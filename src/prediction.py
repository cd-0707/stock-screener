"""Prévision de la volatilité à un mois : un modèle HAR et ses extensions « machine learning ».

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.prediction
    python -m src.prediction --k 15        # taille de la sélection de l'expérience finale

Références (version adaptée à des données quotidiennes, voir « Adaptations ») :
  - Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
    Journal of Financial Econometrics, 7(2), 174-196.
    Modèle HAR : la volatilité future est expliquée par des volatilités passées
    mesurées sur des horizons différents (jour, semaine, mois).
  - Christensen, K., Siggaard, M. et Veliyev, B. (2022). A Machine Learning Approach
    to Volatility Forecasting. Journal of Financial Econometrics, doi:10.1093/jjfinec/nbac020.
    Compare des modèles HAR à des méthodes de machine learning (régularisation, arbres,
    réseaux de neurones) sur les actions du Dow Jones.
  - Gu, S., Kelly, B. et Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
    Review of Financial Studies, 33(5), 2223-2273.
    Sert de référence méthodologique : validation hors échantillon, chronologique.

Adaptations, à écrire dans le README :
  - Ces articles utilisent des données intrajournalières (variance réalisée à haute
    fréquence). Ici on n'a que des clôtures quotidiennes : la variance est approchée par
    le carré du rendement quotidien, une mesure beaucoup plus bruitée.
  - Horizon d'un mois (21 jours de bourse) et non d'un jour.
  - Pas de composante « quotidienne » du HAR (le carré d'un seul rendement est trop bruité) :
    on utilise les composantes hebdomadaire (5 j), mensuelle (21 j) et trimestrielle (63 j).
  - Univers : 89 actions, soit environ 10 000 observations mensuelles, très loin des
    bases des articles : on garde donc des modèles simples.

Rigueur de l'évaluation :
  - toutes les caractéristiques d'une date n'utilisent que des prix jusqu'à cette date ;
  - l'entraînement ne contient que des observations dont la période à prévoir est
    entièrement terminée avant le début du test (« purge » : pas de chevauchement) ;
  - validation par avancement : on entraîne sur le passé, on teste sur l'année suivante,
    puis on recommence en avançant ;
  - les modèles sont comparés à des références simples, pas seulement entre eux.
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.collecte import CHEMIN_BASE, RACINE
from src.indicateurs import charger_prix
from src.univers import INDICES

log = logging.getLogger(__name__)

JOURS_PAR_AN = 252
HORIZON = 21            # un mois de bourse
PAS = 21                # une observation par mois et par action (cibles qui ne se chevauchent pas)
HISTORIQUE_MIN = 252    # un an de prix avant la première observation
EPSILON = 1e-4          # plancher avant de passer au logarithme

CARACTERISTIQUES_HAR = ["vol_5j", "vol_21j", "vol_63j"]
CARACTERISTIQUES_ETENDUES = CARACTERISTIQUES_HAR + [
    "vol_252j", "vol_marche_21j", "rend_21j", "rend_63j", "drawdown_63j",
]

NOMS_MODELES = {
    "naif": "Naïf (volatilité du dernier mois)",
    "moyenne_longue": "Moyenne sur 1 an",
    "har": "HAR (Corsi, 2009)",
    "har_etendu": "HAR étendu (linéaire)",
    "ml": "Gradient boosting",
}


# ---------------------------------------------------------------------------
# Construction des données
# ---------------------------------------------------------------------------

def _blocs(prix, horizon=HORIZON):
    """Caractéristiques (tableaux date x action) et volatilité future, à partir des prix.

    Tout ce qui est calculé avec des fenêtres « rolling » regarde en arrière ; seule la
    volatilité future (la cible) regarde en avant, et elle est isolée dans un tableau à part.
    """
    actions = [c for c in prix.columns if c not in INDICES]
    prix = prix[actions]
    rendements = prix / prix.shift(1) - 1
    carres = rendements ** 2

    def vol(n):
        return np.sqrt(carres.rolling(n).mean() * JOURS_PAR_AN)

    def log_plancher(tableau):
        return np.log(tableau.clip(lower=EPSILON))

    vol_21 = vol(21)
    marche = vol_21.mean(axis=1)        # volatilité moyenne de l'univers : facteur « marché »
    blocs = {
        "vol_5j": log_plancher(vol(5)),
        "vol_21j": log_plancher(vol_21),
        "vol_63j": log_plancher(vol(63)),
        "vol_252j": log_plancher(vol(252)),
        "vol_marche_21j": log_plancher(pd.DataFrame({c: marche for c in actions})),
        "rend_21j": prix / prix.shift(21) - 1,
        "rend_63j": prix / prix.shift(63) - 1,
        "drawdown_63j": prix / prix.rolling(63).max() - 1,
    }
    futur = np.sqrt(carres.rolling(horizon).mean().shift(-horizon) * JOURS_PAR_AN)
    return blocs, futur


def _empiler(blocs, positions, index):
    """Passe de tableaux date x action à un tableau « une ligne par action et par date »."""
    colonnes = {nom: tableau.iloc[positions].unstack() for nom, tableau in blocs.items()}
    long = pd.DataFrame(colonnes)
    long.index.names = ["ticker", "date"]
    return long.reset_index()


def construire_jeu_de_donnees(prix, horizon=HORIZON, pas=PAS, historique_min=HISTORIQUE_MIN):
    """Jeu de données de prévision : une ligne par action et par mois.

    Colonnes : ticker, date, les caractéristiques (calculées avec les prix jusqu'à « date »),
    « cible » (logarithme de la volatilité annualisée des `horizon` jours SUIVANTS) et
    « fin_cible » (dernier jour de la période de la cible, utilisé pour la purge).
    """
    blocs, futur = _blocs(prix, horizon)
    blocs["cible"] = np.log(futur.clip(lower=EPSILON))
    positions = np.arange(historique_min, len(prix), pas)
    donnees = _empiler(blocs, positions, prix.index)

    fins = pd.Series(prix.index[np.minimum(positions + horizon, len(prix) - 1)],
                     index=prix.index[positions])
    donnees["fin_cible"] = donnees["date"].map(fins)
    return donnees.dropna().reset_index(drop=True)


def caracteristiques_actuelles(prix):
    """Caractéristiques de chaque action à la dernière date disponible (pas de cible)."""
    blocs, _ = _blocs(prix)
    return _empiler(blocs, [len(prix) - 1], prix.index).dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

def _predire(nom, entrainement, test):
    """Prévision (en logarithme de volatilité) du modèle « nom » pour les lignes de `test`."""
    if nom == "naif":
        return test["vol_21j"].to_numpy()
    if nom == "moyenne_longue":
        return test["vol_252j"].to_numpy()

    if nom == "har":
        colonnes, modele = CARACTERISTIQUES_HAR, LinearRegression()
    elif nom == "har_etendu":
        colonnes, modele = CARACTERISTIQUES_ETENDUES, LinearRegression()
    elif nom == "ml":
        colonnes = CARACTERISTIQUES_ETENDUES
        modele = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
            early_stopping=False, random_state=0,
        )
    else:
        raise ValueError(f"Modèle inconnu : {nom!r}. Choix : {list(NOMS_MODELES)}")

    modele.fit(entrainement[colonnes], entrainement["cible"])
    return modele.predict(test[colonnes])


# ---------------------------------------------------------------------------
# Validation par avancement
# ---------------------------------------------------------------------------

def jeu_d_entrainement(donnees, debut_test):
    """Observations dont la période à prévoir est entièrement terminée avant `debut_test`."""
    return donnees[donnees["fin_cible"] < debut_test]


def valider_par_avancement(donnees, min_dates_entrainement=24):
    """Prévisions hors échantillon de tous les modèles, année de test après année de test.

    Pour chaque année : entraînement sur tout le passé (avec purge), prévision de l'année.
    Renvoie un tableau « une ligne par action et par date de test » avec la cible réelle
    et la prévision de chaque modèle.
    """
    resultats = []
    for annee in sorted(donnees["date"].dt.year.unique()):
        test = donnees[donnees["date"].dt.year == annee]
        entrainement = jeu_d_entrainement(donnees, test["date"].min())
        if entrainement["date"].nunique() < min_dates_entrainement:
            continue
        sortie = test[["ticker", "date", "cible"]].copy()
        sortie["annee"] = annee
        for nom in NOMS_MODELES:
            sortie[nom] = _predire(nom, entrainement, test)
        resultats.append(sortie)
    if not resultats:
        raise ValueError("Pas assez de données pour une validation par avancement.")
    return pd.concat(resultats, ignore_index=True)


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------

def mesurer(predictions):
    """Qualité des prévisions de chaque modèle, sur toute la période de test.

    - RMSE et MAE : erreur moyenne sur le logarithme de la volatilité (plus bas = mieux) ;
    - R² hors échantillon : 1 - (erreur du modèle / erreur du modèle naïf), en erreurs
      quadratiques. Positif : meilleur que le naïf ; négatif : pire ;
    - QLIKE : perte standard en prévision de volatilité, calculée sur la variance
      (plus bas = mieux).
    """
    y = predictions["cible"]
    sse_naif = ((y - predictions["naif"]) ** 2).sum()
    lignes = {}
    for nom, libelle in NOMS_MODELES.items():
        erreur = y - predictions[nom]
        rapport = np.exp(2 * y) / np.exp(2 * predictions[nom])
        lignes[libelle] = {
            "RMSE (log)": np.sqrt((erreur ** 2).mean()),
            "MAE (log)": erreur.abs().mean(),
            "R² vs naïf": 1 - (erreur ** 2).sum() / sse_naif,
            "QLIKE": (rapport - np.log(rapport) - 1).mean(),
        }
    return pd.DataFrame(lignes).T


def mesurer_par_annee(predictions):
    """Rapport RMSE(modèle) / RMSE(naïf) pour chaque année de test (< 1 : meilleur que le naïf)."""
    lignes = {}
    for annee, groupe in predictions.groupby("annee"):
        rmse_naif = np.sqrt(((groupe["cible"] - groupe["naif"]) ** 2).mean())
        lignes[annee] = {
            NOMS_MODELES[nom]: np.sqrt(((groupe["cible"] - groupe[nom]) ** 2).mean()) / rmse_naif
            for nom in NOMS_MODELES if nom != "naif"
        }
    return pd.DataFrame(lignes).T


def experience_de_selection(predictions, k=10):
    """Choisir les k actions de plus faible volatilité PRÉVUE donne-t-il une volatilité réelle plus basse ?

    À chaque date de test, chaque modèle retient les k actions dont il prévoit la
    volatilité la plus basse ; on mesure leur volatilité réellement observée le mois
    suivant. Comparaison avec le hasard (moyenne de tout l'univers) et avec un oracle
    qui connaîtrait l'avenir (borne basse, inatteignable).
    """
    reel = np.exp(predictions["cible"])
    mesures = {NOMS_MODELES[nom]: [] for nom in NOMS_MODELES}
    mesures["Hasard (tout l'univers)"] = []
    mesures["Oracle (connaît l'avenir)"] = []

    for _, groupe in predictions.assign(reel=reel).groupby("date"):
        if len(groupe) < 2 * k:
            continue
        for nom, libelle in NOMS_MODELES.items():
            choix = groupe.nsmallest(k, nom)
            mesures[libelle].append(choix["reel"].mean())
        mesures["Hasard (tout l'univers)"].append(groupe["reel"].mean())
        mesures["Oracle (connaît l'avenir)"].append(groupe.nsmallest(k, "reel")["reel"].mean())

    return pd.Series({nom: np.mean(valeurs) for nom, valeurs in mesures.items() if valeurs},
                     name="Volatilité réelle moyenne du mois suivant")


# ---------------------------------------------------------------------------
# Prévision à la date la plus récente
# ---------------------------------------------------------------------------

def prevoir_volatilite(prix, modele="har"):
    """Volatilité annualisée prévue pour le mois qui suit la dernière date, par action."""
    donnees = construire_jeu_de_donnees(prix)
    actuel = caracteristiques_actuelles(prix)
    prevision = _predire(modele, donnees, actuel)
    return pd.Series(np.exp(prevision), index=actuel["ticker"], name="volatilite_prevue")


# ---------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Prévision de la volatilité à un mois.")
    parser.add_argument("--k", type=int, default=10, help="taille de la sélection (défaut : 10)")
    args = parser.parse_args()

    chemin = CHEMIN_BASE if CHEMIN_BASE.exists() else RACINE / "data" / "snapshot.db"
    conn = sqlite3.connect(chemin)
    prix = charger_prix(conn)
    conn.close()

    donnees = construire_jeu_de_donnees(prix)
    log.info("%d observations (%d actions, %d dates), du %s au %s",
             len(donnees), donnees["ticker"].nunique(), donnees["date"].nunique(),
             donnees["date"].min().date(), donnees["date"].max().date())
    predictions = valider_par_avancement(donnees)
    log.info("Test hors échantillon : %d prévisions, de %d à %d",
             len(predictions), predictions["annee"].min(), predictions["annee"].max())

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200
    print("\n1. Qualité des prévisions hors échantillon (logarithme de la volatilité) :\n")
    metriques = mesurer(predictions)
    print(metriques.to_string())

    print("\n2. Rapport d'erreur au modèle naïf, année par année (< 1 : meilleur que le naïf) :\n")
    print(mesurer_par_annee(predictions).to_string())

    print(f"\n3. Sélection des {args.k} actions de plus faible volatilité prévue (volatilité réelle ensuite) :\n")
    selection = experience_de_selection(predictions, k=args.k)
    print((selection * 100).map("{:.1f} %".format).to_string())

    sortie = RACINE / "data" / "prediction_metriques.csv"
    metriques.to_csv(sortie)
    print(f"\nMétriques enregistrées : {sortie}")
    print("\n⚠️  Évaluation sur une seule période et un univers soumis au biais de survie. "
          "Projet pédagogique, ceci n'est pas un conseil en investissement.")