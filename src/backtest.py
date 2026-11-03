"""Backtest : que serait-il arrivé si on avait suivi les recommandations dans le passé ?

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.backtest --profil prudent
    python -m src.backtest --profil dynamique --nb 8 --frequence mensuel --frais 0.002

Principe : à chaque date de rééquilibrage t (par exemple chaque fin de trimestre)
  1. on calcule les indicateurs avec UNIQUEMENT les prix disponibles jusqu'à t ;
  2. on score les actions, on retient les meilleures, on optimise les poids ;
  3. on ajuste le portefeuille (en payant des frais sur ce qu'on achète ou vend) ;
  4. on le laisse évoluer jusqu'au rééquilibrage suivant.

Absence de fuite de données : rien de ce qui se passe après t n'est utilisé à t.
Les fondamentaux (PER, dividende) sont exclus, car la base ne contient que leurs
valeurs actuelles. Un test automatique vérifie qu'un choc futur ne change pas le passé.

Simplifications connues, à écrire dans le README :
  - l'univers est celui d'aujourd'hui : biais de survie (les entreprises disparues
    ou sorties des indices n'y figurent pas) ;
  - pas de conversion de devises : chaque action est suivie dans sa devise ;
  - exécution au cours de clôture du jour du signal ;
  - frais proportionnels aux montants échangés, sans écart acheteur/vendeur.
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.collecte import CHEMIN_BASE, RACINE
from src.indicateurs import (
    FENETRE_DEFAUT, MIN_OBS, calculer_indicateurs, charger_prix, drawdown_max,
    ratio_sharpe, rendement_annualise, volatilite_annualisee,
)
from src.portefeuille import POIDS_MAX_DEFAUT, construire_portefeuille
from src.univers import INDICES

log = logging.getLogger(__name__)

MOIS_PAR_FREQUENCE = {"mensuel": 1, "trimestriel": 3, "semestriel": 6, "annuel": 12}
FRAIS_DEFAUT = 0.001          # 0,1 % du montant échangé
HISTORIQUE_MIN = 504          # 2 ans de données avant le premier investissement


def dates_de_rebalancement(index, frequence="trimestriel", historique_min=HISTORIQUE_MIN):
    """Dernier jour de bourse de chaque période, à partir de `historique_min` jours de données."""
    if frequence not in MOIS_PAR_FREQUENCE:
        raise ValueError(f"Fréquence inconnue : {frequence!r}. Choix : {list(MOIS_PAR_FREQUENCE)}")
    if len(index) <= historique_min:
        raise ValueError("Pas assez de données pour lancer le backtest.")

    jours = pd.Series(index[historique_min:], index=index[historique_min:])
    periode = (jours.index.year * 12 + jours.index.month - 1) // MOIS_PAR_FREQUENCE[frequence]
    dates = list(jours.groupby(periode).max())
    if dates[-1] != index[-1]:
        dates.append(index[-1])
    return pd.DatetimeIndex(dates)


def _tableau_a_la_date(prix, date, secteurs):
    """Indicateurs tels qu'ils auraient été calculés à `date`, avec les seules données passées."""
    tableau = calculer_indicateurs(prix, date=date)
    derniers = prix.loc[:date].iloc[-1]
    tableau = tableau[derniers.reindex(tableau.index).notna()]   # écarte les valeurs sans cours à cette date
    tableau["secteur"] = secteurs.reindex(tableau.index)
    return tableau


def simuler(prix, rendements, dates, fonction_cible, frais):
    """Fait vivre un portefeuille entre les dates de rééquilibrage.

    fonction_cible(t) renvoie les poids visés à la date t (Series, somme = 1).
    Entre deux dates, les poids dérivent avec les prix (on n'échange rien).
    Renvoie (courbe de valeur, compositions visées, rotation à chaque date).
    """
    valeur = 1.0
    poids_actuels = pd.Series(dtype=float)
    morceaux = [pd.Series([1.0], index=[dates[0]])]
    compositions, rotations = {}, {}

    for t, suivante in zip(dates[:-1], dates[1:]):
        cible = fonction_cible(t)

        union = cible.index.union(poids_actuels.index)
        ecart = cible.reindex(union, fill_value=0.0) - poids_actuels.reindex(union, fill_value=0.0)
        rotation = ecart.abs().sum()               # part du portefeuille échangée
        valeur *= 1.0 - frais * rotation
        compositions[t], rotations[t] = cible, rotation

        jours = rendements.loc[(rendements.index > t) & (rendements.index <= suivante)]
        if cible.empty:                            # rien à acheter : on reste en liquidités
            segment = pd.Series(valeur, index=jours.index)
            poids_actuels = pd.Series(dtype=float)
        else:
            cumul = (1.0 + jours[cible.index].fillna(0.0)).cumprod()
            investi = cumul.mul(cible, axis=1)
            segment = valeur * investi.sum(axis=1)
            fin = investi.iloc[-1]
            poids_actuels = fin / fin.sum()
        valeur = segment.iloc[-1]
        morceaux.append(segment)

    courbe = pd.concat(morceaux)
    composition = pd.DataFrame(compositions).T.fillna(0.0)
    return courbe, composition, pd.Series(rotations)


def metriques(courbes):
    """Tableau de performance : une ligne par courbe de valeur."""
    lignes = {}
    for nom in courbes.columns:
        c = courbes[nom].dropna()
        lignes[nom] = {
            "Rendement total": c.iloc[-1] / c.iloc[0] - 1,
            "Rendement annualisé": rendement_annualise(c),
            "Volatilité annualisée": volatilite_annualisee(c),
            "Sharpe": ratio_sharpe(c),
            "Drawdown max": drawdown_max(c),
        }
    return pd.DataFrame(lignes).T


def lancer_backtest(prix, secteurs, profil, horizon="moyen", nb_actions=10, max_par_secteur=None,
                    poids_max=POIDS_MAX_DEFAUT, methode=None, frais=FRAIS_DEFAUT,
                    frequence="trimestriel", historique_min=HISTORIQUE_MIN,
                    fenetre=FENETRE_DEFAUT, previsions_volatilite=None):
    """Backtest complet d'un profil, comparé à un portefeuille équipondéré et aux indices.

    prix     : DataFrame (dates x tickers) de clôtures ajustées, avec les indices (voir charger_prix)
    secteurs : Series ticker -> secteur
    previsions_volatilite : dictionnaire {date de rééquilibrage: Series de volatilités prévues}
        (voir prediction.previsions_aux_dates). Si fourni, le critère « volatilité » du score
        utilise la volatilité PRÉVUE à la place de la volatilité passée sur 3 ans. Chaque
        prévision doit avoir été faite avec les seuls prix disponibles à sa date.
    """
    rendements = prix / prix.shift(1) - 1
    dates = dates_de_rebalancement(prix.index, frequence, historique_min)
    actions = [c for c in prix.columns if c not in INDICES]

    def cible_strategie(t):
        tableau = _tableau_a_la_date(prix, t, secteurs)
        if len(tableau) < 3:
            return pd.Series(dtype=float)
        if previsions_volatilite is not None and t in previsions_volatilite:
            prevue = previsions_volatilite[t].reindex(tableau.index)
            tableau["volatilite_annuelle"] = prevue.fillna(tableau["volatilite_annuelle"])
        poids, _ = construire_portefeuille(
            tableau, rendements.loc[:t].tail(fenetre), profil, horizon=horizon,
            nb_actions=nb_actions, max_par_secteur=max_par_secteur, poids_max=poids_max,
            methode=methode, liquidite_min=0,
        )
        return poids

    def cible_equipondere(t):
        fenetre_prix = prix.loc[:t, actions].tail(fenetre + 1)
        eligibles = fenetre_prix.columns[(fenetre_prix.notna().sum() > MIN_OBS) & fenetre_prix.iloc[-1].notna()]
        return pd.Series(1.0 / len(eligibles), index=eligibles) if len(eligibles) else pd.Series(dtype=float)

    log.info("Backtest « %s » : %d rééquilibrages du %s au %s",
             profil, len(dates) - 1, dates[0].date(), dates[-1].date())
    courbe, compositions, rotations = simuler(prix, rendements, dates, cible_strategie, frais)
    equipondere, _, _ = simuler(prix, rendements, dates, cible_equipondere, frais)

    courbes = pd.DataFrame({"Stratégie": courbe, "Équipondéré (tout l'univers)": equipondere})
    for ticker, nom in INDICES.items():
        if ticker in prix.columns:
            reference = prix.loc[dates[0]:dates[-1], ticker].ffill()
            courbes[nom] = (reference / reference.iloc[0]).reindex(courbes.index)

    return {
        "courbes": courbes,
        "metriques": metriques(courbes),
        "compositions": compositions,
        "rotations": rotations,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Backtest d'un profil d'investisseur.")
    parser.add_argument("--profil", default="equilibre", help="prudent, equilibre ou dynamique")
    parser.add_argument("--horizon", default="moyen")
    parser.add_argument("--nb", type=int, default=10, help="nombre d'actions détenues")
    parser.add_argument("--max-par-secteur", type=int, default=None)
    parser.add_argument("--poids-max", type=float, default=POIDS_MAX_DEFAUT)
    parser.add_argument("--frais", type=float, default=FRAIS_DEFAUT, help="par défaut 0.001 (0,1 %%)")
    parser.add_argument("--frequence", default="trimestriel", choices=list(MOIS_PAR_FREQUENCE))
    args = parser.parse_args()

    conn = sqlite3.connect(CHEMIN_BASE)
    prix = charger_prix(conn)
    secteurs = pd.read_sql_query("SELECT ticker, secteur FROM actions", conn).set_index("ticker")["secteur"]
    conn.close()

    resultat = lancer_backtest(
        prix, secteurs, args.profil, horizon=args.horizon, nb_actions=args.nb,
        max_par_secteur=args.max_par_secteur, poids_max=args.poids_max,
        frais=args.frais, frequence=args.frequence,
    )

    tableau = resultat["metriques"].copy()
    for colonne in ["Rendement total", "Rendement annualisé"]:
        tableau[colonne] = (tableau[colonne] * 100).map("{:+.1f} %".format)
    for colonne in ["Volatilité annualisée", "Drawdown max"]:
        tableau[colonne] = (tableau[colonne] * 100).map("{:.1f} %".format)
    tableau["Sharpe"] = tableau["Sharpe"].map("{:.2f}".format)
    print("\nPerformance sur la période (frais inclus pour la stratégie et l'équipondéré) :\n")
    print(tableau.to_string())

    rotation_moyenne = resultat["rotations"].iloc[1:].mean()
    print(f"\nRotation moyenne à chaque rééquilibrage : {rotation_moyenne:.0%} du portefeuille")
    print(f"Frais payés en moyenne par an : environ "
          f"{rotation_moyenne * args.frais * 12 / {'mensuel': 1, 'trimestriel': 3, 'semestriel': 6, 'annuel': 12}[args.frequence]:.2%}")

    dossier = RACINE / "data"
    prefixe = dossier / f"backtest_{args.profil}"
    resultat["courbes"].to_csv(f"{prefixe}_courbes.csv")
    resultat["compositions"].to_csv(f"{prefixe}_compositions.csv")
    try:
        import plotly.express as px
        figure = px.line(resultat["courbes"], title=f"Backtest : profil {args.profil} (valeur de 1 investi au départ)")
        figure.write_html(f"{prefixe}.html", include_plotlyjs="cdn")
        print(f"\nGraphique enregistré : {prefixe}.html (à ouvrir dans le navigateur)")
    except ImportError:
        pass
    print(f"Courbes et compositions enregistrées dans {dossier}")
    print("\n⚠️  Performances passées : aucune garantie pour l'avenir. Projet pédagogique, "
          "ceci n'est pas un conseil en investissement.")