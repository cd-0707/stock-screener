"""Moteur de scoring et de recommandation d'actions.

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.scoring --profil prudent
    python -m src.scoring --profil dynamique --nb 8 --secteurs Technology Healthcare
    python -m src.scoring --profil equilibre --horizon long --prix-max 150 --dividendes

Principe, en six temps :
 1. Chaque indicateur est converti en note de 0 à 1 : son rang en percentile
    parmi TOUT l'univers (1 = le meilleur). On normalise avant de filtrer, pour
    qu'une action garde la même note quels que soient les filtres choisis.
 2. Chaque profil (prudent, équilibré, dynamique) pondère ces notes différemment.
 3. L'horizon et la préférence pour les dividendes ajustent légèrement les poids.
 4. Un clustering (K-means) regroupe les actions en 4 familles de comportement ;
    la compatibilité entre la famille et le profil ajoute un petit bonus.
 5. On filtre selon les contraintes de l'utilisateur (secteur, prix, liquidité).
 6. On classe, on retient les N meilleures et on explique chaque choix.

Les pondérations ci-dessous sont des HYPOTHÈSES de modélisation, pas des
vérités : elles sont à justifier et à discuter dans le README.
"""

import argparse
import logging
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.collecte import CHEMIN_BASE
from src.indicateurs import construire_tableau

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paramètres du modèle
# ---------------------------------------------------------------------------

# Chaque critère : (colonnes du tableau d'indicateurs, sens)
#   sens = +1 : plus c'est haut, mieux c'est ; -1 : plus c'est bas, mieux c'est.
# Le drawdown est négatif (-0,2 = -20 %) : un drawdown « plus haut » est donc meilleur.
CRITERES = {
    "volatilite": (["volatilite_annuelle"], -1),
    "drawdown": (["drawdown_max"], +1),
    "sharpe": (["sharpe"], +1),
    "rendement": (["rendement_annuel"], +1),
    "momentum": (["momentum_6m", "momentum_12m"], +1),
    "dividende": (["rendement_dividende"], +1),
}

# Pondérations par profil (la somme de chaque ligne vaut 1)
PROFILS = {
    "prudent":   {"volatilite": 0.35, "drawdown": 0.25, "sharpe": 0.15,
                  "rendement": 0.10, "momentum": 0.05, "dividende": 0.10},
    "equilibre": {"volatilite": 0.20, "drawdown": 0.15, "sharpe": 0.25,
                  "rendement": 0.20, "momentum": 0.20, "dividende": 0.00},
    "dynamique": {"volatilite": 0.05, "drawdown": 0.05, "sharpe": 0.20,
                  "rendement": 0.30, "momentum": 0.40, "dividende": 0.00},
}

# Position de risque visée par chaque profil sur l'échelle des familles
# (0 = la famille la plus défensive, 1 = la plus risquée)
CIBLE_RISQUE = {"prudent": 0.0, "equilibre": 0.5, "dynamique": 1.0}

# Multiplicateurs de poids selon l'horizon de placement (avant renormalisation).
# Hypothèse : à court terme, on craint les baisses et on suit la tendance ;
# à long terme, on tolère la volatilité et on vise le rapport rendement/risque.
HORIZONS = {
    "court": {"momentum": 1.5, "drawdown": 1.3, "rendement": 0.8},
    "moyen": {},
    "long": {"momentum": 0.6, "sharpe": 1.2, "volatilite": 0.8},
}

BONUS_DIVIDENDE = 0.15       # poids ajouté si l'utilisateur veut des dividendes
POIDS_CLUSTER = 0.10         # part du score venant de la compatibilité de famille
LIQUIDITE_MIN = 1_000_000    # montant moyen échangé par jour (devise locale)

COLONNES_CLUSTER = ["volatilite_annuelle", "drawdown_max", "beta",
                    "rendement_annuel", "momentum_12m"]
FAMILLES = ["Défensif", "Modéré", "Dynamique", "Spéculatif"]

LIBELLES_FORTS = {
    "volatilite": "faible volatilité",
    "drawdown": "baisses passées limitées",
    "sharpe": "bon rapport rendement/risque",
    "rendement": "rendement élevé",
    "momentum": "forte dynamique récente",
    "dividende": "dividende attractif",
}
LIBELLES_FAIBLES = {
    "volatilite": "volatilité élevée",
    "drawdown": "fortes baisses passées",
    "sharpe": "rapport rendement/risque modeste",
    "rendement": "rendement faible",
    "momentum": "dynamique récente faible",
    "dividende": "peu ou pas de dividende",
}


# ---------------------------------------------------------------------------
# Poids
# ---------------------------------------------------------------------------

def _cle(texte):
    """Normalise un nom saisi : minuscules, sans accents (« Équilibré » -> « equilibre »)."""
    sans_accents = unicodedata.normalize("NFD", str(texte)).encode("ascii", "ignore").decode()
    return sans_accents.strip().lower()


def poids_effectifs(profil, horizon="moyen", preference_dividende=False):
    """Poids finaux des critères (somme = 1) selon le profil, l'horizon et les dividendes."""
    profil, horizon = _cle(profil), _cle(horizon)
    if profil not in PROFILS:
        raise ValueError(f"Profil inconnu : {profil!r}. Choix possibles : {list(PROFILS)}")
    if horizon not in HORIZONS:
        raise ValueError(f"Horizon inconnu : {horizon!r}. Choix possibles : {list(HORIZONS)}")

    poids = dict(PROFILS[profil])
    for critere, facteur in HORIZONS[horizon].items():
        poids[critere] *= facteur
    if preference_dividende:
        poids["dividende"] += BONUS_DIVIDENDE

    total = sum(poids.values())
    return {critere: p / total for critere, p in poids.items()}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def ajouter_clusters(tableau):
    """Ajoute les colonnes « famille » (étiquette) et « risque_famille » (0 à 1).

    Les familles sont numérotées par volatilité moyenne croissante : la plus
    calme s'appelle « Défensif », la plus agitée « Spéculatif ». Les étiquettes
    sont indicatives : elles décrivent le comportement passé, pas la qualité.
    """
    t = tableau.copy()
    k = len(FAMILLES)
    colonnes = [c for c in COLONNES_CLUSTER if c in t.columns and t[c].notna().any()]

    if len(t) < 2 * k or not colonnes:
        t["famille"] = None
        t["risque_famille"] = np.nan
        return t

    X = t[colonnes].fillna(t[colonnes].median())
    X = StandardScaler().fit_transform(X)
    t["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    ordre = t.groupby("cluster")["volatilite_annuelle"].mean().sort_values().index
    rang = {cluster: i for i, cluster in enumerate(ordre)}
    t["famille"] = t["cluster"].map(rang).map(lambda i: FAMILLES[i])
    t["risque_famille"] = t["cluster"].map(rang) / (k - 1)
    return t.drop(columns="cluster")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _notes_par_critere(tableau):
    """Note de 0 à 1 par critère : rang en percentile dans l'univers (1 = meilleur)."""
    notes = pd.DataFrame(index=tableau.index)
    for critere, (colonnes, sens) in CRITERES.items():
        rangs = []
        for col in colonnes:
            if col not in tableau.columns:
                continue
            valeurs = tableau[col]
            if col == "rendement_dividende":
                valeurs = valeurs.fillna(0.0)    # pas de dividende = rendement nul
            rangs.append(valeurs.rank(pct=True, ascending=(sens > 0)))
        if rangs:
            notes[critere] = pd.concat(rangs, axis=1).mean(axis=1)
    return notes.fillna(0.5)                      # donnée manquante : note neutre


def calculer_scores(tableau, poids, cible_risque=None, poids_cluster=POIDS_CLUSTER):
    """Calcule le score (0 à 100) de chaque action de l'univers.

    Renvoie deux tableaux :
      - contributions : points apportés par chaque critère ; leur somme = « score »
      - notes : notes de 0 à 1 par critère (servent aux explications)

    Un critère dont la donnée est absente du tableau (par exemple le dividende
    dans un backtest) est ignoré, et les poids restants sont renormalisés.
    """
    notes = _notes_par_critere(tableau)
    criteres = [c for c in notes.columns if poids.get(c, 0) > 0]
    if not criteres:
        raise ValueError("Aucun critère exploitable avec ces poids et ces données.")

    p = pd.Series({c: poids[c] for c in criteres})
    p = p / p.sum()

    avec_cluster = (
        cible_risque is not None
        and "risque_famille" in tableau.columns
        and tableau["risque_famille"].notna().any()
    )
    part_cluster = poids_cluster if avec_cluster else 0.0

    contributions = notes[criteres].mul(p) * (1 - part_cluster) * 100
    if avec_cluster:
        compatibilite = 1 - (tableau["risque_famille"] - cible_risque).abs()
        contributions["bonus_famille"] = compatibilite.fillna(0.5) * part_cluster * 100
    contributions["score"] = contributions.sum(axis=1)
    return contributions, notes


# ---------------------------------------------------------------------------
# Filtrage, sélection, explication
# ---------------------------------------------------------------------------

def filtrer(tableau, secteurs=None, prix_min=None, prix_max=None, liquidite_min=LIQUIDITE_MIN):
    """Garde les actions qui respectent les contraintes de l'utilisateur.

    Les prix sont exprimés dans la devise de cotation de chaque action
    (euros pour Paris, dollars pour les États-Unis) : pas de conversion.
    """
    masque = pd.Series(True, index=tableau.index)
    if secteurs:
        masque &= tableau["secteur"].isin(secteurs)
    if prix_min is not None:
        masque &= tableau["dernier_prix"] >= prix_min
    if prix_max is not None:
        masque &= tableau["dernier_prix"] <= prix_max
    if liquidite_min and "liquidite_moyenne" in tableau.columns:
        masque &= tableau["liquidite_moyenne"] >= liquidite_min
    return tableau[masque]


def _selectionner(classement, nb_actions, max_par_secteur):
    """Retient les N premières, en limitant éventuellement le nombre par secteur."""
    if not max_par_secteur:
        return classement.head(nb_actions)
    retenus, compteur = [], {}
    for ticker, secteur in classement["secteur"].items():
        cle = secteur if pd.notna(secteur) else "Inconnu"
        if compteur.get(cle, 0) >= max_par_secteur:
            continue
        retenus.append(ticker)
        compteur[cle] = compteur.get(cle, 0) + 1
        if len(retenus) == nb_actions:
            break
    return classement.loc[retenus]


def _expliquer(notes, poids, seuil_fort=0.75, seuil_faible=0.30, poids_min=0.10):
    """Phrase d'explication : critères importants pour le profil où l'action brille ou déçoit."""
    importants = [c for c in notes.index if poids.get(c, 0) >= poids_min]

    forts = sorted((c for c in importants if notes[c] >= seuil_fort),
                   key=lambda c: poids[c] * notes[c], reverse=True)[:3]
    faibles = sorted((c for c in importants if notes[c] <= seuil_faible),
                     key=lambda c: poids[c] * (1 - notes[c]), reverse=True)[:2]

    morceaux = []
    if forts:
        morceaux.append("Points forts : " + ", ".join(
            f"{LIBELLES_FORTS[c]} ({round(notes[c] * 100)}e centile)" for c in forts))
    else:
        morceaux.append("Aucun point fort marqué sur les critères du profil")
    if faibles:
        morceaux.append("À surveiller : " + ", ".join(LIBELLES_FAIBLES[c] for c in faibles))
    return ". ".join(morceaux) + "."


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def recommander(profil, horizon="moyen", secteurs=None, prix_min=None, prix_max=None,
                dividendes=False, nb_actions=10, max_par_secteur=None,
                liquidite_min=LIQUIDITE_MIN, tableau=None):
    """Classement expliqué des actions adaptées à un profil d'investisseur.

    profil          : « prudent », « équilibré » ou « dynamique »
    horizon         : « court » (< 1 an), « moyen » (1 à 5 ans), « long » (> 5 ans)
    secteurs        : liste de secteurs à garder (None = tous)
    prix_min/max    : fourchette de prix de l'action, en devise locale
    dividendes      : True si l'investisseur souhaite favoriser les dividendes
    nb_actions      : nombre d'actions à renvoyer
    max_par_secteur : plafond d'actions d'un même secteur (None = pas de plafond)
    tableau         : tableau d'indicateurs ; chargé depuis la base si omis

    Renvoie un DataFrame trié par score décroissant : informations descriptives,
    score de 0 à 100, points apportés par chaque critère et phrase d'explication.
    Les poids utilisés sont dans classement.attrs["poids"].
    """
    cle = _cle(profil)
    poids = poids_effectifs(profil, horizon, dividendes)

    if tableau is None:
        tableau = charger_univers()
    if "risque_famille" not in tableau.columns:
        tableau = ajouter_clusters(tableau)

    contributions, notes = calculer_scores(tableau, poids, CIBLE_RISQUE[cle])
    candidats = filtrer(tableau, secteurs, prix_min, prix_max, liquidite_min)

    descriptives = [c for c in ["nom", "secteur", "pays", "famille", "dernier_prix"]
                    if c in tableau.columns]
    classement = candidats[descriptives].join(contributions)
    classement = classement.sort_values("score", ascending=False)
    classement = _selectionner(classement, nb_actions, max_par_secteur).copy()

    classement["explication"] = [_expliquer(notes.loc[t], poids) for t in classement.index]
    colonnes_points = [c for c in contributions.columns if c != "score"]
    classement[["score"] + colonnes_points] = classement[["score"] + colonnes_points].round(1)
    classement.attrs["poids"] = poids
    return classement


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def _donnees_de_marche(conn, date=None, jours_tolerance=10):
    """Dernier prix (non ajusté) et montant moyen échangé par jour sur 60 séances."""
    long = pd.read_sql_query(
        "SELECT ticker, date, cloture, volume FROM prix", conn, parse_dates=["date"]
    )
    if date is not None:
        long = long[long["date"] <= pd.Timestamp(date)]
    long = long.sort_values(["ticker", "date"])

    derniere = long.groupby("ticker").tail(1).set_index("ticker")
    marche = pd.DataFrame({"dernier_prix": derniere["cloture"],
                           "derniere_date": derniere["date"]})

    recents = long.groupby("ticker").tail(60).assign(valeur=lambda d: d["cloture"] * d["volume"])
    marche["liquidite_moyenne"] = recents.groupby("ticker")["valeur"].mean()

    # On écarte les valeurs dont les données s'arrêtent bien avant les autres
    # (retirées de la cote, par exemple).
    limite = marche["derniere_date"].max() - pd.Timedelta(days=jours_tolerance)
    perimees = marche.index[marche["derniere_date"] < limite]
    if len(perimees):
        log.warning("Données périmées, valeurs écartées : %s", ", ".join(perimees))
    return marche.drop(index=perimees).drop(columns="derniere_date")


def charger_univers(conn=None, date=None):
    """Tableau complet prêt pour le scoring : indicateurs, prix, liquidité et familles.

    Avec une date, tout est calculé à cette date et les fondamentaux (PER,
    dividende) sont exclus, car la base ne contient que leurs valeurs actuelles.
    """
    ferme = conn is None
    if ferme:
        conn = sqlite3.connect(CHEMIN_BASE)
    try:
        tableau = construire_tableau(conn, date=date, avec_fondamentaux=date is None)
        marche = _donnees_de_marche(conn, date)
    finally:
        if ferme:
            conn.close()
    tableau = tableau.join(marche, how="inner")
    return ajouter_clusters(tableau)


# ---------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Recommandation d'actions selon un profil.")
    parser.add_argument("--profil", default="equilibre", help="prudent, equilibre ou dynamique")
    parser.add_argument("--horizon", default="moyen", help="court, moyen ou long")
    parser.add_argument("--secteurs", nargs="*", default=None, help="un ou plusieurs secteurs")
    parser.add_argument("--prix-min", type=float, default=None)
    parser.add_argument("--prix-max", type=float, default=None)
    parser.add_argument("--dividendes", action="store_true", help="favoriser les dividendes")
    parser.add_argument("--nb", type=int, default=10, help="nombre d'actions (défaut : 10)")
    parser.add_argument("--max-par-secteur", type=int, default=None)
    parser.add_argument("--liquidite-min", type=float, default=LIQUIDITE_MIN,
                        help="montant moyen échangé par jour minimal (défaut : 1 000 000)")
    args = parser.parse_args()

    resultat = recommander(
        args.profil, horizon=args.horizon, secteurs=args.secteurs,
        prix_min=args.prix_min, prix_max=args.prix_max, dividendes=args.dividendes,
        nb_actions=args.nb, max_par_secteur=args.max_par_secteur,
        liquidite_min=args.liquidite_min,
    )

    print("\nPoids utilisés :", {c: round(p, 2) for c, p in resultat.attrs["poids"].items() if p})
    if resultat.empty:
        print("\nAucune action ne respecte ces contraintes.")
    else:
        pd.options.display.width = 200
        pd.options.display.float_format = "{:,.1f}".format
        affichage = resultat[["nom", "secteur", "famille", "dernier_prix", "score"]].copy()
        affichage["nom"] = affichage["nom"].str.slice(0, 28)
        print("\n" + affichage.to_string())
        print("\nExplications (3 premières) :")
        for ticker, ligne in resultat.head(3).iterrows():
            print(f"  {ticker} : {ligne['explication']}")
    print("\n  Projet pédagogique : ceci n'est pas un conseil en investissement.")