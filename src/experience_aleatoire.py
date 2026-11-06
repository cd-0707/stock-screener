"""Le scoring fait-il mieux qu'un tirage au hasard ?

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.experience_aleatoire
    python -m src.experience_aleatoire --n 2000 --nb 10 --profils prudent

La question à laquelle ce test répond : « les actions choisies par le score sont-elles
meilleures que 10 actions tirées au hasard dans le même univers ? »

Principe : à chaque date de rééquilibrage, on compare
  - la STRATÉGIE : les N meilleures actions selon le score du profil, à poids égaux ;
  - des milliers de portefeuilles ALÉATOIRES : N actions tirées au hasard parmi les mêmes
    actions candidates, à poids égaux, avec les mêmes dates, les mêmes frais et les mêmes
    règles de rééquilibrage.
Les deux versions détiennent N actions à poids égaux : la différence ne peut venir que du
choix des actions, ce qui écarte l'effet de la diversification (le défaut de la comparaison
avec le portefeuille équipondéré de tout l'univers) et celui de l'optimisation des poids.

Lecture : pour chaque indicateur, on regarde la part des portefeuilles aléatoires que la
stratégie bat, et une p-valeur unilatérale empirique (la part des portefeuilles aléatoires
au moins aussi bons que la stratégie). Une p-valeur faible (par exemple moins de 5 %) est un
indice que le score apporte quelque chose ; une valeur élevée signifie que la stratégie
est indiscernable du hasard.

Limites : une seule période, un seul univers (celui d'aujourd'hui : biais de survie qui
avantage aussi les portefeuilles aléatoires) ; les rendements d'un même portefeuille aléatoire
sont corrélés dans le temps, mais chaque simulation est indépendante des autres ; la p-valeur
dépend du nombre de simulations ; on teste plusieurs indicateurs et plusieurs profils, donc
quelques résultats « significatifs » peuvent apparaître par hasard.
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.backtest import (
    FRAIS_DEFAUT, HISTORIQUE_MIN, _tableau_a_la_date, dates_de_rebalancement, simuler,
)
from src.collecte import CHEMIN_BASE, RACINE, SUFFIXE_BASE
from src.indicateurs import (
    charger_prix, drawdown_max, ratio_sharpe, rendement_annualise, volatilite_annualisee,
)
from src.scoring import recommander

log = logging.getLogger(__name__)

# Sens de lecture : +1 = plus c'est haut, mieux c'est ; -1 = plus c'est bas, mieux c'est.
# (le drawdown est négatif : -0,2 est meilleur que -0,4)
SENS = {
    "Rendement annualisé": +1,
    "Volatilité annualisée": -1,
    "Sharpe": +1,
    "Drawdown max": +1,
}


def indicateurs_courbe(courbe):
    """Les quatre indicateurs de performance d'une courbe de valeur."""
    return {
        "Rendement annualisé": rendement_annualise(courbe),
        "Volatilité annualisée": volatilite_annualisee(courbe),
        "Sharpe": ratio_sharpe(courbe),
        "Drawdown max": drawdown_max(courbe),
    }


def poids_egaux(tickers):
    tickers = list(tickers)
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers)


def preparer_dates(prix, secteurs, profils, dates, nb_actions, eligibilite=None):
    """À chaque date : les actions candidates, et la sélection du score pour chaque profil.

    Les candidates sont exactement celles que le backtest principal pourrait choisir
    (assez d'historique, cours disponible, appartenance à l'indice si demandée).
    """
    candidats = {}
    selections = {profil: {} for profil in profils}
    for t in dates[:-1]:
        tableau = _tableau_a_la_date(prix, t, secteurs, eligibilite)
        candidats[t] = list(tableau.index)
        for profil in profils:
            if len(tableau) >= 3:
                classement = recommander(profil, nb_actions=nb_actions, tableau=tableau, liquidite_min=0)
                selections[profil][t] = list(classement.index)
            else:
                selections[profil][t] = []
    return candidats, selections


def courbe_selection(prix, rendements, dates, selection, frais):
    """Courbe de valeur d'une stratégie qui détient, à chaque date, les actions de `selection[date]`."""
    courbe, _, _ = simuler(prix, rendements, dates, lambda t: poids_egaux(selection[t]), frais)
    return courbe


def simuler_portefeuilles_aleatoires(prix, rendements, dates, candidats, nb_actions, frais,
                                     n_simulations, graine=0):
    """Indicateurs de n_simulations stratégies qui tirent au hasard leurs actions à chaque date."""
    hasard = np.random.default_rng(graine)
    lignes = []
    for _ in range(n_simulations):
        def cible(t):
            possibles = candidats[t]
            if not possibles:
                return pd.Series(dtype=float)
            k = min(nb_actions, len(possibles))
            return poids_egaux(hasard.choice(possibles, size=k, replace=False))

        courbe, _, _ = simuler(prix, rendements, dates, cible, frais)
        lignes.append(indicateurs_courbe(courbe))
    return pd.DataFrame(lignes)


def positionner(strategie, aleatoires):
    """Place la stratégie par rapport à la distribution des portefeuilles aléatoires.

    Pour chaque indicateur : valeur de la stratégie, médiane et bornes à 5 % et 95 % des
    portefeuilles aléatoires, part des portefeuilles aléatoires que la stratégie bat, et
    p-valeur unilatérale empirique (1 + nombre de portefeuilles au moins aussi bons) / (1 + n).
    """
    lignes = {}
    for nom, sens in SENS.items():
        valeurs = aleatoires[nom].dropna()
        v = strategie[nom]
        if sens > 0:
            battus = (valeurs < v).mean()
            au_moins_aussi_bons = (valeurs >= v).sum()
        else:
            battus = (valeurs > v).mean()
            au_moins_aussi_bons = (valeurs <= v).sum()
        lignes[nom] = {
            "Stratégie": v,
            "Hasard : 5 %": valeurs.quantile(0.05),
            "Hasard : médiane": valeurs.median(),
            "Hasard : 95 %": valeurs.quantile(0.95),
            "Part du hasard battue": battus,
            "p-valeur": (1 + au_moins_aussi_bons) / (1 + len(valeurs)),
        }
    return pd.DataFrame(lignes).T


def comparer_au_hasard(prix, secteurs, profils=("prudent", "equilibre", "dynamique"), nb_actions=10,
                       n_simulations=1000, frais=FRAIS_DEFAUT, frequence="trimestriel",
                       historique_min=HISTORIQUE_MIN, eligibilite=None, graine=0):
    """Compare chaque profil à des portefeuilles de nb_actions actions tirées au hasard.

    Les mêmes portefeuilles aléatoires servent de référence aux trois profils : leur
    distribution ne dépend pas du profil.
    Renvoie {"positions": {profil: tableau}, "aleatoires": DataFrame, "equipondere": dict}.
    """
    rendements = prix / prix.shift(1) - 1
    dates = dates_de_rebalancement(prix.index, frequence, historique_min)
    candidats, selections = preparer_dates(prix, secteurs, profils, dates, nb_actions, eligibilite)
    log.info("%d dates de rééquilibrage, %d actions candidates en moyenne",
             len(dates) - 1, np.mean([len(c) for c in candidats.values()]))

    aleatoires = simuler_portefeuilles_aleatoires(
        prix, rendements, dates, candidats, nb_actions, frais, n_simulations, graine)

    positions = {}
    for profil in profils:
        courbe = courbe_selection(prix, rendements, dates, selections[profil], frais)
        positions[profil] = positionner(indicateurs_courbe(courbe), aleatoires)

    tout_l_univers = courbe_selection(prix, rendements, dates, candidats, frais)
    return {
        "positions": positions,
        "aleatoires": aleatoires,
        "equipondere": indicateurs_courbe(tout_l_univers),
        "nb_candidats_moyen": float(np.mean([len(c) for c in candidats.values()])),
    }


def _formater(tableau):
    """Mise en forme des valeurs pour l'affichage (pourcentages, Sharpe à 2 décimales)."""
    colonnes = ["Stratégie", "Hasard : 5 %", "Hasard : médiane", "Hasard : 95 %"]
    formats = {
        "Rendement annualisé": lambda v: f"{v * 100:+.1f} %",
        "Volatilité annualisée": lambda v: f"{v * 100:.1f} %",
        "Sharpe": lambda v: f"{v:.2f}",
        "Drawdown max": lambda v: f"{v * 100:.1f} %",
    }
    affichage = tableau.copy().astype(object)
    for indicateur, formateur in formats.items():
        for colonne in colonnes:
            affichage.loc[indicateur, colonne] = formateur(tableau.loc[indicateur, colonne])
    affichage["Part du hasard battue"] = (tableau["Part du hasard battue"] * 100).map("{:.0f} %".format)
    affichage["p-valeur"] = tableau["p-valeur"].map("{:.3f}".format)
    return affichage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Le scoring fait-il mieux qu'un tirage au hasard ?")
    parser.add_argument("--profils", nargs="*", default=["prudent", "equilibre", "dynamique"])
    parser.add_argument("--nb", type=int, default=10, help="nombre d'actions par portefeuille")
    parser.add_argument("--n", type=int, default=1000, help="nombre de portefeuilles aléatoires")
    parser.add_argument("--frais", type=float, default=FRAIS_DEFAUT)
    parser.add_argument("--frequence", default="trimestriel")
    parser.add_argument("--appartenance", action="store_true",
                        help="n'autorise une action du S&P 500 qu'à partir de sa date d'entrée dans l'indice")
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()

    chemin = CHEMIN_BASE if CHEMIN_BASE.exists() else RACINE / "data" / "snapshot.db"
    conn = sqlite3.connect(chemin)
    prix = charger_prix(conn)
    secteurs = pd.read_sql_query("SELECT ticker, secteur FROM actions", conn).set_index("ticker")["secteur"]
    conn.close()

    eligibilite = None
    if args.appartenance:
        from src.sp_500 import eligibilite as charger_eligibilite
        eligibilite = charger_eligibilite()

    resultat = comparer_au_hasard(
        prix, secteurs, profils=args.profils, nb_actions=args.nb, n_simulations=args.n,
        frais=args.frais, frequence=args.frequence, eligibilite=eligibilite, graine=args.graine)

    print(f"\n{args.n} portefeuilles aléatoires de {args.nb} actions, tirés parmi "
          f"{resultat['nb_candidats_moyen']:.0f} actions candidates en moyenne.")
    for profil, tableau in resultat["positions"].items():
        print(f"\n=== Profil « {profil} » : sélection par le score (poids égaux) contre le hasard ===\n")
        print(_formater(tableau).to_string())

    reference = resultat["equipondere"]
    print("\nPour mémoire, portefeuille équipondéré de tout l'univers : "
          f"rendement {reference['Rendement annualisé']:+.1%}, volatilité {reference['Volatilité annualisée']:.1%}, "
          f"Sharpe {reference['Sharpe']:.2f}, drawdown {reference['Drawdown max']:.1%}")
    print("\n« Part du hasard battue » : part des portefeuilles aléatoires que la stratégie dépasse "
          "(pour la volatilité et le drawdown, « battre » veut dire faire moins mal).")
    print("« p-valeur » : part des portefeuilles aléatoires au moins aussi bons que la stratégie.")
    print("Une p-valeur élevée signifie que la stratégie ne se distingue pas du hasard.")

    sortie = RACINE / "data" / f"hasard{SUFFIXE_BASE}{'_appartenance' if args.appartenance else ''}.csv"
    pd.concat(resultat["positions"], names=["profil", "indicateur"]).to_csv(sortie)
    print(f"\nRésultats enregistrés : {sortie}")
    print("\n⚠️  Une seule période, un univers soumis au biais de survie, plusieurs comparaisons à la fois. "
          "Projet pédagogique, ceci n'est pas un conseil en investissement.")