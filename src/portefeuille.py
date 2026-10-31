"""Construction de portefeuille : de la liste d'actions recommandées aux poids à investir.

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.portefeuille --profil prudent
    python -m src.portefeuille --profil dynamique --nb 8 --poids-max 0.25
    python -m src.portefeuille --profil equilibre --methode equipondere

Trois méthodes de répartition :
  - equipondere  : le même poids pour chaque action (référence simple) ;
  - variance_min : le portefeuille le moins risqué possible ;
  - sharpe_max   : le meilleur rapport rendement/risque.

Contraintes : pas de vente à découvert (poids >= 0), somme des poids = 1 et un
plafond par action (20 % par défaut) pour éviter un portefeuille concentré.

Deux précautions techniques, car l'optimisation de Markowitz est fragile :
  - la covariance est estimée avec un « rétrécissement » de Ledoit-Wolf, bien
    plus stable que la covariance brute sur peu d'observations ;
  - les rendements moyens passés sont très bruités : on les rapproche de leur
    moyenne générale (rétrécissement de 50 %) pour éviter qu'un seul coup de
    chance passé ne domine tout le portefeuille.
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.collecte import CHEMIN_BASE
from src.indicateurs import (
    FENETRE_DEFAUT, JOURS_PAR_AN, charger_prix, drawdown_max, ratio_sharpe,
    rendement_annualise, volatilite_annualisee,
)
from src.scoring import LIQUIDITE_MIN, _cle, charger_univers, recommander

log = logging.getLogger(__name__)

METHODES = ("equipondere", "variance_min", "sharpe_max")
METHODE_PAR_PROFIL = {"prudent": "variance_min", "equilibre": "sharpe_max", "dynamique": "sharpe_max"}
POIDS_MAX_DEFAUT = 0.20
RETRECISSEMENT_RENDEMENTS = 0.5
POIDS_NEGLIGEABLE = 1e-4


def calculer_rendements(prix, date=None, fenetre=FENETRE_DEFAUT):
    """Rendements quotidiens des `fenetre` derniers jours, jusqu'à `date` incluse."""
    if date is not None:
        prix = prix.loc[:pd.Timestamp(date)]
    prix = prix.tail(fenetre + 1)
    return (prix / prix.shift(1) - 1).iloc[1:]


def optimiser(rendements, methode="sharpe_max", poids_max=POIDS_MAX_DEFAUT, taux_sans_risque=0.0):
    """Poids optimaux (Series indexée par ticker, somme = 1) pour les rendements fournis."""
    if methode not in METHODES:
        raise ValueError(f"Méthode inconnue : {methode!r}. Choix possibles : {list(METHODES)}")

    tickers = list(rendements.columns)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1 or methode == "equipondere":
        return pd.Series(1.0 / n, index=tickers)

    donnees = rendements.fillna(0.0).to_numpy()
    covariance = LedoitWolf().fit(donnees).covariance_ * JOURS_PAR_AN
    mu = donnees.mean(axis=0) * JOURS_PAR_AN
    mu = (1 - RETRECISSEMENT_RENDEMENTS) * mu + RETRECISSEMENT_RENDEMENTS * mu.mean()

    # Le plafond doit permettre d'investir 100 % : avec 4 actions, 20 % maximum est impossible.
    plafond = max(poids_max, 1.0 / n + 1e-9)
    depart = np.full(n, 1.0 / n)

    if methode == "variance_min":
        objectif = lambda w: w @ covariance @ w
    else:
        objectif = lambda w: -(w @ mu - taux_sans_risque) / np.sqrt(w @ covariance @ w)

    resultat = minimize(
        objectif, depart, method="SLSQP",
        bounds=[(0.0, plafond)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500},
    )
    if not resultat.success:
        log.warning("Optimisation non convergée (%s) : retour à l'équipondération.", resultat.message)
        return pd.Series(depart, index=tickers)

    poids = np.clip(resultat.x, 0.0, None)
    poids[poids < POIDS_NEGLIGEABLE] = 0.0
    poids = poids / poids.sum()
    return pd.Series(poids, index=tickers)


def statistiques_portefeuille(poids, rendements):
    """Caractéristiques historiques du portefeuille s'il avait gardé ces poids constants."""
    poids = poids[poids > 0]
    r = (rendements[poids.index].fillna(0.0) * poids).sum(axis=1)
    valeur = pd.Series(np.concatenate([[1.0], (1 + r).cumprod().to_numpy()]))
    return {
        "rendement_annuel": rendement_annualise(valeur),
        "volatilite_annuelle": volatilite_annualisee(valeur),
        "sharpe": ratio_sharpe(valeur),
        "drawdown_max": drawdown_max(valeur),
    }


def construire_portefeuille(tableau, rendements, profil, horizon="moyen", nb_actions=10,
                            max_par_secteur=None, poids_max=POIDS_MAX_DEFAUT, methode=None,
                            **contraintes):
    """Sélectionne les actions (scoring) puis calcule leurs poids (optimisation).

    tableau     : tableau d'indicateurs de l'univers (voir scoring.charger_univers)
    rendements  : rendements quotidiens récents de l'univers
    contraintes : secteurs, prix_min, prix_max, dividendes, liquidite_min

    Renvoie (poids, classement) : les poids non nuls, et le classement expliqué
    des actions retenues avec une colonne « poids ».
    """
    classement = recommander(profil, horizon=horizon, nb_actions=nb_actions,
                             max_par_secteur=max_par_secteur, tableau=tableau, **contraintes)
    tickers = [t for t in classement.index if t in rendements.columns]
    if not tickers:
        return pd.Series(dtype=float), classement

    methode = methode or METHODE_PAR_PROFIL[_cle(profil)]
    poids = optimiser(rendements[tickers], methode=methode, poids_max=poids_max)
    classement["poids"] = poids.reindex(classement.index).fillna(0.0)
    return poids[poids > 0], classement


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Portefeuille recommandé selon un profil.")
    parser.add_argument("--profil", default="equilibre", help="prudent, equilibre ou dynamique")
    parser.add_argument("--horizon", default="moyen", help="court, moyen ou long")
    parser.add_argument("--methode", default=None, choices=METHODES,
                        help="défaut : selon le profil")
    parser.add_argument("--secteurs", nargs="*", default=None)
    parser.add_argument("--prix-min", type=float, default=None)
    parser.add_argument("--prix-max", type=float, default=None)
    parser.add_argument("--dividendes", action="store_true")
    parser.add_argument("--nb", type=int, default=10)
    parser.add_argument("--max-par-secteur", type=int, default=None)
    parser.add_argument("--poids-max", type=float, default=POIDS_MAX_DEFAUT)
    parser.add_argument("--liquidite-min", type=float, default=LIQUIDITE_MIN)
    args = parser.parse_args()

    conn = sqlite3.connect(CHEMIN_BASE)
    univers = charger_univers(conn)
    rendements = calculer_rendements(charger_prix(conn))
    conn.close()

    poids, classement = construire_portefeuille(
        univers, rendements, args.profil, horizon=args.horizon, nb_actions=args.nb,
        max_par_secteur=args.max_par_secteur, poids_max=args.poids_max, methode=args.methode,
        secteurs=args.secteurs, prix_min=args.prix_min, prix_max=args.prix_max,
        dividendes=args.dividendes, liquidite_min=args.liquidite_min,
    )

    if poids.empty:
        print("\nAucune action ne respecte ces contraintes.")
    else:
        methode = args.methode or METHODE_PAR_PROFIL[_cle(args.profil)]
        print(f"\nPortefeuille « {args.profil} » (méthode : {methode}, plafond {args.poids_max:.0%} par action)\n")
        affichage = classement[["nom", "secteur", "famille", "score", "poids"]].copy()
        affichage["nom"] = affichage["nom"].str.slice(0, 28)
        affichage["poids"] = (affichage["poids"] * 100).round(1).astype(str) + " %"
        print(affichage.to_string())

        stats = statistiques_portefeuille(poids, rendements)
        print("\nCaractéristiques historiques (3 dernières années, poids constants) :")
        print(f"  rendement annualisé : {stats['rendement_annuel']:+.1%}")
        print(f"  volatilité annualisée : {stats['volatilite_annuelle']:.1%}")
        print(f"  ratio de Sharpe : {stats['sharpe']:.2f}")
        print(f"  drawdown maximal : {stats['drawdown_max']:.1%}")
    print("\n  Projet pédagogique : ceci n'est pas un conseil en investissement.")