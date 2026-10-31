"""Calcul des indicateurs financiers de chaque action.

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.indicateurs                    # indicateurs à aujourd'hui
    python -m src.indicateurs --date 2022-06-30  # indicateurs tels qu'ils étaient à cette date

Principe important : chaque fonction de calcul ne reçoit QUE des prix. La
fonction calculer_indicateurs() peut les couper à une date donnée : à l'étape 6
(backtest), on pourra ainsi calculer les indicateurs « comme si on était » à
une date passée, sans jamais utiliser d'information future.

Conventions :
- les prix utilisés sont les clôtures ajustées (splits et dividendes inclus) ;
- une année compte 252 jours de bourse ;
- le taux sans risque est de 0 % par défaut (simplification, à documenter).
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.collecte import CHEMIN_BASE
from src.univers import INDICES

log = logging.getLogger(__name__)

JOURS_PAR_AN = 252
FENETRE_DEFAUT = 756   # 3 ans de bourse : période sur laquelle on mesure le risque
MIN_OBS = 252          # historique minimal (1 an) pour calculer des indicateurs fiables


# ---------------------------------------------------------------------------
# Indicateurs unitaires (chacun prend une série de prix indexée par date)
# ---------------------------------------------------------------------------

def rendements(prix):
    """Rendements quotidiens simples : prix du jour / prix de la veille - 1."""
    return prix.dropna().pct_change().dropna()


def rendement_annualise(prix):
    """Rendement annualisé composé (CAGR) sur toute la période fournie."""
    prix = prix.dropna()
    n = len(prix) - 1
    if n < 1:
        return np.nan
    return (prix.iloc[-1] / prix.iloc[0]) ** (JOURS_PAR_AN / n) - 1


def volatilite_annualisee(prix):
    """Écart-type des rendements quotidiens, ramené à l'année (x racine de 252)."""
    r = rendements(prix)
    if len(r) < 2:
        return np.nan
    return r.std() * np.sqrt(JOURS_PAR_AN)


def ratio_sharpe(prix, taux_sans_risque=0.0):
    """(Rendement annualisé - taux sans risque) / volatilité annualisée."""
    vol = volatilite_annualisee(prix)
    # Seuil et non « == 0 » : une croissance parfaitement régulière donne une
    # volatilité de l'ordre de 1e-15 (bruit d'arrondi), pas exactement zéro.
    if not np.isfinite(vol) or vol < 1e-9:
        return np.nan
    return (rendement_annualise(prix) - taux_sans_risque) / vol


def drawdown_max(prix):
    """Pire baisse depuis un plus haut (nombre négatif : -0.5 = -50 %)."""
    prix = prix.dropna()
    if prix.empty:
        return np.nan
    return (prix / prix.cummax() - 1).min()


def momentum(prix, jours):
    """Performance sur les `jours` derniers jours de bourse."""
    prix = prix.dropna()
    if len(prix) <= jours:
        return np.nan
    return prix.iloc[-1] / prix.iloc[-1 - jours] - 1


def beta(prix_action, prix_indice):
    """Sensibilité de l'action au marché : covariance / variance de l'indice.

    Beta de 1 : bouge comme le marché ; > 1 : amplifie ; < 1 : amortit.
    """
    r = pd.concat(
        [rendements(prix_action), rendements(prix_indice)],
        axis=1, keys=["action", "indice"], join="inner",
    ).dropna()
    if len(r) < 2:
        return np.nan
    variance = r["indice"].var()
    if variance == 0:
        return np.nan
    return r["action"].cov(r["indice"]) / variance


def indice_de_reference(ticker):
    """Indice servant de référence pour le bêta : CAC 40 pour Paris/Amsterdam, sinon S&P 500."""
    return "^FCHI" if ticker.endswith((".PA", ".AS")) else "^GSPC"


# ---------------------------------------------------------------------------
# Calcul du tableau complet
# ---------------------------------------------------------------------------

def calculer_indicateurs(prix, date=None, fenetre=FENETRE_DEFAUT, min_obs=MIN_OBS):
    """Tableau « une ligne par action, une colonne par indicateur ».

    prix   : DataFrame (index = dates, colonnes = tickers) de clôtures ajustées
    date   : si fournie, seules les données jusqu'à cette date sont utilisées
    fenetre: nombre de jours de bourse pris en compte avant cette date
    min_obs: les actions avec moins d'historique que ça sont exclues
    """
    if date is not None:
        prix = prix.loc[:pd.Timestamp(date)]
    prix = prix.tail(fenetre + 1)

    lignes = {}
    for ticker in prix.columns:
        if ticker in INDICES:
            continue
        serie = prix[ticker].dropna()
        if len(serie) <= min_obs:
            continue
        ref = indice_de_reference(ticker)
        lignes[ticker] = {
            "rendement_annuel": rendement_annualise(serie),
            "volatilite_annuelle": volatilite_annualisee(serie),
            "sharpe": ratio_sharpe(serie),
            "beta": beta(serie, prix[ref]) if ref in prix.columns else np.nan,
            "drawdown_max": drawdown_max(serie),
            "momentum_6m": momentum(serie, 126),
            "momentum_12m": momentum(serie, 252),
            "nb_jours": len(serie),
        }
    return pd.DataFrame.from_dict(lignes, orient="index")


# ---------------------------------------------------------------------------
# Lecture de la base et assemblage final
# ---------------------------------------------------------------------------

def charger_prix(conn):
    """Prix ajustés en tableau date x ticker, avec les trous de calendrier comblés (5 jours max)."""
    long = pd.read_sql_query(
        "SELECT ticker, date, cloture_ajustee FROM prix", conn, parse_dates=["date"]
    )
    prix = long.pivot(index="date", columns="ticker", values="cloture_ajustee").sort_index()
    return prix.ffill(limit=5)


def construire_tableau(conn, date=None, avec_fondamentaux=True):
    """Indicateurs de prix + informations descriptives (+ fondamentaux si demandé).

    Les fondamentaux (PER, dividende) de la base sont ceux d'AUJOURD'HUI : ils ne
    sont pas historisés. Pour un calcul à une date passée (backtest), il faut donc
    mettre avec_fondamentaux=False, sinon on utilise de l'information du futur.
    """
    tableau = calculer_indicateurs(charger_prix(conn), date=date)

    actions = pd.read_sql_query(
        "SELECT ticker, nom, secteur, pays, devise FROM actions", conn
    ).set_index("ticker")
    tableau = tableau.join(actions)

    if avec_fondamentaux:
        fonda = pd.read_sql_query(
            "SELECT ticker, capitalisation, per, rendement_dividende FROM fondamentaux", conn
        ).set_index("ticker")
        tableau = tableau.join(fonda)

    tableau.index.name = "ticker"
    return tableau


def enregistrer(conn, tableau):
    """Écrit le tableau dans la table « indicateurs » de la base (remplacée à chaque fois)."""
    tableau.to_sql("indicateurs", conn, if_exists="replace", index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Calcul des indicateurs financiers.")
    parser.add_argument("--date", default=None, help="date d'arrêt (AAAA-MM-JJ), défaut : aujourd'hui")
    args = parser.parse_args()

    conn = sqlite3.connect(CHEMIN_BASE)
    tableau = construire_tableau(conn, date=args.date, avec_fondamentaux=args.date is None)
    if args.date is None:
        enregistrer(conn, tableau)
        log.info("Table « indicateurs » enregistrée dans la base.")
    conn.close()

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200
    log.info("%d actions avec assez d'historique.", len(tableau))
    print(tableau.drop(columns=["nom", "devise"], errors="ignore").head(10).to_string())