"""La prévision de volatilité améliore-t-elle les portefeuilles recommandés ?

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.experience_prevision
    python -m src.experience_prevision --profils prudent --modele har

Principe : on lance deux backtests identiques pour chaque profil (mêmes dates, mêmes frais,
mêmes règles). Une seule chose change : le critère « volatilité » du score.
  - Version de base : volatilité passée mesurée sur les 3 dernières années ;
  - Version « prévision » : volatilité prévue pour le mois suivant par un modèle HAR
    (voir prediction.py), réentraîné à chaque date avec les seuls prix disponibles alors.

Pour savoir si l'écart est autre chose que du hasard, on calcule des intervalles de
confiance par bootstrap par blocs : on retire au hasard, par blocs de 21 jours pour
conserver la dépendance entre jours voisins, des journées de rendements et on recalcule
l'écart de performance des deux versions. Si l'intervalle contient zéro, l'écart observé
ne permet pas de conclure que la prévision aide.

Limites : même période et même univers que le backtest principal ; une seule réalisation
du passé ; l'intervalle de confiance suppose que les blocs de 21 jours sont à peu près
indépendants entre eux.
"""

import argparse
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.backtest import FRAIS_DEFAUT, HISTORIQUE_MIN, dates_de_rebalancement, lancer_backtest
from src.collecte import CHEMIN_BASE, RACINE
from src.indicateurs import charger_prix
from src.portefeuille import POIDS_MAX_DEFAUT
from src.prediction import previsions_aux_dates

log = logging.getLogger(__name__)

JOURS_PAR_AN = 252


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def rendement_annuel_moyen(rendements):
    """Rendement annualisé arithmétique (moyenne quotidienne x 252)."""
    return float(np.mean(rendements) * JOURS_PAR_AN)


def sharpe_arithmetique(rendements):
    """Ratio de Sharpe à partir des rendements quotidiens, taux sans risque nul."""
    r = np.asarray(rendements, dtype=float)
    ecart_type = r.std(ddof=1)
    # Seuil et non « == 0 » : une série constante donne un écart-type d'environ 1e-19 (arrondi).
    return np.nan if ecart_type < 1e-12 else float(r.mean() / ecart_type * np.sqrt(JOURS_PAR_AN))


def difference_par_bootstrap(rendements_a, rendements_b, statistique, longueur_bloc=21,
                             n_tirages=2000, niveau=0.90, graine=0):
    """Écart statistique(a) - statistique(b), et son intervalle de confiance par bootstrap par blocs.

    Les mêmes journées sont tirées pour a et pour b (tirage « apparié »), car les deux
    stratégies sont exposées aux mêmes marchés : ce qui compte est leur écart.
    Renvoie (écart observé, borne basse, borne haute).
    """
    a = np.asarray(rendements_a, dtype=float)
    b = np.asarray(rendements_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Les deux séries de rendements doivent avoir la même longueur.")
    n = len(a)
    nb_blocs = int(np.ceil(n / longueur_bloc))
    decalages = np.arange(longueur_bloc)
    hasard = np.random.default_rng(graine)

    ecarts = np.empty(n_tirages)
    for i in range(n_tirages):
        debuts = hasard.integers(0, n - longueur_bloc + 1, nb_blocs)
        indices = (debuts[:, None] + decalages).ravel()[:n]
        ecarts[i] = statistique(a[indices]) - statistique(b[indices])

    bas, haut = np.nanpercentile(ecarts, [(1 - niveau) / 2 * 100, (1 + niveau) / 2 * 100])
    return statistique(a) - statistique(b), float(bas), float(haut)


def recouvrement(composition_a, composition_b):
    """Part du portefeuille commune aux deux versions à chaque date (1 = portefeuilles identiques)."""
    colonnes = composition_a.columns.union(composition_b.columns)
    a = composition_a.reindex(columns=colonnes, fill_value=0.0)
    b = composition_b.reindex(index=a.index, columns=colonnes, fill_value=0.0)
    return np.minimum(a, b).sum(axis=1)


# ---------------------------------------------------------------------------
# Expérience
# ---------------------------------------------------------------------------

def comparer(prix, secteurs, profils=("prudent", "equilibre", "dynamique"), modele="har_etendu",
             frequence="trimestriel", historique_min=HISTORIQUE_MIN, **parametres):
    """Compare, pour chaque profil, le backtest de base et le backtest avec volatilité prévue.

    Renvoie {"performance": tableau des deux versions, "differences": écarts avec intervalles
    de confiance, "n_previsions": nombre de dates où une prévision a pu être faite}.
    """
    dates = dates_de_rebalancement(prix.index, frequence, historique_min)
    previsions = previsions_aux_dates(prix, dates, modele=modele)
    log.info("Prévisions disponibles à %d dates de rééquilibrage sur %d", len(previsions), len(dates) - 1)

    performance, differences = {}, {}
    for profil in profils:
        log.info("Profil « %s » : backtest de base puis avec prévision", profil)
        reglages = dict(frequence=frequence, historique_min=historique_min, **parametres)
        base = lancer_backtest(prix, secteurs, profil, **reglages)
        avec = lancer_backtest(prix, secteurs, profil, previsions_volatilite=previsions, **reglages)

        performance[(profil, "Volatilité passée (3 ans)")] = base["metriques"].loc["Stratégie"]
        performance[(profil, "Volatilité prévue (HAR)")] = avec["metriques"].loc["Stratégie"]

        r_base = base["courbes"]["Stratégie"].pct_change().dropna()
        r_avec = avec["courbes"]["Stratégie"].pct_change().reindex(r_base.index).fillna(0.0)
        ecart_rend, rend_bas, rend_haut = difference_par_bootstrap(r_avec, r_base, rendement_annuel_moyen)
        ecart_sharpe, sh_bas, sh_haut = difference_par_bootstrap(r_avec, r_base, sharpe_arithmetique)
        differences[profil] = {
            "Δ rendement (pts/an)": ecart_rend * 100,
            "IC 90 % bas": rend_bas * 100,
            "IC 90 % haut": rend_haut * 100,
            "Δ Sharpe": ecart_sharpe,
            "Sharpe IC bas": sh_bas,
            "Sharpe IC haut": sh_haut,
            "Recouvrement moyen": recouvrement(base["compositions"], avec["compositions"]).mean(),
        }

    return {
        "performance": pd.DataFrame(performance).T,
        "differences": pd.DataFrame(differences).T,
        "n_previsions": len(previsions),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="La prévision de volatilité améliore-t-elle les portefeuilles ?")
    parser.add_argument("--profils", nargs="*", default=["prudent", "equilibre", "dynamique"])
    parser.add_argument("--modele", default="har_etendu", choices=["har", "har_etendu", "ml"])
    parser.add_argument("--nb", type=int, default=10)
    parser.add_argument("--frais", type=float, default=FRAIS_DEFAUT)
    parser.add_argument("--frequence", default="trimestriel")
    args = parser.parse_args()

    chemin = CHEMIN_BASE if CHEMIN_BASE.exists() else RACINE / "data" / "snapshot.db"
    conn = sqlite3.connect(chemin)
    prix = charger_prix(conn)
    secteurs = pd.read_sql_query("SELECT ticker, secteur FROM actions", conn).set_index("ticker")["secteur"]
    conn.close()

    resultat = comparer(prix, secteurs, profils=args.profils, modele=args.modele,
                        nb_actions=args.nb, frais=args.frais, frequence=args.frequence,
                        poids_max=POIDS_MAX_DEFAUT)

    performance = resultat["performance"].copy()
    for colonne in ["Rendement total", "Rendement annualisé"]:
        performance[colonne] = (performance[colonne] * 100).map("{:+.1f} %".format)
    for colonne in ["Volatilité annualisée", "Drawdown max"]:
        performance[colonne] = (performance[colonne] * 100).map("{:.1f} %".format)
    performance["Sharpe"] = performance["Sharpe"].map("{:.2f}".format)
    print("\n1. Performance du backtest, avec ou sans volatilité prévue :\n")
    print(performance.to_string())

    pd.options.display.float_format = "{:,.2f}".format
    pd.options.display.width = 200
    print("\n2. Écart (prévision - base) avec intervalle de confiance à 90 % par bootstrap :\n")
    print(resultat["differences"].to_string())
    print("\nSi un intervalle contient 0, l'écart observé ne permet pas de conclure à un effet de la prévision.")

    sortie = RACINE / "data" / "experience_prevision.csv"
    resultat["differences"].to_csv(sortie)
    print(f"\nÉcarts enregistrés : {sortie}")
    print("\n⚠️  Une seule période et un univers soumis au biais de survie. Projet pédagogique, "
          "ceci n'est pas un conseil en investissement.")