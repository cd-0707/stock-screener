"""Tests de l'expérience « la prévision de volatilité aide-t-elle le portefeuille ? ».

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experience_prevision import (
    comparer, difference_par_bootstrap, recouvrement, rendement_annuel_moyen,
    sharpe_arithmetique,
)
from tests.test_backtest import PARAMETRES, SECTEUR_PAR_TICKER, prix_synthetiques


def rendements_aleatoires(n=1000, moyenne=0.0004, ecart=0.01, graine=0):
    return np.random.default_rng(graine).normal(moyenne, ecart, n)


# --- bootstrap ---------------------------------------------------------------

def test_series_identiques_ecart_nul():
    r = rendements_aleatoires()
    ecart, bas, haut = difference_par_bootstrap(r, r, rendement_annuel_moyen, n_tirages=200)
    assert ecart == 0 and bas == 0 and haut == 0


def test_un_vrai_ecart_est_detecte():
    """Une stratégie qui gagne 0,1 % de plus chaque jour : l'intervalle exclut zéro."""
    r = rendements_aleatoires()
    ecart, bas, haut = difference_par_bootstrap(r + 0.001, r, rendement_annuel_moyen, n_tirages=300)
    assert ecart == pytest.approx(0.001 * 252)
    assert bas > 0


def test_un_ecart_du_au_hasard_contient_zero():
    """Deux séries sans lien : leur écart observé n'est que du bruit, l'intervalle contient zéro."""
    a = rendements_aleatoires(graine=1)
    b = rendements_aleatoires(graine=2)
    _, bas, haut = difference_par_bootstrap(a, b, rendement_annuel_moyen, n_tirages=500)
    assert bas < 0 < haut


def test_bootstrap_reproductible_et_longueurs_verifiees():
    a, b = rendements_aleatoires(graine=1), rendements_aleatoires(graine=2)
    premier = difference_par_bootstrap(a, b, sharpe_arithmetique, n_tirages=100)
    second = difference_par_bootstrap(a, b, sharpe_arithmetique, n_tirages=100)
    assert premier == second
    with pytest.raises(ValueError):
        difference_par_bootstrap(a, b[:-1], sharpe_arithmetique)


def test_sharpe_arithmetique_serie_constante():
    assert np.isnan(sharpe_arithmetique(np.full(100, 0.001)))


# --- recouvrement ------------------------------------------------------------

def test_recouvrement_de_portefeuilles():
    dates = pd.to_datetime(["2020-03-31", "2020-06-30"])
    a = pd.DataFrame({"X": [0.5, 1.0], "Y": [0.5, 0.0]}, index=dates)
    b = pd.DataFrame({"X": [0.5, 0.0], "Z": [0.5, 1.0]}, index=dates)
    assert recouvrement(a, b).tolist() == [0.5, 0.0]
    assert recouvrement(a, a).tolist() == [1.0, 1.0]


# --- expérience complète -----------------------------------------------------

def test_comparer_sur_donnees_simulees():
    prix = prix_synthetiques()
    parametres = {k: v for k, v in PARAMETRES.items() if k != "profil"}
    resultat = comparer(prix, SECTEUR_PAR_TICKER, profils=("equilibre",), modele="har", **parametres)

    assert resultat["n_previsions"] > 0
    assert len(resultat["performance"]) == 2
    diff = resultat["differences"].loc["equilibre"]
    assert diff["IC 90 % bas"] <= diff["IC 90 % haut"]
    assert 0 <= diff["Recouvrement moyen"] <= 1