"""Tests des indicateurs : on vérifie sur de petits exemples dont on connaît la réponse.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicateurs import (
    beta, calculer_indicateurs, drawdown_max, indice_de_reference, momentum,
    rendement_annualise, rendements, ratio_sharpe, volatilite_annualisee,
)


def serie(valeurs):
    """Série de prix avec des jours ouvrés consécutifs."""
    index = pd.bdate_range("2020-01-01", periods=len(valeurs))
    return pd.Series(valeurs, index=index, dtype=float)


# --- rendements -------------------------------------------------------------

def test_rendements_simples():
    r = rendements(serie([100, 110, 99]))
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)


# --- rendement annualisé ----------------------------------------------------

def test_rendement_annualise_prix_double_en_un_an():
    # 252 rendements quotidiens identiques qui doublent le prix : +100 % par an
    prix = serie(100 * 2 ** (np.arange(253) / 252))
    assert rendement_annualise(prix) == pytest.approx(1.0)


def test_rendement_annualise_historique_trop_court():
    assert np.isnan(rendement_annualise(serie([100])))


# --- volatilité -------------------------------------------------------------

def test_volatilite_nulle_si_croissance_constante():
    prix = serie(100 * 1.001 ** np.arange(300))
    assert volatilite_annualisee(prix) == pytest.approx(0.0, abs=1e-12)


def test_volatilite_rendements_alternes():
    # rendements +1 % / -1 % en alternance : écart-type = 1 % x racine(n/(n-1))
    n = 1000
    facteurs = np.where(np.arange(n) % 2 == 0, 1.01, 0.99)
    prix = serie(np.concatenate([[100.0], 100 * np.cumprod(facteurs)]))
    attendu = 0.01 * np.sqrt(n / (n - 1)) * np.sqrt(252)
    # les rendements réels valent +1 % puis -0,99 %... on vérifie l'ordre de grandeur
    assert volatilite_annualisee(prix) == pytest.approx(attendu, rel=0.02)


# --- Sharpe -----------------------------------------------------------------

def test_sharpe_indefini_si_volatilite_nulle():
    prix = serie(100 * 1.001 ** np.arange(300))
    assert np.isnan(ratio_sharpe(prix))


def test_sharpe_positif_si_rendement_positif():
    rng = np.random.default_rng(0)
    prix = serie(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 500)))
    assert ratio_sharpe(prix) > 0


# --- drawdown ---------------------------------------------------------------

def test_drawdown_max():
    # plus haut à 120, puis 60 : baisse de 50 %
    assert drawdown_max(serie([100, 120, 60, 90])) == pytest.approx(-0.5)


def test_drawdown_nul_si_prix_croissant():
    assert drawdown_max(serie([100, 101, 102, 103])) == pytest.approx(0.0)


# --- momentum ---------------------------------------------------------------

def test_momentum():
    prix = serie(np.arange(100, 400))        # 100, 101, ..., 399
    # il y a 126 jours : 399 - 126 = 273
    assert momentum(prix, 126) == pytest.approx(399 / 273 - 1)


def test_momentum_historique_insuffisant():
    assert np.isnan(momentum(serie([100, 101, 102]), 126))


# --- bêta -------------------------------------------------------------------

def _indice_et_action(coefficient, constante=0.0):
    rng = np.random.default_rng(1)
    r_indice = rng.normal(0.0005, 0.01, 400)
    indice = serie(100 * np.cumprod(1 + r_indice))
    action = serie(100 * np.cumprod(1 + coefficient * r_indice + constante))
    return action, indice


def test_beta_egal_a_un_pour_lindice_lui_meme():
    _, indice = _indice_et_action(1.0)
    assert beta(indice, indice) == pytest.approx(1.0)


def test_beta_double_si_action_amplifie_par_deux():
    action, indice = _indice_et_action(2.0)
    assert beta(action, indice) == pytest.approx(2.0, rel=1e-6)


def test_beta_ignore_une_constante():
    action, indice = _indice_et_action(0.5, constante=0.0003)
    assert beta(action, indice) == pytest.approx(0.5, rel=1e-6)


# --- indice de référence ----------------------------------------------------

def test_indice_de_reference():
    assert indice_de_reference("MC.PA") == "^FCHI"
    assert indice_de_reference("AAPL") == "^GSPC"


# --- tableau complet --------------------------------------------------------

def _prix_test(n=900, saut_apres=None):
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2018-01-01", periods=n)
    marche = rng.normal(0.0003, 0.01, n)
    donnees = {
        "^FCHI": 100 * np.cumprod(1 + marche),
        "^GSPC": 100 * np.cumprod(1 + marche),
        "A.PA": 100 * np.cumprod(1 + 1.5 * marche),
        "B": 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n)),
    }
    prix = pd.DataFrame(donnees, index=index)
    if saut_apres is not None:                      # choc brutal sur B après une date
        prix.loc[prix.index > saut_apres, "B"] *= 5
    return prix


def test_tableau_a_les_bonnes_colonnes_et_exclut_les_indices():
    t = calculer_indicateurs(_prix_test())
    assert set(t.index) == {"A.PA", "B"}
    for col in ["rendement_annuel", "volatilite_annuelle", "sharpe", "beta",
                "drawdown_max", "momentum_6m", "momentum_12m"]:
        assert col in t.columns


def test_beta_utilise_lindice_du_bon_pays():
    t = calculer_indicateurs(_prix_test())
    assert t.loc["A.PA", "beta"] == pytest.approx(1.5, rel=1e-6)


def test_exclut_les_actions_sans_assez_dhistorique():
    prix = _prix_test()
    prix.loc[prix.index[:700], "B"] = np.nan        # B n'a que 200 jours de données
    t = calculer_indicateurs(prix)
    assert "B" not in t.index
    assert "A.PA" in t.index


def test_pas_de_fuite_de_donnees_futures():
    """Un choc survenu APRÈS la date de calcul ne doit pas changer les indicateurs."""
    date = "2020-06-30"
    sans_choc = calculer_indicateurs(_prix_test(), date=date)
    avec_choc = calculer_indicateurs(_prix_test(saut_apres=date), date=date)
    pd.testing.assert_frame_equal(sans_choc, avec_choc)