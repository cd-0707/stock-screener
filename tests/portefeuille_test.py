"""Tests de l'optimisation de portefeuille.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.portefeuille import (
    METHODES, calculer_rendements, construire_portefeuille, optimiser,
    statistiques_portefeuille,
)
from tests.test_scoring import univers


def rendements_test(n_actifs=6, n_jours=500, seed=0):
    """Rendements fictifs : chaque actif a sa propre volatilité."""
    rng = np.random.default_rng(seed)
    volatilites = np.linspace(0.005, 0.03, n_actifs)
    donnees = rng.normal(0.0003, 1.0, (n_jours, n_actifs)) * volatilites
    return pd.DataFrame(donnees, columns=[f"A{i}" for i in range(n_actifs)])


@pytest.mark.parametrize("methode", METHODES)
def test_poids_valides_pour_chaque_methode(methode):
    poids = optimiser(rendements_test(8), methode=methode, poids_max=0.25)
    assert poids.sum() == pytest.approx(1.0)
    assert (poids >= 0).all()
    assert (poids <= 0.25 + 1e-3).all()


def test_equipondere():
    poids = optimiser(rendements_test(5), methode="equipondere")
    assert np.allclose(poids, 0.2)


def test_variance_min_moins_risque_que_equipondere():
    r = rendements_test(6)
    cov = r.cov().to_numpy()
    w_min = optimiser(r, "variance_min", poids_max=1.0).to_numpy()
    w_eq = optimiser(r, "equipondere").to_numpy()
    assert w_min @ cov @ w_min < w_eq @ cov @ w_eq


def test_variance_min_favorise_les_actifs_calmes():
    poids = optimiser(rendements_test(6), "variance_min", poids_max=1.0)
    assert poids["A0"] > poids["A5"]


def test_plafond_atteint_par_un_actif_dominant():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0.0, 0.02, (600, 4)), columns=list("ABCD"))
    r["A"] = rng.normal(0.003, 0.004, 600)          # rendement élevé, risque faible
    poids = optimiser(r, "sharpe_max", poids_max=0.4)
    assert poids["A"] == pytest.approx(0.4, abs=1e-3)


def test_plafond_ajuste_quand_il_est_impossible():
    """3 actions avec un plafond de 20 % : impossible d'investir 100 %, on relâche le plafond."""
    poids = optimiser(rendements_test(3), "variance_min", poids_max=0.2)
    assert poids.sum() == pytest.approx(1.0)


def test_un_seul_actif_recoit_tout():
    poids = optimiser(rendements_test(1))
    assert poids.iloc[0] == pytest.approx(1.0)


def test_methode_inconnue():
    with pytest.raises(ValueError):
        optimiser(rendements_test(3), methode="magique")


def test_statistiques_dun_actif_unique():
    r = rendements_test(3)
    poids = pd.Series({"A1": 1.0})
    stats = statistiques_portefeuille(poids, r)
    attendu = r["A1"].std() * np.sqrt(252)
    assert stats["volatilite_annuelle"] == pytest.approx(attendu, rel=1e-6)
    assert -1 <= stats["drawdown_max"] <= 0


def test_calculer_rendements_fenetre_et_date():
    index = pd.bdate_range("2020-01-01", periods=100)
    prix = pd.DataFrame({"X": np.arange(100.0, 200.0)}, index=index)
    r = calculer_rendements(prix, date=index[50], fenetre=10)
    assert len(r) == 10
    assert r.index.max() == index[50]


def test_construire_portefeuille_poids_et_classement():
    t = univers()
    rendements = rendements_test(0)  # remplacé ci-dessous
    rng = np.random.default_rng(3)
    rendements = pd.DataFrame(rng.normal(0.0003, 0.012, (400, len(t))), columns=t.index)
    poids, classement = construire_portefeuille(t, rendements, "prudent", nb_actions=8)
    assert len(classement) == 8
    assert poids.sum() == pytest.approx(1.0)
    assert set(poids.index) <= set(classement.index)
    assert classement["poids"].sum() == pytest.approx(1.0)


def test_construire_portefeuille_sans_candidat():
    t = univers()
    rendements = pd.DataFrame(np.zeros((10, len(t))), columns=t.index)
    poids, classement = construire_portefeuille(t, rendements, "prudent", prix_min=10_000)
    assert poids.empty and classement.empty
    