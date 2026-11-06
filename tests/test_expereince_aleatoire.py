"""Tests du test des portefeuilles aléatoires.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experience_aleatoire import (
    SENS, comparer_au_hasard, indicateurs_courbe, poids_egaux, positionner,
)


def prix_avec_gagnantes(drift, n=1000, nb=16, nb_gagnants=4, seed=0):
    """Actions corrélées à un marché ; les `nb_gagnants` premières ont en plus un gain quotidien `drift`."""
    rng = np.random.default_rng(seed)
    marche = rng.normal(0.0002, 0.008, n)
    colonnes = {}
    for i in range(nb):
        gain = drift if i < nb_gagnants else 0.0
        colonnes[f"S{i}"] = 100 * np.cumprod(1 + marche + gain + rng.normal(0, 0.01, n))
    colonnes["^GSPC"] = 100 * np.cumprod(1 + marche)
    colonnes["^FCHI"] = 100 * np.cumprod(1 + marche)
    return pd.DataFrame(colonnes, index=pd.bdate_range("2016-01-01", periods=n))


SECTEURS = pd.Series({f"S{i}": "A" for i in range(16)})
PARAMETRES = dict(profils=("dynamique",), nb_actions=4, n_simulations=120, historique_min=300)


# --- fonctions de base ------------------------------------------------------

def test_poids_egaux():
    assert poids_egaux(["A", "B", "C", "D"]).tolist() == [0.25] * 4
    assert poids_egaux([]).empty


def test_positionner_sur_une_distribution_connue():
    """100 portefeuilles aléatoires dont les rendements vont de 0 à 0,99 : une stratégie à 0,50 en bat la moitié."""
    aleatoires = pd.DataFrame({nom: np.arange(100) / 100 for nom in SENS})
    strategie = {nom: 0.505 for nom in SENS}
    position = positionner(strategie, aleatoires)
    assert position.loc["Rendement annualisé", "Part du hasard battue"] == pytest.approx(0.51)
    assert position.loc["Rendement annualisé", "p-valeur"] == pytest.approx((1 + 49) / 101)
    # volatilité : plus c'est bas, mieux c'est, donc la lecture est inversée
    assert position.loc["Volatilité annualisée", "Part du hasard battue"] == pytest.approx(0.49)


def test_une_strategie_excellente_a_une_p_valeur_minimale():
    aleatoires = pd.DataFrame({nom: np.arange(100) / 100 for nom in SENS})
    position = positionner({nom: 10.0 if SENS[nom] > 0 else -10.0 for nom in SENS}, aleatoires)
    assert (position["Part du hasard battue"] == 1.0).all()
    assert np.allclose(position["p-valeur"], 1 / 101)


def test_indicateurs_courbe():
    courbe = pd.Series(2 ** (np.arange(253) / 252))
    resultat = indicateurs_courbe(courbe)
    assert set(resultat) == set(SENS)
    assert resultat["Rendement annualisé"] == pytest.approx(1.0)


# --- l'expérience complète --------------------------------------------------

def test_detecte_un_vrai_talent():
    """Quatre actions gagnent durablement : un score fondé sur la performance passée doit les trouver."""
    prix = prix_avec_gagnantes(drift=0.001)
    resultat = comparer_au_hasard(prix, SECTEURS, **PARAMETRES)
    position = resultat["positions"]["dynamique"]
    assert position.loc["Rendement annualisé", "Part du hasard battue"] > 0.9
    assert position.loc["Rendement annualisé", "p-valeur"] < 0.1


def test_ne_trouve_rien_quand_il_ny_a_rien_a_trouver():
    """Aucune action n'est meilleure qu'une autre : la stratégie est indiscernable du hasard."""
    prix = prix_avec_gagnantes(drift=0.0)
    resultat = comparer_au_hasard(prix, SECTEURS, **PARAMETRES)
    position = resultat["positions"]["dynamique"]
    assert 0.02 < position.loc["Rendement annualisé", "Part du hasard battue"] < 0.98


def test_resultats_reproductibles_et_forme_des_sorties():
    prix = prix_avec_gagnantes(drift=0.0005)
    a = comparer_au_hasard(prix, SECTEURS, graine=7, **PARAMETRES)
    b = comparer_au_hasard(prix, SECTEURS, graine=7, **PARAMETRES)
    assert a["aleatoires"].equals(b["aleatoires"])
    assert len(a["aleatoires"]) == PARAMETRES["n_simulations"]
    assert set(a["aleatoires"].columns) == set(SENS)
    assert a["nb_candidats_moyen"] == pytest.approx(16.0)