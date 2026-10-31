"""Tests du backtest.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import dates_de_rebalancement, lancer_backtest, metriques

SECTEURS = ["Technology", "Healthcare", "Energy", "Financial Services"]
PARAMETRES = dict(profil="equilibre", nb_actions=5, historique_min=300, fenetre=300)


def prix_synthetiques(n=1000, nb_actions=12, seed=0):
    """Actions fictives corrélées à un marché, plus deux indices."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-01", periods=n)
    marche = rng.normal(0.0003, 0.008, n)
    colonnes = {}
    for i in range(nb_actions):
        beta = rng.uniform(0.5, 1.5)
        colonnes[f"S{i}"] = 100 * np.cumprod(1 + beta * marche + rng.normal(0.0002, rng.uniform(0.005, 0.02), n))
    colonnes["^GSPC"] = 100 * np.cumprod(1 + marche)
    colonnes["^FCHI"] = 100 * np.cumprod(1 + marche)
    return pd.DataFrame(colonnes, index=index)


SECTEUR_PAR_TICKER = pd.Series({f"S{i}": SECTEURS[i % 4] for i in range(12)})


@pytest.fixture(scope="module")
def prix():
    return prix_synthetiques()


@pytest.fixture(scope="module")
def resultat(prix):
    return lancer_backtest(prix, SECTEUR_PAR_TICKER, **PARAMETRES)


# --- dates ------------------------------------------------------------------

def test_dates_de_rebalancement(prix):
    dates = dates_de_rebalancement(prix.index, "trimestriel", historique_min=300)
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates.isin(prix.index).all()
    assert dates[0] >= prix.index[300]
    ecarts = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)
    assert 60 <= np.median(ecarts) <= 100


def test_frequence_mensuelle_plus_dense_que_trimestrielle(prix):
    mensuelle = dates_de_rebalancement(prix.index, "mensuel", 300)
    trimestrielle = dates_de_rebalancement(prix.index, "trimestriel", 300)
    assert len(mensuelle) > len(trimestrielle)


def test_frequence_inconnue(prix):
    with pytest.raises(ValueError):
        dates_de_rebalancement(prix.index, "quotidien")


def test_pas_assez_de_donnees(prix):
    with pytest.raises(ValueError):
        dates_de_rebalancement(prix.index[:100], "trimestriel", 504)


# --- résultat ---------------------------------------------------------------

def test_les_courbes_demarrent_a_un(resultat):
    premiere_ligne = resultat["courbes"].iloc[0]
    assert np.allclose(premiere_ligne, 1.0)


def test_courbes_attendues(resultat):
    assert {"Stratégie", "Équipondéré (tout l'univers)", "S&P 500", "CAC 40"} <= set(resultat["courbes"].columns)
    assert resultat["courbes"].notna().all().all()
    assert (resultat["courbes"] > 0).all().all()


def test_compositions_valides(resultat):
    compo = resultat["compositions"]
    assert np.allclose(compo.sum(axis=1), 1.0)
    assert (compo >= 0).all().all()
    assert (compo.max(axis=1) <= 0.20 + 1e-3).all()
    assert ((compo > 0).sum(axis=1) <= PARAMETRES["nb_actions"]).all()


def test_premiere_rotation_egale_un(resultat):
    """Au départ, on achète tout : 100 % du portefeuille est échangé."""
    assert resultat["rotations"].iloc[0] == pytest.approx(1.0)


# --- frais ------------------------------------------------------------------

def test_les_frais_reduisent_la_performance(prix, resultat):
    sans_frais = lancer_backtest(prix, SECTEUR_PAR_TICKER, frais=0.0, **PARAMETRES)
    cher = lancer_backtest(prix, SECTEUR_PAR_TICKER, frais=0.01, **PARAMETRES)
    final = lambda r: r["courbes"]["Stratégie"].iloc[-1]
    assert final(sans_frais) > final(resultat) > final(cher)


# --- le test le plus important : aucune fuite de données --------------------

def test_un_choc_futur_ne_change_pas_le_passe(prix, resultat):
    """Si les prix changent APRÈS une date D, la courbe jusqu'à D doit rester identique."""
    date = prix.index[700]
    prix_choc = prix.copy()
    prix_choc.loc[prix_choc.index > date, ["S1", "S2", "S3", "S4"]] *= 5
    autre = lancer_backtest(prix_choc, SECTEUR_PAR_TICKER, **PARAMETRES)

    avant = resultat["courbes"].loc[:date, "Stratégie"]
    apres = autre["courbes"].loc[:date, "Stratégie"]
    assert np.allclose(avant, apres)
    assert not np.allclose(resultat["courbes"]["Stratégie"], autre["courbes"]["Stratégie"])


# --- métriques --------------------------------------------------------------

def test_metriques_dune_courbe_qui_double_en_un_an():
    courbe = pd.DataFrame({"X": 2 ** (np.arange(253) / 252)})
    m = metriques(courbe)
    assert m.loc["X", "Rendement annualisé"] == pytest.approx(1.0)
    assert m.loc["X", "Rendement total"] == pytest.approx(1.0)
    assert m.loc["X", "Drawdown max"] == pytest.approx(0.0)