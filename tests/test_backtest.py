"""Tests du backtest.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import dates_de_rebalancement, lancer_backtest, metriques
from src.prediction import previsions_aux_dates

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


# --- volatilité prévue dans le backtest -------------------------------------

@pytest.fixture(scope="module")
def dates_reequilibrage(prix):
    return dates_de_rebalancement(prix.index, "trimestriel", PARAMETRES["historique_min"])


def test_la_prevision_change_les_resultats(prix, resultat, dates_reequilibrage):
    previsions = previsions_aux_dates(prix, dates_reequilibrage, modele="har")
    assert len(previsions) > 0
    avec = lancer_backtest(prix, SECTEUR_PAR_TICKER, previsions_volatilite=previsions, **PARAMETRES)
    assert not np.allclose(avec["courbes"]["Stratégie"], resultat["courbes"]["Stratégie"])
    assert np.allclose(avec["courbes"]["Équipondéré (tout l'univers)"],
                       resultat["courbes"]["Équipondéré (tout l'univers)"])


def test_un_choc_futur_ne_change_pas_le_passe_avec_les_previsions(prix, dates_reequilibrage):
    """Même test que ci-dessus, mais en utilisant la volatilité prévue : aucune fuite non plus."""
    date = prix.index[700]
    previsions = previsions_aux_dates(prix, dates_reequilibrage, modele="har")
    normal = lancer_backtest(prix, SECTEUR_PAR_TICKER, previsions_volatilite=previsions, **PARAMETRES)

    prix_choc = prix.copy()
    prix_choc.loc[prix_choc.index > date, ["S1", "S2", "S3", "S4"]] *= 5
    previsions_choc = previsions_aux_dates(prix_choc, dates_reequilibrage, modele="har")
    avec_choc = lancer_backtest(prix_choc, SECTEUR_PAR_TICKER, previsions_volatilite=previsions_choc, **PARAMETRES)

    assert np.allclose(normal["courbes"].loc[:date, "Stratégie"], avec_choc["courbes"].loc[:date, "Stratégie"])
    anterieures = [d for d in previsions if d <= date]
    assert len(anterieures) > 0
    for d in anterieures:
        assert np.allclose(previsions[d], previsions_choc[d])


# --- appartenance à l'indice (réduction du biais de survie) -----------------

def test_une_action_nest_utilisable_qua_partir_de_sa_date_d_entree(prix):
    entree = prix.index[800]
    eligibilite = pd.Series({f"S{i}": entree for i in range(6)})
    avec = lancer_backtest(prix, SECTEUR_PAR_TICKER, eligibilite=eligibilite, **PARAMETRES)

    compositions = avec["compositions"]
    avant = compositions[compositions.index < entree]
    interdites = [c for c in compositions.columns if c in eligibilite.index]
    assert len(avant) > 0
    assert (avant[interdites] == 0).all().all()


def test_les_tickers_absents_de_leligibilite_restent_autorises(prix, resultat):
    """Une éligibilité qui ne mentionne que des tickers inconnus ne doit rien changer."""
    inconnue = pd.Series({"ZZZ": prix.index[800]})
    avec = lancer_backtest(prix, SECTEUR_PAR_TICKER, eligibilite=inconnue, **PARAMETRES)
    assert np.allclose(avec["courbes"]["Stratégie"], resultat["courbes"]["Stratégie"])