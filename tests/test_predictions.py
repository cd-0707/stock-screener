"""Tests de la prévision de volatilité, sur des prix simulés dont on connaît la dynamique.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.prediction import (
    CARACTERISTIQUES_ETENDUES, HORIZON, NOMS_MODELES, construire_jeu_de_donnees,
    experience_de_selection, jeu_d_entrainement, mesurer, mesurer_par_annee,
    prevoir_volatilite, previsions_aux_dates, valider_par_avancement,
)


def prix_simules(volatilite_persistante, n=2400, nb_actions=20, seed=0):
    """Prix simulés. Persistante : la volatilité varie lentement (prévisible).
    Sinon : volatilité constante (rien à prévoir à part sa valeur moyenne)."""
    rng = np.random.default_rng(seed)
    base = np.log(0.012)
    log_vol = np.full((n, nb_actions), base)
    if volatilite_persistante:
        for t in range(1, n):
            log_vol[t] = base + 0.98 * (log_vol[t - 1] - base) + 0.12 * rng.normal(size=nb_actions)
    rendements = np.exp(log_vol) * rng.normal(size=(n, nb_actions))
    index = pd.bdate_range("2012-01-02", periods=n)
    prix = pd.DataFrame(100 * np.cumprod(1 + rendements, axis=0), index=index,
                        columns=[f"A{i}" for i in range(nb_actions)])
    prix["^GSPC"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return prix


@pytest.fixture(scope="module")
def prix_persistants():
    return prix_simules(True)


@pytest.fixture(scope="module")
def donnees(prix_persistants):
    return construire_jeu_de_donnees(prix_persistants)


@pytest.fixture(scope="module")
def predictions(donnees):
    return valider_par_avancement(donnees)


# --- jeu de données ---------------------------------------------------------

def test_jeu_de_donnees_complet_et_sans_indices(donnees):
    assert not donnees.isna().any().any()
    assert "^GSPC" not in set(donnees["ticker"])
    assert set(CARACTERISTIQUES_ETENDUES + ["cible", "fin_cible"]) <= set(donnees.columns)
    assert donnees["ticker"].nunique() == 20


def test_une_observation_par_mois_et_par_action(donnees):
    ecarts = donnees[donnees["ticker"] == "A0"]["date"].diff().dropna().dt.days
    assert 25 <= ecarts.median() <= 35


def test_la_periode_de_la_cible_se_termine_apres_la_date(donnees):
    assert (donnees["fin_cible"] > donnees["date"]).all()


# --- absence de fuite de données --------------------------------------------

def test_les_caracteristiques_nutilisent_que_le_passe(prix_persistants, donnees):
    """Si les prix changent APRÈS une date, les caractéristiques de cette date restent identiques,
    alors que la cible (qui regarde en avant) change."""
    date = donnees["date"].iloc[len(donnees) // 2]
    modifie = prix_persistants.copy()
    modifie.loc[modifie.index > date, [c for c in modifie.columns if c.startswith("A")]] *= 3
    autres = construire_jeu_de_donnees(modifie)

    avant = donnees[donnees["date"] == date].set_index("ticker")
    apres = autres[autres["date"] == date].set_index("ticker")
    assert np.allclose(avant[CARACTERISTIQUES_ETENDUES], apres.loc[avant.index, CARACTERISTIQUES_ETENDUES])
    assert not np.allclose(avant["cible"], apres.loc[avant.index, "cible"])


def test_la_purge_exclut_les_periodes_qui_chevauchent_le_test(donnees):
    debut_test = pd.Timestamp("2018-01-02")
    entrainement = jeu_d_entrainement(donnees, debut_test)
    assert (entrainement["fin_cible"] < debut_test).all()
    # le dernier mois avant le test est exclu : sa cible empiète sur la période de test
    juste_avant = donnees[(donnees["date"] < debut_test) & (donnees["fin_cible"] >= debut_test)]
    assert len(juste_avant) > 0
    assert not entrainement.index.isin(juste_avant.index).any()


def test_chaque_test_est_posterieur_a_lentrainement(predictions):
    assert predictions["date"].min() > pd.Timestamp("2013-12-31")
    assert predictions["annee"].is_monotonic_increasing


# --- comportement des modèles -----------------------------------------------

def test_toutes_les_previsions_sont_finies(predictions):
    for nom in NOMS_MODELES:
        assert np.isfinite(predictions[nom]).all()


def test_har_bat_le_naif_quand_la_volatilite_est_persistante(predictions):
    metriques = mesurer(predictions)
    har = metriques.loc[NOMS_MODELES["har"]]
    naif = metriques.loc[NOMS_MODELES["naif"]]
    assert har["RMSE (log)"] < naif["RMSE (log)"]
    assert har["R² vs naïf"] > 0
    assert naif["R² vs naïf"] == pytest.approx(0.0)


def test_la_moyenne_longue_bat_le_naif_quand_rien_nest_previsible():
    """Contrôle : avec une volatilité constante, la mesure du dernier mois n'est que du bruit."""
    donnees = construire_jeu_de_donnees(prix_simules(False, nb_actions=15, seed=3))
    metriques = mesurer(valider_par_avancement(donnees))
    assert (metriques.loc[NOMS_MODELES["moyenne_longue"], "RMSE (log)"]
            < metriques.loc[NOMS_MODELES["naif"], "RMSE (log)"])


def test_resultats_reproductibles(donnees):
    a = valider_par_avancement(donnees)["ml"]
    b = valider_par_avancement(donnees)["ml"]
    assert np.allclose(a, b)


# --- mesures ----------------------------------------------------------------

def test_mesures_par_annee(predictions):
    tableau = mesurer_par_annee(predictions)
    assert NOMS_MODELES["har"] in tableau.columns
    assert NOMS_MODELES["naif"] not in tableau.columns
    assert (tableau > 0).all().all()


def test_la_selection_par_volatilite_prevue_bat_le_hasard(predictions):
    selection = experience_de_selection(predictions, k=5)
    hasard = selection["Hasard (tout l'univers)"]
    oracle = selection["Oracle (connaît l'avenir)"]
    assert selection[NOMS_MODELES["har"]] < hasard
    assert oracle <= selection[NOMS_MODELES["har"]]


# --- prévision actuelle -----------------------------------------------------

def test_prevoir_volatilite(prix_persistants):
    previsions = prevoir_volatilite(prix_persistants, modele="har")
    assert len(previsions) == 20
    assert previsions.between(0.02, 3.0).all()
    assert set(previsions.index) == {f"A{i}" for i in range(20)}


def test_pas_de_prevision_sans_assez_de_donnees(prix_persistants):
    with pytest.raises(ValueError):
        prevoir_volatilite(prix_persistants.iloc[:300])


def test_previsions_aux_dates_nutilisent_que_le_passe(prix_persistants):
    dates = [prix_persistants.index[1500], prix_persistants.index[2000]]
    completes = previsions_aux_dates(prix_persistants, dates, modele="har")
    assert set(completes) == set(dates)
    for date in dates:
        tronquees = prevoir_volatilite(prix_persistants.loc[:date], modele="har")
        assert np.allclose(completes[date], tronquees)


def test_dates_sans_assez_de_donnees_sont_ignorees(prix_persistants):
    dates = [prix_persistants.index[300], prix_persistants.index[2000]]
    previsions = previsions_aux_dates(prix_persistants, dates, modele="har")
    assert dates[0] not in previsions and dates[1] in previsions