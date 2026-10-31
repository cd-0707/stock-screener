"""Tests du moteur de scoring, sur un petit univers fictif et reproductible.

Lancer depuis la racine du projet :  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring import (
    CIBLE_RISQUE, FAMILLES, PROFILS, ajouter_clusters, calculer_scores, filtrer,
    poids_effectifs, recommander,
)

SECTEURS = ["Technology", "Healthcare", "Energy", "Financial Services"]


def univers(n=40, seed=0):
    """Tableau d'indicateurs fictif de n actions."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "nom": [f"Société {i}" for i in range(n)],
            "secteur": rng.choice(SECTEURS, n),
            "pays": "US",
            "rendement_annuel": rng.normal(0.08, 0.10, n),
            "volatilite_annuelle": rng.uniform(0.15, 0.50, n),
            "sharpe": rng.normal(0.4, 0.3, n),
            "beta": rng.uniform(0.5, 1.5, n),
            "drawdown_max": -rng.uniform(0.15, 0.60, n),
            "momentum_6m": rng.normal(0.04, 0.15, n),
            "momentum_12m": rng.normal(0.08, 0.25, n),
            "rendement_dividende": np.where(rng.random(n) < 0.6, rng.uniform(0, 0.05, n), np.nan),
            "dernier_prix": rng.uniform(10, 500, n),
            "liquidite_moyenne": rng.uniform(2e6, 1e8, n),
        },
        index=[f"T{i}" for i in range(n)],
    )


def univers_avec_extremes():
    """Univers où T0 est très défensif et T1 très spéculatif."""
    t = univers()
    t.loc["T0", ["volatilite_annuelle", "drawdown_max", "rendement_annuel",
                 "momentum_6m", "momentum_12m", "rendement_dividende"]] = [0.10, -0.05, 0.01, -0.10, -0.10, 0.04]
    t.loc["T1", ["volatilite_annuelle", "drawdown_max", "rendement_annuel",
                 "momentum_6m", "momentum_12m", "rendement_dividende"]] = [0.90, -0.80, 0.60, 1.00, 1.00, np.nan]
    t.loc[["T0", "T1"], "sharpe"] = 0.4
    return t


# --- poids ------------------------------------------------------------------

@pytest.mark.parametrize("profil", list(PROFILS))
def test_les_poids_de_chaque_profil_valent_un(profil):
    assert sum(PROFILS[profil].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("horizon", ["court", "moyen", "long"])
@pytest.mark.parametrize("dividendes", [False, True])
def test_poids_effectifs_somment_a_un(horizon, dividendes):
    for profil in PROFILS:
        assert sum(poids_effectifs(profil, horizon, dividendes).values()) == pytest.approx(1.0)


def test_profil_accepte_les_accents_et_les_majuscules():
    assert poids_effectifs("Équilibré") == poids_effectifs("equilibre")


def test_profil_inconnu_leve_une_erreur():
    with pytest.raises(ValueError):
        poids_effectifs("aventurier")


def test_horizon_court_renforce_le_momentum():
    assert poids_effectifs("equilibre", "court")["momentum"] > poids_effectifs("equilibre", "moyen")["momentum"]


def test_preference_dividendes_augmente_son_poids():
    assert poids_effectifs("prudent", "moyen", True)["dividende"] > poids_effectifs("prudent")["dividende"]


# --- scores -----------------------------------------------------------------

def test_les_contributions_somment_au_score_et_restent_entre_0_et_100():
    t = ajouter_clusters(univers())
    contributions, _ = calculer_scores(t, poids_effectifs("equilibre"), CIBLE_RISQUE["equilibre"])
    points = contributions.drop(columns="score")
    assert np.allclose(points.sum(axis=1), contributions["score"])
    assert contributions["score"].between(0, 100).all()


def test_le_profil_prudent_prefere_laction_defensive():
    t = univers_avec_extremes()
    c, _ = calculer_scores(t, poids_effectifs("prudent"))
    assert c.loc["T0", "score"] > c.loc["T1", "score"]


def test_le_profil_dynamique_prefere_laction_speculative():
    t = univers_avec_extremes()
    c, _ = calculer_scores(t, poids_effectifs("dynamique"))
    assert c.loc["T1", "score"] > c.loc["T0", "score"]


def test_critere_absent_est_ignore_sans_erreur():
    """Cas du backtest : pas de dividende dans les données."""
    t = univers().drop(columns="rendement_dividende")
    contributions, _ = calculer_scores(t, poids_effectifs("prudent"))
    assert "dividende" not in contributions.columns
    assert contributions["score"].max() <= 100 + 1e-9


# --- clusters ---------------------------------------------------------------

def test_clusters_donnent_quatre_familles_ordonnees_par_risque():
    t = ajouter_clusters(univers())
    assert set(t["famille"].unique()) <= set(FAMILLES)
    assert t["risque_famille"].between(0, 1).all()
    moyennes = t.groupby("famille")["volatilite_annuelle"].mean()
    ordre = [f for f in FAMILLES if f in moyennes.index]
    assert list(moyennes[ordre]) == sorted(moyennes[ordre])


def test_clusters_sont_reproductibles():
    a = ajouter_clusters(univers())["famille"]
    b = ajouter_clusters(univers())["famille"]
    assert a.equals(b)


def test_clusters_ignores_si_univers_trop_petit():
    t = ajouter_clusters(univers(n=5))
    assert t["famille"].isna().all()


# --- filtrage ---------------------------------------------------------------

def test_filtre_par_secteur():
    t = filtrer(univers(), secteurs=["Energy"])
    assert len(t) > 0 and (t["secteur"] == "Energy").all()


def test_filtre_par_fourchette_de_prix():
    t = filtrer(univers(), prix_min=100, prix_max=200)
    assert len(t) > 0 and t["dernier_prix"].between(100, 200).all()


def test_filtre_de_liquidite():
    t = filtrer(univers(), liquidite_min=5e7)
    assert len(t) > 0 and (t["liquidite_moyenne"] >= 5e7).all()


# --- recommandation ---------------------------------------------------------

def test_recommander_renvoie_le_bon_nombre_trie_par_score():
    r = recommander("prudent", nb_actions=7, tableau=univers())
    assert len(r) == 7
    assert r["score"].is_monotonic_decreasing
    assert r["explication"].str.len().min() > 0


def test_profils_differents_donnent_des_classements_differents():
    t = univers()
    prudent = recommander("prudent", nb_actions=10, tableau=t).index
    dynamique = recommander("dynamique", nb_actions=10, tableau=t).index
    assert list(prudent) != list(dynamique)


def test_la_note_dune_action_ne_depend_pas_des_filtres():
    """La normalisation se fait sur tout l'univers, avant filtrage."""
    t = univers()
    tous = recommander("equilibre", nb_actions=40, liquidite_min=0, tableau=t)
    energie = recommander("equilibre", nb_actions=40, liquidite_min=0,
                          secteurs=["Energy"], tableau=t)
    commun = energie.index
    assert len(commun) > 0
    assert np.allclose(tous.loc[commun, "score"], energie.loc[commun, "score"])


def test_plafond_par_secteur_respecte():
    r = recommander("dynamique", nb_actions=8, max_par_secteur=2, tableau=univers())
    assert r["secteur"].value_counts().max() <= 2


def test_aucun_candidat_renvoie_un_tableau_vide():
    r = recommander("prudent", prix_min=10_000, tableau=univers())
    assert r.empty


def test_les_poids_utilises_sont_exposes():
    r = recommander("prudent", horizon="long", dividendes=True, tableau=univers())
    assert sum(r.attrs["poids"].values()) == pytest.approx(1.0)
    