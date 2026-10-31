"""Tests de la collecte, sans accès à Internet (Yahoo Finance est remplacé par un faux).

Lancer depuis la racine du projet :  pytest -v
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import src.collecte as collecte


class FauxTicker:
    """Imite yf.Ticker : 300 jours de prix, dont la journée d'aujourd'hui."""

    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, period, auto_adjust):
        index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=300, freq="D", tz="Europe/Paris")
        cloture = 100 + np.arange(300, dtype=float)
        return pd.DataFrame(
            {"Open": cloture, "High": cloture + 1, "Low": cloture - 1, "Close": cloture,
             "Adj Close": cloture, "Volume": 1000.0},
            index=index,
        )

    @property
    def info(self):
        return {"longName": f"{self.ticker} SA", "sector": "Technology", "currentPrice": 100.0}


@pytest.fixture
def environnement(tmp_path, monkeypatch):
    monkeypatch.setattr(collecte, "CHEMIN_BASE", tmp_path / "test.db")
    monkeypatch.setattr(collecte.yf, "Ticker", FauxTicker)
    monkeypatch.setattr(collecte.time, "sleep", lambda s: None)
    monkeypatch.setattr(collecte, "INDICES", {})
    monkeypatch.setattr(collecte, "ACTIONS", ["X", "Y"])
    monkeypatch.setattr(collecte, "DEBUT_VALIDE", {})
    return tmp_path / "test.db"


def lire(chemin, requete, parametres=()):
    conn = sqlite3.connect(chemin)
    try:
        return conn.execute(requete, parametres).fetchone()
    finally:
        conn.close()


def test_la_journee_en_cours_est_exclue(environnement):
    collecte.mettre_a_jour(annees=1)
    derniere = lire(environnement, "SELECT MAX(date) FROM prix WHERE ticker = 'X'")[0]
    aujourdhui = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    assert derniere < aujourdhui


def test_historique_tronque_a_partir_de_la_date_valide(environnement, monkeypatch):
    collecte.mettre_a_jour(annees=1)                       # première collecte : tout l'historique
    debut = (pd.Timestamp.today().normalize() - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    avant = lire(environnement, "SELECT MIN(date) FROM prix WHERE ticker = 'X'")[0]
    assert avant < debut

    monkeypatch.setattr(collecte, "DEBUT_VALIDE", {"X": debut})
    collecte.mettre_a_jour(annees=1)                       # nouvelle collecte avec la règle

    assert lire(environnement, "SELECT MIN(date) FROM prix WHERE ticker = 'X'")[0] == debut
    # l'autre action n'est pas touchée
    assert lire(environnement, "SELECT MIN(date) FROM prix WHERE ticker = 'Y'")[0] == avant


def test_relancer_la_collecte_ne_cree_pas_de_doublons(environnement):
    collecte.mettre_a_jour(annees=1)
    premier = lire(environnement, "SELECT COUNT(*) FROM prix")[0]
    collecte.mettre_a_jour(annees=1)
    assert lire(environnement, "SELECT COUNT(*) FROM prix")[0] == premier