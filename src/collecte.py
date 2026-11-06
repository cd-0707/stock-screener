"""Collecte des données de marché (yfinance) et stockage dans SQLite.

Utilisation (depuis la racine du projet, environnement virtuel actif) :
    python -m src.collecte                 # collecte complète (10 ans)
    python -m src.collecte --annees 5      # sur 5 ans
    python -m src.collecte --verifier      # contrôle de la base, sans télécharger

Choix de conception :
- À chaque exécution, tout l'historique est retéléchargé puis écrit avec
  INSERT OR REPLACE. Un nouveau dividende ou split modifie rétroactivement
  tous les prix ajustés passés : un téléchargement partiel les rendrait
  incohérents entre eux.
- Les prix sont stockés bruts (clôture) ET ajustés (clôture ajustée). Les
  calculs de rendement utiliseront la clôture ajustée.
- La journée en cours est exclue (séance non terminée) : relance la collecte le
  lendemain pour obtenir la clôture d'hier.
- Les valeurs manquantes ne sont pas « remplies » ici : la base garde les
  données réelles, le traitement (alignement des calendriers Paris/New York)
  se fera au moment de l'analyse.
"""

import argparse
import logging
import math
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.univers import ACTIONS, CAC40, DEBUT_VALIDE, INDICES

RACINE = Path(__file__).resolve().parent.parent
# La variable d'environnement STOCK_SCREENER_DB permet d'utiliser une autre base (par exemple
# pour un univers élargi) sans toucher à celle de l'application. PowerShell :
#   $env:STOCK_SCREENER_DB = "data/stock_screener_sp500.db"
_base_perso = os.environ.get("STOCK_SCREENER_DB")
CHEMIN_BASE = (RACINE / _base_perso).resolve() if _base_perso else RACINE / "data" / "stock_screener.db"

# Suffixe ajouté aux fichiers de résultats quand on n'utilise pas la base par défaut,
# pour ne jamais écraser les fichiers lus par l'application.
SUFFIXE_BASE = "" if CHEMIN_BASE.name == "stock_screener.db" else "_" + CHEMIN_BASE.stem.removeprefix("stock_screener_")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    ticker      TEXT PRIMARY KEY,
    nom         TEXT,
    secteur     TEXT,
    industrie   TEXT,
    pays        TEXT,
    devise      TEXT,
    est_indice  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prix (
    ticker           TEXT NOT NULL,
    date             TEXT NOT NULL,
    ouverture        REAL,
    haut             REAL,
    bas              REAL,
    cloture          REAL,
    cloture_ajustee  REAL,
    volume           INTEGER,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker) REFERENCES actions (ticker)
);

CREATE TABLE IF NOT EXISTS fondamentaux (
    ticker               TEXT PRIMARY KEY,
    capitalisation       REAL,
    per                  REAL,
    dividende_annuel     REAL,
    rendement_dividende  REAL,
    date_maj             TEXT,
    FOREIGN KEY (ticker) REFERENCES actions (ticker)
);
"""


# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------

def connecter():
    """Ouvre la base SQLite (la crée si besoin) et s'assure que les tables existent."""
    CHEMIN_BASE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHEMIN_BASE)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _nombre(valeur):
    """Convertit en float, ou renvoie None si la valeur est absente, non numérique ou infinie."""
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        return None
    if not math.isfinite(valeur):
        return None
    return float(valeur)


# ---------------------------------------------------------------------------
# Téléchargement
# ---------------------------------------------------------------------------

def telecharger_prix(ticker, annees):
    """Télécharge l'historique journalier et renvoie un DataFrame nettoyé (ou None si vide)."""
    df = yf.Ticker(ticker).history(period=f"{annees}y", auto_adjust=False)
    if df is None or df.empty:
        return None

    df = df.rename(columns={
        "Open": "ouverture", "High": "haut", "Low": "bas",
        "Close": "cloture", "Adj Close": "cloture_ajustee", "Volume": "volume",
    })
    if "cloture_ajustee" not in df.columns:  # certains indices n'ont pas de prix ajusté
        df["cloture_ajustee"] = df["cloture"]
    df = df[["ouverture", "haut", "bas", "cloture", "cloture_ajustee", "volume"]].copy()

    # Dates : on retire le fuseau horaire et l'heure, on garde le jour
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    # Nettoyage : doublons, clôtures manquantes ou aberrantes (<= 0)
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["cloture", "cloture_ajustee"])
    df = df[(df["cloture"] > 0) & (df["cloture_ajustee"] > 0)]
    df["volume"] = df["volume"].fillna(0)

    # La séance du jour n'est peut-être pas terminée : son dernier cours serait un cours
    # en cours de journée, pas une clôture. On ne garde que les journées terminées.
    df = df[df.index < pd.Timestamp.today().normalize()]
    return df.sort_index()


def recuperer_infos(ticker):
    """Récupère les informations descriptives et fondamentales d'une action."""
    info = yf.Ticker(ticker).info

    prix = (_nombre(info.get("currentPrice"))
            or _nombre(info.get("regularMarketPrice"))
            or _nombre(info.get("previousClose")))
    dividende = _nombre(info.get("dividendRate"))
    # On recalcule le rendement nous-mêmes : le format du champ « dividendYield »
    # a changé selon les versions de yfinance (fraction ou pourcentage).
    rendement = dividende / prix if dividende and prix else None

    return {
        "nom": info.get("longName") or info.get("shortName") or ticker,
        "secteur": info.get("sector"),
        "industrie": info.get("industry"),
        "pays": info.get("country"),
        "devise": info.get("currency"),
        "capitalisation": _nombre(info.get("marketCap")),
        "per": _nombre(info.get("trailingPE")),
        "dividende_annuel": dividende,
        "rendement_dividende": rendement,
    }


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------

def enregistrer_action(conn, ticker, infos, est_indice):
    conn.execute(
        """
        INSERT INTO actions (ticker, nom, secteur, industrie, pays, devise, est_indice)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            nom = excluded.nom,
            secteur = COALESCE(excluded.secteur, actions.secteur),
            industrie = COALESCE(excluded.industrie, actions.industrie),
            pays = COALESCE(excluded.pays, actions.pays),
            devise = COALESCE(excluded.devise, actions.devise),
            est_indice = excluded.est_indice
        """,
        (ticker, infos.get("nom", ticker), infos.get("secteur"), infos.get("industrie"),
         infos.get("pays"), infos.get("devise"), int(est_indice)),
    )


def enregistrer_prix(conn, ticker, df):
    lignes = [
        (ticker, date.strftime("%Y-%m-%d"),
         _nombre(r.ouverture), _nombre(r.haut), _nombre(r.bas),
         float(r.cloture), float(r.cloture_ajustee), int(r.volume))
        for date, r in zip(df.index, df.itertuples(index=False))
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO prix VALUES (?, ?, ?, ?, ?, ?, ?, ?)", lignes
    )
    return len(lignes)


def enregistrer_fondamentaux(conn, ticker, infos):
    conn.execute(
        "INSERT OR REPLACE INTO fondamentaux VALUES (?, ?, ?, ?, ?, ?)",
        (ticker, infos.get("capitalisation"), infos.get("per"),
         infos.get("dividende_annuel"), infos.get("rendement_dividende"),
         datetime.now().isoformat(timespec="seconds")),
    )


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def mettre_a_jour(annees=10, pause=0.5, actions=None):
    """Télécharge et enregistre prix et fondamentaux pour tout l'univers (ou la liste `actions`)."""
    conn = connecter()
    liste = ACTIONS if actions is None else actions
    univers = [(t, True) for t in INDICES] + [(t, False) for t in liste]
    echecs, sans_fondamentaux = [], []

    for i, (ticker, est_indice) in enumerate(univers, start=1):
        log.info("[%d/%d] %s", i, len(univers), ticker)
        try:
            prix = telecharger_prix(ticker, annees)
            if prix is None:
                raise ValueError("aucune donnée de prix reçue")

            # Historique tronqué (fusion, scission...) : voir DEBUT_VALIDE dans univers.py
            debut_valide = DEBUT_VALIDE.get(ticker)
            if debut_valide:
                prix = prix[prix.index >= pd.Timestamp(debut_valide)]
                conn.execute("DELETE FROM prix WHERE ticker = ? AND date < ?", (ticker, debut_valide))
            if prix.empty:
                raise ValueError("aucune donnée de prix exploitable")

            # Les prix sont indispensables ; les fondamentaux sont facultatifs
            infos = {"nom": INDICES.get(ticker, ticker)}
            if not est_indice:
                try:
                    infos = recuperer_infos(ticker)
                except Exception as erreur:
                    log.warning("   fondamentaux indisponibles pour %s : %s", ticker, erreur)
                    sans_fondamentaux.append(ticker)

            enregistrer_action(conn, ticker, infos, est_indice)
            nb = enregistrer_prix(conn, ticker, prix)
            if not est_indice and len(infos) > 1:
                enregistrer_fondamentaux(conn, ticker, infos)
            conn.commit()
            log.info("   %d jours enregistrés (%s -> %s)", nb,
                     prix.index.min().date(), prix.index.max().date())
        except Exception as erreur:
            conn.rollback()
            log.warning("   ÉCHEC pour %s : %s", ticker, erreur)
            echecs.append(ticker)
        time.sleep(pause)  # évite d'être bloqué par Yahoo pour trop de requêtes

    conn.close()

    log.info("Terminé : %d/%d tickers collectés.", len(univers) - len(echecs), len(univers))
    if echecs:
        log.warning("Tickers en échec (à vérifier dans src/univers.py) : %s", ", ".join(echecs))
    if sans_fondamentaux:
        log.warning("Sans fondamentaux : %s", ", ".join(sans_fondamentaux))


def verifier():
    """Affiche un contrôle de qualité de la base (sans rien télécharger)."""
    conn = connecter()

    resume = pd.read_sql_query(
        """
        SELECT a.ticker, a.nom, a.secteur,
               COUNT(p.date) AS nb_jours,
               MIN(p.date)   AS debut,
               MAX(p.date)   AS fin
        FROM actions a LEFT JOIN prix p ON p.ticker = a.ticker
        GROUP BY a.ticker
        ORDER BY nb_jours ASC
        """,
        conn,
    )
    print(f"\n{len(resume)} tickers en base. Les 10 historiques les plus courts :\n")
    print(resume.head(10).to_string(index=False))

    manquants = pd.read_sql_query(
        """
        SELECT COUNT(*) AS nb_actions,
               SUM(per IS NULL)                 AS sans_per,
               SUM(rendement_dividende IS NULL) AS sans_dividende,
               SUM(capitalisation IS NULL)      AS sans_capitalisation
        FROM fondamentaux
        """,
        conn,
    )
    print("\nFondamentaux manquants :\n")
    print(manquants.to_string(index=False))

    # Variations quotidiennes extrêmes : souvent le signe d'une donnée erronée
    prix = pd.read_sql_query(
        "SELECT ticker, date, cloture_ajustee FROM prix ORDER BY ticker, date", conn
    )
    if not prix.empty:
        prix["rendement"] = prix.groupby("ticker")["cloture_ajustee"].pct_change()
        extremes = prix[prix["rendement"].abs() > 0.30]
        print(f"\nVariations quotidiennes supérieures à 30 % : {len(extremes)}")
        if not extremes.empty:
            print(extremes.head(10).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte des données de marché.")
    parser.add_argument("--annees", type=int, default=10, help="profondeur d'historique (défaut : 10)")
    parser.add_argument("--verifier", action="store_true", help="contrôle la base sans télécharger")
    parser.add_argument("--univers", choices=["defaut", "sp500"], default="defaut",
                        help="sp500 : CAC 40 + tout le S&P 500 (lance d'abord python -m src.univers_sp500)")
    args = parser.parse_args()

    if args.verifier:
        verifier()
    else:
        liste = None
        if args.univers == "sp500":
            from src.sp_500 import tickers_sp500
            try:
                liste = sorted(set(CAC40) | set(tickers_sp500()))
            except FileNotFoundError as erreur:
                raise SystemExit(str(erreur))
            log.info("Univers élargi : %d actions. Base utilisée : %s", len(liste), CHEMIN_BASE)
        mettre_a_jour(annees=args.annees, actions=liste)