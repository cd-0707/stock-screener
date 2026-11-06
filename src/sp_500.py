"""Liste des constituants actuels du S&P 500, avec leur date d'entrée dans l'indice.

Utilisation (depuis la racine du projet, avec accès à Internet) :
    python -m src.univers_sp500

Le script télécharge la liste publiée sur Wikipédia et l'enregistre dans
data/sp500_constituants.csv, avec les colonnes : ticker (au format Yahoo Finance), nom,
secteur_gics, date_ajout.

Deux usages :
  - élargir l'univers : python -m src.collecte --univers sp500 ;
  - limiter le biais de survie dans le backtest : la « date d'ajout » permet de n'autoriser
    une action qu'à partir du moment où elle est entrée dans l'indice
    (python -m src.backtest --appartenance).

Limites à connaître :
  - c'est la liste d'AUJOURD'HUI : les entreprises sorties de l'indice (faillites, rachats,
    déclassements) n'y figurent pas. Le biais de survie n'est donc que PARTIELLEMENT corrigé :
    la date d'ajout évite d'utiliser une action avant son entrée dans l'indice, mais pas
    d'ignorer celles qui en sont sorties ;
  - la page Wikipédia est maintenue par des contributeurs : sa structure ou ses données
    peuvent changer ou contenir des erreurs. Vérifie le nombre de lignes obtenues.
"""

import io
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_CSV = RACINE / "data" / "sp500_constituants.csv"
URL_WIKIPEDIA = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
EN_TETE = {"User-Agent": "stock-screener (projet pedagogique)"}


def parser_tableau(html):
    """Extrait la table des constituants d'une page HTML et normalise les colonnes."""
    tableaux = pd.read_html(io.StringIO(html))
    brut = next((t for t in tableaux if "Symbol" in t.columns and "Security" in t.columns), None)
    if brut is None:
        raise ValueError("Table des constituants introuvable : la page a peut-être changé de structure.")

    resultat = pd.DataFrame({
        # Wikipédia écrit BRK.B, Yahoo Finance écrit BRK-B
        "ticker": brut["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
        "nom": brut["Security"],
        "secteur_gics": brut.get("GICS Sector"),
        "date_ajout": pd.to_datetime(brut.get("Date added"), errors="coerce"),
    })
    return resultat.drop_duplicates("ticker").reset_index(drop=True)


def telecharger():
    """Télécharge la page Wikipédia et renvoie la table normalisée."""
    import requests

    reponse = requests.get(URL_WIKIPEDIA, headers=EN_TETE, timeout=30)
    reponse.raise_for_status()
    return parser_tableau(reponse.text)


def enregistrer(table, chemin=CHEMIN_CSV):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(chemin, index=False)


def charger_csv(chemin=CHEMIN_CSV):
    """Relit le fichier enregistré. Lève FileNotFoundError s'il n'existe pas encore."""
    if not Path(chemin).exists():
        raise FileNotFoundError(f"{chemin} introuvable : lance d'abord  python -m src.univers_sp500")
    return pd.read_csv(chemin, parse_dates=["date_ajout"])


def tickers_sp500(chemin=CHEMIN_CSV):
    return list(charger_csv(chemin)["ticker"])


def eligibilite(chemin=CHEMIN_CSV):
    """Series ticker -> date d'entrée dans l'indice (pour le backtest)."""
    table = charger_csv(chemin)
    return table.set_index("ticker")["date_ajout"]


if __name__ == "__main__":
    table = telecharger()
    enregistrer(table)
    sans_date = table["date_ajout"].isna().sum()
    print(f"{len(table)} constituants enregistrés dans {CHEMIN_CSV}")
    print(f"Entrées dans l'indice : de {table['date_ajout'].min():%Y-%m-%d} à {table['date_ajout'].max():%Y-%m-%d} "
          f"({sans_date} sans date)")
    print("Vérifie que le nombre de lignes est proche de 500 avant de lancer la collecte.")