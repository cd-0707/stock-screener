"""Tests de la récupération de la liste du S&P 500 (sans accès à Internet : page HTML simulée).

Lancer depuis la racine du projet :  pytest -v
"""

import pandas as pd
import pytest

from src.sp_500 import (
    charger_csv, eligibilite, enregistrer, parser_tableau, tickers_sp500,
)

PAGE = """
<html><body>
<table><tr><th>Autre</th></tr><tr><td>x</td></tr></table>
<table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>1982-11-30</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>2010-02-16</td></tr>
<tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td><td>1982-10-31</td></tr>
<tr><td>XYZ</td><td>Sans date</td><td>Industrials</td><td></td></tr>
<tr><td>AAPL</td><td>Doublon</td><td>Information Technology</td><td>1982-11-30</td></tr>
</table>
</body></html>
"""


def test_parser_choisit_la_bonne_table_et_normalise_les_tickers():
    table = parser_tableau(PAGE)
    assert list(table["ticker"]) == ["AAPL", "BRK-B", "BF-B", "XYZ"]     # points -> tirets, doublon supprimé
    assert list(table.columns) == ["ticker", "nom", "secteur_gics", "date_ajout"]


def test_parser_lit_les_dates_et_tolere_les_dates_manquantes():
    table = parser_tableau(PAGE).set_index("ticker")
    assert table.loc["BRK-B", "date_ajout"] == pd.Timestamp("2010-02-16")
    assert pd.isna(table.loc["XYZ", "date_ajout"])


def test_parser_signale_une_page_inattendue():
    with pytest.raises(ValueError):
        parser_tableau("<html><body><table><tr><th>Nom</th></tr><tr><td>x</td></tr></table></body></html>")


def test_enregistrement_et_relecture(tmp_path):
    chemin = tmp_path / "sp500.csv"
    enregistrer(parser_tableau(PAGE), chemin)
    assert tickers_sp500(chemin) == ["AAPL", "BRK-B", "BF-B", "XYZ"]
    relu = charger_csv(chemin)
    assert pd.api.types.is_datetime64_any_dtype(relu["date_ajout"])
    serie = eligibilite(chemin)
    assert serie["AAPL"] == pd.Timestamp("1982-11-30")
    assert pd.isna(serie["XYZ"])


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        charger_csv(tmp_path / "absent.csv")