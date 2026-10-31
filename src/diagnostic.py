"""Diagnostic du backtest : repère les plus gros mouvements et cherche leur origine.

Utilisation (depuis la racine du projet, après avoir lancé le backtest du profil) :
    python -m src.diagnostic --profil prudent

Le script répond à trois questions :
  1. Quels jours la courbe de la stratégie a-t-elle le plus bougé ?
  2. Quelles actions détenait-elle ce jour-là, et laquelle a bougé ?
  3. Quelles variations quotidiennes extrêmes contient la base de prix ?
     Une variation extrême est soit un vrai événement (krach, rachat, scission),
     soit une erreur de données : seule une vérification sur une source externe
     (par exemple la page de l'action sur finance.yahoo.com) permet de trancher.
"""

import argparse
import sqlite3

import pandas as pd

from src.collecte import CHEMIN_BASE, RACINE
from src.indicateurs import charger_prix
from src.univers import INDICES

CHEMIN_SNAPSHOT = RACINE / "data" / "snapshot.db"


def mouvements_extremes(rendements, seuil=0.30, nb=15):
    """Variations quotidiennes supérieures au seuil, avec ce qui se passe le lendemain.

    Un retour en sens inverse dès le lendemain évoque une erreur ponctuelle ;
    un lendemain « normal » évoque un changement durable de niveau (événement
    réel, ou ajustement de prix mal géré).
    """
    lignes = []
    for ticker in rendements.columns:
        if ticker in INDICES:
            continue
        serie = rendements[ticker].dropna()
        for date, valeur in serie[serie.abs() > seuil].items():
            suivant = serie.shift(-1).get(date)
            lignes.append({"ticker": ticker, "date": date.date(), "variation": valeur, "lendemain": suivant})
    resultat = pd.DataFrame(lignes)
    if resultat.empty:
        return resultat
    ordre = resultat["variation"].abs().sort_values(ascending=False).index
    return resultat.loc[ordre].head(nb).reset_index(drop=True)


def diagnostiquer(profil, nb_jours=5, seuil=0.30):
    dossier = RACINE / "data"
    fichier_courbes = dossier / f"backtest_{profil}_courbes.csv"
    fichier_compositions = dossier / f"backtest_{profil}_compositions.csv"
    if not fichier_courbes.exists():
        raise SystemExit(f"Backtest introuvable : lance d'abord  python -m src.backtest --profil {profil}")

    courbes = pd.read_csv(fichier_courbes, index_col=0, parse_dates=True)
    compositions = pd.read_csv(fichier_compositions, index_col=0, parse_dates=True).sort_index()

    chemin = CHEMIN_BASE if CHEMIN_BASE.exists() else CHEMIN_SNAPSHOT
    conn = sqlite3.connect(chemin)
    prix = charger_prix(conn)
    conn.close()
    rendements = prix / prix.shift(1) - 1

    variations = courbes["Stratégie"].pct_change().dropna()
    plus_fortes = variations.reindex(variations.abs().sort_values(ascending=False).index).head(nb_jours)
    print(f"\n1. Les {nb_jours} plus fortes variations quotidiennes de la stratégie « {profil} » :\n")
    print(plus_fortes.map("{:+.1%}".format).to_string())

    date = plus_fortes.index[0]
    detenues = compositions.loc[:date].iloc[-1]
    detenues = detenues[detenues > 0]
    jour = rendements.loc[date, detenues.index]
    tableau = pd.DataFrame({
        "poids_initial": detenues,
        "variation_du_jour": jour,
        "contribution": detenues * jour,
    }).sort_values("contribution", key=abs, ascending=False)
    print(f"\n2. Le {date.date()} (variation : {plus_fortes.iloc[0]:+.1%}), actions détenues et leur variation :\n")
    pd.options.display.float_format = "{:,.3f}".format
    print(tableau.to_string())

    extremes = mouvements_extremes(rendements, seuil=seuil)
    print(f"\n3. Variations quotidiennes de plus de {seuil:.0%} dans la base de prix :\n")
    if extremes.empty:
        print("   Aucune.")
    else:
        print(extremes.to_string(index=False))
        print("\n   À vérifier sur finance.yahoo.com (graphique de l'action à cette date) : événement réel ou erreur ?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic des mouvements extrêmes du backtest.")
    parser.add_argument("--profil", default="prudent", help="prudent, equilibre ou dynamique")
    parser.add_argument("--seuil", type=float, default=0.30, help="seuil de variation quotidienne (défaut : 0.30)")
    args = parser.parse_args()
    diagnostiquer(args.profil, seuil=args.seuil)