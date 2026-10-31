"""Crée data/snapshot.db : une copie allégée de la base, à versionner sur GitHub.

Pourquoi : data/stock_screener.db est ignorée par Git (trop lourde, régénérable).
Or l'application mise en ligne a besoin de données. Cette copie ne garde que ce
dont l'application se sert : elle retire les prix d'ouverture, plus haut et plus
bas, ainsi que la table « indicateurs » (recalculée à la volée).

Utilisation (depuis la racine du projet) :
    python -m src.snapshot

À refaire de temps en temps seulement : chaque version commitée alourdit
l'historique Git de la taille du fichier.
"""

import sqlite3

from src.collecte import CHEMIN_BASE, RACINE

CHEMIN_SNAPSHOT = RACINE / "data" / "snapshot.db"


def exporter():
    if not CHEMIN_BASE.exists():
        raise SystemExit("Base introuvable : lance d'abord  python -m src.collecte")

    CHEMIN_SNAPSHOT.unlink(missing_ok=True)

    source = sqlite3.connect(CHEMIN_BASE)
    source.execute("VACUUM INTO ?", (str(CHEMIN_SNAPSHOT),))
    source.close()

    copie = sqlite3.connect(CHEMIN_SNAPSHOT)
    copie.execute("DROP TABLE IF EXISTS indicateurs")
    copie.execute("UPDATE prix SET ouverture = NULL, haut = NULL, bas = NULL")
    copie.commit()
    copie.execute("VACUUM")
    copie.close()

    taille = CHEMIN_SNAPSHOT.stat().st_size / 1e6
    print(f"Copie créée : {CHEMIN_SNAPSHOT} ({taille:.1f} Mo)")
    return taille


if __name__ == "__main__":
    exporter()