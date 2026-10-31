"""Univers d'actions suivi par le projet (tickers au format Yahoo Finance).

Les compositions des indices changent avec le temps : si un ticker ne
fonctionne plus, le script de collecte le signale en fin d'exécution.
Il suffit alors de le corriger ou de le retirer de cette liste.
"""

# Indices de référence (benchmarks) : ils ne sont pas recommandés,
# ils servent à comparer les performances.
INDICES = {
    "^FCHI": "CAC 40",
    "^GSPC": "S&P 500",
}

# Actions du CAC 40 (suffixe .PA = Euronext Paris)
CAC40 = [
    "AC.PA", "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA",
    "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA",
    "EL.PA", "ERF.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA",
    "ML.PA", "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA",
    "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA",
    "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
]

# Environ 50 grandes valeurs du S&P 500, réparties sur plusieurs secteurs
SP500 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", "MA", "CVX",
    "LLY", "ABBV", "KO", "PEP", "MRK", "COST", "AVGO", "ORCL", "CSCO",
    "ADBE", "CRM", "MCD", "NKE", "DIS", "INTC", "AMD", "TXN", "QCOM",
    "IBM", "CAT", "BA", "GE", "HON", "UPS", "LMT", "PFE", "T", "VZ",
    "BAC", "WFC", "GS", "NEE",
]

ACTIONS = CAC40 + SP500

# Historiques à tronquer : on ne garde les prix qu'à partir de cette date.
# Une fusion ou une scission peut recoller deux entreprises sous un même ticker,
# avec un « saut » de prix qui n'est pas un rendement réel. Exemple : Stellantis est
# née le 16 janvier 2021 de la fusion de PSA et FCA, et cote depuis le 18 janvier 2021.
# Avant cette date, l'historique du ticker STLAP.PA n'est pas comparable.
DEBUT_VALIDE = {
    "STLAP.PA": "2021-01-18",
}