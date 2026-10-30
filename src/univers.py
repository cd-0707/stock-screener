"""Univers d'actions suivi par le projet (tickers au format Yahoo Finance).

Les compositions des indices changent avec le temps : si un ticker ne
fonctionne plus, le script de collecte le signale en fin d'exécution.
Il suffit alors de le corriger ou de le retirer de cette liste.
"""

# Indices de référence 
INDICES = {
    "^FCHI": "CAC 40",
    "^GSPC": "S&P 500",
}

# CAC 40 
CAC40 = [
    "AC.PA", "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA",
    "EN.PA", "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA",
    "EL.PA", "ERF.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA",
    "ML.PA", "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA",
    "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA",
    "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
]

# 50 of S&P 500 (top 50 by market cap, as of 2024-06-01)
SP500 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", "MA", "CVX",
    "LLY", "ABBV", "KO", "PEP", "MRK", "COST", "AVGO", "ORCL", "CSCO",
    "ADBE", "CRM", "MCD", "NKE", "DIS", "INTC", "AMD", "TXN", "QCOM",
    "IBM", "CAT", "BA", "GE", "HON", "UPS", "LMT", "PFE", "T", "VZ",
    "BAC", "WFC", "GS", "NEE",
]
 
ACTIONS = CAC40 + SP500
 