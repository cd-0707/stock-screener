# Cahier des charges : Stock Screener

## Objectif
Un outil qui classe des actions selon le profil d’un investisseur (risque, horizon, secteurs, budget) et propose un portefeuille dont les performances passées sont évaluées par backtest.

## Univers
- CAC 40 + ~50 valeurs du S&P 500
- Indices de référence : ^FCHI, ^GSPC

## Entrées utilisateur
Critère	Valeurs possibles
Profil de risque : 	prudent / équilibré / dynamique
Secteurs d’intérêt : 	un ou plusieurs secteurs, ou « tous »
Fourchette de prix de l’action : 	minimum et maximum
Horizon de placement : 	court (< 1 an), moyen (1 à 5 ans), long (> 5 ans)
Préférence dividendes : 	oui / indifférent
Nombre d’actions souhaité : par exemple 5 à 15

## Profils et pondérations
Indicateur	            Prudent	Équilibré	Dynamique
Faible volatilité	    35 %	20 %	    5 %
Faible drawdown maximal	25 %	15 %	    5 %
Ratio de Sharpe	        15 %	25 %	    20 %
Rendement annualisé	    10 %	20 %	    30 %
Momentum (6 à 12 mois)	5 %	    20 %	    40 %
Dividende	            10 %	0 %	        0 %

## Sorties
Un classement des actions avec un score de 0 à 100
Pour chaque action, la contribution de chaque critère à son score (l’explication)
Une répartition de portefeuille avec son risque global
Un graphique de backtest face aux indices de référence

## Backtest
- Période, rééquilibrage trimestriel, frais de 0,1 %
- Uniquement des indicateurs calculés à partir des prix
- Comparaison : indices et portefeuille équipondéré

## Hors périmètre

pas de trading en temps réel, ni de connexion à un courtier
pas d’analyse de news ou de sentiment
pas de produits autres que les actions (ni crypto, ni obligations)
pas de conseil personnalisé : outil pédagogique


## Limites connues
- Biais de survie (univers composé des actions actuelles)
- Données gratuites yfinance, parfois incomplètes
- Fondamentaux non historisés
- simplification du taux à 0%.
