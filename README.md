# Stock Screener

Système de recommandation d'actions personnalisé selon le profil de l'investisseur
(risque, horizon, secteurs, budget), avec scoring multicritère, optimisation de
portefeuille et backtest.

> Projet pédagogique. Ceci n'est pas un conseil en investissement.

## Statut
En cours de développement

## Installation
    git clone https://github.com/cd-0707/stock-screener.git
    cd stock-screener
    python -m venv .venv
    source .venv/bin/activate   # Windows : .venv\Scripts\activate
    pip install -r requirements.txt

## Feuille de route
- [ ] Collecte des données
- [ ] Indicateurs financiers
- [ ] Moteur de scoring
- [ ] Optimisation de portefeuille et backtest
- [ ] Interface Streamlit

## Objectifs 
Un outil qui classe des actions selon le profil d’un investisseur (risque, horizon, secteurs, budget) et propose un portefeuille dont les performances passées sont évaluées par backtest.

## Périmètre 
^FCHI (CAC 40) et ^GSPC (S&P 500)
+infos sur docs/cahier_des_charges.md 

## Méthodologie
Fenêtre de 3 ans (756 jours de bourse) : le risque se mesure sur une période récente, pas sur 10 ans qui incluraient   des régimes de marché très différents.
Minimum 1 an d’historique : en dessous, une volatilité ou un bêta ne sont pas fiables.
Bêta par rapport à l’indice du pays : CAC 40 pour les tickers .PA et .AS, S&P 500 pour les autres.
Taux sans risque à 0 % dans le Sharpe : une simplification.
Fondamentaux exclus quand on calcule à une date passée. Le PER et le dividende de la base sont ceux d’aujourd’hui. Les utiliser dans un backtest injecterait de l’information du futur. C’est le choix fait à l’étape 1, et le code l’applique automatiquement avec --date.

##

Biais de survie : l’univers est celui d’aujourd’hui, donc il exclut les entreprises sorties des indices ou disparues. Tous les résultats sont un peu trop optimistes.
Pas de conversion de devises : euros et dollars sont mélangés.
Exécution au cours de clôture du jour du signal.
Frais simplifiés : pas d’écart acheteur/vendeur.
Fondamentaux exclus du backtest (PER, dividende non historisés).
Une seule période : un backtest n’est pas une preuve.


## 
Sur 2018-2026, avec des rééquilibrages trimestriels et 0,1 % de frais, le profil prudent réduit la pire baisse (−27 % contre −33 % pour un portefeuille équipondéré de l’univers) mais n’améliore pas le rendement (15,1 % contre 17,0 % par an). Aucun profil ne bat le portefeuille équipondéré sur le ratio de Sharpe. Ces résultats portent sur une seule période et un univers soumis au biais de survie : ils ne permettent pas de conclure à l’efficacité du scoring.


##
Sur 2018-2026 (rééquilibrage trimestriel, 0,1 % de frais), aucun des trois profils ne dépasse le portefeuille équipondéré de l’univers en rendement (10,0 %, 13,5 % et 15,9 % par an contre 16,8 %) ni en ratio de Sharpe (0,73, 0,81 et 0,79 contre 1,02). Les profils se différencient bien sur la volatilité (13,7 %, 16,7 %, 20,2 %), mais le profil prudent ne réduit pas la pire baisse (−32,4 % contre −33,8 %). Une anomalie de données liée à la fusion de Stellantis gonflait le rendement du profil prudent de 5 points par an avant sa détection. Ces résultats portent sur une seule période et un univers soumis au biais de survie : ils ne permettent pas de conclure à l’efficacité du scoring.

Contrôle qualité : toutes les variations quotidiennes supérieures à 20 % ont été examinées. Une anomalie liée à la fusion PSA/FCA (Stellantis) a été détectée, expliquée et corrigée.