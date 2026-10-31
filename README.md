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