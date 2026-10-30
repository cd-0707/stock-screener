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
