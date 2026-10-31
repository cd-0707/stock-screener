"""Application Streamlit : recommandation d'actions selon le profil de l'investisseur.

Lancer en local (depuis la racine du projet, environnement virtuel actif) :
    streamlit run app.py

Données : la base locale data/stock_screener.db si elle existe, sinon la copie
allégée data/snapshot.db (c'est elle qui est versionnée et utilisée en ligne).
"""

import math
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from src.backtest import metriques
from src.collecte import CHEMIN_BASE, RACINE
from src.indicateurs import charger_prix
from src.portefeuille import (
    METHODE_PAR_PROFIL, POIDS_MAX_DEFAUT, calculer_rendements,
    construire_portefeuille, statistiques_portefeuille,
)
from src.scoring import CRITERES, HORIZONS, PROFILS, _cle, charger_univers

CHEMIN_SNAPSHOT = RACINE / "data" / "snapshot.db"

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")

PROFILS_AFFICHES = {"Prudent": "prudent", "Équilibré": "equilibre", "Dynamique": "dynamique"}
HORIZONS_AFFICHES = {"Court (moins d'un an)": "court", "Moyen (1 à 5 ans)": "moyen", "Long (plus de 5 ans)": "long"}
NOMS_CRITERES = {
    "volatilite": "Faible volatilité", "drawdown": "Baisses limitées", "sharpe": "Sharpe",
    "rendement": "Rendement", "momentum": "Momentum", "dividende": "Dividende",
    "bonus_famille": "Compatibilité de famille",
}


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement des données...")
def charger_donnees():
    chemin = CHEMIN_BASE if CHEMIN_BASE.exists() else CHEMIN_SNAPSHOT
    conn = sqlite3.connect(chemin)
    try:
        univers = charger_univers(conn)
        prix = charger_prix(conn)
    finally:
        conn.close()
    return univers, calculer_rendements(prix), prix.index.max()


@st.cache_data
def charger_backtest(cle):
    chemin = RACINE / "data" / f"backtest_{cle}_courbes.csv"
    if not chemin.exists():
        return None
    return pd.read_csv(chemin, index_col=0, parse_dates=True)


try:
    univers, rendements, date_donnees = charger_donnees()
except Exception as erreur:
    st.error("Impossible de charger les données. Lance d'abord `python -m src.collecte` "
             "(ou ajoute le fichier data/snapshot.db).")
    st.caption(f"Détail technique : {erreur}")
    st.stop()


# ---------------------------------------------------------------------------
# Questionnaire (barre latérale)
# ---------------------------------------------------------------------------

st.sidebar.header("Votre profil")

profil_affiche = st.sidebar.radio(
    "Profil de risque", list(PROFILS_AFFICHES), index=1,
    help="Prudent : privilégie la stabilité. Dynamique : accepte plus de risque pour viser plus de rendement.",
)
horizon_affiche = st.sidebar.selectbox("Horizon de placement", list(HORIZONS_AFFICHES), index=1)
secteurs = st.sidebar.multiselect(
    "Secteurs d'intérêt", sorted(univers["secteur"].dropna().unique()),
    help="Laisser vide pour considérer tous les secteurs.",
)

prix_plafond = int(math.ceil(univers["dernier_prix"].max()))
prix_bas, prix_haut = st.sidebar.slider(
    "Fourchette de prix de l'action", 0, prix_plafond, (0, prix_plafond), step=5,
    help="Dans la devise de cotation de chaque action (euros ou dollars), sans conversion.",
)
dividendes = st.sidebar.checkbox("Favoriser les dividendes")
nb_actions = st.sidebar.slider("Nombre d'actions", 5, 15, 10)

with st.sidebar.expander("Réglages avancés"):
    max_par_secteur = st.slider("Maximum d'actions par secteur (0 = aucune limite)", 0, 6, 3)
    poids_max = st.slider("Poids maximum par action", 10, 40, int(POIDS_MAX_DEFAUT * 100), step=5,
                          format="%d %%") / 100

st.sidebar.caption("⚠️ Projet pédagogique. Ceci n'est pas un conseil en investissement.")

profil = PROFILS_AFFICHES[profil_affiche]
horizon = HORIZONS_AFFICHES[horizon_affiche]

# Avec peu de secteurs choisis, un plafond trop strict empêcherait d'atteindre le nombre voulu
if max_par_secteur and secteurs:
    max_par_secteur = max(max_par_secteur, math.ceil(nb_actions / len(secteurs)))


# ---------------------------------------------------------------------------
# Calcul du portefeuille
# ---------------------------------------------------------------------------

poids, classement = construire_portefeuille(
    univers, rendements, profil, horizon=horizon, nb_actions=nb_actions,
    max_par_secteur=max_par_secteur or None, poids_max=poids_max,
    secteurs=secteurs or None,
    prix_min=prix_bas if prix_bas > 0 else None,
    prix_max=prix_haut if prix_haut < prix_plafond else None,
    dividendes=dividendes,
)


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

st.title("📈 Stock Screener")
st.warning(
    "**Projet pédagogique : ceci n'est pas un conseil en investissement.** "
    "Les classements décrivent le comportement passé de quelques actions ; "
    "ils ne prédisent pas l'avenir et peuvent faire perdre de l'argent."
)
st.caption(f"Données de marché arrêtées au {date_donnees:%d/%m/%Y} · {len(univers)} actions analysées "
           "(CAC 40 et grandes valeurs du S&P 500).")

if classement.empty:
    st.error("Aucune action ne respecte ces critères (secteurs, fourchette de prix, liquidité minimale). "
             "Essaie d'élargir la fourchette de prix ou de retirer un secteur.")
    st.stop()
if len(classement) < nb_actions:
    st.info(f"Seulement {len(classement)} actions respectent ces critères (au lieu de {nb_actions}).")

stats = statistiques_portefeuille(poids, rendements)
colonnes = st.columns(4)
colonnes[0].metric("Rendement annualisé", f"{stats['rendement_annuel']:+.1%}")
colonnes[1].metric("Volatilité annualisée", f"{stats['volatilite_annuelle']:.1%}")
colonnes[2].metric("Ratio de Sharpe", f"{stats['sharpe']:.2f}")
colonnes[3].metric("Pire baisse (drawdown)", f"{stats['drawdown_max']:.1%}")
st.caption(
    "Mesuré sur les 3 dernières années avec ces poids constants. Cette période est celle qui a servi à "
    "choisir les actions : **ce n'est pas une prévision**. L'onglet « Backtest » donne une évaluation plus honnête."
)

onglet_portefeuille, onglet_detail, onglet_carte, onglet_backtest, onglet_methode = st.tabs(
    ["Portefeuille", "Pourquoi ces actions ?", "Carte rendement / risque", "Backtest", "Méthode et limites"]
)

# --- Portefeuille -----------------------------------------------------------
with onglet_portefeuille:
    gauche, droite = st.columns([3, 2])
    with gauche:
        affichage = pd.DataFrame({
            "Société": classement["nom"],
            "Secteur": classement["secteur"],
            "Famille": classement["famille"],
            "Prix": classement["dernier_prix"],
            "Devise": univers["devise"].reindex(classement.index),
            "Score": classement["score"],
            "Poids (%)": classement["poids"] * 100,
        })
        st.dataframe(
            affichage,
            column_config={
                "Prix": st.column_config.NumberColumn(format="%.2f"),
                "Score": st.column_config.NumberColumn(format="%.1f", help="De 0 à 100"),
                "Poids (%)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            },
        )
        st.caption("Un poids de 0 signifie que l'optimisation a écarté l'action malgré son bon score.")
    with droite:
        par_secteur = (classement.assign(secteur=classement["secteur"].fillna("Inconnu"))
                       .groupby("secteur")["poids"].sum().reset_index())
        figure = px.pie(par_secteur, names="secteur", values="poids", hole=0.45,
                        title="Répartition par secteur")
        st.plotly_chart(figure)

# --- Détail du score --------------------------------------------------------
with onglet_detail:
    st.write("Chaque barre montre les points de score (sur 100) apportés par chaque critère.")
    points = [c for c in classement.columns if c in CRITERES or c == "bonus_famille"]
    long = (classement[points].rename(columns=NOMS_CRITERES).reset_index()
            .melt(id_vars="ticker", var_name="Critère", value_name="Points"))
    figure = px.bar(long, x="Points", y="ticker", color="Critère", orientation="h",
                    title=f"Score par critère : profil {profil_affiche.lower()}")
    figure.update_yaxes(autorange="reversed", title="")
    st.plotly_chart(figure)

    for ticker, ligne in classement.iterrows():
        with st.expander(f"{ticker} · {ligne['nom']} · score {ligne['score']:.1f}"):
            st.write(ligne["explication"])

# --- Carte rendement / risque -----------------------------------------------
with onglet_carte:
    carte = univers.copy()
    carte["Groupe"] = "Autres actions"
    carte.loc[classement.index, "Groupe"] = "Actions retenues"
    figure = px.scatter(
        carte, x="volatilite_annuelle", y="rendement_annuel", color="Groupe", hover_name="nom",
        hover_data={"secteur": True, "famille": True, "Groupe": False},
        labels={"volatilite_annuelle": "Volatilité annualisée", "rendement_annuel": "Rendement annualisé"},
        color_discrete_map={"Autres actions": "#B0B7C3", "Actions retenues": "#E4572E"},
        title="Rendement et risque des 3 dernières années",
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure)

# --- Backtest ---------------------------------------------------------------
with onglet_backtest:
    st.write(
        "Le backtest rejoue le passé : à chaque fin de trimestre, le portefeuille est reconstruit avec "
        "**uniquement** les données disponibles à cette date, puis détenu jusqu'au trimestre suivant."
    )
    courbes = charger_backtest(profil)
    if courbes is None:
        st.info(f"Pas de backtest enregistré pour le profil {profil_affiche.lower()}. "
                f"Lance `python -m src.backtest --profil {profil}` puis relance l'application.")
    else:
        figure = px.line(courbes, title="Valeur d'un capital de 1 investi au départ")
        figure.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Valeur")
        st.plotly_chart(figure)

        tableau = metriques(courbes)
        for colonne in ["Rendement total", "Rendement annualisé"]:
            tableau[colonne] = (tableau[colonne] * 100).map("{:+.1f} %".format)
        for colonne in ["Volatilité annualisée", "Drawdown max"]:
            tableau[colonne] = (tableau[colonne] * 100).map("{:.1f} %".format)
        tableau["Sharpe"] = tableau["Sharpe"].map("{:.2f}".format)
        st.dataframe(tableau)
        st.caption(
            "Paramètres : 10 actions, rééquilibrage trimestriel, frais de 0,1 % du montant échangé, sans "
            "filtres (le backtest porte sur le profil seul). L'univers est celui d'aujourd'hui, ce qui "
            "avantage tous les résultats (biais de survie). Les performances passées ne garantissent pas "
            "les performances futures."
        )

# --- Méthode et limites -----------------------------------------------------
with onglet_methode:
    st.subheader("Comment ça marche")
    st.markdown(
        """
1. **Données** : prix historiques ajustés (splits et dividendes) de 10 ans, via Yahoo Finance.
2. **Indicateurs** : rendement, volatilité, ratio de Sharpe, bêta, drawdown maximal et momentum,
   calculés sur les 3 dernières années.
3. **Score** : chaque indicateur devient une note de 0 à 1 (rang parmi toutes les actions), puis les notes
   sont pondérées selon le profil.
4. **Familles** : un algorithme de clustering (K-means) regroupe les actions en 4 familles de comportement ;
   la famille compatible avec le profil apporte un petit bonus.
5. **Portefeuille** : les meilleures actions sont retenues, puis leurs poids sont optimisés (méthode de
   Markowitz, avec plafond par action).
        """
    )
    poids_profils = pd.DataFrame(PROFILS).T[list(NOMS_CRITERES)[:6]]
    poids_profils = (poids_profils * 100).round().astype(int).astype(str) + " %"
    poids_profils.columns = [NOMS_CRITERES[c] for c in poids_profils.columns]
    poids_profils.index = ["Prudent", "Équilibré", "Dynamique"]
    st.write("**Pondérations de départ par profil** (hypothèses de modélisation, ajustées ensuite selon "
             "l'horizon et la préférence pour les dividendes) :")
    st.table(poids_profils)

    st.subheader("Limites")
    st.markdown(
        """
- **Biais de survie** : l'univers est composé des actions actuelles ; les entreprises disparues ou sorties des
  indices n'y figurent pas.
- **Passé, pas avenir** : le score décrit le comportement passé. Les pondérations sont des hypothèses.
- **Devises** : euros et dollars ne sont pas convertis.
- **Données gratuites** : Yahoo Finance peut contenir des erreurs ou des valeurs manquantes.
- **Fondamentaux** (PER, dividende) : seules les valeurs actuelles sont disponibles ; ils ne sont donc pas
  utilisés dans le backtest.
- **Frais** simplifiés : pas d'écart acheteur/vendeur ni de fiscalité.
        """
    )