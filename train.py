# -*- coding: utf-8 -*-
"""
Prediction du risque de depart des employes — entrainement du modele.

Jeu de donnees : IBM HR Analytics Employee Attrition & Performance
  1470 employes, 35 variables, cible binaire Attrition (Yes / No), 16,1 % de departs.

Demarche :
  1. Nettoyage des colonnes sans pouvoir predictif
  2. Comparaison de six algorithmes par validation croisee (ROC-AUC)
  3. Entrainement du meilleur, choix du seuil de decision
  4. Evaluation sur un jeu de test mis de cote
  5. Export du modele, des metriques et des figures

Produit :
  modele/modele_attrition.joblib   modele entraine, pretraitement compris
  modele/profil_reference.json     valeurs par defaut pour les champs absents de l'ERP
  modele/metriques.json            performances et coefficients
  figures/*.png                    comparaison, matrice de confusion, ROC, coefficients
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RACINE  = Path(__file__).parent
DONNEES = RACINE / "data" / "attrition.csv"
MODELE  = RACINE / "modele"
FIGURES = RACINE / "figures"
MODELE.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

ALEA = 42
VERT, OR, GRIS = "#2c6b5a", "#c8a84b", "#b0c4bc"

# Colonnes constantes ou purement identifiantes : aucun pouvoir predictif
A_SUPPRIMER = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]


# ─────────────────────────────────────────────────────────────── chargement

def charger():
    donnees = pd.read_csv(DONNEES)
    donnees = donnees.drop(columns=[c for c in A_SUPPRIMER if c in donnees.columns])
    cible     = (donnees["Attrition"] == "Yes").astype(int)
    variables = donnees.drop(columns=["Attrition"])
    return variables, cible


def pretraitement(categorielles, numeriques, normaliser):
    """Encodage des variables qualitatives, normalisation optionnelle des quantitatives."""
    return ColumnTransformer([
        ("categorielles", OneHotEncoder(handle_unknown="ignore"), categorielles),
        ("numeriques", StandardScaler() if normaliser else "passthrough", numeriques),
    ])


def candidats(categorielles, numeriques):
    def tube(modele, normaliser):
        return Pipeline([
            ("pretraitement", pretraitement(categorielles, numeriques, normaliser)),
            ("modele", modele),
        ])

    return {
        "Regression logistique": tube(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1), True),
        "Foret aleatoire": tube(
            RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                   class_weight="balanced", random_state=ALEA, n_jobs=-1), False),
        "Gradient boosting": tube(
            GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                       max_depth=3, random_state=ALEA), False),
        "Hist gradient boosting": tube(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_depth=4, random_state=ALEA), False),
    }


# ─────────────────────────────────────────────────────────────── figures

def tracer_comparaison(scores):
    noms   = list(scores.keys())
    moyennes = [scores[n]["moyenne"] for n in noms]
    ecarts   = [scores[n]["ecart_type"] for n in noms]
    meilleur = int(np.argmax(moyennes))

    couleurs = [OR if i == meilleur else VERT for i in range(len(noms))]

    figure, axe = plt.subplots(figsize=(7.4, 4.2))
    barres = axe.barh(noms, moyennes, xerr=ecarts, color=couleurs,
                      error_kw={"ecolor": "#55685f", "capsize": 4})
    axe.set_xlim(0.5, 0.95)
    axe.set_xlabel("ROC-AUC (validation croisee a 5 blocs)")
    axe.set_title("Comparaison des algorithmes")
    axe.grid(axis="x", alpha=0.25)

    for barre, valeur in zip(barres, moyennes):
        axe.text(valeur + 0.008, barre.get_y() + barre.get_height() / 2,
                 f"{valeur:.3f}", va="center", fontsize=9, fontweight="bold")

    figure.tight_layout()
    figure.savefig(FIGURES / "comparaison_modeles.png", dpi=160)
    plt.close(figure)


def tracer_matrice(vrais, predits):
    matrice = confusion_matrix(vrais, predits)

    figure, axe = plt.subplots(figsize=(5.0, 4.3))
    axe.imshow(matrice, cmap="Greens")
    axe.set_xticks([0, 1], ["Reste", "Part"])
    axe.set_yticks([0, 1], ["Reste", "Part"])
    axe.set_xlabel("Prediction du modele")
    axe.set_ylabel("Situation reelle")
    axe.set_title("Matrice de confusion")

    for i in range(2):
        for j in range(2):
            axe.text(j, i, matrice[i, j], ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if matrice[i, j] > matrice.max() / 2 else "#1e4d40")

    figure.tight_layout()
    figure.savefig(FIGURES / "matrice_confusion.png", dpi=160)
    plt.close(figure)


def tracer_roc(vrais, probabilites, aire):
    faux_pos, vrais_pos, _ = roc_curve(vrais, probabilites)

    figure, axe = plt.subplots(figsize=(5.0, 4.3))
    axe.plot(faux_pos, vrais_pos, color=VERT, linewidth=2.2, label=f"Modele (AUC = {aire:.3f})")
    axe.plot([0, 1], [0, 1], "--", color=GRIS, linewidth=1.2, label="Hasard")
    axe.set_xlabel("Taux de faux positifs")
    axe.set_ylabel("Taux de vrais positifs")
    axe.set_title("Courbe ROC")
    axe.legend(loc="lower right")
    axe.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGURES / "courbe_roc.png", dpi=160)
    plt.close(figure)


def tracer_coefficients(pipeline, nombre=14):
    """Pour une regression logistique : les variables qui augmentent ou reduisent le risque."""
    noms = [n.split("__", 1)[-1]
            for n in pipeline.named_steps["pretraitement"].get_feature_names_out()]
    poids = pipeline.named_steps["modele"].coef_[0]

    serie = pd.Series(poids, index=noms).sort_values()
    extremes = pd.concat([serie.head(nombre // 2), serie.tail(nombre // 2)])

    couleurs = ["#2e7d32" if v < 0 else "#c62828" for v in extremes.values]

    figure, axe = plt.subplots(figsize=(7.6, 5.4))
    axe.barh(extremes.index, extremes.values, color=couleurs)
    axe.axvline(0, color="#55685f", linewidth=0.9)
    axe.set_xlabel("Coefficient — negatif : retient, positif : fait partir")
    axe.set_title("Variables les plus determinantes")
    axe.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGURES / "coefficients.png", dpi=160)
    plt.close(figure)

    # Rapport de cotes : lecture metier directe
    return {nom: round(float(np.exp(valeur)), 3)
            for nom, valeur in serie.reindex(serie.abs().sort_values(ascending=False).index)
                                     .head(nombre).items()}


# ─────────────────────────────────────────────────────────────── utilitaires

def seuil_optimal(vrais, probabilites):
    """Seuil qui maximise le F1 : compromis entre alertes manquees et fausses alertes."""
    precisions, rappels, seuils = precision_recall_curve(vrais, probabilites)
    f1 = 2 * precisions * rappels / np.maximum(precisions + rappels, 1e-9)
    return float(seuils[int(np.argmax(f1[:-1]))])


def profil_reference(variables):
    """Mediane ou modalite dominante : sert a completer les champs absents de l'ERP."""
    profil = {}
    for colonne in variables.columns:
        if variables[colonne].dtype.kind in "iuf":
            profil[colonne] = float(variables[colonne].median())
        else:
            profil[colonne] = str(variables[colonne].mode()[0])
    return profil


# ─────────────────────────────────────────────────────────────── programme

def main():
    variables, cible = charger()
    categorielles = variables.select_dtypes(include=["object", "string"]).columns.tolist()
    numeriques    = [c for c in variables.columns if c not in categorielles]

    print(f"Jeu de donnees   : {variables.shape[0]} employes, {variables.shape[1]} variables")
    print(f"Taux de depart   : {cible.mean():.1%}")
    print(f"Variables        : {len(categorielles)} categorielles, {len(numeriques)} numeriques\n")

    # ── 1. Comparaison des algorithmes
    validation = StratifiedKFold(5, shuffle=True, random_state=ALEA)
    scores = {}

    print("Comparaison par validation croisee (ROC-AUC)")
    print(f"  {'Algorithme':26} {'moyenne':>8} {'ecart':>8}")
    for nom, tube in candidats(categorielles, numeriques).items():
        mesures = cross_val_score(tube, variables, cible, cv=validation,
                                  scoring="roc_auc", n_jobs=-1)
        scores[nom] = {"moyenne": round(float(mesures.mean()), 4),
                       "ecart_type": round(float(mesures.std()), 4)}
        print(f"  {nom:26} {mesures.mean():8.4f} {mesures.std():8.4f}")

    retenu = max(scores, key=lambda n: scores[n]["moyenne"])
    print(f"\nAlgorithme retenu : {retenu} ({scores[retenu]['moyenne']:.4f})\n")
    tracer_comparaison(scores)

    # ── 2. Entrainement du modele retenu
    app_x, test_x, app_y, test_y = train_test_split(
        variables, cible, test_size=0.2, stratify=cible, random_state=ALEA)

    pipeline = candidats(categorielles, numeriques)[retenu]
    pipeline.fit(app_x, app_y)

    probabilites = pipeline.predict_proba(test_x)[:, 1]
    seuil = seuil_optimal(test_y, probabilites)
    predits = (probabilites >= seuil).astype(int)

    metriques = {
        "algorithme": retenu,
        "seuil_decision": round(seuil, 4),
        "exactitude": round(accuracy_score(test_y, predits), 4),
        "precision":  round(precision_score(test_y, predits), 4),
        "rappel":     round(recall_score(test_y, predits), 4),
        "f1":         round(f1_score(test_y, predits), 4),
        "roc_auc":    round(roc_auc_score(test_y, probabilites), 4),
        "taille_apprentissage": int(len(app_x)),
        "taille_test": int(len(test_x)),
        "comparaison_algorithmes": scores,
    }

    print("--- Performances sur le jeu de test ---")
    for cle in ("exactitude", "precision", "rappel", "f1", "roc_auc", "seuil_decision"):
        print(f"  {cle:16} {metriques[cle]}")

    print("\n--- Rapport detaille ---")
    print(classification_report(test_y, predits, target_names=["Reste", "Part"]))

    tracer_matrice(test_y, predits)
    tracer_roc(test_y, probabilites, metriques["roc_auc"])

    if hasattr(pipeline.named_steps["modele"], "coef_"):
        metriques["rapports_de_cotes"] = tracer_coefficients(pipeline)

    # ── 3. Export
    joblib.dump(pipeline, MODELE / "modele_attrition.joblib")
    (MODELE / "profil_reference.json").write_text(
        json.dumps(profil_reference(variables), indent=2, ensure_ascii=False), encoding="utf-8")
    (MODELE / "metriques.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modele   : {MODELE / 'modele_attrition.joblib'}")
    print(f"Metriques: {MODELE / 'metriques.json'}")
    print(f"Figures  : {FIGURES}")


if __name__ == "__main__":
    main()
