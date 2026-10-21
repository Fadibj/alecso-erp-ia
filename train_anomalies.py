# -*- coding: utf-8 -*-
"""
Detection d'anomalies dans les depenses — entrainement du modele.

Apprentissage NON SUPERVISE : aucune depense n'est etiquetee comme
suspecte. Le modele apprend ce qui est habituel dans l'historique et
signale ce qui s'en ecarte.

Algorithme : Isolation Forest. Il isole chaque observation par des
decoupages aleatoires ; une observation atypique est isolee en peu de
decoupages, ce qui lui vaut un score d'anormalite eleve.

Produit :
  modele/modele_anomalies.joblib   modele entraine
  modele/reference_depenses.json   statistiques de reference par categorie et service
  modele/metriques_anomalies.json  seuil, taux de signalement, depenses signalees
  figures/                         trois figures pour le rapport
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RACINE  = Path(__file__).parent
DONNEES = RACINE / "data" / "depenses.csv"
MODELE  = RACINE / "modele"
FIGURES = RACINE / "figures"
MODELE.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

ALEA = 42
PART_ANORMALE = 0.12        # proportion attendue de depenses atypiques
VERT, ROUGE, GRIS = "#2c6b5a", "#c62828", "#b0c4bc"


# ───────────────────────────────────────────── construction des variables

def statistiques_reference(donnees: pd.DataFrame) -> dict:
    """Mediane et dispersion par categorie et par service.

    On utilise la mediane et l'ecart absolu median plutot que la moyenne
    et l'ecart type : ces deux mesures ne sont pas tirees par les valeurs
    extremes, ce qui compte justement quand on cherche des valeurs extremes.
    """
    def resume(groupe):
        mediane = float(groupe.median())
        ecart   = float((groupe - mediane).abs().median())
        return {"mediane": mediane, "ecart": max(ecart, 1.0)}

    return {
        "categories":   {str(c): resume(g) for c, g in donnees.groupby("category")["amount"]},
        "departements": {str(d): resume(g) for d, g in donnees.groupby("department")["amount"]},
        "fournisseurs": donnees["vendor"].value_counts().to_dict(),
        "montant_median_global": float(donnees["amount"].median()),
        "montant_q90": float(donnees["amount"].quantile(0.90)),
        "montant_max": float(donnees["amount"].max()),
        "ecart_global": max(float((donnees["amount"] - donnees["amount"].median()).abs().median()), 1.0),
    }


def construire_variables(donnees: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Transforme chaque depense en six indicateurs numeriques interpretables."""
    dates = pd.to_datetime(donnees["expense_date"])

    def ecart_relatif(ligne, cle, colonne):
        stats = reference[cle].get(str(ligne[colonne]))
        if stats is None:
            stats = {"mediane": reference["montant_median_global"],
                     "ecart":   reference["ecart_global"]}
        return (ligne["amount"] - stats["mediane"]) / stats["ecart"]

    variables = pd.DataFrame(index=donnees.index)

    # Montant en echelle logarithmique : les depenses vont de 1 450 a 42 000 TND
    variables["log_montant"] = np.log1p(donnees["amount"])

    # Ecart au comportement habituel de la categorie et du service
    variables["ecart_categorie"]   = donnees.apply(lambda l: ecart_relatif(l, "categories", "category"), axis=1)
    variables["ecart_departement"] = donnees.apply(lambda l: ecart_relatif(l, "departements", "department"), axis=1)

    # Moment dans l'exercice : les depenses de fin d'annee meritent un regard
    variables["mois"] = dates.dt.month
    variables["fin_exercice"] = dates.dt.month.isin([11, 12]).astype(int)

    # Un fournisseur qui n'apparait qu'une fois pour un montant eleve est atypique
    occurrences = donnees["vendor"].map(
        lambda v: reference["fournisseurs"].get(str(v), 0))
    variables["fournisseur_rare"] = (occurrences <= 1).astype(int)

    return variables


# ───────────────────────────────────────────── figures

def tracer_distribution(scores, seuil):
    figure, axe = plt.subplots(figsize=(6.6, 4.0))

    axe.hist(scores, bins=18, color=VERT, alpha=0.82, edgecolor="white")
    axe.axvline(seuil, color=ROUGE, linestyle="--", linewidth=2,
                label=f"Seuil de signalement ({seuil:.3f})")

    axe.set_xlabel("Score d'anormalite — plus il est bas, plus la depense est atypique")
    axe.set_ylabel("Nombre de depenses")
    axe.set_title("Distribution des scores")
    axe.legend()
    axe.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGURES / "distribution_scores.png", dpi=160)
    plt.close(figure)


def tracer_nuage(donnees, anomalies):
    figure, axe = plt.subplots(figsize=(7.6, 4.4))
    dates = pd.to_datetime(donnees["expense_date"])

    normales = ~anomalies
    axe.scatter(dates[normales], donnees["amount"][normales],
                s=55, color=VERT, alpha=0.75, label="Depense habituelle")
    axe.scatter(dates[anomalies], donnees["amount"][anomalies],
                s=130, color=ROUGE, marker="X", label="Signalee")

    for i in donnees.index[anomalies]:
        axe.annotate(donnees.loc[i, "title"][:26],
                     (dates[i], donnees.loc[i, "amount"]),
                     textcoords="offset points", xytext=(0, 11),
                     fontsize=7.5, ha="center", color="#7a1c1c")

    axe.set_ylabel("Montant (TND)")
    axe.set_title("Depenses de l'exercice et signalements")
    axe.legend()
    axe.grid(alpha=0.25)
    figure.autofmt_xdate()

    figure.tight_layout()
    figure.savefig(FIGURES / "nuage_depenses.png", dpi=160)
    plt.close(figure)


def tracer_par_service(donnees, anomalies):
    comptes = (donnees.assign(signalee=anomalies)
                      .groupby("department")["signalee"]
                      .agg(["sum", "count"]))
    comptes = comptes.sort_values("count")

    figure, axe = plt.subplots(figsize=(7.0, 4.0))
    axe.barh(comptes.index, comptes["count"], color=GRIS, label="Depenses")
    axe.barh(comptes.index, comptes["sum"],   color=ROUGE, label="Signalees")

    axe.set_xlabel("Nombre de depenses")
    axe.set_title("Repartition des signalements par service")
    axe.legend()
    axe.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(FIGURES / "signalements_par_service.png", dpi=160)
    plt.close(figure)


# ───────────────────────────────────────────── programme

def main():
    donnees = pd.read_csv(DONNEES)
    print(f"Historique      : {len(donnees)} depenses")
    print(f"Periode         : {donnees['expense_date'].min()} au {donnees['expense_date'].max()}")
    print(f"Montant total   : {donnees['amount'].sum():,.0f} TND\n")

    reference = statistiques_reference(donnees)
    variables = construire_variables(donnees, reference)

    normaliseur = StandardScaler()
    normalisees = normaliseur.fit_transform(variables)

    modele = IsolationForest(
        n_estimators=300,
        contamination=PART_ANORMALE,
        max_samples="auto",
        random_state=ALEA,
        n_jobs=-1,
    )
    modele.fit(normalisees)

    scores    = modele.score_samples(normalisees)
    anomalies = modele.predict(normalisees) == -1
    seuil     = float(scores[anomalies].max()) if anomalies.any() else float(scores.min())

    print(f"Variables       : {', '.join(variables.columns)}")
    print(f"Signalees       : {anomalies.sum()} sur {len(donnees)} "
          f"({anomalies.mean():.0%})\n")

    print("--- Depenses signalees, de la plus atypique a la moins ---")
    signalees = donnees[anomalies].assign(score=scores[anomalies]).sort_values("score")
    for _, ligne in signalees.iterrows():
        print(f"  {ligne['score']:+.3f}  {ligne['amount']:>10,.0f} TND  "
              f"{ligne['category']:<12} {ligne['department']:<20} {ligne['title'][:40]}")

    tracer_distribution(scores, seuil)
    tracer_nuage(donnees, anomalies)
    tracer_par_service(donnees, anomalies)

    metriques = {
        "algorithme": "Isolation Forest",
        "apprentissage": "non supervise",
        "taille_historique": int(len(donnees)),
        "part_anormale_attendue": PART_ANORMALE,
        "nombre_signalees": int(anomalies.sum()),
        "taux_signalement": round(float(anomalies.mean()), 4),
        "seuil_score": round(seuil, 4),
        "variables": list(variables.columns),
        "score_min": round(float(scores.min()), 4),
        "score_max": round(float(scores.max()), 4),
    }

    joblib.dump({"modele": modele, "normaliseur": normaliseur,
                 "colonnes": list(variables.columns)},
                MODELE / "modele_anomalies.joblib")

    (MODELE / "reference_depenses.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODELE / "metriques_anomalies.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModele   : {MODELE / 'modele_anomalies.joblib'}")
    print(f"Figures  : {FIGURES}")


if __name__ == "__main__":
    main()
