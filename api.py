# -*- coding: utf-8 -*-
"""
Service de prediction du risque de depart — ALECSO ERP.

Expose le modele entraine par train.py sous forme d'API REST, consommee
par le backend Spring Boot.

Lancement :
    uvicorn api:app --host 0.0.0.0 --port 8000

Points d'entree :
    GET  /sante      etat du service et performances du modele
    POST /predire    score de risque pour un employe
    GET  /docs       documentation interactive (fournie par FastAPI)
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anomalies
from subventions_ia import routeur as routeur_subventions

RACINE = Path(__file__).parent
MODELE = RACINE / "modele"

# ─────────────────────────────────────────────── chargement au demarrage

try:
    pipeline  = joblib.load(MODELE / "modele_attrition.joblib")
    profil    = json.loads((MODELE / "profil_reference.json").read_text(encoding="utf-8"))
    metriques = json.loads((MODELE / "metriques.json").read_text(encoding="utf-8"))
except FileNotFoundError as erreur:
    raise SystemExit(
        f"Modele introuvable ({erreur}). Lancez d'abord : python train.py"
    )

SEUIL = metriques.get("seuil_decision", 0.5)

app = FastAPI(
    title="ALECSO ERP — Prediction du risque de depart",
    description="Modele de regression logistique entraine sur le jeu "
                "IBM HR Analytics Employee Attrition (1470 employes).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(anomalies.routeur)
app.include_router(routeur_subventions)


# ─────────────────────────────────────────────── contrats d'echange

class DemandePrediction(BaseModel):
    """Champs fournis par l'ERP. Tous facultatifs : les manquants sont
    completes par le profil median de la population de reference."""

    age:                    Optional[int]   = Field(None, ge=18, le=70)
    departement:            Optional[str]   = None
    genre:                  Optional[str]   = Field(None, description="MALE ou FEMALE")
    situationFamiliale:     Optional[str]   = Field(None, description="CELIBATAIRE, MARIE, DIVORCE")
    salaireMensuel:         Optional[float] = Field(None, ge=0)
    anciennete:             Optional[int]   = Field(None, ge=0, description="annees dans l'organisation")
    experienceTotale:       Optional[int]   = Field(None, ge=0)
    anneesDepuisPromotion:  Optional[int]   = Field(None, ge=0)
    anneesMemeManager:      Optional[int]   = Field(None, ge=0)
    heuresSupplementaires:  Optional[bool]  = None
    deplacementsFrequents:  Optional[bool]  = None
    niveauPoste:            Optional[int]   = Field(None, ge=1, le=5)
    satisfactionTravail:    Optional[int]   = Field(None, ge=1, le=4)
    satisfactionEnvironnement: Optional[int] = Field(None, ge=1, le=4)
    equilibreViePro:        Optional[int]   = Field(None, ge=1, le=4)
    distanceDomicile:       Optional[int]   = Field(None, ge=0)
    formationsAnneePassee:  Optional[int]   = Field(None, ge=0)
    nombreEmployeursPrecedents: Optional[int] = Field(None, ge=0)


class Facteur(BaseModel):
    libelle: str
    effet:   str   # "RISQUE" (pousse au depart) ou "STABILITE" (retient)
    poids:   float


class ReponsePrediction(BaseModel):
    probabilite:   float
    pourcentage:   int
    niveau:        str          # FAIBLE, MODERE, ELEVE
    seuilAlerte:   float
    facteurs:      list[Facteur]
    champsFournis: int
    champsTotal:   int
    modele:        str
    rocAuc:        float


# ─────────────────────────────────────────────── traduction ERP → modele

# Les services de l'ALECSO n'ont pas d'equivalent exact dans le jeu IBM.
# On les rapproche du departement le plus proche par nature d'activite.
DEPARTEMENTS = {
    "Direction":           "Sales",
    "Ressources Humaines": "Human Resources",
    "Finance":             "Sales",
    "Informatique":        "Research & Development",
    "Logistique":          "Research & Development",
    "Communication":       "Sales",
}

SITUATIONS = {
    "CELIBATAIRE": "Single",
    "MARIE":       "Married",
    "MARIE(E)":    "Married",
    "DIVORCE":     "Divorced",
    "DIVORCE(E)":  "Divorced",
    "VEUF":        "Divorced",
    "VEUF / VEUVE": "Divorced",
}

# Libelles lisibles pour les variables renvoyees dans l'explication
LIBELLES = {
    "OverTime_Yes":                     "Heures supplementaires frequentes",
    "OverTime_No":                      "Pas d'heures supplementaires",
    "BusinessTravel_Travel_Frequently": "Deplacements frequents",
    "BusinessTravel_Non-Travel":        "Aucun deplacement",
    "BusinessTravel_Travel_Rarely":     "Deplacements occasionnels",
    "MaritalStatus_Single":             "Celibataire",
    "MaritalStatus_Married":            "Marie",
    "MaritalStatus_Divorced":           "Divorce",
    "YearsSinceLastPromotion":          "Anciennete depuis la derniere promotion",
    "YearsAtCompany":                   "Anciennete dans l'organisation",
    "YearsWithCurrManager":             "Anciennete avec le meme responsable",
    "YearsInCurrentRole":               "Anciennete dans le poste",
    "TotalWorkingYears":                "Experience professionnelle totale",
    "NumCompaniesWorked":               "Nombre d'employeurs precedents",
    "MonthlyIncome":                    "Niveau de remuneration",
    "Age":                              "Age",
    "DistanceFromHome":                 "Distance domicile-travail",
    "JobSatisfaction":                  "Satisfaction au travail",
    "EnvironmentSatisfaction":          "Satisfaction de l'environnement",
    "WorkLifeBalance":                  "Equilibre vie professionnelle / privee",
    "JobLevel":                         "Niveau hierarchique",
    "StockOptionLevel":                 "Interessement",
    "TrainingTimesLastYear":            "Formations suivies l'an dernier",
    "JobInvolvement":                   "Implication dans le poste",
    "Department_Sales":                 "Service de rattachement",
    "Department_Research & Development": "Service de rattachement",
    "Department_Human Resources":       "Service de rattachement",
    "Gender_Male":                      "Genre",
    "Gender_Female":                    "Genre",
}


def vers_colonnes_modele(demande: DemandePrediction) -> tuple[pd.DataFrame, list[str]]:
    """Construit la ligne attendue par le modele a partir des champs de l'ERP.

    Renvoie aussi la liste des colonnes reellement renseignees : seules
    celles-la pourront figurer dans l'explication, pour ne pas justifier
    un score par une valeur mediane que l'ERP n'a jamais fournie.
    """
    ligne = dict(profil)          # profil median de la population de reference
    renseignees: list[str] = []

    def poser(colonne, valeur):
        if valeur is not None:
            ligne[colonne] = valeur
            renseignees.append(colonne)

    poser("Age", demande.age)
    poser("MonthlyIncome", demande.salaireMensuel)
    poser("YearsAtCompany", demande.anciennete)
    poser("TotalWorkingYears", demande.experienceTotale)
    poser("YearsSinceLastPromotion", demande.anneesDepuisPromotion)
    poser("YearsWithCurrManager", demande.anneesMemeManager)
    poser("JobLevel", demande.niveauPoste)
    poser("JobSatisfaction", demande.satisfactionTravail)
    poser("EnvironmentSatisfaction", demande.satisfactionEnvironnement)
    poser("WorkLifeBalance", demande.equilibreViePro)
    poser("DistanceFromHome", demande.distanceDomicile)
    poser("TrainingTimesLastYear", demande.formationsAnneePassee)
    poser("NumCompaniesWorked", demande.nombreEmployeursPrecedents)

    if demande.genre is not None:
        poser("Gender", "Male" if demande.genre.upper().startswith("M") else "Female")

    if demande.departement is not None:
        poser("Department", DEPARTEMENTS.get(demande.departement, "Research & Development"))

    if demande.situationFamiliale is not None:
        poser("MaritalStatus", SITUATIONS.get(demande.situationFamiliale.upper(), "Single"))

    if demande.heuresSupplementaires is not None:
        poser("OverTime", "Yes" if demande.heuresSupplementaires else "No")

    if demande.deplacementsFrequents is not None:
        poser("BusinessTravel",
              "Travel_Frequently" if demande.deplacementsFrequents else "Travel_Rarely")

    return pd.DataFrame([ligne]), renseignees


def expliquer(ligne: pd.DataFrame, renseignees: list[str], nombre: int = 4) -> list[Facteur]:
    """Contribution de chaque variable au score de cet employe precis.

    Pour une regression logistique, la contribution vaut coefficient x valeur
    transformee : c'est ce qui rend la prediction explicable individuellement.
    """
    modele = pipeline.named_steps["modele"]
    if not hasattr(modele, "coef_"):
        return []

    transformee = pipeline.named_steps["pretraitement"].transform(ligne)
    if hasattr(transformee, "toarray"):
        transformee = transformee.toarray()

    noms = [n.split("__", 1)[-1]
            for n in pipeline.named_steps["pretraitement"].get_feature_names_out()]
    contributions = transformee[0] * modele.coef_[0]

    classement = sorted(zip(noms, contributions), key=lambda c: abs(c[1]), reverse=True)

    def provient_de_l_erp(nom_colonne: str) -> bool:
        return any(nom_colonne == c or nom_colonne.startswith(c + "_") for c in renseignees)

    facteurs = []
    for nom, contribution in classement:
        if abs(contribution) < 0.01 or not provient_de_l_erp(nom):
            continue
        facteurs.append(Facteur(
            libelle=LIBELLES.get(nom, nom.replace("_", " ")),
            effet="RISQUE" if contribution > 0 else "STABILITE",
            poids=round(float(abs(contribution)), 3),
        ))
        if len(facteurs) == nombre:
            break

    return facteurs


# ─────────────────────────────────────────────── points d'entree

@app.get("/sante")
def sante():
    return {
        "statut": "operationnel",
        "modele": metriques.get("algorithme"),
        "rocAuc": metriques.get("roc_auc"),
        "exactitude": metriques.get("exactitude"),
        "rappel": metriques.get("rappel"),
        "seuilAlerte": SEUIL,
        "tailleApprentissage": metriques.get("taille_apprentissage"),
    }


@app.post("/predire", response_model=ReponsePrediction)
def predire(demande: DemandePrediction):
    try:
        ligne, renseignees = vers_colonnes_modele(demande)
        probabilite = float(pipeline.predict_proba(ligne)[0, 1])
    except Exception as erreur:
        raise HTTPException(status_code=400, detail=f"Prediction impossible : {erreur}")

    if probabilite >= SEUIL:
        niveau = "ELEVE"
    elif probabilite >= SEUIL * 0.6:
        niveau = "MODERE"
    else:
        niveau = "FAIBLE"

    return ReponsePrediction(
        probabilite=round(probabilite, 4),
        pourcentage=int(round(probabilite * 100)),
        niveau=niveau,
        seuilAlerte=round(SEUIL, 4),
        facteurs=expliquer(ligne, renseignees),
        champsFournis=len(renseignees),
        champsTotal=len(profil),
        modele=metriques.get("algorithme", "inconnu"),
        rocAuc=metriques.get("roc_auc", 0.0),
    )
