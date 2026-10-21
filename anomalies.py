# -*- coding: utf-8 -*-
"""
Detection d'anomalies dans les depenses — service de notation.

Expose un routeur FastAPI monte par api.py. Le backend Spring envoie la
liste des depenses, le service renvoie chacune avec son score et, pour
celles qui sont signalees, les raisons du signalement.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

RACINE = Path(__file__).parent
MODELE = RACINE / "modele"

try:
    paquet      = joblib.load(MODELE / "modele_anomalies.joblib")
    reference   = json.loads((MODELE / "reference_depenses.json").read_text(encoding="utf-8"))
    metriques   = json.loads((MODELE / "metriques_anomalies.json").read_text(encoding="utf-8"))
    DISPONIBLE  = True
except FileNotFoundError:
    paquet, reference, metriques, DISPONIBLE = None, {}, {}, False

routeur = APIRouter(prefix="/anomalies", tags=["Anomalies de depenses"])


# ─────────────────────────────────────────────── contrats d'echange

class Depense(BaseModel):
    """Tous les champs tolerent la valeur nulle : la base de l'ERP contient
    des depenses sans service ni fournisseur renseigne."""

    id:          Optional[int]   = None
    title:       Optional[str]   = None
    category:    Optional[str]   = None
    amount:      Optional[float] = Field(None, ge=0)
    expenseDate: Optional[str]   = None     # AAAA-MM-JJ
    department:  Optional[str]   = None
    vendor:      Optional[str]   = None


class Raison(BaseModel):
    libelle: str
    gravite: str            # FORTE | MOYENNE


class DepenseNotee(BaseModel):
    id:          Optional[int]
    title:       str
    category:    str
    amount:      float
    expenseDate: Optional[str]
    department:  Optional[str]
    vendor:      Optional[str]
    score:       float
    signalee:    bool
    raisons:     list[Raison]


class ResultatAnalyse(BaseModel):
    depenses:         list[DepenseNotee]
    nombreAnalysees:  int
    nombreSignalees:  int
    seuilScore:       float
    algorithme:       str
    tailleHistorique: int


# ─────────────────────────────────────────────── calcul des variables

def _stats(cle: str, valeur) -> dict:
    defaut = {"mediane": reference.get("montant_median_global", 1.0),
              "ecart":   reference.get("ecart_global", 1.0)}
    return reference.get(cle, {}).get(str(valeur), defaut)


def _variables(donnees: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(donnees["expenseDate"], errors="coerce")

    variables = pd.DataFrame(index=donnees.index)
    variables["log_montant"] = np.log1p(donnees["amount"])

    variables["ecart_categorie"] = donnees.apply(
        lambda l: (l["amount"] - _stats("categories", l["category"])["mediane"])
                  / _stats("categories", l["category"])["ecart"], axis=1)

    variables["ecart_departement"] = donnees.apply(
        lambda l: (l["amount"] - _stats("departements", l["department"])["mediane"])
                  / _stats("departements", l["department"])["ecart"], axis=1)

    variables["mois"] = dates.dt.month.fillna(6).astype(int)
    variables["fin_exercice"] = dates.dt.month.isin([11, 12]).fillna(False).astype(int)

    occurrences = donnees["vendor"].map(
        lambda v: reference.get("fournisseurs", {}).get(str(v), 0))
    variables["fournisseur_rare"] = (occurrences <= 1).astype(int)

    return variables[paquet["colonnes"]]


def _texte(valeur) -> Optional[str]:
    """None, NaN et chaines vides deviennent uniformement None.

    Pandas convertit les valeurs nulles en NaN, un flottant que pydantic
    refuse pour un champ texte.
    """
    if valeur is None:
        return None
    if isinstance(valeur, float) and np.isnan(valeur):
        return None
    texte = str(valeur).strip()
    return texte or None


def _entier(valeur) -> Optional[int]:
    if valeur is None:
        return None
    if isinstance(valeur, float) and np.isnan(valeur):
        return None
    try:
        return int(valeur)
    except (TypeError, ValueError):
        return None


def _raisons(ligne: pd.Series, vars_ligne: pd.Series) -> list[Raison]:
    """Explique le signalement a partir des variables les plus extremes."""
    raisons: list[Raison] = []

    ecart_cat = vars_ligne["ecart_categorie"]
    if ecart_cat >= 2:
        mediane = _stats("categories", ligne["category"])["mediane"]
        raisons.append(Raison(
            libelle=f"Montant tres superieur a l'habitude de la categorie "
                    f"{ligne['category']} (mediane {mediane:,.0f} TND)".replace(",", " "),
            gravite="FORTE" if ecart_cat >= 4 else "MOYENNE"))

    ecart_dep = vars_ligne["ecart_departement"]
    if ecart_dep >= 2 and _texte(ligne.get("department")):
        mediane = _stats("departements", ligne["department"])["mediane"]
        raisons.append(Raison(
            libelle=f"Montant inhabituel pour le service {ligne['department']} "
                    f"(mediane {mediane:,.0f} TND)".replace(",", " "),
            gravite="FORTE" if ecart_dep >= 4 else "MOYENNE"))

    if vars_ligne["fournisseur_rare"] == 1 and _texte(ligne.get("vendor")):
        raisons.append(Raison(
            libelle=f"Fournisseur sans antecedent dans l'historique : {ligne['vendor']}",
            gravite="MOYENNE"))

    q90 = reference.get("montant_q90")
    if q90 and ligne["amount"] >= q90:
        raisons.append(Raison(
            libelle=f"Montant parmi les 10 % les plus eleves de l'historique "
                    f"(au-dela de {q90:,.0f} TND)".replace(",", " "),
            gravite="FORTE" if ligne["amount"] >= q90 * 1.5 else "MOYENNE"))

    if vars_ligne["fin_exercice"] == 1:
        raisons.append(Raison(
            libelle="Depense engagee en fin d'exercice budgetaire",
            gravite="MOYENNE"))

    if not raisons:
        raisons.append(Raison(
            libelle="Combinaison inhabituelle de montant, categorie et periode",
            gravite="MOYENNE"))

    return raisons[:3]


# ─────────────────────────────────────────────── points d'entree

@routeur.get("/sante")
def sante():
    if not DISPONIBLE:
        raise HTTPException(503, "Modele d'anomalies absent. Lancez train_anomalies.py.")
    return {
        "statut": "operationnel",
        "algorithme": metriques.get("algorithme"),
        "apprentissage": metriques.get("apprentissage"),
        "tailleHistorique": metriques.get("taille_historique"),
        "tauxSignalement": metriques.get("taux_signalement"),
        "variables": metriques.get("variables"),
    }


@routeur.post("/analyser", response_model=ResultatAnalyse)
def analyser(depenses: list[Depense]):
    if not DISPONIBLE:
        raise HTTPException(503, "Modele d'anomalies absent. Lancez train_anomalies.py.")
    if not depenses:
        raise HTTPException(400, "Aucune depense a analyser.")

    donnees = pd.DataFrame([d.model_dump() for d in depenses])
    donnees["category"]   = donnees["category"].map(_texte).fillna("OTHER")
    donnees["amount"]     = pd.to_numeric(donnees["amount"], errors="coerce").fillna(0.0)
    donnees["department"] = donnees["department"].map(_texte)
    donnees["vendor"]     = donnees["vendor"].map(_texte)

    try:
        variables   = _variables(donnees)
        normalisees = paquet["normaliseur"].transform(variables)
        scores      = paquet["modele"].score_samples(normalisees)
        signalees   = paquet["modele"].predict(normalisees) == -1
    except Exception as erreur:
        raise HTTPException(400, f"Analyse impossible : {erreur}")

    resultats = []
    for position, (indice, ligne) in enumerate(donnees.iterrows()):
        est_signalee = bool(signalees[position])
        resultats.append(DepenseNotee(
            id=_entier(ligne.get("id")),
            title=_texte(ligne.get("title")) or "(sans libelle)",
            category=_texte(ligne.get("category")) or "OTHER",
            amount=float(ligne.get("amount") or 0),
            expenseDate=_texte(ligne.get("expenseDate")),
            department=_texte(ligne.get("department")),
            vendor=_texte(ligne.get("vendor")),
            score=round(float(scores[position]), 4),
            signalee=est_signalee,
            raisons=_raisons(ligne, variables.loc[indice]) if est_signalee else [],
        ))

    # Du plus atypique au plus banal
    resultats.sort(key=lambda d: d.score)

    return ResultatAnalyse(
        depenses=resultats,
        nombreAnalysees=len(resultats),
        nombreSignalees=int(signalees.sum()),
        seuilScore=metriques.get("seuil_score", 0.0),
        algorithme=metriques.get("algorithme", "Isolation Forest"),
        tailleHistorique=metriques.get("taille_historique", 0),
    )
