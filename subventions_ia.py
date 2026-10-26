"""
Aide a la decision sur les demandes de subvention — service d'inference.

Monte sous /subventions dans l'application FastAPI principale :

    from subventions_ia import routeur as routeur_subventions
    app.include_router(routeur_subventions)

Le service est sans etat : il ne lit pas la base. L'anteriorite du
beneficiaire (nombre de demandes et d'accords passes) est transmise par le
backend, qui seul connait l'historique.
"""

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

RACINE = Path(__file__).resolve().parent
ARTEFACT = RACINE / "modele" / "subventions.joblib"

routeur = APIRouter(prefix="/subventions", tags=["Subventions"])

# --------------------------------------------------------------- chargement

try:
    _paquet = joblib.load(ARTEFACT)
    MODELE_ACCORD = _paquet["modele_accord"]
    MODELE_MONTANT = _paquet["modele_montant"]
    MEDIANES = _paquet["medianes_demande"]
    REPERES = _paquet["reperes"]
    CHARGE = True
except Exception as erreur:                                    # noqa: BLE001
    MODELE_ACCORD = MODELE_MONTANT = None
    MEDIANES, REPERES, CHARGE = {}, {}, False
    _ERREUR = str(erreur)

TYPES = {"MEMBER_STATE", "INSTITUTION", "ASSOCIATION", "INDIVIDUAL"}

CATEGORIES = {
    "type_beneficiaire_MEMBER_STATE": "Demandeur : État membre",
    "type_beneficiaire_INSTITUTION": "Demandeur : institution",
    "type_beneficiaire_ASSOCIATION": "Demandeur : association",
    "type_beneficiaire_INDIVIDUAL": "Demandeur : personne physique",
}

# Les variables numeriques sont centrees avant l'apprentissage : une valeur
# nulle devient un ecart negatif a la moyenne, et sa contribution n'est donc
# pas nulle. Etiqueter la variable par son nom ferait alors afficher « depot
# en fin d'exercice : favorable » sur un dossier depose en juin. Le libelle
# est donc construit a partir de la valeur reelle du dossier, pas du nom de
# la colonne.
def _libelle_numerique(nom: str, valeur: float) -> Optional[str]:
    if nom == "mois":
        return None                    # redondant avec fin_exercice, et non actionnable
    if nom == "fin_exercice":
        return "Dépôt en fin d'exercice" if valeur >= 0.5 else "Dépôt hors fin d'exercice"
    if nom == "ecart_aux_pairs":
        return ("Montant supérieur aux dossiers comparables" if valeur > 0
                else "Montant inférieur aux dossiers comparables")
    if nom == "candidat_recurrent":
        return "Bénéficiaire déjà candidat par le passé" if valeur >= 0.5 else "Première candidature"
    return nom.replace("_", " ")


def _libelle_categoriel(nom: str) -> str:
    if nom in CATEGORIES:
        return CATEGORIES[nom]
    if nom.startswith("thematique_"):
        return f"Thématique : {nom.removeprefix('thematique_').lower()}"
    return nom.replace("_", " ")


# ------------------------------------------------------------------ schemas

class Demande(BaseModel):
    """Tous les champs sont facultatifs : un dossier en cours de saisie doit
    pouvoir etre evalue avant d'etre complet."""

    typeBeneficiaire: Optional[str] = Field(None, description="MEMBER_STATE, INSTITUTION, ASSOCIATION ou INDIVIDUAL")
    thematique: Optional[str] = Field(None, description="EDUCATION, FORMATION, PATRIMOINE, CULTURE, SCIENCES, NUMERIQUE, TRADUCTION, JEUNESSE")
    montantDemande: Optional[float] = Field(None, ge=0)
    dateDemande: Optional[str] = Field(None, description="AAAA-MM-JJ ; aujourd'hui si absente")
    demandesAnterieures: Optional[int] = Field(None, ge=0)
    accordsAnterieurs: Optional[int] = Field(None, ge=0)


class Facteur(BaseModel):
    libelle: str
    effet: str          # FAVORABLE ou DEFAVORABLE
    poids: float        # intensite relative, 0 a 1


class Levier(BaseModel):
    libelle: str
    probabiliteApres: float
    gain: float


class Evaluation(BaseModel):
    probabiliteAccord: float
    niveau: str                       # FAIBLE, MODERE, FAVORABLE
    montantSuggere: Optional[float]
    partSuggeree: Optional[float]
    facteurs: List[Facteur]
    leviers: List[Levier]
    comparables: dict
    avertissement: str


# ------------------------------------------------------------- preparation

def _variables(d: Demande, montant: Optional[float] = None, jour: Optional[date] = None) -> pd.DataFrame:
    """Reconstruit exactement les variables de l'entrainement."""
    typ = d.typeBeneficiaire if d.typeBeneficiaire in TYPES else "INSTITUTION"
    montant = float(montant if montant is not None else (d.montantDemande or 0) or 1.0)

    if jour is None:
        try:
            jour = datetime.strptime(d.dateDemande, "%Y-%m-%d").date() if d.dateDemande else date.today()
        except ValueError:
            jour = date.today()

    reference = MEDIANES.get(typ) or float(np.median(list(MEDIANES.values()) or [30000.0]))
    demandes = int(d.demandesAnterieures or 0)
    accords = int(d.accordsAnterieurs or 0)

    return pd.DataFrame([{
        "type_beneficiaire": typ,
        "thematique": (d.thematique or "AUTRE").upper(),
        "ecart_aux_pairs": float(np.log(max(montant, 1.0)) - np.log(reference)),
        "mois": jour.month,
        "fin_exercice": int(jour.month >= 10),
        "candidat_recurrent": int(demandes > 0 or accords > 0),
    }])


# Bornes d'affichage. Le modele produit parfois 0,01 ou 0,99 en empilant
# plusieurs effets defavorables, mais il n'a ete valide qu'a 0,74 d'aire sous
# la courbe : annoncer « 1 % » a un instructeur lui promettrait une certitude
# que ce chiffre n'autorise pas. On borne donc l'affichage a la plage ou le
# modele a effectivement ete observe.
PLANCHER, PLAFOND = 0.05, 0.95


def _probabilite(ligne: pd.DataFrame, brute: bool = False) -> float:
    p = float(MODELE_ACCORD.predict_proba(ligne)[0, 1])
    return p if brute else float(np.clip(p, PLANCHER, PLAFOND))


def _expliquer(ligne: pd.DataFrame, nombre: int = 5) -> List[Facteur]:
    """Decompose le score en contributions, lisibles parce que le modele
    retenu est lineaire. Un gradient boosting aurait demande SHAP."""
    prep = MODELE_ACCORD.named_steps["prep"]
    final = MODELE_ACCORD.named_steps["modele"]
    if not hasattr(final, "coef_"):
        return []

    transformee = prep.transform(ligne)[0]
    contributions = transformee * final.coef_[0]
    noms = prep.get_feature_names_out()

    retenus = []
    for i in np.argsort(np.abs(contributions))[::-1]:
        if abs(contributions[i]) < 1e-6:
            continue
        prefixe, _, colonne = noms[i].partition("__")
        if prefixe == "n":                                  # variable numerique
            libelle = _libelle_numerique(colonne, float(ligne[colonne].iloc[0]))
        else:                                               # categorie active
            libelle = _libelle_categoriel(colonne)
        if libelle is None:
            continue
        retenus.append((libelle, float(contributions[i])))
        if len(retenus) == nombre:
            break

    maximum = max((abs(c) for _, c in retenus), default=1.0) or 1.0
    return [Facteur(libelle=libelle,
                    effet="FAVORABLE" if contribution > 0 else "DEFAVORABLE",
                    poids=round(abs(contribution) / maximum, 3))
            for libelle, contribution in retenus]


def _leviers(d: Demande, base: float) -> List[Levier]:
    """Ce qu'il faudrait changer pour ameliorer les chances.

    Annoncer « 31 % » n'aide personne. Dire « en deposant avant octobre,
    58 % » transforme un score en conseil d'instruction.
    """
    trouves: List[Levier] = []

    # Levier 1 : deposer hors de la fin d'exercice
    try:
        jour = datetime.strptime(d.dateDemande, "%Y-%m-%d").date() if d.dateDemande else date.today()
    except ValueError:
        jour = date.today()
    if jour.month >= 10:
        apres = _probabilite(_variables(d, jour=date(jour.year, 6, 15)))
        if apres - base > 0.03:
            trouves.append(Levier(libelle="Déposer hors de la fin d'exercice (avant octobre)",
                                  probabiliteApres=round(apres, 4), gain=round(apres - base, 4)))

    # Levier 2 : ajuster le montant demande
    if d.montantDemande:
        for part in (0.85, 0.70, 0.55):
            reduit = d.montantDemande * part
            apres = _probabilite(_variables(d, montant=reduit))
            if apres - base > 0.05:
                trouves.append(Levier(
                    libelle=f"Ramener la demande à {reduit:,.0f} TND (−{(1-part)*100:.0f} %)".replace(",", " "),
                    probabiliteApres=round(apres, 4), gain=round(apres - base, 4)))
                break

    return trouves


def _niveau(p: float) -> str:
    return "FAIBLE" if p < 0.40 else ("MODERE" if p < 0.65 else "FAVORABLE")


# -------------------------------------------------------------- endpoints

@routeur.get("/sante")
def sante():
    if not CHARGE:
        return {"modele_charge": False, "erreur": _ERREUR,
                "indication": "Lancez train_subventions.py pour produire modele/subventions.joblib"}
    return {
        "modele_charge": True,
        "accord": type(MODELE_ACCORD.named_steps["modele"]).__name__,
        "montant": type(MODELE_MONTANT.named_steps["modele"]).__name__,
        "medianes_demande": {k: round(v, 3) for k, v in MEDIANES.items()},
        "taux_accord_historique": REPERES.get("taux_par_type", {}),
    }


@routeur.post("/evaluer", response_model=Evaluation)
def evaluer(demande: Demande):
    if not CHARGE:
        raise HTTPException(503, "Modèle indisponible : lancez train_subventions.py")

    ligne = _variables(demande)
    probabilite = _probabilite(ligne)

    part = montant = None
    if demande.montantDemande:
        part = float(np.clip(MODELE_MONTANT.predict(ligne)[0], 0.20, 1.0))
        montant = round(demande.montantDemande * part, 3)

    typ = demande.typeBeneficiaire if demande.typeBeneficiaire in TYPES else "INSTITUTION"
    comparables = {
        "tauxAccordType": REPERES.get("taux_par_type", {}).get(typ),
        "tauxAccordThematique": REPERES.get("taux_par_thematique", {}).get((demande.thematique or "").upper()),
        "partMedianeType": REPERES.get("ratio_median_par_type", {}).get(typ),
        "demandeMedianeType": MEDIANES.get(typ),
    }

    return Evaluation(
        probabiliteAccord=round(probabilite, 4),
        niveau=_niveau(probabilite),
        montantSuggere=montant,
        partSuggeree=round(part, 4) if part is not None else None,
        facteurs=_expliquer(ligne),
        leviers=_leviers(demande, probabilite),
        comparables=comparables,
        avertissement=("Estimation statistique fondée sur les dossiers passés. "
                       "Elle éclaire l'instruction, elle ne la remplace pas : "
                       "la décision appartient au comité."),
    )
