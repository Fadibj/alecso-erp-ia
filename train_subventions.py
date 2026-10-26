"""
Aide a la decision sur les demandes de subvention — ALECSO ERP
===============================================================

Deux modeles complementaires, entraines sur l'historique des dossiers deja
instruits :

  1. CLASSIFICATION — probabilite que le dossier soit accorde
  2. REGRESSION     — part du montant demande susceptible d'etre accordee

Le script compare plusieurs algorithmes, retient celui qui gagne en
validation croisee, puis l'evalue sur les exercices les plus recents,
mis de cote et jamais vus pendant l'apprentissage.

    python train_subventions.py
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

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (brier_score_loss, mean_absolute_error, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ALEA = 42
RACINE = Path(__file__).resolve().parent
DONNEES = RACINE / "data" / "subventions.csv"
MODELE = RACINE / "modele"
FIGURES = RACINE / "figures"
MODELE.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Les exercices mis de cote : le modele ne les voit jamais pendant
# l'apprentissage. Un decoupage temporel, et non aleatoire : en production
# le modele predira toujours l'avenir a partir du passe, jamais l'inverse.
ANNEE_TEST = 2024

ACCORDES = {"APPROVED", "PAID"}
DECIDES = ACCORDES | {"REJECTED"}

VERT, OR, ROUGE, BLEU, GRIS = "#2c6b5a", "#c8a84b", "#c62828", "#1565c0", "#8a9a94"


# ---------------------------------------------------------------- donnees

def charger() -> pd.DataFrame:
    if not DONNEES.exists():
        raise SystemExit(
            f"Fichier introuvable : {DONNEES}\n"
            "Exportez la table subventions depuis phpMyAdmin au format CSV."
        )
    d = pd.read_csv(DONNEES)
    for colonne in ("request_date", "decision_date", "payment_date"):
        d[colonne] = pd.to_datetime(d[colonne], errors="coerce")
    d["requested_amount"] = pd.to_numeric(d["requested_amount"], errors="coerce")
    d["granted_amount"] = pd.to_numeric(d["granted_amount"], errors="coerce")
    d["sector"] = d["sector"].fillna("AUTRE")
    d["country"] = d["country"].fillna("INCONNU")
    return d.dropna(subset=["request_date", "requested_amount", "beneficiary_type"])


def antecedents(d: pd.DataFrame) -> pd.DataFrame:
    """Compte, pour chaque dossier, ce que le beneficiaire avait obtenu AVANT.

    Le piege serait de compter toutes ses subventions, y compris posterieures :
    le modele apprendrait alors a partir d'une information que l'instructeur
    n'a pas au moment de decider. On ne retient donc que les dossiers dont la
    decision est anterieure a la date de la demande examinee.
    """
    d = d.sort_values("request_date").reset_index(drop=True)
    anterieurs_demandes, anterieurs_accordes = [], []

    for i, ligne in d.iterrows():
        passe = d.iloc[:i]
        meme = passe[passe["beneficiary_name"] == ligne["beneficiary_name"]]
        anterieurs_demandes.append(len(meme))
        decides = meme[meme["decision_date"].notna()
                       & (meme["decision_date"] < ligne["request_date"])]
        anterieurs_accordes.append(int(decides["status"].isin(ACCORDES).sum()))

    d["demandes_anterieures"] = anterieurs_demandes
    d["accords_anterieurs"] = anterieurs_accordes
    return d


def construire_variables(d: pd.DataFrame, medianes: dict) -> pd.DataFrame:
    """Variables disponibles au moment de la decision, et elles seules.

    Tout ce qui decoule de la decision — montant accorde, date de decision,
    motif — est exclu : l'inclure donnerait un modele parfait et inutile.
    """
    v = pd.DataFrame(index=d.index)

    v["type_beneficiaire"] = d["beneficiary_type"]
    v["thematique"] = d["sector"]

    # Le pays est volontairement absent des variables, pour deux raisons.
    #
    # Statistique : 22 pays pour quelques centaines de dossiers, soit une
    # douzaine de cas chacun. C'est assez pour que le hasard dessine des
    # motifs, pas assez pour qu'ils soient reels — un essai prealable
    # placait quatre pays parmi les facteurs les plus determinants, devant
    # le type de beneficiaire.
    #
    # Deontologique : une organisation qui sert vingt-deux Etats membres a
    # egalite ne peut pas deployer un outil qui avantage structurellement
    # certains d'entre eux. Meme averee, la correlation ne serait pas
    # utilisable.

    # Le montant en valeur absolue n'est pas retenu : il est colineaire a
    # l'ecart aux pairs (l'un se deduit de l'autre et de la mediane du type),
    # et la regression repartit alors l'effet arbitrairement entre les deux,
    # au point de leur donner des signes opposes. Le retirer ne coute rien en
    # performance et rend le graphique des facteurs lisible.

    # La variable decisive : demander beaucoup n'est pas penalisant en soi,
    # demander plus que ses pairs l'est. Un Etat membre qui demande 60 000
    # est dans la norme, une association qui demande autant ne l'est pas.
    reference = d["beneficiary_type"].map(medianes).fillna(np.median(list(medianes.values())))
    v["ecart_aux_pairs"] = np.log(d["requested_amount"]) - np.log(reference)

    v["mois"] = d["request_date"].dt.month
    v["fin_exercice"] = (d["request_date"].dt.month >= 10).astype(int)

    # L'anteriorite decomposee en deux faits disjoints plutot qu'en trois
    # mesures qui se recouvrent. Compter a la fois les demandes, les accords
    # et une indicatrice « deja finance » les rend colineaires : la
    # regression leur donne alors des signes opposes, et un dossier juge
    # favorable affichait « subventions deja obtenues : defavorable ».
    #
    # Une seule variable d'anteriorite : le beneficiaire s'etait-il deja
    # presente ? Trois encodages ont ete compares. Compter en plus les
    # subventions deja obtenues ne gagne que deux millieme d'AUC, mais rend
    # les deux variables colineaires : la regression leur donne alors des
    # signes opposes et l'interface affiche « deja finance deux fois :
    # defavorable » sur un dossier juge a 89 %. Une explication fausse coute
    # plus cher qu'un millieme de performance.
    v["candidat_recurrent"] = (d["demandes_anterieures"] > 0).astype(int)

    return v


CATEGORIELLES = ["type_beneficiaire", "thematique"]
NUMERIQUES = ["ecart_aux_pairs", "mois", "fin_exercice", "candidat_recurrent"]


def tube(modele, normaliser: bool) -> Pipeline:
    etapes = [("c", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), CATEGORIELLES)]
    etapes.append(("n", StandardScaler() if normaliser else "passthrough", NUMERIQUES))
    return Pipeline([("prep", ColumnTransformer(etapes)), ("modele", modele)])


def retenir(scores: dict) -> str:
    """Regle de l'ecart-type : parmi les modeles a moins d'un ecart-type du
    meilleur, on garde le plus simple.

    Choisir systematiquement la moyenne la plus haute revient a choisir du
    bruit quand les candidats sont au coude a coude : un modele qui gagne
    0,004 avec un ecart-type deux fois plus grand n'est pas meilleur, il est
    moins stable. Les candidats sont declares du plus simple au plus complexe.
    """
    meilleur = max(scores, key=lambda k: scores[k][0])
    seuil = scores[meilleur][0] - scores[meilleur][1]
    for nom in scores:                      # ordre de declaration = complexite croissante
        if scores[nom][0] >= seuil:
            return nom
    return meilleur


# ------------------------------------------------- modele 1 : accord ou non

def candidats_accord():
    return {
        "Régression logistique": tube(
            LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"), True),
        "Forêt aléatoire": tube(
            RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=4,
                                   class_weight="balanced", random_state=ALEA, n_jobs=-1), False),
        "Gradient boosting": tube(
            GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                       max_depth=3, random_state=ALEA), False),
        "Hist gradient boosting": tube(
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_depth=4, random_state=ALEA), False),
    }


def entrainer_accord(Xa, ya, Xt, yt):
    print("\n" + "=" * 66)
    print("  MODÈLE 1 — Probabilité d'accord")
    print("=" * 66)
    print(f"  Apprentissage : {len(Xa)} dossiers  ({ya.mean()*100:.1f} % accordés)")
    print(f"  Test          : {len(Xt)} dossiers  ({yt.mean()*100:.1f} % accordés)\n")

    blocs = StratifiedKFold(5, shuffle=True, random_state=ALEA)
    scores = {}
    for nom, pipeline in candidats_accord().items():
        s = cross_val_score(pipeline, Xa, ya, cv=blocs, scoring="roc_auc", n_jobs=-1)
        scores[nom] = (s.mean(), s.std())
        print(f"    {nom:<24} ROC-AUC {s.mean():.4f}  ± {s.std():.4f}")

    meilleur = retenir(scores)
    sommet = max(scores, key=lambda k: scores[k][0])
    if meilleur != sommet:
        print(f"\n  Meilleure moyenne : {sommet} ({scores[sommet][0]:.4f})")
        print(f"  Retenu            : {meilleur} ({scores[meilleur][0]:.4f}) — plus simple, à moins d'un écart-type")
    else:
        print(f"\n  Retenu : {meilleur}")

    modele = candidats_accord()[meilleur]
    modele.fit(Xa, ya)

    p = modele.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, p)
    brier = brier_score_loss(yt, p)

    print(f"\n  Sur les exercices {ANNEE_TEST}+ jamais vus :")
    print(f"    ROC-AUC ........ {auc:.4f}   (capacité à ordonner les dossiers)")
    print(f"    Score de Brier . {brier:.4f}   (justesse de la probabilité elle-même)")
    print(f"    Référence ...... {np.mean((yt.mean()-yt)**2):.4f}   (en prédisant toujours le taux moyen)")

    return modele, meilleur, scores, p, auc, brier


# ------------------------------------------------ modele 2 : montant accorde

def candidats_montant():
    return {
        "Régression ridge": tube(RidgeCV(alphas=np.logspace(-2, 2, 20)), True),
        "Forêt aléatoire": tube(
            RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=4,
                                  random_state=ALEA, n_jobs=-1), False),
        "Gradient boosting": tube(
            GradientBoostingRegressor(n_estimators=250, learning_rate=0.05,
                                      max_depth=3, random_state=ALEA), False),
    }


def entrainer_montant(Xa, ra, Xt, rt):
    print("\n" + "=" * 66)
    print("  MODÈLE 2 — Part du montant demandé qui serait accordée")
    print("=" * 66)
    print(f"  Apprentissage : {len(Xa)} dossiers accordés")
    print(f"  Part médiane  : {np.median(ra):.1%} du montant demandé\n")

    scores = {}
    for nom, pipeline in candidats_montant().items():
        s = cross_val_score(pipeline, Xa, ra, cv=5, scoring="r2", n_jobs=-1)
        scores[nom] = (s.mean(), s.std())
        print(f"    {nom:<24} R²      {s.mean():.4f}  ± {s.std():.4f}")

    meilleur = retenir(scores)
    sommet = max(scores, key=lambda k: scores[k][0])
    if meilleur != sommet:
        print(f"\n  Meilleure moyenne : {sommet} ({scores[sommet][0]:.4f})")
        print(f"  Retenu            : {meilleur} ({scores[meilleur][0]:.4f}) — plus simple, à moins d'un écart-type")
    else:
        print(f"\n  Retenu : {meilleur}")

    modele = candidats_montant()[meilleur]
    modele.fit(Xa, ra)

    pred = np.clip(modele.predict(Xt), 0.2, 1.0)
    print(f"\n  Sur les exercices {ANNEE_TEST}+ :")
    print(f"    R² ............. {r2_score(rt, pred):.4f}")
    print(f"    Erreur moyenne . {mean_absolute_error(rt, pred)*100:.1f} points de pourcentage")

    return modele, meilleur, scores, pred


# ----------------------------------------------------------------- figures

def figure_comparaison(scores, titre, unite, fichier):
    noms = list(scores)
    moyennes = [scores[n][0] for n in noms]
    ecarts = [scores[n][1] for n in noms]
    couleurs = [VERT if m == max(moyennes) else GRIS for m in moyennes]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    barres = ax.barh(noms, moyennes, xerr=ecarts, color=couleurs,
                     error_kw=dict(ecolor="#55615d", capsize=4, lw=1.2))
    for barre, m in zip(barres, moyennes):
        ax.text(barre.get_width() + 0.012, barre.get_y() + barre.get_height() / 2,
                f"{m:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel(unite)
    ax.set_title(titre, fontweight="bold", color="#1e4d40")
    ax.set_xlim(0, max(moyennes) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / fichier, dpi=150)
    plt.close(fig)


def figure_roc(yt, p, auc):
    fpr, tpr, _ = roc_curve(yt, p)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(fpr, tpr, color=VERT, lw=2.4, label=f"Modèle — AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1.2, label="Hasard — AUC 0.500")
    ax.fill_between(fpr, tpr, alpha=.12, color=VERT)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Capacité à distinguer les dossiers accordés", fontweight="bold", color="#1e4d40")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_accord.png", dpi=150)
    plt.close(fig)


def figure_calibration(yt, p, brier):
    """Une probabilite affichee a un instructeur doit vouloir dire ce qu'elle dit :
    sur les dossiers annonces a 70 %, environ 70 % doivent etre accordes."""
    reel, prevu = calibration_curve(yt, p, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1.2, label="Calibration parfaite")
    ax.plot(prevu, reel, "o-", color=OR, lw=2.4, ms=8, label=f"Modèle — Brier {brier:.3f}")
    ax.set_xlabel("Probabilité annoncée par le modèle")
    ax.set_ylabel("Proportion réellement accordée")
    ax.set_title("La probabilité annoncée est-elle fiable ?", fontweight="bold", color="#1e4d40")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close(fig)


def figure_facteurs(modele, nom):
    """Lecture metier : chaque coefficient devient un rapport de cotes."""
    if "ogistique" not in nom:
        return
    noms = modele.named_steps["prep"].get_feature_names_out()
    coefs = modele.named_steps["modele"].coef_[0]
    ordre = np.argsort(np.abs(coefs))[-12:]
    noms, coefs = noms[ordre], coefs[ordre]
    cotes = np.exp(coefs)

    etiquettes = [n.split("__")[-1].replace("type_beneficiaire_", "")
                   .replace("thematique_", "").replace("_", " ")
                  for n in noms]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(etiquettes, cotes, color=[VERT if c > 1 else ROUGE for c in cotes])
    ax.axvline(1, color="#333", lw=1.1)
    for i, c in enumerate(cotes):
        ax.text(c + .03, i, f"×{c:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Effet sur les chances d'accord (× 1 = sans effet)")
    ax.set_title("Ce qui pèse sur une décision", fontweight="bold", color="#1e4d40")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "facteurs_accord.png", dpi=150)
    plt.close(fig)


def figure_montant(rt, pred):
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.scatter(rt * 100, pred * 100, color=VERT, alpha=.65, s=46, edgecolor="white")
    bornes = [min(rt.min(), pred.min()) * 100 - 4, 104]
    ax.plot(bornes, bornes, "--", color=GRIS, lw=1.3, label="Prédiction parfaite")
    ax.set_xlabel("Part réellement accordée (%)")
    ax.set_ylabel("Part prédite (%)")
    ax.set_title("Montant accordé — prédiction contre réalité", fontweight="bold", color="#1e4d40")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "montant_accorde.png", dpi=150)
    plt.close(fig)


def figure_saisonnalite(d):
    decides = d[d["status"].isin(DECIDES)].copy()
    decides["mois"] = decides["request_date"].dt.month
    taux = decides.groupby("mois")["status"].apply(lambda s: s.isin(ACCORDES).mean() * 100)
    moyenne = decides["status"].isin(ACCORDES).mean() * 100

    fig, ax = plt.subplots(figsize=(9, 4.4))
    couleurs = [ROUGE if m >= 10 else VERT for m in taux.index]
    ax.bar(taux.index, taux.values, color=couleurs)
    ax.axhline(moyenne, ls="--", color="#333", lw=1.2, label=f"Moyenne {moyenne:.0f} %")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["jan", "fév", "mar", "avr", "mai", "juin",
                        "juil", "août", "sep", "oct", "nov", "déc"])
    ax.set_ylabel("Taux d'accord (%)")
    ax.set_title("Déposer en fin d'exercice coûte cher au demandeur",
                 fontweight="bold", color="#1e4d40")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "saisonnalite.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    d = antecedents(charger())
    decides = d[d["status"].isin(DECIDES)].copy()
    decides["accorde"] = decides["status"].isin(ACCORDES).astype(int)

    print(f"\n{len(d)} dossiers lus, dont {len(decides)} déjà instruits.")
    print(f"Période : {d['request_date'].min():%Y-%m-%d} → {d['request_date'].max():%Y-%m-%d}")

    appr = decides[decides["request_date"].dt.year < ANNEE_TEST]
    test = decides[decides["request_date"].dt.year >= ANNEE_TEST]
    if len(test) < 20:
        raise SystemExit("Pas assez de dossiers récents pour une évaluation honnête.")

    # Les medianes de reference se calculent sur l'apprentissage seul : les
    # calculer sur tout le jeu ferait fuiter le futur dans le passe.
    medianes = appr.groupby("beneficiary_type")["requested_amount"].median().to_dict()

    Xa, Xt = construire_variables(appr, medianes), construire_variables(test, medianes)
    ya, yt = appr["accorde"].values, test["accorde"].values

    m_accord, nom_accord, sc_accord, p, auc, brier = entrainer_accord(Xa, ya, Xt, yt)

    # --- Modele 2, sur les seuls dossiers accordes ---
    acc_a = appr[appr["granted_amount"].notna()]
    acc_t = test[test["granted_amount"].notna()]
    ra = (acc_a["granted_amount"] / acc_a["requested_amount"]).values
    rt = (acc_t["granted_amount"] / acc_t["requested_amount"]).values

    m_montant, nom_montant, sc_montant, pred = entrainer_montant(
        construire_variables(acc_a, medianes), ra,
        construire_variables(acc_t, medianes), rt)

    # --- Figures ---
    figure_comparaison(sc_accord, "Probabilité d'accord — comparaison",
                       "ROC-AUC (validation croisée à 5 blocs)", "comparaison_accord.png")
    figure_comparaison(sc_montant, "Montant accordé — comparaison",
                       "R² (validation croisée à 5 blocs)", "comparaison_montant.png")
    figure_roc(yt, p, auc)
    figure_calibration(yt, p, brier)
    figure_facteurs(m_accord, nom_accord)
    figure_montant(rt, pred)
    figure_saisonnalite(d)

    # --- Reperes historiques, pour situer un dossier parmi ses comparables ---
    reperes = {
        "taux_par_type": decides.groupby("beneficiary_type")["accorde"].mean().round(4).to_dict(),
        "taux_par_thematique": decides.groupby("sector")["accorde"].mean().round(4).to_dict(),
        "ratio_median_par_type": (decides[decides["granted_amount"].notna()]
                                  .assign(r=lambda x: x["granted_amount"] / x["requested_amount"])
                                  .groupby("beneficiary_type")["r"].median().round(4).to_dict()),
        "medianes_demande": {k: round(v, 3) for k, v in medianes.items()},
    }

    joblib.dump({
        "modele_accord": m_accord,
        "modele_montant": m_montant,
        "medianes_demande": medianes,
        "reperes": reperes,
        "categorielles": CATEGORIELLES,
        "numeriques": NUMERIQUES,
    }, MODELE / "subventions.joblib")

    metriques = {
        "n_dossiers": int(len(d)),
        "n_instruits": int(len(decides)),
        "n_apprentissage": int(len(appr)),
        "n_test": int(len(test)),
        "annee_test": ANNEE_TEST,
        "accord": {
            "algorithme": nom_accord,
            "validation_croisee": {k: round(v[0], 4) for k, v in sc_accord.items()},
            "roc_auc_test": round(float(auc), 4),
            "brier_test": round(float(brier), 4),
            "taux_accord_historique": round(float(decides["accorde"].mean()), 4),
        },
        "montant": {
            "algorithme": nom_montant,
            "validation_croisee": {k: round(v[0], 4) for k, v in sc_montant.items()},
            "r2_test": round(float(r2_score(rt, pred)), 4),
            "erreur_absolue_test": round(float(mean_absolute_error(rt, pred)), 4),
        },
        "reperes": reperes,
    }
    (MODELE / "metriques.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 66)
    print("  Enregistré")
    print("=" * 66)
    print(f"    {MODELE / 'subventions.joblib'}")
    print(f"    {MODELE / 'metriques.json'}")
    for f in sorted(FIGURES.glob("*.png")):
        print(f"    {f}")
    print()


if __name__ == "__main__":
    main()
