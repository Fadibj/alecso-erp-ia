# Prediction du risque de depart des employes

Module d'intelligence artificielle de l'ERP ALECSO. Estime, pour chaque
employe, la probabilite qu'il quitte l'organisation, et explique les
facteurs qui pesent le plus dans ce score.

## Jeu de donnees

IBM HR Analytics Employee Attrition & Performance : 1 470 employes,
35 variables, 16,1 % de departs observes. Jeu public de reference en
analyse des ressources humaines.

Les donnees de l'ERP ne peuvent pas servir a l'entrainement : trois
employes ne permettent d'apprendre aucune regularite. Le modele est donc
entraine sur ce jeu externe, puis applique aux employes de l'ERP. Cette
limite est assumee et documentee.

## Installation

    pip install -r requirements.txt

## Entrainement

    python train.py

Compare quatre algorithmes par validation croisee, retient le meilleur,
choisit le seuil de decision qui maximise le F1, puis exporte :

    modele/modele_attrition.joblib   modele entraine, pretraitement compris
    modele/profil_reference.json     valeurs medianes de la population
    modele/metriques.json            performances et rapports de cotes
    figures/                         quatre figures pretes pour le rapport

## Service de prediction

    uvicorn api:app --host 0.0.0.0 --port 8000

Documentation interactive : http://localhost:8000/docs

### Points d'entree

    GET  /sante     etat du service et performances du modele
    POST /predire   score de risque pour un employe

### Exemple

    curl -X POST http://localhost:8000/predire \
      -H "Content-Type: application/json" \
      -d '{"age":27,"departement":"Informatique","salaireMensuel":2400,
           "anciennete":2,"anneesDepuisPromotion":2,
           "heuresSupplementaires":true,"satisfactionTravail":1}'

Tous les champs sont facultatifs : ceux que l'ERP ne connait pas sont
completes par la mediane de la population de reference, et seuls les
champs reellement transmis peuvent figurer dans l'explication du score.

## Resultats obtenus

| Algorithme             | ROC-AUC (validation croisee) |
|------------------------|------------------------------|
| Regression logistique  | 0,829                        |
| Gradient boosting      | 0,814                        |
| Foret aleatoire        | 0,809                        |
| Hist gradient boosting | 0,801                        |

Modele retenu : regression logistique.
Sur le jeu de test mis de cote (294 employes) : exactitude 0,83,
rappel 0,66, ROC-AUC 0,81, seuil de decision 0,574.

Le rappel prime ici sur la precision : manquer un employe sur le point de
partir coute plus cher qu'une alerte injustifiee.

## Lecture metier

Les coefficients de la regression se lisent en rapports de cotes :

    Heures supplementaires frequentes   x 1,99
    Deplacements frequents              x 1,79
    Anciennete depuis derniere promotion x 1,53
    Celibataire                         x 1,46
    Experience professionnelle totale   x 0,68  (retient)
    Aucun deplacement                   x 0,56  (retient)

Les heures supplementaires doublent le risque de depart. C'est le facteur
le plus determinant du modele, et il est actionnable.
