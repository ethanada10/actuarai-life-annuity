# ActuarAI — Life Insurance Project (Deferred Temporary Life Annuity)

Ce projet implémente un **moteur de tarification actuarielle** (table TGF05) pour une **rente viagère temporaire différée**, génère un dataset synthétique conforme à l’énoncé, entraîne **deux modèles de Machine Learning** (Ridge + RandomForest via GridSearchCV) pour approximer les résultats actuariels, et fournit une **application Streamlit** permettant de comparer **Actuariel vs ML** sur un contrat donné.

---

## 1) Produit et conventions (rappel)

### Produit : rente viagère temporaire différée
- Souscription à l’âge **x**
- Différé **m’** années (aucun paiement pendant le différé)
- Paiement de rente **A** en **fin d’année** (annuity-immediate)
- Rente payée au maximum pendant **n** années après le différé
- Conditionné à la **survie** via la table de mortalité

### Primes
- Prime annuelle nette **P** payée en **début d’année** (annuity-due)
- Payable pendant **m** années au maximum

### Table de mortalité
- Table **TGF05** (survivants `l_x`) chargée depuis `TGF05-TGH05.xls`
- Probabilité de survie :  
  \[
  {}_k p_x = \frac{l_{x+k}}{l_x}
  \]

### Formules utilisées (implémentées)
- Valeur actuelle de la rente différée temporaire :
  \[
  {}_{m'}a_{x:\overline{n}|} = \sum_{k=m'+1}^{m'+n} v^k \, {}_k p_x
  \]
- Prime unique nette :
  \[
  \Pi^{(1)} = A \cdot {}_{m'}a_{x:\overline{n}|}
  \]
- Valeur actuelle des primes (annuity-due temporaire) :
  \[
  \ddot{a}_{x:\overline{m}|} = \sum_{k=0}^{m-1} v^k \, {}_k p_x
  \]
- Prime annuelle nette :
  \[
  P = \frac{\Pi^{(1)}}{\ddot{a}_{x:\overline{m}|}}
  \]

---

## 2) Contenu du projet

### Arborescence minimale
LIFE INSURANCE PROJECT/
├── app.py
├── main.py
├── mortality.py
├── actuarial.py
├── data_gen.py
├── ml_models.py
├── TGF05-TGH05.xls
├── requirements.txt
└── README.md

### Rôle des fichiers
- **mortality.py**  
  Charge la table TGF05 (`l_x`) et expose les fonctions de mortalité (ex: `kpx`).
- **actuarial.py**  
  Implémente les formules actuarielles : `single_premium` (Pi1) et `annual_premium` (P).
- **data_gen.py**  
  Génère un dataset synthétique (features + targets actuariels).
- **ml_models.py**  
  Entraîne 2 modèles (Ridge + RandomForest) via **GridSearchCV**, calcule MAE/RMSE/R², renvoie modèles + scores + best params.
- **main.py**  
  Script de tests étape par étape (mortalité, pricing actuariel, dataset, ML).
- **app.py**  
  Application Streamlit interactive (local) : dataset + training + prédictions + comparaison actuariel vs ML.
- **TGF05-TGH05.xls**  
  Table de mortalité fournie (TGF05).
- **requirements.txt**  
  Dépendances.

---

## 3) Paramètres du dataset (conformes à l’énoncé)

Les features sont tirées aléatoirement dans les ensembles discrets :

- `x ∈ {20, 30, 40, 50, 60}`
- `m ∈ {1, 5, 10, 20, 30, 40}`
- `mprime ∈ {0, 1, 5, 10, 20, 30, 40}`
- `n ∈ {1, 5, 10, 20, 30, 40, 50, 60}`
- `i ∈ {0, 0.005, 0.01, 0.015, 0.02, 0.025}`
- `A ∈ {50, 100, 200, 400, 800, 1000, 2000}`

Variable additionnelle :
- `generation = tariff_year - x` (par défaut `tariff_year = 2025`)

Targets (sorties) :
- `Pi1` = prime unique nette \(\Pi^{(1)}\)
- `P` = prime annuelle nette

Contrainte mortalité (table limitée à un âge max) :
- Rejet si `x + mprime + n > age_max` (souvent 121)

Taille dataset :
- `N ∈ [1, 1000]`

---

## 4) Installation (local)

### Pré-requis
- Python **3.10+** (recommandé)
- macOS / Linux / Windows (OK)

### Installation
Dans un terminal, se placer dans le dossier du projet puis :

```bash
pip install -r requirements.txt
```
## 5) Exécution des tests (console)
Pour exécuter les tests étape par étape :
python3 main.py
Tu dois voir :
Test mortalité (valeurs l_x et kpx)
Test pricing (Pi1 et P)
Test dataset (aperçu + stats descriptives)
Test ML (scores + hyperparamètres)


## 6) Application Streamlit (local)
Lancer l’app
streamlit run app.py
Fonctionnalités
Choix de N (taille du dataset) + seed
Génération dataset + training ML (GridSearchCV)
Saisie d’un contrat (x,m,mprime,n,i,A)
Calcul actuariel Pi1/P
Prédictions ML Pi1/P (Ridge + RF)
Erreurs absolues/relatives vs actuariel
Scores MAE/RMSE/R² + best hyperparams
Preview du dataset généré
⚠️ Avec GridSearchCV, l’entraînement peut prendre du temps (selon la machine et N).
Streamlit met les résultats en cache : si tu ne changes pas N ou seed, il ne retrain pas.


## 7) Métriques de performance (exemple)
Exemple obtenu sur un dataset d’entraînement N=500 (train/test split) :
Target	Model	MAE	RMSE	R²
Pi1	Ridge	5638.95	8550.53	0.6693
Pi1	RandomForest	1508.50	2983.94	0.9597
P	Ridge	2045.24	2844.28	0.0708
P	RandomForest	337.69	1094.67	0.8624
Interprétation :
RandomForest >> Ridge (non-linéarités + interactions)
P est plus difficile car P = Pi1 / ä (ratio non-linéaire)


## 8) (Option) Validation Excel
Les fichiers fournis Deferred_Life_Annuity.xlsm / jeu_de_test servent principalement à :
vérifier les conventions (primes en début d’année, rente en fin d’année, différé)
comparer les valeurs attendues sur quelques jeux de paramètres
Nous n’utilisons pas l’Excel directement dans le code, mais il peut servir de référence de validation.

## 9) Dépannage (problèmes fréquents)
“Module not found”
Assure-toi d’être dans le bon dossier avant d’exécuter :
cd "/Users/xxxxx/xxxx/Projet ActuarAI/LIFE INSURANCE PROJECT"
Lecture de TGF05-TGH05.xls
Si erreur Excel :
vérifie que TGF05-TGH05.xls est bien présent dans le dossier
vérifie que xlrd est installé (pip show xlrd)
si besoin : réinstalle les dépendances pip install -r requirements.txt
Dataset lent
La génération peut être plus lente si beaucoup de tirages sont rejetés (contrainte âge max).
Réduis N (ex : 300–500) ou augmente la puissance de la machine.


​	
 
Ethan Ada & Tom Cohen

