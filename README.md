# Machine Learning — Step by Step

A hands-on, notebook-driven machine learning learning series. Each **module** is
a self-contained Jupyter notebook that walks through one complete ML workflow —
from importing libraries and loading data all the way to training, evaluating
and cross-validating a model.

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository is a growing series of end-to-end **scikit-learn** walkthroughs.
Every module is standalone, deliberately reproducible (fixed `random_state`s) and
written for beginners who want to see a full machine-learning workflow in one
readable notebook.

| Module | Notebook | Theme | Algorithms | Dataset |
| --- | --- | --- | --- | --- |
| 1 | [`machine_learning001.ipynb`](machine_learning001.ipynb) | Linear classification | `SGDClassifier` (linear) | Iris |
| 2 | [`machine_learning002.ipynb`](machine_learning002.ipynb) | Intro to supervised & unsupervised learning | `SGDClassifier`, `KMeans`, `SGDRegressor` | Iris + California housing |
| 3 | [`machine_learning003.ipynb`](machine_learning003.ipynb) | Supervised learning — image recognition | `SVC(kernel="linear")` | Olivetti faces |

---

## Module 1 · Scikit-Learn: Linear Classification on the Iris Dataset

A complete, beginner-friendly introduction to **supervised classification** with
scikit-learn. Using the classic **Iris** dataset, it builds up a linear
classifier (`SGDClassifier`) step by step and measures its performance honestly
with both a held-out test set and 5-fold cross-validation.

### The notebook

| Item | Detail |
| --- | --- |
| File | [`machine_learning001.ipynb`](machine_learning001.ipynb) |
| Dataset | Iris (`sklearn.datasets.load_iris`) — 150 samples, 4 features, 3 classes |
| Features used | `sepal length`, `sepal width` (first two attributes, for 2-D visualisation) |
| Model | `SGDClassifier` (linear), wrapped in a `Pipeline` with `StandardScaler` |
| Test split | 25% hold-out (`random_state=33`) → 112 train / 38 test |

### What the notebook covers

1. **Setup** — import `IPython`, `scikit-learn`, `pandas`, `numpy` and `matplotlib`, and print their versions.
2. **Load data** — `datasets.load_iris()` into `x_iris` (150 × 4) and `y_iris` (150,).
3. **Split** — `train_test_split(test_size=0.25, random_state=33)`.
4. **Standardise** — `StandardScaler` fitted **on the training set only**, then applied to both sets.
5. **Visualise** — scatter plot of the standardised training data, coloured by class.
6. **Train** — `SGDClassifier` linear model; plots the "one class versus the rest" decision boundaries using `clf.coef_` and `clf.intercept_`.
7. **Predict** — inspect a single prediction with `clf.predict(scaler.transform([[4.7, 3.1]]))`.
8. **Evaluate** — accuracy on train and test, plus `classification_report` and `confusion_matrix`.
9. **Cross-validate** — 5-fold `KFold` (shuffled, `random_state=33`) over a `Pipeline`, then report mean ± standard error.

### Results

**Accuracy**

| Set | Accuracy |
| --- | --- |
| Training set (112 samples) | 0.8125 |
| Test set (38 samples) | 0.6842 |

**Classification report (test set)**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 | 8 |
| versicolor | 0.43 | 0.27 | 0.33 | 11 |
| virginica | 0.65 | 0.79 | 0.71 | 19 |
| **accuracy** | | | **0.68** | 38 |

> `setosa` separates perfectly, while `versicolor` and `virginica` overlap —
> expected, since only the first two features are used.

**Confusion matrix (test set)**

```
[[ 8  0  0]     # setosa     -> 8 correct
 [ 0  3  8]     # versicolor -> 3 correct, 8 predicted virginica
 [ 0  4 15]]    # virginica  -> 15 correct, 4 predicted versicolor
```

**5-fold cross-validation**

```
fold scores : [0.667, 0.800, 0.767, 0.867, 0.867]
mean ± SEM  : 0.793 (±0.037)
```

---

## Module 2 · A Gentle Introduction to Machine Learning with Python and Scikit-learn

A single, longer notebook that tours the three pillars of classical machine
learning with scikit-learn — **classification**, **clustering** and
**regression** — all on familiar, ready-to-use datasets.

### The notebook

| Item | Detail |
| --- | --- |
| File | [`machine_learning002.ipynb`](machine_learning002.ipynb) |
| Datasets | Iris (`load_iris`) — 150 × 4; California housing (`fetch_california_housing`) — 20,640 × 8 |
| Models | `SGDClassifier(loss="log_loss")`, `KMeans(n_clusters=3, init="k-means++")`, `SGDRegressor` |
| Test split | 25% hold-out (`random_state=33`); scale-then-model, scaler fitted on the training set only |

> **Note:** `fetch_california_housing()` downloads the dataset on first use, so an
> internet connection is needed for the regression section.

### What the notebook covers

1. **Setup** — print Python, IPython, NumPy, scikit-learn and Matplotlib versions.
2. **Load data** — Iris into `x_iris` / `y_iris`; inspect feature names, target classes and the first instance.
3. **Visualise** — 2-D scatter plots of the sepal and petal measurements, coloured by class.
4. **Classification — split & scale** — 25% hold-out (`random_state=33`); `StandardScaler` fitted on the training set only, with a mean/σ sanity check.
5. **Binary classifier** — collapse the problem to *setosa vs. rest* and train `SGDClassifier(loss="log_loss")`; plot the decision boundary.
6. **Prediction** — classify a single flower and print the decision function scores.
7. **Three-class problem** — retrain on the original three classes and draw the three "one-vs-rest" boundaries.
8. **Evaluation** — training/testing accuracy, confusion matrix and classification report.
9. **All four features** — repeat with all four attributes to show the accuracy jump.
10. **Clustering** — `KMeans` on the sepal, petal and all-four-attribute spaces; plot Voronoi regions and centroids.
11. **Regression** — standardise the California housing data, define a reusable `train_and_evaluate` helper and compare `SGDRegressor` **without** penalty and **with** `l2`.

### Results

**Classification — Iris, two features (`SGDClassifier`, `random_state=33`)**

| Set | Accuracy |
| --- | --- |
| Training set | 0.69 |
| Test set | 0.71 |

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 | 8 |
| versicolor | 0.00 | 0.00 | 0.00 | 11 |
| virginica | 0.63 | 1.00 | 0.78 | 19 |
| **accuracy** | | | **0.71** | 38 |

```
[[ 8  0  0]     # setosa     -> 8 correct
 [ 0  0 11]     # versicolor -> 0 correct, 11 predicted virginica
 [ 0  0 19]]    # virginica  -> 19 correct
```

> With only two features, `versicolor` and `virginica` collapse together — a
> clear motivation for using more features.

**Classification — Iris, all four features (`SGDClassifier`, `random_state=33`)**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 | 8 |
| versicolor | 1.00 | 0.73 | 0.84 | 11 |
| virginica | 0.86 | 1.00 | 0.93 | 19 |
| **accuracy** | | | **0.92** | 38 |

> Accuracy jumps from **0.71** to **0.92** simply by adding the petal features.

**Regression — California housing (`SGDRegressor`, 5-fold CV)**

| Penalty | Training score (R²) | 5-fold CV score |
| --- | --- | --- |
| `None` | -5817.55 | -15,302,479.59 |
| `l2` | -5722.70 | -15,093,226.69 |

> The regression section is intentionally left "raw": with default settings the
> `SGDRegressor` is unstable on unscaled targets, which makes it a good teaching
> example of why feature/target scaling and hyper-parameter tuning matter.

---

## Module 3 · Supervised Learning: Image Recognition with Support Vector Machines

The series' first step beyond tabular data. Using the **Olivetti faces** dataset —
400 grayscale 64×64 portraits of 40 different people — this notebook builds a
linear **Support Vector Classifier** and uses it for two progressively harder
problems: *who is this person?* and *is this person wearing glasses?*

### The notebook

| Item | Detail |
| --- | --- |
| File | [`machine_learning003.ipynb`](machine_learning003.ipynb) |
| Dataset | Olivetti faces (`sklearn.datasets.fetch_olivetti_faces`) — 400 images, 4096 features (64×64), 40 classes |
| Model | `SVC(kernel="linear")` — one hyperplane per class, one-vs-rest |
| Split | 25% hold-out (`random_state=0`), plus a custom 390/10 split for the "same person" test |

> **Note:** `fetch_olivetti_faces()` downloads the dataset on first use, so an
> internet connection is needed the first time the notebook runs. Please give
> credit to AT&T Laboratories Cambridge when reusing these images.

### What the notebook covers

1. **Setup** — print IPython, NumPy, scikit-learn and Matplotlib versions.
2. **Load data** — `fetch_olivetti_faces()` into `faces`; inspect `data`, `images`, `target` and the value range (0.0 – 1.0).
3. **Visualise** — a reusable `print_faces()` helper that renders a 20×20 grid of faces with the class label and index in each corner; used for the first 20 faces, then all 400.
4. **Model** — `SVC(kernel="linear")` as the hyperplane that separates one class from the rest.
5. **Split** — 25% hold-out (`random_state=0`).
6. **Cross-validate** — a reusable `evaluate_cross_validation()` helper running 5-fold shuffled `KFold` and reporting mean ± `scipy.stats.sem`.
7. **Evaluate** — a reusable `train_and_evaluate()` helper printing train/test accuracy, `classification_report` and `confusion_matrix` across all 40 classes.
8. **Task B — with or without glasses** — hand-label the image index ranges for subjects wearing glasses and rebuild a binary target with `create_target()`.
9. **Linear kernel on the binary problem** — cross-validate and evaluate the glasses classifier.
10. **Same-subject test** — hold out images 30–40 (one person, sometimes with and sometimes without glasses), train on the other 390, and check the model learned glasses rather than faces.
11. **Inspect the errors** — plot the 10 held-out faces with their predicted labels.

### Results

**Identity — 40-class face recognition (`SVC(kernel="linear")`, `random_state=0`)**

| Set | Accuracy |
| --- | --- |
| Training set (300 samples) | 1.00 |
| Test set (100 samples) | 0.99 |

```
5-fold CV scores : [0.9875 0.975  0.9875 0.95   0.975 ]
mean ± SEM       : 0.975 (±0.007)
macro / weighted : 1.00 / 0.99 f1 (0.99 accuracy)
```

> Only one person is ever confused with another; a hyperplane in 4096 dimensions
> is already enough to separate 40 identities almost perfectly.

**Glasses — binary "with / without" (`SVC(kernel="linear")`)**

| Set | Accuracy |
| --- | --- |
| Training set | 1.00 |
| Test set (100 samples) | 0.99 |

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| no glasses (0.0) | 1.00 | 0.99 | 0.99 | 67 |
| glasses (1.0) | 0.97 | 1.00 | 0.99 | 33 |
| **accuracy** | | | **0.99** | 100 |

```
5-fold CV scores : [1.0  0.95  0.983 0.983 0.933]
mean ± SEM       : 0.970 (±0.012)
confusion matrix : [[66  1]
                    [ 0 33]]
```

**Glasses — held-out subject (train on 390, test on images 30–40)**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| no glasses (0.0) | 0.83 | 1.00 | 0.91 | 5 |
| glasses (1.0) | 1.00 | 0.80 | 0.89 | 5 |
| **accuracy** | | | **0.90** | 10 |

```
confusion matrix : [[5 0]
                    [1 4]]
```

> A single image is misclassified as "no glasses" — plausibly because the
> subject's eyes are closed. The model is therefore reacting to a glasses-shaped
> feature rather than memorising faces.

---

## Project Structure

```
Machine-learning-series/
├── machine_learning001.ipynb   # Module 1 — linear classification on Iris
├── machine_learning002.ipynb   # Module 2 — classification, clustering & regression
├── machine_learning003.ipynb   # Module 3 — SVM image recognition on Olivetti faces
├── requirements.txt            # pinned dependencies
├── .gitattributes              # line-ending / diff normalisation
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13 (the notebook was last executed on **3.13.9**)
- `pip`

### Installation

```bash
git clone https://github.com/owilioduor-a11y/Machine-learning-series.git
cd Machine-learning-series

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Then open `machine_learning001.ipynb`, `machine_learning002.ipynb` or
`machine_learning003.ipynb` and **Run All** cells. The Iris dataset ships with
scikit-learn, so no download is required; the California housing dataset
(Module 2) and the Olivetti faces dataset (Module 3) download on first run.

## Key concepts covered

- Train/test splitting and using `random_state` for reproducibility
- Feature standardisation, and why the scaler is fitted on the training set only
- Linear classification: decision functions and decision boundaries
- Accuracy, precision, recall, F1-score and the confusion matrix
- Why cross-validation gives a more reliable estimate than a single split
- Reproducible pipelines with `sklearn.pipeline.Pipeline`
- Binary vs. multiclass ("one-vs-rest") classification
- Unsupervised clustering with `KMeans` (Voronoi regions and centroids)
- Regression scoring (R²) and cross-validating regressors
- Kernel methods: how a linear `SVC` draws a hyperplane between classes
- Working with high-dimensional image data (flat 4096-dim vectors) and human-constructed labels
- Designing an evaluation split that tests the hypothesis instead of the training set

## Roadmap

- [x] **Module 1** — Linear classification with scikit-learn (Iris dataset)
- [x] **Module 2** — Intro to classification, clustering & regression
- [x] **Module 3** — Supervised learning: SVM image recognition (Olivetti faces)
- [ ] Model selection & hyper-parameter tuning
- [ ] Tree-based ensembles
- [ ] Unsupervised learning (deep dive)

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

## Author

**Peter Owili**

## Acknowledgements

- The **Iris** dataset (R. A. Fisher) — bundled with scikit-learn.
- The **California housing** dataset — available via `sklearn.datasets.fetch_california_housing`.
- The **Olivetti faces** dataset — courtesy of AT&T Laboratories Cambridge, available via `sklearn.datasets.fetch_olivetti_faces`.
- The scikit-learn, NumPy, pandas, SciPy and Matplotlib documentation and communities.

---

⭐ If you find this series useful, consider giving the repository a star.
