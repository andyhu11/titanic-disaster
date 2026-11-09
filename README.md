# Titanic Disaster: Survival Prediction in Docker (Python & R)

This repository predicts which passengers survived the sinking of the Titanic
(April 15, 1912) using **logistic regression**, implemented twice:

| Container | Language | Location | Model |
|---|---|---|---|
| `titanic-python` | Python 3.11 (pandas, scikit-learn) | `src/python_app/` | scikit-learn `LogisticRegression` |
| `titanic-r` | R 4.4.1 (readr, dplyr) | `src/r_app/` | base R `glm(family = binomial)` |

Each container loads the data, prints every cleaning and feature-engineering step,
trains the model, reports accuracy, and predicts on the test set. All output goes to the terminal.

> The dataset is **not** included in this repository. Follow Step 2 to download it.

---

## Repository structure

```
titanic-disaster/
├── README.md
├── .gitignore               # excludes the data, OS files, virtual environments
├── .github/CODEOWNERS
└── src/
    ├── data/                # put train.csv and test.csv here (git-ignored)
    ├── python_app/
    │   ├── Dockerfile
    │   ├── requirements.txt # pinned package versions
    │   └── titanic.py
    └── r_app/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic.R
```

---

## Prerequisites

1. **Git**: <https://git-scm.com/downloads>
2. **Docker Desktop** (or Docker Engine on Linux): <https://www.docker.com/products/docker-desktop/>.
   Make sure Docker is **running** before you continue. `docker info` should print without errors.
3. A free **Kaggle account**, to download the data: <https://www.kaggle.com>

You do **not** need Python or R installed locally. Everything runs inside Docker.

---

## Step 1: Clone the repository

```bash
git clone https://github.com/andyhu11/titanic-disaster.git
cd titanic-disaster
```

All commands below are run from this `titanic-disaster` folder (the repository root).

## Step 2: Download the data

**Option A: Browser**

1. Go to <https://www.kaggle.com/competitions/titanic/data> and sign in.
2. Click **"Join Competition"** / accept the rules. Kaggle requires this before downloading.
3. Click **"Download All"** to get `titanic.zip`.
4. Unzip it and move **`train.csv`** and **`test.csv`** into `src/data/`.

**Option B: Kaggle CLI** (requires an API token in `~/.kaggle/kaggle.json`, and the rules must already be accepted on the website)

```bash
pip install kaggle==1.7.4.5
kaggle competitions download -c titanic -p src/data
```

Then unzip it:

- macOS / Linux / Git Bash: `unzip src/data/titanic.zip -d src/data`
- Windows PowerShell: `Expand-Archive src/data/titanic.zip -DestinationPath src/data`

When you are done, the folder must look like this:

```
src/data/
├── train.csv   (891 rows)
└── test.csv    (418 rows)
```

## Step 3: Run the Python container

Build the image:

```bash
docker build -t titanic-python src/python_app
```

Run it, mounting the data folder into the container:

**macOS / Linux / Git Bash**
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

**Windows PowerShell**
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

**Windows Command Prompt (cmd.exe)**
```bat
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

## Step 4: Run the R container

Build the image (the first build downloads R packages and takes a few minutes).
Package versions are reproducible: the `rocker/r-ver:4.4.1` base image installs packages from a
CRAN snapshot dated 2024-10-30, which resolves to readr 2.1.5 and dplyr 1.1.4.

```bash
docker build -t titanic-r src/r_app
```

Run it:

**macOS / Linux / Git Bash**
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

**Windows PowerShell**
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

**Windows Command Prompt (cmd.exe)**
```bat
docker run --rm -v "%cd%/src/data:/app/data" titanic-r
```

---

## What the output shows

Both scripts print the same numbered sections:

| Section | Content |
|---|---|
| STEP 0 | Language and package versions |
| STEP 1 | `train.csv` shape, first rows, missing values |
| STEP 2 | Interesting notes: survival rate overall, by sex, by class, children vs adults |
| STEP 3 | Every data change, tagged `[ADD]`, `[FILL]`, `[DROP]`, `[ENCODE]`, `[CHECK]` |
| STEP 4 | Train/validation split and the fitted logistic regression coefficients |
| STEP 5 | Accuracy on the training set (and on a held-out validation set of approximately 20%) |
| STEP 6 | `test.csv` cleaned with the **same** rules learned from `train.csv`, then predicted |
| STEP 7 | Test-set accuracy (see note below) |

Predictions are also saved in Kaggle submission format to
`src/data/predictions_python.csv` and `src/data/predictions_r.csv`.

### Data processing

| Change | Rule |
|---|---|
| `Title` (new) | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare) |
| `Age` missing | Filled with the median age of the passenger's Title |
| `Fare` missing | Filled with the median fare of the passenger's class (one missing value in `test.csv`) |
| `Embarked` missing | Filled with the most common port |
| `FamilySize` (new) | `SibSp + Parch + 1` |
| `IsAlone` (new) | 1 if `FamilySize == 1` |
| `HasCabin` (new) | 1 if `Cabin` is recorded (Cabin itself is ~77% missing) |
| Dropped | `PassengerId`, `Name`, `Ticket`, `Cabin` |

All fill values are learned from `train.csv` only and reused on `test.csv`, so there is no data leakage.

**Model differences:** the Python model uses all features above, with Sex, Embarked
and Title one-hot encoded and standardized. The R model uses
`Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, HasCabin`. It drops `SibSp`
and `Parch` because they are perfectly collinear with `FamilySize`, which unregularized `glm` cannot handle.

### Note on test-set accuracy

Kaggle's `test.csv` **does not contain the `Survived` column**. The true labels are
hidden for the competition. Test accuracy therefore cannot be computed locally. The scripts:

1. report the accuracy on a **stratified hold-out of approximately 20%** of `train.csv` as an estimate of test performance, and
2. save the test predictions so they can be submitted on Kaggle to get the official score.

If a labeled `test.csv` is provided instead, the scripts detect the `Survived`
column and print the real test accuracy automatically.

## Results

Results from running both containers on the official Kaggle data. The test-set accuracy comes from
submitting `predictions_python.csv` and `predictions_r.csv` to the
[Titanic competition](https://www.kaggle.com/competitions/titanic) (submitted 2026-11-08).

| Metric | Python (scikit-learn) | R (glm) |
|---|---|---|
| Training accuracy (100% of `train.csv`) | 0.8350 | 0.8047 |
| Validation accuracy (stratified hold-out of approximately 20% of `train.csv`) | 0.8436 | 0.8371 |
| **Test accuracy: Kaggle public leaderboard (`test.csv`)** | **0.77033** | **0.75837** |

The Kaggle leaderboard for this competition is calculated on all of `test.csv` (418 passengers).
The test score is lower than the validation accuracy. This is expected because the validation
set is small (approximately 20% of `train.csv`: 179 rows in Python, 178 in R because of per-class
rounding) and the models are simple linear ones.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `[ERROR] /app/data/train.csv not found` | The CSVs are not in `src/data/`, or you ran `docker run` from a folder other than the repository root. |
| `Cannot connect to the Docker daemon` / `failed to connect to the docker API` | Start Docker Desktop and wait until it says it is running. |
| Volume mount fails on Windows | Use the PowerShell or cmd.exe command above that matches your terminal. In Docker Desktop, make sure the drive is shared (Settings → Resources → File sharing). |
| Rebuilding after editing code | Run the `docker build` command again before `docker run`. |
