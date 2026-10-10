# MongoDB-Based Big Data Analytics System for Student Performance and Placement Prediction

## Abstract

This project is a runnable college-level Big Data Analytics system that stores student records in MongoDB, processes them with PySpark, produces Spark SQL analytics, trains placement-classification models, and presents the results through Streamlit.

The included dataset is **synthetic**. It contains 10,000 generated student records with realistic-looking academic, skill, experience, and placement fields. Replace `data/students.csv` with a compatible CSV when using real institutional data.

## Objectives

- Store student documents in MongoDB.
- Process and clean records with PySpark DataFrames.
- Demonstrate Spark SQL aggregation and analytics.
- Compare Logistic Regression, Decision Tree, and Random Forest classifiers.
- Persist the selected model for reuse.
- Provide an interactive academic project demonstration dashboard.

## Technologies

Python 3.14.7, MongoDB/MongoDB Atlas, PyMongo, Apache Spark, PySpark, Spark SQL, pandas, NumPy, scikit-learn, Joblib, Plotly, and Streamlit.

## Architecture

```text
CSV / student sources
        |
        v
MongoDB student_analytics.students
        |
        v
PySpark ingestion -> cleaning -> encoding -> Spark SQL analytics
        |                                      |
        v                                      v
Feature engineering -> ML training       CSV analytics outputs
        |
        v
Saved Joblib model
        |
        v
Streamlit dashboard and prediction form
```

## Project structure

```text
student-placement-bigdata/
├── config/                 Environment and path configuration
├── data/                   Synthetic generator and students.csv
├── database/               MongoDB connection and CSV import
├── spark/                  Spark session, ingestion, preprocessing, analytics
├── ml/                     Features, training, evaluation, prediction
├── dashboard/              Streamlit application
├── models/trained_model/   Saved model and metadata
├── outputs/                Cleaned data and analytics CSVs
├── scripts/                Project validation tooling
├── requirements.txt
├── .env.example
├── run_project.py
├── README.md
├── RUN_INSTRUCTIONS.md
└── FINAL_TEST_REPORT.md
```

## Dataset

`data/students.csv` currently contains 10,000 records and 20 columns:

`student_id`, `department`, `semester`, `gender`, `10th_percentage`, `12th_percentage`, `cgpa`, `attendance`, `backlogs`, `programming_score`, `aptitude_score`, `communication_score`, `projects`, `internships`, `certifications`, `hackathons`, `extracurricular_score`, `placement_training`, `placement_status`, and `salary`.

The validated distribution is 9,230 `Placed` and 770 `Not Placed` records. The data generator intentionally includes overlap and randomness; placement is not determined by one field.

## MongoDB setup

Install and start MongoDB locally, or create a MongoDB Atlas cluster. The project uses:

- Database: `student_analytics`
- Collection: `students`

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Set `MONGODB_URI` in `.env`. Passwords and connection strings are not committed; `.env` is ignored by `.gitignore`.

Generate and import the sample dataset:

```powershell
.\.venv\Scripts\python.exe run_project.py --rows 10000 --import-to-mongodb
```

The import utility is also available directly:

```powershell
.\.venv\Scripts\python.exe -m database.import_data --csv data\students.csv
```

## Installation

Use Python 3.10+ (the validated environment uses Python 3.14.7):

```powershell
cd D:\BDA\student-placement-bigdata
C:\Users\vinay\AppData\Local\Programs\Python\Python314\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

PySpark requires Java. The validated Windows setup uses:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

## Running each phase

### Phase 1: generate data

```powershell
python run_project.py --rows 10000
```

### Phase 2: Spark processing and analytics

```powershell
python run_project.py --phase2
```

To load from MongoDB instead of CSV:

```powershell
python run_project.py --phase2 --from-mongo
```

Outputs are written to `outputs/`, including cleaned/transformed records, department summaries, placement comparisons, and salary analytics.

### Phase 3: train and save models

```powershell
python run_project.py --phase3
```

The selected model and metadata are saved under `models/trained_model/`.

### Phase 4: launch Streamlit

```powershell
python -m streamlit run dashboard\app.py
```

Open `http://localhost:8501`.

The dashboard loads the project CSV and renders its dataset analytics separately
from the machine-learning stack. If a Windows Application Control policy blocks
SciPy's native extension, the dataset pages remain available and the app displays
a warning; prediction and uploaded-data model training are disabled rather than
returning placeholder predictions or metrics. Resolve that dependency issue
through the device administrator to re-enable ML features.

### Validate the project

Run the repeatable Windows validation script from PowerShell:

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
.\scripts\validate_project.ps1
```

It uses the project `.venv` when present and reports `PASS`, `FAIL`, `BLOCKED`,
or `SKIPPED` per check. It validates paths and dataset schema, checks MongoDB,
runs a small Spark DataFrame operation with the project interpreter configured
as the worker, checks SciPy/scikit-learn and the saved prediction API when
available, tests the rendered Streamlit pages and CSV upload, and starts a
temporary local Streamlit server for an HTTP smoke check. A missing test suite
is reported as `SKIPPED`; security-policy blocks are reported as `BLOCKED`.
Required failures and blocks return a nonzero exit code. The script does not
display MongoDB credentials, retrain/overwrite the model, modify security
settings, or delete project files.

### Dynamic dataset upload

The dashboard also provides an **Upload New Dataset** workspace for CSV files. It is
separate from the preloaded project dataset and can be reset with **Reset Dataset**.
For each upload, the dashboard reports file size, row/column counts, data types,
missing values, duplicate rows, descriptive statistics, available numeric
distributions, department/placement distributions, correlations, and placement
comparisons for columns that are actually present. Column aliases are detected for
common student fields, and missing expected fields are reported clearly.

Uploaded data is not passed to the saved project model automatically. If and only if
the exact Phase 3 feature schema and `placement_status` target are present, the
dashboard offers optional training and comparison of Logistic Regression, Decision
Tree, and Random Forest models on the uploaded data. CSV is currently supported;
Excel and JSON are intentionally reserved for a later extension.

## Spark configuration

Spark runs locally with `local[*]`, eight shuffle partitions, and eight default parallelism units. CSV loading is used by default for reproducible local Windows execution. MongoDB ingestion is available through PyMongo and conversion to a Spark DataFrame.

## Machine-learning methodology

The target is binary `placement_status` (`Placed` or `Not Placed`). The pipeline:

1. Selects academic, attendance, skill, experience, department, gender, semester, and training features.
2. Excludes target and salary to avoid leakage.
3. Imputes numeric values and scales them.
4. One-hot encodes department and gender.
5. Performs a fixed stratified 80/20 train/test split.
6. Trains Logistic Regression, Decision Tree, and Random Forest.
7. Reports accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
8. Saves the best model as `placement_model.joblib`.

## Validated model results

The validated 10,000-record run selected Random Forest:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 52.35% | 91.61% | 53.25% | 67.35% | 0.4585 |
| Decision Tree | 19.75% | 92.88% | 14.14% | 24.54% | 0.5038 |
| Random Forest | **89.30%** | **92.15%** | **96.64%** | **94.34%** | **0.5080** |

The Random Forest confusion matrix is `[[2, 152], [62, 1784]]` with label order `Not Placed, Placed`.

The ROC-AUC of approximately 0.5080 is documented rather than hidden. It is a characteristic of the current synthetic generator: placement probabilities are heavily saturated, so threshold-based metrics are strong while ranking quality is close to random. This result should not be interpreted as real-world model quality.

## Prediction workflow

The dashboard collects all features required by the saved model, calls `ml.predict.predict_placement`, validates the saved class order (`Not Placed`, `Placed`), and displays status plus the probability for `Placed`. Every prediction page includes a notice that the result is a model estimate, not a guaranteed placement result.

## Expected outputs

- `outputs/cleaned_students.csv`
- `outputs/transformed_students.csv`
- `outputs/department_counts.csv`
- `outputs/department_placement_rate.csv`
- `outputs/cgpa_by_placement.csv`
- `outputs/internships_by_placement.csv`
- `outputs/projects_by_placement.csv`
- `outputs/certifications_by_placement.csv`
- `outputs/backlogs_by_placement.csv`
- `models/trained_model/placement_model.joblib`
- `models/trained_model/training_metadata.json`

## Troubleshooting

- **Java gateway error:** set `JAVA_HOME` to a Java 17 installation before running PySpark.
- **MongoDB connection failure:** start MongoDB or replace `MONGODB_URI` with a reachable Atlas URI.
- **Missing model:** run `python run_project.py --phase3`.
- **Missing dataset:** run `python run_project.py --rows 10000`.
- **Dashboard cannot load the dataset:** run Phase 1 from the project root.
- **Prediction or model training is unavailable:** verify the ML dependencies can
  be imported. If Windows Application Control blocked a SciPy extension, contact
  the device administrator; dataset analytics can still run without ML.
- **MongoDB import failure:** verify the URI, database permissions, and that the CSV is non-empty.
- **Upload validation warning:** inspect the reported missing columns; arbitrary schemas
  cannot use the saved project model and must be trained only when the exact ML schema
  is available.

## Limitations

- The current dataset is synthetic and not representative of a particular institution.
- The class distribution is imbalanced.
- ROC-AUC is weak for the current generated data.
- MongoDB must be supplied separately; the repository does not bundle a database server.
- Predictions should not be used for admission, placement, or employment decisions.

## Future enhancements

- Replace synthetic data with governed institutional data.
- Add calibrated probabilities and cross-validation.
- Add model drift and fairness monitoring.
- Add authentication and role-based dashboard access.
- Add native MongoDB Spark Connector ingestion for distributed deployments.
