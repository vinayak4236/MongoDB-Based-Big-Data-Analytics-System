# Windows Dashboard Run Instructions

## Project files and sample upload datasets

Two distinct, synthetic, ML-compatible CSV files are included for testing the
dashboard's **Dataset Upload** workflow:

- `data\upload_samples\student_upload_sample_a.csv` — 1,000 records
- `data\upload_samples\student_upload_sample_b.csv` — 1,000 records

Both use the same 20-column schema as the project dataset, include `Placed` and
`Not Placed` target values, and are separate from `data\students.csv`. The
samples are generated demonstration data; do not treat their metrics as
real-world evidence.

## Manual Windows setup and run

### 1. Open PowerShell in the project folder

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
```

The project virtual environment is `.venv`. Use its Python executable for the
commands below; activating the environment is optional.

If you do not have the virtual environment yet, create it first:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { python -m venv .venv }
```

### 2. Install or verify Python dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure MongoDB

For a local MongoDB Community Server, start the MongoDB service and confirm
`.env` has the appropriate local URI. To create `.env` from the template only if
it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The defaults target `mongodb://localhost:27017/`, database
`student_analytics`, collection `students`. For Atlas, put the Atlas URI in the
local `.env`; do not paste credentials into source files or commit `.env`.
The dashboard can read the preloaded CSV without MongoDB, but uploading a
dataset requires a working MongoDB connection. Uploaded rows are saved separately
in `uploaded_students`, with upload and training status in
`uploaded_dataset_batches`; the existing `students` collection is not replaced.

To check MongoDB connectivity:

```powershell
.\.venv\Scripts\python.exe -B -c "from database.mongo_connection import ping_mongodb; print('Connected' if ping_mongodb() else 'Connection failed')"
```

To import the project CSV:

```powershell
.\.venv\Scripts\python.exe -m database.import_data --csv data\students.csv
```

**Caution:** the importer clears existing documents in the target collection
before inserting by default. Confirm the target database and collection before
running it. MongoDB is not required to view the preloaded CSV dashboard.

### 4. Generate or retrain project data/models (optional)

The project already includes `data\students.csv` and a saved model. Do not run
these commands unless you intend to regenerate or retrain those project
artifacts:

```powershell
.\.venv\Scripts\python.exe run_project.py --rows 10000
.\.venv\Scripts\python.exe run_project.py --phase3
```

The first command overwrites the preloaded CSV. Phase 3 retrains and replaces the
saved model and metadata.

### 5. Run Spark processing (optional)

PySpark needs Java. In the PowerShell session, configure the validated Java 17
location and use the project venv for Spark worker processes:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$env:PYSPARK_PYTHON = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
```

Process the CSV dataset:

```powershell
.\.venv\Scripts\python.exe run_project.py --phase2
```

To process the MongoDB collection instead:

```powershell
.\.venv\Scripts\python.exe run_project.py --phase2 --from-mongo
```

Spark writes analytics CSV files into `outputs\`. Windows may show a
non-blocking `winutils.exe` warning.

### 6. Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open <http://localhost:8501>. To use another port if 8501 is occupied:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
```

Keep the PowerShell server process open while using the site. Stop it with
**Ctrl+C**.

### 7. Upload either sample CSV in Streamlit

1. In the dashboard sidebar, select **Upload New Dataset** in the **Dataset**
   selector.
2. In **Upload student dataset (CSV)**, browse to one of:
   - `D:\BDA\student-placement-bigdata\data\upload_samples\student_upload_sample_a.csv`
   - `D:\BDA\student-placement-bigdata\data\upload_samples\student_upload_sample_b.csv`
3. The dashboard automatically saves the uploaded rows to MongoDB, trains and
   evaluates Logistic Regression, Decision Tree, and Random Forest when the
   complete model feature schema and ML dependencies are available, and displays
   the results.
4. The upload's selected model is saved under
   `models\trained_model\uploads\<upload-id>\`. The upload-specific training
   summary is also recorded in `uploaded_dataset_batches`. The existing
   `students` collection and project model are not replaced.
5. The dataset can then be selected for dashboard analytics. Select **Reset
   Dataset** in the sidebar to return to the preloaded project dataset.

Uploads are identified by their content hash so the same file is not duplicated
in MongoDB during Streamlit reruns. CSVs missing model features are still saved
for analysis, but automatic model training is explicitly reported as skipped.
MongoDB must be running and reachable for an upload to complete its database
save.

## If machine-learning imports are blocked

Dataset analytics and CSV upload profiling do not require successful
scikit-learn/SciPy imports. When an ML dependency fails, the dashboard warns that
prediction and uploaded-data model training are unavailable. It does not show
fabricated predictions or metrics. Re-enable ML only through an authorized
administrator resolution of any Windows Application Control block.

## Re-enable prediction and model training

This environment previously reported an intermittent Application Control block
for SciPy's `_distance_wrap` native extension; the most recent direct import and
full validation passed. The project virtual-environment interpreter is
`D:\BDA\student-placement-bigdata\.venv\Scripts\python.exe` (Python 3.14.7). The
reported extension path was
`D:\BDA\student-placement-bigdata\.venv\Lib\site-packages\scipy\spatial\_distance_wrap.cp314-win_amd64.pyd`
(SciPy 1.18.1; scikit-learn 1.9.0). Ask the IT/security administrator to review
the matching Application Control event and resolve it using the organization's
approved policy. Do not disable Application Control, add a local exclusion, or
download replacement DLLs.

The issue was intermittent: a later direct `cdist` import and the full validation
script succeeded, while a simulated blocked-import regression test confirmed that
dataset analytics remain available and ML actions are disabled if the block
recurs.

## Run validation

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
.\scripts\validate_project.ps1
```

The script reports individual check statuses and exits nonzero when a required
check fails or is blocked. See [FINAL_TEST_REPORT.md](./FINAL_TEST_REPORT.md) for
the latest recorded verification.

After an authorized resolution, check whether prediction dependencies import:

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
.\.venv\Scripts\python.exe -c "from scipy.spatial.distance import cdist; print(cdist([[0,0]], [[1,1]]))"
```

The import check should print a distance matrix. If it still reports a blocked
DLL, the administrator needs to review the policy event; reinstalling packages
or changing dashboard code cannot override Application Control.

## Troubleshooting

- If the SciPy import check still reports that a DLL is blocked, contact the
  administrator who manages Application Control. Reinstalling packages or
  changing dashboard code cannot override that policy.
- If Streamlit reports that port 8501 is already in use, stop the other
  authorized Streamlit instance or choose another port:

  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
  ```
- If the app reports missing project data or model files, follow the Phase 1
  and Phase 3 setup instructions in [README.md](./README.md).
