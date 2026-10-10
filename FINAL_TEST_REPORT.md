# Project Cleanup and Validation Report

**Verification date:** 2026-10-10
**Project:** MongoDB-Based Big Data Analytics System for Student Performance and Placement Prediction
**Validation command:** `.\scripts\validate_project.ps1`

## Repository and safety inspection

- Project path: `D:\BDA\student-placement-bigdata`.
- This directory and its parent are not Git repositories. Git status, tracked/untracked
  state, and project Git history are therefore unavailable.
- No existing backup files or backup directories were found in the project.
- `.env` exists locally and remains untouched; its contents and credentials were not
  read or printed. `.env` is ignored by `.gitignore`.
- The project virtual environment, data, model, analytics outputs, source files, and
  configuration were preserved.
- Four project `__pycache__` directories were confirmed to contain only Python
  bytecode. Their 13 files were copied and hash-verified before removing the caches.
  Backup: `D:\BDA\maintenance-backups\project-cleanup-20261010-192552`.
- No unused source modules, duplicate datasets, empty project directories, or
  temporary log files were confirmed for removal. No source files were moved.

## Current validation results

| Check | Result | Evidence |
|---|---|---|
| Required project structure | PASS | All 20 required project paths, including this validation script, are present |
| Environment-file protection | PASS | `.env` ignore rule exists; credential values were not inspected |
| Project interpreter | PASS | `D:\BDA\student-placement-bigdata\.venv\Scripts\python.exe`, Python 3.14.7 |
| Dependency inventory | PASS | NumPy 2.5.2, pandas 3.0.5, PyMongo 4.18.0, python-dotenv 1.2.3, scikit-learn 1.9.0, PySpark 4.2.0, Streamlit 1.63.0, Plotly 7.0.0, Joblib 1.6.0 |
| Configuration and paths | PASS | Configuration loaded and required paths validated without displaying the connection string |
| Dataset schema | PASS | `data/students.csv`: 10,000 rows, 20 columns, required schema present |
| Dataset quality | PASS | 0 missing cells, 0 duplicate rows; 9,230 Placed and 770 Not Placed |
| MongoDB | PASS | Ping succeeded; `student_analytics.students` contains 10,000 documents |
| PySpark | PASS | Spark 4.2.0 small DataFrame count returned 2 with the project venv configured as worker Python |
| SciPy/scikit-learn | PASS | SciPy 1.18.1 `cdist` and scikit-learn 1.9.0 imported successfully during this run |
| Saved model | PASS | Existing model loaded; test call returned `Placed`, probability 0.513 (51.30%). No retraining or artifact replacement occurred |
| Streamlit rendering | PASS | AppTest rendered all eight pages with zero exceptions; six Overview KPIs matched values calculated from the CSV |
| Dataset upload | PASS | Existing project CSV upload was profiled in AppTest; 10,000 rows and no Streamlit exception |
| Streamlit HTTP startup | PASS | Validation script launched an isolated local server on an ephemeral port and received HTTP 200; browser inspection separately confirmed real rendered dashboard content on port 8503 |
| Automated test suite | SKIPPED | No `tests/` or `test/` directory exists. `python -B -m unittest discover -v` found 0 tests and returned code 5 |
| Simulated ML security block | PASS | Regression test simulated the SciPy Application Control ImportError; analytics remained visible and prediction/model training were disabled with clear warnings |

## Diagnostics and environment notes

- A prior startup traceback showed Windows Application Control blocking
  `scipy.spatial._distance_wrap` in the project venv:
  `D:\BDA\student-placement-bigdata\.venv\Lib\site-packages\scipy\spatial\_distance_wrap.cp314-win_amd64.pyd`.
- The same environment later imported `cdist` successfully. The block appears
  intermittent; no security setting was changed or bypassed. If it recurs, the
  organization’s administrator should review the policy event.
- The first Spark smoke test failed because the Spark worker attempted to launch
  the Windows Store `python` alias. Setting `PYSPARK_PYTHON` and
  `PYSPARK_DRIVER_PYTHON` to the project venv interpreter made the Spark operation
  pass. The validation script performs this setup.
- Spark emits non-blocking Windows `winutils.exe` and native-Hadoop warnings.
  PySpark also warns that pandas 3.0 is newer than its fully supported range.
- The initial validator draft exposed a PowerShell argument-quoting issue. The
  script was corrected to pass encoded Python source; the complete validation then
  passed.
- No MongoDB credentials, values from `.env`, or other secrets are included in
  validation output.

## Files changed

- Added `scripts/validate_project.ps1` for repeatable structure, environment,
  configuration, dataset, MongoDB, Spark, ML, dashboard, upload, and test checks.
- Updated `README.md` with the actual project structure and validation instructions.
- Updated `RUN_INSTRUCTIONS.md` with the current intermittent-block status and
  validation command.
- Replaced this report with the current validation evidence.
- Removed only four generated cache directories after backup; no source, data,
  configuration, model, or analytics output was deleted.

## Run commands

From PowerShell:

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
.\scripts\validate_project.ps1
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The validator returns nonzero if a required check fails or is blocked. A
Windows Application Control block is reported as `BLOCKED`; the dashboard’s
dataset analytics can still be exercised independently, while ML actions remain
disabled until the extension is available.

## Final status

**Current validation script run: 12 PASS, 1 SKIPPED, 0 FAIL, 0 BLOCKED.** The
separate simulated security-block regression also passed. Automated unit tests
are not present. The previous Application Control denial remains recorded as an
intermittent environmental issue, but the latest direct import, model prediction,
rendered dashboard, and full validation run succeeded. The project is not claimed
to have a unit-test suite or uninterrupted ML availability.
