# Windows Dashboard Run Instructions

## Dataset analytics can run without machine-learning dependencies

Start the dashboard from PowerShell:

```powershell
Set-Location "D:\BDA\student-placement-bigdata"
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open <http://localhost:8501> in a browser. Stop the server with **Ctrl+C** in
the PowerShell window. The dataset analytics pages can load without scikit-learn
or SciPy. If an ML dependency fails to load, the dashboard displays available
dataset analytics and warns that prediction and uploaded model training are
unavailable; it does not generate substitute predictions or metrics.

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
