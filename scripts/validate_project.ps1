param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$StartupTimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Checks = [System.Collections.Generic.List[object]]::new()

function Add-Check {
    param(
        [string]$Name,
        [ValidateSet("PASS", "FAIL", "BLOCKED", "SKIPPED")]
        [string]$Status,
        [string]$Details,
        [bool]$Required = $true
    )

    $Checks.Add([pscustomobject]@{
        Check = $Name
        Status = $Status
        Required = $Required
        Details = $Details
    })
}

function Invoke-PythonCheck {
    param(
        [string]$Name,
        [string]$Code,
        [bool]$Required = $true,
        [string]$BlockedPattern = ""
    )

    $stderrPath = Join-Path $env:TEMP ("student-placement-python-" + [guid]::NewGuid().ToString("N") + ".err.log")
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $encodedCode = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
        $runner = "import base64; exec(base64.b64decode('$encodedCode'))"
        $stdout = @(& $script:PythonExe -B -c $runner 2> $stderrPath)
        $exitCode = $LASTEXITCODE
        $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
    $output = (@($stdout) + @($stderr -split "`r?`n" | Where-Object { $_ })) -join "`n"
    $output = $output.Trim()
    if ($exitCode -eq 0) {
        $detailLines = @($stdout | ForEach-Object { "$_".Trim() } | Where-Object {
            $_ -and $_ -notmatch "^SUCCESS: The process with PID" -and $_ -notmatch "^Stage \d+:"
        } | Select-Object -Last 4)
        Add-Check -Name $Name -Status "PASS" -Details ($detailLines -join " | ") -Required $Required
        return "PASS"
    }

    $status = "FAIL"
    if ($BlockedPattern -and $output -match $BlockedPattern) {
        $status = "BLOCKED"
    }
    $details = ($output -split "`r?`n" | Select-Object -Last 8) -join " | "
    Add-Check -Name $Name -Status $status -Details $details -Required $Required
    return $status
}

Push-Location $ProjectRoot
try {
    $requiredPaths = @(
        "config\config.py",
        "dashboard\app.py",
        "dashboard\dataset_analysis.py",
        "data\students.csv",
        "database\mongo_connection.py",
        "database\import_data.py",
        "ml\predict.py",
        "ml\train_models.py",
        "models\trained_model\placement_model.joblib",
        "models\trained_model\training_metadata.json",
        "spark\analytics.py",
        "spark\data_ingestion.py",
        "spark\preprocessing.py",
        "spark\spark_session.py",
        "scripts\validate_project.ps1",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "run_project.py"
    )
    $missingPaths = @($requiredPaths | Where-Object { -not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_)) })
    if ($missingPaths.Count -eq 0) {
        Add-Check "Project structure" "PASS" "$($requiredPaths.Count) required paths present"
    }
    else {
        Add-Check "Project structure" "FAIL" ("Missing: " + ($missingPaths -join ", "))
    }

    $ignorePath = Join-Path $ProjectRoot ".gitignore"
    $ignoreLines = if (Test-Path -LiteralPath $ignorePath) { Get-Content -LiteralPath $ignorePath } else { @() }
    if ($ignoreLines | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq ".env" }) {
        Add-Check "Environment-file protection" "PASS" ".env ignore rule is present; credential values were not inspected"
    }
    else {
        Add-Check "Environment-file protection" "FAIL" ".env is not explicitly ignored by .gitignore"
    }

    if (Test-Path -LiteralPath $PythonExe) {
        $pythonVersion = (& $PythonExe -B -c "import sys; print(sys.version.split()[0]); print(sys.executable)" 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) {
            Add-Check "Project virtual environment" "PASS" ($pythonVersion -replace "`r?`n", " | ")
        }
        else {
            Add-Check "Project virtual environment" "FAIL" $pythonVersion
        }
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $PythonExe = $pythonCommand.Source
            $pythonVersion = (& $PythonExe -B -c "import sys; print(sys.version.split()[0]); print(sys.executable)" 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -eq 0) {
                Add-Check "Python interpreter" "PASS" ("Fallback interpreter: " + ($pythonVersion -replace "`r?`n", " | "))
            }
            else {
                Add-Check "Python interpreter" "FAIL" $pythonVersion
            }
        }
        else {
            Add-Check "Python interpreter" "FAIL" "Project virtual environment and system Python were not found"
        }
    }

    if (-not (Test-Path -LiteralPath $PythonExe) -and -not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
        Add-Check "Python-dependent checks" "SKIPPED" "No usable Python interpreter" -Required $false
    }
    else {
        $dependencyCode = @'
import importlib.metadata as metadata
names = ["numpy", "pandas", "pymongo", "python-dotenv", "scikit-learn", "pyspark", "streamlit", "plotly", "joblib"]
versions = [f"{name}={metadata.version(name)}" for name in names]
print("; ".join(versions))
'@
        Invoke-PythonCheck "Dependency inventory" $dependencyCode | Out-Null

        $configCode = @'
from config.config import BASE_DIR, DEFAULT_CSV_PATH, MODEL_DIR, MONGODB_URI
assert BASE_DIR.is_dir() and DEFAULT_CSV_PATH.is_file() and MODEL_DIR.is_dir()
assert isinstance(MONGODB_URI, str) and bool(MONGODB_URI.strip())
print("Configuration loaded; required paths exist; connection string not displayed")
'@
        Invoke-PythonCheck "Configuration and paths" $configCode | Out-Null

        $datasetCode = @'
import pandas as pd
from config.config import DEFAULT_CSV_PATH
data = pd.read_csv(DEFAULT_CSV_PATH)
required = {"student_id", "department", "placement_status", "cgpa", "attendance", "salary"}
missing = sorted(required - set(data.columns))
assert not data.empty and not missing, f"invalid dataset schema; missing={missing}"
print(f"{len(data)} rows, {len(data.columns)} columns; required schema valid")
'@
        Invoke-PythonCheck "Dataset schema" $datasetCode | Out-Null

        $mongoCode = @'
from config.config import MONGODB_COLLECTION, MONGODB_URI, MONGODB_DB_NAME
from pymongo import MongoClient
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=4000)
try:
    client.admin.command("ping")
    count = client[MONGODB_DB_NAME][MONGODB_COLLECTION].count_documents({})
    print(f"MongoDB reachable; configured collection contains {count} documents")
finally:
    client.close()
'@
        Invoke-PythonCheck "MongoDB connectivity and collection" $mongoCode -BlockedPattern "ServerSelectionTimeout|Connection refused|timed out|Network is unreachable|No route to host" | Out-Null

        $javaHome = $env:JAVA_HOME
        if (-not $javaHome -and (Test-Path -LiteralPath "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot\bin\java.exe")) {
            $javaHome = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
        }
        if ($javaHome -and (Test-Path -LiteralPath (Join-Path $javaHome "bin\java.exe"))) {
            $env:JAVA_HOME = $javaHome
            $env:Path = (Join-Path $javaHome "bin") + ";" + $env:Path
        }
        $env:PYSPARK_PYTHON = $PythonExe
        $env:PYSPARK_DRIVER_PYTHON = $PythonExe
        $sparkCode = @'
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").appName("project-validation").config("spark.ui.enabled", "false").getOrCreate()
try:
    count = spark.createDataFrame([(1,), (2,)], ["value"]).count()
    assert count == 2, f"unexpected DataFrame count: {count}"
    print(f"PySpark {spark.version}; small DataFrame count={count}")
finally:
    spark.stop()
'@
        Invoke-PythonCheck "PySpark operation" $sparkCode -BlockedPattern "Python worker failed|Python was not found|JAVA_GATEWAY_EXITED|Java gateway|JAVA_HOME|Access is denied|Application Control" | Out-Null

        $mlCode = @'
import scipy
from scipy.spatial.distance import cdist
import sklearn
result = cdist([[0, 0]], [[1, 1]])
assert result.shape == (1, 1)
print(f"SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}; cdist import succeeded")
'@
        $mlStatus = Invoke-PythonCheck "SciPy and scikit-learn imports" $mlCode -BlockedPattern "Application Control|blocked this file|Access is denied"

        if ($mlStatus -eq "PASS") {
            $predictionCode = @'
from ml.predict import predict_placement
values = {
    "department": "CSE", "semester": 8, "gender": "Female",
    "10th_percentage": 85.0, "12th_percentage": 84.0, "cgpa": 8.0,
    "attendance": 85.0, "backlogs": 0, "programming_score": 80,
    "aptitude_score": 75, "communication_score": 75, "projects": 2,
    "internships": 1, "certifications": 2, "hackathons": 1,
    "extracurricular_score": 70, "placement_training": 1
}
result = predict_placement(values)
assert result["predicted_status"] in {"Placed", "Not Placed"}
assert 0 <= result["placement_probability"] <= 1
print("Saved model loaded and returned a valid class/probability")
'@
            Invoke-PythonCheck "Saved-model prediction" $predictionCode | Out-Null
        }
        else {
            $status = if ($mlStatus -eq "BLOCKED") { "BLOCKED" } else { "SKIPPED" }
            Add-Check "Saved-model prediction" $status "Not run because SciPy/scikit-learn imports did not pass" -Required ($status -eq "BLOCKED")
        }

        $appTestCode = @'
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from config.config import DEFAULT_CSV_PATH

pages = ["Overview", "Performance Analytics", "Placement Analytics", "Correlation Patterns", "Placement Prediction", "Model Performance", "Data View", "Dataset Upload"]
for name in pages:
    app = AppTest.from_file(str(Path("dashboard") / "app.py")).run(timeout=60)
    app.sidebar.radio[0].set_value(name).run(timeout=60)
    errors = [str(item.value) for item in app.exception]
    assert not errors, f"{name} rendered exceptions: {errors}"

app = AppTest.from_file(str(Path("dashboard") / "app.py")).run(timeout=60)
data = pd.read_csv(DEFAULT_CSV_PATH)
expected = [
    ("Total Students", f"{len(data):,}"),
    ("Placed Students", f"{int((data['placement_status'] == 'Placed').sum()):,}"),
    ("Not Placed", f"{int((data['placement_status'] != 'Placed').sum()):,}"),
    ("Placement Rate", f"{(data['placement_status'] == 'Placed').mean() * 100:.2f}%"),
    ("Average CGPA", f"{data['cgpa'].mean():.2f}"),
    ("Average Salary", f"{data.loc[data['salary'] > 0, 'salary'].mean():,.2f} LPA"),
]
actual = [(metric.label, metric.value) for metric in app.metric]
assert actual == expected, f"dashboard KPI mismatch: {actual} != {expected}"

app.sidebar.radio[0].set_value("Dataset Upload").run(timeout=60)
app.file_uploader[0].upload("students.csv", DEFAULT_CSV_PATH.read_bytes(), "text/csv").run(timeout=60)
assert not app.exception, [str(item.value) for item in app.exception]
assert any(metric.label == "Rows" and metric.value == f"{len(data):,}" for metric in app.metric)
print(f"All {len(pages)} pages rendered; six live KPIs match CSV; project CSV upload profiled")
'@
        Invoke-PythonCheck "Streamlit rendered-page regression" $appTestCode | Out-Null

        $testDirectory = @(
            Get-ChildItem -LiteralPath $ProjectRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -in @("tests", "test") } |
                Select-Object -First 1
        )
        if ($testDirectory.Count -eq 0) {
            Add-Check "Automated test suite" "SKIPPED" "No tests or test directory exist in the project" -Required $false
        }
        else {
            $testRoot = $testDirectory[0].Name
            $unitCode = @"
import unittest
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("$testRoot"))
raise SystemExit(0 if result.wasSuccessful() else 1)
"@
            Invoke-PythonCheck "Automated test suite" $unitCode | Out-Null
        }
    }

    $portListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $portListener.Start()
    $port = $portListener.LocalEndpoint.Port
    $portListener.Stop()
    $stdoutPath = Join-Path $env:TEMP ("student-placement-streamlit-" + [guid]::NewGuid().ToString("N") + ".out.log")
    $stderrPath = Join-Path $env:TEMP ("student-placement-streamlit-" + [guid]::NewGuid().ToString("N") + ".err.log")
    $server = $null
    $serverHealthy = $false
    try {
        $arguments = @(
            "-B", "-m", "streamlit", "run", "dashboard\app.py",
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=$port",
            "--browser.gatherUsageStats=false"
        )
        $server = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $ProjectRoot -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        while ((Get-Date) -lt $deadline -and -not $server.HasExited) {
            try {
                $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port" -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    $serverHealthy = $true
                    break
                }
            }
            catch {
                Start-Sleep -Seconds 1
            }
        }
        if ($serverHealthy) {
            Add-Check "Streamlit HTTP startup" "PASS" "Streamlit server responded HTTP 200 on an ephemeral local port; rendered content is separately checked by AppTest"
        }
        else {
            $logText = ""
            if (Test-Path -LiteralPath $stderrPath) {
                $logText = Get-Content -LiteralPath $stderrPath -Raw
            }
            if ($logText -match "Application Control|blocked this file|Access is denied") {
                $status = "BLOCKED"
            }
            else {
                $status = "FAIL"
            }
            $logSummary = ($logText -split "`r?`n" | Select-Object -Last 8) -join " | "
            Add-Check "Streamlit HTTP startup" $status ("No HTTP 200 before timeout. Log: " + $logSummary)
        }
    }
    finally {
        if ($server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
            $server.WaitForExit()
        }
        if ($serverHealthy) {
            Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Output "Streamlit diagnostic logs preserved: $stdoutPath ; $stderrPath"
        }
    }
}
finally {
    Pop-Location
}

Write-Output ""
Write-Output "Validation results:"
$Checks | Format-Table -AutoSize -Wrap

$requiredFailures = @($Checks | Where-Object { $_.Required -and $_.Status -eq "FAIL" })
$requiredBlocked = @($Checks | Where-Object { $_.Required -and $_.Status -eq "BLOCKED" })
if ($requiredBlocked.Count -gt 0) {
    exit 2
}
if ($requiredFailures.Count -gt 0) {
    exit 1
}
exit 0
