from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DEFAULT_CSV_PATH, MODEL_DIR  # noqa: E402
try:
    from ml.predict import predict_placement  # noqa: E402
except ImportError as exc:
    predict_placement = None
    ML_IMPORT_ERROR = str(exc)
else:
    ML_IMPORT_ERROR = None

from dashboard.dataset_analysis import (  # noqa: E402
    dynamic_numeric_columns,
    dataset_profile,
    load_uploaded_csv,
    train_uploaded_models,
    validate_uploaded_dataset,
)


st.set_page_config(
    page_title="Student Performance & Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --blue: #2563eb;
        --muted: #64748b;
        --surface: #ffffff;
        --line: #e2e8f0;
    }
    .stApp { background: #f4f7fb; color: #102a43; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #173f5f 100%);
        border-right: 1px solid #1e4667;
    }
    [data-testid="stSidebar"] * { color: #e8f1f8; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 8px 10px;
        margin: 2px 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.12); }
    .dashboard-brand {
        padding: 8px 4px 18px;
        border-bottom: 1px solid rgba(255,255,255,.18);
        margin-bottom: 18px;
    }
    .dashboard-brand-title { font-size: 1.05rem; font-weight: 700; line-height: 1.25; }
    .dashboard-brand-subtitle { color: #b9d3e8 !important; font-size: .76rem; margin-top: 6px; }
    .page-kicker { color: #2563eb; font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .page-description { color: #64748b; margin-top: -8px; margin-bottom: 20px; }
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    .result-card {
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        box-shadow: 0 8px 22px rgba(37, 99, 235, .1);
    }
    .result-status { color: #102a43; font-size: 1.6rem; font-weight: 750; }
    .result-probability { color: #2563eb; font-size: 2.25rem; font-weight: 800; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    .st-key-overview-kpis [data-testid="stMetric"] {
        background-color: #ffffff;
        border-color: #dbe5ef;
    }
    .st-key-overview-kpis [data-testid="stMetricValue"],
    .st-key-overview-kpis [data-testid="stMetricLabel"],
    .st-key-overview-kpis [data-testid="stMetricLabel"] p {
        color: #1f2937;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 650;
    }
    h1, h2, h3 { color: #102a43; letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Student dataset not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required = {"student_id", "department", "placement_status", "cgpa", "attendance", "salary"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Student dataset is missing required columns: {', '.join(missing)}")
    return dataframe


@st.cache_data
def load_metadata(path: str) -> dict:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def money(value: float) -> str:
    return f"{value:,.2f} LPA"


def page_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


def style_chart(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#334e68"},
        title={"font": {"size": 18, "color": "#102a43"}},
        margin={"l": 18, "r": 18, "t": 54, "b": 18},
        legend={"bgcolor": "rgba(255,255,255,.75)"},
    )
    return figure


def overview_page(dataframe: pd.DataFrame, metadata: dict) -> None:
    page_intro("Executive summary", "Overview", "A Spark-processed snapshot of student performance, placement outcomes, and academic readiness.")
    total = len(dataframe)
    placed = int((dataframe["placement_status"] == "Placed").sum())
    not_placed = total - placed
    metrics = [
        ("Total Students", f"{total:,}"),
        ("Placed Students", f"{placed:,}"),
        ("Not Placed", f"{not_placed:,}"),
        ("Placement Rate", f"{placed / total * 100:.2f}%"),
        ("Average CGPA", f"{dataframe['cgpa'].mean():.2f}"),
        ("Average Salary", money(dataframe.loc[dataframe["salary"] > 0, "salary"].mean())),
    ]
    with st.container(key="overview-kpis"):
        columns = st.columns(6)
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)

    left, right = st.columns(2)
    with left:
        counts = dataframe["placement_status"].value_counts().rename_axis("status").reset_index(name="students")
        st.plotly_chart(style_chart(px.pie(counts, names="status", values="students", hole=0.45, title="Placement distribution")), width="stretch")
    with right:
        department = dataframe.groupby("department", as_index=False).size().rename(columns={"size": "students"})
        st.plotly_chart(style_chart(px.bar(department, x="department", y="students", title="Students by department", text_auto=True)), width="stretch")
    st.info("The current dataset is synthetic and was generated for project demonstration. It is not a guaranteed placement outcome.")
    st.caption(f"Best saved model: {metadata.get('best_model', 'Unavailable')}")


def performance_page(dataframe: pd.DataFrame) -> None:
    page_intro("Big data analytics", "Student Performance", "Explore academic readiness, attendance patterns, and department-level performance.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(style_chart(px.histogram(dataframe, x="cgpa", color="placement_status", nbins=25, barmode="overlay", title="CGPA distribution")), width="stretch")
    with right:
        st.plotly_chart(style_chart(px.histogram(dataframe, x="attendance", color="placement_status", nbins=25, barmode="overlay", title="Attendance distribution")), width="stretch")
    department_cgpa = dataframe.groupby("department", as_index=False)["cgpa"].mean().sort_values("cgpa", ascending=False)
    st.plotly_chart(style_chart(px.bar(department_cgpa, x="department", y="cgpa", title="Average CGPA by department", text_auto=".2f")), width="stretch")
    st.plotly_chart(
        style_chart(px.box(dataframe, x="placement_status", y="cgpa", color="placement_status", title="CGPA comparison by placement status")),
        width="stretch",
    )


def placement_page(dataframe: pd.DataFrame) -> None:
    page_intro("Outcome intelligence", "Placement Analytics", "Compare placement outcomes across departments, experience indicators, and assessed skills.")
    department_rate = (
        dataframe.assign(is_placed=(dataframe["placement_status"] == "Placed").astype(int))
        .groupby("department", as_index=False)
        .agg(students=("student_id", "count"), placement_rate=("is_placed", "mean"))
    )
    department_rate["placement_rate"] *= 100
    st.plotly_chart(style_chart(px.bar(department_rate, x="department", y="placement_rate", text_auto=".2f", title="Department-wise placement rate (%)")), width="stretch")

    comparison_features = ["attendance", "cgpa", "internships", "projects", "certifications", "backlogs", "programming_score", "aptitude_score", "communication_score"]
    selected_feature = st.selectbox("Compare a feature by placement status", comparison_features, format_func=lambda value: value.replace("_", " ").title())
    comparison = dataframe.groupby("placement_status", as_index=False)[selected_feature].mean()
    st.plotly_chart(style_chart(px.bar(comparison, x="placement_status", y=selected_feature, color="placement_status", text_auto=".2f", title=f"Average {selected_feature.replace('_', ' ').title()} by placement status")), width="stretch")

    feature_rates = []
    for feature in ["internships", "projects", "certifications", "backlogs"]:
        grouped = dataframe.groupby(feature, as_index=False).agg(placement_rate=("placement_status", lambda values: (values == "Placed").mean() * 100), students=("student_id", "count"))
        grouped["feature"] = feature
        grouped["value"] = grouped[feature]
        feature_rates.append(grouped[["feature", "value", "placement_rate", "students"]])
    patterns = pd.concat(feature_rates, ignore_index=True)
    st.plotly_chart(style_chart(px.line(patterns, x="value", y="placement_rate", color="feature", markers=True, hover_data=["students"], title="Placement rate across experience indicators")), width="stretch")


def pattern_page(dataframe: pd.DataFrame) -> None:
    page_intro("Relationships", "Correlation & Patterns", "Identify associations across academic, skills, experience, and compensation signals.")
    numeric_columns = ["10th_percentage", "12th_percentage", "cgpa", "attendance", "backlogs", "programming_score", "aptitude_score", "communication_score", "projects", "internships", "certifications", "hackathons", "extracurricular_score", "salary"]
    correlation = dataframe[numeric_columns].corr()
    st.plotly_chart(style_chart(px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature correlation heatmap")), width="stretch")
    strongest = correlation["cgpa"].drop("cgpa").abs().sort_values(ascending=False).head(5)
    st.subheader("Strongest relationships with CGPA")
    strongest_table = strongest.rename("absolute_correlation").reset_index()
    strongest_table = strongest_table.rename(columns={"index": "feature"})
    st.dataframe(strongest_table, width="stretch", hide_index=True)
    st.warning("Correlation shows association between variables; it does not prove that one variable causes another.")


def prediction_page() -> None:
    page_intro("Decision support", "Placement Prediction", "Estimate placement likelihood using the persisted Random Forest model.")
    if predict_placement is None:
        st.warning("Placement prediction is unavailable because the machine-learning dependencies could not be loaded.")
        st.caption(f"ML import error: {ML_IMPORT_ERROR}")
        return

    st.warning("This is a model prediction, not a guaranteed placement result. The model was trained on synthetic data.")
    with st.form("placement_prediction_form"):
        st.markdown("#### Academic profile")
        left, middle, right = st.columns(3)
        with left:
            department = st.selectbox("Department", ["CSE", "ECE", "EEE", "MECH", "IT", "CIVIL"])
            semester = st.slider("Semester", 1, 8, 8)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            tenth = st.number_input("10th percentage", 0.0, 100.0, 85.0)
            twelfth = st.number_input("12th percentage", 0.0, 100.0, 84.0)
            cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
        with middle:
            attendance = st.number_input("Attendance", 0.0, 100.0, 85.0)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            programming = st.slider("Programming score", 0, 100, 80)
            aptitude = st.slider("Aptitude score", 0, 100, 75)
            communication = st.slider("Communication score", 0, 100, 75)
            extracurricular = st.slider("Extracurricular score", 0, 100, 70)
        st.markdown("#### Experience and engagement")
        with right:
            projects = st.number_input("Projects", 0, 20, 2)
            internships = st.number_input("Internships", 0, 10, 1)
            certifications = st.number_input("Certifications", 0, 20, 2)
            hackathons = st.number_input("Hackathons", 0, 20, 1)
            training = st.checkbox("Placement training completed", value=True)
        submitted = st.form_submit_button("Predict Placement", type="primary")

    if submitted:
        values = {
            "department": department, "semester": semester, "gender": gender,
            "10th_percentage": tenth, "12th_percentage": twelfth, "cgpa": cgpa,
            "attendance": attendance, "backlogs": backlogs, "programming_score": programming,
            "aptitude_score": aptitude, "communication_score": communication, "projects": projects,
            "internships": internships, "certifications": certifications, "hackathons": hackathons,
            "extracurricular_score": extracurricular, "placement_training": int(training),
        }
        try:
            result = predict_placement(values)
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(str(exc))
        else:
            status = result["predicted_status"]
            st.markdown(
                f'<div class="result-card"><div class="result-status">Prediction: {status}</div>'
                f'<div class="result-probability">{result["placement_probability_percent"]:.2f}%</div>'
                '<div style="color:#64748b;">Estimated probability of placement</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Class labels used by the saved model: {', '.join(result['class_labels'])}")


def model_page(metadata: dict | None) -> None:
    page_intro("Model governance", "Model Performance", "Compare validated classifiers and inspect the selected model's confusion matrix.")
    if metadata is None:
        st.warning("Saved model metrics are unavailable because the training metadata file could not be loaded.")
        return
    rows = []
    for name, values in metadata.get("models", {}).items():
        rows.append({"Model": name.replace("_", " ").title(), **{key: values[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}})
    metric_columns = st.columns(len(rows))
    for column, row in zip(metric_columns, rows):
        column.metric(row["Model"], f'{row["f1"]:.2%}', f'Accuracy {row["accuracy"]:.2%}')
    st.dataframe(pd.DataFrame(rows).style.format({key: "{:.4f}" for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}), width="stretch", hide_index=True)
    selected = st.selectbox("Confusion matrix model", list(metadata.get("models", {}).keys()), index=list(metadata.get("models", {})).index(metadata.get("best_model")) if metadata.get("best_model") in metadata.get("models", {}) else 0)
    values = metadata["models"][selected]
    matrix = values["confusion_matrix"]
    figure = go.Figure(go.Heatmap(z=matrix, x=["Predicted Not Placed", "Predicted Placed"], y=["Actual Not Placed", "Actual Placed"], text=matrix, texttemplate="%{text}", colorscale="Blues"))
    figure.update_layout(title=f"{selected.replace('_', ' ').title()} confusion matrix")
    st.plotly_chart(style_chart(figure), width="stretch")
    if selected == "random_forest":
        st.info("Random Forest ROC-AUC is approximately 0.508. This is expected for the current synthetic generator: placement probabilities are heavily saturated, so threshold metrics are strong while probability ranking is close to random.")


def data_page(dataframe: pd.DataFrame) -> None:
    page_intro("Records", "Data Explorer", "Filter, search, and inspect the student records feeding the dashboard.")
    filter_left, filter_middle, filter_right = st.columns([1.2, 1.2, 1])
    departments = filter_left.multiselect("Department", sorted(dataframe["department"].unique()), default=sorted(dataframe["department"].unique()))
    statuses = filter_middle.multiselect("Placement status", ["Placed", "Not Placed"], default=["Placed", "Not Placed"])
    search = filter_right.text_input("Search student ID", placeholder="e.g. STU00042")
    filtered = dataframe[dataframe["department"].isin(departments) & dataframe["placement_status"].isin(statuses)]
    if search:
        filtered = filtered[filtered["student_id"].astype(str).str.contains(search, case=False, na=False)]
    st.caption(f"Showing {len(filtered):,} of {len(dataframe):,} records")
    st.dataframe(filtered, width="stretch", hide_index=True)


def upload_page(dataframe: pd.DataFrame | None, upload_name: str | None) -> None:
    page_intro("Flexible ingestion", "Dataset Upload", "Upload a CSV to automatically profile, validate, analyze, and optionally train compatible models.")
    uploaded_file = st.file_uploader("Upload student dataset (CSV)", type=["csv"], help="Excel and JSON support can be added later without changing the analysis layer.")
    if uploaded_file is not None:
        try:
            dataframe = load_uploaded_csv(uploaded_file.getvalue())
            upload_name = uploaded_file.name
            st.session_state["uploaded_dataframe"] = dataframe
            st.session_state["uploaded_name"] = upload_name
            st.session_state["uploaded_size"] = uploaded_file.size
        except ValueError as exc:
            st.error(str(exc))
            return

    if dataframe is None:
        st.info("Upload a CSV to begin. The preloaded project dataset remains available from the Dataset selector.")
        return

    profile = dataset_profile(dataframe, uploaded_file.size if uploaded_file else st.session_state.get("uploaded_size", 0))
    validation = validate_uploaded_dataset(dataframe)
    st.success(f"Analyzing uploaded dataset: {upload_name or 'uploaded CSV'}")
    metric_columns = st.columns(5)
    metric_columns[0].metric("File size", f"{profile['file_size_kb']:.1f} KB")
    metric_columns[1].metric("Rows", f"{profile['rows']:,}")
    metric_columns[2].metric("Columns", profile["columns"])
    metric_columns[3].metric("Missing cells", profile["missing_cells"])
    metric_columns[4].metric("Duplicates", profile["duplicate_rows"])

    if validation["missing_expected_columns"]:
        st.warning("Expected student columns not detected: " + ", ".join(validation["missing_expected_columns"]))
    else:
        st.success("Core student-performance columns detected.")
    if profile["missing_by_column"]:
        st.dataframe(pd.DataFrame({"column": list(profile["missing_by_column"]), "missing_values": list(profile["missing_by_column"].values())}), width="stretch", hide_index=True)
    else:
        st.info("No missing values detected.")

    numeric_columns = dynamic_numeric_columns(dataframe)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        if "department" in dataframe.columns:
            department_counts = dataframe["department"].astype(str).value_counts().rename_axis("department").reset_index(name="students")
            st.plotly_chart(style_chart(px.bar(department_counts, x="department", y="students", title="Department distribution", text_auto=True)), width="stretch", key="uploaded_department_distribution")
        elif numeric_columns:
            st.plotly_chart(style_chart(px.histogram(dataframe, x=numeric_columns[0], title=f"{numeric_columns[0]} distribution")), width="stretch", key="uploaded_numeric_distribution_left")
    with chart_right:
        if "placement_status" in dataframe.columns:
            placement_counts = dataframe["placement_status"].astype(str).value_counts().rename_axis("placement_status").reset_index(name="students")
            st.plotly_chart(style_chart(px.pie(placement_counts, names="placement_status", values="students", hole=.45, title="Placement distribution")), width="stretch", key="uploaded_placement_distribution")
        elif numeric_columns:
            st.plotly_chart(style_chart(px.histogram(dataframe, x=numeric_columns[-1], title=f"{numeric_columns[-1]} distribution")), width="stretch", key="uploaded_numeric_distribution_right")

    st.subheader("Automatic dataset analysis")
    if numeric_columns:
        st.dataframe(dataframe[numeric_columns].describe().T.reset_index().rename(columns={"index": "feature"}), width="stretch")
        correlation = dataframe[numeric_columns].corr()
        if len(numeric_columns) > 1:
            st.plotly_chart(style_chart(px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Available numeric feature correlations")), width="stretch", key="uploaded_numeric_correlations")
    analysis_columns = [column for column in ["cgpa", "attendance", "internships", "projects", "certifications", "backlogs", "programming_score", "aptitude_score", "communication_score"] if column in dataframe.columns]
    if "placement_status" in dataframe.columns and analysis_columns:
        selected = st.selectbox("Analyze an available feature by placement status", analysis_columns, format_func=lambda value: value.replace("_", " ").title(), key="uploaded_feature")
        comparison = dataframe.groupby("placement_status", as_index=False)[selected].mean()
        st.plotly_chart(style_chart(px.bar(comparison, x="placement_status", y=selected, color="placement_status", text_auto=".2f", title=f"{selected.replace('_', ' ').title()} by placement status")), width="stretch", key="uploaded_placement_comparison")
    else:
        st.info("Placement comparisons require a placement_status column and at least one supported numeric feature.")

    st.subheader("Optional uploaded-dataset model training")
    if not validation["ml_compatible"]:
        if validation["ml_dependency_error"]:
            st.warning("Uploaded-dataset model training is unavailable because machine-learning dependencies could not be loaded.")
            st.caption(f"ML import error: {validation['ml_dependency_error']}")
        else:
            st.info("Model training is unavailable for this upload because the exact Phase 3 feature schema is incomplete.")
            st.caption("Missing ML columns: " + ", ".join(validation["ml_missing_columns"]))
    elif st.button("Train and compare models on uploaded data", type="primary"):
        try:
            with st.spinner("Training Logistic Regression, Decision Tree, and Random Forest..."):
                results = train_uploaded_models(dataframe)
            st.success(f"Training complete. Best model: {results['best_model']}")
            rows = [{"Model": name.replace("_", " ").title(), **{key: values[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}} for name, values in results["models"].items()]
            st.dataframe(pd.DataFrame(rows).style.format({key: "{:.4f}" for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}), width="stretch", hide_index=True)
        except (ValueError, KeyError) as exc:
            st.error(f"Uploaded dataset model training failed: {exc}")


def reset_uploaded_dataset() -> None:
    st.session_state.pop("uploaded_dataframe", None)
    st.session_state.pop("uploaded_name", None)
    st.session_state.pop("uploaded_size", None)
    st.session_state["dataset_mode"] = "Project Dataset"


def main() -> None:
    try:
        dataframe = load_dataset(str(DEFAULT_CSV_PATH))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Dashboard could not load required project files: {exc}")
        st.info("Run Phase 1 to generate the project dataset.")
        st.stop()

    try:
        metadata = load_metadata(str(MODEL_DIR / "training_metadata.json"))
    except (FileNotFoundError, ValueError, OSError) as exc:
        metadata = None
        metadata_error = str(exc)
    else:
        metadata_error = None

    with st.sidebar:
        st.markdown(
            '<div class="dashboard-brand"><div class="dashboard-brand-title">🎓 Student Analytics</div>'
            '<div class="dashboard-brand-subtitle">MongoDB • PySpark • Machine Learning</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Workspace")
        dataset_mode = st.selectbox("Dataset", ["Project Dataset", "Upload New Dataset"], key="dataset_mode")
        page = st.radio(
            "Navigate",
            ["Overview", "Performance Analytics", "Placement Analytics", "Correlation Patterns", "Placement Prediction", "Model Performance", "Data View", "Dataset Upload"],
            label_visibility="collapsed",
        )
        st.button("Reset Dataset", use_container_width=True, on_click=reset_uploaded_dataset)
        st.markdown("---")
        st.caption("Preloaded project data • 10,000 synthetic records")
    st.markdown("## MongoDB-Based Big Data Analytics System")
    st.caption("Student performance and placement prediction | Academic project demonstration")
    if predict_placement is None:
        st.warning("Machine-learning dependencies are unavailable in this environment. Dataset statistics and analytics remain available; placement prediction and uploaded model training are disabled.")
    if metadata_error:
        st.warning(f"Saved model metrics are unavailable: {metadata_error}")
    if dataset_mode == "Upload New Dataset":
        upload_page(st.session_state.get("uploaded_dataframe"), st.session_state.get("uploaded_name"))
        return
    if page == "Dataset Upload":
        upload_page(st.session_state.get("uploaded_dataframe"), st.session_state.get("uploaded_name"))
        return
    if page == "Overview":
        overview_page(dataframe, metadata)
    elif page == "Performance Analytics":
        performance_page(dataframe)
    elif page == "Placement Analytics":
        placement_page(dataframe)
    elif page == "Correlation Patterns":
        pattern_page(dataframe)
    elif page == "Placement Prediction":
        prediction_page()
    elif page == "Model Performance":
        model_page(metadata)
    else:
        data_page(dataframe)


if __name__ == "__main__":
    main()
