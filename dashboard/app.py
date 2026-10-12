from __future__ import annotations

import hashlib
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
from database.uploaded_datasets import (  # noqa: E402
    persist_uploaded_dataset,
    save_uploaded_training_result,
)
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
        --canvas: #f4f7fb;
        --ink: #102a43;
        --shadow: 0 8px 24px rgba(15, 42, 67, .06);
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, .92);
        border-bottom: 1px solid rgba(226, 232, 240, .75);
        backdrop-filter: blur(12px);
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #173f5f 100%);
        border-right: 1px solid #1e4667;
        box-shadow: 8px 0 28px rgba(15, 42, 67, .08);
    }
    [data-testid="stSidebar"] * { color: #e8f1f8; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 9px 10px;
        margin: 3px 0;
        transition: background-color .18s ease, border-color .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, .1);
        border-color: rgba(255, 255, 255, .12);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(96, 165, 250, .2);
        border-color: rgba(191, 219, 254, .3);
    }
    .dashboard-brand {
        padding: 8px 4px 18px;
        border-bottom: 1px solid rgba(255,255,255,.18);
        margin-bottom: 18px;
    }
    .dashboard-brand-title { font-size: 1.08rem; font-weight: 750; line-height: 1.25; letter-spacing: -.01em; }
    .dashboard-brand-subtitle { color: #b9d3e8 !important; font-size: .78rem; margin-top: 7px; }
    .page-kicker {
        color: var(--blue);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: .4rem;
    }
    .page-description { color: var(--muted); margin-top: -8px; margin-bottom: 24px; max-width: 70rem; line-height: 1.65; }
    .section-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        box-shadow: var(--shadow);
    }
    .result-card {
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        box-shadow: 0 10px 28px rgba(37, 99, 235, .1);
    }
    .result-status { color: #102a43; font-size: 1.6rem; font-weight: 750; }
    .result-probability { color: #2563eb; font-size: 2.25rem; font-weight: 800; }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 18px;
        box-shadow: 0 4px 14px rgba(15, 42, 67, .045);
        transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #bfdbfe;
        box-shadow: 0 10px 24px rgba(37, 99, 235, .09);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #1f2937 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        font-size: .82rem;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 750;
    }
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #1f2937 !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }
    .stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 650;
        transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        border-color: #93c5fd;
        box-shadow: 0 5px 14px rgba(37, 99, 235, .12);
        transform: translateY(-1px);
    }
    .stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible,
    [data-testid="stDownloadButton"] > button:focus-visible {
        outline: 3px solid rgba(37, 99, 235, .35);
        outline-offset: 2px;
    }
    [data-testid="stPlotlyChart"] {
        overflow: visible;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: 0 4px 14px rgba(15, 42, 67, .04);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #93c5fd !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #2563eb !important;
        border: 1px solid #2563eb !important;
        border-radius: 9px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, .22) !important;
    }
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #334e68 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #526579 !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 42, 67, .04);
    }
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 12px;
        background: rgba(255, 255, 255, .72);
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        border-radius: 10px;
    }
    :focus-visible {
        outline-color: rgba(37, 99, 235, .7);
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
    h1 { font-size: clamp(2rem, 3vw, 2.65rem) !important; line-height: 1.12 !important; }
    h2 { line-height: 1.2 !important; }
    @media (max-width: 1200px) {
        div[data-testid="stMetric"] { padding: 14px 12px; }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            font-size: .64rem;
            line-height: 1.2;
            white-space: normal !important;
            word-break: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
            font-size: .72rem !important;
            line-height: 1.2;
            white-space: normal !important;
            word-break: normal !important;
            overflow: visible !important;
            overflow-wrap: normal;
            text-overflow: clip !important;
        }
    }
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 2rem; }
        .page-description { margin-bottom: 18px; }
        div[data-testid="stMetric"] { padding: 13px; border-radius: 12px; }
        [data-testid="stPlotlyChart"] { border-radius: 11px; }
    }
    @media (prefers-reduced-motion: reduce) {
        .stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] > button,
        [data-testid="stSidebar"] [role="radiogroup"] label, div[data-testid="stMetric"] {
            transition-duration: .01ms !important;
        }
    }
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
        legend={"bgcolor": "rgba(255,255,255,.75)", "font": {"color": "#334e68"}},
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#cbd5e1",
            "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#102a43", "size": 13},
        },
        xaxis={"tickfont": {"color": "#526579"}, "title_font": {"color": "#526579"}},
        yaxis={"tickfont": {"color": "#526579"}, "title_font": {"color": "#526579"}},
    )
    return figure


def overview_page(dataframe: pd.DataFrame, metadata: dict, is_uploaded: bool = False) -> None:
    page_intro("Executive summary", "Overview", "Review student performance, placement outcomes, and academic readiness for the selected dataset.")
    total = len(dataframe)
    metrics = [("Total Students", f"{total:,}")]
    if "placement_status" in dataframe:
        placed = int((dataframe["placement_status"] == "Placed").sum())
        metrics.extend([
            ("Placed Students", f"{placed:,}"),
            ("Not Placed", f"{total - placed:,}"),
            ("Placement Rate", f"{placed / total * 100:.2f}%"),
        ])
    if "cgpa" in dataframe:
        metrics.append(("Average CGPA", f"{dataframe['cgpa'].mean():.2f}"))
    if "salary" in dataframe:
        salary_values = dataframe.loc[dataframe["salary"] > 0, "salary"]
        metrics.append(("Average Salary", money(salary_values.mean()) if not salary_values.empty else "N/A"))
    with st.container(key="overview-kpis"):
        columns = st.columns(len(metrics))
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)

    charts = []
    if "placement_status" in dataframe:
        counts = dataframe["placement_status"].value_counts().rename_axis("status").reset_index(name="students")
        charts.append(style_chart(px.pie(counts, names="status", values="students", hole=0.45, title="Placement distribution")))
    if "department" in dataframe:
        department = dataframe.groupby("department", as_index=False).size().rename(columns={"size": "students"})
        charts.append(style_chart(px.bar(department, x="department", y="students", title="Students by department", text_auto=True)))
    for chart in charts:
        st.plotly_chart(chart, width="stretch")
    if not charts:
        st.info("Add a department or placement_status column to see overview charts.")
    if is_uploaded:
        st.info("Overview values and charts are calculated from the selected uploaded dataset.")
    else:
        st.info("The current dataset is synthetic and was generated for project demonstration. It is not a guaranteed placement outcome.")
        st.caption(f"Best saved model: {metadata.get('best_model', 'Unavailable') if metadata else 'Unavailable'}")


def performance_page(dataframe: pd.DataFrame) -> None:
    page_intro("Big data analytics", "Student Performance", "Explore academic readiness, attendance patterns, and department-level performance.")
    charts = []
    for feature in ("cgpa", "attendance"):
        if feature in dataframe:
            charts.append(style_chart(px.histogram(
                dataframe, x=feature, color="placement_status" if "placement_status" in dataframe else None,
                nbins=25, barmode="overlay", title=f"{feature.replace('_', ' ').title()} distribution",
                labels={"placement_status": "Placement", feature: feature.replace("_", " ").title(), "count": "Students"},
            )))
    if {"department", "cgpa"}.issubset(dataframe.columns):
        department_cgpa = dataframe.groupby("department", as_index=False)["cgpa"].mean().sort_values("cgpa", ascending=False)
        charts.append(style_chart(px.bar(department_cgpa, x="department", y="cgpa", title="Average CGPA by department", text_auto=".2f")))
    if {"placement_status", "cgpa"}.issubset(dataframe.columns):
        charts.append(style_chart(px.box(dataframe, x="placement_status", y="cgpa", color="placement_status", title="CGPA comparison by placement status")))
    for chart in charts:
        st.plotly_chart(chart, width="stretch")
    if not charts:
        st.info("No supported performance columns were found. Include numeric cgpa or attendance columns to see charts.")


def placement_page(dataframe: pd.DataFrame) -> None:
    page_intro("Outcome intelligence", "Placement Analytics", "Compare placement outcomes across departments, experience indicators, and assessed skills.")
    comparison_features = ["attendance", "cgpa", "internships", "projects", "certifications", "backlogs", "programming_score", "aptitude_score", "communication_score"]
    if {"department", "placement_status"}.issubset(dataframe.columns):
        department_rate = dataframe.assign(is_placed=(dataframe["placement_status"] == "Placed").astype(int)).groupby("department", as_index=False).agg(
            students=("is_placed", "size"), placement_rate=("is_placed", "mean")
        )
        department_rate["placement_rate"] *= 100
        st.plotly_chart(style_chart(px.bar(department_rate, x="department", y="placement_rate", text_auto=".2f", title="Department-wise placement rate (%)")), width="stretch")
    else:
        st.info("Department-wise placement rates require department and placement_status columns.")

    feature_rates = []
    if "placement_status" in dataframe:
        for feature in ["internships", "projects", "certifications", "backlogs"]:
            if feature in dataframe:
                grouped = dataframe.groupby(feature, as_index=False).agg(
                    placement_rate=("placement_status", lambda values: (values == "Placed").mean() * 100),
                    students=("placement_status", "size"),
                )
                grouped["feature"] = feature
                grouped["value"] = grouped[feature]
                feature_rates.append(grouped[["feature", "value", "placement_rate", "students"]])
    if feature_rates:
        patterns = pd.concat(feature_rates, ignore_index=True)
        st.plotly_chart(style_chart(px.line(patterns, x="value", y="placement_rate", color="feature", markers=True, hover_data=["students"], title="Placement rate across experience indicators")), width="stretch")
    available_features = [feature for feature in comparison_features if feature in dataframe and pd.api.types.is_numeric_dtype(dataframe[feature])]
    if "placement_status" in dataframe and available_features:
        selected_feature = st.selectbox("Compare a feature by placement status", available_features, format_func=lambda value: value.replace("_", " ").title())
        comparison = dataframe.groupby("placement_status", as_index=False)[selected_feature].mean()
        st.plotly_chart(style_chart(px.bar(comparison, x="placement_status", y=selected_feature, color="placement_status", text_auto=".2f", title=f"Average {selected_feature.replace('_', ' ').title()} by placement status")), width="stretch")
    else:
        st.info("Feature comparisons require placement_status and at least one supported numeric feature.")


def pattern_page(dataframe: pd.DataFrame) -> None:
    page_intro("Relationships", "Correlation & Patterns", "Identify associations across academic, skills, experience, and compensation signals.")
    numeric_columns = dynamic_numeric_columns(dataframe)
    if len(numeric_columns) < 2:
        st.info("Correlation analysis requires at least two numeric columns.")
        return
    correlation = dataframe[numeric_columns].corr()
    st.plotly_chart(style_chart(px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature correlation heatmap")), width="stretch")
    if "cgpa" in correlation:
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
                '<div style="color:#475569;">Estimated probability of placement</div></div>',
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
    with st.container(key="model-performance-kpis"):
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
    filtered = dataframe
    if "department" in dataframe:
        departments = sorted(dataframe["department"].dropna().astype(str).unique())
        selected_departments = filter_left.multiselect("Department", departments, default=departments)
        filtered = filtered[filtered["department"].astype(str).isin(selected_departments)]
    else:
        filter_left.caption("Department filter unavailable")
    if "placement_status" in dataframe:
        statuses = sorted(dataframe["placement_status"].dropna().astype(str).unique())
        selected_statuses = filter_middle.multiselect("Placement status", statuses, default=statuses)
        filtered = filtered[filtered["placement_status"].astype(str).isin(selected_statuses)]
    else:
        filter_middle.caption("Placement filter unavailable")
    if "student_id" in dataframe:
        search = filter_right.text_input("Search student ID", placeholder="e.g. STU00042")
        if search:
            filtered = filtered[filtered["student_id"].astype(str).str.contains(search, case=False, na=False)]
    else:
        filter_right.caption("ID search unavailable")
    st.caption(f"Showing {len(filtered):,} of {len(dataframe):,} records")
    st.dataframe(filtered, width="stretch", hide_index=True)


def dataset_comparison_page(project_dataframe: pd.DataFrame, uploaded_dataframe: pd.DataFrame, upload_id: str | None) -> None:
    st.subheader("Project vs uploaded dataset")
    st.caption("Compare dataset size, data quality, shared columns, and the distribution of a shared numeric feature.")

    shared_columns = [column for column in project_dataframe.columns if column in uploaded_dataframe.columns]
    summary = pd.DataFrame([
        {
            "Dataset": "Project dataset",
            "Records": len(project_dataframe),
            "Columns": len(project_dataframe.columns),
            "Shared columns": len(shared_columns),
            "Missing cells": int(project_dataframe.isna().sum().sum()),
            "Duplicate rows": int(project_dataframe.duplicated().sum()),
        },
        {
            "Dataset": "Uploaded dataset",
            "Records": len(uploaded_dataframe),
            "Columns": len(uploaded_dataframe.columns),
            "Shared columns": len(shared_columns),
            "Missing cells": int(uploaded_dataframe.isna().sum().sum()),
            "Duplicate rows": int(uploaded_dataframe.duplicated().sum()),
        },
    ])
    st.dataframe(summary, width="stretch", hide_index=True)

    project_only = [column for column in project_dataframe.columns if column not in uploaded_dataframe.columns]
    upload_only = [column for column in uploaded_dataframe.columns if column not in project_dataframe.columns]
    with st.expander("Compare dataset columns"):
        st.write("**Shared columns:** " + (", ".join(map(str, shared_columns)) if shared_columns else "None"))
        st.write("**Only in project dataset:** " + (", ".join(map(str, project_only)) if project_only else "None"))
        st.write("**Only in uploaded dataset:** " + (", ".join(map(str, upload_only)) if upload_only else "None"))

    numeric_shared_columns = []
    for column in shared_columns:
        project_values = pd.to_numeric(project_dataframe[column], errors="coerce")
        uploaded_values = pd.to_numeric(uploaded_dataframe[column], errors="coerce")
        if project_values.notna().any() and uploaded_values.notna().any():
            numeric_shared_columns.append(column)

    if not numeric_shared_columns:
        st.info("The datasets have no shared numeric columns to compare as distributions.")
        return

    feature = st.selectbox(
        "Shared numeric feature",
        numeric_shared_columns,
        format_func=lambda value: str(value).replace("_", " ").title(),
        key=f"dataset_comparison_feature_{upload_id or 'current'}",
    )
    comparison = pd.concat(
        [
            pd.DataFrame({
                "Dataset": "Project dataset",
                feature: pd.to_numeric(project_dataframe[feature], errors="coerce"),
            }),
            pd.DataFrame({
                "Dataset": "Uploaded dataset",
                feature: pd.to_numeric(uploaded_dataframe[feature], errors="coerce"),
            }),
        ],
        ignore_index=True,
    ).dropna(subset=[feature])
    figure = px.histogram(
        comparison,
        x=feature,
        color="Dataset",
        histnorm="percent",
        barmode="overlay",
        opacity=.65,
        nbins=25,
        labels={feature: str(feature).replace("_", " ").title(), "percent": "Records (%)"},
        title=f"{str(feature).replace('_', ' ').title()} distribution comparison",
    )
    figure.update_layout(yaxis_title="Records (%)")
    st.plotly_chart(style_chart(figure), width="stretch", key="project_upload_comparison")


def upload_page(project_dataframe: pd.DataFrame, dataframe: pd.DataFrame | None, upload_name: str | None) -> None:
    page_intro("Flexible ingestion", "Dataset Upload", "Upload a CSV to save it to MongoDB, analyze it, and automatically train compatible models.")
    uploaded_file = st.file_uploader("Upload student dataset (CSV)", type=["csv"], help="Excel and JSON support can be added later without changing the analysis layer.")
    upload_id = st.session_state.get("uploaded_id")
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            dataframe = load_uploaded_csv(file_bytes)
            upload_name = uploaded_file.name
            upload_id = hashlib.sha256(file_bytes).hexdigest()
            st.session_state["uploaded_dataframe"] = dataframe
            st.session_state["uploaded_name"] = upload_name
            st.session_state["uploaded_size"] = uploaded_file.size
            st.session_state["uploaded_id"] = upload_id
        except ValueError as exc:
            st.error(str(exc))
            return

    if dataframe is None:
        st.info("Upload a CSV to begin. The preloaded project dataset remains available from the Dataset selector.")
        return

    profile = dataset_profile(dataframe, uploaded_file.size if uploaded_file else st.session_state.get("uploaded_size", 0))
    validation = validate_uploaded_dataset(dataframe)
    st.success(f"Analyzing uploaded dataset: {upload_name or 'uploaded CSV'}")
    if upload_id:
        processed_uploads = st.session_state.setdefault("uploaded_processing_results", {})
        if upload_id not in processed_uploads:
            dataset_saved = False
            try:
                with st.spinner("Saving uploaded records to MongoDB..."):
                    database_result = persist_uploaded_dataset(dataframe, upload_id, upload_name or "uploaded.csv")
                dataset_saved = True
                if validation["ml_compatible"]:
                    with st.spinner("Training and evaluating models on the uploaded dataset..."):
                        training_result = train_uploaded_models(
                            dataframe,
                            MODEL_DIR / "uploads" / upload_id,
                        )
                    save_uploaded_training_result(upload_id, "trained", result=training_result)
                    processed_uploads[upload_id] = {
                        "database": database_result,
                        "training_status": "trained",
                        "training_result": training_result,
                    }
                else:
                    reason = (
                        "Machine-learning dependencies are unavailable: "
                        f"{validation['ml_dependency_error']}"
                        if validation["ml_dependency_error"]
                        else "Missing required model columns: " + ", ".join(validation["ml_missing_columns"])
                    )
                    save_uploaded_training_result(upload_id, "skipped", error=reason)
                    processed_uploads[upload_id] = {
                        "database": database_result,
                        "training_status": "skipped",
                        "training_error": reason,
                    }
            except (RuntimeError, ImportError, ValueError, KeyError, OSError) as exc:
                if dataset_saved:
                    try:
                        save_uploaded_training_result(upload_id, "failed", error=str(exc))
                    except RuntimeError as database_exc:
                        st.error(f"Model training failed: {exc}. Could not record its status in MongoDB: {database_exc}")
                    else:
                        st.error(f"Model training failed after the dataset was saved: {exc}")
                else:
                    st.error(str(exc))

        processing_result = processed_uploads.get(upload_id)
        if processing_result:
            database_result = processing_result["database"]
            st.success(
                f"Saved {database_result['rows_saved']:,} records to "
                f"{database_result['database']}.{database_result['collection']}."
            )
            if processing_result["training_status"] == "trained":
                training_result = processing_result["training_result"]
                st.success(f"Automatic training complete. Best model: {training_result['best_model']}.")
                st.caption(f"Selected model saved to {training_result['model_path']}")
                model_rows = [
                    {"Model": name.replace("_", " ").title(), **{
                        key: values[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                    }}
                    for name, values in training_result["models"].items()
                ]
                st.dataframe(
                    pd.DataFrame(model_rows).style.format(
                        {key: "{:.4f}" for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning(f"Dataset saved, but automatic model training was skipped: {processing_result['training_error']}")

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

    dataset_comparison_page(project_dataframe, dataframe, upload_id)

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

    st.subheader("Uploaded-dataset model training")
    if not validation["ml_compatible"]:
        if validation["ml_dependency_error"]:
            st.warning("Uploaded-dataset model training is unavailable because machine-learning dependencies could not be loaded.")
            st.caption(f"ML import error: {validation['ml_dependency_error']}")
        else:
            st.info("Model training is unavailable for this upload because the exact Phase 3 feature schema is incomplete.")
            st.caption("Missing ML columns: " + ", ".join(validation["ml_missing_columns"]))
    else:
        st.info("Compatible datasets are trained automatically when uploaded.")


def reset_uploaded_dataset() -> None:
    st.session_state.pop("uploaded_dataframe", None)
    st.session_state.pop("uploaded_name", None)
    st.session_state.pop("uploaded_size", None)
    st.session_state.pop("uploaded_id", None)
    st.session_state.pop("uploaded_processing_results", None)
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
        if dataset_mode == "Upload New Dataset":
            uploaded_dataframe = st.session_state.get("uploaded_dataframe")
            uploaded_name = st.session_state.get("uploaded_name", "uploaded CSV")
            if uploaded_dataframe is None:
                st.caption("Select Dataset Upload to choose a CSV.")
            else:
                st.caption(f"Active dataset: {uploaded_name} • {len(uploaded_dataframe):,} records")
        else:
            st.caption(f"Active dataset: project data • {len(dataframe):,} records")
    st.markdown("## MongoDB-Based Big Data Analytics System")
    st.caption("Student performance and placement prediction | Academic project demonstration")
    if predict_placement is None:
        st.warning("Machine-learning dependencies are unavailable in this environment. Dataset statistics and analytics remain available; placement prediction and uploaded model training are disabled.")
    if metadata_error:
        st.warning(f"Saved model metrics are unavailable: {metadata_error}")
    if dataset_mode == "Upload New Dataset":
        uploaded_dataframe = st.session_state.get("uploaded_dataframe")
        if uploaded_dataframe is None or page == "Dataset Upload":
            upload_page(dataframe, uploaded_dataframe, st.session_state.get("uploaded_name"))
            return
        dataframe = uploaded_dataframe
    if page == "Dataset Upload":
        upload_page(dataframe, st.session_state.get("uploaded_dataframe"), st.session_state.get("uploaded_name"))
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
        if dataset_mode == "Upload New Dataset":
            st.info("The prediction form uses the saved model trained on the project dataset; uploading a CSV does not retrain or replace that model.")
        prediction_page()
    elif page == "Model Performance":
        if dataset_mode == "Upload New Dataset":
            st.info("These saved model metrics are from training on the project dataset, not the selected upload. Uploaded data can be evaluated from the Dataset Upload page when it has the required ML schema.")
        model_page(metadata)
    else:
        data_page(dataframe)


if __name__ == "__main__":
    main()
