"""
Early Academic Risk Prediction and Student Support System
Page 2: Batch Student Risk Prediction (app/pages/2_Batch_Prediction.py)

This page allows educators to upload a cohort CSV file containing student records
available on or before Day 30 of the course, run academic risk predictions for all
records in memory, inspect and filter results by operational priority, and export
the findings for institutional support workflows.
"""

import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Ensure Project Root is in sys.path
# When Streamlit runs multi-page apps, the script directory (app/pages or app)
# is placed in sys.path, but the project root containing 'src' is not. Adding it
# ensures 'src.predict' can always be imported cleanly and reliably.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the shared prediction module (reusing verified pipeline logic)
from src.predict import predict_student_risk

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Batch Risk Prediction | Student Support",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clear, professional card containers and status badges
st.markdown("""
<style>
    .page-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .guidance-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .result-card-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 6px solid #EF4444;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .result-card-watch {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 6px solid #F59E0B;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .result-card-low {
        background-color: #F0FDF4;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #10B981;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .ethical-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 6px solid #2563EB;
        border-radius: 8px;
        padding: 18px;
        margin-top: 25px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Sidebar Metadata & System Overview
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=70
    )
    st.title("System Overview")
    st.markdown("**Early Academic Risk Prediction & Student Support System**")
    st.caption("Batch Cohort Screening Module")

    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown("""
    * **Algorithm:** Logistic Regression (Scikit-Learn)
    * **Decision Threshold:** `0.40`
    * **Prediction Cutoff:** Day 30 of Presentation
    * **Target:** Academic Risk (1 = Fail / Post-Day 30 Withdrawn)
    * **Dataset:** OULAD Distance Learning
    """)

    st.markdown("---")
    st.subheader("Classification Rules")
    st.markdown("""
    * **Binary Threshold (0.40):**
      * Probability $\\ge 0.40$: **At Risk**
      * Probability $< 0.40$: **Not At Risk**
    * **Operational Support Priority:**
      * **High Priority:** $P \\ge 0.50$
      * **Watchlist:** $0.35 \\le P < 0.50$
      * **Low Risk:** $P < 0.35$
    """)

    st.markdown("---")
    st.info("🔒 **Data Privacy:** Uploaded CSVs are processed strictly in memory and are never saved to disk.")

# -----------------------------------------------------------------------------
# 3. Model Constants & Required Features
# -----------------------------------------------------------------------------
REQUIRED_CATEGORICAL = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]

REQUIRED_NUMERICAL = [
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "total_clicks",
    "active_days",
    "unique_sites",
    "pre_course_clicks",
    "early_assessments_submitted",
    "early_avg_score",
    "has_early_submission"
]

ALL_REQUIRED_COLUMNS = REQUIRED_CATEGORICAL + REQUIRED_NUMERICAL

# Target leakage columns (post-Day-30 course outcomes)
LEAKAGE_COLUMNS = [
    "final_result",
    "date_unregistration",
    "final_score",
    "passed",
    "course_outcome"
]

# Non-nullable mandatory columns (engagement counts and course codes)
# Note: early_avg_score, imd_band, and date_registration may legitimately contain missing values
STRICT_MANDATORY_COLS = [
    "code_module",
    "code_presentation",
    "gender",
    "highest_education",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks",
    "active_days",
    "unique_sites",
    "pre_course_clicks",
    "early_assessments_submitted",
    "has_early_submission"
]

# -----------------------------------------------------------------------------
# 4. Helper: CSV Validation Function
# -----------------------------------------------------------------------------
def validate_batch_dataframe(df: pd.DataFrame):
    """
    Validate uploaded DataFrame for batch risk prediction.
    
    Returns:
        tuple: (is_valid: bool, errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    # 1. Empty DataFrame check
    if df.empty or len(df.columns) == 0:
        errors.append("The uploaded CSV file is empty. Please upload a file containing student records.")
        return False, errors, warnings

    if len(df) == 0:
        errors.append("The uploaded CSV contains column headers but zero student data records.")
        return False, errors, warnings

    # 2. Target leakage check (Reject post-Day-30 outcome columns)
    found_leakage = [col for col in df.columns if col.lower() in LEAKAGE_COLUMNS]
    if found_leakage:
        errors.append(
            f"Target leakage detected: The uploaded file contains post-cutoff outcome column(s): {found_leakage}. "
            f"To preserve early-warning integrity, please remove final course outcomes and re-upload."
        )

    # 3. Required columns check
    missing_cols = [col for col in ALL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required column(s) ({len(missing_cols)}): {missing_cols}. "
            f"All 18 model features must be present in the CSV file."
        )

    # Return early if fundamental structure is missing
    if errors:
        return False, errors, warnings

    # 4. Numerical columns validation (Check for non-numeric unparseable strings)
    for col in REQUIRED_NUMERICAL:
        non_null_series = df[col].dropna()
        # Coerce to numeric and find any non-null strings that fail conversion
        coerced = pd.to_numeric(non_null_series, errors="coerce")
        bad_mask = coerced.isna()
        if bad_mask.any():
            bad_samples = non_null_series[bad_mask].head(3).tolist()
            errors.append(
                f"Column '{col}' contains invalid non-numeric values: {bad_samples}. Please ensure this column contains numbers."
            )

    # 5. Check for missing values in strictly mandatory columns
    for col in STRICT_MANDATORY_COLS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(
                f"Column '{col}' contains {null_count} missing (blank/NaN) value(s). "
                f"This engagement feature must be populated for all student records."
            )

    # 6. Check for duplicate student IDs (Informative warning, not an error)
    if "id_student" in df.columns:
        duplicate_count = df["id_student"].duplicated().sum()
        if duplicate_count > 0:
            warnings.append(
                f"Found {duplicate_count} duplicate student ID(s). In distance education, students frequently "
                f"enroll in multiple courses or presentations simultaneously; all {len(df)} enrollment records are preserved."
            )

    # 7. Check if target column exists (Informative note)
    if "target" in df.columns:
        warnings.append(
            "Column 'target' was detected in the file. It will be excluded from the model input "
            "and will not influence predictions."
        )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


# -----------------------------------------------------------------------------
# 5. Page Header
# -----------------------------------------------------------------------------
st.markdown('<div class="page-title">📁 Batch Student Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">'
    'Upload a cohort CSV of Day 1–30 student records to screen and triage academic risk at scale.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 6. Guidance & Format Requirements Expander
# -----------------------------------------------------------------------------
with st.expander("📋 CSV Requirements, Input Format & Sample Download", expanded=False):
    st.markdown("""
    The uploaded CSV must contain student data available on or before **Day 30** of the course presentation.
    
    **Required Features (18 Model Inputs):**
    * **Student Demographics (8):** `code_module`, `code_presentation`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`
    * **Academic History (3):** `num_of_prev_attempts`, `studied_credits`, `date_registration` (days relative to start, e.g. -30)
    * **Early VLE Engagement (4):** `total_clicks`, `active_days`, `unique_sites`, `pre_course_clicks`
    * **Early Assessments (3):** `early_assessments_submitted`, `early_avg_score`, `has_early_submission` (0 or 1)
    
    **Important Notes:**
    * `id_student`: Optional but recommended. If provided, it is preserved to identify results in tables and downloads.
    * `early_avg_score`: May be left blank/NaN for students with 0 early assessment submissions. The pipeline will impute missing scores automatically.
    * `target` & Outcome Columns: Any post-cutoff outcome columns (such as `final_result`) will be rejected to prevent target leakage.
    """)

    # Provide a sample template download button
    sample_file_path = PROJECT_ROOT / "scratch" / "sample_batch_students.csv"
    if sample_file_path.exists():
        sample_csv_bytes = sample_file_path.read_bytes()
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=sample_csv_bytes,
            file_name="sample_batch_students.csv",
            mime="text/csv",
            help="Download a synthetic 12-student CSV template to test batch prediction."
        )

# -----------------------------------------------------------------------------
# 7. File Uploader
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a student cohort CSV file:",
    type=["csv"],
    help="Select a CSV file containing student records with the 18 required features."
)

if uploaded_file is not None:
    # Build unique file identifier to manage session state cleanly
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Reset batch results if a different file is uploaded
    if st.session_state.get("active_batch_file_id") != file_id:
        st.session_state["active_batch_file_id"] = file_id
        st.session_state["batch_results"] = None

    # Load uploaded CSV in memory
    try:
        if uploaded_file.size == 0:
            st.error("❌ The uploaded CSV file is empty (0 bytes). Please upload a file containing student records.")
            st.stop()
        df_uploaded = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded CSV file is empty. Please upload a file containing student records.")
        st.stop()
    except Exception as read_err:
        st.error(f"❌ Error parsing CSV file: {read_err}. Please ensure the file is a standard, uncorrupted CSV.")
        st.stop()

    # Validate uploaded DataFrame
    is_valid, validation_errors, validation_warnings = validate_batch_dataframe(df_uploaded)

    # Display warnings if any
    for warn_msg in validation_warnings:
        st.info(f"ℹ️ {warn_msg}")

    # Display validation errors and stop if invalid
    if not is_valid:
        st.error("❌ CSV Validation Failed. Please address the following issue(s):")
        for err_msg in validation_errors:
            st.error(f"• {err_msg}")
        st.stop()

    st.success(f"✅ CSV Validation Passed: Found {len(df_uploaded):,} student records with all 18 required features.")

    # Show data preview
    with st.expander(f"🔍 Preview Uploaded Data ({len(df_uploaded)} records)", expanded=True):
        st.dataframe(df_uploaded.head(5), use_container_width=True)

    # -------------------------------------------------------------------------
    # 8. Prediction Execution Trigger
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">⚡ Run Cohort Risk Evaluation</div>', unsafe_allow_html=True)
    st.caption("Click below to evaluate risk probabilities and assign operational priority categories.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_prediction = st.button("🚀 Run Batch Prediction", type="primary")

    if run_prediction:
        with st.spinner("Processing cohort predictions with Logistic Regression pipeline..."):
            try:
                results_df = predict_student_risk(df_uploaded)
                st.session_state["batch_results"] = results_df
                st.success("🎉 Batch prediction complete! Results are displayed below.")
            except Exception as pred_err:
                st.error(f"Prediction failed: {pred_err}")
                st.stop()

# -----------------------------------------------------------------------------
# 9. Results Dashboard & Priority Filtering
# -----------------------------------------------------------------------------
if st.session_state.get("batch_results") is not None:
    results_df = st.session_state["batch_results"]

    st.markdown('<div class="section-title">📊 Cohort Risk Summary & Triage Dashboard</div>', unsafe_allow_html=True)

    # Compute key cohort summary metrics
    total_records = len(results_df)
    high_count = int((results_df["priority"] == "High Priority").sum())
    watch_count = int((results_df["priority"] == "Watchlist").sum())
    low_count = int((results_df["priority"] == "Low Risk").sum())

    # Summary KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(
            label="Total Students",
            value=f"{total_records:,}"
        )
    with kpi_col2:
        st.metric(
            label="🔴 High Priority",
            value=f"{high_count} ({high_count / total_records * 100:.1f}%)",
            help="Probability >= 0.50. Urgent outreach recommended."
        )
    with kpi_col3:
        st.metric(
            label="🟡 Watchlist",
            value=f"{watch_count} ({watch_count / total_records * 100:.1f}%)",
            help="0.35 <= Probability < 0.50. Proactive check-in recommended."
        )
    with kpi_col4:
        st.metric(
            label="🟢 Low Risk",
            value=f"{low_count} ({low_count / total_records * 100:.1f}%)",
            help="Probability < 0.35. Standard course communications."
        )

    # -------------------------------------------------------------------------
    # Priority Filter & Table Display
    # -------------------------------------------------------------------------
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        selected_priority = st.selectbox(
            "Filter by Operational Priority:",
            options=["All Students", "High Priority", "Watchlist", "Low Risk"],
            index=0,
            help="Filter the student cohort table by priority tier."
        )

    # Apply priority filter
    if selected_priority == "All Students":
        filtered_df = results_df.copy()
    else:
        filtered_df = results_df[results_df["priority"] == selected_priority].copy()

    st.caption(f"Displaying **{len(filtered_df)}** of **{total_records}** student records.")

    # Prepare user-friendly display table
    display_df = filtered_df.copy()
    display_df["Risk Probability"] = display_df["risk_probability"].map(lambda p: f"{p * 100:.1f}%")
    display_df["Binary Status"] = display_df["prediction"]
    display_df["Priority Tier"] = display_df["priority"]

    # Select and order columns for display clarity
    display_columns = []
    if "id_student" in display_df.columns:
        display_columns.append("id_student")
    
    display_columns.extend([
        "code_module",
        "code_presentation",
        "Risk Probability",
        "Binary Status",
        "Priority Tier",
        "active_days",
        "total_clicks",
        "early_assessments_submitted",
        "early_avg_score",
        "highest_education"
    ])

    # Show interactive data table
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # 10. Download Results
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">📥 Export Prediction Results</div>', unsafe_allow_html=True)
    st.caption("Download the prediction findings as a CSV file to share with student support advisors.")

    # Prepare export DataFrame preserving original row order
    export_df = results_df.copy()
    
    # Place identifiers and prediction outputs first
    front_cols = []
    if "id_student" in export_df.columns:
        front_cols.append("id_student")
    front_cols.extend(["risk_probability", "prediction", "priority"])
    
    other_cols = [c for c in export_df.columns if c not in front_cols]
    export_df = export_df[front_cols + other_cols]

    # Convert to CSV buffer
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        st.download_button(
            label="📥 Download Full Batch Predictions (All Records)",
            data=csv_bytes,
            file_name="student_risk_predictions_all.csv",
            mime="text/csv",
            help="Download all processed student records in their original row order."
        )

    # Also allow downloading only the filtered cohort
    if selected_priority != "All Students":
        filtered_export = filtered_df[front_cols + other_cols]
        filtered_buffer = io.StringIO()
        filtered_export.to_csv(filtered_buffer, index=False)
        filtered_bytes = filtered_buffer.getvalue().encode("utf-8")
        
        with dl_col2:
            st.download_button(
                label=f"📥 Download {selected_priority} Records Only",
                data=filtered_bytes,
                file_name=f"student_risk_{selected_priority.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                help=f"Download only the {len(filtered_df)} students currently triaged as {selected_priority}."
            )

# -----------------------------------------------------------------------------
# 11. Responsible Use & Ethical Guidance
# -----------------------------------------------------------------------------
st.markdown("""
<div class="ethical-card">
    <div style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;">
        ⚖️ Responsible Use & Ethical Framework for Educators
    </div>
    <ul style="margin-bottom: 0; padding-left: 20px; color: #1F2937; line-height: 1.6;">
        <li><strong>Early-Warning Indicators, Not Definite Outcomes:</strong> Model predictions reflect statistical associations identified in historical cohorts. They are early indicators of potential disengagement, not predetermined student outcomes or measures of student capability.</li>
        <li><strong>Human Context First:</strong> Model predictions should always be reviewed alongside direct student communication, attendance notes, and personal circumstances before taking outreach actions.</li>
        <li><strong>Operational Triage Categories:</strong> The High Priority, Watchlist, and Low Risk categories are proposed operational triage categories designed to help manage outreach workloads, not institutionally validated student tracks.</li>
        <li><strong>Supportive, Non-Punitive Intent:</strong> This early warning system is designed solely to connect students with academic resources, tutoring, and advisor support. It must never be used to penalize students, restrict course access, or justify automated decisions.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
