"""
Early Academic Risk Prediction & Student Support System
Streamlit Home Dashboard (app/Home.py)

This module provides the main landing dashboard for the portfolio project,
summarizing project objectives, historical dataset statistics, verified
model performance metrics, operational priority workflows, and ethical use guidelines.
"""

import sys
from pathlib import Path
import streamlit as st

# -----------------------------------------------------------------------------
# Ensure Project Root is in sys.path
# When Streamlit runs 'app/Home.py', Python sets sys.path[0] to 'app/'. Adding
# the project root ensures 'src' and modules can be imported across all pages.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Risk Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for subtle, professional KPI cards and badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.3rem;
    }
    .workflow-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        height: 100%;
    }
    .workflow-step {
        font-weight: 700;
        color: #2563EB;
        font-size: 1.1rem;
        margin-bottom: 6px;
    }
    .workflow-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.4;
    }
    .priority-card-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .priority-card-watch {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .priority-card-low {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: System Metadata & Context
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=70
    )
    st.title("System Overview")
    st.markdown("**Early Academic Risk Prediction & Student Support System**")
    st.caption("Machine Learning Portfolio Project")

    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown("""
    * **Algorithm:** Logistic Regression (Scikit-Learn)
    * **Decision Threshold:** `0.40`
    * **Prediction Cutoff:** Day 30 of Presentation
    * **Target:** Academic Risk (1 = Fail / Post-Day 30 Withdrawn)
    * **Dataset:** Open University Learning Analytics Dataset (OULAD)
    * **Validation:** 5-Fold GroupKFold (`id_student`)
    """)

    st.markdown("---")
    st.subheader("How Educators Use This")
    st.markdown("""
    1. **Early Triage:** Identify students showing slow initial engagement by Week 4.
    2. **Targeted Support:** Reach out with tutoring resources, study skills, or advising.
    3. **Human-in-the-Loop:** Combine model risk probabilities with direct classroom observations.
    """)

    st.markdown("---")
    st.caption("Version 1.0 | Streamlit 1.65.0 | Python 3.10")

# -----------------------------------------------------------------------------
# Section 1: Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎓 Early Academic Risk Prediction & Student Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An ML-powered early warning system designed to help educators identify students who may need additional academic support.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Section 2: Project Overview
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">📌 Project Overview</div>', unsafe_allow_html=True)

st.markdown("""
In higher education distance learning, many students who encounter academic difficulties or drop out do so without early educator awareness. 
This system uses **student demographic information, previous academic history, and early virtual learning engagement data** 
to estimate the probability of academic risk early in the academic presentation.

* **Day 30 Early Warning Cutoff:** All prediction features are strictly calculated from information available on or before **Day 30** (approximately Week 4 of a multi-month course). This allows educators to intervene weeks before major mid-term assessments or final exams.
* **Decision Support, Not Decision Automation:** The system is built to assist and inform educators—**not to replace human judgement**, automate academic evaluations, or penalize students.
""")

# -----------------------------------------------------------------------------
# Section 3: Key Project Statistics (Historical Modeling Cohort)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">📊 Historical Cohort Statistics</div>', unsafe_allow_html=True)
st.caption("These statistics describe the historical OULAD modeling cohort (active enrollments at Day 30), not live institutional students.")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(
        label="Eligible Enrollments",
        value="27,373",
        help="Total student-module enrollments active at Day 30 across 7 courses and 4 presentation years."
    )

with kpi_col2:
    st.metric(
        label="Not At Risk (Pass / Distinction)",
        value="15,385",
        delta="56.21%",
        delta_color="normal",
        help="Students who successfully completed their course with a Pass or Distinction."
    )

with kpi_col3:
    st.metric(
        label="At Risk (Fail / Withdrawn)",
        value="11,988",
        delta="-43.79%",
        delta_color="inverse",
        help="Students who subsequently failed or withdrew after the Day 30 cutoff."
    )

with kpi_col4:
    st.metric(
        label="Model Input Features",
        value="18",
        help="8 demographic attributes and 10 engagement/academic features observed on or before Day 30."
    )

# -----------------------------------------------------------------------------
# Section 4: Model Performance
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">🎯 Model Performance on Held-Out Test Set</div>', unsafe_allow_html=True)
st.caption("Verified test-set evaluation results on 5,462 held-out enrollments (4,942 completely unseen students) at the selected 0.40 threshold.")

perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)

with perf_col1:
    st.metric(
        label="Recall (Sensitivity)",
        value="73.41%",
        help="Percentage of actual at-risk enrollments correctly caught by the model at Day 30 (1,739 of 2,369)."
    )

with perf_col2:
    st.metric(
        label="Precision",
        value="63.14%",
        help="Percentage of flagged alerts that are genuinely at risk (1,739 of 2,754 alerts)."
    )

with perf_col3:
    st.metric(
        label="F1-Score",
        value="0.6789",
        help="Harmonic mean of Recall and Precision, balancing risk detection with educator alert workload."
    )

with perf_col4:
    st.metric(
        label="ROC-AUC Score",
        value="0.7745",
        help="Area under the Receiver Operating Characteristic curve, measuring overall ranking capability."
    )

with perf_col5:
    st.metric(
        label="PR-AUC Score",
        value="0.7413",
        help="Area under the Precision-Recall curve, substantially outperforming the 0.43 no-skill baseline."
    )

with st.expander("ℹ️ Understanding These Evaluation Metrics"):
    st.markdown("""
    * **Recall (73.41%):** In an educational support setting, missing a struggling student has a high educational cost. At threshold `0.40`, the model catches **nearly 3 out of every 4 struggling enrollments**, compared to only 58.7% at the default 0.50 cutoff.
    * **Precision (63.14%):** When the model generates an alert, approximately **63%** of flagged enrollments actually need support. The remaining 37% ("false alarms") represent students who exhibited slow initial engagement but recovered. In practice, a supportive check-in to a passing student is low-risk and constructive.
    * **Selected Binary Classification Threshold (0.40):** We selected `0.40` strictly using out-of-fold training predictions to achieve an optimal balance between catching at-risk students and avoiding educator alert fatigue.
    """)

# -----------------------------------------------------------------------------
# Section 5: How the System Works (Visual Workflow)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">⚙️ How the System Works</div>', unsafe_allow_html=True)

step_col1, step_col2, step_col3, step_col4, step_col5 = st.columns(5)

with step_col1:
    st.markdown("""
    <div class="workflow-card">
        <div class="workflow-step">1. Student Data</div>
        <div class="workflow-desc">
            Demographic background, course registration, and Day 1–30 VLE interactions.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step_col2:
    st.markdown("""
    <div class="workflow-card">
        <div class="workflow-step">2. Preprocessing</div>
        <div class="workflow-desc">
            Missing value imputation, standard scaling, and one-hot encoding in a unified pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step_col3:
    st.markdown("""
    <div class="workflow-card">
        <div class="workflow-step">3. ML Model</div>
        <div class="workflow-desc">
            Logistic Regression evaluates weighted associations across 18 features.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step_col4:
    st.markdown("""
    <div class="workflow-card">
        <div class="workflow-step">4. Risk Probability</div>
        <div class="workflow-desc">
            Continuous risk probability P(At Risk) computed between 0.00 and 1.00.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step_col5:
    st.markdown("""
    <div class="workflow-card">
        <div class="workflow-step">5. Support Priority</div>
        <div class="workflow-desc">
            Threshold (0.40) determines At Risk status; triaged into 3 operational priority tiers.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 6: Operational Priority Categories
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">🚦 Operational Support Priority Categories</div>', unsafe_allow_html=True)

st.markdown("""
To help teachers triage outreach workloads effectively, predicted probabilities are mapped into **three proposed operational support categories**. 
These tiers provide actionable guidance for scheduling advisor check-ins:
""")

prio_col1, prio_col2, prio_col3 = st.columns(3)

with prio_col1:
    st.markdown("""
    <div class="priority-card-high">
        <strong style="color: #B91C1C; font-size: 1.1rem;">🔴 High Priority</strong><br/>
        <span style="font-size: 0.95rem; font-weight: 600;">Predicted Probability &ge; 0.50</span>
        <p style="margin-top: 8px; font-size: 0.88rem; color: #4B5563;">
            Strong early indicators of risk (low VLE clicks, missing early submissions). 
            Recommended for direct advisor outreach, academic coaching, or tutoring referral.
        </p>
    </div>
    """, unsafe_allow_html=True)

with prio_col2:
    st.markdown("""
    <div class="priority-card-watch">
        <strong style="color: #D97706; font-size: 1.1rem;">🟡 Watchlist / Moderate</strong><br/>
        <span style="font-size: 0.95rem; font-weight: 600;">0.35 &le; Predicted Probability &lt; 0.50</span>
        <p style="margin-top: 8px; font-size: 0.88rem; color: #4B5563;">
            Moderate or borderline early engagement. Recommended for passive monitoring, 
            automated reminder emails, and tracking the next assignment submission.
        </p>
    </div>
    """, unsafe_allow_html=True)

with prio_col3:
    st.markdown("""
    <div class="priority-card-low">
        <strong style="color: #047857; font-size: 1.1rem;">🟢 Low Risk</strong><br/>
        <span style="font-size: 0.95rem; font-weight: 600;">Predicted Probability &lt; 0.35</span>
        <p style="margin-top: 8px; font-size: 0.88rem; color: #4B5563;">
            Steady early clicks, frequent login days, and successful early submission. 
            Standard course communications with routine periodic monitoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "**Important Clarification:** These priority categories are proposed operational support tiers to guide educator time allocation, "
    "not institutionally validated risk guarantees. The model's formal binary classification threshold remains **0.40** "
    "(Probabilities $\\ge 0.40$ are classified as *At Risk*, while probabilities $< 0.40$ are classified as *Not At Risk*)."
)

# -----------------------------------------------------------------------------
# Section 7: Responsible Use & Ethical Disclaimer
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">⚖️ Responsible Use & Ethical Disclaimer</div>', unsafe_allow_html=True)

st.warning("""
**Disclaimer & Limitations:**
* **Prototype Research Tool:** This system is a research and portfolio prototype developed using historical data from the Open University Learning Analytics Dataset (OULAD). It has not been validated or calibrated for deployment in an active educational institution.
* **Statistical Associations, Not Causes:** Predictions represent statistical correlation observed in historical data, not confirmed causes, student potential, or predetermined outcomes. Demographic attributes reflect historical disparities, never individual capability.
* **Human-Centered Decision Making:** Model outputs should never be used to penalize, grade, reject, or discourage students. Educators must always review individual circumstances, communicate directly with students, and exercise professional judgement before deciding on support interventions.
""")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>"
    "Early Academic Risk Prediction & Student Support System &bull; Machine Learning Portfolio Project &bull; Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
