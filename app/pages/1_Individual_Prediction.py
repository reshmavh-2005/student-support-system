"""
Early Academic Risk Prediction and Student Support System
Page 1: Individual Student Risk Prediction (app/pages/1_Individual_Prediction.py)

This page allows educators to enter student information available on or before
Day 30 of the course, and generates:
1. Academic Risk Probability
2. Binary Risk Prediction (Decision Threshold: 0.40)
3. Operational Support Priority (High Priority, Watchlist, Low Risk)
4. Tailored, actionable support suggestions for teachers.
"""

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

# Import the shared prediction module (reusing tested pipeline logic)
from src.predict import predict_student_risk

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Individual Risk Prediction | Student Support",
    page_icon="👤",
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
    .form-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .result-card-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 6px solid #EF4444;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .result-card-watch {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 6px solid #F59E0B;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .result-card-low {
        background-color: #F0FDF4;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #10B981;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .action-header {
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 10px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Helper: Load Dropdown Options from Processed Dataset
# -----------------------------------------------------------------------------
@st.cache_data
def load_dropdown_options():
    """
    Populate categorical selectbox options directly from the processed dataset.
    This guarantees options match the exact values seen during model training.
    """
    data_path = PROJECT_ROOT / "data" / "processed" / "student_model_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return {
            "code_module": sorted(df["code_module"].dropna().unique().tolist()),
            "code_presentation": sorted(df["code_presentation"].dropna().unique().tolist()),
            "gender": ["M", "F"],
            "region": sorted(df["region"].dropna().unique().tolist()),
            "highest_education": [
                "No Formal quals",
                "Lower Than A Level",
                "A Level or Equivalent",
                "HE Qualification",
                "Post Graduate Qualification"
            ],
            "imd_band": [
                "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
            ],
            "age_band": ["0-35", "35-55", "55<="],
            "disability": ["N", "Y"]
        }
    else:
        # Fallback options if dataset file is moved
        return {
            "code_module": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
            "code_presentation": ["2013J", "2013B", "2014J", "2014B"],
            "gender": ["M", "F"],
            "region": [
                "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
                "North Region", "North Western Region", "Scotland", "South East Region",
                "South Region", "South West Region", "Wales", "West Midlands Region",
                "Yorkshire Region"
            ],
            "highest_education": [
                "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                "HE Qualification", "Post Graduate Qualification"
            ],
            "imd_band": [
                "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
            ],
            "age_band": ["0-35", "35-55", "55<="],
            "disability": ["N", "Y"]
        }

options = load_dropdown_options()

# -----------------------------------------------------------------------------
# 3. Sidebar: Context & Guidelines
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 Individual Assessment")
    st.markdown("""
    This tool allows an educator to estimate the academic risk probability for an
    individual student enrollment based on early signals observed on or before **Day 30**.
    """)

    st.markdown("---")
    st.subheader("Threshold Reference")
    st.markdown("""
    * **Binary Threshold (0.40):**
      * Probability $\\ge 0.40$: **At Risk**
      * Probability $< 0.40$: **Not At Risk**
    * **Priority Tiers:**
      * **High Priority:** $P \\ge 0.50$
      * **Watchlist:** $0.35 \\le P < 0.50$
      * **Low Risk:** $P < 0.35$
    """)

    st.markdown("---")
    st.info("💡 **Tip:** All engagement metrics should reflect the cumulative activity up to Day 30 of the course.")

# -----------------------------------------------------------------------------
# 4. Page Header
# -----------------------------------------------------------------------------
st.markdown('<div class="page-title">👤 Individual Student Academic Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">'
    'Enter student information and Day 1–30 engagement metrics to generate a risk assessment and supportive guidance.'
    '</div>',
    unsafe_allow_html=True
)

# Optional Profile Preset Selector for Quick Demo Testing
preset_col1, preset_col2 = st.columns([2, 1])
with preset_col1:
    preset_choice = st.selectbox(
        "Load Example Student Profile (Optional):",
        options=[
            "Custom (Enter values manually below)",
            "Example 1: High Early Engagement (Low Risk Profile)",
            "Example 2: Moderate / Borderline Engagement (Watchlist Profile)",
            "Example 3: Disengaged / Slow Starter (High Priority Profile)"
        ],
        index=0,
        help="Select a representative profile to populate the form fields quickly for testing."
    )

# Set defaults based on preset choice
if preset_choice == "Example 1: High Early Engagement (Low Risk Profile)":
    def_module = "BBB"
    def_pres = "2013J"
    def_gender = "M"
    def_region = "South Region"
    def_edu = "A Level or Equivalent"
    def_imd = "60-70%"
    def_age = "0-35"
    def_dis = "N"
    def_prev = 0
    def_credits = 60
    def_reg = -45
    def_clicks = 450
    def_active = 18
    def_sites = 35
    def_pre_clicks = 60
    def_submitted = 1
    def_has_score = True
    def_score = 85.0
elif preset_choice == "Example 2: Moderate / Borderline Engagement (Watchlist Profile)":
    def_module = "BBB"
    def_pres = "2013J"
    def_gender = "M"
    def_region = "South Region"
    def_edu = "Lower Than A Level"
    def_imd = "20-30%"
    def_age = "0-35"
    def_dis = "N"
    def_prev = 0
    def_credits = 60
    def_reg = -30
    def_clicks = 160
    def_active = 10
    def_sites = 18
    def_pre_clicks = 15
    def_submitted = 1
    def_has_score = True
    def_score = 70.0
elif preset_choice == "Example 3: Disengaged / Slow Starter (High Priority Profile)":
    def_module = "BBB"
    def_pres = "2013J"
    def_gender = "F"
    def_region = "Scotland"
    def_edu = "Lower Than A Level"
    def_imd = "0-10%"
    def_age = "0-35"
    def_dis = "N"
    def_prev = 1
    def_credits = 90
    def_reg = -10
    def_clicks = 40
    def_active = 3
    def_sites = 7
    def_pre_clicks = 5
    def_submitted = 0
    def_has_score = False
    def_score = 0.0
else:
    # Default custom starting values
    def_module = "BBB"
    def_pres = "2013J"
    def_gender = "F"
    def_region = "South East Region"
    def_edu = "A Level or Equivalent"
    def_imd = "40-50%"
    def_age = "0-35"
    def_dis = "N"
    def_prev = 0
    def_credits = 60
    def_reg = -30
    def_clicks = 250
    def_active = 12
    def_sites = 24
    def_pre_clicks = 25
    def_submitted = 1
    def_has_score = True
    def_score = 75.0

# -----------------------------------------------------------------------------
# 5. Prediction Input Form
# -----------------------------------------------------------------------------
with st.form("individual_prediction_form"):

    # SECTION A: Student Demographics & Course Information
    st.markdown('<div class="form-header">1. Student Information & Course Registration</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        code_module = st.selectbox(
            "Course Module Code",
            options=options["code_module"],
            index=options["code_module"].index(def_module) if def_module in options["code_module"] else 0,
            help="Course subject area code (AAA through GGG)."
        )
        gender = st.selectbox(
            "Gender",
            options=options["gender"],
            index=options["gender"].index(def_gender),
            format_func=lambda x: "Female (F)" if x == "F" else "Male (M)",
            help="Student's declared gender."
        )

    with c2:
        code_presentation = st.selectbox(
            "Course Presentation",
            options=options["code_presentation"],
            index=options["code_presentation"].index(def_pres) if def_pres in options["code_presentation"] else 0,
            help="Academic cohort presentation (e.g., 2013J = October 2013 start; 2014B = February 2014 start)."
        )
        age_band = st.selectbox(
            "Age Band",
            options=options["age_band"],
            index=options["age_band"].index(def_age),
            help="Student age bracket at the start of the course."
        )

    with c3:
        region = st.selectbox(
            "Geographic Region",
            options=options["region"],
            index=options["region"].index(def_region) if def_region in options["region"] else 0,
            help="Geographic region of the student's residence."
        )
        highest_education = st.selectbox(
            "Highest Prior Education",
            options=options["highest_education"],
            index=options["highest_education"].index(def_edu) if def_edu in options["highest_education"] else 2,
            help="Highest qualification held by the student upon enrollment."
        )

    with c4:
        imd_band = st.selectbox(
            "Deprivation Band (IMD)",
            options=options["imd_band"],
            index=options["imd_band"].index(def_imd) if def_imd in options["imd_band"] else 4,
            help="Index of Multiple Deprivation decile (0-10% indicates highest socioeconomic deprivation)."
        )
        disability = st.selectbox(
            "Declared Disability",
            options=options["disability"],
            index=options["disability"].index(def_dis),
            format_func=lambda x: "Yes (Declared)" if x == "Y" else "No (None Declared)",
            help="Whether the student declared a recognized disability."
        )

    # SECTION B: Previous Academic Record
    st.markdown('<div class="form-header">2. Previous Academic History</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)

    with c5:
        num_of_prev_attempts = st.number_input(
            "Previous Attempts at This Course",
            min_value=0,
            max_value=6,
            value=int(def_prev),
            step=1,
            help="Number of times the student has previously enrolled in this module."
        )

    with c6:
        studied_credits = st.number_input(
            "Currently Studied Credits",
            min_value=30,
            max_value=360,
            value=int(def_credits),
            step=30,
            help="Total credit load currently undertaken by the student across all enrolled modules."
        )

    with c7:
        date_registration = st.number_input(
            "Registration Date (Days Relative to Start)",
            min_value=-310,
            max_value=30,
            value=int(def_reg),
            step=1,
            help="Days relative to course start date. Negative values indicate early enrollment prior to Day 0 (e.g., -30 = registered 30 days early)."
        )

    # SECTION C: Early Learning Engagement (Days 1–30)
    st.markdown('<div class="form-header">3. Early Online Learning Engagement (Days 1–30)</div>', unsafe_allow_html=True)
    c8, c9, c10, c11 = st.columns(4)

    with c8:
        active_days = st.number_input(
            "Active Days on Platform",
            min_value=0,
            max_value=31,
            value=int(def_active),
            step=1,
            help="Number of distinct days on or before Day 30 on which the student interacted with the virtual learning environment."
        )

    with c9:
        total_clicks = st.number_input(
            "Total Interactions (Clicks)",
            min_value=0,
            max_value=5000,
            value=int(def_clicks),
            step=10,
            help="Total number of recorded click interactions on course materials up to Day 30."
        )

    with c10:
        unique_sites = st.number_input(
            "Unique Course Resources Visited",
            min_value=0,
            max_value=200,
            value=int(def_sites),
            step=1,
            help="Number of distinct course activities, content pages, or forum threads viewed."
        )

    with c11:
        pre_course_clicks = st.number_input(
            "Pre-Course Clicks (Prior to Day 0)",
            min_value=0,
            max_value=2000,
            value=int(def_pre_clicks),
            step=5,
            help="Interactions recorded prior to the official start of the module."
        )

    # SECTION D: Early Assessment Information (On or Before Day 30)
    st.markdown('<div class="form-header">4. Early Assessment Results (On or Before Day 30)</div>', unsafe_allow_html=True)
    c12, c13, c14 = st.columns(3)

    with c12:
        early_assessments_submitted = st.number_input(
            "Early Assessments Submitted",
            min_value=0,
            max_value=6,
            value=int(def_submitted),
            step=1,
            help="Number of tutor-marked or computer-marked assessments submitted on or before Day 30."
        )

    with c13:
        # Determine submission flag and grade availability
        has_submission = 1 if early_assessments_submitted > 0 else 0
        if early_assessments_submitted > 0:
            score_available = st.checkbox(
                "Assessment has been graded",
                value=def_has_score,
                help="Uncheck if the assessment was submitted but has not yet been marked by the tutor."
            )
        else:
            score_available = False
            st.caption("ℹ️ *No submissions on or before Day 30.*")

    with c14:
        if early_assessments_submitted > 0 and score_available:
            early_avg_score = st.number_input(
                "Average Score (0–100)",
                min_value=0.0,
                max_value=100.0,
                value=float(def_score),
                step=1.0,
                help="Average mark on assignments submitted on or before Day 30."
            )
        else:
            early_avg_score = np.nan
            st.text_input(
                "Average Score (0–100)",
                value="Missing / No Graded Submission",
                disabled=True,
                help="Treated as missing (NaN). The pipeline's SimpleImputer will handle this automatically."
            )

    st.markdown("<br/>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔮 Predict Student Academic Risk",
        use_container_width=True,
        type="primary"
    )

# -----------------------------------------------------------------------------
# 6. Prediction Execution & Results Rendering
# -----------------------------------------------------------------------------
if submitted:
    # Construct a single-row DataFrame using the exact 18 feature names expected by the model
    input_data = pd.DataFrame([{
        "code_module": code_module,
        "code_presentation": code_presentation,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "disability": disability,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "date_registration": float(date_registration),
        "total_clicks": float(total_clicks),
        "active_days": float(active_days),
        "unique_sites": float(unique_sites),
        "pre_course_clicks": float(pre_course_clicks),
        "early_assessments_submitted": float(early_assessments_submitted),
        "early_avg_score": early_avg_score,  # NaN if missing/unsubmitted
        "has_early_submission": int(has_submission)
    }])

    try:
        # Generate predictions using the shared prediction module
        prediction_result = predict_student_risk(input_data)

        risk_prob = float(prediction_result.loc[0, "risk_probability"])
        binary_pred = str(prediction_result.loc[0, "prediction"])
        priority_prio = str(prediction_result.loc[0, "priority"])

        st.markdown("---")
        st.subheader("🎯 Assessment Results")

        # Top Metric Display
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(
                label="Predicted Risk Probability",
                value=f"{risk_prob * 100:.1f}%",
                help="Estimated likelihood of academic risk based on Day 30 features."
            )
        with r2:
            st.metric(
                label="Binary Model Prediction",
                value=binary_pred,
                delta="At Risk (Cutoff 0.40)" if binary_pred == "At Risk" else "Not At Risk",
                delta_color="inverse" if binary_pred == "At Risk" else "normal",
                help="Binary classification using the optimized decision threshold of 0.40."
            )
        with r3:
            st.metric(
                label="Operational Support Priority",
                value=priority_prio,
                help="Proposed operational tier to guide educator outreach priority."
            )

        # Detailed Card Breakdown with Context and Suggested Actions
        if priority_prio == "High Priority":
            card_class = "result-card-high"
            badge_icon = "🔴"
            explanation = (
                f"The model estimates a **{risk_prob * 100:.1f}% risk probability**, which exceeds both the **0.40 classification threshold** "
                "and the **0.50 High Priority boundary**. Early engagement indicators (such as low activity days or missing assessments) "
                "strongly associate with student attrition."
            )
            actions = [
                "**Direct Outreach:** Initiate a supportive, non-evaluative check-in (personalized email or phone call) to see how the student is managing course demands.",
                "**Tutoring & Academic Coaching:** Refer to subject tutoring or study skills advisors for foundational topic review.",
                "**Pacing & Workload Review:** Discuss study schedules, upcoming assignment deadlines, and time management strategies.",
                "**Access & Barrier Check:** Verify that the student has reliable internet access, required textbooks, and necessary assistive tools."
            ]

        elif priority_prio == "Watchlist":
            card_class = "result-card-watch"
            badge_icon = "🟡"
            explanation = (
                f"The model estimates a **{risk_prob * 100:.1f}% risk probability**, placing the student in the **Watchlist / Moderate range (0.35 to 0.49)**. "
                + (
                    "Notice that because the probability is $\\ge 0.40$, the binary model predicts **At Risk**, but the priority remains **Watchlist** to indicate elevated uncertainty."
                    if risk_prob >= 0.40 else
                    "Because the probability is $< 0.40$, the binary prediction is **Not At Risk**, but the student is close to the decision boundary."
                )
            )
            actions = [
                "**Monitor Next Assignment:** Track whether the upcoming Day 45/60 assessment is submitted on time.",
                "**Encourage Peer Collaboration:** Prompt participation in module discussion forums and virtual study groups.",
                "**Proactive Resource Reminder:** Send an automated or gentle reminder about available academic office hours.",
                "**Observe Interaction Trends:** Check if login frequency and click activity increase over the next two weeks."
            ]

        else:  # Low Risk
            card_class = "result-card-low"
            badge_icon = "🟢"
            explanation = (
                f"The model estimates a **{risk_prob * 100:.1f}% risk probability**, which is below both the **0.40 classification threshold** "
                "and the **0.35 Low Risk boundary**. Early engagement and assessment signals suggest healthy initial progress."
            )
            actions = [
                "**Standard Communications:** Maintain regular course announcements and encouraging feedback.",
                "**Acknowledge Early Progress:** Provide positive reinforcement on early assessment submissions.",
                "**Routine Periodic Monitoring:** Continue standard check-ins around major mid-term milestones, noting that Day 30 indicators cannot guarantee safety against late-term obstacles."
            ]

        # Render styled result card
        action_items_html = "".join([f"<li>{a}</li>" for a in actions])
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
                {badge_icon} Operational Category: {priority_prio} (Prediction: {binary_pred})
            </div>
            <p style="font-size: 0.95rem; color: #374151; margin-bottom: 12px;">
                {explanation}
            </p>
            <div class="action-header">📋 Suggested Educator Support Options:</div>
            <ul style="font-size: 0.9rem; color: #4B5563; margin-top: 4px; padding-left: 20px;">
                {action_items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Key Distinction Callout
        st.info(
            "📌 **Important Distinction for Educators:**\n\n"
            "* **Binary Classification Threshold (0.40):** The model classifies any probability $\\ge 0.40$ as *At Risk*. "
            "This threshold was calibrated to identify $73.41\%$ of struggling enrollments early.\n"
            "* **Operational Priority Categories (0.35 & 0.50):** These three tiers help prioritize staff time. "
            "A student with $P = 42\\%$ is classified as *At Risk* by the binary model, but assigned to the *Watchlist* tier "
            "to reflect borderline model certainty.\n\n"
            "*Disclaimer: Suggested actions are possible support options for educator consideration, not mandatory decisions or guaranteed outcomes.*"
        )

    except Exception as err:
        st.error(f"❌ An error occurred during prediction: {err}")

# -----------------------------------------------------------------------------
# 7. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>"
    "Early Academic Risk Prediction & Student Support System &bull; Individual Student Assessment Module"
    "</div>",
    unsafe_allow_html=True
)
