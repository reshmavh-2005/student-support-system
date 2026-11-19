# Early Academic Risk Prediction and Student Support System Using Machine Learning

An end-to-end Machine Learning decision-support system designed to assist higher education educators and academic advisors in identifying students at risk of course withdrawal or academic failure early in the term (by Day 30).

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Abstract](#2-abstract)
3. [Key Features](#3-key-features)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Machine Learning Details](#6-machine-learning-details)
7. [Technology Stack](#7-technology-stack)
8. [Project Structure](#8-project-structure)
9. [Installation and Local Setup](#9-installation-and-local-setup)
10. [AWS Deployment](#10-aws-deployment)
11. [Limitations and Responsible Use](#11-limitations-and-responsible-use)
12. [Future Improvements](#12-future-improvements)

---

## 1. Project Overview

In higher education distance and blended learning environments, students who face academic difficulties or decide to withdraw frequently do so without early faculty awareness. By the time mid-term assessments or final examinations occur, it is often too late for academic advisors to organize effective support.

### The Problem Addressed
* **Silent Disengagement:** Students studying online can experience feelings of isolation and fall behind on reading, virtual activities, or initial assignments without raising immediate red flags.
* **Delayed Interventions:** Traditional institutional interventions occur reactively after major exam failures or formal withdrawal requests.
* **Advisory Caseload Constraints:** Academic advisors and course tutors handle hundreds of enrollments per presentation and lack automated tools to triage which students need immediate outreach.

### Who Can Use the System
* **Course Instructors & Tutors:** To monitor early student engagement patterns and identify students who may need assignment support or study-group referrals.
* **Academic Advisors & Counselors:** To triage advising workloads and prioritize personalized check-in conversations during early academic weeks.
* **Institutional Retention Teams:** To review cohort-wide risk distributions and plan proactive academic assistance programs.

### How It Assists Educators
The system operates as an **early warning decision-support tool**. The model uses 18 features available by Day 30, including demographic background, prior academic load, pre-course engagement information, and student virtual learning environment (VLE) activity recorded during Days 1–30. Based on these features, the system estimates a continuous risk probability and assigns students to operational support tiers.

> **Important Principle:** The system provides decision support, **not automated decisions**. It is designed to assist human educators, not to replace professional judgement, assign academic penalties, or predetermine student capability.

---

## 2. Abstract

Early identification of academic risk is critical for improving student retention and providing timely academic assistance in higher education distance learning. Using the Open University Learning Analytics Dataset (OULAD), the project constructs a modeling cohort of 27,373 eligible student-course enrollments across seven modules and four presentation periods.

Restricting feature extraction strictly to information available by Day 30 of the course presentation, we engineer 18 early indicators covering student background, prior academic load, pre-course engagement, and early virtual learning environment activity recorded during Days 1–30. To prevent data leakage across repeating students, model training and evaluation are conducted using a student-level group-aware split (80% train / 20% test).

Among several candidate algorithms, a regularized Logistic Regression pipeline was selected for its balanced predictive performance, probability estimates used for risk scoring and threshold-based classification, and transparency. Evaluated on a held-out test cohort of 5,462 enrollments at an operational decision threshold of 0.40, the model achieves **73.41% Recall**, **63.14% Precision**, an **F1-score of 67.89%**, an **ROC-AUC of 77.45%**, and a **PR-AUC of 74.13%**.

The model is integrated into a multi-page interactive Streamlit web application supporting both individual student risk assessment and cohort-wide batch CSV predictions with operational priority triaging. The application is packaged and deployed on an Amazon Web Services (AWS) EC2 cloud instance, delivering an accessible, transparent prototype for educational support teams.

---

## 3. Key Features

The application provides a modern, educator-oriented interface built with Streamlit:

### 1. Home Dashboard (`Home.py`)
* **Project Context & Rationale:** Outlines the early academic intervention problem and the 30-day cutoff strategy.
* **Historical Cohort Overview:** Displays dataset statistics across eligible historical enrollments, class distributions, and feature counts.
* **Model Performance Metrics:** Highlights verified held-out test set metrics (Recall, Precision, F1-score, ROC-AUC, PR-AUC).
* **Interactive Metric Explanations:** Explains how each metric relates to student support and alert workload management.
* **Ethical Use Guidelines:** Clear statements on responsible use, data limitations, and human-in-the-loop workflows.

### 2. Individual Student Risk Prediction (`1_Individual_Prediction.py`)
* **Interactive Assessment Form:** Organizes the 18 model features into intuitive categories:
  * Course presentation and student demographic background.
  * Previous academic study and registration timing.
  * Day 1–30 virtual learning environment engagement (clicks, active days, unique resources visited).
  * Early assessment submissions and scores.
* **Dynamic Dropdown Options:** Sourced directly from historical cohort data to prevent invalid category entry.
* **Real-Time Risk Calculation:** Reuses the verified Scikit-learn pipeline to compute the continuous probability $P(\text{At Risk})$.
* **Visual Status Display:** Displays a risk probability progress gauge, binary risk status (threshold 0.40), and operational support priority badge.
* **Educator Support Suggestions:** Displays general educator support suggestions corresponding to the assigned operational priority tier to assist tutors in considering constructive next steps.

### 3. Batch Student Risk Prediction (`2_Batch_Prediction.py`)
* **CSV File Upload:** Allows educators to upload an entire student cohort CSV file for in-memory batch evaluation.
* **Automated Column Validation:** Checks for all 18 required model input features before execution and reports missing columns clearly.
* **In-Memory Cohort Inference:** Processes batch records efficiently using the shared Scikit-learn prediction module.
* **Cohort Summary KPIs:** Reports total enrollments uploaded, percentage flagged as at-risk, and distribution across priority tiers.
* **Interactive Priority Filtering:** Allows educators to isolate specific tiers (High Priority, Watchlist, or Low Risk).
* **Enriched Data Table:** Presents enrollment records with predicted risk probabilities, classifications, and priority categories.
* **CSV Results Export:** Enables one-click download of the complete prediction results for integration with academic advising spreadsheets.
* **Sample Template Download:** Offers a downloadable CSV template with sample records for testing cohort uploads.

### 4. Operational Support Priority Categories
To help advisors schedule outreach workloads efficiently, continuous risk probabilities are mapped into three actionable support tiers:

| Priority Category | Probability Range | Recommended Educator Action |
|---|---|---|
| **High Priority** | P ≥ 0.50 | Consider direct advisor contact, tutoring referral, or academic coaching. |
| **Watchlist** | 0.35 ≤ P < 0.50 | Consider a supportive check-in and monitor upcoming assessments. |
| **Low Risk** | P < 0.35 | Continue routine academic monitoring and standard communications. |

These recommended actions are general educator support suggestions associated with operational priority tiers, not personalized interventions dynamically generated from individual risk drivers. Furthermore, the application provides decision support and does not automatically contact students or send intervention emails; all outreach requires educator initiative.

### 5. Distinguishing the Two Threshold Systems
The project utilizes two separate threshold mechanisms that serve distinct operational purposes:
* **Binary Classification Threshold (0.40):** The formal statistical decision boundary used to classify each student enrollment as *At Risk* ($P \ge 0.40$) or *Not At Risk* ($P < 0.40$). This cutoff is used to compute model evaluation metrics (Recall, Precision, F1-score).
* **Operational Priority Boundaries (0.50 and 0.35):** Separate triage boundaries used to assign students into *High Priority* ($P \ge 0.50$), *Watchlist* ($0.35 \le P < 0.50$), and *Low Risk* ($P < 0.35$) operational tiers to assist educators in scheduling check-ins.
* **Interpreting Intersecting Ranges:** An enrollment with a predicted probability between 0.40 and 0.49 is classified as *At Risk* by the binary model, but assigned to the *Watchlist* operational tier to indicate moderate risk and proximity to the decision boundary.

---

## 4. Dataset

This project is built using the **Open University Learning Analytics Dataset (OULAD)**, an internationally recognized benchmark dataset released by The Open University (UK).

### Dataset Scope
* **Historical Data:** Contains longitudinal distance-learning records from 7 selected undergraduate courses (modules `AAA` through `GGG`) offered across 4 academic terms (`2013B`, `2013J`, `2014B`, `2014J`).
* **Raw Relational Tables:**
  * `studentInfo.csv`: Student demographic records, highest prior education, region, age band, deprivation decile, declared disability, and final module result.
  * `studentRegistration.csv`: Course registration date and formal withdrawal date.
  * `courses.csv`: Module codes and presentation lengths.
  * `vle.csv` & `studentVle.csv`: Virtual learning platform resources and daily interaction click logs.
  * `assessments.csv` & `studentAssessment.csv`: Assessment types, submission dates, and awarded scores.

### Target Formulation & Eligible Cohort
* **Day 30 Early Warning Cutoff:** To create an early warning system rather than an end-of-term summary, features are constructed strictly from information available on or before Day 30 of each presentation, including pre-course interactions and engagement recorded during Days 1–30.
* **Cohort Exclusion Criteria:**
  * Students who formally withdrew on or before Day 30 were excluded because their departure preceded the early intervention window.
  * An additional 93 ambiguous Withdrawn records with missing unregistration dates were excluded from the initial prediction cohort.
* **Eligible Cohort Size:** **27,373** student-course enrollments.
* **Binary Target (`target`):**
  * `0` (Not At Risk): **15,385** enrollments (56.21%) who successfully completed the course with a *Pass* or *Distinction*.
  * `1` (At Risk): **11,988** enrollments (43.79%) who subsequently *Failed* or *Withdrew* after Day 30.

> **Data Privacy Notice:** In compliance with repository best practices, raw student data files are gitignored and are not stored in this GitHub repository.

---

## 5. Methodology

The project follows a rigorous, reproducible Data Science workflow:

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   1. Data Understanding  │ ──> │ 2. Exploratory Analysis │ ──> │  3. Feature Engineering  │
│  Schema, keys & missing  │     │ Engagement patterns &   │     │  Day 1–30 VLE & scores  │
│      value auditing     │     │   target distributions  │     │  Leakage-free filtering │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   6. Threshold Tuning   │ <── │   5. Model Evaluation   │ <── │    4. Model Training    │
│  OOF trade-off analysis │     │ Test set metrics: ROC,  │     │ Group-aware split (80/20)│
│     Set cutoff to 0.40  │     │   PR-AUC & confusion    │     │  5 algorithms compared  │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
             │
             ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  7. Web App Development │ ──> │    8. Cloud Deployment  │
│ Streamlit multi-page UI │     │     AWS EC2 Instance    │
│  Single & Batch modes   │     │   Web server on :8501   │
└─────────────────────────┘     └─────────────────────────┘
```

1. **Data Understanding (`01_dataset_understanding.ipynb`):** Examined table relationships, composite keys (`id_student`, `code_module`, `code_presentation`), and data types.
2. **Exploratory Data Analysis (`02_exploratory_data_analysis.ipynb`):** Analyzed target distribution, interaction frequency, and relationships between early clicks and course completion.
3. **Feature Engineering (`03_feature_engineering.ipynb`):** Formulated Day-30 features, resolved missing data imputation rules, and constructed the modeling dataset (`student_model_data.csv`).
4. **Model Training & Comparison (`04_model_training.ipynb`):** Evaluated five candidate algorithms under student-level cross-validation.
5. **Model Evaluation (`05_model_evaluation.ipynb`):** Analyzed performance on unseen test data across multiple classification thresholds, ROC curves, and Precision-Recall dynamics.
6. **Error & Fairness Analysis (`06_error_analysis.ipynb` & `07_feature_comparison.ipynb`):** Examined false positive and false negative characteristics across demographic and educational subgroups.
7. **Application Integration (`app/` and `src/predict.py`):** Encapsulated the trained pipeline into a reusable prediction module and built the Streamlit frontend.
8. **Cloud Deployment:** Hosted the web application on an AWS EC2 instance.

---

## 6. Machine Learning Details

### 1. Problem Formulation
* **Task Type:** Supervised Binary Classification.
* **Positive Class ($1$):** At Risk (Course Failure or Post-Day 30 Withdrawal).
* **Negative Class ($0$):** Not At Risk (Course Pass or Distinction).
* **Prediction Horizon:** Day 30 of the course presentation.

### 2. Selected Model Features (18 Total)
The model uses 18 features available by Day 30, including pre-course engagement information and student activity recorded during Days 1–30:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `code_module` | Categorical | Course code identifier (7 modules: AAA through GGG). |
| `code_presentation` | Categorical | Course offering cohort (e.g., 2013B, 2013J). |
| `gender` | Categorical | Student gender (`M` / `F`). |
| `region` | Categorical | Geographic region of the student. |
| `highest_education` | Categorical | Prior educational level upon course enrollment. |
| `imd_band` | Categorical | Index of Multiple Deprivation decile band. |
| `age_band` | Categorical | Age bracket (`0-35`, `35-55`, `55<=`). |
| `disability` | Categorical | Declared disability status (`N` / `Y`). |
| `num_of_prev_attempts` | Numerical | Number of previous attempts on the module. |
| `studied_credits` | Numerical | Total credit load currently studied by the student. |
| `date_registration` | Numerical | Days between student registration and course start date. |
| `total_clicks` | Numerical | Total VLE interactions recorded between Day 1 and Day 30. |
| `active_days` | Numerical | Number of distinct active login days between Day 1 and Day 30. |
| `unique_sites` | Numerical | Count of distinct VLE materials/pages visited. |
| `pre_course_clicks` | Numerical | Total VLE clicks logged prior to Day 0 (pre-course study). |
| `early_assessments_submitted`| Numerical | Number of assessment items submitted on or before Day 30. |
| `early_avg_score` | Numerical | Average percentage score on early assessments (median imputed if none). |
| `has_early_submission` | Numerical | Binary indicator: `1` if at least one assessment was submitted, `0` otherwise. |

### 3. Student-Level Group-Aware Splitting
In educational data, individual students frequently enroll in multiple courses simultaneously or retake modules in subsequent terms. Splitting records randomly by rows would place records from the same individual student into both the training and testing sets, causing **data leakage** and overly optimistic performance estimates.

To prevent this:
* Splitting was performed using **`GroupShuffleSplit`** (80% train / 20% test, `random_state=42`) grouped strictly on **`id_student`**.
* **Training Partition:** 21,911 enrollments across **19,768 unique students**.
* **Testing Partition:** 5,462 enrollments across **4,942 unique students**.
* **Verified Student Overlap:** Exactly **0 overlapping students** between training and testing partitions.

### 4. Algorithm Selection: Logistic Regression
During model benchmarking (`04_model_training.ipynb`), Logistic Regression was selected over tree-based ensembles (Decision Tree, Random Forest) and K-Nearest Neighbors for key reasons:
1. **Continuous Probability Estimates:** Logistic Regression outputs continuous risk probability estimates via the logistic sigmoid function, which are directly utilized for operational risk scoring, priority tiering, and threshold-based classification.
2. **Model Coefficients & Relative Interpretability:** Logistic Regression provides model coefficients that can be examined to understand how input features contribute directionally to predictions. These coefficients offer a degree of interpretability compared with more complex non-linear ensemble models, although they do not provide complete causal explanations for individual student predictions.
3. **Comparative Model Evaluation:** In 5-fold cross-validation on the training set (`04_model_training.ipynb`), Logistic Regression achieved performance comparable to non-linear tree ensembles (CV F1 = 0.6394 vs. 0.6273 for Random Forest), while utilizing a simpler linear model structure with fewer hyperparameters.
4. **Lightweight Deployment Footprint:** The serialized pipeline has a compact storage footprint (~7.5 KB), making it convenient to package and load within the Streamlit application.

### 5. Final Model Performance on Held-Out Test Set ($N = 5,462$)
Evaluated at the operational threshold of **0.40**:

| Metric | Score | Educational Meaning |
| :--- | :---: | :--- |
| **Recall (Sensitivity)** | **73.41%** | Identifies approximately 73% of at-risk enrollments in the held-out test set (1,739 of 2,369). |
| **Precision** | **63.14%** | When an alert is raised, 63.14% of flagged enrollments are genuinely at risk (1,739 of 2,754 flagged enrollments). |
| **F1-Score** | **67.89%** | Harmonic mean of Recall and Precision, balancing detection with alert caseload. |
| **Accuracy** | **69.88%** | Total correct classifications across both passing and at-risk cohorts. |
| **ROC-AUC** | **77.45%** | ROC-AUC measures the model's ability to distinguish between at-risk and not-at-risk enrollments across classification thresholds. |
| **PR-AUC** | **74.13%** | PR-AUC summarizes the precision-recall trade-off across thresholds, evaluated in the context of the 43.37% positive-class prevalence in the test cohort. |

#### Confusion Matrix Breakdown:
* **True Positives (TP):** 1,739 (At-risk enrollments successfully identified)
* **False Negatives (FN):** 630 (At-risk enrollments missed)
* **False Positives (FP):** 1,015 (Passing enrollments flagged for support)
* **True Negatives (TN):** 2,078 (Passing enrollments correctly unflagged)

### 6. Rationale for the 0.40 Decision Threshold
Standard binary classification models default to a decision threshold of `0.50`. In this early warning system, the project prioritizes identifying more potentially at-risk enrollments early in the term while accepting additional false-positive alerts and educator follow-up as an operational trade-off:

| Metric | Threshold 0.50 | Threshold 0.40 |
|---|---:|---:|
| **Recall** | 58.72% | 73.41% |
| **Precision** | 69.79% | 63.14% |
| **F1-score** | 63.78% | 67.89% |

Adjusting the decision threshold from `0.50` down to `0.40`:
* **Increases Recall by approximately 14.69 percentage points** (from 58.72% to 73.41%), identifying **348 additional at-risk enrollments** (1,739 vs. 1,391).
* **Reduces Precision from 69.79% to 63.14%**, generating **413 additional false-positive alerts** (1,015 vs. 602).
* **Total flagged alerts increase** from 1,993 to 2,754 (+761 alerts across the 5,462 test enrollments).

The operational trade-off accepts a higher rate of false-positive check-ins in order to identify approximately 73% of at-risk enrollments early in the presentation.

#### Distinguishing Binary Classification from Operational Priority Tiers
* **Binary Classification Threshold (0.40):** Classifies each student enrollment formally as *At Risk* ($P \ge 0.40$) or *Not At Risk* ($P < 0.40$) for statistical evaluation and model metrics.
* **Operational Priority Boundaries (0.50 and 0.35):** Separate triage thresholds used to assign students into *High Priority* ($P \ge 0.50$), *Watchlist* ($0.35 \le P < 0.50$), and *Low Risk* ($P < 0.35$) operational tiers to assist educators in planning check-in schedules.

---

## 7. Technology Stack

The project uses established open-source Python libraries and frameworks:

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language (Local Dev)** | Python | 3.10 | Local development environment for notebooks, scripts, and testing. |
| **Language (AWS EC2)** | Python | 3.11 | Cloud runtime environment on Amazon Linux 2023. |
| **Data Manipulation** | Pandas | 2.3.3 | Tabular data processing, aggregation, and feature extraction. |
| **Numerical Computing** | NumPy | 2.2.6 | Array calculations and numerical operations. |
| **Machine Learning** | Scikit-learn | 1.7.2 | Preprocessing pipelines, imputation, scaling, models, and evaluation metrics. |
| **Artifact Persistence**| Joblib | 1.6.0 | Serializing and loading the trained Scikit-learn pipeline. |
| **Visualization** | Matplotlib & Seaborn | 3.10.9 / 0.13.2 | EDA charts, ROC curves, confusion matrix heatmaps, and threshold plots. |
| **Web Frontend** | Streamlit | 1.65.0 | Interactive multi-page dashboard for educators. |
| **Cloud Hosting** | AWS EC2 | — | Virtual server hosting the running Streamlit web application. |
| **Version Control** | Git & GitHub | — | Repository tracking, versioning, and collaborative development. |

---

## 8. Project Structure

A clean, modular repository organization:

```text
student-support-system/
│
├── .gitignore                      # Git exclusion rules for virtualenv, raw data, and cache
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Exact project dependencies and pinned versions
│
├── app/                            # Streamlit web application
│   ├── Home.py                     # Main dashboard: overview, metrics, and workflow
│   └── pages/
│       ├── 1_Individual_Prediction.py # Single student risk assessment form & recommendations
│       └── 2_Batch_Prediction.py      # Cohort CSV upload, validation, filtering & export
│
├── src/                            # Shared application source code
│   ├── __init__.py                 # Package initializer
│   └── predict.py                  # Core inference module: model loading, threshold, and scoring
│
├── artifacts/                      # Serialized model components
│   ├── .gitkeep                    # Tracks directory in git
│   ├── logistic_regression_pipeline.joblib # Full trained Scikit-learn preprocessing & model pipeline
│   └── model_threshold.txt         # Plaintext operational threshold value (0.40)
│
├── notebooks/                      # Sequential Jupyter research and development notebooks
│   ├── 01_environment_test.ipynb   # Environment verification and library testing
│   ├── 01_dataset_understanding.ipynb # Relational OULAD exploration and schema checks
│   ├── 02_exploratory_data_analysis.ipynb # Feature distributions, correlations, and target balance
│   ├── 03_feature_engineering.ipynb # Day-30 feature extraction and dataset preparation
│   ├── 04_model_training.ipynb     # Group-aware splitting and algorithm benchmarking
│   ├── 05_model_evaluation.ipynb   # Held-out evaluation, threshold tuning, and ROC curves
│   ├── 06_error_analysis.ipynb     # Error breakdown across demographic and engagement subgroups
│   └── 07_feature_comparison.ipynb # Experimental 18 vs 12 feature comparison study
│
└── data/                           # Data directory (contents gitignored)
    ├── raw/                        # Original OULAD CSV files
    ├── processed/                  # Final modeling dataset (student_model_data.csv)
    └── external/                   # Optional external reference datasets
```

---

## 9. Installation and Local Setup

Follow these step-by-step instructions to set up and run the project locally on **Windows**:

### Prerequisites
* **Python:** Python 3.10 ([Python Downloads](https://www.python.org/downloads/)).
* **Git:** Git ([Git SCM](https://git-scm.com/)).

### Step 1: Clone the Repository
Open PowerShell or Command Prompt and clone the repository:
```powershell
git clone https://github.com/reshmavh-2005/student-support-system.git
cd student-support-system
```

### Step 2: Create a Virtual Environment
Create an isolated Python virtual environment named `.venv`:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
Activate the environment:
* In **PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first).*
* In **Command Prompt**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
Upgrade pip and install all required packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Launch the Streamlit Application
Run the Streamlit application using the official entry point:
```powershell
python -m streamlit run app/Home.py
```

### Step 6: Open the Dashboard
Once launched, Streamlit will display the local URL. Open your web browser and navigate to:
```text
http://localhost:8501
```

Use the left sidebar navigation to switch between the **Home Dashboard**, **Individual Prediction**, and **Batch Prediction** pages.

---

## 10. AWS Deployment

The interactive Streamlit application is deployed to the cloud using an **Amazon Web Services (AWS) EC2** virtual server:

### Architecture Overview
* **Virtual Machine:** AWS EC2 Instance running **Amazon Linux 2023**.
* **Network & Security:** The EC2 Security Group is configured to allow inbound TCP traffic on port `8501` for web browser access. As a security best practice, inbound access should be restricted to trusted institutional or administrative IP addresses where possible.
* **Server Runtime:** **Python 3.11** virtual environment named `student-support-env`.
* **File Deployment:** Project files and application source code were uploaded directly under `/home/ec2-user` via **WinSCP**.

### Actual Cloud Deployment Steps
1. **EC2 Provisioning:**
   * Launch an Amazon Linux 2023 EC2 instance.
   * Attach a security group configured with an inbound rule allowing Custom TCP traffic on port `8501` (restricted to trusted IP ranges where practical).
2. **System Package Setup:** Connect to the instance via SSH and install Python 3.11 and package tools using `dnf`:
   ```bash
   sudo dnf update -y
   sudo dnf install -y python3.11 python3.11-pip
   ```
3. **Project File Transfer:**
   * Transfer the project directories (`app/`, `src/`, `artifacts/`, `requirements.txt`) directly to `/home/ec2-user/` using WinSCP via SFTP.
4. **Virtual Environment & Dependencies:**
   * In `/home/ec2-user`, create and activate the Python 3.11 virtual environment named `student-support-env`:
   ```bash
   python3.11 -m venv student-support-env
   source student-support-env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Run the Streamlit Application:**
   * Launch the application bound to port `8501`:
   ```bash
   streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0
   ```
6. **Web Access:**
   * Access the dashboard in any modern web browser via:
   ```text
   http://<EC2-PUBLIC-IP-OR-DNS>:8501
   ```

*(Note: Sensitive cloud credentials, access keys, private `.pem` keys, and specific host IP addresses are deliberately excluded from this repository for security purposes).*

---

## 11. Limitations and Responsible Use

To maintain ethical and responsible machine learning standards, users must recognize the system's operational boundaries:

1. **Decision Support, Not Automated Evaluation:**
   * This system is an **advisory decision-support prototype**. Predictions must never be used to make automated academic decisions, alter student grades, deny course admission, or assign disciplinary actions.
2. **Use of Demographic and Background Attributes:**
   * The production model utilizes six student demographic and socioeconomic features: `gender`, `region`, `age_band`, `highest_education`, `imd_band`, and `disability`.
   * Model error rates and performance metrics can differ across demographic subgroups. Subgroup analysis examined differences in prediction errors across educational and demographic groups, highlighting the importance of further fairness evaluation.
   * Removing demographic attributes from predictive models does not automatically eliminate bias or guarantee algorithmic fairness, as online engagement patterns and platform usage may themselves correlate with underlying structural, technological, or accessibility factors.
   * Rigorous subgroup fairness auditing and local institutional validation are essential before any practical adoption.
3. **Historical Data Limitations:**
   * The underlying model was trained exclusively on historical distance-learning records from The Open University (UK). Student demographics, course structures, and LMS usage patterns in other universities or high schools will differ.
   * The model **must be evaluated and validated on institutional data** before being deployed in an active academic environment.
4. **Statistical Correlation vs. Causality:**
   * Model outputs indicate statistical correlations observed during the early study period. A high risk probability does **not** identify the root cause of a student's difficulty (e.g., illness, employment shifts, technical hurdles, personal circumstances).
5. **Human-in-the-Loop Requirement:**
   * Educators must interpret risk scores in combination with personal check-ins, classroom discussions, and institutional context. An alert is simply a prompt to initiate a supportive conversation.
6. **Transparent, Supportive, and Ethical Engagement:**
   * Communication and outreach prompted by the system should remain supportive, empathetic, and resource-oriented (such as sharing tutoring schedules, writing center resources, or advising appointments).
   * Interactions should be transparent, ethical, and grounded in direct educator-student dialogue. Educators must review all context before initiating check-ins and ensure the tool is used constructively rather than punitively.

---

## 12. Future Improvements

Practical avenues for future development include:

* **Dynamic Time-Series Tracking:** Extend predictions beyond Day 30 to incorporate ongoing Day 60 and Day 90 interaction trajectories as courses progress.
* **Local Explainability (SHAP):** Integrate per-student SHAP (SHapley Additive exPlanations) visualizations to show educators the top positive and negative factors driving an individual risk score.
* **Institutional Pilot Testing:** Test the pipeline against modern Learning Management Systems (Canvas, Moodle, Blackboard) via secure LMS API integrations.
* **Investigation of Reduced Feature Sets (12-Feature Model B):** Further explore the 12-feature model investigated in `07_feature_comparison.ipynb`, which excludes six demographic and background features (`gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`) and relies strictly on course context and Day 1–30 VLE engagement. On the held-out test set, Model B identified approximately 98.9% of the true-positive at-risk enrollments identified by Model A (1,720 vs. 1,739), although its overall predictive performance was lower. Removing demographic features does not guarantee fairness or eliminate bias, as engagement patterns may themselves correlate with underlying disparities. Further study is required to evaluate fairness and operational feasibility across diverse cohorts.
* **Intervention Outcome Tracking:** Implement feedback tracking to log which advisory interventions were conducted and measure whether student engagement improved after outreach.

---

## Author & Acknowledgments

* **Project:** Early Academic Risk Prediction and Student Support System Using Machine Learning
* **Dataset:** Open University Learning Analytics Dataset (OULAD), provided by Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017).
* **Tools:** Built with Python, Scikit-learn, and Streamlit.
