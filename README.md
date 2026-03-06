# 🔍 Universal Data Quality Auditor and Auto-Cleaning Pipeline

Upload any CSV or Excel dataset and this tool automatically detects **8 types of data quality issues**, calculates a **quality score out of 100**, cleans the data, and exports a **JSON audit report** — all without writing a single line of code.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 📌 What It Does

Upload your dataset. The tool immediately runs a full audit across 8 issue types:

| # | Issue Type | Detection Method |
|---|---|---|
| 1 | Missing Values | Null count per column |
| 2 | Duplicate Rows | Exact row matching |
| 3 | IQR Outliers | Interquartile Range on coerced numeric columns |
| 4 | Type Mismatches | Numeric columns containing text values |
| 5 | Formatting Issues | Inconsistent casing, whitespace |
| 6 | Low Variance Columns | Near-constant columns flagged |
| 7 | Domain Range Violations | Values outside valid numeric bounds |
| 8 | Invalid Categories | Values not in known valid lists |
| ⚙️ | ML Anomalies | Isolation Forest unsupervised detection |

---

## 📊 Dashboard Tabs

| Tab | Content |
|---|---|
| 📋 Issue Summary | Full breakdown of every issue found with severity tags |
| 📈 Score Report | Quality score (0–100), grade, and deduction breakdown |
| 🔍 Issue Details | Expandable rows showing exact violating values |
| 🤖 ML Anomalies | Isolation Forest anomaly detection with score histogram |
| 🧹 Cleaned Data | Auto-cleaned dataset with full audit log |
| 📊 Data Profile | Column-by-column statistical profile |

---

## 🧹 Auto-Cleaning Pipeline

The cleaning runs in a strict order to ensure accuracy:

1. **Type coercion** — converts object columns to numeric where appropriate  
2. **Remove invalid category rows** — before filling missing values  
3. **Remove duplicates**  
4. **Fill missing values** — median for numeric, mode for categorical  
5. **Domain range clamping** — enforces valid bounds (e.g. review_score → [0, 5])  
6. **IQR capping** — caps remaining outliers for non-domain columns  
7. **Standardize text case** — Title Case for all text columns  
8. **Post-coercion fill** — handles any NaNs created during type conversion  

---

## 📤 Downloads

- **Cleaned CSV** — ready-to-use dataset with all issues fixed  
- **JSON Audit Report** — structured report with score, deductions, and every change made  

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading, profiling, cleaning |
| NumPy | Numerical operations |
| Scikit-learn | Isolation Forest ML anomaly detection, StandardScaler |
| Matplotlib | Anomaly score histogram |
| Streamlit | Interactive web dashboard |

---

## 📁 Project Structure

```
data-quality-auditor/
│
├── app.py               # Main Streamlit app (all logic included)
├── requirements.txt     # All dependencies
└── README.md
```

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/KartikeyaWarhade2002/data-quality-auditor.git
cd data-quality-auditor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
openpyxl
```

Save as `requirements.txt` in the project folder.

---

## 🔍 Quality Scoring

```
Score starts at 100.
Deductions applied per issue type:

Missing Values      → up to  -10
Duplicate Rows      → up to  -10
IQR Outliers        → up to   -5
Type Mismatches     → up to  -10
Formatting Issues   → up to  -10
Domain Violations   → up to   -8
Invalid Categories  → up to  -15
Low Variance        → up to   -5
```

| Score | Grade |
|---|---|
| 90–100 | Excellent ✅ |
| 75–89 | Good 🟢 |
| 60–74 | Needs Cleaning 🟡 |
| Below 60 | Poor 🔴 |

---

## 📸 Screenshots

**Quality Score Cards and Deductions Chart**
<img width="7684" height="4322" alt="quality_score" src="https://github.com/user-attachments/assets/9ce22d3b-533e-4c11-b172-980eb82e2296" />

**Outliers and Type Mismatches Tab**
<img width="7684" height="4322" alt="issue_details" src="https://github.com/user-attachments/assets/a1d1bb98-1ff5-4ac6-9929-6981847c5fe0" />

**Audit Log — Every Change Made**
<img width="7684" height="4322" alt="audit_log" src="https://github.com/user-attachments/assets/c9016bf6-e22c-4513-a275-a57848eadf11" />

**Cleaned Data and Download Buttons**
<img width="7684" height="4322" alt="cleaned_data" src="https://github.com/user-attachments/assets/15f74ad5-9f53-4b8b-8e60-f4ce1e46845e" />


---

## 👤 Developer

**Kartikeya Warhade**
[LinkedIn](https://linkedin.com/in/kartikeya-warhade) · [GitHub](https://github.com/KartikeyaWarhade2002)
