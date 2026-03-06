# =============================================================================
# PROJECT 2: Universal Data Quality Auditor and Auto-Cleaning Pipeline
# Developer: Kartikeya Warhade
#
# BUGS FIXED vs ORIGINAL:
#   BUG 1:  Type mismatch now reports exact bad values + count (not just "should be numeric")
#   BUG 2:  IQR now runs on COERCED copy — catches 9.8 review_score, 5000 player_count
#   BUG 3:  indian_popularity -3 no longer missed — domain rules catch it regardless of IQR
#   BUG 4:  auto_clean now enforces domain rules (review_score→[0,5], india_pop→[1,10], etc.)
#   BUG 5:  auto_clean now coerces type mismatches FIRST before any numeric operations
#   BUG 6:  Added full domain/range validation module for all numeric columns
#   BUG 7:  Added invalid category detection (platform, genre against known valid lists)
#   BUG 8:  Replaced ALL st.metric() with HTML cards — no more white boxes in dark mode
#   BUG 9:  Added schema validator — checks required columns exist, warns on mismatch
#   BUG 10: Isolation Forest contamination is now a sidebar slider (user-controlled)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =============================================================================
# DOMAIN RULES
# Defines valid numeric ranges and valid categorical values per column.
# Used by BOTH the detector and the cleaner — single source of truth.
# =============================================================================

CURRENT_YEAR = datetime.now().year

DOMAIN_RULES = {
    "numeric_ranges": {
        "review_score":          {"min": 0.0,  "max": 5.0},
        "indian_popularity":     {"min": 1.0,  "max": 10.0},
        "player_count_millions": {"min": 0.0,  "max": 5000.0},
        "release_year":          {"min": 1970, "max": CURRENT_YEAR},
        "multiplayer":           {"min": 0,    "max": 1},
        "battle_royale":         {"min": 0,    "max": 1},
        "open_world":            {"min": 0,    "max": 1},
        "fps_shooter":           {"min": 0,    "max": 1},
        "sports":                {"min": 0,    "max": 1},
        "rpg":                   {"min": 0,    "max": 1},
        "indie":                 {"min": 0,    "max": 1},
    },
    "valid_categories": {
        "platform": [
            "PC", "Mobile", "PlayStation", "Xbox", "Nintendo",
            "Console", "PS4", "PS5", "Xbox One", "Xbox Series X",
            "PC/Mobile", "PC/Console", "Mobile/Console"
        ],
        "genre": [
            "Action", "RPG", "FPS", "Battle Royale", "Sports", "Racing",
            "Strategy", "MOBA", "Simulation", "Puzzle", "Platformer",
            "Horror", "Adventure", "Sandbox", "Fighting", "Co-op",
            "Card", "Board", "Social Deduction", "Endless Runner",
            "Casual", "AR Adventure", "Roguelike", "Survival",
            "Open World", "eSports", "Fantasy Sports", "Indie",
            "Shooter", "MMO", "FPS RPG", "Action RPG", "Turn-Based RPG"
        ]
    }
}

REQUIRED_COLUMNS = [
    "name", "genre", "platform", "multiplayer", "battle_royale",
    "open_world", "fps_shooter", "sports", "rpg", "indie",
    "review_score", "player_count_millions", "indian_popularity",
    "release_year", "developer", "publisher"
]

# =============================================================================
# HTML CARD HELPER — replaces ALL st.metric() calls
# BUG FIX 8: No more white boxes in dark mode
# =============================================================================

def info_card(value, label, bg="#1565C0"):
    return (
        f'<div style="flex:1; min-width:140px; background:{bg}; border-radius:10px;'
        f'padding:16px 14px; text-align:center; color:#ffffff;'
        f'box-shadow:0 2px 6px rgba(0,0,0,0.3); margin:4px;">'
        f'<div style="font-size:22px; font-weight:800;">{value}</div>'
        f'<div style="font-size:10px; margin-top:4px; opacity:0.85;'
        f'text-transform:uppercase; letter-spacing:0.8px;">{label}</div>'
        f'</div>'
    )

def cards_row(*cards):
    return (
        '<div style="display:flex; gap:10px; margin:12px 0 18px 0; flex-wrap:wrap;">'
        + "".join(cards) + '</div>'
    )

# =============================================================================
# MODULE 1: SCHEMA VALIDATOR (BUG FIX 9)
# =============================================================================

def validate_schema(df):
    df_cols   = [c.strip().lower() for c in df.columns]
    expected  = [c.lower() for c in REQUIRED_COLUMNS]
    missing   = [c for c in expected if c not in df_cols]
    extra     = [c for c in df_cols  if c not in expected]
    warnings  = []
    if missing:
        warnings.append(f"Missing expected columns: {missing}")
    if extra:
        warnings.append(f"Extra columns found (will still process): {extra}")
    return len(missing) == 0, missing, extra, warnings

# =============================================================================
# MODULE 2: DATA PROFILER
# =============================================================================

def profile_dataframe(df):
    profile = {}
    for col in df.columns:
        col_data = df[col]
        dtype    = str(col_data.dtype)
        p = {
            "column_name":     col,
            "data_type":       dtype,
            "total_rows":      len(col_data),
            "missing_count":   int(col_data.isnull().sum()),
            "missing_percent": round(col_data.isnull().mean() * 100, 2),
            "unique_count":    int(col_data.nunique()),
            "unique_percent":  round(col_data.nunique() / max(len(col_data),1) * 100, 2),
        }
        if dtype in ['int64','float64']:
            p.update({
                "mean":        round(float(col_data.mean()),   4) if not col_data.isnull().all() else None,
                "median":      round(float(col_data.median()), 4) if not col_data.isnull().all() else None,
                "std":         round(float(col_data.std()),    4) if not col_data.isnull().all() else None,
                "min":         round(float(col_data.min()),    4) if not col_data.isnull().all() else None,
                "max":         round(float(col_data.max()),    4) if not col_data.isnull().all() else None,
                "is_constant": col_data.nunique() <= 1,
            })
        else:
            p.update({"mean":None,"median":None,"std":None,"min":None,"max":None,"is_constant":col_data.nunique()<=1})
        profile[col] = p
    return profile

# =============================================================================
# MODULE 3: ISSUE DETECTOR — ALL 8 ISSUE TYPES
#
# KEY FIX EXPLAINED:
# The original code ran df.select_dtypes(include=[np.number]) FIRST,
# which completely skipped review_score and player_count_millions because
# they were stored as object (due to injected text values like "many").
#
# FIX: We build df_coerced — a working copy where object columns that
# are MOSTLY numeric get force-converted. ALL checks then run on df_coerced.
# The original df is untouched and only used for category checks.
# =============================================================================

def detect_issues(df):
    issues = {
        "missing_values":          [],
        "duplicate_rows":          [],
        "outliers_iqr":            [],
        "type_mismatches":         [],
        "formatting_inconsistencies": [],
        "low_variance_columns":    [],
        "domain_range_violations": [],   # BUG FIX 6
        "invalid_categories":      [],   # BUG FIX 7
    }

    # ── Build coerced working copy (BUG FIX 2 + 5) ───────────────────────────
    # Convert object columns that are mostly numeric so IQR + domain checks work
    df_coerced = df.copy()
    for col in df_coerced.select_dtypes(include=['object','string']).columns:
        attempt = pd.to_numeric(df_coerced[col], errors='coerce')
        if attempt.notna().mean() > 0.7:   # >70% converts → treat as numeric
            df_coerced[col] = attempt

    # ── ISSUE 1: Missing values ───────────────────────────────────────────────
    for col in df.columns:
        mc = df[col].isnull().sum()
        if mc > 0:
            pct = round(mc / len(df) * 100, 2)
            issues["missing_values"].append({
                "column":          col,
                "missing_count":   int(mc),
                "missing_percent": pct,
                "severity":        "HIGH" if pct > 40 else "MEDIUM" if pct > 10 else "LOW",
                "recommendation":  "Drop column" if pct > 40 else "Fill with median/mode"
            })

    # ── ISSUE 2: Duplicate rows ───────────────────────────────────────────────
    dup = int(df.duplicated().sum())
    if dup > 0:
        issues["duplicate_rows"].append({
            "count":          dup,
            "percent":        round(dup / len(df) * 100, 2),
            "severity":       "HIGH" if dup / len(df) > 0.1 else "MEDIUM",
            "recommendation": "Remove duplicate rows"
        })

    # ── ISSUE 3: Type mismatches (BUG FIX 1 — now reports bad values) ────────
    for col in df.select_dtypes(include=['object','string']).columns:
        non_null     = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted    = pd.to_numeric(non_null, errors='coerce')
        convertible  = converted.notna().sum()
        pct          = convertible / len(non_null)
        bad_vals     = non_null[converted.isna()].unique()[:5].tolist()
        bad_count    = int((converted.isna()).sum())

        if pct > 0.7 and bad_count > 0:   # mostly numeric but has text values
            issues["type_mismatches"].append({
                "column":              col,
                "current_type":        "text (object)",
                "suggested_type":      "numeric",
                "convertible_percent": round(pct * 100, 2),
                "bad_value_count":     bad_count,         # BUG FIX 1
                "bad_value_examples":  str(bad_vals),     # BUG FIX 1
                "severity":            "MEDIUM",
                "recommendation":      f"Convert '{col}' to numeric; fix/remove {bad_count} invalid text entries"
            })

    # ── ISSUE 4: IQR Outliers (BUG FIX 2 — now runs on coerced copy) ─────────
    numeric_cols = df_coerced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df_coerced[col].dropna()
        if len(col_data) < 4:
            continue
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR    = Q3 - Q1
        if IQR == 0:
            continue
        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outlier_mask  = (df_coerced[col] < lb) | (df_coerced[col] > ub)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            issues["outliers_iqr"].append({
                "column":          col,
                "outlier_count":   outlier_count,
                "outlier_percent": round(outlier_count / len(df_coerced) * 100, 2),
                "lower_bound":     round(float(lb), 4),
                "upper_bound":     round(float(ub), 4),
                "severity":        "HIGH" if outlier_count/len(df_coerced) > 0.05 else "LOW",
                "recommendation":  "Cap outliers at IQR bounds"
            })

    # ── ISSUE 5: Formatting inconsistencies ──────────────────────────────────
    for col in df.select_dtypes(include=['object','string']).columns:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        lower_uniq    = sample.str.lower().nunique()
        original_uniq = sample.nunique()
        if lower_uniq < original_uniq:
            issues["formatting_inconsistencies"].append({
                "column":                        col,
                "original_unique_values":        int(original_uniq),
                "standardized_unique_values":    int(lower_uniq),
                "extra_duplicates_due_to_casing":int(original_uniq - lower_uniq),
                "severity":                      "LOW",
                "recommendation":                f"Standardize text case in '{col}'"
            })

    # ── ISSUE 6: Low variance ─────────────────────────────────────────────────
    for col in numeric_cols:
        col_data = df_coerced[col].dropna()
        if len(col_data) == 0:
            continue
        if col_data.std() < 0.001 or col_data.nunique() == 1:
            issues["low_variance_columns"].append({
                "column":         col,
                "unique_values":  int(col_data.nunique()),
                "std_dev":        round(float(col_data.std()), 6),
                "severity":       "MEDIUM",
                "recommendation": f"Consider removing '{col}' — adds no predictive value"
            })

    # ── ISSUE 7: Domain range violations (BUG FIX 6) ─────────────────────────
    # These catch things IQR MISSES — e.g. indian_popularity = -3 has
    # IQR lower bound of -5.5 so IQR would NOT flag it. Domain rule catches it.
    for col, bounds in DOMAIN_RULES["numeric_ranges"].items():
        if col not in df_coerced.columns:
            continue
        col_data = df_coerced[col].dropna()
        viol_low  = int((col_data < bounds["min"]).sum())
        viol_high = int((col_data > bounds["max"]).sum())
        total     = viol_low + viol_high
        if total > 0:
            ex_low  = df_coerced.loc[df_coerced[col] < bounds["min"], col].head(3).tolist()
            ex_high = df_coerced.loc[df_coerced[col] > bounds["max"], col].head(3).tolist()
            issues["domain_range_violations"].append({
                "column":            col,
                "valid_min":         bounds["min"],
                "valid_max":         bounds["max"],
                "violations_below":  viol_low,
                "violations_above":  viol_high,
                "total_violations":  total,
                "examples_too_low":  str(ex_low),
                "examples_too_high": str(ex_high),
                "severity":          "HIGH",
                "recommendation":    f"Clamp '{col}' to [{bounds['min']}, {bounds['max']}]"
            })

    # ── ISSUE 8: Invalid categories (BUG FIX 7) ──────────────────────────────
    for col, valid_vals in DOMAIN_RULES["valid_categories"].items():
        if col not in df.columns:
            continue
        col_str      = df[col].dropna().astype(str).str.strip()
        invalid_mask = ~col_str.isin(valid_vals)
        invalid_count= int(invalid_mask.sum())
        if invalid_count > 0:
            issues["invalid_categories"].append({
                "column":           col,
                "invalid_count":    invalid_count,
                "invalid_percent":  round(invalid_count / len(df) * 100, 2),
                "invalid_examples": str(col_str[invalid_mask].unique()[:5].tolist()),
                "valid_options_sample": str(valid_vals[:6]) + "...",
                "severity":         "HIGH",
                "recommendation":   f"Remove or correct invalid '{col}' values"
            })

    return issues

# =============================================================================
# MODULE 4: QUALITY SCORE — updated to include domain + category issues
# =============================================================================

def calculate_quality_score(df, issues):
    score      = 100
    deductions = {}

    def deduct(key, label, amount):
        nonlocal score
        d = round(min(amount[0], amount[1]), 1)
        score -= d
        deductions[label] = d

    if issues["missing_values"]:
        worst = max(i["missing_percent"] for i in issues["missing_values"])
        deduct(None, "Missing Values",     (min(20, worst * 0.5), 20))
    if issues["duplicate_rows"]:
        deduct(None, "Duplicate Rows",    (min(15, issues["duplicate_rows"][0]["percent"]*1.5), 15))
    if issues["outliers_iqr"]:
        worst = max(i["outlier_percent"] for i in issues["outliers_iqr"])
        deduct(None, "IQR Outliers",      (min(10, worst * 0.5), 10))
    if issues["type_mismatches"]:
        deduct(None, "Type Mismatches",   (min(10, len(issues["type_mismatches"])*5), 10))
    if issues["formatting_inconsistencies"]:
        deduct(None, "Formatting Issues", (min(8,  len(issues["formatting_inconsistencies"])*4), 8))
    if issues["low_variance_columns"]:
        deduct(None, "Low Variance",      (min(5,  len(issues["low_variance_columns"])*3), 5))
    if issues["domain_range_violations"]:
        total = sum(i["total_violations"] for i in issues["domain_range_violations"])
        deduct(None, "Domain Violations", (min(15, total * 2), 15))
    if issues["invalid_categories"]:
        deduct(None, "Invalid Categories",(min(15, len(issues["invalid_categories"])*7), 15))

    score = max(0, round(score, 1))
    grade = ("Excellent"           if score >= 85 else
             "Good"                if score >= 70 else
             "Needs Cleaning"      if score >= 50 else
             "Poor — Major Issues")
    return score, grade, deductions

# =============================================================================
# MODULE 5: ISOLATION FOREST (BUG FIX 10 — contamination is now a parameter)
# =============================================================================

def run_isolation_forest(df, contamination=0.05):
    """
    Isolation Forest finds contextually anomalous rows.
    contamination = expected fraction of anomalous rows (user-controlled via slider).

    HOW IT WORKS:
    Randomly isolates data points by picking a random feature and a random split.
    Anomalies are isolated quickly (few splits needed — they are already unusual).
    Normal points need many splits to isolate (they cluster with similar rows).
    Score = average path length. Short path = anomaly.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return None, []

    filled = numeric_df.fillna(numeric_df.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)

    iso   = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=42, n_jobs=-1)
    preds  = iso.fit_predict(scaled)
    scores = iso.score_samples(scaled)

    result             = df.copy()
    result['anomaly_flag']  = preds
    result['anomaly_score'] = scores
    result['is_anomaly']    = preds == -1

    return result, list(result[result['is_anomaly']].index)

# =============================================================================
# MODULE 6: AUTO CLEANER — fixed all 5 cleaning bugs
#
# CLEANING ORDER MATTERS:
#   1. Coerce type mismatches first  (so numeric ops work on them)
#   2. Remove duplicates             (before filling — avoid filling dupes)
#   3. Fill missing values           (after type coercion so medians are correct)
#   4. Clamp domain violations       (BUG FIX 4 — was completely absent before)
#   5. Cap IQR outliers              (only for columns WITHOUT domain rules)
#   6. Remove invalid category rows  (BUG FIX 7)
#   7. Standardize text case
# =============================================================================

def auto_clean(df, issues):
    cleaned  = df.copy()
    log      = []

    # Step 1 — Type coercion (BUG FIX 5 — was completely missing before)
    for issue in issues["type_mismatches"]:
        col = issue["column"]
        if col not in cleaned.columns:
            continue
        before_nulls = cleaned[col].isnull().sum()
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        new_nulls    = cleaned[col].isnull().sum() - before_nulls
        log.append({
            "action":        "Type Coercion",
            "details":       f"'{col}': converted to numeric; {new_nulls} uncoercible values → NaN",
            "rows_affected": int(issue["bad_value_count"]),
            "severity":      "MEDIUM"
        })

    # Step 2a — Remove invalid category rows FIRST (before filling missing)
    # Removing bad rows first means we don't fill NaNs in rows we are about to delete,
    # and NaNs created by type coercion in bad rows don't inflate the missing count.
    for _ic_issue in issues["invalid_categories"]:
        _col = _ic_issue["column"]
        if _col not in cleaned.columns:
            continue
        _valid_vals = DOMAIN_RULES["valid_categories"].get(_col, [])
        _before = len(cleaned)
        cleaned = cleaned[
            cleaned[_col].isna() | cleaned[_col].astype(str).str.strip().isin(_valid_vals)
        ]
        _removed = _before - len(cleaned)
        if _removed > 0:
            log.append({"action":"Removed Invalid Categories",
                        "details":f"'{_col}': removed {_removed} rows with invalid values",
                        "rows_affected":_removed,"severity":"HIGH"})

    # Step 2b — Remove duplicates
    before  = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)
    if removed > 0:
        log.append({
            "action": "Removed Duplicates",
            "details": f"Removed {removed} exact duplicate rows",
            "rows_affected": removed, "severity": "HIGH"
        })

    # Step 3 — Fill missing values (after row removal — accurate counts)
    for issue in issues["missing_values"]:
        col = issue["column"]
        if col not in cleaned.columns:
            continue
        if issue["missing_percent"] > 40:
            cleaned = cleaned.drop(columns=[col])
            log.append({"action":"Dropped Column",
                        "details":f"'{col}' had {issue['missing_percent']}% missing — dropped",
                        "rows_affected":issue["missing_count"],"severity":"HIGH"})
        else:
            if cleaned[col].dtype in ['int64','float64']:
                fv = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(fv)
                log.append({"action":"Filled Missing",
                            "details":f"'{col}': filled {issue['missing_count']} NaNs with median ({round(float(fv),3)})",
                            "rows_affected":issue["missing_count"],"severity":issue["severity"]})
            else:
                fv = cleaned[col].mode()[0] if len(cleaned[col].mode()) > 0 else "Unknown"
                cleaned[col] = cleaned[col].fillna(fv)
                log.append({"action":"Filled Missing",
                            "details":f"'{col}': filled {issue['missing_count']} NaNs with mode ('{fv}')",
                            "rows_affected":issue["missing_count"],"severity":issue["severity"]})

    # Step 4 — Domain range clamping (BUG FIX 4 — was completely absent before)
    domain_cols = set()
    for issue in issues["domain_range_violations"]:
        col = issue["column"]
        if col not in cleaned.columns:
            continue
        lo, hi = issue["valid_min"], issue["valid_max"]
        viol_count = int(((cleaned[col] < lo) | (cleaned[col] > hi)).sum())
        cleaned[col] = cleaned[col].clip(lower=lo, upper=hi)
        domain_cols.add(col)
        log.append({"action":"Domain Clamp",
                    "details":f"'{col}': clamped {viol_count} values to valid range [{lo}, {hi}]",
                    "rows_affected":viol_count,"severity":"HIGH"})

    # Step 5 — IQR capping (only for columns NOT already handled by domain rules)
    for issue in issues["outliers_iqr"]:
        col = issue["column"]
        if col not in cleaned.columns or col in domain_cols:
            continue
        cleaned[col] = cleaned[col].clip(lower=issue["lower_bound"], upper=issue["upper_bound"])
        log.append({"action":"IQR Cap",
                    "details":f"'{col}': capped {issue['outlier_count']} outliers to [{issue['lower_bound']}, {issue['upper_bound']}]",
                    "rows_affected":issue["outlier_count"],"severity":issue["severity"]})

    # Step 6 — Standardize text case
    for issue in issues["formatting_inconsistencies"]:
        col = issue["column"]
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.title()
            log.append({"action":"Standardized Case",
                        "details":f"'{col}': converted to Title Case",
                        "rows_affected":issue["extra_duplicates_due_to_casing"],"severity":"LOW"})

    # Step final — Catch any NaNs created by type coercion that weren't in original missing list
    # (e.g. 'many' in player_count_millions was object before — not detected as missing
    #  but becomes NaN after coercion in step 1)
    for col in cleaned.columns:
        remaining_nulls = cleaned[col].isnull().sum()
        if remaining_nulls > 0:
            if cleaned[col].dtype in ['int64','float64']:
                fv = cleaned[col].median()
                if pd.notna(fv):
                    cleaned[col] = cleaned[col].fillna(fv)
                    log.append({"action":"Post-Coercion Fill",
                                "details":f"'{col}': filled {remaining_nulls} NaN(s) created by type coercion, median={round(float(fv),3)}",
                                "rows_affected":int(remaining_nulls),"severity":"LOW"})

    return cleaned, log

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def main():
    st.set_page_config(
        page_title="Data Quality Auditor",
        page_icon="🔍",
        layout="wide"
    )

    # ── Make tab labels bigger and more prominent ─────────────────────────────
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: #1e1e2e;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565C0 !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 Universal Data Quality Auditor & Auto-Cleaning Pipeline")
    st.markdown(
        "**Upload any CSV or Excel file. Get a full 8-issue quality audit, "
        "domain validation, ML anomaly detection, and a cleaned dataset — all in seconds.**  \n"
        "Built by **Kartikeya Warhade**"
    )
    st.markdown("---")

    # contamination fixed at 5% — sidebar removed for simplicity
    contamination = 0.05

    # ── FILE UPLOAD ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Drop any CSV or Excel file here",
        type=["csv","xlsx","xls"]
    )

    if uploaded_file is None:
        st.info("Upload any CSV or Excel file to begin the quality audit.")
        st.markdown("""
        | # | Issue Type | Detection Method |
        |---|------------|-----------------|
        | 1 | Missing Values | Null count per column |
        | 2 | Duplicate Rows | Exact row hash matching |
        | 3 | IQR Outliers | Q1−1.5×IQR / Q3+1.5×IQR |
        | 4 | Type Mismatches | Numeric coercion test — reports exact bad values |
        | 5 | Formatting Issues | Case normalisation check |
        | 6 | Low Variance | Std dev < 0.001 |
        | 7 | Domain Range Violations | Predefined valid min/max per column |
        | 8 | Invalid Categories | Known valid values list for platform/genre |
        | + | ML Anomalies | Isolation Forest unsupervised detection |
        """)
        return

    # ── LOAD FILE ─────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        df = (pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv')
              else pd.read_excel(uploaded_file))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.success(f"Loaded in {round(time.time()-t0,2)}s — {len(df):,} rows × {len(df.columns)} columns")

    # ── SCHEMA VALIDATION (BUG FIX 9) ────────────────────────────────────────
    is_valid, missing_cols, extra_cols, schema_warnings = validate_schema(df)
    for w in schema_warnings:
        if "Missing" in w:
            st.warning(f"⚠️ Schema Warning: {w}")
        else:
            st.info(f"ℹ️ {w}")

    # ── RUN FULL ANALYSIS ─────────────────────────────────────────────────────
    with st.spinner("Running full 8-issue quality analysis + ML anomaly detection..."):
        t1      = time.time()
        profile = profile_dataframe(df)
        issues  = detect_issues(df)
        score, grade, deductions = calculate_quality_score(df, issues)
        anomaly_df, anomaly_idx  = run_isolation_forest(df, contamination)
        elapsed = round(time.time()-t1, 2)

    st.success(f"Analysis complete in {elapsed}s")

    # ── QUALITY SCORE CARDS (BUG FIX 8 — HTML cards, no white boxes) ─────────
    st.header("📊 Data Quality Report")

    total_issues = sum(len(v) for v in issues.values())
    score_bg  = "#2E7D32" if score >= 85 else "#E65100" if score >= 60 else "#B71C1C"

    st.markdown(
        cards_row(
            info_card(f"{score}/100", "Quality Score",   score_bg),
            info_card(grade,           "Grade",           "#1565C0"),
            info_card(str(total_issues),"Issues Found",  "#37474F"),
            info_card(f"{len(anomaly_idx)}", "ML Anomalies", "#4A148C" if anomaly_idx else "#2E7D32"),
            info_card(f"{len(df):,}",  "Total Rows",     "#00695C"),
        ),
        unsafe_allow_html=True
    )

    # Score deduction chart
    if deductions:
        fig, ax = plt.subplots(figsize=(9, max(2.5, len(deductions)*0.45)))
        colors = ['#EF5350'] * len(deductions)
        ax.barh(list(deductions.keys()), list(deductions.values()),
                color=colors, edgecolor='white', alpha=0.9)
        ax.set_xlabel("Points Deducted")
        ax.set_title("Quality Score Deductions by Issue Type", fontsize=12, fontweight='bold')
        for i,(k,v) in enumerate(deductions.items()):
            ax.text(v+0.1, i, f"−{v}", va='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── ISSUE TABS ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "❌ Missing & Dupes",
        "📈 Outliers & Types",
        "🚨 Domain & Categories",
        "🤖 ML Anomalies",
        "✅ Cleaned Data",
        "📋 Full Profile"
    ])

    # ── TAB 1: Missing + Duplicates ───────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing Values")
            if issues["missing_values"]:
                st.dataframe(pd.DataFrame(issues["missing_values"]),
                             use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(6,3))
                md = {i["column"]:i["missing_percent"] for i in issues["missing_values"]}
                ax.bar(md.keys(), md.values(), color='#EF5350', edgecolor='white')
                ax.set_ylabel("Missing %")
                ax.set_title("Missing Values by Column", fontsize=11, fontweight='bold')
                plt.xticks(rotation=30, ha='right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            else:
                st.success("✅ No missing values found.")

        with c2:
            st.subheader("Duplicate Rows")
            if issues["duplicate_rows"]:
                st.dataframe(pd.DataFrame(issues["duplicate_rows"]),
                             use_container_width=True, hide_index=True)
                st.warning(
                    f"**{issues['duplicate_rows'][0]['count']} duplicate rows** detected. "
                    "Will be removed during auto-cleaning."
                )
            else:
                st.success("✅ No duplicate rows found.")

            st.subheader("Formatting Issues")
            if issues["formatting_inconsistencies"]:
                st.dataframe(pd.DataFrame(issues["formatting_inconsistencies"]),
                             use_container_width=True, hide_index=True)
            else:
                st.success("✅ No formatting inconsistencies.")

    # ── TAB 2: Outliers + Type Mismatches ────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("IQR Outliers")
            st.caption("Detects values beyond 1.5× the interquartile range.")
            if issues["outliers_iqr"]:
                st.dataframe(pd.DataFrame(issues["outliers_iqr"]),
                             use_container_width=True, hide_index=True)
            else:
                st.success("✅ No IQR outliers found.")

            st.subheader("Low Variance Columns")
            if issues["low_variance_columns"]:
                st.dataframe(pd.DataFrame(issues["low_variance_columns"]),
                             use_container_width=True, hide_index=True)
            else:
                st.success("✅ No low variance columns.")

        with c2:
            st.subheader("Type Mismatches")
            st.caption("Columns storing text where numbers are expected — shows exact bad values.")
            if issues["type_mismatches"]:
                st.dataframe(pd.DataFrame(issues["type_mismatches"]),
                             use_container_width=True, hide_index=True)
                st.warning(
                    "These columns contain text where numbers are expected. "
                    "Auto-clean will coerce them — uncoercible values become NaN."
                )
            else:
                st.success("✅ No type mismatches found.")

    # ── TAB 3: Domain Violations + Invalid Categories (NEW) ──────────────────
    with tab3:
        st.subheader("🚨 Domain Range Violations")
        st.markdown(
            "Values outside the **physically valid range** for that column. "
            "IQR cannot reliably catch these — e.g. `indian_popularity = -3` "
            "has IQR lower bound of −5.5, so IQR would **miss it**. "
            "Domain rules always catch it."
        )
        if issues["domain_range_violations"]:
            st.dataframe(pd.DataFrame(issues["domain_range_violations"]),
                         use_container_width=True, hide_index=True)

            # Show the actual offending rows
            df_temp = df.copy()
            for col in df_temp.select_dtypes(include=['object','string']).columns:
                attempt = pd.to_numeric(df_temp[col], errors='coerce')
                if attempt.notna().mean() > 0.7:
                    df_temp[col] = attempt

            for issue in issues["domain_range_violations"]:
                col = issue["column"]
                if col not in df_temp.columns:
                    continue
                lo, hi   = issue["valid_min"], issue["valid_max"]
                bad_rows = df_temp[
                    (df_temp[col] < lo) | (df_temp[col] > hi)
                ]
                name_cols = ["name", col] if "name" in df_temp.columns else [col]
                with st.expander(
                    f"📍 {len(bad_rows)} violating rows — {col} (valid: {lo}–{hi})"
                ):
                    st.dataframe(bad_rows[name_cols].head(10),
                                 use_container_width=True, hide_index=True)
        else:
            st.success("✅ No domain range violations found.")

        st.markdown("---")
        st.subheader("🏷️ Invalid Category Values")
        st.markdown(
            "Values in categorical columns that are **not in the known valid list**. "
            "Examples: `'Smart Fridge'` as platform, `'Shooterish'` as genre."
        )
        if issues["invalid_categories"]:
            st.dataframe(pd.DataFrame(issues["invalid_categories"]),
                         use_container_width=True, hide_index=True)
            for issue in issues["invalid_categories"]:
                col       = issue["column"]
                valid_vals = DOMAIN_RULES["valid_categories"].get(col, [])
                if col not in df.columns:
                    continue
                bad_rows = df[
                    ~df[col].isna() &
                    ~df[col].astype(str).str.strip().isin(valid_vals)
                ]
                with st.expander(
                    f"📍 {len(bad_rows)} rows with invalid '{col}'"
                ):
                    cols_show = ["name", col] if "name" in df.columns else [col]
                    st.dataframe(bad_rows[cols_show].head(10),
                                 use_container_width=True, hide_index=True)
        else:
            st.success("✅ No invalid category values found.")

    # ── TAB 4: ML Anomalies ───────────────────────────────────────────────────
    with tab4:
        st.subheader("Isolation Forest ML Anomaly Detection")
        st.markdown(
            "Finds rows that are **contextually anomalous** across ALL numeric features together. "
            "A row can look normal column-by-column but be anomalous in combination. "
            "Isolation Forest algorithm — 100 trees, 5% contamination threshold."
        )
        if anomaly_df is not None and len(anomaly_idx) > 0:
            st.warning(f"{len(anomaly_idx)} anomalous rows detected.")
            anomaly_rows = (
                anomaly_df[anomaly_df['is_anomaly']]
                .drop(columns=['anomaly_flag','is_anomaly'])
                .sort_values('anomaly_score')
            )
            st.dataframe(anomaly_rows.head(30), use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8,3))
            ax.hist(anomaly_df['anomaly_score'], bins=30,
                    color='#7B1FA2', edgecolor='white', alpha=0.85)
            threshold = anomaly_df[anomaly_df['is_anomaly']]['anomaly_score'].max()
            ax.axvline(threshold, color='red', linestyle='--', label='Anomaly threshold')
            ax.set_xlabel("Anomaly Score (lower = more anomalous)")
            ax.set_title("Score Distribution", fontsize=11, fontweight='bold')
            ax.legend(); plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.success("✅ No significant anomalies at current sensitivity.")

    # ── TAB 5: Cleaned Data ───────────────────────────────────────────────────
    with tab5:
        st.subheader("Auto-Cleaned Dataset")
        with st.spinner("Cleaning in progress..."):
            cleaned_df, audit_log = auto_clean(df, issues)
        st.success(f"✅ Cleaning complete — {len(audit_log)} changes applied.")

        # Before / After — HTML cards, no white boxes
        st.markdown(
            cards_row(
                info_card(f"{len(df):,}",               "Original Rows",    "#37474F"),
                info_card(f"{len(cleaned_df):,}",       "Cleaned Rows",     "#2E7D32"),
                info_card(str(len(df.columns)),          "Original Columns", "#37474F"),
                info_card(str(len(cleaned_df.columns)), "Cleaned Columns",  "#1565C0"),
                info_card(str(df.isnull().sum().sum()),  "Nulls Before",     "#B71C1C"),
                info_card(str(cleaned_df.isnull().sum().sum()), "Nulls After","#2E7D32"),
            ),
            unsafe_allow_html=True
        )

        # Audit log with dark colour coding — readable text
        st.subheader("Audit Log — Every Change Made")
        if audit_log:
            adf = pd.DataFrame(audit_log)
            def color_severity(row):
                c = ('#B71C1C' if row.get('severity')=='HIGH' else
                     '#E65100' if row.get('severity')=='MEDIUM' else '#1B5E20')
                return [f'background-color:{c}; color:white']*len(row)
            st.dataframe(adf.style.apply(color_severity, axis=1),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No changes were made — data was already clean.")

        st.markdown("---")

        # ── DOWNLOAD SECTION ──────────────────────────────────────────────────
        st.subheader("Downloads")
        dl1, dl2 = st.columns(2)

        with dl1:
            st.markdown("**Cleaned Dataset (CSV)**")
            buf = io.StringIO()
            cleaned_df.to_csv(buf, index=False)
            fname = uploaded_file.name.replace('.xlsx','.csv').replace('.xls','.csv')
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=buf.getvalue(),
                file_name=f"cleaned_{fname}",
                mime="text/csv",
                use_container_width=True
            )

        with dl2:
            st.markdown("**Data Quality Report (JSON)**")
            import json
            # Convert numpy types to plain Python so JSON serializer doesn't crash
            clean_deductions = {k: float(v) for k, v in deductions.items()}
            clean_log = [{k: (int(v) if hasattr(v,'item') else v) for k,v in e.items()} for e in audit_log]
            report = {
                "report_generated":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file":         uploaded_file.name,
                "rows_processed":      len(df),
                "clean_rows":          len(cleaned_df),
                "rows_removed":        len(df) - len(cleaned_df),
                "columns":             len(df.columns),
                "data_quality_score":  score,
                "grade":               grade,
                "issues_detected": {
                    "missing_values":          len(issues["missing_values"]),
                    "duplicate_rows":          issues["duplicate_rows"][0]["count"] if issues["duplicate_rows"] else 0,
                    "iqr_outliers":            len(issues["outliers_iqr"]),
                    "type_mismatches":         len(issues["type_mismatches"]),
                    "formatting_issues":       len(issues["formatting_inconsistencies"]),
                    "low_variance_columns":    len(issues["low_variance_columns"]),
                    "domain_range_violations": len(issues["domain_range_violations"]),
                    "invalid_categories":      len(issues["invalid_categories"]),
                    "ml_anomalies":            len(anomaly_idx),
                },
                "score_deductions":    clean_deductions,
                "cleaning_actions":    len(audit_log),
                "audit_log":           clean_log,
            }
            st.download_button(
                label="⬇️ Download Quality Report JSON",
                data=json.dumps(report, indent=2),
                file_name=f"quality_report_{fname.replace('.csv','')}.json",
                mime="application/json",
                use_container_width=True
            )

        st.subheader("Cleaned Data Preview (first 50 rows)")
        st.dataframe(cleaned_df.head(50), use_container_width=True, hide_index=True)

    # ── TAB 6: Full Data Profile ──────────────────────────────────────────────
    with tab6:
        st.subheader("Column-by-Column Data Profile")
        rows = []
        for col, p in profile.items():
            rows.append({
                "Column":        p["column_name"],
                "Type":          p["data_type"],
                "Missing":       p["missing_count"],
                "Missing %":     p["missing_percent"],
                "Unique Values": p["unique_count"],
                "Mean":          p["mean"],
                "Median":        p["median"],
                "Std Dev":       p["std"],
                "Min":           p["min"],
                "Max":           p["max"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
