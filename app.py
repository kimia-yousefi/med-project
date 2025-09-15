"""
Surrogacy matching app (Exact implementation of the weighted static scoring + dynamic euclidean method).

Required:
- dynamicData.xlsx in same folder
- staticData.xlsx in same folder

Libraries:
pip install pandas numpy flask openpyxl
"""

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import math
import os
import json

# ---------------------------
# CONFIG: paths to Excel files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_DATA_FILE = os.path.join(BASE_DIR, 'dynamicData.xlsx')
STATIC_DATA_FILE = os.path.join(BASE_DIR, 'staticData.xlsx')

# Relative importance between static and dynamic parts (can be chamge)
STATIC_WEIGHT = 1.0
DYNAMIC_WEIGHT = 1.0
# Final score = STATIC_WEIGHT * static_distance + DYNAMIC_WEIGHT * dynamic_distance
# Lower final score is better.

# ---------------------------
# Helper to find possible column names (Persian/English tolerant)
def find_col(df, candidates, required=True):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    if required:
        raise ValueError(f"Required column not found. Tried: {candidates}. Available cols: {list(df.columns)}")
    return None

def is_true_like(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "بله", "طبیعی", "نرمال", "ok", "okey")

# ---------------------------
# Load and merge data from two separate Excel files
def load_donors():
    if not os.path.exists(DYNAMIC_DATA_FILE):
        raise FileNotFoundError(f"{DYNAMIC_DATA_FILE} not found. Please place this Excel file in the same folder.")
    if not os.path.exists(STATIC_DATA_FILE):
        raise FileNotFoundError(f"{STATIC_DATA_FILE} not found. Please place this Excel file in the same folder.")

    # Load data, skipping the complex 2-row headers and assigning names manually
    df_dynamic = pd.read_excel(DYNAMIC_DATA_FILE, header=None, skiprows=2)
    df_static = pd.read_excel(STATIC_DATA_FILE, header=None, skiprows=2)

    # --- Data Cleaning and Preparation ---

    # 1. Manually assign clean column names
    df_dynamic.columns = ['Name', 'Location', 'Payment', 'Education', 'Marriage', 'ethnicity', 'religion', 'faith']
    df_static.columns = ['Name', 'Age', 'BMI','Anatomy', 'endometrium', 'hormonal factor', 'Para', 'Abort', 
        'Infertility', 'Fetal Death', 'Gravida', 'Premature', 'Complications', 'Ep', 'C/S',
        'chronic_illness', 'smoke', 'infection', 'EF', 'GFR',
        'autoimmune', 'MMPI', 'MPSS'
    ]

    
    df_static = df_static.drop(columns=['Name'])

    # 2. Parse the 'Location' column into 'city' and 'province'
    if 'Location' in df_dynamic.columns:
        # The Persian comma is '،'
        split_locs = df_dynamic['Location'].str.split('،', n=1, expand=True)
        df_dynamic['city'] = split_locs[0].str.strip()
        df_dynamic['province'] = split_locs[1].str.strip() if split_locs.shape[1] > 1 else ''

    # 3. Clean the 'Payment' column (remove commas and convert to number)
    if 'Payment' in df_dynamic.columns:
        df_dynamic['Payment'] = df_dynamic['Payment'].astype(str).str.replace(',', '').astype(float)


    # Merge the two dataframes side-by-side.
    # This assumes that row N in the static file corresponds to row N in the dynamic file.
    if len(df_dynamic) != len(df_static):
        print(f"Warning: Dynamic data has {len(df_dynamic)} rows but static data has {len(df_static)} rows. Merging might be incorrect.")

    df_merged = pd.concat([df_dynamic, df_static], axis=1)

    # Rename the identifier column to 'donor_id' for compatibility with the rest of the code.
    df_merged = df_merged.rename(columns={'Name': 'donor_id'})

    return df_merged

# ---------------------------
# Scoring functions for static features (return 0..100)

# 1) Age mapping
AGE_TABLE = {
    21:100,22:98,23:96,24:94,25:92,26:90,27:88,28:86,29:84,30:80,
    31:76,32:72,33:68,34:64,35:60,36:53,37:46,38:39,39:32,40:25,41:18,42:11
}
def score_age(age):
    try:
        age = float(age)
    except:
        return 0.0
    if age < 21:
        return 0.0
    if age >= 42:
        return 11.0 if age == 42 else 0.0
    lo = math.floor(age)
    hi = math.ceil(age)
    lo_sc = AGE_TABLE.get(int(lo), 0)
    hi_sc = AGE_TABLE.get(int(hi), 0)
    if lo == hi:
        return float(lo_sc)
    frac = age - lo
    return lo_sc + (hi_sc - lo_sc) * frac

# 2) BMI buckets
def score_bmi(bmi):
    try:
        bmi = float(bmi)
    except:
        return 0.0
    if bmi < 19: return 87.5
    if 19 <= bmi <= 25: return 100.0
    if 25 < bmi <= 30: return 93.75
    if 30 < bmi <= 35: return 84.375
    if 35 < bmi <= 40: return 81.25
    if 40 < bmi <= 45: return 75.0
    if 45 < bmi <= 50: return 71.8
    return 62.5

# 3) Uterus health
def score_uterus(row, colmap):
    anatomy_col = find_col(row.index.to_series().to_frame().T, [colmap.get('anatomy', 'Anatomy'), 'Anatomy', 'ساختار رحم', 'anatomy_status'], required=False)
    endometrium_col = find_col(row.index.to_series().to_frame().T, [colmap.get('endometrium', 'endometrium'), 'endometrium', 'آندومتر'], required=False)
    hormonal_col = find_col(row.index.to_series().to_frame().T, [colmap.get('hormonal_factor', 'hormonal factor'), 'hormonal factor', 'پروژسترون', 'عامل هورمونی'], required=False)

    anatomy_val = row.get(anatomy_col, None)
    anatomy_score = 30.0 if is_true_like(anatomy_val) or str(anatomy_val).strip().lower() in ('طبیعی', 'normal', 'نرمال') else 0.0

    endometrium_val = row.get(endometrium_col, None)
    endometrium_score = 10.0 if is_true_like(endometrium_val) or str(endometrium_val).strip().lower() in ('طبیعی', 'normal', 'نرمال') else 0.0

    hormonal_val = row.get(hormonal_col, None)
    hormonal_score = 20.0 if is_true_like(hormonal_val) or str(hormonal_val).strip().lower() in ('مطلوب', 'desired', 'ok', 'good') else 0.0

    return float(anatomy_score + endometrium_score + hormonal_score)

# 4) Pregnancy history
def score_preg_history(row, colmap):
    para_col = find_col(row.index.to_series().to_frame().T, [colmap.get('para','para'),'para','PARA','زایمان_موفق','para_count'], required=False)
    abort_col = find_col(row.index.to_series().to_frame().T, [colmap.get('abort','abort'),'abort','سقط'], required=False)
    infertility_col = find_col(row.index.to_series().to_frame().T, [colmap.get('infertility_years','infertility_years'),'infertility','ناباروری'], required=False)
    fetal_col = find_col(row.index.to_series().to_frame().T, [colmap.get('fetal_death','fetal_death'),'fetal_death','مرده زایی'], required=False)
    gravida_col = find_col(row.index.to_series().to_frame().T, [colmap.get('gravida','gravida'),'gravida','N.O gravida','تعداد بارداری'], required=False)
    premature_col = find_col(row.index.to_series().to_frame().T, [colmap.get('premature','premature'),'premature','زایمان_زودرس'], required=False)
    complications_col = find_col(row.index.to_series().to_frame().T, [colmap.get('complications','complications'),'complications','مشکلات'], required=False)
    ectopic_col = find_col(row.index.to_series().to_frame().T, [colmap.get('ectopic','EP'),'EP','ectopic','بیرون رحمی'], required=False)
    cs_col = find_col(row.index.to_series().to_frame().T, [colmap.get('c_section','C/S'),'C/S','c_section','سزارین'], required=False)

    try: para = int(float(row.get(para_col, 0) or 0))
    except: para = 0
    if para >= 2: para_score = 25.0
    elif para == 1: para_score = 20.0
    else:
        try: grava = int(float(row.get(gravida_col, 0) or 0))
        except: grava = 0
        para_score = 10.0 if grava > 0 else 0.0

    try: aborts = int(float(row.get(abort_col, 0) or 0))
    except: aborts = 0
    if aborts == 0: abort_score = 15.0
    elif aborts == 1: abort_score = 12.0
    elif aborts == 2: abort_score = 8.0
    elif aborts == 3: abort_score = 4.0
    else: abort_score = 0.0

    # Infertility
    infertility_val = row.get(infertility_col, '')
    infertility_str = str(infertility_val).strip().lower()
    if infertility_str in ('بدون سابقه', 'no history', 'none', ''):
        infert_score = 10.0
    elif infertility_str in ('باسابقه', 'with history'):
        infert_score = 0.0  
    else:
        try:
            infertility_years = float(infertility_val)
            if infertility_years <= 1: infert_score = 10.0
            elif 1 < infertility_years <= 2: infert_score = 7.0
            elif 2 < infertility_years <= 4: infert_score = 3.0
            else: infert_score = 0.0
        except:
            infert_score = 0.0 

    try: fetal = int(float(row.get(fetal_col, 0) or 0))
    except: fetal = 0
    if fetal == 0: fetal_score = 10.0
    elif fetal == 1: fetal_score = 5.0
    else: fetal_score = 0.0

    try: gravida = int(float(row.get(gravida_col, 0) or 0))
    except: gravida = 0
    if 1 <= gravida <= 3: gravida_score = 10.0
    elif 4 <= gravida <= 5: gravida_score = 7.0
    elif gravida >= 6: gravida_score = 3.0
    else: gravida_score = 0.0

    try: prem = int(float(row.get(premature_col, 0) or 0))
    except: prem = 0
    if prem == 0: prem_score = 10.0
    elif prem == 1: prem_score = 6.0
    else: prem_score = 2.0

    try: comp = int(float(row.get(complications_col, 0) or 0))
    except: comp = 0
    if comp == 0: comp_score = 10.0
    elif comp == 1: comp_score = 5.0
    else: comp_score = 0.0

    try: ep = int(float(row.get(ectopic_col, 0) or 0))
    except: ep = 0
    if ep == 0: ep_score = 5.0
    elif ep == 1: ep_score = 2.0
    else: ep_score = 0.0

    try: cs = int(float(row.get(cs_col, 0) or 0))
    except: cs = 0
    if cs <= 1: cs_score = 5.0
    elif cs == 2: cs_score = 3.0
    else: cs_score = 0.0

    total = para_score + abort_score + infert_score + fetal_score + gravida_score + prem_score + comp_score + ep_score + cs_score
    return float(total)

# 5) Physical health
def score_physical(row, colmap):
    chronic_col = find_col(row.index.to_series().to_frame().T, [colmap.get('chronic_illness', 'Chronic Illness'), 'Chronic Illness', 'بیماری مزمن'], required=False)
    smoke_col = find_col(row.index.to_series().to_frame().T, [colmap.get('smoke', 'Smoking / Drugs'), 'Smoking / Drugs', 'smoke', 'سیگار'], required=False)
    infection_col = find_col(row.index.to_series().to_frame().T, [colmap.get('infection', 'Infectious Disease'), 'Infectious Disease', 'بیماری عفونی'], required=False)
    ef_col = find_col(row.index.to_series().to_frame().T, [colmap.get('cardiovascular', 'Cardiovascular'), 'Cardiovascular', 'قلب و عروق'], required=False)
    gfr_col = find_col(row.index.to_series().to_frame().T, [colmap.get('kidney_health', 'Kidney Health'), 'Kidney Health', 'سلامت کلیه'], required=False)
    auto_col = find_col(row.index.to_series().to_frame().T, [colmap.get('immunological_disease', 'Immunological Disease'), 'Immunological Disease'], required=False)

    s = str(row.get(chronic_col, '')).strip().lower()
    if s in ('بدون بیماری مزمن'): 
        chronic_score = 25.0
    elif any(x in s for x in ['بیماری مزمن کنترل‌شده با دارو']): 
        chronic_score = 18.0
    elif any(x in s for x in ['بیماری مزمن با کنترل نسبی']): 
        chronic_score = 10.0
    else: 
        chronic_score = 0.0

    s = str(row.get(smoke_col, '')).strip().lower()
    if s in ('', 'هیچ گونه مصرف'): 
        smoke_score = 20.0
    elif any(x in s for x in ['مصرف گهگاهی سیگار یا الکل']): 
        smoke_score = 12.0
    elif any(x in s for x in ['مصرف روزانه سیگار یا الکل']): 
        smoke_score = 6.0
    else: 
        smoke_score = 0.0

    s = str(row.get(infection_col, '')).strip().lower()
    if s in ('بدون بیماری'): 
        inf_score = 20.0
    elif any(x in s for x in ['سابقه بیماری درمان شده یا غیر فعال']): 
        inf_score = 10.0
    else: 
        inf_score = 0.0

    ef_val = str(row.get(ef_col, '')).strip().lower()
    if ef_val in ('طبیعی', 'normal', 'نرمال'): 
        ef_score = 15.0
    elif ef_val in ('کاهش خفیف', 'mild reduction'): 
        ef_score = 10.0
    elif ef_val in ('کاهش متوسط', 'moderate reduction'): 
        ef_score = 5.0
    else: 
        ef_score = 0.0

    gfr_val = str(row.get(gfr_col, '')).strip().lower()
    if gfr_val in ('طبیعی', 'normal', 'نرمال'): 
        gfr_score = 10.0
    elif gfr_val in ('کاهش خفیف', 'mild reduction'): 
        gfr_score = 5.0
    elif gfr_val in ('کاهش متوسط', 'moderate reduction'): 
        gfr_score = 3.0
    else: 
        gfr_score = 0.0

    s = str(row.get(auto_col, '')).strip().lower()
    if s in ('بدون مشکل'): 
        auto_score = 10.0
    elif any(x in s for x in ['آلرژی یا بیماری خود ایمنی خفیف']): 
        auto_score = 5.0
    else: 
        auto_score = 0.0

    return float(chronic_score + smoke_score + inf_score + ef_score + gfr_score + auto_score)

# 6) Mental health (MMPI-2)
def score_mmpi(mmpi_val):
    try:
        t = float(mmpi_val)
        if 45 <= t <= 55: return 100.0
        if 40 <= t <= 44 or 56 <= t <= 60: return 80.0
        if 35 <= t <= 39 or 61 <= t <= 65: return 60.0
        if 30 <= t <= 34 or 66 <= t <= 70: return 40.0
        if 25 <= t <= 29 or 71 <= t <= 75: return 20.0
        return 0.0
    except:
        s = str(mmpi_val).strip().lower()
        if s in ('normal','نرمال'): return 100.0
        if s in ('near normal','نزدیک به نرمال'): return 80.0
        if s in ('slightly off','کمی دور از نرمال'): return 60.0
        if s in ('mild clinical','افزایش بالینی ملایم'): return 40.0
        if s in ('افزایش بالینی مشخص'): return 20.0
        return 0.0

# 7) Family support (MSPSS)
def score_support(mspss_val):
    try:
        v = float(mspss_val)
        if 1 <= v <= 7:
            if 6 <= v <= 7: return 100.0
            if 5 <= v < 6: return 80.0
            if 4 <= v < 5: return 60.0
            if 3 <= v < 4: return 40.0
            if 2 <= v < 3: return 20.0
            return 0.0
        if 0 <= v <= 100:
            return float(max(0.0, min(100.0, v)))
    except:
        s = str(mspss_val).strip().lower()
        if s in ('very high','خیلی بالا'): return 100.0
        if s in ('high','بالا'): return 80.0
        if s in ('medium','متوسط'): return 60.0
        if s in ('low','پایین'): return 40.0
        if s in ('very low','خیلی پایین'): return 20.0
        return 0.0

# ---------------------------
# Compute static score percent (0..100) given a donor row
WEIGHTS = {
    'age': 0.22, 'bmi': 0.13, 'preg_history': 0.18, 'physical': 0.18,
    'mental': 0.06, 'uterus': 0.18, 'support': 0.05
}
def compute_static_percent(row):
    age_col = find_col(row.index.to_series().to_frame().T, ['Age'], required=False)
    age_sc = score_age(row.get(age_col, np.nan))

    bmi_col = find_col(row.index.to_series().to_frame().T, ['BMI'], required=False)
    bmi_sc = score_bmi(row.get(bmi_col, np.nan))

    uterus_sc = score_uterus(row, {})
    preg_sc = score_preg_history(row, {})
    physical_sc = score_physical(row, {})

    mmpi_col = find_col(row.index.to_series().to_frame().T, ['MMPI','MMPI Result','MMPI_result'], required=False)
    mental_sc = score_mmpi(row.get(mmpi_col, None))

    support_col = find_col(row.index.to_series().to_frame().T, ['MPSS','MPSS Result'], required=False)
    support_sc = score_support(row.get(support_col, None))

    static_percent = (
        age_sc * WEIGHTS['age'] + bmi_sc * WEIGHTS['bmi'] +
        preg_sc * WEIGHTS['preg_history'] + physical_sc * WEIGHTS['physical'] +
        mental_sc * WEIGHTS['mental'] + uterus_sc * WEIGHTS['uterus'] +
        support_sc * WEIGHTS['support']
    )
    return float(static_percent), {
        'age_sc': age_sc, 'bmi_sc': bmi_sc, 'uterus_sc': uterus_sc,
        'preg_sc': preg_sc, 'physical_sc': physical_sc, 'mental_sc': mental_sc,
        'support_sc': support_sc
    }

# ---------------------------
# Dynamic distance between family input and donor row (normalized 0..1)
def compute_dynamic_distance(row, family_input):
    components = []
    colmap = {
        'city': 'city',
        'province': 'province',
        'Education': 'Education',
        'Marriage': 'Marriage',
        'ethnicity': 'ethnicity',
        'religion': 'religion',
        'faith': 'faith',
        'Payment': 'Payment'
    }

    # city and province
    donor_city = str(row.get(colmap.get('city', 'city'), '')).strip()
    donor_prov = str(row.get(colmap.get('province', 'province'), '')).strip()
    fam_city = family_input.get('city', '').strip()
    fam_prov = family_input.get('province', '').strip()

    if fam_city or fam_prov:
        if donor_city and fam_city and donor_city == fam_city:
            loc_d = 0.0  # samecity (value 3)
        elif donor_prov and fam_prov and donor_prov == fam_prov:
            loc_d = 0.5  # sameprovince (value 2)
        else:
            loc_d = 1.0  # far (value 1)
    else:
        loc_d = 0.0  
    components.append(loc_d)

    #payment
    try: donor_payment = float(row.get('Payment', np.nan)) 
    except: donor_payment = np.nan
    fam_budget_str = family_input.get('budget', '')
    if fam_budget_str is None or fam_budget_str == '':
        pay_d = 0.0
    else:
        try: fam_budget = float(fam_budget_str)
        except: fam_budget = 0.0
        all_pay_max = 1.0
        if not np.isnan(donor_payment): all_pay_max = max(all_pay_max, abs(donor_payment))
        all_pay_max = max(all_pay_max, abs(fam_budget))
        if all_pay_max > 0:
            diff = abs((donor_payment if not np.isnan(donor_payment) else 0.0) - fam_budget)
            pay_d = min(1.0, diff / all_pay_max)
        else: pay_d = 0.0
    components.append(pay_d)

    # Education mapping: 1=دیپلم, 2=فوق دیپلم, 3=لیسانس, 4=ارشد, 5=دکترا
    edu_map = {'دیپلم': 1, 'فوق دیپلم': 2, 'لیسانس': 3, 'ارشد': 4, 'دکترا': 5}
    donor_edu_str = str(row.get('Education', '')).strip()
    donor_edu = edu_map.get(donor_edu_str, np.nan)
    fam_edu_str = family_input.get('education', '')
    if fam_edu_str is None or fam_edu_str == '':
        edu_d = 0.0
    else:
        try: fam_edu = float(fam_edu_str)
        except: fam_edu = np.nan
        if np.isnan(donor_edu) or np.isnan(fam_edu): edu_d = 1.0
        else: edu_d = min(1.0, abs(donor_edu - fam_edu) / 4.0)
    components.append(edu_d)

    # Marriage mapping: 1=مطلقه, 2=متاهل
    mar_map = {'مطلقه': 1, 'متاهل': 2}
    donor_mar_str = str(row.get('Marriage', '')).strip()
    donor_mar = mar_map.get(donor_mar_str, np.nan)
    fam_mar_str = family_input.get('marriage', '')
    if fam_mar_str is None or fam_mar_str == '':
        mar_d = 0.0
    else:
        try: fam_mar = float(fam_mar_str)
        except: fam_mar = np.nan
        if np.isnan(donor_mar) or np.isnan(fam_mar): mar_d = 1.0
        else: mar_d = 0.0 if donor_mar == fam_mar else 1.0
    components.append(mar_d)

    # Socio-cultural
    fam_eth = family_input.get('ethnicity','').strip().lower()
    fam_religion = family_input.get('religion','').strip().lower()
    fam_faith = family_input.get('faith','').strip().lower()

    pref_eth_same = family_input.get('prefer_same_eth', False)
    pref_rel_same = family_input.get('prefer_same_rel', False)
    pref_faith_same = family_input.get('prefer_same_faith', False)

    donor_eth_raw = str(row.get('ethnicity','')).strip().lower()
    donor_rel = str(row.get('religion','')).strip().lower()
    donor_faith = str(row.get('faith','')).strip().lower()

    donor_ethnicities = [e.strip() for e in donor_eth_raw.split("/") if e.strip()]
    if not pref_eth_same or fam_eth == '':
        eth_d = 0.0
    else:
        if fam_eth in donor_ethnicities:
            eth_d = 0.0   
        else:
            eth_d = 1.0 

    rel_d = 0.0 if not pref_rel_same or fam_religion == '' or donor_rel == fam_religion else 1.0
    faith_d = 0.0 if not pref_faith_same or fam_faith == '' or donor_faith == fam_faith else 1.0
    components.extend([eth_d, rel_d, faith_d])

    arr = np.array(components, dtype=float)
    if arr.size == 0: return 0.0
    euc = math.sqrt(np.sum(arr ** 2))
    max_possible = math.sqrt(arr.size)
    return float(euc / max_possible) if max_possible > 0 else 0.0

# ---------------------------
# Flask app and web UI
app = Flask(__name__)

# Data for dynamic dropdowns 
SECTS_BY_RELIGION = {
    'اسلام': ['شیعه', 'سنی'],
    'مسیحی': ['کاتولیک', 'پروتستان', 'ارتودکس'],
    'یهودی': ['ارتودکس', 'رفورم', 'محافظه‌کار']
    # 'زرتشتی' has no sects in this model, so it won't be in the keys
}


# FORM_HTML
FORM_HTML = """
<!doctype html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="utf-8">
    <title>سیستم تطبیق رحم جایگزین</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2em; background-color: #f9f9f9; }
        h2, h3 { color: #333; }
        form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: grid; grid-template-columns: 150px 1fr; gap: 10px; align-items: center;}
        label, .label { font-weight: bold; }
        input[type="text"], input[type="number"], select { width: 250px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        .full-width { grid-column: 1 / -1; }
        hr { border: none; border-top: 1px solid #eee; margin: 10px 0; }
        input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; width: auto; justify-self: start; }
        input[type="submit"]:hover { background-color: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
    </style>
</head>
<body>
<h2>سیستم تطبیق رحم جایگزین — فرم خانواده</h2>
<p>لطفاً مشخصات مورد نظر خود را برای اهداکننده وارد کنید. فیلدهای خالی در نظر گرفته نمی‌شوند.</p>
<form method="post">

  <label for="province_id">استان:</label>
  <select name="province" id="province_id">
      <option value="" {% if not form_data.get('province') %}selected{% endif %}>همه استان‌ها</option>
      <option value="آذربایجان شرقی" {% if form_data.get('province') == 'آذربایجان شرقی' %}selected{% endif %}>آذربایجان شرقی</option>
      <option value="آذربایجان غربی" {% if form_data.get('province') == 'آذربایجان غربی' %}selected{% endif %}>آذربایجان غربی</option>
      <option value="اردبیل" {% if form_data.get('province') == 'اردبیل' %}selected{% endif %}>اردبیل</option>
      <option value="اصفهان" {% if form_data.get('province') == 'اصفهان' %}selected{% endif %}>اصفهان</option>
      <option value="البرز" {% if form_data.get('province') == 'البرز' %}selected{% endif %}>البرز</option>
      <option value="ایلام" {% if form_data.get('province') == 'ایلام' %}selected{% endif %}>ایلام</option>
      <option value="بوشهر" {% if form_data.get('province') == 'بوشهر' %}selected{% endif %}>بوشهر</option>
      <option value="تهران" {% if form_data.get('province') == 'تهران' %}selected{% endif %}>تهران</option>
      <option value="چهارمحال و بختیاری" {% if form_data.get('province') == 'چهارمحال و بختیاری' %}selected{% endif %}>چهارمحال و بختیاری</option>
      <option value="خراسان جنوبی" {% if form_data.get('province') == 'خراسان جنوبی' %}selected{% endif %}>خراسان جنوبی</option>
      <option value="خراسان رضوی" {% if form_data.get('province') == 'خراسان رضوی' %}selected{% endif %}>خراسان رضوی</option>
      <option value="خراسان شمالی" {% if form_data.get('province') == 'خراسان شمالی' %}selected{% endif %}>خراسان شمالی</option>
      <option value="خوزستان" {% if form_data.get('province') == 'خوزستان' %}selected{% endif %}>خوزستان</option>
      <option value="زنجان" {% if form_data.get('province') == 'زنجان' %}selected{% endif %}>زنجان</option>
      <option value="سمنان" {% if form_data.get('province') == 'سمنان' %}selected{% endif %}>سمنان</option>
      <option value="سیستان و بلوچستان" {% if form_data.get('province') == 'سیستان و بلوچستان' %}selected{% endif %}>سیستان و بلوچستان</option>
      <option value="فارس" {% if form_data.get('province') == 'فارس' %}selected{% endif %}>فارس</option>
      <option value="قزوین" {% if form_data.get('province') == 'قزوین' %}selected{% endif %}>قزوین</option>
      <option value="قم" {% if form_data.get('province') == 'قم' %}selected{% endif %}>قم</option>
      <option value="کردستان" {% if form_data.get('province') == 'کردستان' %}selected{% endif %}>کردستان</option>
      <option value="کرمان" {% if form_data.get('province') == 'کرمان' %}selected{% endif %}>کرمان</option>
      <option value="کرمانشاه" {% if form_data.get('province') == 'کرمانشاه' %}selected{% endif %}>کرمانشاه</option>
      <option value="کهگیلویه و بویراحمد" {% if form_data.get('province') == 'کهگیلویه و بویراحمد' %}selected{% endif %}>کهگیلویه و بویراحمد</option>
      <option value="گلستان" {% if form_data.get('province') == 'گلستان' %}selected{% endif %}>گلستان</option>
      <option value="گیلان" {% if form_data.get('province') == 'گیلان' %}selected{% endif %}>گیلان</option>
      <option value="لرستان" {% if form_data.get('province') == 'لرستان' %}selected{% endif %}>لرستان</option>
      <option value="مازندران" {% if form_data.get('province') == 'مازندران' %}selected{% endif %}>مازندران</option>
      <option value="مرکزی" {% if form_data.get('province') == 'مرکزی' %}selected{% endif %}>مرکزی</option>
      <option value="هرمزگان" {% if form_data.get('province') == 'هرمزگان' %}selected{% endif %}>هرمزگان</option>
      <option value="همدان" {% if form_data.get('province') == 'همدان' %}selected{% endif %}>همدان</option>
      <option value="یزد" {% if form_data.get('province') == 'یزد' %}selected{% endif %}>یزد</option>
  </select>

  <label for="city_id">شهر:</label>
  <input name="city" id="city_id" type="text" value="{{ form_data.get('city', '') }}">

  <label for="budget_id">بودجه:</label>
  <input name="budget" id="budget_id" type="number" step="1000000" value="{{ form_data.get('budget', '') }}">

  <label for="education_id">سطح تحصیلات:</label>
  <select name="education" id="education_id">
    <option value="" {% if not form_data.get('education') %}selected{% endif %}>مهم نیست</option>
    <option value="1" {% if form_data.get('education') == '1' %}selected{% endif %}>دیپلم</option>
    <option value="2" {% if form_data.get('education') == '2' %}selected{% endif %}>فوق دیپلم</option>
    <option value="3" {% if form_data.get('education') == '3' %}selected{% endif %}>لیسانس</option>
    <option value="4" {% if form_data.get('education') == '4' %}selected{% endif %}>ارشد</option>
    <option value="5" {% if form_data.get('education') == '5' %}selected{% endif %}>دکترا</option>
  </select>

  <label for="marriage_id">وضعیت تاهل:</label>
  <select name="marriage" id="marriage_id">
    <option value="" {% if not form_data.get('marriage') %}selected{% endif %}>مهم نیست</option>
    <option value="1" {% if form_data.get('marriage') == '1' %}selected{% endif %}>مطلقه</option>
    <option value="2" {% if form_data.get('marriage') == '2' %}selected{% endif %}>متاهل</option>
  </select>

  <hr class="full-width">

  <label for="ethnicity_id">قومیت شما:</label>
  <select name="ethnicity" id="ethnicity_id">
    <option value="" {% if not form_data.get('ethnicity') %}selected{% endif %}>انتخاب کنید</option>
    <option value="فارس" {% if form_data.get('ethnicity') == 'فارس' %}selected{% endif %}>فارس</option>
    <option value="ترک" {% if form_data.get('ethnicity') == 'ترک' %}selected{% endif %}>ترک</option>
    <option value="کرد" {% if form_data.get('ethnicity') == 'کرد' %}selected{% endif %}>کرد</option>
    <option value="لر" {% if form_data.get('ethnicity') == 'لر' %}selected{% endif %}>لر</option>
    <option value="عرب" {% if form_data.get('ethnicity') == 'عرب' %}selected{% endif %}>عرب</option>
    <option value="بلوچ" {% if form_data.get('ethnicity') == 'بلوچ' %}selected{% endif %}>بلوچ</option>
    <option value="ترکمن" {% if form_data.get('ethnicity') == 'ترکمن' %}selected{% endif %}>ترکمن</option>
    <option value="گیلک" {% if form_data.get('ethnicity') == 'گیلک' %}selected{% endif %}>گیلک</option>
    <option value="آشوری" {% if form_data.get('ethnicity') == 'آشوری' %}selected{% endif %}>آشوری</option>
    <option value="ارمنی" {% if form_data.get('ethnicity') == 'ارمنی' %}selected{% endif %}>ارمنی</option>
    <option value="کلیمی" {% if form_data.get('ethnicity') == 'کلیمی' %}selected{% endif %}>کلیمی</option>
  </select>
  
  <div class="full-width">
    <input type="checkbox" name="prefer_same_eth" id="eth_check" value="on" {% if form_data.get('prefer_same_eth') %}checked{% endif %}>
    <label for="eth_check">ترجیح می‌دهم قومیت یکسان باشد</label>
  </div>

  <label for="faith_id">دین شما:</label>
  <select name="faith" id="faith_id">
       <option value="" {% if not form_data.get('faith') %}selected{% endif %}>انتخاب کنید</option>
       <option value="اسلام" {% if form_data.get('faith') == 'اسلام' %}selected{% endif %}>اسلام</option>
       <option value="مسیحی" {% if form_data.get('faith') == 'مسیحی' %}selected{% endif %}>مسیحی</option>
       <option value="یهودی" {% if form_data.get('faith') == 'یهودی' %}selected{% endif %}>یهودی</option>
       <option value="زرتشتی" {% if form_data.get('faith') == 'زرتشتی' %}selected{% endif %}>زرتشتی</option>
  </select>
  
  <div class="full-width">
    <input type="checkbox" name="prefer_same_faith" id="faith_check" value="on" {% if form_data.get('prefer_same_faith') %}checked{% endif %}>
    <label for="faith_check">ترجیح می‌دهم دین یکسان باشد</label>
  </div>

  <label for="religion_id">مذهب شما:</label>
  <select name="religion" id="religion_id">
    <option value="">ابتدا دین را انتخاب کنید</option>
  </select>
 
  <div class="full-width">
    <input type="checkbox" name="prefer_same_rel" id="rel_check" value="on" {% if form_data.get('prefer_same_rel') %}checked{% endif %}>
    <label for="rel_check">ترجیح می‌دهم مذهب یکسان باشد</label>
  </div>

  <hr class="full-width">
 
  <label for="topk_id">تعداد نتایج برتر:</label>
  <input name="topk" id="topk_id" value="{{ form_data.get('topk', 5) }}" type="number">
 
  <div class="full-width">
    <input type="submit" value="پیدا کردن بهترین گزینه‌ها">
  </div>
</form>

{% if results %}
<h3>نتایج برتر</h3>
<table>
<tr>
  <th>رتبه</th><th>کد اهداکننده</th><th>امتیاز نهایی</th><th>درصد استاتیک</th><th>فاصله استاتیک</th><th>فاصله دینامیک</th>
  <th>سن</th><th>BMI</th><th>شهر</th><th>مبلغ</th><th>تحصیلات</th><th>تاهل</th><th>قومیت</th><th>مذهب</th><th>دین</th>
</tr>
{% for r in results %}
<tr>
  <td>{{ loop.index }}</td>
  <td>{{ r['donor_id'] }}</td>
  <td>{{ "%.4f"|format(r['final_score']) }}</td>
  <td>{{ "%.2f"|format(r['static_percent']) }}</td>
  <td>{{ "%.4f"|format(r['static_distance']) }}</td>
  <td>{{ "%.4f"|format(r['dynamic_distance']) }}</td>
  <td>{{ r.get('age','') }}</td>
  <td>{{ r.get('BMI','') }}</td>
  <td>{{ r.get('city','') }}</td>
  <td>{{ r.get('Payment','') }}</td>
  <td>{{ r.get('Education','') }}</td>
  <td>{{ r.get('Marriage','') }}</td>
  <td>{{ r.get('ethnicity','') }}</td>
  <td>{{ r.get('religion','') }}</td>
  <td>{{ r.get('faith','') }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const sectsData = {{ sects_data_json | safe }};
        const religionSelect = document.getElementById('faith_id');
        const sectSelect = document.getElementById('religion_id');
        const previouslySelectedSect = "{{ form_data.get('religion', '') }}";

        function updateSectDropdown() {
            const selectedReligion = religionSelect.value;
            const sects = sectsData[selectedReligion] || [];
            
            // Clear previous options
            sectSelect.innerHTML = '';

            // Add a default "Doesn't Matter" option first
            const defaultOption = document.createElement('option');
            defaultOption.value = "";
            defaultOption.textContent = "مهم نیست";
            sectSelect.appendChild(defaultOption);

            if (sects.length > 0) {
                sectSelect.disabled = false;
                sects.forEach(function(sect) {
                    const option = document.createElement('option');
                    option.value = sect;
                    option.textContent = sect;
                    sectSelect.appendChild(option);
                });
            } else {
                // If no sects, disable the dropdown
                sectSelect.disabled = true;
                defaultOption.textContent = "مذهب تعریف نشده";
            }
            
            // Re-select the previous value if it exists in the new list
            if (previouslySelectedSect) {
                sectSelect.value = previouslySelectedSect;
            }
        }

        // Add event listener for changes
        religionSelect.addEventListener('change', updateSectDropdown);

        // Initial call to set the dropdown on page load
        updateSectDropdown();
    });
</script>

</body>
</html>
"""


@app.route("/", methods=["GET","POST"])
def index():
    results = None
    form_data = {} 
    if request.method == "POST":
        form_data = request.form 
        family_input = {
            'city': request.form.get('city','').strip(),
            'province': request.form.get('province','').strip(),
            'budget': request.form.get('budget','').strip(),
            'education': request.form.get('education','').strip(),
            'marriage': request.form.get('marriage','').strip(),
            'ethnicity': request.form.get('ethnicity','').strip(),
            'prefer_same_eth': bool(request.form.get('prefer_same_eth')),
            'religion': request.form.get('religion','').strip(),
            'prefer_same_rel': bool(request.form.get('prefer_same_rel')),
            'faith': request.form.get('faith','').strip(),
            'prefer_same_faith': bool(request.form.get('prefer_same_faith'))
        }
        try:
            topk = int(request.form.get('topk', 5))
        except ValueError:
            topk = 5

        try:
            donors_df = load_donors()
            all_scores = []
            for idx, row in donors_df.iterrows():
                static_percent, static_breakdown = compute_static_percent(row)
                dyn_dist = compute_dynamic_distance(row, family_input)

                # Lower score is better. Static percent is inverted to a distance.
                static_dist = (100.0 - static_percent) / 100.0

                final_score = STATIC_WEIGHT * static_dist + DYNAMIC_WEIGHT * dyn_dist
                
                bmi_val = row.get('BMI')
                res = {
                    'donor_id': row.get('donor_id'),
                    'final_score': final_score,
                    'static_percent': static_percent,
                    'static_distance': static_dist,
                    'dynamic_distance': dyn_dist,
                    'age': row.get('Age'),
                    'BMI': f"{bmi_val:.1f}" if pd.notna(bmi_val) else '',
                    'city': row.get('city'),
                    'province': row.get('province'),
                    'Payment': row.get('Payment'),
                    'Education': row.get('Education'),
                    'Marriage': row.get('Marriage'),
                    'ethnicity': row.get('ethnicity'),
                    'religion': row.get('religion'),
                    'faith': row.get('faith')
                }
                all_scores.append(res)
            
            all_scores.sort(key=lambda x: x['final_score'])
            results = all_scores[:topk]

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            # You can pass an error message to the template here if you want
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Pass the sects data as a JSON string for the JavaScript
    return render_template_string(FORM_HTML,
                                  results=results,
                                  form_data=form_data,
                                  sects_data_json=json.dumps(SECTS_BY_RELIGION))

if __name__ == "__main__":
    app.run(debug=True)
