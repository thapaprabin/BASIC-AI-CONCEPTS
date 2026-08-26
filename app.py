import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2rem 2.2rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        color: white;
    }
    .header-box h1 { font-size: 1.7rem; margin: 0 0 .4rem; font-weight: 700; color: white; }
    .header-box p  { font-size: .92rem; margin: 0; opacity: .75; color: #cdd3e0; }

    .section-label {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: #6b7a99;
        margin: 1.4rem 0 .5rem;
    }

    .result-churn {
        background: #fff1f1;
        border: 1.5px solid #f5a0a0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .result-stay {
        background: #f0faf4;
        border: 1.5px solid #7fcca0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .result-label  { font-size: 1.05rem; font-weight: 600; margin-bottom: .3rem; }
    .result-prob   { font-size: 2.6rem; font-weight: 800; line-height: 1; }
    .result-sub    { font-size: .82rem; color: #666; margin-top: .4rem; }

    .tip-box {
        background: #fff8e1;
        border-left: 4px solid #f0b429;
        border-radius: 0 8px 8px 0;
        padding: .8rem 1rem;
        font-size: .84rem;
        margin-top: 1rem;
        color: #5c4a00;
    }
    .metric-row { display: flex; gap: 12px; margin-top: .8rem; flex-wrap: wrap; }
    .metric-pill {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: .3rem .9rem;
        font-size: .8rem;
        color: #334155;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white;
        border: none;
        border-radius: 8px;
        padding: .6rem 2.2rem;
        font-size: .95rem;
        font-weight: 600;
        width: 100%;
        margin-top: .6rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .88; color: white; }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label { font-size: .87rem; font-weight: 500; color: #334155; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv('User churn.csv')

    # Same feature engineering as your notebook
    def group(tenure):
        if tenure <= 12:   return '0-12 months'
        elif tenure <= 24: return '12-24 months'
        elif tenure <= 48: return '24-48 months'
        else:              return 'More than 48 months'

    df['Tenure Cluster'] = df['tenure'].apply(group)

    X = df.drop(['Churn', 'customerID'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(max_depth=6, random_state=42)
    rf.fit(X_train_smote, y_train_smote)

    return rf, X.columns.tolist()
model, trained_columns= load_model()
threshold=0.5

FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'Tenure Cluster_More than 48 months',
    'Tenure Cluster_12-24 months',
    'Tenure Cluster_24-48 months',
    'Tenure Cluster_More than 48 months',
]

def tenure_group(t):
    if t < 13:   return '0-12 months'
    elif t < 25: return '12-24 months'
    elif t < 49: return '24-48 months'
    else:        return 'More than 48 months'

def build_input(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Numeric
    row['SeniorCitizen']   = inputs['SeniorCitizen']
    row['tenure']          = inputs['tenure']
    row['MonthlyCharges']  = inputs['MonthlyCharges']
    row['TotalCharges']    = inputs['TotalCharges']

    # Binary yes/no features
    if inputs['gender']           == 'Male':  row['gender_Male']           = 1
    if inputs['Partner']          == 'Yes':   row['Partner_Yes']            = 1
    if inputs['Dependents']       == 'Yes':   row['Dependents_Yes']         = 1
    if inputs['PhoneService']     == 'Yes':   row['PhoneService_Yes']       = 1
    if inputs['PaperlessBilling'] == 'Yes':   row['PaperlessBilling_Yes']   = 1

    # MultipleLines
    if inputs['MultipleLines'] == 'No phone service':
        row['MultipleLines_No phone service'] = 1
    elif inputs['MultipleLines'] == 'Yes':
        row['MultipleLines_Yes'] = 1

    # InternetService
    if inputs['InternetService'] == 'Fiber optic':
        row['InternetService_Fiber optic'] = 1
    elif inputs['InternetService'] == 'No':
        row['InternetService_No'] = 1

    # Internet-dependent features
    for feat in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        val = inputs[feat]
        col_no_svc = f'{feat}_No internet service'
        col_yes    = f'{feat}_Yes'
        if val == 'No internet service' and col_no_svc in row:
            row[col_no_svc] = 1
        elif val == 'Yes' and col_yes in row:
            row[col_yes] = 1

    # Contract
    if inputs['Contract'] == 'One year':    row['Contract_One year']  = 1
    elif inputs['Contract'] == 'Two year':  row['Contract_Two year']  = 1

    # PaymentMethod
    pm_map = {
        'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)',
        'Electronic check':        'PaymentMethod_Electronic check',
        'Mailed check':            'PaymentMethod_Mailed check',
    }
    if inputs['PaymentMethod'] in pm_map:
        row[pm_map[inputs['PaymentMethod']]] = 1

    # Tenure cluster
    cluster = tenure_group(inputs['tenure'])
    if cluster == '12-24 months':
        row['Tenure Cluster_12-24 months'] = 1
    elif cluster == '24-48 months':
        row['Tenure Cluster_24-48 months'] = 1
    elif cluster == 'More than 48 months':
        row['Tenure Cluster_More than 48 months'] = 1

    return pd.DataFrame([row])[trained_columns]


st.markdown("""
<div class="header-box">
  <h1>📡 Customer Churn Predictor</h1>
  <p>Enter customer details below to predict the likelihood of churn.<br>
     Model: Random Forest Classifier &nbsp;|&nbsp; Trained on SMOTE-balanced data</p>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    gender         = c1.selectbox("Gender",         ["Female", "Male"])
    senior         = c2.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = c3.selectbox("Partner",         ["No", "Yes"])
    dependents     = c1.selectbox("Dependents",      ["No", "Yes"])

    st.markdown('<div class="section-label">Account Information</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    tenure         = c4.slider("Tenure (months)", 0, 72, 12)
    monthly        = c5.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total          = c6.number_input("Total Charges ($)",   0.0, 10000.0, float(tenure * 65), step=1.0)
    contract       = c4.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
    billing        = c5.selectbox("Paperless Billing",["No", "Yes"])
    payment        = c6.selectbox("Payment Method",  [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])

    st.markdown('<div class="section-label">Phone Services</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    phone          = c7.selectbox("Phone Service",   ["No", "Yes"])
    multi_lines    = c8.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])

    st.markdown('<div class="section-label">Internet Services</div>', unsafe_allow_html=True)
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    internet_opts  = ["No", "Yes"] if internet != "No" else ["No internet service"]
    c9,  c10, c11 = st.columns(3)
    c12, c13, c14 = st.columns(3)
    online_sec     = c9.selectbox("Online Security",    internet_opts)
    online_bkp     = c10.selectbox("Online Backup",     internet_opts)
    device_prot    = c11.selectbox("Device Protection", internet_opts)
    tech_sup       = c12.selectbox("Tech Support",      internet_opts)
    stream_tv      = c13.selectbox("Streaming TV",      internet_opts)
    stream_mv      = c14.selectbox("Streaming Movies",  internet_opts)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    inputs = {
        'gender':           gender,
        'SeniorCitizen':    1 if senior == "Yes" else 0,
        'Partner':          partner,
        'Dependents':       dependents,
        'tenure':           tenure,
        'MonthlyCharges':   monthly,
        'TotalCharges':     total,
        'PhoneService':     phone,
        'MultipleLines':    multi_lines,
        'InternetService':  internet,
        'OnlineSecurity':   online_sec,
        'OnlineBackup':     online_bkp,
        'DeviceProtection': device_prot,
        'TechSupport':      tech_sup,
        'StreamingTV':      stream_tv,
        'StreamingMovies':  stream_mv,
        'Contract':         contract,
        'PaperlessBilling': billing,
        'PaymentMethod':    payment,
    }

    X_input  = build_input(inputs)
    proba    = model.predict_proba(X_input)[0][1]   
    churns   = proba >= threshold

    st.markdown("---")
    st.markdown("### Prediction Result")

    if churns:
        st.markdown(f"""
        <div class="result-churn">
          <div class="result-label" style="color:#c0392b;">⚠️ High Churn Risk</div>
          <div class="result-prob" style="color:#e74c3c;">{proba:.0%}</div>
          <div class="result-sub">probability this customer will churn</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
          💡 <strong>Recommended action:</strong> Flag this customer for a retention campaign.
          Consider offering a contract upgrade, discount, or proactive support outreach.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-stay">
          <div class="result-label" style="color:#1e7a4a;">✅ Low Churn Risk</div>
          <div class="result-prob" style="color:#27ae60;">{proba:.0%}</div>
          <div class="result-sub">probability this customer will churn</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box" style="border-color:#27ae60; background:#f0fff4; color:#145a32;">
          ✅ <strong>Customer appears stable.</strong> Continue standard engagement.
          Monitor monthly if on a month-to-month contract.
        </div>""", unsafe_allow_html=True)

    # Key factors summary
    st.markdown("#### Key factors in this prediction")
    cluster = tenure_group(tenure)
    cols = st.columns(4)
    cols[0].metric("Tenure",    f"{tenure} months", delta=cluster, delta_color="off")
    cols[1].metric("Contract",  contract)
    cols[2].metric("Monthly $", f"${monthly:.0f}")
    cols[3].metric("Internet",  internet)