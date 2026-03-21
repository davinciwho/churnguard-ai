import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score
import plotly.express as px

# 1. DATA LAYER

@st.cache_data
def generate_business_data(n=2500):
    fake = Faker()
    np.random.seed(42)
    data = []

    for _ in range(n):
        tenure = np.random.randint(1, 72)
        monthly = np.random.uniform(20, 150)
        calls = np.random.poisson(1.2)
        contract = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            p=[0.5, 0.2, 0.3]
        )

        risk = 0.05
        if contract == 'Month-to-month': risk += 0.35
        if calls > 3: risk += 0.40
        if tenure < 12: risk += 0.10

        churn = 1 if np.random.random() < min(risk, 0.95) else 0

        data.append({
            'CustomerID': fake.uuid4()[:8],
            'Tenure': tenure,
            'MonthlyCharges': round(monthly, 2),
            'SupportCalls': calls,
            'Contract': contract,
            'Churn': churn
        })

    return pd.DataFrame(data)


# 2. MODEL LAYER

def train_model(df):
    df_ml = df.copy()

    # Manual encoding 
    mapping = {'Month-to-month':0, 'One year':1, 'Two year':2}
    df_ml['Contract_Enc'] = df_ml['Contract'].map(mapping)

    X = df_ml[['Tenure', 'MonthlyCharges', 'SupportCalls', 'Contract_Enc']]
    y = df_ml['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Evaluation
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    precision = precision_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    return model, precision, auc, X


# 3. STREAMLIT UI

st.set_page_config(page_title="ChurnGuard AI", layout="wide", page_icon="🛡️")

st.title(" 🛡️ ChurnGuard AI: Retention & Revenue Platform")

df = generate_business_data()
model, precision, auc, X = train_model(df)

# Revenue Risk Modeling
df['Churn_Prob'] = model.predict_proba(X)[:,1]
df['Revenue_Risk'] = df['MonthlyCharges'] * 12 * df['Churn_Prob']


# KPI SECTION

c1, c2, c3, c4 = st.columns(4)

c1.metric("Predicted Revenue at Risk", f"${df['Revenue_Risk'].sum():,.0f}")
c2.metric("Avg Churn Probability", f"{df['Churn_Prob'].mean():.1%}")
c3.metric("Model Precision", f"{precision:.1%}")
c4.metric("ROC-AUC Score", f"{auc:.2f}")


# SIDEBAR

st.sidebar.header("Retention Strategy")
discount_pct = st.sidebar.slider("Discount %", 5, 50, 15)
acq_cost = st.sidebar.number_input("Acquisition Cost", 100, 1000, 450)


# USER INPUT

st.subheader("🔍 Customer Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (Months)", 1, 72, 24)
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

with col2:
    calls = st.number_input("Support Calls", 0, 20, 1)
    monthly = st.number_input("Monthly Charges", 20.0, 200.0, 75.0)

# Prediction
mapping = {'Month-to-month':0, 'One year':1, 'Two year':2}
contract_enc = mapping[contract]

features = np.array([[tenure, monthly, calls, contract_enc]])
risk = model.predict_proba(features)[0][1]


# OUTPUT

with col3:
    st.write("### Risk Level")

    if risk > 0.7:
        st.error(f"CRITICAL RISK: {risk:.1%}")
    elif risk > 0.4:
        st.warning(f"MODERATE RISK: {risk:.1%}")
    else:
        st.success(f"LOW RISK: {risk:.1%}")

# Financial logic
st.subheader("💰 Financial Impact")

revenue_risk = monthly * 12 * risk
retention_cost = monthly * (discount_pct/100) * 12

st.write(f"Estimated Revenue at Risk: **${revenue_risk:,.2f}**")

if risk > 0.5:
    st.info(f"Retention Cost: ${retention_cost:,.0f} vs Acquisition Cost: ${acq_cost}")


# VISUALS

st.subheader("📊 Insights")

fig1 = px.histogram(df, x="Churn_Prob", nbins=30,
                    title="Churn Probability Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Feature Importance
importance = model.feature_importances_
features = ['Tenure','MonthlyCharges','SupportCalls','Contract']

fig2 = px.bar(x=features, y=importance,
              title="Feature Importance")
st.plotly_chart(fig2, use_container_width=True)
