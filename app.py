import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# 1. DATA LAYER
# ─────────────────────────────────────────────

@st.cache_data
def generate_business_data(n=2500):
    fake = Faker()
    np.random.seed(42)
    data = []

    for _ in range(n):
        tenure      = np.random.randint(1, 72)
        monthly     = np.random.uniform(20, 150)
        calls       = np.random.poisson(1.2)
        contract    = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            p=[0.5, 0.2, 0.3]
        )

        risk = 0.05
        if contract == 'Month-to-month': risk += 0.35
        if calls > 3:                    risk += 0.40
        if tenure < 12:                  risk += 0.10

        churn = 1 if np.random.random() < min(risk, 0.95) else 0

        data.append({
            'CustomerID':     fake.uuid4()[:8],
            'Tenure':         tenure,
            'MonthlyCharges': round(monthly, 2),
            'SupportCalls':   calls,
            'Contract':       contract,
            'Churn':          churn
        })

    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# 2. MODEL LAYER
# ─────────────────────────────────────────────

CONTRACT_MAP  = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
FEATURE_COLS  = ['Tenure', 'MonthlyCharges', 'SupportCalls', 'Contract_Enc']
FEATURE_NAMES = ['Tenure', 'Monthly Charges', 'Support Calls', 'Contract']


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric contract encoding without mutating the original df."""
    df = df.copy()
    df['Contract_Enc'] = df['Contract'].map(CONTRACT_MAP)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    df_ml = encode(df)

    X = df_ml[FEATURE_COLS]
    y = df_ml['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, pred)
    auc       = roc_auc_score(y_test, prob)

    return model, precision, auc


# ─────────────────────────────────────────────
# 3. STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnGuard AI",
    layout="wide",
    page_icon="🛡️"
)

st.title("🛡️ ChurnGuard AI: Retention & Revenue Platform")

# ── Load data and model ───────────────────────
df = generate_business_data()
model, precision, auc = train_model(df)

# Score the full dataset for BI dashboarding
# (separate from model evaluation — no leakage concern here)
df_encoded          = encode(df)
df['Churn_Prob']    = model.predict_proba(df_encoded[FEATURE_COLS])[:, 1]
df['Revenue_Risk']  = df['MonthlyCharges'] * 12 * df['Churn_Prob']
churn_rate          = df['Churn'].mean()


# ── KPI Row ───────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Predicted Revenue at Risk", f"${df['Revenue_Risk'].sum():,.0f}")
c2.metric("Overall Churn Rate",        f"{churn_rate:.1%}")
c3.metric("Avg Churn Probability",     f"{df['Churn_Prob'].mean():.1%}")
c4.metric("Model Precision",           f"{precision:.1%}")
c5.metric("ROC-AUC Score",             f"{auc:.2f}")


# ── Sidebar: Retention Strategy ───────────────
st.sidebar.header("💼 Retention Strategy")
discount_pct = st.sidebar.slider("Discount % to Offer", 5, 50, 15)
acq_cost     = st.sidebar.number_input(
    "New Customer Acquisition Cost ($)", 100, 1000, 450
)


# ── Customer Risk Assessment ──────────────────
st.subheader("🔍 Customer Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    tenure   = st.slider("Tenure (Months)", 1, 72, 24)
    contract = st.selectbox("Contract Type", list(CONTRACT_MAP.keys()))

with col2:
    calls   = st.number_input("Support Calls", 0, 20, 1)
    monthly = st.number_input("Monthly Charges ($)", 20.0, 200.0, 75.0)

# Predict for the single input customer
# Variable named 'input_features' to avoid collision with FEATURE_NAMES list
input_features = np.array([[tenure, monthly, calls, CONTRACT_MAP[contract]]])
churn_risk     = model.predict_proba(input_features)[0][1]

with col3:
    st.write("### Risk Level")
    if churn_risk > 0.7:
        st.error(f"🔴 CRITICAL RISK: {churn_risk:.1%}")
    elif churn_risk > 0.4:
        st.warning(f"🟡 MODERATE RISK: {churn_risk:.1%}")
    else:
        st.success(f"🟢 LOW RISK: {churn_risk:.1%}")


# ── Financial Impact (wired to sidebar controls) ──
st.subheader("💰 Financial Impact & ROI")

revenue_risk   = monthly * 12 * churn_risk
retention_cost = monthly * (discount_pct / 100) * 12
roi_vs_acq     = acq_cost - retention_cost

fi1, fi2, fi3 = st.columns(3)
fi1.metric("Est. Annual Revenue at Risk", f"${revenue_risk:,.2f}")
fi2.metric(f"Retention Offer Cost ({discount_pct}% disc.)", f"${retention_cost:,.2f}")
fi3.metric(
    "Saved vs. Re-acquiring",
    f"${roi_vs_acq:,.2f}",
    delta="Cheaper to retain" if roi_vs_acq > 0 else "Consider re-acquiring"
)

if churn_risk > 0.5:
    st.info(
        f"💡 Offering a **{discount_pct}% discount** costs **${retention_cost:,.0f}/yr** — "
        f"**${roi_vs_acq:,.0f} cheaper** than replacing this customer "
        f"(acquisition cost: ${acq_cost:,})."
    )


# ── Top At-Risk Customers Table ───────────────
st.subheader("🚨 Top 10 Customers by Revenue at Risk")

top_at_risk = (
    df.sort_values('Revenue_Risk', ascending=False)
      .head(10)
    [['CustomerID', 'Contract', 'Tenure', 'SupportCalls',
      'MonthlyCharges', 'Churn_Prob', 'Revenue_Risk']]
      .rename(columns={
          'CustomerID':     'Customer ID',
          'SupportCalls':   'Support Calls',
          'MonthlyCharges': 'Monthly Charges ($)',
          'Churn_Prob':     'Churn Probability',
          'Revenue_Risk':   'Revenue at Risk ($)'
      })
      .reset_index(drop=True)
)

top_at_risk['Churn Probability']    = top_at_risk['Churn Probability'].map('{:.1%}'.format)
top_at_risk['Revenue at Risk ($)']  = top_at_risk['Revenue at Risk ($)'].map('${:,.2f}'.format)
top_at_risk['Monthly Charges ($)']  = top_at_risk['Monthly Charges ($)'].map('${:,.2f}'.format)

st.dataframe(top_at_risk, use_container_width=True)


# ── Insight Charts ────────────────────────────
st.subheader("📊 Insights")

chart1, chart2 = st.columns(2)

with chart1:
    fig1 = px.histogram(
        df, x="Churn_Prob", nbins=30,
        title="Churn Probability Distribution",
        color_discrete_sequence=["#EF553B"]
    )
    fig1.update_layout(xaxis_title="Churn Probability", yaxis_title="Customer Count")
    st.plotly_chart(fig1, use_container_width=True)

with chart2:
    # Key business insight: churn rate breakdown by contract type
    churn_by_contract = (
        df.groupby('Contract')['Churn']
          .mean()
          .reset_index()
          .rename(columns={'Churn': 'Churn Rate'})
          .sort_values('Churn Rate', ascending=False)
    )
    fig2 = px.bar(
        churn_by_contract, x='Contract', y='Churn Rate',
        title="Churn Rate by Contract Type",
        color='Churn Rate',
        color_continuous_scale='Reds',
        text=churn_by_contract['Churn Rate'].map('{:.1%}'.format)
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(yaxis_tickformat='.0%', coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

chart3, chart4 = st.columns(2)

with chart3:
    # Feature importance — uses FEATURE_NAMES list (different var from input_features)
    importances = model.feature_importances_
    fig3 = px.bar(
        x=FEATURE_NAMES, y=importances,
        title="Feature Importance (XGBoost)",
        color=importances,
        color_continuous_scale='Blues',
        labels={'x': 'Feature', 'y': 'Importance'}
    )
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with chart4:
    # Revenue at risk pie by contract
    risk_by_contract = (
        df.groupby('Contract')['Revenue_Risk']
          .sum()
          .reset_index()
          .rename(columns={'Revenue_Risk': 'Total Revenue at Risk ($)'})
    )
    fig4 = px.pie(
        risk_by_contract,
        names='Contract',
        values='Total Revenue at Risk ($)',
        title="Revenue at Risk by Contract Type",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig4, use_container_width=True)
