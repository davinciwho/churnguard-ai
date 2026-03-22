ChurnGuard AI — Customer Retention & Revenue Risk Analytics

ChurnGuard AI is an interactive analytics dashboard that predicts customer churn, quantifies revenue at risk, and evaluates retention strategies using ROI-based decision logic.

The application combines machine learning with business metrics to simulate a real-world customer analytics workflow.

Problem Statement

Customer churn directly impacts revenue and long-term growth. Businesses need to:

Identify high-risk customers
Estimate potential revenue loss
Evaluate whether retention is cost-effective
Prioritize customers based on financial impact

This project builds an end-to-end churn analytics platform to support those decisions.

Solution

The application:

Predicts churn probability using XGBoost
Calculates annual revenue at risk
Compares retention cost vs acquisition cost
Identifies top at-risk customers
Visualizes churn drivers and business insights
Key Features

Churn Prediction

Machine learning-based churn probability scoring
Customer-level risk classification

Revenue Risk Modeling

Annual revenue loss estimation
Probability-weighted financial impact

Retention Strategy Simulator

Adjustable discount strategy
Acquisition vs retention cost comparison
ROI-based decision guidance

Business Dashboard

KPI metrics
Churn rate tracking
Feature importance visualization
Risk segmentation

At-Risk Customer Prioritization

Top customers ranked by revenue risk
Decision-ready analytics table
Model Details

Model: XGBoost Classifier

Features used:

Customer tenure
Monthly charges
Support calls
Contract type

Evaluation metrics:

Precision
ROC-AUC
Churn rate analysis
Example Insights
Month-to-month contracts show the highest churn risk
High support call frequency strongly predicts churn
Short-tenure customers contribute disproportionate revenue risk
Retention discounts are often cheaper than reacquisition
Tech Stack

Python
Pandas / NumPy
XGBoost
Scikit-learn
Streamlit
Plotly
Faker (synthetic data generation)
