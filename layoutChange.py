import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import shap
from optbinning import OptimalBinning
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(123)

# Streamlit App Title
st.title("Final Credit Risk Management App")
st.markdown("""
This app predicts credit default risk using an XGBoost model, addressing class imbalance, threshold optimization, and age dependency.
It includes Expected Loss (EL) calculations, advanced visualizations, and detailed analysis.
""")

# 1. Load and Preprocess the "Give Me Some Credit" Dataset
@st.cache_data
def load_data():
    # Load from local file (place cs-training.csv in the same directory as this script)
    file_path = "cs-training.csv"  # Ensure this file is in your project directory
    df = pd.read_csv(file_path)
    df = df.rename(columns={'SeriousDlqin2yrs': 'default'})
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop index column if present

    # Handle missing values early
    num_imputer = SimpleImputer(strategy='median')
    df[df.columns] = num_imputer.fit_transform(df)

    # Balance the dataset using random undersampling
    df_majority = df[df['default'] == 0]
    df_minority = df[df['default'] == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=123)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced

df = load_data()
st.write("Dataset Loaded:", df.head())

# Feature Engineering: Bin Age into Groups and Add Interaction Term
bins = [20, 30, 40, 50, 60, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60+']
df['AGE_GROUP'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
df['AGE_PAY_0_INTERACTION'] = df['age'] * df['NumberOfTime30-59DaysPastDueNotWorse']

# Define feature types
numerical_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
                  'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                  'NumberRealEstateLoansOrLines', 'NumberOfDependents',
                  'AGE_PAY_0_INTERACTION']
categorical_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
                   'NumberOfTimes90DaysLate', 'AGE_GROUP']

# Handle missing values for categorical
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 2. WoE and IV Calculation
def calculate_woe_iv(df, feature, target):
    optb = OptimalBinning(name=feature, dtype="numerical" if df[feature].dtype in ['int64', 'float64'] else "categorical")
    try:
        optb.fit(df[feature].values, df[target].values)
        binning_table = optb.binning_table.build()
        iv = binning_table['IV'].sum()
        woe = binning_table[['Bin', 'WoE']].set_index('Bin').to_dict()['WoE']
        return woe, iv, optb
    except:
        return {}, 0, None

woe_dict = {}
iv_dict = {}
optb_dict = {}
features = [col for col in df_encoded.columns if col != 'default']
for feature in features:
    woe, iv, optb = calculate_woe_iv(df_encoded, feature, 'default')
    woe_dict[feature] = woe
    iv_dict[feature] = iv
    optb_dict[feature] = optb

# Lower IV threshold
iv_threshold = 0.01
selected_features = [f for f, iv in iv_dict.items() if iv > iv_threshold]
if not selected_features:
    st.warning("No features passed IV threshold. Using all features.")
    selected_features = features

# Transform features to WoE values
df_woe = df_encoded.copy()
for feature in selected_features:
    if optb_dict[feature] is not None:
        df_woe[feature] = optb_dict[feature].transform(df_woe[feature], metric="woe")
    else:
        df_woe = df_woe.drop(columns=[feature])
        selected_features.remove(feature)

# 3. Train-Test Split
X = df_woe[selected_features]
y = df_woe['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# 4. Model Training with XGBoost
# Since dataset is balanced, scale_pos_weight is set to 1
model = xgb.XGBClassifier(scale_pos_weight=1, random_state=123, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Linear Regression (Variant 1: WoE features)
lin_reg1 = LinearRegression()
lin_reg1.fit(X_train, y_train)
y_pred_lin1 = lin_reg1.predict(X_test)
y_pred_lin1 = np.clip(y_pred_lin1, 0, 1)

# Linear Regression (Variant 2: Raw features)
X_no_woe = df_encoded[selected_features]
X_train_no_woe, X_test_no_woe = train_test_split(X_no_woe, test_size=0.3, random_state=123)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_train_no_woe, y_train)
y_pred_lin2 = lin_reg2.predict(X_test_no_woe)
y_pred_lin2 = np.clip(y_pred_lin2, 0, 1)

# 5. Prediction Interface with Threshold and EL
st.header("Predict Default Risk")
st.markdown("Enter client details to predict the probability of default and calculate Expected Loss (EL). Adjust the prediction threshold.")

# Create input fields and threshold slider
input_data = {}
with st.form("prediction_form"):
    input_data['RevolvingUtilizationOfUnsecuredLines'] = st.slider("Revolving Utilization (%)", 0.0, 1.0, 0.5, 0.01)
    input_data['age'] = st.slider("Age", 20, 80, 30)
    input_data['NumberOfTime30-59DaysPastDueNotWorse'] = st.selectbox("Past Due 30-59 Days", [0, 1, 2, 3, 4, 5])
    input_data['MonthlyIncome'] = st.slider("Monthly Income", 0, 100000, 5000, 100)
    input_data['DebtRatio'] = st.slider("Debt Ratio", 0.0, 1.0, 0.3, 0.01)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    submitted = st.form_submit_button("Predict")

# Process input data, predict, and calculate EL
if submitted:
    input_df = pd.DataFrame([input_data])
    input_df['AGE_GROUP'] = pd.cut(input_df['age'], bins=bins, labels=labels, include_lowest=True)
    input_df['AGE_PAY_0_INTERACTION'] = input_df['age'] * input_df['NumberOfTime30-59DaysPastDueNotWorse']
    input_df = pd.get_dummies(input_df)
    for col in selected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[selected_features]
    numerical_input_cols = [col for col in numerical_cols if col in input_df.columns]
    if numerical_input_cols:
        input_df[numerical_input_cols] = scaler.transform(input_df[numerical_input_cols])
    for feature in selected_features:
        if feature in optb_dict and optb_dict[feature] is not None:
            input_df[feature] = optb_dict[feature].transform(input_df[feature], metric="woe")
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob > threshold else 0
    st.write(f"**Predicted Default Probability (PD):** {prob:.2%}")
    st.write(f"**Predicted Default (Threshold {threshold:.2f}):** {pred}")
    if prob > threshold:
        st.error("High risk of default!")
    else:
        st.success("Low risk of default.")
    # Calculate Expected Loss (EL)
    lgd = 0.8  # Assumption: 80% Loss Given Default
    ead = input_data['MonthlyIncome'] * input_data['RevolvingUtilizationOfUnsecuredLines']  # Proxy for exposure
    el = prob * lgd * ead
    st.write(f"**Expected Loss (EL):** ${el:,.2f}")
    st.markdown(f"**EL Breakdown:** PD = {prob:.2%}, LGD = {lgd:.0%}, EAD = ${ead:,.2f}")

# 6. Enhanced Model Performance and Visualizations
st.header("Model Performance and Insights")

# Model Performance Metrics with Adjustable Threshold
st.subheader("XGBoost Model Performance")
threshold = st.slider("Select Threshold for Performance Metrics", 0.0, 1.0, 0.5, 0.05)
y_pred = (y_pred_prob > threshold).astype(int)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
st.write(f"**Balanced Accuracy:** {balanced_acc:.3f}")
st.write(f"**F1-Score:** {f1:.3f}")

# Correlation for Linear Models
corr_lin1 = np.corrcoef(y_test, y_pred_lin1)[0, 1]
corr_lin2 = np.corrcoef(y_test, y_pred_lin2)[0, 1]
st.write("**Correlation with Actual Default:**")
st.write(f"Linear Regression (WoE Features): {corr_lin1:.3f}")
st.write(f"Linear Regression (Raw Features): {corr_lin2:.3f}")

# Visualizations
st.subheader("Visualizations")

# IV Bar Plot
fig, ax = plt.subplots()
iv_df = pd.DataFrame({'Feature': iv_dict.keys(), 'IV': iv_dict.values()})
iv_df = iv_df.sort_values('IV', ascending=False).head(10)
sns.barplot(x='IV', y='Feature', data=iv_df, ax=ax)
ax.set_title('Top 10 Features by Information Value (IV)')
st.pyplot(fig)

# ROC Curve
fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - XGBoost')
ax.legend(loc='lower right')
st.pyplot(fig)

# Precision-Recall Curve with Optimal Threshold
fig, ax = plt.subplots()
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
ax.plot(recall, precision, label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - XGBoost')
optimal_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-10))
optimal_threshold = thresholds[optimal_idx]
ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', label=f'Optimal Threshold = {optimal_threshold:.2f}')
ax.legend()
st.pyplot(fig)

# SHAP Feature Importance
fig, ax = plt.subplots()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False)
plt.title('SHAP Feature Importance - XGBoost')
st.pyplot(fig)

# WoE Distribution for Top Feature
fig, ax = plt.subplots()
top_feature = iv_df['Feature'].iloc[0]
if optb_dict[top_feature] is not None:
    binning_table = optb_dict[top_feature].binning_table.build()
    sns.barplot(x='Bin', y='WoE', data=binning_table, ax=ax)
    ax.set_title(f'WoE Distribution for {top_feature}')
    ax.tick_params(axis='x', rotation=45)
else:
    ax.text(0.5, 0.5, 'WoE not available for top feature', ha='center')
    ax.set_title(f'WoE Distribution for {top_feature}')
st.pyplot(fig)

# Distribution of Revolving Utilization
fig, ax = plt.subplots()
sns.histplot(data=df, x='RevolvingUtilizationOfUnsecuredLines', hue='default', bins=30, ax=ax)
ax.set_title('Distribution of Revolving Utilization by Default Status')
ax.set_xlabel('Revolving Utilization')
ax.set_ylabel('Count')
st.pyplot(fig)

# Confusion Matrix
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix - XGBoost')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Age vs. Default Rate
fig, ax = plt.subplots()
age_default_rate = df.groupby('AGE_GROUP')['default'].mean()
sns.barplot(x='AGE_GROUP', y='default', data=age_default_rate.reset_index(), ax=ax)
ax.set_title('Default Rate by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Default Rate')
st.pyplot(fig)

# Age vs. Predicted PD (Scatter)
fig, ax = plt.subplots()
df_test = X_test.copy()
df_test['default'] = y_test
df_test['pred_prob'] = y_pred_prob
df_test['age'] = df['age'].iloc[X_test.index]
sns.scatterplot(x='age', y='pred_prob', hue='default', data=df_test, alpha=0.5, ax=ax)
ax.set_title('Age vs. Predicted Probability of Default')
ax.set_xlabel('Age')
ax.set_ylabel('Predicted Probability of Default')
st.pyplot(fig)

# Age vs. Average Predicted PD
fig, ax = plt.subplots()
age_pred = df_test.groupby(df['AGE_GROUP'].iloc[X_test.index])['pred_prob'].mean().reset_index()
sns.barplot(x='AGE_GROUP', y='pred_prob', data=age_pred, ax=ax)
ax.set_title('Average Predicted PD by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Average Predicted Probability of Default')
st.pyplot(fig)

# Feature Correlation Heatmap
fig, ax = plt.subplots()
corr = df[numerical_cols + ['default']].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

# PD Distribution
fig, ax = plt.subplots()
sns.histplot(y_pred_prob, bins=30, kde=True, ax=ax)
ax.set_title('Distribution of Predicted Default Probabilities')
ax.set_xlabel('Predicted Probability of Default')
ax.set_ylabel('Count')
st.pyplot(fig)

# Calibration Plot
fig, ax = plt.subplots()
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
ax.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Plot - XGBoost')
ax.legend()
st.pyplot(fig)

# Lift Chart
fig, ax = plt.subplots()
sorted_probs = np.sort(y_pred_prob)[::-1]
sorted_indices = np.argsort(y_pred_prob)[::-1]
sorted_labels = y_test.iloc[sorted_indices].values
cumulative_positives = np.cumsum(sorted_labels)
cumulative_population = np.arange(1, len(sorted_labels) + 1)
baseline = cumulative_positives[-1] / len(sorted_labels) * cumulative_population
lift = cumulative_positives / baseline
ax.plot(cumulative_population / len(sorted_labels), lift, label='Lift Curve')
ax.set_xlabel('Fraction of Population')
ax.set_ylabel('Lift')
ax.set_title('Lift Chart - XGBoost')
ax.legend()
st.pyplot(fig)

# Enhanced Analysis
st.subheader("Enhanced Analysis")

# Dataset Insights
default_rate = df['default'].mean()
st.write(f"**Dataset Default Rate:** {default_rate:.3f} ({default_rate*100:.1f}%)")
st.write(f"**Number of Features Selected (IV > {iv_threshold}):** {len(selected_features)}")
st.write(f"**Total Features Analyzed:** {len(features)}")

# Age Dependency Analysis
st.write("**Age Dependency Analysis:**")
age_default_rate_df = df.groupby('AGE_GROUP')['default'].mean().reset_index()
st.write(age_default_rate_df.rename(columns={'default': 'Default Rate'}))
st.write("**Average Predicted PD by Age Group:**")
st.write(age_pred)

# Feature Importance Tables
st.write("**Top 5 Features by IV:**")
st.write(iv_df.head(5)[['Feature', 'IV']])

st.write("**Top 5 Features by SHAP Importance:**")
shap_df = pd.DataFrame({
    'Feature': selected_features,
    'SHAP Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP Importance', ascending=False).head(5)
st.write(shap_df)

# Confusion Matrix Insights
st.write("**Confusion Matrix Insights:**")
st.write(f"True Negatives (No Default, Predicted No Default): {cm[0,0]}")
st.write(f"False Positives (No Default, Predicted Default): {cm[0,1]}")
st.write(f"False Negatives (Default, Predicted No Default): {cm[1,0]}")
st.write(f"True Positives (Default, Predicted Default): {cm[1,1]}")
st.write(f"**Recall for Defaults (Sensitivity):** {cm[1,1] / (cm[1,1] + cm[1,0]):.3f}")
st.write(f"**Precision for Defaults:** {cm[1,1] / (cm[1,1] + cm[0,1]):.3f}")
