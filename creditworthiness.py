import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit settings
st.set_page_config(page_title="Creditworthiness Predictor", layout="wide")
st.title("💳 Creditworthiness Prediction using Random Forest")

# Load dataset
@st.cache_data
def load_data():
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y

X, y = load_data()

# Encode features
def preprocess_data(X, y):
    X_encoded = X.copy()
    label_encoders = {}

    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

    y_encoded = y.iloc[:, 0]
    if y_encoded.dtype == "object":
        y_encoded = LabelEncoder().fit_transform(y_encoded)

    return X_encoded, y_encoded, label_encoders

X_encoded, y_encoded, label_encoders = preprocess_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display performance
st.subheader("📊 Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.dataframe(pd.DataFrame(report).transpose())
st.write("**Confusion Matrix:**")
st.dataframe(pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# Feature importance plot
st.subheader("📌 Feature Importance")
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="coolwarm")
st.pyplot(fig)

# ----------------------------------------
# 💡 User Input Section for Prediction
# ----------------------------------------

st.subheader("🧠 Predict Creditworthiness from User Input")

# Dynamically generate input fields based on original columns
user_input = {}
for column in X.columns:
    if column in label_encoders:
        options = list(label_encoders[column].classes_)
        user_input[column] = st.selectbox(f"{column}", options)
    else:
        user_input[column] = st.number_input(f"{column}", value=float(X[column].mean()))

# Prepare input for model
input_df = pd.DataFrame([user_input])

# Encode categorical fields
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Predict
if st.button("🔍 Predict Creditworthiness"):
    prediction = rf_model.predict(input_df)[0]
    prediction_text = "Creditworthy ✅" if prediction == 1 else "Not Creditworthy ❌"
    st.success(f"Prediction: **{prediction_text}**")
