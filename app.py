import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

st.title("🔐 Network Intrusion Detection System (NIDS)")

st.write("Upload NSL-KDD / network traffic dataset to detect attacks using Machine Learning")

# =============================
# Upload Dataset
# =============================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # =============================
    # Basic Cleaning
    # =============================
    df = df.dropna()

    # Assume last column is label
    label_col = df.columns[-1]

    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # Encode categorical columns
    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode labels
    y = LabelEncoder().fit_transform(y)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # =============================
    # Train/Test Split
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # =============================
    # Train Model
    # =============================
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model Accuracy: {round(acc*100,2)}%")

    # =============================
    # Predictions Table
    # =============================
    results = pd.DataFrame()
    results["Actual"] = y_test
    results["Predicted"] = y_pred

    results["Attack"] = np.where(results["Predicted"] == 0, "Normal", "Attack")

    st.subheader("🚨 Detection Results")
    st.dataframe(results.head(20))

    # =============================
    # Charts
    # =============================
    st.subheader("📈 Attack Distribution")

    attack_counts = results["Attack"].value_counts()

    fig = px.pie(
        values=attack_counts.values,
        names=attack_counts.index,
        title="Normal vs Attack Traffic"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Attack Count Bar Chart")

    bar = px.bar(
        x=attack_counts.index,
        y=attack_counts.values,
        labels={"x": "Traffic Type", "y": "Count"}
    )

    st.plotly_chart(bar, use_container_width=True)

    # =============================
    # Suspicious Traffic
    # =============================
    st.subheader("⚠️ Suspicious Traffic Samples")

    suspicious = results[results["Attack"] == "Attack"]

    st.dataframe(suspicious.head(20))

    st.download_button(
        "Download Suspicious Traffic CSV",
        suspicious.to_csv(index=False),
        file_name="suspicious_traffic.csv"
    )

else:
    st.info("Please upload NSL-KDD dataset CSV to start detection.")
