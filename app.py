import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fake Product Review Detection")
st.write("Machine Learning based product review analysis")

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("dataset/reviews.csv")

X = data["review"]
y = data["label"]

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vector, y_train)

# -----------------------------
# Test model
# -----------------------------
y_pred = model.predict(X_test_vector)

# -----------------------------
# Calculate metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label="fake"
)
recall = recall_score(
    y_test, y_pred, pos_label="fake"
)
f1 = f1_score(
    y_test, y_pred, pos_label="fake"
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["fake", "genuine"]
)

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Precision", f"{precision * 100:.2f}%")
col3.metric("Recall", f"{recall * 100:.2f}%")
col4.metric("F1 Score", f"{f1 * 100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📈 Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Genuine"],
    columns=["Predicted Fake", "Predicted Genuine"]
)

st.dataframe(cm_df)

# -----------------------------
# Review prediction
# -----------------------------
st.subheader("📝 Check a Product Review")

review = st.text_area(
    "Enter your product review:"
)

if st.button("Check Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        if prediction == "fake":
            st.error("🚨 Fake Review Detected")

        else:
            st.success("✅ Genuine Review")