
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="LoanAI | Smart Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------- PREMIUM DARK UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080D1B, #15152F, #0B1C30);
    color: #E8EDFF;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 {
    color: #A78BFA !important;
    font-weight: 800 !important;
}
h2, h3 {
    color: #67E8F9 !important;
}
p, label, .stMarkdown {
    color: #D9E2F5;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1A2340, #262044);
    border: 1px solid #48436F;
    border-radius: 16px;
    padding: 18px;
}
[data-testid="stMetricLabel"] {
    color: #B8C6E5 !important;
}
[data-testid="stMetricValue"] {
    color: #67E8F9 !important;
}
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: #151D33;
    border: 1px solid #353D5C;
    border-radius: 14px;
}
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #7C3AED, #4F46E5);
    color: white !important;
    border: 1px solid #8B5CF6;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #67E8F9;
}
[data-testid="stFileUploader"] {
    background: #151D33;
    border: 1px dashed #7C5CFC;
    border-radius: 14px;
    padding: 14px;
}
hr {
    border-color: #343C5B;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## 🏦 LoanAI")
st.caption("SMART BANKING • MACHINE LEARNING • LOAN ANALYTICS")
st.title("AI Loan Prediction Dashboard")
st.write(
    "Explore your dataset, analyse model performance, "
    "and predict loan status using Machine Learning."
)
st.divider()

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload your Loan Prediction CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif Path("Loan Prediction Dataset.csv").exists():
    df = pd.read_csv("Loan Prediction Dataset.csv")
else:
    st.info(
        "Please upload your Loan Prediction Dataset.csv file to continue."
    )
    st.stop()

df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

if "Loan_Status" not in df.columns:
    st.error(
        "Loan_Status column not found. "
        "Please check your CSV column names."
    )
    st.stop()

df = df.dropna(subset=["Loan_Status"]).copy()

if df.empty or df["Loan_Status"].nunique() < 2:
    st.error("The dataset must contain at least two Loan_Status classes.")
    st.stop()

# ---------------- PREPARE DATA ----------------
X = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
y = df["Loan_Status"].astype(str).str.strip()

# Remove rows with missing target labels
valid = y.ne("") & y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].copy()

if X.empty or X.shape[1] == 0 or y.nunique() < 2:
    st.error("The dataset does not contain enough usable data.")
    st.stop()

# Detect numeric and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Prepare preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_cols:
    transformers.append(("numeric", numeric_pipeline, numeric_cols))

if categorical_cols:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Split dataset safely
class_counts = y.value_counts()
stratify_target = y if class_counts.min() >= 2 else None

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# ---------------- TRAIN MODEL ----------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Dashboard",
        "🔮 Predict Loan",
        "📊 Dataset Insights",
        "🧠 Model Insights"
    ]
)

st.sidebar.divider()
st.sidebar.caption("LoanAI • Machine Learning Project")

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    st.subheader("✨ Project Overview")

    approved_count = int((y.str.upper() == "Y").sum())
    rejected_count = int((y.str.upper() == "N").sum())
    approval_rate = approved_count / len(y) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Training Rows", f"{len(X_train):,}")
    c3.metric("Testing Rows", f"{len(X_test):,}")
    c4.metric("Model Accuracy", f"{accuracy:.2%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Loan Status Distribution")
        status_counts = y.value_counts().rename_axis(
            "Loan Status"
        ).to_frame("Records")
        st.bar_chart(status_counts)

    with right:
        st.subheader("🎯 Approval Overview")
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
        st.metric("Approved Records", approved_count)
        st.metric("Rejected Records", rejected_count)

    st.info(
        "Note: Model accuracy is measured on the held-out test dataset. "
        "It does not guarantee that a real loan application will be approved."
    )

    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head(20), use_container_width=True)

# ---------------- LOAN PREDICTION ----------------
elif page == "🔮 Predict Loan":

    st.subheader("🔮 Smart Loan Predictor")
    st.write("Enter the applicant details below.")

    with st.form("loan_prediction_form"):

        applicant = {}

        cols = st.columns(2)

        for i, column in enumerate(X.columns):
            with cols[i % 2]:

                if column in numeric_cols:
                    values = pd.to_numeric(
                        X[column], errors="coerce"
                    ).dropna()

                    default = (
                        float(values.median())
                        if not values.empty else 0.0
                    )

                    if column == "Credit_History":
                        applicant[column] = st.selectbox(
                            "Credit History",
                            options=[1.0, 0.0],
                            index=0
                        )
                    elif column == "Dependents":
                        applicant[column] = st.selectbox(
                            "Dependents",
                            options=[0, 1, 2, 3],
                            index=0
                        )
                    else:
                        applicant[column] = st.number_input(
                            column.replace("_", " "),
                            value=default,
                            step=1.0,
                            format="%.2f"
                        )

                else:
                    options = (
                        X[column]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .index
                        .tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    applicant[column] = st.selectbox(
                        column.replace("_", " "),
                        options=options
                    )

        submitted = st.form_submit_button(
            "🚀 Predict Loan Status",
            use_container_width=True
        )

    if submitted:
        applicant_df = pd.DataFrame(
            [applicant],
            columns=X.columns
        )

        prediction = model.predict(applicant_df)[0]

        probabilities = model.predict_proba(applicant_df)[0]
        classes = model.named_steps["model"].classes_
        confidence = float(
            probabilities[list(classes).index(prediction)] * 100
        )

        is_approved = str(prediction).strip().upper() in [
            "Y", "YES", "APPROVED", "1", "TRUE"
        ]

        if is_approved:
            st.success("✅ Model Prediction: Loan Approved")
        else:
            st.error("❌ Model Prediction: Loan Rejected")

        st.metric("Model Confidence", f"{confidence:.1f}%")

        st.caption(
            "This is a model prediction, not an official bank decision. "
            "Confidence is not a guarantee of correctness."
        )

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        result_row = applicant.copy()
        result_row["Predicted_Status"] = prediction
        result_row["Model_Confidence_Percent"] = round(confidence, 2)

        st.session_state.prediction_history.append(result_row)

        st.subheader("📋 Applicant Details")
        st.dataframe(
            applicant_df,
            use_container_width=True
        )

    if st.session_state.get("prediction_history"):
        st.divider()
        st.subheader("🗂️ Prediction History")

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Prediction History (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="loan_prediction_results.csv",
            mime="text/csv"
        )

# ---------------- DATASET INSIGHTS ----------------
elif page == "📊 Dataset Insights":

    st.subheader("📊 Explore Your Dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}"
    )

    st.subheader("Loan Status Distribution")
    st.bar_chart(
        y.value_counts().rename_axis("Status").to_frame("Count")
    )

    if "ApplicantIncome" in df.columns:
        income = pd.to_numeric(
            df["ApplicantIncome"], errors="coerce"
        ).dropna()

        if not income.empty:
            st.subheader("💰 Applicant Income Distribution")
            st.bar_chart(
                income.clip(upper=income.quantile(0.95))
                .reset_index(drop=True)
            )

    st.subheader("🧹 Missing Values by Column")
    missing = df.isna().sum().sort_values(ascending=False)
    st.bar_chart(missing)

    st.subheader("📄 Full Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="loan_dataset.csv",
        mime="text/csv"
    )

# ---------------- MODEL INSIGHTS ----------------
elif page == "🧠 Model Insights":

    st.subheader("🧠 Machine Learning Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Trees", "200")
    c3.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("📊 Confusion Matrix")

    labels = model.named_steps["model"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )

    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        "Rows represent actual labels; columns represent predicted labels."
    )

    st.subheader("⭐ Feature Importance")

    try:
        feature_names = (
            model.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importance = model.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(15)

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"]
        )

    except Exception:
        st.warning(
            "Feature importance could not be displayed for this dataset."
        )

    st.subheader("📝 Model Details")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Train-test split: 80% training, 20% testing")
    st.write("Preprocessing: Missing-value imputation and one-hot encoding")
    st.write("Random state: 42")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "🏦 LoanAI | Built with Python, Pandas, Scikit-learn and Streamlit"
)
st.caption(
    "Educational project only • Predictions are not financial advice."
)
