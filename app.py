
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import hashlib

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Intelligence | AI Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>
    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #14213d, #284b75);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dce6f5;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: #f5f7fb;
        border: 1px solid #e3e8f0;
        padding: 16px;
        border-radius: 12px;
    }

    .result-card {
        background: #f5f7fb;
        border: 1px solid #e3e8f0;
        padding: 20px;
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL FEATURE DEFINITIONS
# =========================================================

numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

feature_order = numeric_features + categorical_features


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "models/preprocessor.joblib"
    )

    classifier = xgb.XGBClassifier()

    classifier.load_model(
        "models/xgb_model.json"
    )

    return preprocessor, classifier


@st.cache_data
def load_config():

    with open("models/model_config.json", "r") as file:
        return json.load(file)


try:

    preprocessor, classifier = load_models()
    config = load_config()

except Exception as error:

    st.error("Unable to load the trained AI model.")
    st.exception(error)
    st.stop()


threshold = float(config["decision_threshold"])


# =========================================================
# 5. GET CATEGORIES FROM TRAINED PREPROCESSOR
# =========================================================

def find_encoders(obj, visited=None):

    if visited is None:
        visited = set()

    if id(obj) in visited:
        return []

    visited.add(id(obj))

    found = []

    if hasattr(obj, "categories_"):
        found.append(obj)

    if hasattr(obj, "named_steps"):

        for step in obj.named_steps.values():
            found.extend(find_encoders(step, visited))

    if hasattr(obj, "transformers_"):

        for _, transformer, _ in obj.transformers_:

            if transformer not in ["drop", "passthrough"]:
                found.extend(
                    find_encoders(transformer, visited)
                )

    if hasattr(obj, "steps"):

        for _, step in obj.steps:
            found.extend(find_encoders(step, visited))

    return found


category_options = {}

for encoder in find_encoders(preprocessor):

    if hasattr(encoder, "feature_names_in_"):

        columns = list(encoder.feature_names_in_)

        for column, values in zip(
            columns,
            encoder.categories_
        ):

            category_options[column] = [
                str(value) for value in values
            ]


missing_categories = [
    column
    for column in categorical_features
    if column not in category_options
]

if missing_categories:

    st.error(
        "Unable to retrieve categories from the trained "
        "preprocessor: " + ", ".join(missing_categories)
    )

    st.stop()


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def validate_dataframe(data):

    missing = [
        column
        for column in feature_order
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data = data[feature_order].copy()

    if data.isnull().any().any():

        columns = data.columns[
            data.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing values found in: "
            + ", ".join(columns)
            + ". Please clean the dataset first."
        )

    for column in numeric_features:

        data[column] = pd.to_numeric(
            data[column],
            errors="raise"
        )

    for column in categorical_features:

        data[column] = data[column].astype(str)

    return data


def make_prediction(data):

    transformed = preprocessor.transform(data)

    scores = classifier.predict_proba(
        transformed
    )[:, 1]

    predictions = (
        scores >= threshold
    ).astype(int)

    return transformed, scores, predictions


def get_feature_names():

    try:

        names = list(
            preprocessor.get_feature_names_out()
        )

        return names

    except Exception:

        booster_names = (
            classifier.get_booster().feature_names
        )

        if booster_names:
            return booster_names

        return [
            f"Feature {i + 1}"
            for i in range(
                classifier.get_booster().num_features()
            )
        ]


def calculate_shap(transformed):

    matrix = xgb.DMatrix(transformed)

    contributions = classifier.get_booster().predict(
        matrix,
        pred_contribs=True
    )

    # Last column is the expected/base value
    shap_values = contributions[:, :-1]

    return shap_values


def make_explanation_dataframe(shap_values):

    feature_names = get_feature_names()

    if len(feature_names) != shap_values.shape[1]:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(shap_values.shape[1])
        ]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": shap_values
    })

    explanation["Absolute Impact"] = (
        explanation["Contribution"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return explanation


def add_prediction_columns(original, scores, predictions):

    results = original.copy()

    results["Risk_Score"] = scores

    results["Predicted_Default"] = predictions

    results["Risk_Classification"] = np.where(
        predictions == 1,
        "Higher Predicted Risk",
        "Lower Predicted Risk"
    )

    return results


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🏦 Loan Intelligence")

    st.caption("Explainable AI Risk Assessment")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Dashboard",
            "Individual Analysis",
            "Dataset Analysis",
            "Model Explainability"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** XGBoost")

    st.write(
        f"**Threshold:** {threshold:.2f}"
    )

    st.write(
        f"**Training Records:** "
        f"{config.get('dataset_rows', 0):,}"
    )

    st.divider()

    st.success("AI Model Operational")

    st.caption(
        "Educational and research decision-support system."
    )


# =========================================================
# 8. APPLICATION HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🏦 Loan Default Risk Analyzer</h1>
    <p>
        Intelligent loan risk assessment powered by
        Explainable Artificial Intelligence and XGBoost.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Overview Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Machine Learning Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    with col3:
        st.metric(
            "Training Dataset",
            f"{config.get('dataset_rows', 0):,}"
        )

    st.divider()

    st.markdown("### What can this system do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 👤 Individual Analysis")

        st.write(
            "Enter borrower details and predict "
            "the default-risk class for one borrower."
        )

        st.markdown("#### 📂 Dataset Analysis")

        st.write(
            "Upload a CSV file and generate predictions "
            "for multiple borrowers."
        )

    with col2:

        st.markdown("#### 🧠 Explainable AI")

        st.write(
            "Understand which borrower features "
            "influence the model's predictions."
        )

        st.markdown("#### 📥 Download Reports")

        st.write(
            "Export prediction results for "
            "further analysis."
        )

    st.divider()

    st.info(
        "Use the navigation menu in the sidebar "
        "to explore the application."
    )


# =========================================================
# 10. INDIVIDUAL BORROWER ANALYSIS
# =========================================================

elif page == "Individual Analysis":

    st.markdown(
        '<div class="section-title">Individual Borrower Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the borrower's information to generate "
        "an AI-based loan default risk assessment."
    )

    with st.form("borrower_form"):

        st.markdown("### Financial Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            income = st.number_input(
                "Annual Income",
                min_value=0,
                value=50000,
                step=1000
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650
            )

        with c2:

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=10000,
                step=1000
            )

            months_employed = st.number_input(
                "Months Employed",
                min_value=0,
                value=60
            )

            num_credit_lines = st.number_input(
                "Number of Credit Lines",
                min_value=0,
                value=3
            )

        with c3:

            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0
            )

            loan_term = st.number_input(
                "Loan Term (Months)",
                min_value=1,
                value=36
            )

            dti_ratio = st.number_input(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.01
            )

        st.divider()

        st.markdown("### Borrower Background")

        c4, c5 = st.columns(2)

        with c4:

            education = st.selectbox(
                "Education",
                category_options["Education"]
            )

            employment_type = st.selectbox(
                "Employment Type",
                category_options["EmploymentType"]
            )

            marital_status = st.selectbox(
                "Marital Status",
                category_options["MaritalStatus"]
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                category_options["LoanPurpose"]
            )

        with c5:

            has_mortgage = st.selectbox(
                "Has Mortgage?",
                category_options["HasMortgage"]
            )

            has_dependents = st.selectbox(
                "Has Dependents?",
                category_options["HasDependents"]
            )

            has_cosigner = st.selectbox(
                "Has Co-Signer?",
                category_options["HasCoSigner"]
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Borrower",
            use_container_width=True
        )

    if submitted:

        borrower = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "Education": education,
            "EmploymentType": employment_type,
            "MaritalStatus": marital_status,
            "HasMortgage": has_mortgage,
            "HasDependents": has_dependents,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": has_cosigner
        }])

        try:

            transformed, scores, predictions = make_prediction(
                borrower
            )

            score = float(scores[0])
            prediction = int(predictions[0])

            st.session_state["individual_result"] = {
                "borrower": borrower,
                "transformed": transformed,
                "score": score,
                "prediction": prediction
            }

        except Exception as error:

            st.error("Prediction failed.")
            st.exception(error)

    if "individual_result" in st.session_state:

        result = st.session_state["individual_result"]

        score = result["score"]
        prediction = result["prediction"]

        st.divider()

        st.markdown("### 📊 Prediction Results")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Model Risk Score",
                f"{score * 100:.2f}%"
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

        with c2:

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )

            if prediction == 1:

                st.error(
                    "Higher Predicted Default Risk Class"
                )

            else:

                st.success(
                    "Lower Predicted Default Risk Class"
                )

        st.caption(
            "The model risk score is not a calibrated "
            "real-world probability of default."
        )

        st.markdown("### Borrower Details")

        st.dataframe(
            result["borrower"],
            use_container_width=True,
            hide_index=True
        )

        # LOCAL SHAP EXPLANATION

        st.divider()

        st.markdown("### 🧠 Individual AI Explanation")

        st.write(
            "These TreeSHAP contributions show how "
            "each feature influenced the model's raw score."
        )

        with st.spinner("Calculating feature contributions..."):

            shap_values = calculate_shap(
                result["transformed"]
            )[0]

            explanation = make_explanation_dataframe(
                shap_values
            )

        top_features = explanation.head(10).copy()

        st.bar_chart(
            top_features.set_index("Feature")[
                "Contribution"
            ],
            horizontal=True
        )

        st.dataframe(
            top_features[
                ["Feature", "Contribution", "Absolute Impact"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive SHAP values push the raw model output "
            "toward the default class. Negative values push "
            "it away. These are not percentage-point changes."
        )


# =========================================================
# 11. DATASET UPLOAD AND BATCH ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.markdown(
        '<div class="section-title">Dataset Upload & Batch Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload your loan dataset to analyze multiple "
        "borrowers using the trained XGBoost model."
    )

    uploaded_file = st.file_uploader(
        "Upload Loan Dataset",
        type=["csv"],
        help="Upload a CSV containing the 16 predictor columns."
    )

    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.getvalue()

            fingerprint = hashlib.sha256(
                file_bytes
            ).hexdigest()

            if st.session_state.get("batch_fingerprint") != fingerprint:

                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_fingerprint", None)

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.success("Dataset uploaded successfully!")

            c1, c2 = st.columns(2)

            c1.metric(
                "Total Records",
                f"{len(uploaded_df):,}"
            )

            c2.metric(
                "Total Columns",
                len(uploaded_df.columns)
            )

            st.markdown("### Dataset Preview")

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )

            missing_features = [
                column
                for column in feature_order
                if column not in uploaded_df.columns
            ]

            if missing_features:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            elif len(uploaded_df) == 0:

                st.error("The uploaded dataset is empty.")

            else:

                st.info(
                    "The model will use the 16 trained "
                    "predictor columns. LoanID and an existing "
                    "Default column will not be used for prediction."
                )

                if st.button(
                    "🚀 Analyze Entire Dataset",
                    use_container_width=True
                ):

                    try:

                        clean_data = validate_dataframe(
                            uploaded_df
                        )

                        with st.spinner(
                            "Analyzing borrower records..."
                        ):

                            transformed, scores, predictions = (
                                make_prediction(clean_data)
                            )

                            results = add_prediction_columns(
                                uploaded_df,
                                scores,
                                predictions
                            )

                            st.session_state["batch_results"] = results

                            st.session_state["batch_fingerprint"] = fingerprint

                    except Exception as error:

                        st.error(
                            "Dataset analysis failed."
                        )

                        st.exception(error)

            # DISPLAY BATCH RESULTS

            if (
                st.session_state.get("batch_fingerprint")
                == fingerprint
                and "batch_results" in st.session_state
            ):

                results = st.session_state["batch_results"]

                st.divider()

                st.markdown("### 📊 Batch Analysis Results")

                total = len(results)

                high_risk = int(
                    (results["Predicted_Default"] == 1).sum()
                )

                lower_risk = total - high_risk

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Borrowers",
                    f"{total:,}"
                )

                c2.metric(
                    "Higher Predicted Risk",
                    f"{high_risk:,}"
                )

                c3.metric(
                    "Lower Predicted Risk",
                    f"{lower_risk:,}"
                )

                st.markdown("### Risk Distribution")

                distribution = pd.DataFrame({
                    "Classification": [
                        "Higher Predicted Risk",
                        "Lower Predicted Risk"
                    ],
                    "Borrowers": [
                        high_risk,
                        lower_risk
                    ]
                })

                st.bar_chart(
                    distribution.set_index("Classification")
                )

                st.markdown("### Prediction Results")

                st.dataframe(
                    results.head(100),
                    use_container_width=True
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Complete Prediction Report",
                    data=csv_data,
                    file_name="loan_risk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # GLOBAL TREE SHAP EXPLANATION

                st.divider()

                st.markdown(
                    "### 🌍 Global Explainability"
                )

                st.write(
                    "This analysis shows which features "
                    "had the greatest average absolute "
                    "contribution across a sample of borrowers."
                )

                sample_size = min(500, len(results))

                if st.button(
                    f"Calculate Global Explanations "
                    f"(sample of {sample_size} borrowers)"
                ):

                    try:

                        with st.spinner(
                            "Calculating global TreeSHAP values..."
                        ):

                            sample_data = clean_data.sample(
                                n=sample_size,
                                random_state=42
                            )

                            sample_transformed = (
                                preprocessor.transform(sample_data)
                            )

                            sample_shap = calculate_shap(
                                sample_transformed
                            )

                            mean_absolute_shap = np.abs(
                                sample_shap
                            ).mean(axis=0)

                            global_explanation = (
                                make_explanation_dataframe(
                                    mean_absolute_shap
                                )
                            )

                            st.session_state[
                                "global_explanation"
                            ] = global_explanation

                            st.session_state[
                                "global_fingerprint"
                            ] = fingerprint

                    except Exception as error:

                        st.error(
                            "Global explanation failed."
                        )

                        st.exception(error)

                if (
                    st.session_state.get("global_fingerprint")
                    == fingerprint
                    and "global_explanation" in st.session_state
                ):

                    global_explanation = st.session_state[
                        "global_explanation"
                    ]

                    st.markdown(
                        "#### Top 15 Influential Features"
                    )

                    st.bar_chart(
                        global_explanation.head(15).set_index(
                            "Feature"
                        )["Absolute Impact"],
                        horizontal=True
                    )

                    st.dataframe(
                        global_explanation.head(15),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Global explanations are calculated on "
                        "a sample of up to 500 uploaded borrowers. "
                        "Mean absolute TreeSHAP values represent "
                        "average influence on the raw model output."
                    )

        except Exception as error:

            st.error("Unable to read the uploaded CSV file.")
            st.exception(error)


# =========================================================
# 12. MODEL EXPLAINABILITY
# =========================================================

elif page == "Model Explainability":

    st.markdown(
        '<div class="section-title">Explainable AI & Model Insights</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand how the trained XGBoost model "
        "uses borrower features to produce predictions."
    )

    st.markdown("### Global Model Feature Importance")

    importance = classifier.get_booster().get_score(
        importance_type="gain"
    )

    if importance:

        feature_names = get_feature_names()

        importance_rows = []

        for key, value in importance.items():

            if key.startswith("f") and key[1:].isdigit():

                index = int(key[1:])

                if index < len(feature_names):
                    name = feature_names[index]
                else:
                    name = key

            else:
                name = key

            importance_rows.append({
                "Feature": name,
                "Importance": float(value)
            })

        importance_df = pd.DataFrame(
            importance_rows
        ).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.head(15).set_index("Feature"),
            horizontal=True
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "This chart uses XGBoost gain-based feature importance. "
            "It is not a SHAP explanation and does not show "
            "the direction of an individual prediction."
        )

    else:

        st.info(
            "The model does not contain gain-based "
            "feature importance values."
        )

    st.divider()

    st.markdown("### Model Configuration")

    st.json(config)


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "Loan Default Risk Analyzer | Explainable AI Portfolio Project"
)

st.caption(
    "This application is intended for educational and research "
    "purposes. It is not a substitute for professional credit "
    "assessment or a real-world lending decision."
)
