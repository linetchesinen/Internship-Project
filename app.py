import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IGOJI PWD Inclusion Intelligence",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Spark Data.xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "IGOJI_REGISTRATION_GAP_MODEL.joblib"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
        linear-gradient(
            135deg,
            #f4fbfc 0%,
            #ffffff 50%,
            #eef8f7 100%
        );
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #063b4c 0%,
            #075e6b 50%,
            #07838d 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main headings */
    h1 {
        color: #063b4c;
        font-weight: 800;
    }

    h2 {
        color: #075e6b;
        font-weight: 750;
    }

    h3 {
        color: #087d87;
        font-weight: 700;
    }

    /* Hero */
    .hero {
        padding: 45px;
        border-radius: 25px;
        margin-bottom: 30px;

        background:
        linear-gradient(
            135deg,
            rgba(6,59,76,0.98),
            rgba(7,131,141,0.94)
        );

        box-shadow:
        0 15px 40px rgba(6,59,76,0.20);
    }

    .hero-title {
        color: white;
        font-size: 45px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-text {
        color: #e8ffff;
        font-size: 18px;
        line-height: 1.7;
    }

    /* Cards */
    .card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        min-height: 180px;

        box-shadow:
        0 8px 28px rgba(6,59,76,0.08);

        border: 1px solid #e2eff1;
    }

    .card:hover {
        box-shadow:
        0 15px 35px rgba(6,59,76,0.14);
    }

    /* KPI */
    .kpi {
        background: white;
        padding: 24px;
        border-radius: 18px;
        text-align: center;

        box-shadow:
        0 7px 24px rgba(6,59,76,0.08);

        border-top: 5px solid #07838d;
    }

    .kpi-number {
        font-size: 32px;
        font-weight: 800;
        color: #063b4c;
    }

    .kpi-label {
        color: #66777d;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Section */
    .section {
        background: white;
        padding: 28px;
        border-radius: 20px;
        margin: 20px 0;

        box-shadow:
        0 7px 25px rgba(6,59,76,0.06);
    }

    /* Prediction */
    .prediction-gap {
        background: #fff3e8;
        border-left: 7px solid #e27b25;
        padding: 28px;
        border-radius: 18px;
    }

    .prediction-normal {
        background: #eaf8f0;
        border-left: 7px solid #208b5a;
        padding: 28px;
        border-radius: 18px;
    }

    .prediction-title {
        font-size: 25px;
        font-weight: 800;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        padding: 30px;
        text-align: center;
        border-top: 1px solid #dce8eb;
        color: #68777d;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    try:
        df = pd.read_excel(DATA_PATH)

        # Remove completely empty rows
        df = df.dropna(
            how="all"
        ).reset_index(drop=True)

        return df

    except Exception as e:

        st.error(
            f"Could not read the Excel file: {e}"
        )

        return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        return joblib.load(
            MODEL_PATH
        )

    except Exception as e:

        st.error(
            f"Could not load the trained model: {e}"
        )

        return None


df = load_data()
model = load_model()

# ============================================================
# IMAGE FUNCTION
# ============================================================

def get_image(filename):

    path = os.path.join(
        ASSETS_DIR,
        filename
    )

    if os.path.exists(path):

        try:
            return Image.open(path)

        except:
            return None

    return None


# ============================================================
# CLEANING FUNCTIONS FOR DASHBOARD
# ============================================================

def clean_sex(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().upper()

    if value in ["F", "FEMALE", "FAMELE", "FEMBLE"]:
        return "Female"

    if value in ["M", "MALE", "SEMALE", "EKMALE"]:
        return "Male"

    return "Unknown"


def clean_disability(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().upper()

    if "PHYS" in value:
        return "Physical"

    if any(
        x in value
        for x in [
            "VISUAL",
            "BLIND"
        ]
    ):
        return "Visual"

    if any(
        x in value
        for x in [
            "DEAF",
            "HEARING"
        ]
    ):
        return "Hearing"

    if "SPEECH" in value:
        return "Speech"

    if "EPILEP" in value:
        return "Epilepsy"

    if any(
        x in value
        for x in [
            "MENTAL",
            "PSYCHO"
        ]
    ):
        return "Mental/Psychosocial"

    if any(
        x in value
        for x in [
            "INTELLECT",
            "AUTISM",
            "DEVELOP"
        ]
    ):
        return "Intellectual/Developmental"

    if value in [
        "",
        "NAN",
        "NONE"
    ]:
        return "Unknown"

    return "Other"


def clean_age(value):

    if pd.isna(value):
        return np.nan

    try:

        text = str(value).strip()

        # Remove common words
        text = (
            text
            .replace("yrs", "")
            .replace("YRS", "")
            .replace("years", "")
            .replace("YEAR", "")
            .replace("yrs.", "")
            .strip()
        )

        number = float(text)

        # Possible birth year
        if 1900 <= number <= 2026:

            age = 2026 - number

            if 0 <= age <= 120:
                return age

        if 0 <= number <= 120:
            return number

    except:
        pass

    return np.nan


def age_group(age):

    if pd.isna(age):
        return "Unknown"

    age = float(age)

    if age <= 5:
        return "0-5"

    if age <= 12:
        return "6-12"

    if age <= 17:
        return "13-17"

    if age <= 35:
        return "18-35"

    if age <= 60:
        return "36-60"

    return "61+"


# ============================================================
# PREPARE ANALYTICS DATA
# ============================================================

if df is not None:

    analytics = df.copy()

    analytics["Sex_Clean"] = (
        analytics["Sex"]
        .apply(clean_sex)
    )

    analytics["Age_Clean"] = (
        analytics["Age"]
        .apply(clean_age)
    )

    analytics["Age_Group"] = (
        analytics["Age_Clean"]
        .apply(age_group)
    )

    analytics["Disability_Clean"] = (
        analytics["Type of Disability"]
        .apply(clean_disability)
    )

    analytics["Belongs_To_Group"] = (
        analytics["Group Name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    # Registration gap proxy
    ncpwd_recorded = (
        analytics["NCPWD Certificate No."]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    sacco_recorded = (
        analytics["PWD SACCO No."]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    analytics["Registration_Gap"] = (
        (~ncpwd_recorded)
        &
        (~sacco_recorded)
    ).astype(int)

else:

    analytics = None


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    logo = get_image(
        "logo.png"
    )

    if logo is not None:

        st.image(
            logo,
            use_container_width=True
        )

    else:

        st.markdown(
            "<div style='text-align:center;"
            "font-size:55px;'>♿</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <h2 style="
        text-align:center;
        color:white !important;
        ">
        IGOJI PWD
        </h2>

        <p style="
        text-align:center;
        color:#e4ffff !important;
        ">
        Inclusion Intelligence
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "MENU",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🤖 Prediction",
            "📈 Model",
            "📸 Gallery",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    if model is not None:

        st.success(
            "✓ Model connected"
        )

    else:

        st.warning(
            "Model not connected"
        )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    hero = get_image(
        "hero.jpg"
    )

    if hero is not None:

        st.image(
            hero,
            use_container_width=True
        )

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        IGOJI PWD Inclusion Intelligence
        </div>

        <div class="hero-text">

        Predicting Gaps in Disability Programme
        Registration and Inclusion in Igoji Sub-County

        <br><br>

        A machine-learning decision-support system
        designed to transform disability programme data
        into useful insights for community planning,
        outreach and inclusion.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI
    if analytics is not None:

        total = len(analytics)

        groups = analytics[
            "Belongs_To_Group"
        ].sum()

        gaps = analytics[
            "Registration_Gap"
        ].sum()

        gap_rate = (
            gaps / total * 100
            if total > 0 else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="kpi">
                <div class="kpi-number">
                {total:,}
                </div>
                <div class="kpi-label">
                Total Records
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="kpi">
                <div class="kpi-number">
                {groups:,}
                </div>
                <div class="kpi-label">
                Group Records
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="kpi">
                <div class="kpi-number">
                {gaps:,}
                </div>
                <div class="kpi-label">
                Documented Gap Proxy
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="kpi">
                <div class="kpi-number">
                {gap_rate:.1f}%
                </div>
                <div class="kpi-label">
                Gap Proxy Rate
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "## 🌍 Turning Community Data Into Insight"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="card">

            <h3>📊 Data Driven</h3>

            <p>
            Analyze demographic, disability and geographic
            patterns contained in the community dataset.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

            <h3>🤖 Machine Learning</h3>

            <p>
            Use the trained classification model to identify
            patterns associated with the documented registration
            and inclusion gap proxy.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">

            <h3>🤝 Inclusive Planning</h3>

            <p>
            Support evidence-informed outreach, documentation
            support, programme planning and inclusion initiatives.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 IGOJI PWD Community Dashboard"
    )

    if analytics is None:

        st.error(
            "Excel data could not be loaded."
        )

    else:

        # --------------------------------------------
        # FILTERS
        # --------------------------------------------

        st.markdown(
            "### 🔎 Filter the Dashboard"
        )

        f1, f2, f3 = st.columns(3)

        with f1:

            subcounties = sorted(
                analytics[
                    "Subcounty"
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            selected_subcounty = st.selectbox(
                "Subcounty",
                ["All"] + subcounties
            )

        with f2:

            sex_filter = st.selectbox(
                "Sex",
                [
                    "All",
                    "Female",
                    "Male",
                    "Unknown"
                ]
            )

        with f3:

            disability_filter = st.selectbox(
                "Disability",
                [
                    "All"
                ]
                +
                sorted(
                    analytics[
                        "Disability_Clean"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        filtered = analytics.copy()

        if selected_subcounty != "All":

            filtered = filtered[
                filtered[
                    "Subcounty"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                ==
                selected_subcounty.strip().upper()
            ]

        if sex_filter != "All":

            filtered = filtered[
                filtered["Sex_Clean"]
                ==
                sex_filter
            ]

        if disability_filter != "All":

            filtered = filtered[
                filtered["Disability_Clean"]
                ==
                disability_filter
            ]

        # --------------------------------------------
        # KPI
        # --------------------------------------------

        total = len(filtered)

        gaps = int(
            filtered[
                "Registration_Gap"
            ].sum()
        )

        groups = int(
            filtered[
                "Belongs_To_Group"
            ].sum()
        )

        gap_rate = (
            gaps / total * 100
            if total > 0
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Records",
                f"{total:,}"
            )

        with c2:
            st.metric(
                "Group Records",
                f"{groups:,}"
            )

        with c3:
            st.metric(
                "Documented Gap Proxy",
                f"{gaps:,}"
            )

        with c4:
            st.metric(
                "Gap Proxy Rate",
                f"{gap_rate:.1f}%"
            )

        st.markdown("---")

        # --------------------------------------------
        # SEX AND DISABILITY
        # --------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "👥 Sex Distribution"
            )

            sex_counts = (
                filtered[
                    "Sex_Clean"
                ]
                .value_counts()
            )

            st.bar_chart(
                sex_counts
            )

        with c2:

            st.subheader(
                "♿ Disability Categories"
            )

            disability_counts = (
                filtered[
                    "Disability_Clean"
                ]
                .value_counts()
            )

            st.bar_chart(
                disability_counts
            )

        # --------------------------------------------
        # AGE
        # --------------------------------------------

        st.subheader(
            "🎂 Age Group Distribution"
        )

        age_counts = (
            filtered[
                "Age_Group"
            ]
            .value_counts()
            .reindex(
                [
                    "0-5",
                    "6-12",
                    "13-17",
                    "18-35",
                    "36-60",
                    "61+",
                    "Unknown"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            age_counts
        )

        # --------------------------------------------
        # GEOGRAPHY
        # --------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "📍 Subcounty"
            )

            subcounty_counts = (
                filtered[
                    "Subcounty"
                ]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .value_counts()
                .head(15)
            )

            st.bar_chart(
                subcounty_counts
            )

        with c2:

            st.subheader(
                "🗺️ Ward"
            )

            ward_counts = (
                filtered[
                    "Ward"
                ]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .value_counts()
                .head(15)
            )

            st.bar_chart(
                ward_counts
            )

        # --------------------------------------------
        # DATA
        # --------------------------------------------

        st.markdown("---")

        st.subheader(
            "📋 Dataset"
        )

        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            hide_index=True
        )

        # Download
        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered Data",
            data=csv,
            file_name="igoji_pwd_filtered_data.csv",
            mime="text/csv"
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.title(
        "🤖 Registration & Inclusion Gap Prediction"
    )

    st.markdown(
        """
        <div class="section">

        Enter the available information below.
        The trained machine-learning model will process
        the seven features used during model development.

        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "The trained model was not found."
        )

        st.info(
            "Place the model here:"
        )

        st.code(
            "IGOJI_REGISTRATION_GAP_MODEL.joblib"
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            sex = st.selectbox(
                "Sex",
                [
                    "Female",
                    "Male",
                    "Unknown"
                ]
            )

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
                step=1
            )

            disability = st.selectbox(
                "Disability Category",
                [
                    "Physical",
                    "Visual",
                    "Hearing",
                    "Speech",
                    "Epilepsy",
                    "Mental/Psychosocial",
                    "Intellectual/Developmental",
                    "Other",
                    "Unknown"
                ]
            )

        with c2:

            # Get actual values from dataset
            if analytics is not None:

                subcounty_options = sorted(
                    analytics[
                        "Subcounty"
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                ward_options = sorted(
                    analytics[
                        "Ward"
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

            else:

                subcounty_options = [
                    "IGOJI"
                ]

                ward_options = [
                    "IGOJI WEST",
                    "IGOJI EAST"
                ]

            subcounty = st.selectbox(
                "Subcounty",
                subcounty_options
            )

            ward = st.selectbox(
                "Ward",
                ward_options
            )

            group_status = st.selectbox(
                "Group Membership",
                [
                    "Group recorded",
                    "No group recorded"
                ]
            )

        calculated_age_group = age_group(
            age
        )

        st.info(
            f"Calculated Age Group: "
            f"**{calculated_age_group}**"
        )

        st.markdown(
            """
            **The model does not use:**

            - Name
            - ID
            - Telephone numbers
            - Contact person
            - NCPWD certificate number
            - PWD SACCO number

            These fields are intentionally excluded from prediction.
            """
        )

        st.markdown("---")

        predict = st.button(
            "🔍 GENERATE PREDICTION",
            type="primary",
            use_container_width=True
        )

        if predict:

            try:

                belongs_to_group = (
                    1
                    if group_status ==
                    "Group recorded"
                    else 0
                )

                # EXACT MODEL FEATURES
                input_data = pd.DataFrame(
                    {
                        "Sex_Clean": [
                            sex
                        ],

                        "Age_Clean": [
                            float(age)
                        ],

                        "Age_Group": [
                            calculated_age_group
                        ],

                        "Disability_Clean": [
                            disability
                        ],

                        "Subcounty_Clean": [
                            subcounty.strip()
                        ],

                        "Ward_Clean": [
                            ward.strip()
                        ],

                        "Belongs_To_Group": [
                            belongs_to_group
                        ]
                    }
                )

                # Ensure correct order
                input_data = input_data[
                    [
                        "Sex_Clean",
                        "Age_Clean",
                        "Age_Group",
                        "Disability_Clean",
                        "Subcounty_Clean",
                        "Ward_Clean",
                        "Belongs_To_Group"
                    ]
                ]

                # Prediction
                prediction = model.predict(
                    input_data
                )

                prediction = int(
                    np.asarray(
                        prediction
                    ).ravel()[0]
                )

                # Probability
                probability = None

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            input_data
                        )
                    )

                    if (
                        probabilities is not None
                        and len(probabilities) > 0
                    ):

                        probability = float(
                            probabilities[0][1]
                        )

                # ----------------------------------
                # RESULT
                # ----------------------------------

                st.markdown("---")

                if prediction == 1:

                    st.markdown(
                        """
                        <div class="prediction-gap">

                        <div class="prediction-title">
                        ⚠️ Documented Registration /
                        Inclusion Gap Pattern
                        </div>

                        <p>
                        The model identified a pattern associated
                        with the documented registration/inclusion
                        gap proxy.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="prediction-normal">

                        <div class="prediction-title">
                        ✓ No Documented Registration /
                        Inclusion Gap Pattern
                        </div>

                        <p>
                        The model did not identify the documented
                        gap pattern represented by the target.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if probability is not None:

                    st.markdown(
                        "### 📊 Prediction Probability"
                    )

                    probability_percent = (
                        probability * 100
                    )

                    st.progress(
                        probability
                    )

                    st.metric(
                        "Probability of Documented Gap Pattern",
                        f"{probability_percent:.1f}%"
                    )

                # ----------------------------------
                # SAVE HISTORY
                # ----------------------------------

                history_row = {
                    "Sex": sex,
                    "Age": age,
                    "Age Group": calculated_age_group,
                    "Disability": disability,
                    "Subcounty": subcounty,
                    "Ward": ward,
                    "Group": group_status,
                    "Prediction": (
                        "Gap Pattern"
                        if prediction == 1
                        else "No Gap Pattern"
                    )
                }

                if probability is not None:

                    history_row[
                        "Probability"
                    ] = round(
                        probability * 100,
                        2
                    )

                st.session_state.history.append(
                    history_row
                )

                st.warning(
                    "This output is a machine-learning "
                    "decision-support indicator. It should not "
                    "be used by itself to determine eligibility "
                    "for disability services, benefits or other "
                    "support."
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

        # History
        if st.session_state.history:

            st.markdown("---")

            st.subheader(
                "🧾 Prediction History"
            )

            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

            history_csv = (
                history_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Prediction History",
                data=history_csv,
                file_name="igoji_prediction_history.csv",
                mime="text/csv"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📈 Model":

    st.title(
        "📈 Machine Learning Model"
    )

    st.markdown(
        """
        <div class="section">

        <h3>🎯 What is the model predicting?</h3>

        <p>
        The dataset does not contain a direct official registration
        status variable. Therefore, the project uses a proxy target
        called <b>Registration_Gap</b>.
        </p>

        <p>
        The proxy is coded as <b>1</b> when neither an NCPWD
        certificate number nor a PWD SACCO number is recorded,
        and <b>0</b> when at least one is recorded.
        </p>

        <p>
        This represents a documented information gap in the
        available data. It should not automatically be interpreted
        as proof of official non-registration.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "🧠 Model Features"
    )

    features = pd.DataFrame(
        {
            "Feature": [
                "Sex_Clean",
                "Age_Clean",
                "Age_Group",
                "Disability_Clean",
                "Subcounty_Clean",
                "Ward_Clean",
                "Belongs_To_Group"
            ],

            "Purpose": [
                "Cleaned sex category",
                "Validated age",
                "Age category",
                "Standardized disability category",
                "Subcounty",
                "Ward",
                "Whether group information is recorded"
            ]
        }
    )

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "🚫 Leakage Prevention"
    )

    st.write(
        """
        NCPWD certificate numbers and PWD SACCO numbers are not
        used as model predictors because they were used to construct
        the target. Including them would create target leakage.

        Personal identifiers and contact information are also excluded.
        """
    )

    st.subheader(
        "🔬 Models Used During Development"
    )

    model_table = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],

            "Role": [
                "Interpretable baseline",
                "Tree-based ensemble",
                "Gradient boosting"
            ]
        }
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# GALLERY
# ============================================================

elif page == "📸 Gallery":

    st.title(
        "📸 Project Gallery"
    )

    st.write(
        "Add appropriate photographs from the project, "
        "community activities, data collection or outreach."
    )

    gallery_files = [
        "pwd1.jpg",
        "pwd2.jpg",
        "pwd3.jpg"
    ]

    available_images = []

    for filename in gallery_files:

        image = get_image(
            filename
        )

        if image is not None:

            available_images.append(
                (
                    filename,
                    image
                )
            )

    if available_images:

        columns = st.columns(
            min(
                3,
                len(available_images)
            )
        )

        for i, (
            filename,
            image
        ) in enumerate(
            available_images
        ):

            with columns[
                i % len(columns)
            ]:

                st.image(
                    image,
                    caption=filename,
                    use_container_width=True
                )

    else:

        st.info(
            "No project photographs have been added yet."
        )

    st.markdown("---")

    st.subheader(
        "📤 Upload a Photograph"
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded:

        image = Image.open(
            uploaded
        )

        st.image(
            image,
            caption="Uploaded Project Image",
            use_container_width=True
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## Predicting Gaps in Disability Programme
        ## Registration and Inclusion in Igoji Sub-County

        This project applies machine-learning techniques to
        community disability data to identify patterns associated
        with documented gaps in disability programme registration
        and inclusion.

        The system is intended to support evidence-informed
        community planning and disability inclusion activities.
        """
    )

    st.markdown("---")

    st.subheader(
        "🎯 Project Objectives"
    )

    objectives = [
        "Understand disability population patterns",
        "Explore geographic distribution",
        "Examine demographic characteristics",
        "Identify documented registration/inclusion gaps",
        "Support community outreach planning",
        "Support evidence-informed programme planning"
    ]

    for objective in objectives:

        st.write(
            f"✓ {objective}"
        )

    st.markdown("---")

    st.subheader(
        "🔐 Responsible AI"
    )

    st.warning(
        """
        The system is a decision-support tool.

        Predictions should not independently determine whether
        an individual receives services, benefits, registration
        or other forms of support.

        Human review and local records remain important.

        The model does not use names, IDs, telephone numbers,
        NCPWD certificate numbers or PWD SACCO numbers as
        prediction features.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>IGOJI PWD Inclusion Intelligence</b>

    <br>

    Predicting Gaps in Disability Programme Registration
    and Inclusion in Igoji Sub-County

    <br><br>

    Developed with Python • Pandas • Scikit-learn •
    XGBoost • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)