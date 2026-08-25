import streamlit as st
import pandas as pd
import joblib


# ------- LOADING MODELS ------

math_model = joblib.load(
    "models/math_score_model.pkl"
)

reading_model = joblib.load(
    "models/reading_score_model.pkl"
)

writing_model = joblib.load(
    "models/writing_score_model.pkl"
)


# ------FUNCTIONS------

def get_performance_level(score):

    if score >= 80:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 60:
        return "Average"

    elif score >= 50:
        return "Below Average"

    else:
        return "Poor"


def get_profile(
    math,
    reading,
    writing,
    average
):

    gap = (
        max(
            math,
            reading,
            writing
        )
        -
        min(
            math,
            reading,
            writing
        )
    )


    if average < 50:
        return "At-Risk Performer"


    if (
        math >= 70
        and reading >= 70
        and writing >= 70
    ):
        return "Balanced High Performer"


    if (
        math - reading >= 10
        and math - writing >= 10
    ):
        return "Math-Oriented"


    if (
        reading - math >= 10
        and writing - math >= 10
    ):
        return "Language-Oriented"


    if gap >= 20:
        return "Inconsistent Performer"


    return "Developing Performer"


# ------ PAGE CONFIGURATION-------

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)


# -----TITLE-----

st.title(
    "Student Academic Performance Prediction"
)

st.write(
    """
    Predict Mathematics, Reading and Writing scores
    and calculate the student's predicted overall
    performance.
    """
)


# ------- STUDENT INFORMATION---------

st.header(
    "Student Information"
)


col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        [
            "female",
            "male"
        ]
    )


    race = st.selectbox(
        "Race/Ethnicity",
        [
            "group A",
            "group B",
            "group C",
            "group D",
            "group E"
        ]
    )


    parental_education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]
    )


with col2:

    lunch = st.selectbox(
        "Lunch Type",
        [
            "standard",
            "free/reduced"
        ]
    )


    preparation = st.selectbox(
        "Test Preparation",
        [
            "none",
            "completed"
        ]
    )


# --------PREDICT--------

if st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
):


    # -----------CREATE STUDENT DATA-------

    student = pd.DataFrame({

        "gender": [
            gender
        ],

        "race/ethnicity": [
            race
        ],

        "parental level of education": [
            parental_education
        ],

        "lunch": [
            lunch
        ],

        "test preparation course": [
            preparation
        ]

    })


    # --------PREDICT INDIVIDUAL SUBJECT SCORES--------

    predicted_math = math_model.predict(
        student
    )[0]


    predicted_reading = reading_model.predict(
        student
    )[0]


    predicted_writing = writing_model.predict(
        student
    )[0]


    # ---------KEEP SCORES BETWEEN 0 AND 100-------

    predicted_math = max(
        0,
        min(
            100,
            predicted_math
        )
    )


    predicted_reading = max(
        0,
        min(
            100,
            predicted_reading
        )
    )


    predicted_writing = max(
        0,
        min(
            100,
            predicted_writing
        )
    )


    # ---------- CALCULATE PREDICTED AVERAGE--------

    predicted_average = (

        predicted_math

        + predicted_reading

        + predicted_writing

    ) / 3


    # ----------- PERFORMANCE LEVEL ----------

    level = get_performance_level(
        predicted_average
    )


    # ---------- PERFORMANCE PROFILE ----------

    profile = get_profile(

        predicted_math,

        predicted_reading,

        predicted_writing,

        predicted_average
    )


    # ---------- RESULTS ---------

    st.header(
        "📊 Predicted Performance"
    )


    # -------- SUBJECT SCORES -------

    st.subheader(
        "Predicted Subject Scores"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📐 Mathematics",
            f"{predicted_math:.2f} / 100"
        )


    with col2:

        st.metric(
            "📖 Reading",
            f"{predicted_reading:.2f} / 100"
        )


    with col3:

        st.metric(
            "✍️ Writing",
            f"{predicted_writing:.2f} / 100"
        )


    # ---------- OVERALL RESULT -----------

    st.subheader(
        "Overall Performance"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📊 Predicted Average",
            f"{predicted_average:.2f} / 100"
        )


    with col2:

        st.metric(
            "🏆 Performance Level",
            level
        )


    with col3:

        st.metric(
            "📌 Performance Profile",
            profile
        )


    # --------- SCORE SUMMARY ----------

    st.subheader(
        "Prediction Summary"
    )


    result_df = pd.DataFrame({

        "Subject": [
            "Mathematics",
            "Reading",
            "Writing",
            "Overall Average"
        ],

        "Predicted Score": [
            predicted_math,
            predicted_reading,
            predicted_writing,
            predicted_average
        ]

    })


    result_df["Predicted Score"] = (
        result_df["Predicted Score"]
        .round(2)
    )


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


    # --------- IMPORTANT NOTE ---------

    st.warning(
        """
        These scores are predictions generated by
        machine learning models based on patterns
        learned from the dataset. They are estimates
        and should not be considered guaranteed actual
        future scores.
        """
    )


# --------- INFORMATION ---------

st.divider()


st.subheader(
    "How is performance calculated?"
)


st.write(
    """
    The system predicts Mathematics, Reading and
    Writing scores separately. The predicted overall
    score is then calculated using the average of
    these three predicted scores.
    """
)


st.latex(
    r"""
    Predicted\ Average =
    \frac{
    Predicted\ Math +
    Predicted\ Reading +
    Predicted\ Writing
    }{3}
    """
)


st.subheader(
    "Performance Levels"
)


st.write(
    """
    • 80–100 → Excellent

    • 70–79 → Good

    • 60–69 → Average

    • 50–59 → Below Average

    • Below 50 → Poor
    """
)