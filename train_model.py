# STUDENT PERFORMANCE PREDICTION SYSTEM

# -----IMPORTING LIBRARIES-----
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD DATA

DATA_PATH = "data/StudentsPerformance.csv"

df = pd.read_csv(DATA_PATH)

print("\n-------DATASET INFORMATION ---------")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())


# -----DATA CLEANING-----

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Score columns
score_columns = [
    "math score",
    "reading score",
    "writing score"
]

# Convert scores to numeric
for column in score_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove rows where scores are missing
df = df.dropna(
    subset=score_columns
)

print("\n-------- AFTER CLEANING --------")

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# ------ CALCULATE PERFORMANCE METRICS ------

# Overall average score
df["average_score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
) / 3


# Highest score
df["highest_score"] = df[
    score_columns
].max(axis=1)


# Lowest score
df["lowest_score"] = df[
    score_columns
].min(axis=1)


# Difference between highest and lowest
df["score_gap"] = (
    df["highest_score"]
    - df["lowest_score"]
)


# Standard deviation
df["score_std"] = df[
    score_columns
].std(axis=1)


# -----STRONGEST AND WEAKEST SUBJECT -------

subject_names = {

    "math score": "Mathematics",

    "reading score": "Reading",

    "writing score": "Writing"
}


df["strongest_subject"] = (
    df[score_columns]
    .idxmax(axis=1)
    .map(subject_names)
)


df["weakest_subject"] = (
    df[score_columns]
    .idxmin(axis=1)
    .map(subject_names)
)


#  ------ PERFORMANCE LEVEL -----

def performance_level(score):

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


df["performance_level"] = (
    df["average_score"]
    .apply(performance_level)
)


# ----- CONSISTENCY LEVEL -----

def consistency_level(std):

    if std <= 5:
        return "Highly Consistent"

    elif std <= 10:
        return "Consistent"

    elif std <= 15:
        return "Moderately Consistent"

    else:
        return "Inconsistent"


df["consistency"] = (
    df["score_std"]
    .apply(consistency_level)
)


# ----- AT-RISK IDENTIFICATION  -----
df["at_risk"] = np.where(

    df["average_score"] < 50,

    "At Risk",

    "Not At Risk"
)


# ------- PERFORMANCE PROFILE --------
def get_profile(row):

    math = row["math score"]

    reading = row["reading score"]

    writing = row["writing score"]

    average = row["average_score"]

    gap = row["score_gap"]


    # At-risk student
    if average < 50:

        return "At-Risk Performer"


    # Balanced high performer
    if (
        math >= 70
        and reading >= 70
        and writing >= 70
    ):

        return "Balanced High Performer"


    # Math oriented
    if (
        math - reading >= 10
        and math - writing >= 10
    ):

        return "Math-Oriented"


    # Language oriented
    if (
        reading - math >= 10
        and writing - math >= 10
    ):

        return "Language-Oriented"


    # Inconsistent
    if gap >= 20:

        return "Inconsistent Performer"


    # General
    return "Developing Performer"


df["performance_profile"] = (
    df.apply(
        get_profile,
        axis=1
    )
)


# ------- PERFORMANCE SUMMARY ------
print("\n========== PERFORMANCE SUMMARY ==========")

print(
    df[
        [
            "math score",
            "reading score",
            "writing score",
            "average_score",
            "score_gap",
            "score_std"
        ]
    ].describe()
)


print("\n----PERFORMANCE LEVELS -----")

print(
    df[
        "performance_level"
    ].value_counts()
)


print("\n----- PERFORMANCE PROFILES -----")

print(
    df[
        "performance_profile"
    ].value_counts()
)


print("\n----- STRONGEST SUBJECT -----")

print(
    df[
        "strongest_subject"
    ].value_counts()
)


print("\n---- WEAKEST SUBJECT -----")

print(
    df[
        "weakest_subject"
    ].value_counts()
)


# ---- CREATE CHART DIRECTORY ----
os.makedirs(
    "charts",
    exist_ok=True
)


# ----- OVERALL PERFORMANCE DISTRIBUTION -----

plt.figure(
    figsize=(9, 5)
)

sns.histplot(
    df["average_score"],
    bins=20,
    kde=True
)

plt.title(
    "Distribution of Overall Student Performance"
)

plt.xlabel(
    "Overall Score"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    "charts/overall_performance.png"
)

plt.show()

plt.close()


# =---- PERFORMANCE LEVEL DISTRIBUTION -----
order = [
    "Poor",
    "Below Average",
    "Average",
    "Good",
    "Excellent"
]


plt.figure(
    figsize=(9, 5)
)

sns.countplot(
    data=df,
    x="performance_level",
    order=order
)

plt.title(
    "Student Performance Levels"
)

plt.xlabel(
    "Performance Level"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    "charts/performance_levels.png"
)

plt.show()

plt.close()


# ----- GENDER VS PERFORMANCE ----
plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=df,
    x="gender",
    y="average_score"
)

plt.title(
    "Average Performance by Gender"
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Average Score"
)

plt.tight_layout()

plt.savefig(
    "charts/gender_performance.png"
)

plt.show()

plt.close()


# -----LUNCH VS PERFORMANCE -----
plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=df,
    x="lunch",
    y="average_score"
)

plt.title(
    "Average Performance by Lunch Type"
)

plt.xlabel(
    "Lunch Type"
)

plt.ylabel(
    "Average Score"
)

plt.tight_layout()

plt.savefig(
    "charts/lunch_performance.png"
)

plt.show()

plt.close()


# -----TEST PREPARATION VS PERFORMANCE -----
plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=df,
    x="test preparation course",
    y="average_score"
)

plt.title(
    "Average Performance by Test Preparation"
)

plt.xlabel(
    "Test Preparation"
)

plt.ylabel(
    "Average Score"
)

plt.tight_layout()

plt.savefig(
    "charts/test_preparation_performance.png"
)

plt.show()

plt.close()


# ----- PARENTAL EDUCATION VS PERFORMANCE -----
plt.figure(
    figsize=(11, 6)
)

sns.barplot(
    data=df,
    x="parental level of education",
    y="average_score"
)

plt.title(
    "Average Performance by Parental Education"
)

plt.xlabel(
    "Parental Level of Education"
)

plt.ylabel(
    "Average Score"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "charts/parental_education_performance.png"
)

plt.show()

plt.close()


# ----- RACE/ETHNICITY VS PERFORMANCE -----
plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=df,
    x="race/ethnicity",
    y="average_score"
)

plt.title(
    "Average Performance by Race/Ethnicity"
)

plt.xlabel(
    "Race/Ethnicity"
)

plt.ylabel(
    "Average Score"
)

plt.tight_layout()

plt.savefig(
    "charts/race_performance.png"
)

plt.show()

plt.close()


# ----- SUBJECT COMPARISON -----
subject_means = df[
    score_columns
].mean()


plt.figure(
    figsize=(8, 5)
)

subject_means.plot(
    kind="bar"
)

plt.title(
    "Average Score by Subject"
)

plt.xlabel(
    "Subject"
)

plt.ylabel(
    "Average Score"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "charts/subject_comparison.png"
)

plt.show()

plt.close()


# -----CORRELATION-----
correlation = df[
    score_columns + ["average_score"]
].corr()


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Score Correlation"
)

plt.tight_layout()

plt.savefig(
    "charts/correlation.png"
)

plt.show()

plt.close()


# ----- MACHINE LEARNING -----
# PREDICT MATH, READING AND WRITING SCORES

features = [

    "gender",

    "race/ethnicity",

    "parental level of education",

    "lunch",

    "test preparation course"
]


score_targets = [

    "math score",

    "reading score",

    "writing score"
]


X = df[
    features
]


# -----TRAIN / TEST SPLIT----
X_train, X_test, y_train, y_test = train_test_split(

    X,

    df[score_targets],

    test_size=0.20,

    random_state=42
)


print("\n========== TRAIN / TEST DATA ==========")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ----- ENCODING-----
preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            features
        )
    ]
)


# ------ 24. MACHINE LEARNING MODELS-----
models = {

    "Linear Regression":

        LinearRegression(),


    "Decision Tree":

        DecisionTreeRegressor(
            random_state=42
        ),


    "Random Forest":

        RandomForestRegressor(
            n_estimators=200,

            random_state=42
        )
}


# ----- TRAIN AND EVALUATE MODELS ----
best_models = {}

model_results = {}


for subject in score_targets:

    print("\n")
    print("============================================")
    print("PREDICTING:", subject)
    print("============================================")


    y_train_subject = y_train[
        subject
    ]


    y_test_subject = y_test[
        subject
    ]


    best_model = None

    best_model_name = None

    best_r2 = -float("inf")


    for name, regressor in models.items():

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",

                    preprocessor
                ),

                (
                    "regressor",

                    regressor
                )
            ]
        )


        # Train
        pipeline.fit(

            X_train,

            y_train_subject
        )


        # Predict
        predictions = pipeline.predict(

            X_test
        )


        # MAE
        mae = mean_absolute_error(

            y_test_subject,

            predictions
        )


        # RMSE
        rmse = np.sqrt(

            mean_squared_error(

                y_test_subject,

                predictions
            )
        )


        # R2
        r2 = r2_score(

            y_test_subject,

            predictions
        )


        # Save results
        model_results[

            subject + " - " + name

        ] = {

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2
        }


        print("\nModel:", name)

        print(
            "MAE:",
            round(mae, 2)
        )

        print(
            "RMSE:",
            round(rmse, 2)
        )

        print(
            "R²:",
            round(r2, 3)
        )


        # Select best model
        if r2 > best_r2:

            best_r2 = r2

            best_model = pipeline

            best_model_name = name


    # Save best model
    best_models[subject] = best_model


    print(
        "\nBest model:",
        best_model_name
    )

    print(
        "Best R²:",
        round(best_r2, 3)
    )


# ------MODEL COMPARISON-------
results_df = pd.DataFrame(
    model_results
).T


print(
    "\n========== MODEL COMPARISON =========="
)

print(
    results_df.round(3)
)


# ----- PREDICT THREE SUBJECT SCORES-----
predicted_math = best_models[
    "math score"
].predict(
    X_test
)


predicted_reading = best_models[
    "reading score"
].predict(
    X_test
)


predicted_writing = best_models[
    "writing score"
].predict(
    X_test
)


# ------ CALCULATE PREDICTED AVERAGE-----
predicted_average = (

    predicted_math

    + predicted_reading

    + predicted_writing

) / 3


# -----PREDICT PERFORMANCE LEVEL-----
def predicted_performance(score):

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


predicted_performance_level = [

    predicted_performance(score)

    for score in predicted_average
]


# ---- CREATE PREDICTION RESULT DATAFRAME------
prediction_results = X_test.copy()


prediction_results[
    "predicted_math_score"
] = predicted_math


prediction_results[
    "predicted_reading_score"
] = predicted_reading


prediction_results[
    "predicted_writing_score"
] = predicted_writing


prediction_results[
    "predicted_average_score"
] = predicted_average


prediction_results[
    "predicted_performance"
] = predicted_performance_level


# ------ DISPLAY PREDICTION RESULTS-----
print(
    "\n-------- PREDICTION RESULTS -------"
)


print(

    prediction_results[

        [

            "predicted_math_score",
            "predicted_reading_score",
            "predicted_writing_score",
            "predicted_average_score",
            "predicted_performance"

        ]

    ].head(10).round(2)

)


# -------PERFORMANCE DISTRIBUTION OF PREDICTIONS-----
print(
    "\n------ PREDICTED PERFORMANCE DISTRIBUTION -----"
)


print(

    prediction_results[
        "predicted_performance"
    ].value_counts()

)


# ----- SAVE MODELS----

os.makedirs(
    "models",
    exist_ok=True
)


for subject, model in best_models.items():

    filename = (

        subject
        .replace(" ", "_")
        .replace("/", "_")

        + "_model.pkl"

    )


    joblib.dump(

        model,

        os.path.join(
            "models",
            filename
        )
    )


print(
    "\nAll best models saved in the 'models' folder."
)


# ------SAVE PREDICTION RESULTS-----
os.makedirs(
    "data",
    exist_ok=True
)


prediction_results.to_csv(

    "data/predicted_students.csv",

    index=False
)


print(
    "Prediction results saved to:"
)

print(
    "data/predicted_students.csv"
)


# ----- SAVE ANALYZED DATA----
df.to_csv(

    "data/analyzed_students.csv",

    index=False
)


print(
    "Analyzed dataset saved to:"
)

print(
    "data/analyzed_students.csv"
)


# -----FUNCTION FOR NEW STUDENT PREDICTION-----
def predict_student(

    gender,

    race,

    parental_education,

    lunch,

    test_preparation

):

    # Create student dataframe

    student = pd.DataFrame({

        "gender": [gender],

        "race/ethnicity": [race],

        "parental level of education":
            [parental_education],

        "lunch": [lunch],

        "test preparation course":
            [test_preparation]

    })


    # Predict Math

    math_prediction = best_models[
        "math score"
    ].predict(
        student
    )[0]


    # Predict Reading

    reading_prediction = best_models[
        "reading score"
    ].predict(
        student
    )[0]


    # Predict Writing

    writing_prediction = best_models[
        "writing score"
    ].predict(
        student
    )[0]


    # Calculate average

    average_prediction = (

        math_prediction

        + reading_prediction

        + writing_prediction

    ) / 3


    # Performance

    performance = predicted_performance(

        average_prediction

    )


    # Display result

    print(
        "\n----------------"
    )

    print(
        "      STUDENT PERFORMANCE PREDICTION"
    )

    print(
        "-----------------"
    )


    print(

        "Predicted Math Score:",

        round(
            math_prediction,
            2
        )

    )


    print(

        "Predicted Reading Score:",

        round(
            reading_prediction,
            2
        )

    )


    print(

        "Predicted Writing Score:",

        round(
            writing_prediction,
            2
        )

    )


    print(

        "Predicted Average Score:",

        round(
            average_prediction,
            2
        )

    )


    print(

        "Predicted Performance:",

        performance

    )


    print(
        "-------------------"
    )


    return {

        "Math Score":
            round(
                math_prediction,
                2
            ),

        "Reading Score":
            round(
                reading_prediction,
                2
            ),

        "Writing Score":
            round(
                writing_prediction,
                2
            ),

        "Average Score":
            round(
                average_prediction,
                2
            ),

        "Performance":
            performance
    }


# ------- TEST THE PREDICTION SYSTEM----
result = predict_student(

    gender="female",

    race="group B",

    parental_education="bachelor's degree",

    lunch="standard",

    test_preparation="completed"

)


print("\nFinal Result:")

print(result)


# ------ PROJECT COMPLETED----
print(
    "\n----- PROJECT COMPLETED ------"
)