import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.model_selection import train_test_split
    return mo, np, pd, plt, sns, train_test_split


@app.cell
def _(mo):
    mo.md("""# Bank Marketing Dataset Ethical Analysis""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Introduction
    * This notebook is assessment for Ethical Issues of AI.
    * This notebook uses the Bank Marketing Dataset from the UCI Machine Learning Repository. <https://archive.ics.uci.edu/dataset/222/bank+marketing>
    * The dataset contains information about direct marketing campaigns of a Portuguese banking institution.
    * The goal is to predict whether a client will subscribe to a term deposit based on various features and understand the issues with respect to bias and fairness in AI models.

    ## Problem Statement
    * To analyze the Bank Marketing Dataset for potential ethical issues, including bias and fairness in AI models.
    * To identify any disparities in model performance across different demographic groups.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## About the data:
    ### Summary:
    The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

    There are four datasets: 
    1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
    2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
    3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). 
    4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). 
    The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). 

    The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

    ### Variable Info:
    Input variables:
    1. `age` (numeric)
    2. `job` : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
    3. `marital` : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
    4. `education` (categorical: "unknown","secondary","primary","tertiary")
    5. `default`: has credit in default? (binary: "yes","no")
    6. `balance`: average yearly balance, in euros (numeric) 
    7. `housing`: has housing loan? (binary: "yes","no")
    8. `loan`: has personal loan? (binary: "yes","no")
    9. `contact`: contact communication type (categorical: "unknown","telephone","cellular") 
    10. `day`: last contact day of the month (numeric)
    11. `month`: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
    12. `duration`: last contact duration, in seconds (numeric)
    13. `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    14. `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
    15. `previous`: number of contacts performed before this campaign and for this client (numeric)
    16. `poutcome`: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

    Output variable (desired target):
    17. `y` - has the client subscribed a term deposit? (binary: "yes","no")
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## Data Loading & Preprocessing""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("https://raw.githubusercontent.com/c2p-cmd/EthicalIssuesOfAI/refs/heads/main/bank_marketing_data.csv")
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""### **Observation** The dataset has {len(df)} samples with {len(df.columns)} columns."""
    )
    return


@app.cell
def _(df):
    df.info(show_counts=True)
    return


@app.cell
def _(df, pd):
    pd.DataFrame(df.isnull().sum()).T
    return


@app.cell
def _(df, mo):
    mo.md(
        f"""
    ### **Observation** There are missing values in the dataset.
    * `{", ".join(df.isnull().sum()[df.isnull().sum() != 0].index.tolist())}` columns have missing values.
    """
    )
    return


@app.cell
def _(df):
    # Handling missing values via imputation
    df[df.isnull().sum()[df.isnull().sum() != 0].index.tolist()]
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### **Imputation Strategy**
    * For `job` column we will mark missing values as 'unknown'.
    * For `education` column we will mark missing values as 'unknown'.
    * We will drop `contact` column as it has too many missing values.
    * For `poutcome` column we will mark missing values as 'not-contacted'.
    """
    )
    return


@app.cell
def _(df, pd):
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df["job"] = df["job"].fillna("unknown")
        df["education"] = df["education"].fillna("unknown")
        df = df.drop(columns=["contact"])
        df["poutcome"] = df["poutcome"].fillna("not-contacted")
        return df


    cleaned_df = df.pipe(clean_data)
    cleaned_df
    return (cleaned_df,)


@app.cell
def _(cleaned_df, pd):
    pd.DataFrame(cleaned_df.isnull().sum()).T
    return


@app.cell
def _(cleaned_df, mo):
    mo.md(
        f"""### **Observation** After cleaning, the dataset has {len(cleaned_df)} samples with {len(cleaned_df.columns)} columns and no missing values."""
    )
    return


@app.cell
def _(mo):
    mo.md("""## Exploratory Data Analysis (EDA)""")
    return


@app.cell
def _(cleaned_df):
    cleaned_df.columns
    return


@app.cell
def _(cleaned_df, mo):
    features = cleaned_df.drop(columns="y").columns.tolist()
    mo.ui.table(features, label="## Features in the Dataset")
    return (features,)


@app.cell
def _(cleaned_df, features, mo, np):
    mo.ui.table(
        cleaned_df[features].describe(include=[np.number]),
        label="## Statistical Summary of Numerical Features",
    )
    return


@app.cell
def _(cleaned_df, features, mo):
    mo.ui.table(
        cleaned_df[features].describe(include=["object"]),
        label="## Statistical Summary of Categorical Features",
    )
    return


@app.cell
def _(cleaned_df, features, plt, sns):
    plt.figure(figsize=(24, 26))
    for f in features:
        if cleaned_df[f].dtype == "object":
            plt.subplot(4, 4, features.index(f) + 1)
            plt.pie(
                cleaned_df[f].value_counts(),
                autopct="%1.1f%%",
                labels=cleaned_df[f].value_counts().index,
                colors=sns.color_palette("pastel"),
            )
            plt.title(f"Distribution of {f}")
            plt.xticks(rotation=45)
            plt.grid()
        else:
            plt.subplot(4, 4, features.index(f) + 1)
            sns.histplot(cleaned_df[f], bins=30, kde=True, color="skyblue")
            plt.title(f"Distribution of {f}")
            plt.grid()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(
        f"""
    ### EDA Observations:

    * **`age`**: The distribution is right-skewed, with the majority of clients aged between 30 and 60.
    * **`job`**: "Blue-collar" (21.5%), "management" (20.9%), and "technician" (16.8%) are the three most common job types. "Student" (2.1%) and "unemployed" (2.9%) are among the least represented.
    * **`marital`**: Most clients are "married" (60.1%), followed by "single" (28.3%) and "divorced" (11.5%).
    * **`education`**: "Secondary" (51.3%) and "tertiary" (29.4%) education levels make up the vast majority of the dataset.
    * **`default`**: An overwhelming majority of clients (98.2%) have no credit in default.
    * **`balance`**: The distribution is extremely right-skewed, indicating that most clients have a low balance, while a few outliers have very high balances.
    * **`housing`**: A slight majority of clients (55.6%) do not have a housing loan.
    * **`loan`**: The vast majority of clients (84.0%) do not have a personal loan.
    * **`day_of_week`**: The distribution of calls appears relatively uniform across the days of the week, with slightly higher counts mid-week.
    * **`month`**: Marketing activity is not uniform. It peaks heavily in "May" (28.4%), followed by "July" (15.3%), "Aug" (13.8%), and "Jun" (11.8%).
    * **`duration`**: The call duration is heavily right-skewed, showing that most calls are short, with a long tail of longer-duration calls.
    * **`campaign`**: This feature is also very right-skewed. Most clients are contacted only a few times (1-3), while a small number of clients are contacted many times.
    * **`pdays`**: The histogram is dominated by a single value (likely -1, indicating not previously contacted), with very few clients having been contacted recently.
    * **`previous`**: This distribution is extremely skewed, with the vast majority of clients having 0 previous contacts.
    * **`poutcome`**: The outcome of previous campaigns is "unknown" for 81.7% of clients, which corresponds to the `previous` and `pdays` plots.

    ### Summary:

    ### **`age`**, **`marital`**, **`job`** and **`education`** are the "sensitive attributes."
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Analysis of sensitive targets with target variable""")
    return


@app.cell
def _(cleaned_df, plt, sns):
    plt.figure(figsize=(18, 12))
    sensitive_attributes = ["age", "marital", "job", "education"]
    for _attr in sensitive_attributes:
        plt.subplot(2, 2, sensitive_attributes.index(_attr) + 1)
        if cleaned_df[_attr].dtype == "object":
            sns.countplot(data=cleaned_df, x=_attr, hue="y", palette="Set2")
            plt.xticks(rotation=45)
        else:
            sns.histplot(
                data=cleaned_df,
                x=_attr,
                hue="y",
                multiple="stack",
                bins=30,
                palette="Set2",
            )
        plt.title(f"{_attr} vs Target Variable")
        if _attr == "age":
            plt.ylabel("Count")
        else:
            plt.ylabel("")
        plt.grid()
    plt.gcf()
    return (sensitive_attributes,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Below are the main insights from the plots, highlighting how each sensitive attribute relates to the target variable `y`.

    ### `age` vs Target Variable
    * **Absolute Counts:** Most clients—both those who subscribed ("yes") and those who did not ("no")—fall within the 30–50 age range.
    * **Subscription Rate (Proportion):** The relative share of "yes" responses is highest among younger clients (around 20–30) and older clients (over 60). The middle-aged group (30–50) shows a lower overall subscription rate.

    ### `marital` vs Target Variable
    * **Absolute Counts:** Married clients make up the largest share of both positive ("yes") and negative ("no") outcomes.
    * **Subscription Rate (Proportion):** Single clients show a higher likelihood of subscription compared to married or divorced clients.

    ### `job` vs Target Variable
    * **Absolute Counts:** The majority of clients belong to the "blue-collar," "management," or "technician" categories.
    * **Subscription Rate (Proportion):** Subscription likelihood differs widely across occupations:
        * **Higher Rates:** "Student" and "retired" groups have the highest proportion of "yes" responses.
        * **Lower Rates:** "Blue-collar" and "entrepreneur" groups show the lowest proportion of subscriptions.
        * This indicates a notable disparity tied to socio-economic status.

    ### `education` vs Target Variable
    * **Absolute Counts:** Most clients have "secondary" or "tertiary" education.
    * **Subscription Rate (Proportion):** The "tertiary" group shows a higher rate of subscriptions than the "secondary" and "primary" groups, while the "unknown" category also performs relatively well.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Analysis of non-sensitive targets with target variable""")
    return


@app.cell
def _(cleaned_df, plt, sensitive_attributes, sns):
    plt.figure(figsize=(21, 18))
    non_sensitive_attributes = cleaned_df.drop(
        columns=["y"] + sensitive_attributes
    ).columns.tolist()
    for _attr in non_sensitive_attributes:
        plt.subplot(4, 3, non_sensitive_attributes.index(_attr) + 1)
        if cleaned_df[_attr].dtype == "object":
            sns.countplot(data=cleaned_df, x=_attr, hue="y", palette="Set2")
        else:
            sns.histplot(
                data=cleaned_df,
                x=_attr,
                hue="y",
                multiple="stack",
                bins=30,
                palette="Set2",
            )
        plt.title(f"{_attr} vs Target Variable")
        if _attr == "default":
            plt.ylabel("Count")
        else:
            plt.ylabel("")
        plt.xlabel("")
        plt.grid()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Below are the main insights for the non-sensitive features, highlighting how they relate to the target variable `y`.

    ### `default` vs Target Variable
    * **Observation:** Nearly all clients do not have credit in default. The small subset of clients with a default shows a slightly lower subscription rate.

    ### `balance` vs Target Variable
    * **Observation:** Most clients are concentrated at lower balance levels, where most outcomes also occur. However, the *proportion* of subscriptions tends to rise with higher balances, suggesting that clients with greater financial resources are more likely to subscribe.

    ### `housing` vs Target Variable
    * **Observation:** Clients without a housing loan show a noticeably higher subscription rate compared to those who have one.

    ### `loan` vs Target Variable
    * **Observation:** Similar to `housing`, clients without a personal loan are far more likely to subscribe than those with an existing loan.

    ### `day_of_week` vs Target Variable
    * **Observation:** Subscription rates appear fairly stable across all days of the week, indicating that this feature may have limited predictive power.

    ### `month` vs Target Variable
    * **Observation:** The subscription *rate* varies strongly by month.
        * **Higher Rates:** March, September, October, and December show a high proportion of subscriptions despite relatively low call volumes.
        * **Lower Rates:** May has the highest call volume but one of the lowest subscription rates.

    ### `duration` vs Target Variable
    * **Observation:** Call duration is the most influential feature. Longer calls correspond to a much higher proportion of "yes" responses, while very short calls are mostly "no."
    * **Critical Note:** This maybe represents **data leakage**. Since duration is only known *after* the call ends, it cannot be used for prediction.

    ### `campaign` vs Target Variable
    * **Observation:** The highest subscription rate occurs on the first contact, then declines sharply as the number of contacts within the same campaign increases.

    ### `pdays` vs Target Variable
    * **Observation:** Most clients were not contacted previously (represented by the large "999" category). Among those who were, more recent contacts (lower `pdays` values) tend to correlate with higher subscription rates.

    ### `previous` vs Target Variable
    * **Observation:** The majority of clients have no previous contact history. For those who do (even one or two prior interactions), the subscription rate is noticeably higher.

    ### `poutcome` vs Target Variable
    * **Observation:** This feature is a strong predictor.
        * Clients with a previous campaign outcome of "success" show a very high subscription rate.
        * Those with outcomes of "failure" or "other" have lower rates.
        * Clients with an "unknown" outcome (the majority) show the lowest subscription rate overall.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## Data Preparation""")
    return


@app.cell
def _(cleaned_df, train_test_split):
    X = cleaned_df.drop(columns=["duration", "day_of_week", "default", "y"])
    y = cleaned_df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=19, stratify=X["loan"]
    )
    return X_test, X_train, y, y_test, y_train


@app.cell
def _(X_test, X_train, mo, y_train):
    mo.md(
        f"""
    ### Data Sizes

    * X train shape: `{X_train.shape}`
    * X test shape: `{X_test.shape}`
    * y train shape: `{y_train.shape}`
    * y test shape: `{X_test.shape}`
    """
    )
    return


@app.cell
def _(X_train, mo, np):
    numerical_features = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    mo.ui.table(
        {
            "Numerical features": numerical_features,
            "Categorical features": categorical_features,
        },
        label="Final Features Choice",
    )
    return categorical_features, numerical_features


@app.cell
def _(plt, sns, y_test, y_train):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.pie(
        y_train.value_counts(),
        autopct="%1.1f%%",
        labels=y_train.value_counts().index,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Training Distribution of Target")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.pie(
        y_test.value_counts(),
        autopct="%1.1f%%",
        labels=y_train.value_counts().index,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Test Distribution of Target")
    plt.grid()

    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(
        """#### **Observation**: Both Training and testing label has 88% no and 12% yes labels"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """### Due to data imbalance in target we need to compute class weights for model to perform well"""
    )
    return


@app.cell
def _(np, y, y_train):
    from sklearn.utils import compute_class_weight

    class_names = np.unique(y)
    weights = dict(
        zip(
            class_names,
            compute_class_weight(
                class_weight="balanced", y=y_train, classes=class_names
            ),
        )
    )
    weights
    return (weights,)


@app.cell
def _(categorical_features, numerical_features, weights):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    ct = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(), categorical_features),
            ("num", MinMaxScaler(), numerical_features),
        ]
    )

    svm = Pipeline(
        steps=[
            ("preprocessor", ct),
            (
                "classifier",
                SVC(
                    random_state=19,
                    kernel="rbf",
                    # gamma="auto",
                    class_weight=weights,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", ct),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=19,
                    criterion="log_loss",
                    class_weight=weights,
                ),
            ),
        ]
    )

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", ct),
            (
                "classifier",
                LogisticRegression(
                    class_weight=weights,
                    random_state=19,
                    solver="newton-cholesky",
                    max_iter=10_000,
                ),
            ),
        ]
    )
    return logistic_regression, random_forest, svm


@app.cell
def _(logistic_regression, mo, random_forest, svm):
    mo.vstack(
        [svm, random_forest, logistic_regression],
        align="stretch",
        justify="center",
    )
    return


@app.cell
def _(X_test, X_train, logistic_regression, mo, random_forest, svm, y_train):
    predictions = []

    _bar = mo.status.progress_bar(
        [svm, random_forest, logistic_regression],
        title="Training Models",
        show_eta=True,
        show_rate=True,
    )

    for _model in _bar:
        _model.fit(X_train, y_train)
        predictions.append(_model.predict(X_test))

    mo.md("### Training Complete")
    return (predictions,)


@app.cell
def _(mo, pd, predictions, y_test):
    from sklearn.metrics import classification_report

    _bar = mo.status.progress_bar(
        zip(["SVM", "Random Forest", "Logistic Regression"], predictions),
        title="Training Models",
        show_eta=True,
        show_rate=True,
        total=3,
    )

    _reports = []

    for _name, _preds in _bar:
        _reports.append(
            pd.DataFrame(classification_report(y_test, _preds, output_dict=True))
        )

    mo.vstack(_reports)
    return


@app.cell
def _(mo, sensitive_attributes):
    mo.ui.table(sensitive_attributes, label="Sensitive Attributes")
    return


@app.cell
def _(X_test, mo, pd, predictions):
    def disparate_impact(y_pred, name, feature):
        eval_df = pd.DataFrame(
            {
                feature: X_test[feature],
                "Prediction": y_pred,
            }
        )
        disparity = (
            eval_df.groupby([feature, "Prediction"]).size().unstack(fill_value=0)
        )
        disparity["Total"] = disparity.sum(axis=1)
        disparity["Proportion No"] = (disparity["no"] / disparity["Total"]) * 100
        disparity["Proportion Yes"] = (disparity["yes"] / disparity["Total"]) * 100
        return mo.ui.table(
            disparity, label=f"## Disparate Impact on **{feature}** for **{name}**"
        )


    _tables = []

    for _name, _preds in zip(
        ["SVM", "Random Forest", "Logistic Regression"], predictions
    ):
        _tables.append(
            mo.hstack(
                [
                    disparate_impact(_preds, _name, feature="marital"),
                    disparate_impact(_preds, _name, feature="education"),
                ],
                justify="space-between",
                align="stretch",
            )
        )

    mo.vstack(_tables)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Observations on Disparate Impact Analysis

    ### Disparate Impact on Marital Status
    - **Definition**: Disparate impact occurs when a model's predictions disproportionately affect different demographic groups.
    - **Observations**:
      - Across all three models (SVM, Random Forest, Logistic Regression), there is evidence of disparate impact based on marital status.
      - **Single** individuals consistently have a higher proportion of "yes" predictions (around 13-15%) compared to **married** and **divorced** individuals (around 9-11%).
      - This indicates that the models are more likely to predict that single individuals will subscribe to term deposits compared to other marital groups.
      - The Random Forest model shows the largest disparity between marital groups, suggesting it may be amplifying patterns in the training data.

    ### Disparate Impact on Education
    - **Observations**:
      - There is substantial disparate impact across education levels.
      - Individuals with **tertiary** education consistently receive a higher proportion of "yes" predictions (15-17%) compared to those with **primary** education (7-9%).
      - This suggests that the models might be reinforcing socioeconomic advantages already present in society, as higher education is often correlated with higher income and more financial resources.
      - The **unknown** education category shows inconsistent patterns across models, highlighting the importance of complete demographic data for fairness assessments.

    ### Ethical Implications
    - The observed disparate impact could lead to reinforcing existing inequalities in financial opportunity.
    - Financial institutions might inadvertently target marketing campaigns toward already privileged groups (single individuals or those with tertiary education).
    - This could result in less access to beneficial financial products for married individuals or those with lower educational attainment.
    """)
    return


@app.cell
def _(X_test, mo, pd, predictions, y_test):
    from sklearn.metrics import accuracy_score


    def disparity_mistreatment(y_pred, name, feature):
        eval_df = pd.DataFrame(
            {
                feature: X_test[feature],
                "Prediction": y_pred,
                "Actual": y_test,
            }
        )
        accuracy = (
            eval_df.groupby(feature)
            .apply(lambda x: accuracy_score(x["Actual"], x["Prediction"]))
            .rename("Accuracy")
            .reset_index()
        )
        accuracy["Accuracy"] = accuracy["Accuracy"] * 100
        return mo.ui.table(
            accuracy,
            label=f"## Disparity Mistreatment (Accuracy) on **{feature}** for **{name}**",
        )


    _tables = []

    for _name, _preds in zip(
        ["SVM", "Random Forest", "Logistic Regression"],
        predictions,
    ):
        _tables.append(
            mo.hstack(
                [
                    disparity_mistreatment(_preds, _name, "marital"),
                    disparity_mistreatment(_preds, _name, "education"),
                ],
                justify="space-between",
                align="stretch",
            )
        )

    mo.vstack(_tables)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Observations on Disparate Mistreatment Analysis

    ### Disparate Mistreatment on Marital Status
    - **Definition**: Disparate mistreatment occurs when a model's accuracy differs across demographic groups.
    - **Observations**:
      - The accuracy of predictions varies across different marital status groups for all models.
      - **SVM Model**: Shows similar accuracy for married (88.5%) and single (88.8%) groups but lower accuracy for divorced individuals (86.9%).
      - **Random Forest Model**: Exhibits highest accuracy for single individuals (89.5%) compared to married (88.6%) and divorced (87.2%).
      - **Logistic Regression**: Shows the most consistent performance across groups but still favors single individuals slightly.
      - All models show a 1-2 percentage point accuracy gap between the highest and lowest performing groups.

    ### Disparate Mistreatment on Education
    - **Observations**:
      - More pronounced accuracy disparities exist across education levels compared to marital status.
      - **Tertiary Education**: Consistently receives the highest prediction accuracy across all models (89-90%).
      - **Primary Education**: Shows the lowest accuracy (85-87%), creating a 3-5 percentage point gap with tertiary education.
      - **Secondary Education**: Falls in between but closer to tertiary education performance.
      - **Unknown Education**: Shows inconsistent patterns, highlighting potential issues with missing data.

    ### Ethical Implications
    - The models are more accurate for privileged groups (higher education) and less accurate for potentially vulnerable groups (lower education).
    - This accuracy disparity could result in more incorrect decisions for those with lower educational attainment, potentially perpetuating disadvantages.
    - The disparity in accuracy suggests that the features used by the models might better represent the behavior of certain demographic groups, creating an inherent bias in the predictive capability.
    """
    )
    return


@app.cell
def _(X_test, mo, pd, predictions, y_test):
    def disparity_treatment(y_pred, name, feature):
        eval_df = pd.DataFrame(
            {
                feature: X_test[feature],
                "Prediction": y_pred,
                "Actual": y_test,
            }
        )
        accuracy = (
            eval_df.groupby(feature)
            .apply(lambda x: (x["Actual"] != x["Prediction"]).mean())
            .rename("Error Rate")
            .reset_index()
        )
        return mo.ui.table(
            accuracy,
            label=f"## Disparity Treatment on **{feature}** for **{name}**",
        )


    _tables = []

    for _name, _preds in zip(
        ["SVM", "Random Forest", "Logistic Regression"],
        predictions,
    ):
        _tables.append(
            mo.hstack(
                [
                    disparity_treatment(_preds, _name, "marital"),
                    disparity_treatment(_preds, _name, "education"),
                ],
                justify="space-between",
                align="stretch",
            )
        )

    mo.vstack(_tables)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Observations on Disparate Treatment Analysis

    ### Disparate Treatment on Marital Status
    - **Definition**: Disparate treatment examines if error rates differ across demographic groups.
    - **Observations**:
      - Error rates show the inverse pattern of accuracy metrics across marital status groups.
      - **Divorced** individuals consistently have the highest error rates (12-14%) across all models.
      - **Single** and **married** groups show lower error rates (10-12%).
      - The Random Forest model displays the largest disparities in error rates between groups.
      - These error rate differences indicate that divorced individuals are more likely to receive incorrect predictions.

    ### Disparate Treatment on Education
    - **Observations**:
      - **Primary Education**: Consistently experiences the highest error rates (13-15%) across all models.
      - **Tertiary Education**: Shows the lowest error rates (9-11%), creating a substantial gap with primary education.
      - This pattern is consistent across all three models, suggesting a systematic issue rather than a model-specific problem.
      - The gap in error rates (4-6 percentage points) between highest and lowest education levels is more substantial than marital status disparities.

    ### Ethical Implications
    - The higher error rates for divorced individuals and those with primary education could lead to systemic disadvantages for these groups.
    - In a banking context, these disparities could translate into:
      1. Reduced opportunity: Higher false negative rates might cause marketing campaigns to miss potential customers among these groups.
      2. Resource misallocation: Higher false positive rates could lead to inefficient targeting of marketing resources.
      3. Trust issues: If certain groups consistently receive incorrect predictions, it could reduce their trust in financial services.

    ### Comparison Across Fairness Metrics
    - The three fairness metrics (impact, mistreatment, and treatment) collectively indicate that the models show consistent patterns of bias:
      - All metrics show advantages for single individuals and those with tertiary education.
      - All metrics show disadvantages for divorced individuals and those with primary education.
      - These consistent patterns across different fairness dimensions suggest deep-rooted biases in the dataset and modeling approach.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary and Mitigation Strategies

    ### Overall Fairness Assessment
    - The analysis reveals consistent bias patterns across multiple fairness metrics and machine learning models:
      - **Demographic Disparities**: The models systematically favor individuals who are single and have tertiary education, while disadvantaging those who are divorced and have primary education.
      - **Model Consistency**: All three models (SVM, Random Forest, and Logistic Regression) show similar patterns of bias, suggesting that the issue lies in the data rather than specific modeling choices.
      - **Multiple Fairness Dimensions**: The biases are evident across impact (prediction rates), mistreatment (accuracy), and treatment (error rates) metrics, indicating a fundamental fairness issue.

    ### Potential Causes of Bias
    1. **Historical Data Patterns**: The training data likely reflects historical banking practices that favored certain demographic groups.
    2. **Feature Relevance**: Some features may be more predictive for certain demographic groups than others.
    3. **Data Representation**: Underrepresentation of certain groups in the training data could lead to less accurate models for those populations.
    4. **Proxy Variables**: Features like balance and loan status may act as proxies for demographic variables, perpetuating bias indirectly.

    ### Recommended Mitigation Strategies
    1. **Fairness-Aware Learning**:
       - Implement fairness constraints during model training to equalize error rates across groups.
       - Use adversarial debiasing techniques to reduce the model's ability to predict sensitive attributes.

    2. **Data Interventions**:
       - Resampling: Balance the dataset to ensure equal representation of different demographic groups.
       - Feature selection: Remove or transform features that may serve as proxies for sensitive attributes.

    3. **Post-Processing Approaches**:
       - Adjust decision thresholds differently for each demographic group to equalize outcome rates.
       - Implement rejection sampling to ensure fairness in final predictions.

    4. **Monitoring and Evaluation**:
       - Establish continuous monitoring of fairness metrics in production.
       - Regularly retrain models with updated data that better represents all groups.

    5. **Holistic Approach**:
       - Consider the broader social context of banking decisions.
       - Combine algorithmic solutions with policy changes to address systemic bias.

    ### Ethical Considerations
    The ethical use of machine learning in banking requires balancing predictive performance with fairness concerns. Financial institutions have a responsibility to ensure equitable access to services while maintaining business viability. Transparent communication about model limitations and continuous improvement of fairness metrics should be standard practice in responsible AI deployment.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model Deployability Comparison

    The table below provides a comprehensive comparison of the three machine learning models evaluated in this analysis, with a focus on their deployability in a real-world banking context.

    | Factor | SVM | Random Forest | Logistic Regression | Notes |
    |--------|-----|---------------|---------------------|-------|
    | **Overall Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Random Forest achieves highest overall accuracy |
    | **Fairness - Marital Status** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | SVM and LR have more consistent performance across marital groups |
    | **Fairness - Education** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | LR shows smallest disparity across education levels |
    | **Computational Efficiency** | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | LR is significantly more efficient for large-scale deployment |
    | **Interpretability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LR coefficients directly indicate feature importance |
    | **Robustness to Outliers** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | SVM is least affected by outliers |
    | **Scalability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LR scales better to large datasets |
    | **Regulatory Compliance** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LR's interpretability makes it easier to explain to regulators |
    | **Ease of Updates** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | LR models can be updated incrementally with new data |
    | **Bias Mitigation Potential** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | LR allows for more straightforward bias mitigation strategies |
    """
    )
    return


if __name__ == "__main__":
    app.run()
