# Bank Marketing Dataset - Ethical AI Analysis

An analysis of the Bank Marketing Dataset from UCI Machine Learning Repository, focusing on identifying and understanding bias and fairness issues in AI/ML models.

## Demo Link: <https://c2p-cmd.github.io/EthicalIssuesOfAI/>

## 📋 Project Overview

This project examines the ethical implications of using machine learning models on a Portuguese bank's direct marketing campaign data. The analysis identifies potential biases across demographic groups and evaluates model fairness in predicting term deposit subscriptions.

## 🎯 Objectives

- Analyze the Bank Marketing Dataset for potential ethical issues
- Identify bias and fairness concerns in AI models
- Evaluate disparities in model performance across demographic groups
- Examine sensitive attributes: age, marital status, job, and education

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Description:** Data from direct marketing campaigns (phone calls) of a Portuguese banking institution, with 41,188 samples and 17 features.

**Target Variable:** Whether a client subscribed to a term deposit (yes/no)

## 🗂️ Project Structure

```bash
├── notebook.ipynb           # Main analysis notebook
├── data_convert.ipynb       # Data download and conversion script
├── bank_marketing_data.csv  # Dataset (generated from UCI repo)
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository
2. Install dependencies:
3. Run marimo notebook or jupyter lab

```bash
uv venv --python 3.11
uv pip install -r requirements.txt

# option 1
uv run marimo edit notebook.py

# option 2
jupyter lab
```

### Usage

1. **Data Preparation** (optional - dataset already included):
   - Run `data_convert.ipynb` to fetch fresh data from UCI repository

2. **Main Analysis**:
   - Open and run `notebook.py` or `notebook.py` to see the complete analysis

## 📈 Analysis Highlights

- **Data Preprocessing:** Handling missing values with appropriate imputation strategies
- **Exploratory Data Analysis:** Statistical summaries and visualizations of all features
- **Bias Detection:** Analysis of sensitive attributes (age, job, marital status, education)
- **Model Evaluation:** Performance comparison across demographic groups
- **Fairness Metrics:** Examination of disparities in model predictions

## 🔍 Key Findings

The analysis reveals:

- Subscription rates vary significantly by occupation (students/retired vs. blue-collar workers)
- Age groups show different propensities for term deposit subscriptions
- Education level correlates with subscription likelihood
- Potential socio-economic bias in model predictions

## 🛠️ Technologies Used

- **Python 3.11**
- **pandas** - Data loading & manipulation
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning models
- **marimo** - Interactive analysis environment

## 📝 [License](./LICENSE.md)
