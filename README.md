# Travel Insurance Prediction

Predicting whether a customer will buy travel insurance, using an Artificial Neural Network (ANN-MLP) classifier. Built for INFO813 – Artificial Intelligence (Master of Information Technology).

## Overview

A tour and travel company wants to know, ahead of time, which customers are likely to buy a new travel insurance package (including COVID cover) so it can target outreach effectively. This project builds a binary classifier — will a customer buy insurance or not — from customer demographic and travel-history data.

## Dataset

- **Source:** [Travel Insurance Prediction Data](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data) (Kaggle)
- **Size:** 1,987 rows × 9 columns

| Attribute | Type | Description |
|---|---|---|
| Age | Numerical | Customer age |
| Employment Type | Categorical | Government Sector / Private Sector or Self Employed |
| GraduateOrNot | Categorical | Whether the customer graduated |
| AnnualIncome | Numerical | Yearly income (INR) |
| FamilyMembers | Numerical | Family member count |
| ChronicDiseases | Numerical | Whether the customer has a chronic illness |
| FrequentFlyer | Categorical | 4+ flight bookings in 2017–2019 |
| EverTravelledAbroad | Categorical | Whether the customer has travelled abroad |
| TravelInsurance | Numerical (target) | Whether the customer bought the insurance |

## Approach

1. **EDA** — correlation heatmap, histograms, bar charts and pie charts across all categorical and numerical features
2. **Preprocessing** — label encoding for categorical columns, MinMax normalization for `Age`, `AnnualIncome`, `FamilyMembers`
3. **Class balancing** — SMOTE oversampling, since the target class is imbalanced
4. **Model** — a Sequential ANN-MLP: `Dense(32) → BatchNorm → Dense(16) → BatchNorm → Dense(8) → BatchNorm → Dense(1, sigmoid)`, trained with Adam + binary cross-entropy
5. **Evaluation** — accuracy, Cohen's Kappa, confusion matrix, precision, recall, F1

## Results

| Metric | Value |
|---|---|
| Accuracy | 80.08% |
| Cohen's Kappa | 72.80 |
| Precision | 77% |
| Recall | 65% |
| F1 Score | 71% |

## Why ANN-MLP

The assignment report compares ANN-MLP against Decision Tree, Naïve Bayes, and Logistic Regression for this problem — ANN-MLP was selected for its ability to model non-linear relationships between the mixed categorical/numerical features without manual feature interaction terms, at the cost of being less interpretable than the simpler baselines.

## A known issue worth flagging

In the current code, **SMOTE oversampling and MinMax scaling are both fit before the train/test split**, rather than after. This means the test set isn't fully held out — synthetic samples and scaling statistics from the "test" data leak into training, which can inflate the reported accuracy above what the model would actually achieve on genuinely unseen data. Correcting this (split first, then fit SMOTE and the scaler on the training set only) is a natural next improvement and is noted here rather than silently glossed over.

## Repo structure

```
TravelInsurancePrediction/
├── data/
│   └── TravelInsurancePrediction.csv
├── notebooks/
│   └── travel_insurance_prediction.ipynb
├── docs/
│   └── assignment-report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/travel_insurance_prediction.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · imbalanced-learn (SMOTE) · seaborn · matplotlib

## Author

**Tharaneetharan Thavarasan**
ERP Consultant (SAP FICO / Microsoft D365 F&O) | Data & Business Analyst | AI/ML
[LinkedIn](https://www.linkedin.com/in/tharaneetharan-thavarasan-52754940) · [Portfolio](https://tharaneetharan7.github.io/tharanee-portfolio/#home) · [Medium](https://medium.com/@ttharaneetharan7)

## License

Apache-2.0 — see [LICENSE](./LICENSE).
