# Identifying Root Causes of Drug Addiction Using Explainable AI (XAI)

Final Year Design Project, B.Sc. in Computer Science and Engineering, Daffodil International University (DIU), Dhaka, Bangladesh. Presented 6 January 2026.

## Overview

Drug abuse is a major public health concern in Bangladesh, especially among young people. This project combines machine learning with Explainable AI (SHAP and LIME) to predict the reported reason for drug addiction and to identify the socio-economic, psycho-social and environmental factors behind it. The goal is transparent, evidence-based insight that can support prevention and rehabilitation planning.

## Dataset

- Primary data collected through a structured questionnaire (face-to-face at rehabilitation centers with Department of Narcotics Control approval, plus online Google Forms).
- 531 raw survey responses with 19 attributes. After keeping confirmed drug users, 362 samples and 17 features remained (49 after one-hot encoding).
- Target: reason for drug addiction, 10 classes (Peer pressure, Curiosity, Failure in love, Lack of awareness, Easy availability, Loneliness or depression, Family problems, Stress or mental pressure, Failure in life, Others).
- **The dataset is not included in this repository** because it contains sensitive information about individuals. Participation was voluntary and anonymous, and no names, phone numbers or addresses were collected.

## Methodology

1. Data cleaning: median imputation for numeric features, mode for categorical features
2. One-hot encoding of categorical predictors
3. Stratified 80:20 train-test split, class weighting for imbalance
4. Statistical analysis: ANOVA, chi-square, Pearson and Spearman correlation
5. Feature analysis: correlation, mutual information, VIF
6. Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
7. Evaluation: accuracy, precision, recall, F1-score, macro ROC-AUC, 10-fold stratified cross-validation, stability across random seeds
8. Explainability: SHAP (global, class-wise, dependence, grouped), LIME (local), permutation importance

## Results

Test-set performance (weighted metrics):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| XGBoost | 0.7945 | 0.7908 | 0.7945 | 0.7817 |
| Random Forest | 0.7671 | 0.7540 | 0.7671 | 0.7427 |
| Decision Tree | 0.7534 | 0.7331 | 0.7534 | 0.7307 |
| Gradient Boosting | 0.7397 | 0.7441 | 0.7397 | 0.7262 |
| Logistic Regression | 0.5342 | 0.5528 | 0.5342 | 0.5185 |

- Ensemble and tree-based models clearly outperform the linear baseline.
- 10-fold cross-validation: Random Forest had the highest mean accuracy (0.8094 ± 0.0165), followed by XGBoost (0.7791 ± 0.0317).
- Macro-average ROC-AUC: Random Forest 0.98, XGBoost 0.97, Gradient Boosting 0.96, Decision Tree 0.85, Logistic Regression 0.84.
- XGBoost was selected for the XAI analysis for its best test-set performance, stability across random seeds, and compatibility with SHAP and LIME.

### Key findings

- Grouped SHAP impact: Socio-economic (1.83) > Psycho-social (1.70) > Environmental (0.88) > Other (0.79).
- Most influential features: Start Age, Age, Life Failure, Love Failure, Cultural Pressure, Family Relation, Psychological Stress, Peer Pressure, Financial Pressure.
- ANOVA was significant for all numeric features except Satisfaction. Chi-square was significant for employment, education, companionship, family problems, peer pressure and easy availability, but not for gender or location.
- Pearson and Spearman correlations were weak and non-significant, which supports using non-linear models with XAI.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda_statistical_tests.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_xai_shap_lime.ipynb
├── results/
│   ├── figures/        # SHAP plots, confusion matrix, ROC curves
│   └── tables/         # performance and statistical test tables
└── data/               # empty: dataset not published
```

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/explainable-ai-drug-addiction.git
cd explainable-ai-drug-addiction
pip install -r requirements.txt
```

Place your own dataset in `data/` (the original is not shared), then run the notebooks in numeric order. Developed in Python 3 using Jupyter Notebook and Google Colab.

## Limitations

- Small, single-source dataset (362 samples), so results may not generalize
- Self-reported data may include recall bias
- Not yet validated in real-world or clinical settings

## Team

- Shoaib Hossain
- Supervisor: Md. Shah Jalal, Senior Lecturer, Department of CSE, DIU
- Co-supervisor: Md. Monarul Islam, Lecturer, Department of CSE, DIU

## Status

Related manuscript under review.
