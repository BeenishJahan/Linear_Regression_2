# Credit Risk Classification using Machine Learning

A comprehensive machine learning project for predicting loan default risk using peer-to-peer lending data from Lending Club. This project demonstrates the complete ML workflow from data preprocessing to model deployment, comparing three algorithms to achieve optimal performance.

## 🎯 Project Overview

**Objective:** Build a binary classification model to predict whether a borrower will default on their loan

**Dataset:** Lending Club loan data (2007-2018)
- **Initial size:** 2,260,701 loans
- **Final training set:** 1,347,706 loans (after cleaning)
- **Features:** 18 engineered features from 151 original columns
- **Target:** Binary (0 = Fully Paid, 1 = Default)
- **Class distribution:** 82% Paid, 18% Default

**Best Model:** XGBoost with 0.7171 ROC-AUC

---

## 📊 Results Summary

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.7046 | 0.32 | 0.63 | 0.42 |
| Random Forest | 0.7099 | 0.37 | 0.43 | 0.40 |
| **XGBoost** | **0.7171** | **0.32** | **0.67** | **0.43** |

**Key Achievement:** 1.77% improvement over baseline through ensemble methods

---

## 🔧 Technologies Used

- **Python 3.x**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Development:** Jupyter Notebook

---

## 📁 Project Structure
```
credit-risk-classification/
│
├── data/                          # Data files (not tracked in git)
│   └── accepted_2007_to_2018Q4.csv
│
├── notebooks/
│   └── model.ipynb # Main analysis notebook
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore file
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/BeenishJahan/Credit_Risk.git
cd Credit_Risk
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Download `accepted_2007_to_2018Q4.csv.gz`
- Extract and place in `data/` directory


---

## 📈 Project Workflow

### 1. Data Cleaning & Preprocessing
- **Missing value treatment:**
  - Dropped 63 rows (0.003%) with critical missing data
  - Median imputation for numeric features (dti, revol_util, pub_rec_bankruptcies)
  - Filled categorical features (emp_length with "Unknown", mort_acc with 0)
  
- **Target variable preparation:**
  - Filtered to completed loans only (dropped 40% "Current" status loans)
  - Binary encoding: Fully Paid → 0, Charged Off/Default → 1
  - Final dataset: 1,348,069 loans

### 2. Exploratory Data Analysis
- **Correlation analysis:** Identified weak linear correlations (strongest: int_rate +0.26)
- **Distribution analysis:** Found right-skewed features requiring transformation
- **Class imbalance:** Confirmed 82/18 split requiring balanced weighting

### 3. Feature Engineering

**Created features:**
- `log_income`: Log-transformed annual income (normalized skewness)
- `loan_to_income`: Loan amount / annual income ratio (relative loan size)
- `has_pub_rec`: Binary indicator for public records
- `has_bankruptcy`: Binary indicator for bankruptcy records
- `grade_encoded`: Ordinal encoding of Lending Club risk grade (A-G)
- `emp_length_encoded`: Ordinal encoding of employment length

**Encoding strategies:**
- **Ordinal:** grade (A→0, G→6), emp_length (<1yr→0, 10+→10)
- **One-hot:** home_ownership (4 categories), verification_status (2 categories)
- **Binary:** Public records, bankruptcy flags

**Dropped features:**
- Redundant: fico_range_high (correlated 1.0 with fico_range_low)
- Used in ratios: loan_amnt, installment, annual_inc
- Simplified: purpose (14 categories, minimal additional value)
- Weak predictors: open_acc, has_inq, has_mortgage

**Final feature set:** 18 features (6 numeric, 2 engineered, 2 binary, 2 ordinal-encoded, 6 one-hot-encoded)

### 4. Model Training & Evaluation

**Train/Test Split:**
- 70/30 split with stratification
- Training set: 943,648 samples
- Test set: 404,058 samples

**Feature Scaling:**
- StandardScaler fitted on training data only (avoiding data leakage)
- Applied to both train and test sets

**Models Trained:**

1. **Logistic Regression (Baseline)**
   - Parameters: max_iter=5000, class_weight='balanced'
   - Purpose: Establish baseline linear performance
   - Result: 0.7046 ROC-AUC

2. **Random Forest**
   - Parameters: n_estimators=200, max_depth=20, class_weight='balanced'
   - Purpose: Capture non-linear patterns through bagging
   - Result: 0.7099 ROC-AUC (+0.75%)

3. **XGBoost (Final Model)**
   - Parameters: n_estimators=300, max_depth=6, scale_pos_weight=4
   - Purpose: Optimize performance through gradient boosting
   - Result: 0.7171 ROC-AUC (+1.77%)

**Evaluation Metrics:**
- Primary: ROC-AUC (handles imbalanced classes)
- Secondary: Precision, Recall, F1-Score

---

## 🔍 Key Findings

### Model Performance
1. **Incremental improvement validated methodology:** Each model improved over the previous, demonstrating proper progression from simple to complex
2. **Feature engineering impact:** Strong baseline (0.70) limited gains from ensemble methods, proving effective feature engineering reduces need for model complexity
3. **XGBoost superiority:** Best overall discrimination with highest recall (67%), catching most defaults while maintaining competitive precision

### Business Insights
- **Interest rate strongest predictor (+0.26 correlation):** Bank's own risk assessment already embedded in pricing
- **Credit score moderate predictor (-0.13):** FICO scores help but aren't dominant
- **Income weak predictor (-0.04):** Debt burden (DTI) matters more than raw income
- **High false positive cost:** Model rejects 109K good borrowers to catch 51K defaults (Logistic baseline)

### Technical Insights
- **Class imbalance handling critical:** Balanced class weights improved recall from ~18% (random) to 63-67%
- **Minimal overfitting:** Similar performance between train/test suggests good generalization
- **Diminishing returns:** +1.77% total improvement indicates data has inherent prediction limits

---

## 💡 Limitations & Future Work

### Current Limitations
- **Modest AUC (0.71):** Reflects inherent difficulty of credit risk prediction
- **Historical data (2007-2018):** May not reflect current economic conditions
- **Missing features:** Macroeconomic indicators, detailed debt purpose
- **High false positive rate:** 68% of predicted defaults are false alarms (precision trade-off)
- **Minor data leakage:** Median imputation performed before train/test split (negligible impact: 0.1% of data)

### Future Improvements
1. **Hyperparameter tuning:** Cross-validation for optimal parameters (potential +0.01-0.02 AUC)
2. **Feature expansion:** Include dropped features (purpose), external economic data
3. **Ensemble stacking:** Combine predictions from all three models
4. **Threshold optimization:** Business-driven decision boundary based on cost-benefit analysis
5. **Fairness analysis:** Evaluate model equity across demographic groups
6. **Deep learning:** Explore neural networks for complex pattern recognition

---

## 📚 Key Learnings

1. **Feature engineering > Model complexity:** Well-engineered features (ratios, transformations) enabled simple linear model to achieve strong baseline
2. **Domain knowledge matters:** Understanding credit risk (DTI, utilization, delinquency) guided effective feature creation
3. **Class imbalance handling essential:** Balanced weighting critical for catching minority class (defaults)
4. **Proper validation prevents leakage:** Train/test split before scaling, imputation after split
5. **Realistic expectations:** Credit risk is multifactorial; no single feature or model achieves perfect prediction

---

## 📖 References

- **Dataset:** [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Documentation:** [Lending Club Statistics](https://www.lendingclub.com/info/statistics.action)
- **Methodology:** Followed industry best practices for imbalanced classification

---

## 👤 Author

**Your Name**
- GitHub: [@BeenishJahan](https://github.com/BeenishJahan)
- LinkedIn: [Beenish Jahan](www.linkedin.com/in/beenishjahan)
- Email: beenishjahan2003@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

##  Acknowledgments

- Lending Club for providing the dataset
- Kaggle community for dataset hosting and discussions
- scikit-learn and XGBoost teams for excellent ML libraries

---

## 📝 Notes

- Dataset files are not included in repository due to size (2GB+)
- Download from Kaggle link above to reproduce results
- Random state set to 42 for reproducibility
- All models trained on standard laptop (training times may vary)

---

*Last updated: January 2025*
```