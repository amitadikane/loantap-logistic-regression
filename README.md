# LoanTap — Personal Loan Underwriting (Logistic Regression)

**Domain:** FinTech / Credit Risk  
**Tools:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn, SciPy  
**Focus:** EDA → Feature Engineering → Logistic Regression → Precision-Recall Tradeoff

---

## 📌 Problem Statement

LoanTap offers instant, flexible credit products to salaried millennials and needs an underwriting layer for Personal Loans.

Business questions:
1. Given an applicant’s financial and credit attributes, should LoanTap extend a credit line?
2. How do we balance NPA risk (approving defaulters) vs lost interest income (rejecting good applicants)?

---

## 🎯 Project Workflow

1. Exploratory Data Analysis (~396K loan applications)
2. Feature Engineering (term, emp_length, grade, risk flags)
3. Missing value treatment & multicollinearity check (VIF)
4. Build class-weighted Logistic Regression model
5. Evaluate with Classification Report, ROC-AUC, Precision-Recall Curve
6. Threshold trade-off analysis for underwriting decisions
7. Deliver actionable recommendations for LoanTap

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, StandardScaler)
- SciPy, Matplotlib, Seaborn

---

## 💡 Key Results

- ROC-AUC ≈ 0.71
- Strong risk drivers: Grade, DTI, Term, Purpose (small business), Revolving Utilization, Annual Income
- Fully Paid ≈ 80.4% | Charged Off ≈ 19.6%
- Recommend operating below 0.5 threshold when NPA control is priority

---

## 📓 Notebook & Full Report

- **Jupyter Notebook**: [./notebooks/](./notebooks/loantap_analysis.ipynb)
- **PDF Report**: [./reports/](./reports/loantap_analysis.pdf)

---

## 🚀 Skills Demonstrated

- End-to-end credit risk modeling with Logistic Regression
- Class imbalance handling & VIF-based feature cleaning
- Precision-Recall trade-off for business decisions
- Translating model output into underwriting recommendations

---

**Author:** Amit Narendra Adikane  
**GitHub:** [amitadikane](https://github.com/amitadikane)  
**LinkedIn:** [amit-adikane](https://www.linkedin.com/in/amit-adikane-4060a91b1/)
