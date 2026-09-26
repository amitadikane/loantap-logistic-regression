# LoanTap — Personal Loan Underwriting (Logistic Regression)

**Domain:** FinTech / Credit Risk  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy  
**Focus:** EDA → Feature Engineering → Logistic Regression → Precision-Recall Tradeoff

---

## Problem Statement

LoanTap offers instant, flexible credit products to salaried millennials.  
This project builds an **underwriting model** to predict whether a personal loan will be **Fully Paid** or **Charged Off**, so the probability can drive credit decisions.

Approving too many risky applicants raises **NPAs**; rejecting too many good applicants means **lost interest income**. The model balances both.

---

## Dataset

- **~396,000** loan applications  
- **Target:** Fully Paid (~80.4%) vs Charged Off (~19.6%)  
- Features: loan amount, term, interest rate, grade, income, DTI, revolving utilization, home ownership, employment, credit history, etc.

---

## Approach

1. Exploratory Data Analysis (univariate + bivariate)
2. Feature Engineering (term, emp_length, grade ordinal, risk flags)
3. Missing value treatment & multicollinearity check (VIF)
4. Logistic Regression with class weighting
5. Evaluation: Classification Report, ROC-AUC, Precision-Recall Curve
6. Threshold trade-off analysis for bank decision-making
7. Actionable underwriting recommendations

---

## Key Results

| Metric | Value |
|--------|--------|
| ROC-AUC | ≈ 0.71 |
| Recall (Charged Off @ 0.5) | ≈ 63.5% |
| Precision (Charged Off @ 0.5) | ≈ 31.9% |

**Strong risk drivers:** Grade, DTI, Term (60 months), Purpose (small business), Revolving Utilization, Annual Income

---

## Business Recommendations

1. Operate **below 0.5 threshold** (e.g. 0.30–0.35) if NPA control is priority (higher recall)
2. Use **grade, DTI, purpose, term, income** as primary underwriting levers
3. Offer **tiered terms** (shorter term / lower cap) for higher-risk but still-approvable applicants instead of pure reject
4. Do **not** use geography as an underwriting rule (no significant state effect)
5. Enrich later with bureau score / cash-flow data for higher accuracy
6. Recalibrate model periodically

---

## Questionnaire Highlights

- Fully Paid: **80.39%**
- Loan Amount ↔ Installment correlation: **≈ 0.95**
- Majority home ownership: **MORTGAGE**
- Grade A more likely to fully pay: **True** (~94% vs ~52% for G)
- Top job titles: **Teacher**, **Manager**
- Primary metric for bank: **Recall** (to control NPA)
- Geography effect: **No** (not significant)

---

## Files

- `notebooks/loantap_analysis.ipynb` — Full analysis notebook
- `reports/loantap_analysis.pdf` — PDF report
- `requirements.txt` — Dependencies

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/loantap_analysis.ipynb
