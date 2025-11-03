# Power BI End-to-End Churn Analysis Project

**SQL + Power BI + Machine Learning (Random Forest) | 2025**

This project is an end-to-end customer churn analysis created to show the full data workflow — starting from raw data, moving through data cleaning and model building, and finally visualizing everything in Power BI.  
The dataset is from a telecom provider and includes customer demographics, service subscriptions, billing info, and whether they stayed or churned.

---

##  Project Objectives

- Understand which types of customers are more likely to churn.
- Identify patterns in demographics, services, contract types, and payments.
- Build a churn prediction model that can be applied to new customers.
- Present the findings clearly in Power BI so insights are easy to act on.

**Core KPIs:**  
**Total Customers**, **Total Churn**, **Churn Rate**, **New Joiners**

---

##  Workflow / Architecture

### 1) SQL (Data Cleaning & Modeling)

- Loaded raw data into a staging table (`stg_Churn`).
- Cleaned it (handled missing values, formatted fields, etc.).
- Moved final cleaned data into production table (`Prod_Churn`).

**Created 2 key SQL Views:**

| View | Purpose |
|---|---|
| `vw_ChurnData` | For model training (Stayed vs Churned) |
| `vw_JoinData`  | New customers that need churn prediction |

**Example SQL**

**Gender Breakdown**
```sql
SELECT Gender,
       COUNT(*) AS TotalCount,
       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stg_Churn) AS Percentage
FROM stg_Churn
GROUP BY Gender;
```

```sql
CREATE VIEW vw_ChurnData AS
SELECT * FROM Prod_Churn WHERE Customer_Status IN ('Churned', 'Stayed');

CREATE VIEW vw_JoinData AS
SELECT * FROM Prod_Churn WHERE Customer_Status = 'Joined';
```
### 2) Machine Learning (Python - Random Forest)
-Exported SQL view data to Excel for preprocessing.
-Removed ID-type fields that don’t help model learning.
-Encoded categorical columns.
-Converted target to binary labels (Stayed = 0, Churned = 1).
-Train-Test Split: 80/20
-Model: Random Forest (n_estimators=100)
-Evaluation: Confusion matrix, classification report, feature importance plot
-Final Accuracy: ~84%

**Model Training Snippet**

```python
Copy code
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```
Training Data: vw_ChurnData

Prediction Data: vw_JoinData (applied trained model to predict churn for newly joined users)

### 3) Power BI Dashboard
-Imported both the cleaned churn data and the ML predictions.
-Used Power Query for transformations (custom columns, unpivoting where needed).
-Built measures for KPIs in DAX.
-Designed two pages:
-Overview / Summary Dashboard
-Churn Prediction Dashboard
-Power BI Highlights
-Executive Dashboard
-Total Customers, Churn Rate, and Joiner trends
-Churn breakdown by:
-Age Groups
-Gender
-Payment & Contract types
-Internet service type
-Geography-based churn contribution
-Service combination heatmaps
-Prediction Dashboard
-List of customers predicted to churn (with likelihood context)
-Customer attributes for targeting (e.g., contract type, tenure, monthly charges)

---

##  Tech Stack

| Component        | Tools Used                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| ETL & Data Prep | SQL Server (SSMS)                                                           |
| Machine Learning | Python (pandas, scikit-learn, numpy, matplotlib, seaborn)                  |
| Visualization    | Power BI Desktop                                                            |
| Exchange Format  | Excel & CSV                                                                 |

---

##  Key Findings

- Customers on **month-to-month contracts** showed the highest churn (~46%).
- Customers **without add-on protection services** churn more frequently.
- Certain **age groups (especially 50+ female segment)** had noticeably higher churn.
- The **revenue impact** from churned customers is significant, especially among long-tenure users.
- The model successfully **flagged ~378 new joiners** with a high likelihood of churn.

---

##  How to Reproduce

1. Run the SQL scripts (inside `/sql/`) to create tables and views.
2. Open and run the Python notebook `churn_prediction.ipynb` to train the model and generate the prediction file.
3. Open Power BI → Load data from SQL + prediction CSV.
4. Refresh visuals and interact with the dashboard.

---

##  Future Enhancements

- Try more advanced models (XGBoost, LightGBM).
- Address class imbalance to improve precision for high-risk churners.
- Deploy the prediction workflow through **Streamlit / Flask**.
- Add **automated refresh & scheduled model retraining**.
