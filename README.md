# Automated Credit Risk Categorization Pipeline

A machine learning pipeline that automates **credit risk classification** for loan applicants.  
This project cleans raw applicant data, balances classes using **SMOTE**, trains a **Logistic Regression model**, and evaluates performance with standard metrics.

---

## 📂 Project Structure
1. **Data Cleaning**  
   - Removes duplicates and fixes Applicant_ID  
   - Cleans and normalizes categorical fields (Name, City, Employment_Status, Marital_Status, Education_Level, Existing_Loans, Default)  
   - Handles missing values and outliers in numeric fields (Age, Loan_Amount_Requested, Credit_Score, Annual_Income, Debt_to_Income_Ratio)  

2. **Data Scaling & Visualization**  
   - Scales numeric features using MinMaxScaler  
   - Generates histograms, boxplots, and categorical distributions  

3. **Balancing Dataset**  
   - Applies **SMOTE** to handle class imbalance in the Default column  

4. **Model Training
