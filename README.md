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

4. **Model Training**  
   - Splits data into train/test sets  
   - Trains a **Logistic Regression** classifier  

5. **Evaluation**  
   - Outputs classification report, confusion matrix, F1 Score, and Precision  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/credit-risk-pipeline.git
cd credit-risk-pipeline
pip install -r requirements.txt
