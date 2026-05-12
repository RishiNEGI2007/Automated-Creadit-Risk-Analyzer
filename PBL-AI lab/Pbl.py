 # ---------------------------------
# Automated Credit Risk Categorization Pipeline
# ---------------------------------

# 0. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load the dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\rishi\OneDrive\Desktop\PBL-AI lab\dataset1.csv")

# -------------------------------
# 2. Remove duplicates and fix Applicant_ID
# -------------------------------
df = df.drop_duplicates(subset="Applicant_ID")
df["Applicant_ID"] = pd.to_numeric(df["Applicant_ID"], errors="coerce")
df["Applicant_ID"] = df["Applicant_ID"].ffill()

# -------------------------------
# 3. Clean Name and City columns
# -------------------------------
df["Name"] = df["Name"].str.strip().str.title()
df["City"] = df["City"].str.strip().str.title()

# -------------------------------
# 4. Fix Age column
# -------------------------------
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df.loc[(df["Age"] < 18) | (df["Age"] > 80), "Age"] = np.nan
df["Age"] = df["Age"].fillna(df["Age"].median())

# -------------------------------
# 5. Normalize Employment_Status
# -------------------------------
df["Employment_Status"] = df["Employment_Status"].str.lower().str.strip()
df["Employment_Status"] = df["Employment_Status"].replace({
    "selfemployed": "Self-Employed",
    "un-employed": "Unemployed",
    "student": "Student",
    "employed": "Employed"
})

# -------------------------------
# 6. Normalize Marital_Status
# -------------------------------
df["Marital_Status"] = df["Marital_Status"].str.lower().str.strip()
df["Marital_Status"] = df["Marital_Status"].replace({
    "single": "Single",
    "married": "Married",
    "divorced": "Divorced",
    "widowed": "Widowed",
    "?": np.nan
})
df["Marital_Status"] = df["Marital_Status"].fillna("Unknown")

# -------------------------------
# 7. Clean Education_Level
# -------------------------------
df["Education_Level"] = df["Education_Level"].str.lower().str.strip()
df["Education_Level"] = df["Education_Level"].replace({
    "bachelors??": "Bachelor’s",
    "phd": "PhD",
    "high school": "High School",
    "master’s": "Master’s",
    "bachelor’s": "Bachelor’s"
})

# -------------------------------
# 8. Clean Dependents
# -------------------------------
df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")
df.loc[df["Dependents"] < 0, "Dependents"] = np.nan
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].median())

# -------------------------------
# 9. Clean Loan_Amount_Requested
# -------------------------------
df["Loan_Amount_Requested"] = (
    df["Loan_Amount_Requested"].astype(str)
    .str.replace("INR", "", regex=False)
    .str.strip()
)
df["Loan_Amount_Requested"] = pd.to_numeric(df["Loan_Amount_Requested"], errors="coerce")
df.loc[df["Loan_Amount_Requested"] < 0, "Loan_Amount_Requested"] = np.nan
df["Loan_Amount_Requested"] = df["Loan_Amount_Requested"].fillna(df["Loan_Amount_Requested"].median())

# -------------------------------
# 10. Clean Credit_Score
# -------------------------------
df["Credit_Score"] = pd.to_numeric(df["Credit_Score"], errors="coerce")
df.loc[(df["Credit_Score"] < 300) | (df["Credit_Score"] > 850), "Credit_Score"] = np.nan
df["Credit_Score"] = df["Credit_Score"].fillna(df["Credit_Score"].median())

# -------------------------------
# 11. Clean Annual_Income
# -------------------------------
df["Annual_Income"] = (
    df["Annual_Income"].astype(str)
    .str.replace("INR", "", regex=False)
    .str.strip()
)
df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")
df["Annual_Income"] = df["Annual_Income"].fillna(df["Annual_Income"].median())

# -------------------------------
# 12. Clean Debt_to_Income_Ratio
# -------------------------------
df["Debt_to_Income_Ratio"] = (
    df["Debt_to_Income_Ratio"].astype(str)
    .str.replace("percent", "", regex=False)
    .str.strip()
)
df["Debt_to_Income_Ratio"] = pd.to_numeric(df["Debt_to_Income_Ratio"], errors="coerce")
df.loc[(df["Debt_to_Income_Ratio"] < 0) | (df["Debt_to_Income_Ratio"] > 1), "Debt_to_Income_Ratio"] = np.nan
df["Debt_to_Income_Ratio"] = df["Debt_to_Income_Ratio"].fillna(df["Debt_to_Income_Ratio"].median())

# -------------------------------
# 13. Normalize Existing_Loans
# -------------------------------
df["Existing_Loans"] = df["Existing_Loans"].str.lower().str.strip()
df["Existing_Loans"] = df["Existing_Loans"].replace({
    "yes": "Yes",
    "y": "Yes",
    "no": "No",
    "n": "No"
})

# -------------------------------
# 14. Normalize Default column
# -------------------------------
df["Default"] = df["Default"].astype(str).str.lower().str.strip()
df["Default"] = df["Default"].replace({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "nan": np.nan
})
df["Default"] = df["Default"].fillna(0).astype(int)

# -------------------------------
# 15. Scale numeric columns
# -------------------------------
scaler = MinMaxScaler()
numeric_cols = ["Age","Loan_Amount_Requested","Credit_Score","Annual_Income","Debt_to_Income_Ratio"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# 16. Visualizations
# -------------------------------
df[numeric_cols].hist(bins=20, figsize=(12,8))
plt.suptitle("Histograms of Numeric Columns (After Cleaning)")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplots of Numeric Columns (After Cleaning)")
plt.show()

categorical_cols = ["Employment_Status","Marital_Status","Education_Level","Existing_Loans","Default"]
for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col} (After Cleaning)")
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------
# 17. Save cleaned dataset
# -------------------------------
df.to_csv(r"C:\Users\rishi\OneDrive\Desktop\PBL-AI lab\dataset1_cleaned.csv", index=False)
print("Cleaned dataset saved as dataset1_cleaned.csv")

# -------------------------------
# 18. Data Balancing (SMOTE)
# -------------------------------
X = df.drop("Default", axis=1)
y = df["Default"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# -------------------------------
# 19. Train Logistic Regression
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# 20. Evaluation
# -------------------------------
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

