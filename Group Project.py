import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv('C:\Users\anant\OneDrive\Desktop\MBAN\MBAN 6110\Datasets\Final Presentation\train.csv')

# Convert 'Age' column to numeric and fill in missing or erroneous ages
# If the age is missing for a customer ID, replace it with the most frequently occurring age for that customer ID
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].apply(lambda x: x if 0 < x < 100 else None)
df['Age'] = df.groupby('Customer_ID')['Age'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))

# Fill in missing customer names
# If the name is missing for a customer ID, replace it with the most frequently occurring name for that customer ID
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Fill in missing occupations
# If the occupation is missing for a customer ID, replace it with the most frequently occurring occupation for that customer ID
df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# Clean and convert 'Annual_Income' and 'Monthly_Inhand_Salary' columns
# Remove any non-numeric characters and convert the columns to numeric
for column in ['Annual_Income', 'Monthly_Inhand_Salary']:
    df[column] = df[column].replace('[^0-9.]','', regex=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Fill in missing or changed 'Annual_Income' based on occupation
# If the annual income is missing for a customer ID and the occupation has not changed, replace it with the most frequently occurring annual income for that customer ID
# If the occupation has changed, replace the missing annual income with the annual income from the previous month
mode_annual_income = df.groupby('Customer_ID')['Annual_Income'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
mode_annual_income.columns = ['Customer_ID', 'Mode_Annual_Income']
df = pd.merge(df, mode_annual_income, on='Customer_ID', how='left')
df.loc[(df['Annual_Income'].isna()) & (df['Occupation'] == df['Occupation'].shift()), 'Annual_Income'] = df['Mode_Annual_Income']
df.loc[(df['Annual_Income'].isna()) & (df['Occupation'] != df['Occupation'].shift()), 'Annual_Income'] = df['Annual_Income'].fillna(method='ffill')
df.drop('Mode_Annual_Income', axis=1, inplace=True)

# Fill in missing 'Monthly_Inhand_Salary' based on customer ID
# If the monthly salary is missing for a customer ID, replace it with the most frequently occurring monthly salary for that customer ID
mode_monthly_salary = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
mode_monthly_salary.columns = ['Customer_ID', 'Mode_Monthly_Salary']
df = pd.merge(df, mode_monthly_salary, on='Customer_ID', how='left')
df.loc[df['Monthly_Inhand_Salary'].isna(), 'Monthly_Inhand_Salary'] = df['Mode_Monthly_Salary']
df.drop('Mode_Monthly_Salary', axis=1, inplace=True)

# Clean and convert a list of numerical columns
# Remove any non-numeric characters and convert the columns to numeric
num_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
            'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
for column in num_cols:
    df[column] = df[column].replace('[^0-9.]','', regex=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Fill in missing 'Type_of_Loan' with the mode of the customer ID or 'No Loan'
# If the type of loan is missing for a customer ID, replace it with the most frequently occurring type of loan for that customer ID
# If there is no frequently occurring type of loan for a customer ID, replace it with 'No Loan'
df['Type_of_Loan'] = df.groupby('Customer_ID')['Type_of_Loan'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "No Loan"))

# Split the 'Type_of_Loan' column into separate columns for each unique loan type
# Create a binary indicator for each unique loan type
df['Type_of_Loan'] = df['Type_of_Loan'].str.split(',')
mlb = MultiLabelBinarizer()
loan_types = pd.DataFrame(mlb.fit_transform(df['Type_of_Loan']),columns=mlb.classes_, index=df.index)
df = pd.concat([df, loan_types], axis=1)

# Rename 'Not Specified' column to 'Other'
df.rename(columns={'Not Specified': 'Other'}, inplace=True)

# Split 'Credit_History_Age' into years and months, calculate the total age in months, and handle missing values
# If the credit history age is missing for a customer ID, replace it with the average of the credit history age from the previous and next month
df[['Years', 'Months']] = df['Credit_History_Age'].str.split(' and ', expand=True)
df['Years'] = df['Years'].str.replace(' Years', '').astype(float)
df['Months'] = df['Months'].str.replace(' Months', '').astype(float)
df['Total_Age_in_Months'] = df['Years']*12 + df['Months']
na_rows = df['Total_Age_in_Months'].isna()
df.loc[na_rows, 'Total_Age_in_Months'] = df.groupby('Customer_ID')['Total_Age_in_Months'].apply(lambda group: group.fillna(method='bfill').fillna(method='ffill'))
df.drop(['Years', 'Months', 'Credit_History_Age'], axis=1, inplace=True)

df.head()
