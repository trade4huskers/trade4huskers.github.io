---
title: "Financial Transaction Fraud Detection - MSDS"
layout: post
---

This project will attempt to predict fraud transactions on a financial dataset. The dataset comes from Kaggle.com and the author is Rupak Roy. He states that the online payment fraud big dataset is for testing and practice purposes. 


# BUSINESS PROBLEM/HYPOTHESIS
According to Ben Luthi of Experian (2025), “The FTC logged more than 1.1 million identity theft reports in 2024. The federal agency also received roughly 2.6 million cases of related fraud, with total losses of more than $12.7 billion.”
A big concern for financial firms is Account Takeover Fraud (ATO). Account Takeover Fraud is where cyber criminals deliberately gain unauthorized access to a victim's online bank, payroll, health savings or social media account, with the goal of stealing money or information for personal gain (Internet Crime Complaint Center).
Developing machine learning models can help financial firms detect fraud and stop it before significant losses are incurred.

Stakeholders include:
- Banks
- Credit Unions
- Brokerage Firms
- Money Transfer Companies

# METHODS/ANALYSIS
My methods have included standard EDA and charts and then move to transformations. Transformations include Label Encoding and Dummy Encoding. The dataset comes with a pre-determined Target called “isFraud”, which is binary.  After completing data cleaning and transformation, I applied machine learning techniques to forecast fraud on this dataset.

Clean, wrangle and transform the various datasets.
- Build dataframes from the csv files.
- Look for correlation.
- Transform data for machine learning.

Data Visualization
- Matplotlib libraries
- Correlation chart
- Time series plots

Machine Learning
- Sklearn models
- Train, Test, Split, Fit, Predict
- Logistic Regression Model
- Random Forest Model
- XGBoost Classifier

# Visualizations
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/fraud_project_correlation.png)
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/fraud_project_bar_type.png)
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/fraud_project_boxplot.png)

# The Code

```python
# Jeff Thomas
# DSC680
# Professor Amirfarrokh Iranitalab
# Project 2
# Created October 19, 2025
# Updated November 2, 2025

# Import all necessary libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Financial Transaction Fraud Detection 
This project will perform machine learning on a financial transactions dataset, and attempt to predict fraud.

## Step 1
## Read Files
Read in the following dataset. Credit Rupak Roy from kaggle.com
- Online Payment Fraud

# Using the read_csv method, read the online payment fraud csv
df = pd.read_csv('Online_Payment_Fraud.csv')

df.head()

## Step 2
## EDA

# Get the shape of the dataframe to find how big the dataframe is

# We can see it is 11 columns by 6,362,620 rows

df.shape

# Use the describe method to get numerical data

# Note: the dataset is too big for this method to be useful as it is

df.describe()

# Use the Info method to learn more about feature types

df.info()

# Use the correlation method to get a quick glance if any numeric features are correlated to "isFraud"

corr = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']].copy()

corr.corr()

## Step 3
## Visualizations
- Bar chart
- Box plot
- Correlation Heat Map
- Line plot

# Bar chart to count the different Types of transactions

df['type'].value_counts().plot(kind='bar', title='Transaction Type Distribution')

# Box plot to show count of fraud vs. non-fraud

# Note: The fraud amount (1) is concentrated to smaller amounts

sns.boxplot(x='isFraud', y='amount', data=df)
plt.title('Transaction Amounts by Fraud Status')
plt.xlabel('Is Fraud') 
plt.ylabel('Transaction Amount')
plt.show()

# Use seaborn to plot the correlation map

plt.figure(figsize=(8,6))
sns.heatmap(
    df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap of Transaction Features', fontsize=14, fontweight='bold')
plt.show()

# Plot a horizontal bar plot to show the top 10 accounts that commit fraud

# Note the top 10 each had 2 counts of fraud

df[df['isFraud']==1]['nameDest'].value_counts().head(10).plot(kind='barh', title='Top Fraud Destination Accounts')

# Plot to show fraud trends over time, using the step feature

# Note, fraud is increasing over timing

df.groupby('step')['isFraud'].mean().plot(title='Fraud Rate Over Time')

## Step 4
## Transformations
#### Transform non numeric columns

# Use Label Encoder on nameOrig & nameDest

le_orig = LabelEncoder()
le_dest = LabelEncoder()

df['nameOrig'] = le_orig.fit_transform(df['nameOrig'])
df['nameDest'] = le_dest.fit_transform(df['nameDest'])

df.head()

# Get dummy values for the Type column
df = pd.get_dummies(df, columns=['type'])

df

## Step 5
## Model Building
Build three different models well suited for a binary outcome
- Logistic Regression
- Random Forest
- XGBoost Classifier

### Split data

# Split features (X) and target variable (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Model 1 - Logistic Regression

# Set and fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make the predictions
y_pred = model.predict(X_test)

# Show the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

### Model 2 - Random Forest

# Set and fit the model
model = RandomForestClassifier(n_estimators=30, n_jobs=-1, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Show the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

### Model 3 - XGBoost Classifier

# Create XGBoost Classifier model
model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Show the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

# RESULTS
I completed three machine learning (ML) predictive analysis models. The target was pre-determined in the dataset as "isfraud". The theory was to see if enough correlation exists between the features in the dataset to predict fraud on a transaction.

The outcome of the three models follows:

![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/fraud_project_results.png)

# ETHICAL CONSIDERATIONS
One of the biggest ethical considerations is to ensure the model does not include Bias. Fraud enforcement sometimes unfairly focuses on demographics or characteristics. Other ethical concerns include lack of accountability, where the model cannot be explained why it did or did not flag a transaction as fraudulent.

# CONCLUSION & RECOMMENDATIONS
All three models are successful at predicting fraud scoring over 99%. The Logistic Regression model ran considerably faster than the other two models.  If consideration is given to moving forward with this project, the speed of the Logistic Regression model should be considered as a possible advantage. Currently, I don’t believe this project is ready for production, although the models show promise.  The biggest concern is the possibility of overfitting since all three produced such high accuracy scores. Therefore, I recommend continuing to check for class imbalances with other metrics such as Precision, Recall, F1-Score, and ROC-AUC.

# REFERENCES
Internet Crime Complaint Center. (n.d.). Account takeover fraud (ATO). Internet Crime Complaint Center. https://www.ic3.gov/CrimeInfo/AccountTakeover
 
Luthi, B. (2025, May 30). U.S. fraud and identity theft losses topped $12.7 billion in 2024. Experian. https://www.experian.com/blogs/ask-experian/identity-theft-statistics/
 
Roy, R. (2022, April 17). Online payments fraud detection dataset. Kaggle. https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset 

