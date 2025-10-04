---
title: "Predicting FX Trade Signals - Machine Learning"
layout: post
---

This project will attempt to predict buy and sell signals for the currency EURUSD (Euro dollar vs US dollar). I will work with four different datasets in the form of csv files. Each file represents five years of trading data from four currency pairs. The four pairs are EURUSD, AUDUSD, USDCAD, and USDJPY. The data includes 5-years of the daily Open, Close, High and Low prices for each currency.  After transforming, merging, and feature engineering I will compare the outcome of three different machine learning models.


# BUSINESS PROBLEM/HYPOTHESIS
According to the Federal Reserve Bank of New York (2024), “The foreign exchange (FX) market is the largest financial market in the world by trading volume, with average daily turnover of approximately $7.5 trillion”. The foreign exchange market comprises banks, forex dealers, commercial entities, central banks, investment management firms, hedge funds, retail forex dealers, and individual investors (Ganti 2024). I will use data science and machine learning models to make predictions on whether a buy or sell should take place on the EURUSD currency pair.

Stakeholders include:
- Banks
- Forex Dealers
- Central Banks
- Ivestment Management Firms
- Hedge Funds
- Retail Dealers
- Individual Investors

# METHODS/ANALYSIS
The approach will be finding correlation with the data, which will result in a target that can be used for a supervised machine learning model.  We will begin by collecting our datasets, wrangling, and cleaning the data (formatting appropriately in Excel and Jupyter Notebook) using Pandas and Numpy in dataframes.  There will be multiple built in packages used such as correlation matrix function corr() and merging functions such as merge().  Data visualization will be used to help support our findings, and a machine learning model will be deployed.

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
- Gradient Boost Classifier

# Visualizations
![image](images/fx_project_correlation.png)
![image](images/fx_project_timeseries_combined.png)
![image](images/fx_project_timeseries_subset.png)

# The Code

```python
# Import all necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Forex Buy and Sell Signals
This project will use various currency pairs to make a daily Buy or Sell prediction on the Euro versus US Dollar (EURUSD).

## Step 1
## Read Files
Read in the following currency pairs into a dataframe
- EURUSD (Euro vs USD)
- AUDUSD (Australian Dollar vs USD)
- USDCAD (USD vs Canadian Dollar)
- USDJPY (USD vs Japanese Yen)

### EURUSD (Euro vs USD)

# Using the read_csv method, read the EURUSD csv
eur1 = pd.read_csv('EURUSD_HistoricalData.csv')

eur1.head()

### AUDUSD (Australian Dollar vs USD)

# Using the read_csv method, read the AUDUSD csv
aud1 = pd.read_csv('AUDUSD_HistoricalData.csv')

aud1.head()

### USDCAD (USD vs Canadian Dollar)

# Using the read_csv method, read the USDCAD csv
cad1 = pd.read_csv('USDCAD_HistoricalData.csv')

cad1.head()

### USDJPY (USD vs Japanese Yen)

# Using the read_csv method, read the USDJPY csv
jpy1 = pd.read_csv('USDJPY_HistoricalData.csv')

jpy1.head()

## Step 2
## Transformations
Perform the following transformations on each dataframe
- Remove the Volume column as Nasdaq does not provide it on currencies
- Rename each column with the specific currency
- Check the length of each dataframe to ensure they are all the same
- Merge the 4 dataframes

### EURUSD (Euro vs USD)

# Create new df for EURUSD and drop Volume column
eur2 = eur1.drop('Volume', axis=1)

# Rename columns to designate EURUSD using a dictionary and inplace=True
eur2.rename(columns={'Close/Last': 'EURUSD_Close', 'Open': 'EURUSD_Open',
'High': 'EURUSD_High', 'Low': 'EURUSD_Low'}, inplace=True)

# Check the last 3 columns to ensure column title and number of rows
eur2.tail(3)

### AUDUSD (Australian Dollar vs USD)

# Create new df for AUDUSD and drop Volume column
aud2 = aud1.drop('Volume', axis=1)

# Rename columns to designate EURUSD using a dictionary and inplace=True
aud2.rename(columns={'Close/Last': 'AUDUSD_Close', 'Open': 'AUDUSD_Open',
'High': 'AUDUSD_High', 'Low': 'AUDUSD_Low'}, inplace=True)

# Check the last 3 columns to ensure column title and number of rows
aud2.tail(3)

### USDCAD (USD vs Canadian Dollar)

# Create new df for USDCAD and drop Volume column
cad2 = cad1.drop('Volume', axis=1)

# Rename columns to designate EURUSD using a dictionary and inplace=True
cad2.rename(columns={'Close/Last': 'USDCAD_Close', 'Open': 'USDCAD_Open',
'High': 'USDCAD_High', 'Low': 'USDCAD_Low'}, inplace=True)

# Check the last 3 columns to ensure column title and number of rows
cad2.tail(3)

### USDJPY (USD vs Japanese Yen)

# Create new df for USDJPY and drop Volume column
jpy2 = jpy1.drop('Volume', axis=1)

# Rename columns to designate EURUSD using a dictionary and inplace=True
jpy2.rename(columns={'Close/Last': 'USDJPY_Close', 'Open': 'USDJPY_Open',
'High': 'USDJPY_High', 'Low': 'USDJPY_Low'}, inplace=True)

# Check the last 3 columns to ensure column title and number of rows
jpy2.tail(3)

### Merge all 4 to one dataframe
I can see from the previous steps that each dataframe has a slightly different row count. Merging on date should help align. It will remove the rows that don't match, but there should be enough data to allow for that

# First Merge
# Merge EURUSD to AUDUSD using an inner join
merge1 = pd.merge(eur2, aud2, on='Date', how='inner')

merge1.tail(3)

# Second Merge
# Merge USDCAD to merge1 using an inner join
merge2 = pd.merge(merge1, cad2, on='Date', how='inner')

merge2.tail(3)

# Third Merge
# Merge USDJPY to merge2 using an inner join
merge3 = pd.merge(merge2, jpy2, on='Date', how='inner')

merge3.tail(3)

## Step 3
## Visualizations
Create the appropriate dataframes to perform the following visualizations
- Correlation Matrix of closing price of each currency
- Time series chart of each currency in one chart normalized
- Time series chart of each currency seperately as sub plots

### Correlation Matrix
Correlation chart of closing prices

#Create a correlation dataframe of each currency
cor1 = merge3[['EURUSD_Close', 'AUDUSD_Close', 'USDCAD_Close', 'USDJPY_Close']].copy()

# Perform correlation matrix
cor1.corr()

### Time Series plot of all currencies
Time Series plot using normalization to flatten disparity in prices

import matplotlib.pyplot as plt

#Create a time series dataframe of each currency
dt1 = merge3[['Date','EURUSD_Close', 'AUDUSD_Close', 'USDCAD_Close', 'USDJPY_Close']].copy()

# Make sure date column is datetime type
dt1['Date'] = pd.to_datetime(dt1['Date'])

# Set date as index
dt1.set_index('Date', inplace=True)

# Normalize each column to start at 100
normalized_df = dt1 / dt1.iloc[0] * 100

# Plot all currency columns
normalized_df.plot(figsize=(12, 6), title='Currency Closing Prices Over Time (5 years)')

# Show the plot
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid(True)
plt.legend(title='Currency')
plt.tight_layout()
plt.show()

### Time Series plot of each currency as sub plots
Time Series plot as sub plots to show each currency side by side

# Show sub plots of each currency
dt1.plot(subplots=True, figsize=(10, 8), layout=(2, 2), title="Currency Closing Prices (5 years)")
plt.tight_layout()
plt.show()

## Step 4
## Feature Engineering
## Create the Target on EURUSD
Create the Buy and Sell target using a shift method
- Buy = 1
- Sell = 0
- If next day is higher than previous then Buy, else Sell
- Create some moving averages (MA) and try them in the model

# Create a new target dataframe from the merge
tar1 = merge3

# Create a new Target column and compare current day's EUR with previous day's EUR
target = tar1['EURUSD_Close'].gt(tar1['EURUSD_Close'].shift(1)).map({True: 0, False: 1})

# Insert the new column at position 1
tar1.insert(loc=1, column='Target', value=target)

tar1

# Create a 10 day moving average using

ma1 = tar1.copy()

ma1['MA_EUR_Close_10'] = ma1['EURUSD_Close'].rolling(window=10).mean()

ma1.head()

# Create a 10 day moving average using 

ma1['MA_EUR_Close_30'] = ma1['EURUSD_Close'].rolling(window=30).mean()

ma1.head()

Step 5
Model Building¶
Build three different models well suited for a binary outcome

Logistic Regression
Random Forest
Gradient Boost Classifier

# Create new df for training and remove the date column
final = tar1.drop('Date', axis=1).copy()

# Remove all rows with any NaN values which were created from the MA features
final = final.dropna()

final.head()

### Split data

# Split features (X) and target variable (y)
X = final.drop('Target', axis=1)
y = final['Target']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Model 1 - Logistic Regression

# Set and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make the predictions
y_pred = model.predict(X_test)
# Show the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

### Model 2 - Random Forest

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

### Model 3 - Gradient Boost Classifier

# Create Gradient Boosting Classifier model
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

# RESULTS
I completed three machine learning (ML) predictive analysis models. The target was based on the EURUSD closing price. The dataset included the other three currencies (EURUSD, USDCAD, and USDJPY). The theory was to see if enough correlation exists between the four currency pairs to predict whether a buy or a sell should take place on the EURUSD currency.

The outcome of the three models follows:

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/e9890f79-4e0d-42f7-91a6-32eeb09e7e67)

# ETHICAL CONSIDERATIONS
The biggest ethical consideration is how to deal with trading decisions as they impact financial balance sheets. Whether a hedge fund, individual investor, stockbroker, or bank, trading the FX markets can lead to both profits and losses. Therefore, trading based on machine learning results could lead to unexpected losses. Ensuring the use of clean data and thorough model testing is important for producing unbiased outcome predictions.

# CONCLUSION & RECOMMENDATIONS
The Gradient Boost model shows the most promise at 55%, but none of these models are ready for production use. More features could be added or subtracted to attempt a better score. While the score is not absolute, it should be considered. Generally, a score of 75% or higher would be recommended. However, other factors should be considered such as profitability, which could be more important than what percentage of times the model picks the correct buy and sell signals.

Further analysis of this model needs to be completed before it is production ready. First, add more currencies to the data frame to see if the score will improve. Second, additional feature engineering such as moving average (MA) could be added for potential score improvement. Third, an element of profit should be included to help guide the effectiveness of the model. 55% accuracy might be excellent if it results in large profits.

# REFERENCES
FX Market Structure Conference. FEDERAL RESERVE BANK of NEW YORK. (2024, November 19). https://www.newyorkfed.org/newsevents/events/markets/2024/1119-2024 
Ganti, A. (2024, September 5). Foreign Exchange Market: How It Works, history, and pros and cons. Investopedia. https://www.investopedia.com/terms/forex/f/foreign-exchange-markets.asp 
Stock market, data updates, reports & news. Nasdaq. (n.d.). https://www.nasdaq.com/ 
