---
title: "Predicting Walmart Store Sales using Machine Learning – Python"
layout: post
---

Project 3 - This is our third project during the Spring semester at Bellevue University, and we have 
chosen to work on forecasting Walmart sales using data from Kaggle.com. We will be working 
with four different datasets covering the period from 2010-02-05 to 2013-07-26. We select 
important features available in the datasets to predict sales. As you guessed correctly, our target 
variable is sales, given here by the weekly sales average. After cleaning, transforming, and 
splitting our data in training and test datasets, we will choose the appropriate model, and build our 
machine-learning model. If satisfied with its performance after training, we will then use it on the 
testing dataset for prediction. Overall, our model achieved an accuracy score of 97%, meaning 
there is only a 3% chance of error. With this score, we feel that the model is good, and will 
successfully predict the sales. 


# BUSINESS PROBLEM/HYPOTHESIS

Walmart Inc. is an American multinational retail corporation that operates a chain of 
hypermarkets, discount department stores, and grocery stores in the United States. Our objective 
in this project is to predict weekly store sales. In other words, the aim is to build a system by 
which future weekly sales volumes are estimated. The dataset for example has time-related data 
that are given as features, so analyzing if sales are impacted by time-based factors like holidays 
will be useful in this prediction.

# Stakeholders of Walmart

Walmart shareholders include large asset managers, other institutional investors, 
individual retail investors, and its own associates (For example in 2022, 38% of their active full-
time and salaried U.S. associates participated in at least one of Walmart’s stock ownership 
programs, including equity awards and their Associate Stock Purchase Plan). 

# METHODS/ANALYSIS
For our analysis, we will use Python, (an interpreted, object-oriented, high-level 
programming language with dynamic semantics, widely used by Data Scientists to perform data 
analysis).

We start by importing necessary libraries (see list provided below), then importing the 4 datasets, 
and converting them into panda data frames for manipulation. The data was downloaded from 
Kaggle.com in a .csv format and then imported into Python using pandas. 

Python libraries make it easy for us to handle the data and perform typical and complex tasks with 
a single line of code. 

A few of the main libraries we are using here include: 

- Pandas – This library helps to load the data frame in a 2D array format and has multiple 
functions to perform analysis tasks in one go. 
Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and 
manipulation tool, built on top of the Python programming language. 
- NumPy – NumPy which stands for Numerical Python, is a library consisting of 
multidimensional array objects and a collection of routines for processing those arrays. It’s 
very fast and can perform large computations in a very short time. It also has functions for 
working in the domain of linear algebra, Fourier transform, and matrices. 
- Matplotlib: Matplotlib is a library for creating static, animated, and interactive 
visualizations in Python. It makes things easy and possible. 
- Seaborn – Seaborn is a Python data visualization library based on matplotlib. It provides a 
high-level interface for drawing attractive and informative statistical graphics. 
- scikit-learn/sklearn – This module contains multiple libraries that have pre–implemented 
functions to perform tasks from data preprocessing to model development and evaluation. 
-- Simple and efficient tools for predictive data analysis 
-- Accessible to everybody, and reusable in various contexts 
-- Built on NumPy, SciPy, and matplotlib, just to name a few. 

After loading the data frames, we will merge them and get what we need. The main data frame 
is of size 421570 rows and 17 columns. We explored the data frames to better understand them 
and merged them. Some of the tasks to accomplish include Checking for null values, and 
duplicate values, checking the size and, shape, datatype, etc... 

After all the above steps, we’ll end with the machine learning step. we’ll Choose the 
appropriate model (between the Linear regression model and the Random forest model), 
Train the model, Evaluate the model, do Hyperparameter tuning, and finally, make the 
prediction. 

- Sklearn models /Train, Test, Split, Fit, and Predict: 
- Linear Regression (LR): 
-- LR is a type of statistical analysis used to predict the relationship between two 
variables. It assumes a linear relationship between the independent variable and the dependent 
variable and aims to find the best-fitting line that describes the relationship. The line is 
determined by minimizing the sum of the squared differences between the predicted values 
and the actual values. LR is used to predict the value of a variable based on the value of 
another variable. The variable you want to predict is called the dependent variable. The 
variable you are using to predict the other variable's value is called the independent variable. 

- Random Forest: 
-- Random Forest is one of the most popular and commonly used algorithms by Data 
Scientists. Random forest is a Supervised Machine Learning Algorithm that is used widely in 
Classification and Regression problems. It builds decision trees on different samples and 
takes their majority vote for classification and average in case of regression. One of the most 
important features of the Random Forest Algorithm is that it can handle the data set 
containing continuous variables, as in the case of regression, and categorical variables, as in 
the case of classification. It performs better for classification and regression tasks.

- Hyperparameter tuning using GridSearchCV. 

# The Code

```python
##### Import Python Libraries

#Import libraries
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
%matplotlib inline

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

##### Loading the datasets
#Loading the datasets
data1 = pd.read_csv("E:\\Bellevue\\Spring_2023\\DSC410_Predictive Analytics\\Project\\Walmart\\Historical Product Demand.csv")
data_store= pd.read_csv("E:\\Bellevue\\Spring_2023\\DSC410_Predictive Analytics\\Project\\Walmart\\stores.csv")
data_train= pd.read_csv("E:\\Bellevue\\Spring_2023\\DSC410_Predictive Analytics\\Project\\Walmart\\train.csv")
data_test= pd.read_csv("E:\\Bellevue\\Spring_2023\\DSC410_Predictive Analytics\\Project\\Walmart\\test.csv")
data_features= pd.read_csv("E:\\Bellevue\\Spring_2023\\DSC410_Predictive Analytics\\Project\\Walmart\\features.csv")

print(data_test.shape)
data_test.info()

##### Analyse the data

print(data_store.head())
print("\n", data_store.head())
print("\n", data_test.head())
print("\n", data_features.head())

#Let's check the size fo each dataset
print(data_store.shape)
print(data_train.shape)
print(data_test.shape)
print(data_features.shape)

#Let's check the data type
#data_train.info()

#Merging the datasets (3)

# Assuming that features and stores need to be merged with train and test datasets based on common columns
train = data_train.merge(data_features, on=['Store', 'Date'], how='left')
train = train.merge(data_store, on=['Store'], how='left')
test = data_test.merge(data_features, on=['Store', 'Date'], how='left')
test = test.merge(data_store, on=['Store'], how='left')
print(train.shape, test.shape)

#Merging to make only one dataframe
data = data_train.merge(data_features, on=['Store', 'Date'], how='inner').merge(data_store, on=['Store'], how='inner')
print(data.shape)
data.head(5)

data['ln_Weekly_Sales'] = np.log(data.Weekly_Sales)

#Let's delete duplicate columns and rename other ones from our merged new dataframe
data.drop(['IsHoliday_y'], axis=1,inplace=True) # removing the column
data.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) # renaming the column

print("\n", data.shape)
data.head() # last ready data set

#Let's reduce our dataset by selecting all the week sales greater than 0
data = data.loc[data['Weekly_Sales'] > 0]
data.shape

print(data['Date'].head(5))
print(data['Date'].tail(5))

#### We have our data from febuary 5th 2010 to October 26th 2012

#Let's get the full info
print(data.info(show_counts= False), "\n")
data.columns

#changing the Date column datatype
data["Date"] = pd.to_datetime(data["Date"])
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

#### Check for unique values in each column

data.nunique()

#### Missing values Calculation

data.isnull().sum()

#The below code helps to calculate the percentage of missing values in each column
(data.isnull().sum()/(len(data)))*100

# Note: Clearly we see that there are missing values in MarkDowns columns. 
# Because these columns are of type float64, we can replace thr missing values with 0. 
# The business gave markdown columns to see the effect if markdowns on sales.

#Filling null's with 0
df = data.fillna(0)

#Last null check
print(df.isna().sum())
df.shape

##### Creating new Features

#Let's separate data following their datatype
numeric_data=data.loc[:,((data.dtypes=='int64')|(data.dtypes=='int32')| (data.dtypes=='float64')|(data.dtypes=='float32'))]
categoric_data=data.loc[:, (data.dtypes=='object')]
date_data=data.loc[:,(data.dtypes=='datetime64[ns]')]

print(numeric_data.shape)
print(categoric_data.shape)
print(date_data.shape)

# Introducing a new column,Date features

# We will be extracting day,month and year from Date
data['Day']=pd.DatetimeIndex(date_data['Date']).day
data['Month']=pd.DatetimeIndex(date_data['Date']).month
data['Year']=pd.DatetimeIndex(date_data['Date']).year
print(df.shape)
data.head()

#Creating new variable

from datetime import datetime
import calendar

def weekend_or_weekday(year,month,day):
	d = datetime(year,month,day)
	if d.weekday()>4:
		return 1
	else:
		return 0
data['Weekend'] = data.apply(lambda x:weekend_or_weekday(x['Year'], x['Month'], x['Day']), axis=1)
data.Weekend = data.Weekend.astype(bool)
data.head()

#Creating new variable
from datetime import datetime
df_model=data.copy()
n=np.where
d=data['Date']

data['Super_Bowl']=n((d==datetime(2010,2,12))|(d==datetime(2011,2,11))|(d==datetime(2012,2,10))|(d==datetime(2013,2,8)),1,0)
data['Labor_Day']=n((d==datetime(2010,9,10))|(d==datetime(2011,9,9))|(d==datetime(2012,9,7))|(d==datetime(2013,9,6)),1,0)
data['Thanksgiving']=n((d==datetime(2010,11,26))|(d==datetime(2011,11,25))|(d==datetime(2012,11,23))|(d==datetime(2013,11,29)),1,0)
data['Christmas']=n((d==datetime(2010,12,31))|(d==datetime(2011,12,30))|(d==datetime(2012,12,28))|(d==datetime(2013,12,27)),1,0)
data_holy = data[['Super_Bowl', 'Labor_Day','Thanksgiving', 'Christmas', 'Weekend']]
print(data.shape)
print(data_holy.shape)
data.head()

data.info()

#### Exploratory Data Analysis

df = data.copy()
df.describe()

df.describe(include='all').T

#Let's check how many departments and stores are in our dataset
print("The data set has:", df['Store'].nunique(), "stores")
print("and", df['Dept'].nunique(), "departments")

store_dept_look = pd.pivot_table(df, index='Dept', columns='Store',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_look)

#Visualizing sales by store
plt.figure(figsize=(30,10))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=df)

Rows represent the 81 department labeled 1-99 (some numbers are jumped like), columns represent the 45 Stores.

#Visualizing sales by Department
x = df['Dept']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Store')
plt.xlabel('Department')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()

#Lets separate Numerical and categorical variables for easy analysis

num_var=df.loc[:,((df.dtypes=='int64')|(df.dtypes=='int32')| (df.dtypes=='float64')|(df.dtypes=='float32'))]
cat_var=df.loc[:, (df.dtypes=='object')]
date_var=df.loc[:,(df.dtypes=='datetime64[ns]')]

print("Categorical Variables:")
print(cat_var.shape)
print("\nNumerical Variables:")
print(num_var.shape)
print("\nDate Variables:")
print(date_var.shape)

#Let's drop some columns
num_cols1 = df[['Store', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].copy()

## Y should be normal.
df.Weekly_Sales.sum()
sns.distplot(df.Weekly_Sales)

holyday = ['Super_Bowl', 'Labor_Day','Thanksgiving', 'Christmas', 'Weekend']

#let's compare sales during holiday and the rest of the day
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df) # Holiday vs not-Holiday

Let's see the correlation between features (X) and the target (Y)

#There must be a relation between X and Y
corr=df.corr()
corr

def desc_num_feature(feature_name, bins=30, edgecolor='k', **kwargs):
    fig, ax = plt.subplots(figsize=(8,4))
    df[feature_name].hist(bins=bins, edgecolor=edgecolor, ax=ax, **kwargs)
    ax.set_title(feature_name, size=15)
    plt.figtext(1,0.15, str(df[feature_name].describe().round(2).astype(str)), size=17)

desc_num_feature('Weekly_Sales')

#We also evaluate distribution kurtosis and asymmetry:

print(f"Skewness: {df['Weekly_Sales'].skew()}")
print(f"Kurtosis: {df['Weekly_Sales'].kurt()}")

From this information we see how the distribution:
-does not follow a normal curve
-show spikes
-has kurtosis and asymmetry values greater than 1
We do this for each variable, and we will have a pseudo-complete descriptive picture of their behavior.
We need this work to fully understand each variable, and unlocks the study of the relationship between variables.

for col in numeric_data:
    print(col)
    print('Skewness :', round(data[col].skew(), 2))
    print('Kurtosis :', round(data[col].kurt(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    data[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.show()
    
#Categorical variables
print(df.Type.value_counts(), "\n")

#Let's express the data as a percentage by passing normalize = True
df.Type.value_counts(normalize=True)

#Let's plot it
my_data = [51.1132, 38.7824 , 10.1044 ]  #percentages
my_labels = 'Type A','Type B', 'Type C' # labels
plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) #plot pie type and bigger the labels
plt.axis('equal')
mpl.rcParams.update({'font.size': 20}) #bigger percentage labels

plt.show()

#We can also plot with
df.Type.value_counts().plot(kind="bar")
plt.title("Value counts of the type variable")
plt.xlabel("Store type")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.show()

Type of Store and holyday sales

# Avg weekly sales for types on Christmas
df.groupby(['Christmas','Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Labor_Day
df.groupby(['Labor_Day','Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Thanksgiving
df.groupby(['Thanksgiving','Type'])['Weekly_Sales'].mean()

# Avg weekly sales for types on Super bowl
df.groupby(['Super_Bowl','Type'])['Weekly_Sales'].mean()

Let's plot it

# Plotting avg wekkly sales according to holidays by types
plt.style.use('seaborn-poster')
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27370.73, 20605.69, 19973.22, 18031.03]
B_means = [18661.30, 12401.72, 12013.48, 11394.05]
C_means = [9679.90,10126.20,9871.23,7963.23]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x , B_means, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30,color='r') # holidays avg
plt.axhline(y=15952.82,color='green') # not-holiday avg

fig.tight_layout()

plt.show()

Now let’s check the variation of stock (per day) as the month closes to the end.

plt.figure(figsize=(10,5))
df.groupby('Day').mean()['Weekly_Sales'].plot()
plt.title('Average Sales - Daily')
plt.show()

plt.figure(figsize=(10,5))
df.groupby('Month').mean()['Weekly_Sales'].plot()
plt.title('Average Sales - Monthly')
plt.show()

plt.figure(figsize=(15, 10))

# Calculating Simple Moving Average
# for a window period of 30 days
window_size = 30
data = df[df['Year']==2011]
windows = data['Weekly_Sales'].rolling(window_size)
sma = windows.mean()
sma = sma[window_size - 1:]

data['Weekly_Sales'].plot()
sma.plot()
plt.legend()
plt.show()

plt.figure(figsize=(16,6))
df['Weekly_Sales'].plot()
plt.show()

No way to clearly see the trend of sales. Let's visualize them weekly, monthly and yearly

df.groupby('Month')['Weekly_Sales'].mean() # to see the best months for sale

df.groupby('Year')['Weekly_Sales'].mean() # to see the best months for sale

monthly_sales = pd.pivot_table(df, values = "Weekly_Sales", columns = "Year", index = "Month")
monthly_sales.plot()

Note: clearly we can see that 2010 had high sales
Let's check the correlation between the variables.

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr().abs())
plt.show()

Temperature, unemployment, CPI, Fuel_Price and markdown4 and 5 have no significant effect on weekly sales, We drop them.  
It can create multicollinearity problem, maybe. So, first I will try without them.

#Change encoding
df2 = df.copy() # to keep original dataframe taking copy of it
df2['Super_Bowl'] = df2['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
df2['Thanksgiving'] = df2['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
df2['Labor_Day'] = df2['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
df2['Christmas'] = df2['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
df2['IsHoliday'] = df2['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1
df_new = df2.copy() # taking the copy of encoded df to keep it original

df2.fillna(0)
df2.head()

#Dropping columns with negative correction with the target
drop_col = ['Temperature','MarkDown4','MarkDown5','CPI','Unemployment', 'Weekend', 'Fuel_Price', 'Labor_Day']
df_new.drop(drop_col, axis=1, inplace=True) # dropping columns

plt.figure(figsize=(12, 7))
sns.heatmap(df_new.corr(), annot = True, vmin = -1, vmax = 1)
plt.show()

Now Let's represent subgroups from our dataset for the regression analysis (dummy variable)
#### Dummy Variables

aset=df[['Store','Dept']]
bset=df[['Day','Month','Year']]

def create_dummies(df,colname):
    col_dummies=pd.get_dummies(df[colname],prefix=colname,drop_first=True)
    df=pd.concat([df,col_dummies],axis=1)
    df.drop(colname,axis=1,inplace=True)
    return df
    
for c_feature in aset:
    aset[c_feature]=aset[c_feature].astype('category')
    aset=create_dummies(aset, c_feature)
for c_feature in bset:
    bset[c_feature]=bset[c_feature].astype('category')
    bset=create_dummies(bset,c_feature)
    
print(aset.shape)
print(bset.shape)

cat_var['Type']=cat_var['Type'].astype('category')
cat_var = create_dummies(cat_var,'Type')

cat_var.info()

#Changing the data type
#df['Type']=df['Type'].astype('category')
#df = create_dummies(df,'Type')

#Endcoding the cumlumn type
encoder = LabelEncoder()
test['Type'] = encoder.fit_transform(test['Type'])
train['Type'] = encoder.fit_transform(train['Type'])

dff=df.copy()
dff.info()

#Let's drop some variables
dff.drop(columns=['Store','Dept', 'Date'], inplace=True)
dff.drop(columns=['Day','Month','Year'], inplace=True)

#Let's merge all the data sets
dff = pd.concat([dff, aset, bset], axis=1)
dff.shape

dff.describe().T

### Building prediction model
### Feature Engineering

#We are going to replace missing value with 0
df1 = data.copy()

df1 = data.fillna(0)
train = train.fillna(0)
test = test.fillna(0)
print(df1.shape)

df1.isnull().sum() #check for missing value

#Dropping less importance columns, and column with negative correction with the target
#df1 = df1.drop(['Store', 'Temperature','MarkDown4','MarkDown5','CPI','Unemployment', 'Weekend', 'Christmas', 'Year',
#                'Fuel_Price', 'Day', 'Labor_Day', 'Size', 'IsHoliday', 'Super_Bowl'], axis=1) # dropping columns
#Encoding the Colums type
encoder = LabelEncoder()
df1['Type'] = encoder.fit_transform(df1['Type'])

print(df1.info())
df1.head()

df1_train = df1[:-210106]
df1_test = df1[-210106:]
print(df1_train.shape)
print(df1_test.shape)

# Train the model using linear regression
X_test = df1_test.drop(['Weekly_Sales', 'Date'], axis=1)
X_train1 = df1_train.drop(['Weekly_Sales', 'Date'], axis=1)
y_train1 = df1_train['Weekly_Sales']
print(X_test.shape)
print(X_train1.shape)
print(y_train1.shape)

from sklearn.preprocessing import StandardScaler

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test)

# Splitting the data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train1, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the validation set
y_val_pred_LR = model.predict(X_val)
y_val_pred_LR[:5]

# Training the random forest regressor model
model_random = RandomForestRegressor()
model_random.fit(X_train, y_train)

# Making predictions on the validation set
y_val_pred_RF = model_random.predict(X_val)
y_val_pred_RF[:5]

# Step 4: Calculate the accuracy of the model
mse_LR = mean_squared_error(y_val, y_val_pred_LR)
r2_LR = r2_score(y_val, y_val_pred_LR)

mse_RF = mean_squared_error(y_val, y_val_pred_RF)
r2_RF = r2_score(y_val, y_val_pred_RF)

print('1. Mean Squared Error for Logistic Regression model:', mse_LR)
print('2. R-squared Score for Logistic Regression model:', r2_LR)

print('\n3. Mean Squared Error for Random Forest model:', mse_RF)
print('4. R-squared Score for Random Forest model:', r2_RF)

# Make LR predictions on the test data
predictions_LR = model.predict(X_test)

# Make RF predictions on the test data
predictions_RF = model_random.predict(X_test)

# Add a column of predicted sales to the original DataFrame
df1_test['LR_Predicted_Sales'] = predictions_LR
df1_test['RF_Predicted_Sales'] = predictions_RF

df1_predicted = df1_test[['Store', 'Dept', 'Date', 'Weekly_Sales', 'RF_Predicted_Sales', 'Year', 'IsHoliday']].copy()
df1_predicted.rename(columns={"Weekly_Sales": "Actual_Sales", "RF_Predicted_Sales": "Predicted_Sales"}, inplace=True)
df1_predicted.head(15)

df1_train['Weekly_Sales'].plot(figsize=(20,8), title= 'Sales', fontsize=14, label= 'Train')
df1_test['RF_Predicted_Sales'].plot(figsize=(20,8), title= 'Sales Prediction', fontsize=14, label= "Test")
plt.show()

# Step 3: Train the model and make predictions
X_train2 = train.drop(['Date', 'Weekly_Sales', 'Type'], axis=1)
y_train2 = train['Weekly_Sales']

# Splitting the data into train and validation sets
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train2, y_train2, test_size=0.2, random_state=42)

# Training the random forest regressor model
model = RandomForestRegressor()
model.fit(X_train2, y_train2)

# Making predictions on the validation set
y_val_pred2 = model.predict(X_val3)
y_val_pred2[:5]

# Step 4: Calculate the accuracy of the model
mse = mean_squared_error(y_val3, y_val_pred2)
r2 = r2_score(y_val3, y_val_pred2)

print('Mean Squared Error:', mse)
print('R-squared Score:', r2)

# Step 5: Add a column of predicted sales to the original DataFrame
train['Predicted_Sales'] = model.predict(X_train2)

X_test2 = test.drop(['Date', 'Type'], axis=1)
test['Predicted_Sales'] = model.predict(X_test2)

df_pred = train[['Date', 'Weekly_Sales', 'Predicted_Sales']].copy()
df_pred.head(15)

fig, ax = plt.subplots(figsize=(20, 8))
plt.plot(train['Date'], train['Weekly_Sales'], color='red',label='Actual')
plt.plot(test['Date'], test['Predicted_Sales'], color='purple',label="Predicted")
plt.xlabel("Date")
plt.ylabel("Sales")
leg = plt.legend()
plt.show()

# Step 6: Plot Actual Sales vs. Predicted Sales
fig, ax = plt.subplots(figsize=(13, 6))
plt.plot(train['Date'], train['Weekly_Sales'], label='Actual Sales')
plt.plot(test['Date'], test['Predicted_Sales'], label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual Sales vs. Predicted Sales')
plt.legend()
plt.show()

```

# RESULTS
As expected, holiday average sales are higher than on normal dates. During the data 
exploration step, we found that holidays have a high impact on sales. We have used a barplot to 
confirm that, as we all know that visuals speak louder than numerals. Because Data Visualization 
helps in analyzing the data better as well as gives a better understanding than just reading random 
numbers, we have thought to find and visualize the holidays subgroup, by finding and showing 
which one has high impact than others. We have found that Stores type A have the higher weekly 
sales during Thanksgiving, followed by type B, and then C. Superbowl is the second holiday with 
high sales, a bit close to Labor Day, and then Christmas comes at the bottom position in sales. We 
have also found that January sales are significantly less than other months. This is the result of 
November and December high sales (when comparing monthly sales). After two high sales 
months, people prefer to pay less in January.

Moving to the next step of training and testing our model (using the train and validation 
data), we two different regressor models, to keep the best one after testing the accuracy: The 
Linear Regression and the Random Forest. We have achieved 89% with the Linear Regression 
and 97% with the Random Forest regressor and conclude being satisfied with the Random Forest 
model based on its accuracy score of 97%, meaning that our model has only a 3% prediction 
error. 

With our tested and approved model, we finally made the predictions using the test data. Overall, 
we were able to validate that the model is predicting weekly sales. Again, we chose to graph the 
prediction versus the actual sales on top of a short table of numbers we displayed as proof of the 
predictions.

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/b2476a15-31a2-41b8-bb5a-62a3d3c6ea74)

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/e5b68c9c-b095-4a04-b3cd-dd5318a192e5)

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/cb9144a8-f093-4681-8126-48eb1b4fcbe2)

# RECOMMENDATIONS/ETHICAL CONSIDERATIONS

Regardless of the size of the company, sales forecasting allows firms to: 
- Envision future sales revenues. 
- Allocate human and monetary resources according to forecasts. 
- Prepare strategies geared towards future growth. 
- 
For these reasons, building a system by which future sales volumes are estimated should be given 
with care depending on the stakeholder. The sales forecast may be overestimated or 
underestimated as a result, that is why having at least 3 years of data can help to determine the 
seasonality, train, and test the model. While the data used to make these predictions is found on 
public websites and readily available, we are not 100% sure of its authenticity. That should not 
be taken lightly. With that in mind, these predictions should be considered carefully, and we do 
recommend this study should be combined with other models (like Bayesian models for time 
series forecasting, including predictors such as marketing expenditure and industry trends) and 
considerations before using this prediction for any reason. 

# CONCLUSION
We as a team, for project 3, will have pulled 4 different datasets from Kaggle.com, all 
relating to Walmart store sales. We have used numerous libraries available in Python 
programming language throughout this project, PyCharm Professional 2022.2.1 with Jupyter 
Notebook being our IDE of choice. Seaborn and matplotlib have been the libraries utilized for 
our charts/graphs. We have tried and compared two different models, Linear regressor, and 
Random Forest regressor. We were satisfied by the Random Forest accuracy score of 97% we 
achieved. Weekly sales were our target variable, and we are satisfied because we met our goal of 
forecasting sales.

# REFERENCES
E R, S. (2021, June 17). Random Forest | Introduction to Random Forest Algorithm. Analytics 
Vidhya. https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/<br><br>
IBM. (n.d.). What is Random Forest? | IBM. Www.ibm.com. 
https://www.ibm.com/topics/random-forest<br><br> 
Rheude, J. (2020, July 15). Demand Forecasting: Types, Methods, and Examples. Red Stag 
Fulfillment. https://redstagfulfillment.com/what-is-demand-forecasting/<br><br>
Singh, R. (2020, November 3). Predicting Credit Card Approvals using ML Techniques. 
Medium. https://medium.datadriveninvestor.com/predicting-credit-card-approvals-usingml-techniques-9cd8eaeb5b8c<br><br>
Stakeholder Engagement. (n.d.). 2022 ESG. 
https://corporate.walmart.com/esgreport/stakeholder-engagement<br><br> 
Team, G. L. (2020, February 14). How Machine Learning is Simplifying Sales Forecasting & 
Increasing Accuracy. GreatLearning Blog: Free Resources What Matters to Shape Your 
Career! https://www.mygreatlearning.com/blog/how-machine-learning-is-used-in-salesforecasting/<br><br> 
Thete, J. (2022, March 7). A Stochastic Model For Demand Forecating In Python. MLearning.ai. 
https://medium.com/mlearning-ai/a-stochastic-model-for-demand-forecating-in-pythona1b568b80b94<br><br>
Walmart Sales Forecast. (n.d.). Www.kaggle.com. Retrieved May 18, 2023, from 
https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast?select=train.csv

# Video Link

For more details about our project, please check out our video!
<br/>
[Project 3 - Walmart Sales Prediction](https://www.youtube.com/watch?v=8w66_iANhVI)
