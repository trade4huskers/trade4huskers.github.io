---
title: "College Football Ranking Systems - MSDS"
layout: post
---

This project will use machine learning to create a new ranking system for FBS college football teams. The dataset comes from TeamRankings.com. Their website provides statical data for many different sports, including college football.


# BUSINESS PROBLEM/HYPOTHESIS
College Football rankings determine playoff eligibility, seeding, bowl assignments, and guide sports betters on team performance relative to opponents.
According to Wayne Staats (2025) of ncaa.com, “There are many polls to keep track of during the college football season, from the AP Poll and College Football Playoff rankings in the FBS to the FCS and Division II polls.” In addition to the traditional polls, there are many private ranking systems such as ESPN’s College Power Index, and Team Ranking’s Predictive Rankings.
With the ever-evolving need for precise rankings, I will build a new ranking system that uses predictive analytics from past performance.

Stakeholders include:
- Coaches
- Ranking Polls
- Betting companies
- Individual fans and bettors

# METHODS/ANALYSIS
My methods have included standard EDA and charts and then move to transformations. Transformations include Label Encoding. The dataset comes with last year's ranking which will be used as the target.  After completing data cleaning and transformation, I applied machine learning techniques to rank college teams.

Clean, wrangle and transform the various datasets.
- Build dataframes from the csv files.
- Look for correlation.
- Transform data for machine learning.

Data Visualization
- Matplotlib libraries
- Correlation Chart
- Box Plots
- Scatter Plots

Machine Learning
- Sklearn models
- Train, Test, Split, Fit, Predict
- Linear Elastic Net  Model
- Gradient Boost Light Model
- Random Forest Model

# Visualizations
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/cfb_project_correlation.png)
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/cfb_project_boxplot_offense.png)
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/cfb_project_boxplot_defense.png)
![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/cfb_project_scatterplot.png)

# The Code

```python
# Jeff Thomas
# DSC680
# Professor Amirfarrokh Iranitalab
# Project 3
# Created November 8, 2025
# Updated November 15

# Import all necessary libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr, kendalltau

# College Football Ranking System
This project will perform machine learning on a college football team statistics to rank them based on 2024 results

## Step 1
## Read Files
Read in the following csv files. Credit to Team Rankings of teamrankings.com
- cfb_ranking_2024
- cfb_stats_2024

# Using the read_csv method, read the online payment fraud csv
df1 = pd.read_csv('cfb_ranking_2024.csv')

df1.head()

# Using the read_csv method, read the online payment fraud csv
df2 = pd.read_csv('cfb_stats_2024.csv')

df2.head()

## Step 2
## Merge the two dataframes

# Assume df1 and df2 are your DataFrames
merged_df = pd.merge(df1, df2, on='Team')

merged_df.head()

## Step 3
## Visualizations
- Correlation Heat Map
- Box plots
- Scatter Bubble Chart

# Create the correlation heatmap

num_cols = ["Rank","OPOINTPG","OYDPG","OPLAYPG","ORUSHYPG","OPASSYPG",
             "DPOINTPG","DYDPG","DPLAYPG","DRUSHYPG","DPASSYPG"]

plt.figure(figsize=(10,8))
corr = merged_df[num_cols].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Team Stats and Rank")
plt.show()

# Create a box plot for Points per Game by Conference

plt.figure(figsize=(10,6))
sns.boxplot(data=merged_df, x='Conference', y='OPOINTPG')
plt.xticks(rotation=45)
plt.title("Offensive Points per Game by Conference")
plt.show()

# Create a box plot for Defensive Points per Game by Conference

plt.figure(figsize=(10,6))
sns.boxplot(data=merged_df, x='Conference', y='DPOINTPG')
plt.xticks(rotation=45)
plt.title("Defensive Points per Game by Conference")
plt.show()

# Create a scatter bubble chart for Team efficiency

plt.figure(figsize=(8,6))
sns.scatterplot(data=merged_df, x='OPOINTPG', y='DPOINTPG', hue='Conference', size='Rank', sizes=(30,200))
plt.gca().invert_yaxis()  # lower DPOINTPG = better defense
plt.title("Offense vs Defense: Team Efficiency Map")
plt.xlabel("Offensive Points per Game")
plt.ylabel("Defensive Points Allowed per Game")
plt.show()

## Step 4
## Transformations
- Transform non numeric columns
- Drop the Team column

# Use Label Encoder on Conference

le_conf = LabelEncoder()

merged_df['Conference'] = le_conf.fit_transform(merged_df['Conference'])

merged_df.head()

# Drop the Team column

merged_df.drop('Team', axis=1, inplace=True)

merged_df.head()

## Step 5
## Model Building
Build three different models well suited for a binary outcome
- Linear Elastic Net
- Gradient Boost Light
- Random Forests

### Split data

# Split features (X) and target variable (y)
X = merged_df.drop('Rank', axis=1)
y = merged_df['Rank']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Model 1 - Linear Elastic Net

# Pipeline for scaling
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(max_iter=10000, random_state=42))
])

# Hyperparameter grid layout
param_grid = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0],
    "model__l1_ratio": [0.1, 0.5, 0.9]
}

# Setup Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)

# Make predictions
y_pred = grid.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)
kendall_corr, _ = kendalltau(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"Spearman ρ: {spearman_corr:.3f}")
print(f"Kendall τ: {kendall_corr:.3f}")

### Model 2 - Gradient Boost Light

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# fit the model
lgb_model.fit(X_train, y_train)

# Make predictions
y_pred = lgb_model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)
kendall_corr, _ = kendalltau(y_test, y_pred)

# Print the results

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"Spearman ρ: {spearman_corr:.3f}")
print(f"Kendall τ: {kendall_corr:.3f}")

# Predicted Top 25 on full dataset
y_pred_all = lgb_model.predict(X)
leaderboard = pd.DataFrame({
    "Team": df1["Team"],
    "PredictedRank": y_pred_all,
    "ActualRank": df1["Rank"]
})
leaderboard['PredictedRankInt'] = leaderboard['PredictedRank'].rank(method='first')
top_25 = leaderboard.sort_values('PredictedRankInt').head(25)

print("\nPredicted Top 25 Teams:")
print(top_25[["Team","PredictedRank","PredictedRankInt","ActualRank"]])

### Model 3 - Random Forests

#### Run CV Grid Search for best Parameters

# Define the spearman function
def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

spearman_skl_scorer = make_scorer(spearman_scorer, greater_is_better=True)

# Setup the Random Forest Regressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Create the paramater grid
param_dist = {
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 5],
    "max_features": ["sqrt", 0.5, 0.7]
}

# Randomize the search
rand_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring=spearman_skl_scorer,
    verbose=2,
    random_state=42
)
rand_search.fit(X_train, y_train)

# Show the best model
best_rf = rand_search.best_estimator_
print("Best Hyperparameters:", rand_search.best_params_)

# Predict
y_pred = best_rf.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)
kendall_corr, _ = kendalltau(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"Spearman ρ: {spearman_corr:.3f}")
print(f"Kendall τ: {kendall_corr:.3f}")

#### Re-run Model with best Parameters

### Model 3 - Random Forests

# Setup the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=700,        # number of trees
    max_depth=15,          # grow trees fully
    min_samples_split = 2,
    min_samples_leaf=1,      # prevents overfitting
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
spearman_corr, _ = spearmanr(y_test, y_pred)
kendall_corr, _ = kendalltau(y_test, y_pred)

# Print results
print(f"Random Forest R²: {r2:.3f}")
print(f"Random Forest MAE: {mae:.2f}")
print(f"Spearman ρ: {spearman_corr:.3f}")
print(f"Kendall τ: {kendall_corr:.3f}")

# Predict ranks for the full dataset
y_pred_all = best_rf.predict(X)  # X = all features for all 136 teams

# Create full leaderboard
full_leaderboard = pd.DataFrame({
    "Team": df1["Team"],
    "PredictedRank": y_pred_all,
    "ActualRank": df1["Rank"]
})

# Convert predictions to integer-like ranks for sorting
full_leaderboard['PredictedRankInt'] = full_leaderboard['PredictedRank'].rank(method='first')

# Sort by predicted rank and take top 25
top_25_predicted = full_leaderboard.sort_values('PredictedRankInt').head(25)

print("Predicted Top 25 Teams (full dataset):")
print(top_25_predicted[["Team","PredictedRank","PredictedRankInt","ActualRank"]])

### Load and preprocess 2025 data

import pandas as pd

# Load 2025 stats
df_2025 = pd.read_csv("cfb_2025.csv")

# Iniatiate the label enconder
le_conf = LabelEncoder()

# Encode the conference feature
df_2025['Conference'] = le_conf.fit_transform(df_2025['Conference'])

# Ensure feature columns match training data exactly
features = [
    "Conference", "OPOINTPG","OYDPG","OPLAYPG","ORUSHYPG","OPASSYPG",
    "DPOINTPG","DYDPG","DPLAYPG","DRUSHYPG","DPASSYPG"
]

X_2025 = df_2025[features]
y_2025_actual = df_2025["Rank"]

### Make predictions using your trained Random Forest

# Predict ranks for 2025
y_2025_pred = best_rf.predict(X_2025)

# Build a leaderboard
leaderboard_2025 = pd.DataFrame({
    "Team": df_2025["Team"],
    "PredictedRank": y_2025_pred,
    "ActualRank": y_2025_actual
})

# Optional: convert predicted ranks to integer-like order
leaderboard_2025['PredictedRankInt'] = leaderboard_2025['PredictedRank'].rank(method='first')

# Sort by predicted rank
top_25_2025 = leaderboard_2025.sort_values('PredictedRankInt').head(25)

print("Predicted Top 25 for Week 13 2025:")
print(top_25_2025[["Team","PredictedRank","PredictedRankInt","ActualRank"]])

### Evaluate model performance

# Create the metrics
r2_2025 = r2_score(y_2025_actual, y_2025_pred)
mae_2025 = mean_absolute_error(y_2025_actual, y_2025_pred)
spearman_2025, _ = spearmanr(y_2025_actual, y_2025_pred)
kendall_2025, _ = kendalltau(y_2025_actual, y_2025_pred)

# Print results
print(f"R²: {r2_2025:.3f}")
print(f"MAE: {mae_2025:.2f}")
print(f"Spearman ρ: {spearman_2025:.3f}")
print(f"Kendall τ: {kendall_2025:.3f}")

### Evaluate Top 25 accuracy

# Find the accuracy of the top 25
actual_top25 = set(df_2025[df_2025['Rank'] <= 25]['Team'])
predicted_top25 = set(top_25_2025['Team'])
correct_count = len(actual_top25 & predicted_top25)

print(f"Top 25 accuracy: {correct_count}/25")

```

# RESULTS
I completed three machine learning (ML) predictive analysis models. The target was end of season ranking in 2024. The theory was to see if enough correlation exists between the features in the dataset to rank teams in an order that makes sense, but not necessarily an exact match.

The outcome of the three models follows:

![image](https://raw.githubusercontent.com/trade4huskers/trade4huskers.github.io/master/images/cfb_project_results.png)

# ETHICAL CONSIDERATIONS
One ethical consideration is to ensure the model does not include bias or overfitting. It should be fair, and conferences such as power five versus all others can often create bias. Model use is another consideration. Ultimately, these are young men who have worked very hard to get to a college football team. Inaccurate results could impact playoffs, as well as financial gain or loss for anyone using them to place sport bets.

# CONCLUSION & RECOMMENDATIONS
While this model could be used at a high level for individual users, it is not ready to be used as a final ranking for public consumption or betting.  More features should be used to see if the model metrics can improve.  For example, Wins and Losses could be added as additional features for training and testing.

# REFERENCES
ESPN Internet Ventures. (n.d.). 2025 college football power index. ESPN. https://www.espn.com/college-football/fpi
 
NCAA College Football Predictive Rankings & Ratings. Team Rankings. (n.d.). https://www.teamrankings.com/college-football/ranking/predictive-by-other?date=2025-01-21

Staats, W. (2025, January 22). College Football Rankings: Every poll explained and how they work. NCAA.com. https://www.ncaa.com/news/football/article/2025-01-22/college-football-rankings-every-poll-explained-how-they-work 


