---
title: "Predicting FX Trade Signals - Machine Learning"
layout: post
---

This project will attempt to predict buy and sell signals for the currency EURUSD (Euro dollar vs US dollar). I will work with four different datasets in the form of csv files. Each file represents 5 years of trading data from four currency pairs. The four pairs are EURUSD, AUDUSD, USDCAD, and USDJPY.



# BUSINESS PROBLEM/HYPOTHESIS
Baseball is a sport filled with many statistics.  However, there are so many statistics, how can someone easily know which one to use when finding the best players?  Our stakeholders could look at numerous statistics such as Runs, RBIs, Walks, and Strikeouts just to name a few, but still may not know which statistic(s) the best is to use for selecting players.  To help solve this problem, we will use data science and machine learning models to make predictions on the best MLB offensive baseball players based on their current statistics.

Stakeholders include:
- General Managers of MLB teams
- Professional Sports Agents
- Baseball Scouts
- Fantasy Baseball Managers
- Baseball Enthusiast
- College coaches

# METHODS/ANALYSIS
The approach will be finding correlation with the data, which will result in a target that can be used for a supervised machine learning model.  We will begin by collecting our datasets, wrangling, and cleaning the data (formatting appropriately in Excel and Jupyter Notebook) using Pandas and Numpy in dataframes.  There will be multiple built in packages used such as correlation matrix function corr() and merging functions such as merge().  Data visualization will be used to help support our findings, and a machine learning model will be deployed.

Clean, wrangle and transform the various datasets.
- Build dataframes from the csv files.
- Look for correlation.
- Transform data for machine learning.

Data Visualization
- Matplotlib libraries
- Seaborn libraries
- Regression plots
- Correlation heatmaps

Machine Learning
- Sklearn models
- Train, Test, Split, Fit, Predict
- Xgboost Regressor model
- Hyperparameter tuning with Grid Search CV

# The Code

```python
# Import all necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Part 1 Data Collection
## Look at all 2021 MLB Team to determine what offensive factor is important to Winning Percentage

# Pull mlb team stats from 2021 player_stats_2021 into a dataframe

df = pd.read_csv("C:/Users/trade/DSC450/team_stats_2021.csv")

# Create a new dataframe to hold just the Team and PCT columns.  
# This will be merged later

team2021 = df[['TEAM', 'PCT', 'POST_SEASON']]

## Team EDA

#### Our first goal is to find out what individual OFFENSIVE player statistic might impact a team's success.  Success will be based on a team's winning percentage, or PCT.

#### In doing some light EDA we can see that Runs Scored (RS) has high correlation of 77% to winning percentage (PCT).  In other-words, from an offensive perspective, Runs Scored is a good target for us to conintue with at the player level.

#### Reasoning:
- RS (Runs Scored) is 100% controled by offense
- W & L are the root of PCT (Winning Percentage) so we can throw them out
- RA (Runs Against) are not controlled by offense
- DIFF (Difference of Runs Scored and Runs Against) is not completely controlled by offense

# Now we look at the correlation

#specify that all columns should be shown
pd.set_option('max_columns', None)

corrM = df.corr()

### Plot a regression chart using Winning PCT and Runs Scored to visually verify the two are correlated

# Regression plot

# use lmplot
sb.regplot(x = "RS",
           y = "PCT", 
           ci = 90,
           data = df).set(title='2021 MLB Team Winning PCT versus Runs Scored')

## Now we pull in the 2021 player stats

# Pull mlb player stats from 2021 player_stats_2021 into a dataframe

df1 = pd.read_csv("C:/Users/trade/DSC450/player_stats_2021.csv")

# Merge dataframes

df2 = team2021.merge(df1,indicator=False,how='outer')

# Now we need to do some transformation of our player dataset

# Some of the statistics are not based on a percentage. 
# Since many of the players did not play the exact same amount of games,
# we need to divide the stats by the number of games


df2['AB2'] = df2['AB'] / df2['G']
df2['R2'] = df2['R'] / df2['G']
df2['H2'] = df2['H'] / df2['G']
df2['2B2'] = df2['2B'] / df2['G']
df2['3B2'] = df2['3B'] / df2['G']
df2['HR2'] = df2['HR'] / df2['G']
df2['RBI2'] = df2['RBI'] / df2['G']
df2['BB2'] = df2['BB'] / df2['G']
df2['SO2'] = df2['SO'] / df2['G']
df2['SB2'] = df2['SB'] / df2['G']
df2['CS2'] = df2['CS'] / df2['G']

# Because some players did not play in very many games,
# we will require a player plays in at least 10 games and 1 At Bat in order to be considered

df3 = df2[df2['G'] > 10].copy()

df3 = df3[df3['AB'] > 1]

# Drop the original columns and rename the new columns

df3.drop(columns =['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS'], axis=1, inplace=True)

df3.rename(columns={'AB2': 'AB', 'R2': 'R', 'H2': 'H', '2B2': '2B',
                    '3B2': '3B', 'HR2': 'HR', 'RBI2': 'RBI', 'BB2': 'BB',
                    'SO2': 'SO', 'SB2': 'SB', 'CS2': 'CS'}, inplace=True)

# Sort by RBI (Runs Batted In) to see the best RBI hitters of 2021

# Notice the top RBI guy per game is Teoscar Hermandez of Toronto

df3.sort_values(by=['RBI'], ascending=False, inplace=True)

## Step 2

### EDA Process

#### Perform the EDA process on the merged dataframe, with Target Rank as the Target

# Drop TEAM, PCT, PLAYER and POSITION from the dataframe 

df4 = df3.drop(columns=['TEAM', 'PCT', 'POST_SEASON', 'PLAYER', 'POSITION'])

# First I will use the describe function to view some basic statistics about the dataset

df4.describe()

# Now let us look at the correlation

#specify that all columns should be shown
pd.set_option('max_columns', None)

corrM1 = df4.corr()

# Step 3

## Train-Test-Split, Fit and Predict
## Use sklearn's XGBRegressor model

# Create the x & y objects

X = df4.drop(columns=['R'])
y = df4[['R']]

# Convert to numpy

X = X.to_numpy()
y = y.to_numpy()

# Split the data into 80% training and 20% testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model

xgb_model = XGBRegressor(random_state=47)

# make a dictionary of hyperparameter values to search

# Remove 1000 to n_estimators and add 0 to max_depth

search_space = {
    "n_estimators" : [100,200,500],
    "max_depth" : [1,2,3,6,9],
    "gamma" : [0.01, 0.1],
    "learning_rate" : [0.001, 0.01, 0.1, 1]
    }

# Create the grid search for fitting the model

GS = GridSearchCV(estimator = xgb_model,
                 param_grid = search_space,
                 scoring = ["r2", "neg_root_mean_squared_error"],
                 refit = "r2",
                 cv = 5,
                 verbose = 4)

# Get the details of the best model 

print(GS.best_estimator_)

# Get the details of the best hyperparameters

print(GS.best_params_)

# Get the score of the best model

print(GS.best_score_)

# Now we predict!
## Let's predict the best MLB hitters from the 2022 season as it pertains the statistic Runs (Runs Scored)

# Pull mlb player stats from 2022 player_stats_2022 into a dataframe

df6 = pd.read_csv("C:/Users/trade/DSC450/player_stats_2022.csv")

# Because some players did not play in very many games,
# we will require a player plays in at least 10 games and 1 At Bat in order to be considered

df7 = df6[df6['G'] > 10].copy()

df7 = df7[df7['AB'] > 1]

# Now we need to do some transformation of our player dataset

# Some of the statistics are not based on a percentage. 
# Since many of the players did not play the exact same amount of games,
# we need to divide the stats by the number of games


df7['AB2'] = df7['AB'] / df7['G']
df7['R2'] = df7['R'] / df7['G']
df7['H2'] = df7['H'] / df7['G']
df7['2B2'] = df7['2B'] / df7['G']
df7['3B2'] = df7['3B'] / df7['G']
df7['HR2'] = df7['HR'] / df7['G']
df7['RBI2'] = df7['RBI'] / df7['G']
df7['BB2'] = df7['BB'] / df7['G']
df7['SO2'] = df7['SO'] / df7['G']
df7['SB2'] = df7['SB'] / df7['G']
df7['CS2'] = df7['CS'] / df7['G']

# Drop the original columns and rename the new columns

df7.drop(columns =['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'SB', 'CS'], axis=1, inplace=True)

df7.rename(columns={'AB2': 'AB', 'R2': 'R', 'H2': 'H', '2B2': '2B',
                    '3B2': '3B', 'HR2': 'HR', 'RBI2': 'RBI', 'BB2': 'BB',
                    'SO2': 'SO', 'SB2': 'SB', 'CS2': 'CS'}, inplace=True)

# Pull mlb team stats from 2022

df8 = pd.read_csv("C:/Users/trade/DSC450/team_stats_2022.csv")

# Create a new dataframe to hold just the Team , POST_SEASON and PCT columns.  This will be merged next

team2022 = df8[['TEAM', 'POST_SEASON','PCT']]

# Merge

df9 = team2022.merge(df7,indicator=False,how='outer')

# Seperate the descriptive columns since this a regression model

# We will also include Runs in the descriptive column since Runs are the target in the model

desc_col = df9[['TEAM', 'POST_SEASON', 'PCT', 'PLAYER', 'POSITION', 'R']]

num_col = df9.drop(columns=['TEAM', 'POST_SEASON', 'PCT', 'PLAYER', 'POSITION', 'R'])

# Run the prediction from the model

pred = pd.DataFrame(GS.predict(num_col))

pred.rename(columns={0: 'Pred_Runs'}, inplace=True)

# concatenating df3 and df4 along columns

player_rank = pd.concat([pred, desc_col, num_col], axis=1)

# Print the top 5 and bottom 5 predicted players

player_rank = player_rank.sort_values(by=['Pred_Runs'], ascending=False, ignore_index=True)

player_rank.rename(columns={'R': 'Actual_Runs'}, inplace=True)

print("{}Model Validation{}".format('\033[1m', '\033[0m'))
print("{}Top 5 2022{}".format('\033[1m', '\033[0m'))
print("{}Player Runs Per Game Predicted vs. Actual{}".format('\033[1m', '\033[0m'))

# plotting correlation heatmap
dataplot=sb.heatmap(player_rank.corr())
  
# displaying heatmap
plt.title("Player Predicted Runs Heatmap")
plt.show()

```

# RESULTS
We first used the 2021 Team stats dataset to understand what target we might want to use for predicting players.  We found that Runs scored are an important factor in a team’s winning percentage.  We used a regression plot to help show how a team’s winning percentage is closely correlated with how many runs are scored.
Next, we trained and tested a model using the 2021 Player stats dataset.  The XGB Regressor model is a supervised method and we used Runs as our target.  We also employed a Grid Search model that completed over 600 different tests to find the best parameters for the XGB Regressor.  We achieved a 90% accuracy score, which we felt was a good score to move forward with the model.  
Finally, we made predictions using the 2022 team and player datasets.  We were able to validate that the model is predicting some of the best offensive players in the MLB.  We complete this with a Heat Map to show which statistics the model found to be important as it correlates to Runs scored by each player.

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/e9890f79-4e0d-42f7-91a6-32eeb09e7e67)

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/1ce0077f-99dc-47a1-9243-cf1f4720e316)

![image](https://github.com/trade4huskers/trade4huskers.github.io/assets/52306793/13058ff0-1286-41e6-9469-6ec01c9aa93f)

# RECOMMENDATIONS/ETHICAL CONSIDERATIONS
The topic of predicting baseball players should be given with care depending on the stakeholder and reason for making predictions.  While the data used to make these predictions is found on public websites and readily available, it is still human beings that we are making predictions on.  That should not be taken lightly, and in some cases could mean their livelihoods and how they put food on the table.  With that in mind, these predictions should be considered carefully, and we recommend this study should be combined with other models and considerations before using this prediction for monetary or job reasons.

# CONCLUSION
For project 2, we as a team used 4 different MLB datasets pulled from ESPN.com and MLB.com and applied a machine learning model to answer stakeholder questions of predicting the best offensive baseball players.  We use Team statistics to help us find a target of Runs.  We then can apply Runs per game as our target to the player dataset using a machine learning model called Xgboost Regressor, achieving a 90% accuracy score.  We train and test using the 2021 datasets, and then predict using the 2022 datasets.  Our final predictions are successfully validated by comparing runs per game to predicted runs per game.

# REFERENCES
2021 baseball standings. MLB.com. (n.d.). Retrieved May 2, 2023, from https://www.mlb.com/standings/2021<br><br>
2022 baseball standings. MLB.com. (n.d.). Retrieved May 2, 2023, from https://www.mlb.com/standings/2022<br><br>
ESPN Internet Ventures. (n.d.). 2021 MLB player Batting Stats. ESPN. Retrieved May 2, 2023, from https://www.espn.com/mlb/stats/player/_/season/2021/seasontype/2 <br><br>
ESPN Internet Ventures. (n.d.). 2021 MLB team batting stats. ESPN. Retrieved May 2, 2023, from https://www.espn.com/mlb/stats/team/_/season/2021/seasontype/2 <br><br>
ESPN Internet Ventures. (n.d.). 2022 MLB player batting stats. ESPN. Retrieved May 2, 2023, from https://www.espn.com/mlb/stats/player/_/season/2022/seasontype/2 <br><br>
ESPN Internet Ventures. (n.d.). 2022 MLB team batting postseason stats. ESPN. Retrieved May 2, 2023, from https://www.espn.com/mlb/stats/team/_/season/2022/seasontype/3 <br><br>
Goodwin, K. (2021, July 19). A stakeholder interview checklist. Boxes and Arrows. Retrieved May 6, 2023, from https://boxesandarrows.com/a-stakeholder-interview-checklist/ <br><br>
How to know if your machine learning model has good performance: Obviously ai. Data Science without Code. (n.d.). Retrieved May 2, 2023, from https://www.obviously.ai/post/machine-learning-model-performance#:~:text=Good%20accuracy%20in%20machine%20learning,also%20consistent%20with%20industry%20standards.<br><br>
Python API reference. Python API Reference - xgboost 1.7.5 documentation. (n.d.). Retrieved May 5, 2023, from https://xgboost.readthedocs.io/en/stable/python/python_api.html <br><br>
Sklearn.model_selection.GRIDSEARCHCV. scikit. (n.d.). Retrieved May 5, 2023, from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html <br><br>
Yin, T. (n.d.). Successful data science projects: What questions to ask? LinkedIn. Retrieved May 2, 2023, from https://www.linkedin.com/pulse/successful-data-science-projects-what-questions-ask-tiancheng-yin 

# Video Link

For more details about our project, please check out our video!
<br/>
[Project 2 - MLB Player Prediction](https://www.youtube.com/watch?v=oG7GaFo-j4U)
