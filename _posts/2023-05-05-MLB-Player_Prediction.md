---
title: "Predicting The Best MLB Offensive Players - Machine Learning"
layout: post
---

For Project 2, we have chosen to predict the best MLB (Major League Baseball) offensive players using data from ESPN.com.  We are working with four dataset in the form of csv files, broken down by team statistics, player statistics, and seasons (2021 & 2022).  We also used MLB.com to incorporate a feature called ‘POST_SEASON’ in our Team csv files.  We first look at team stats from 2021 to help us determine that Runs are an important part of a team’s winning percentage.  We then use Runs as our target to train the 2021 player dataset.  Once our machine learning model is built with an acceptable score, we will predict using the 2022 datasets.  Overall, our model achieved a 90% accuracy score, and we can validate that our predictions for how many runs a player will score per game in 2022 appear to be accurate.  Therefore, we feel that this model is successful in predicting some of the best offensive players in the game.


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




