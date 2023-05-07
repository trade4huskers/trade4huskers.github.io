---
title: "Amazon and Yelp Reviews Using Sentiment Analysis"
layout: post
---
For Project 1, we have chosen to utilize Yelp and Amazon review data from Kaggle that will be used to analyze the sentiment of the review responses to determine whether the review is overall positive neutral or negative. Our overall business problem is to determine if customers of restaurants and purchased merchandise are more likely to post a review over a positive experience or a negative experience. Overall, between the Amazon and Yelp data, we found the majority of people who post restaurant and merchandise reviews are doing so because they had an overall positive experience.


# Project 1
## AMAZON AND YELP REVIEWS USING SENTIMENT ANALYSIS
DSC450 Applied Data Science
Bellevue University
Wrangler: Jeff Thomas
Scientist: Jerock Kalala
Presenter: Victoria Sukar

# INTRODUCTION
For Project 1, we have chosen to utilize Yelp and Amazon review data from Kaggle that will be used to analyze the 
sentiment of the review responses to determine whether the review is overall positive neutral or negative. Our 
overall business problem is to determine if customers of restaurants and purchased merchandise are more likely to 
post a review over a positive experience or a negative experience. Overall, between the Amazon and Yelp data, we 
found the majority of people who post restaurant and merchandise reviews are doing so because they had an overall 
positive experience. 
# BUSINESS PROBLEM/HYPOTHESIS
Determine if customers of restaurants and purchased merchandise are more likely to post a review over a positive 
experience or a negative experience. This could allow businesses to begin offering incentives for their customers to 
leave reviews regardless of their experience to get a more well-rounded review base. 
# METHODS/ANALYSIS
We will begin by collecting our datasets, wrangling and cleaning the data (formatting appropriately in Excel and 
Jupyter Notebook) using methods such as removing stopwords, standardizing alpha case (either all lower or all upper 
case), and removing punctuation which isn’t necessary for analysis of the English words being analyzed for 
sentiment. We will use modules such as Regex Tokenizer, WordNetLemmatizer and Vader Lexicon. Lastly, we will 
use Seaborn or another library to visualize the results to get a general idea of whether the reviews are more positive, 
neutral or negative as a whole.

Clean/wrangle the various datasets 
- Remove stopwords 
- Convert to lowercase 
- Remove punctuation 

RegexpTokenizer

Tokenize the datasets 

Create a list of all words (string) 

Create a frequency distribution list 

- nltk.probability import FreqDist 

Lemmatization 
- nltk.download(‘wordnet’) 
- from nltk.stem import WordNetLemmatizer 

Vader 
- nltk.download(‘vader_lexicon’) 
- from nltk.sentiment import SentimentIntensityAnalyzer 
- Change the data structure and add a scoring system 
- Create a new variable with sentiment types (positive, neutral, negative) 

Analyze the data 
- Determine the highest positive sentiment review 
- Determine the highest negative sentiment review 
- Visualize the data results using something like seaborn

# RESULTS
After utilizing both the Amazon fine food reviews and Yelp data, we found that the majority of individuals that post 
reviews do so because they had a positive experience. We believe this is far more common practice because if you 
are impressed with the quality of service at a restaurant or love a product you just purchased, you are more prone 
to tell the world about this product to entice more people to go out and support that restaurant or small business 
or to purchase the product van if you had a negative experience, we would assume you are more likely to address 
the issue directly with the establishment or the product provider to find a resolution. 

![image](https://user-images.githubusercontent.com/52306793/236659630-0e958906-6ec7-4f20-9aab-ff69d9934ef3.png)

![image](https://user-images.githubusercontent.com/52306793/236659684-5eea1a93-9233-4993-a5d0-c45af9970c24.png)

Additionally, without telling a Facebook audience our sentiment analysis results, we conducted an anonymous poll 
to see if our results aligned and they did (Due to the free polling program used, we are unable to determine how 
many individuals participated). Most individuals indicated that they post a review on a website after a positive 
experience.

# RECOMMENDATIONS/ETHICAL CONSIDERATIONS
Sentiment analysis is still viewed as a touchy subject in the ethics department of the data science world due to not 
always knowing how emotions will present in the data. Also, things like unacceptable levels of bias and unwitting 
data collection that help build the sentiment models. (Minty, 2022) 
Overall, we believe that sentiment analysis in this current climate still requires human interaction and interference. 
Sentiment analysis models can certainly assist in viewing the broad picture of topics, relations, business schemes 
and more but this is far from a standalone methodology at this point. 

# Video Link
For more details about our project, please check out our video!
<br/>
[Project 1 - Sentiment Analysis](https://www.youtube.com/watch?v=K7zAx8Q-d3Q)



