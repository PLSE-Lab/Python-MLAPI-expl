#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosting to predict hotel's reviews

# <img src="https://i.imgur.com/gK2gwpD.jpg">

# # Table of contents
# 
# [<h3>1. Presentation of the data</h3>](#1)
# 
# [<h3>2. Explorative analysis</h3>](#2)
# 
# [<h3>3. Prediction of the review score with supervised learning</h3>](#3)

# # 1. Presentation of the data<a class="anchor" id="1"></a>
# 
# <strong><u>Data Context:</u></strong><br>
# This dataset contains 515,000 customer reviews and scoring of 1493 luxury hotels across Europe. Meanwhile, the geographical location of hotels are also provided for further analysis.
# 
# <strong><u>Data Content:</u></strong>
# The csv file contains 17 fields. The description of each field is as below:
# <br>- <strong>Hotel_Address:</strong>  Address of hotel.
# <br><br>- <strong>Review_Date:</strong>  Date when reviewer posted the corresponding review.
# <br><br>- <strong>Average_Score:</strong>  Average Score of the hotel, calculated based on the latest comment in the last year.
# <br><br>- <strong>Hotel_Name:</strong>  Name of Hotel
# <br><br>- <strong>Reviewer_Nationality:</strong>  Nationality of Reviewer
# <br><br>- <strong>Negative_Review:</strong>  Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
# <br><br>- <strong>Review_Total_Negative_Word_Counts:</strong>  Total number of words in the negative review.
# <br><br>- <strong>Positive_Review:</strong>  Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'
# <br><br>- <strong>Review_Total_Positive_Word_Counts:</strong>  Total number of words in the positive review.
# <br><br>- <strong>Reviewer_Score:</strong>  Score the reviewer has given to the hotel, based on his/her experience
# <br><br>- <strong>Total_Number_of_Reviews_Reviewer_Has_Given:</strong>  Number of Reviews the reviewers has given in the past.
# <br><br>- <strong>Total_Number_of_Reviews:</strong>  Total number of valid reviews the hotel has.
# <br><br>- <strong>Tags:</strong>  Tags reviewer gave the hotel.
# <br><br>- <strong>days_since_review:</strong>  Duration between the review date and scrape date.
# <br><br>- <strong>Additional_Number_of_Scoring:</strong>  There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there.
# <br><br>- <strong>lat:</strong>  Latitude of the hotel
# <br><br>- <strong>lng:</strong>  longitude of the hotel

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv("/kaggle/input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv")

df.head(5)


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 15)
plt.show()


# Beside of a small quantity of missing values for the latitude and longitude, the dataset doesn't have missing values.

# In[ ]:


nb_missing = df[df["lat"].isnull() & df["lng"].isnull()].shape[0]
print(f"Number of reviews with no latitude or longitude: {nb_missing}\nTotal number of reviews: {df.shape[0]}")


# # 2. Explorative analysis<a class="anchor" id="2"></a>

# In[ ]:


print("Number of hotels:",df['Hotel_Name'].nunique())


# <strong> Location of the hotels:</strong>
# <img src="https://i.imgur.com/DWAh8Om.png">

# In[ ]:


# Create a column with the rounded reviews
df["Reviewer_Score_Round"] = df["Reviewer_Score"].apply(lambda x: int(round(x)))

# Get the number of reviews with which scores
reviews_dist = df["Reviewer_Score_Round"].value_counts().sort_index()
bar = reviews_dist.plot.bar(figsize =(10,7))
plt.title("Distribution of reviews", fontsize = 18)
plt.axvline(df["Reviewer_Score"].mean()-2, 0 ,1, color = "grey", lw = 3)
plt.text(6, -15000, "average", fontsize = 14, color = "grey")
plt.ylabel("Count", fontsize = 18)
bar.tick_params(labelsize=16)

# Remove the column "Reviewer_Score_Round"
df.drop("Reviewer_Score_Round", axis = 1, inplace = True)


# Most of the reviews are positives, therefore the average review score is 8.4/10. It is not similar to a normal distribution like we might expect. 

# ## Correlation:
# Let's see the correlation between the variables.

# In[ ]:


df_corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(df_corr, annot = True)
plt.title("Correlation between the variables", fontsize = 22)
plt.show()


# Considering the number of variables in the representation of the correlation, we will only represent the correlation with reviewer_score.

# In[ ]:


# Get the colors for the graphic
colors = []
dim = df_corr.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,r,0.3))

# Transform each value in a positive value, because what interesses us
# isn't the direction of the correlation but the absolute correlation
df_corr["Reviewer_Score"].apply(lambda x: abs(x)).sort_values().plot.barh(color = colors)
plt.title("Correlation with Reviewer_Score", fontsize = 16)
plt.show()


# ## Mean review score depending on the reviewer nationality

# In[ ]:


# Group the data by nationality
group_nationality = df.pivot_table(values = "Reviewer_Score", 
                                   index = "Reviewer_Nationality", 
                                   aggfunc=["mean","count"])
group_nationality.columns = ["mean_review","review_count"]


# In[ ]:


# Keep only the nationalities with at least 3000 reviews given
reviews_count=group_nationality[group_nationality["review_count"]>3000]["review_count"].sort_values(ascending = False)

# Get the colors for the graphic
colors = []
dim = reviews_count.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,1-r,0.3))

# Display the result
reviews_count.plot.barh(figsize=(10,10), color = colors)
plt.title("Number of reviews given by nationality", fontsize = 18)
plt.ylabel("")
plt.show()


# Most of the reviews were made by a few countries.

# In[ ]:


# Keep only the nationalities with at least 1000 reviews
group_nationality = group_nationality[group_nationality["review_count"] > 1000].sort_values(by = "mean_review", ascending = False)

# Get the colors for the graphic
colors = []
dim = group_nationality.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,1-r,0.3))

# Display the result
group_nationality["mean_review"].plot.barh(figsize = (10,20), color = colors)
plt.title("Who gives the worst review scores to hotels?", fontsize = 17)
plt.axvline(df["Reviewer_Score"].mean(), 0 ,1, color = "grey", lw = 3)
plt.text(8, 55, "average", fontsize = 14, c = "grey")
plt.text(8, -2, "average", fontsize = 14, c = "grey")
plt.xlabel("Average review score given", fontsize = 18)
plt.ylabel("")
plt.show()


# In average, the English speaking countries give the best reviews scores with USA as number 1.

# # 3. Prediction of the review score with supervised learning<a class="anchor" id="3"></a>
# In this part we will train different models to predict the scores of the reviews based on the others variables.
# Before we can build a model, some transformation is needed:
# 

# In[ ]:


# Convert the reviews to lower and delete leading/trailing space
df["Negative_Review"] = df["Negative_Review"].str.lower().str.strip()
df["Positive_Review"] = df["Positive_Review"].str.lower().str.strip()


# Each reviews text will be analysed with nltk. SentimentIntensityAnalyzer makes it possible to see if a text is positive or negative. For example:

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()

rev1 = "The hotel was very good, I love it!"
rev2 = "It was just horrible, the worst ever."

print(f"review 1:\n{rev1}\nScore: {sent_analyzer.polarity_scores(rev1)}")

print(f"\nreview 2:\n{rev2}\nScore: {sent_analyzer.polarity_scores(rev2)}")


# The compound is the general positivity of a text. Above 0, it is positive and under 0, it is negative. This can be used to determine if a text is positive or negative.
# 
# Create two columns, one column for the polarity_scores of the positive reviews and one column for the negative ones. It might happen that the polarity_scores isn't accurate and to avoid issues in the model, the polarity_score will be only >= 0 for the positive reviews and <= 0 for the negative reviews.

# In[ ]:


# Take only a part of the data to speed up
# df = df[:50000].copy()

start_time = time.time()
pos = df["Positive_Review"].apply(lambda x: abs(sent_analyzer.polarity_scores(x)["compound"]))
neg = df["Negative_Review"].apply(lambda x: -abs(sent_analyzer.polarity_scores(x)["compound"]))

df["sentiment_score"] = pos + neg
df["polarity_pos"] = pos
df["polarity_neg"] = neg

time_model = time.time() - start_time
print(f"Execution time: {int(time_model)} seconds")


# In[ ]:


df_corr = df.corr()

# Get the colors for the graphic
colors = []
dim = df_corr.shape[0]
for i in range(dim):
    r = i * 1/dim
    colors.append((0.3,r,0.3))

# Transform each value in a positive value, because what interesses us
# isn't the direction of the correlation but the absolute correlation
df_corr["Reviewer_Score"].apply(lambda x: abs(x)).sort_values().plot.barh(color = colors)
plt.title("Correlation with Reviewer_Score", fontsize = 16)
plt.show()


# The variable which influences the most the reviewer score is the length of the negative review. It is logical, because if someone really didn't like a hotel, he might write a lot about it.

# In[ ]:


# Columns to use to train the models
# Only the columns with the highest correlation were chosen
cols = ['Review_Total_Negative_Word_Counts',
        'polarity_pos',
        'Average_Score',
        'Review_Total_Positive_Word_Counts']
        
X = df[cols].values
y = df["Reviewer_Score"].values

# Use StandardScaler to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.metrics import mean_squared_error

def plot_res(y_test,pred, model = "LinearRegression"):
# Violinplots with the distribution of real scores and predicted scores

    MSRE = round((mean_squared_error(y_test,pred))**0.5,3)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
    
    sns.violinplot(y_test, ax = axes[0])
    axes[0].set_title("Distribution of\n scores")
    axes[0].set_xlim(0,11)
    
    sns.violinplot(pred, ax = axes[1])
    title = f"Predictions of scores with {model}\nMSRE:{MSRE}"
    axes[1].set_title(title)
    axes[1].set_xlim(0,11)
    plt.show()
    
# LinearRegression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "LinearRegression")

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "GradientBoostingRegressor")

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plot_res(y_test,pred, model = "RandomForestRegressor")


# The GradientBoostingRegressor gives the best result for the prediction
# 

# **Thank you and good night!**
# 
# <img src="https://i.imgur.com/r1y0aA3.png">
