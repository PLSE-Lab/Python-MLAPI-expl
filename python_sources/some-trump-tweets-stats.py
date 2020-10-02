#!/usr/bin/env python
# coding: utf-8

# Hello everyone, today we're going to analyze Trump's tweets ! My main questions are:
# 
# - What are his most popular tweets ?
# - How popular its tweets became over the years ?
# - Is is possible to predict a tweet's number of retweets ?
# - What is the distribution of the number of retweets ?
# - What are the most common words used in his tweets ?
# 
# First, you have to know, I decided that a popular tweet was a tweet with a lot of retweets. It may be wrong or approximative, but this is how I will analyze the data for the rest of the kernel. Let's go deep in the dataset !

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import datetime

get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 7)

df = pd.read_csv("../input/trump-tweets/trumptweets.csv", sep=",", engine="python")
df['date'] = pd.to_datetime(df['date']) # So pandas understands it's a date

number_of_days = (df["date"].max() - df["date"].min()).days
print("Amount of tweets: {}".format(len(df)))
print("Published in {} days, which makes {:.2f} tweets per day".format(number_of_days, len(df)/number_of_days))
df.head()


# It seems like there are a lot of NaN values here, let's check.

# In[ ]:


print("Number of NaN values")
df.isna().sum()


# We can see that geo, hashtags and mentions are filled with a lot of NaN values (especially the geo column, that have 100% of NaN values !). So let's get rid of those columns, among with the link column, that I know I won't use.

# In[ ]:


df = df.drop(["geo", "hashtags", "mentions", "link"], axis=1) # Not so much to deal with


# Let's see top 10 Trump's best tweets.

# In[ ]:


max_retweets = df.sort_values("retweets", ascending=False)
content, date, retweets, favorites = 1, 2, 3, 4
place = 1

for tweet in max_retweets.values[:10]:
    print("\033[94mBest retweet {}:\033[0m \n{}\n{}\t\033[93m{} retweets\t\t{} favorites\033[0m\n".format(
        place, tweet[content], tweet[date], tweet[retweets], tweet[favorites]))
    place += 1


# Here we can see that we're going to have a problem with pictures, so when we'll analyze the content of each tweets, it is going to be a parasite. We can also see that even in the top 10 tweets, the number of retweets are very disparate. Moreover, some tweets are really olds ! And it looks like his best tweets are the old ones also.
# 
# Anyway, let's look at the tweets' distribution among the years. Let's keep in mind that my dataset has been taken in January 2020.

# In[ ]:


plt.subplot(2, 2, 1)
plt.title("Number of Tweets per Year")
df["date"].groupby(df["date"].dt.year).count().plot(kind="bar")
plt.xlabel("year")

plt.subplot(2, 2, 2)
plt.title("Number of Tweets per Day")
tweet_cpt = df["date"].groupby(df["date"].dt.day_name()).count()
tweet_cpt.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).plot(kind="bar")
plt.xlabel("year")
plt.show()


# In his best shape, Mister Trump posted ~8000 tweets ! That makes more than 22 tweets a day !
# 
# Ok so after some reflexion, I decided to analyze his tweets only since a particular date. Since his campaign for running president was in 2015, I thought about comparing his stats before and after 2015. I also want to see the stats since 2013 (when he really started to post a lot of tweets), and since 2017, because I want to look at the fresh data. When looking at the histogram, retweets' distribution are really not doing us a favour. So I use a logarithmic scale to have a better look at the distribution.

# In[ ]:


after_campaign = df[df["date"] >= np.datetime64("2015-01-01")]
before_campaign = df[df["date"] < np.datetime64("2015-01-01")]
after_2013 = df[df["date"] >= np.datetime64("2013-01-01")]
past_years = df[df["date"] >= np.datetime64("2017-01-01")]


# In[ ]:


X1 = np.log(after_campaign["retweets"]+1) # +1 to ensure values != 0
X2 = np.log(before_campaign["retweets"]+1)
plt.hist(X1, bins=20, alpha=0.5, label="After 2015")
plt.hist(X2, bins=20, alpha=0.5, label="Before 2015")

plt.title("Retweets Histogram")
plt.xlabel("# of Retweets (logarithmic scale)")
plt.ylabel("# of Rows")
plt.legend(loc='upper left')
plt.show()


# In[ ]:


after_2013["retweets"].describe()


# In[ ]:


after_campaign["retweets"].describe() # 2015 and +


# In[ ]:


past_years["retweets"].describe() # 2017 and +


# Okay ! So as expected, standard deviation is huge. And when looking at the median number of retweets, we can see the number has grown a lot along the years. It not surprising since he became president and more and more controversial, but still. So, to decide which part of the data we're going to analyze, I thought about plotting the median number of retweets, years by years. Indeed, I feel like since his position has changed over the years, the "retweeting potential" has also changed. If we want to predict the number of retweets, we'll have to select the time where tweets have the same "retweeting potential".

# In[ ]:


df["retweets"].groupby(df["date"].dt.year).median().plot()
plt.title("Median Number of Retweets")
plt.xlabel("year")
plt.plot()


# So, it seems that after 2017, tweets can unified under the same "potential" (I honestly don't know what I am doing right now). I will now use the "past_years" dataframe. Let's have a look at how much there is deviation between the dataset.

# In[ ]:


plt.plot(range(0, len(past_years["retweets"])), sorted(np.log(past_years["retweets"])))
plt.title("Sorted # of Retweets")
plt.ylabel("# of Retweets (logarithmic scale)")
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
plt.show()


# Ok so, we have here a regression objective, to make a prediction, I use a CountVectorizer that select words, and transforms the content of a tweet into a matrix saying if each selected words appears.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

train = past_years.sample(frac=0.8,random_state=200)
test = past_years.drop(train.index)
y_train = train["retweets"]
y_test = test["retweets"]


# In[ ]:


def draw_predicted(model, X_test, y_test):
    """
    Draw the model's prediction, and superpose it with the
    expected values.
    """
    predicts = model.predict(X_test)
    df = pd.DataFrame({"predicted":predicts, "actual":y_test})
    df = df.sort_values("actual")
    plt.plot(range(0, len(df)), df["actual"], label="Actual", color='r', ls='dotted')
    plt.plot(range(0, len(df)), df["predicted"], label="Predicted", color='b', ls='dotted', alpha=0.3)
    plt.title("Model evaluation")
    plt.ylabel("# of Retweets")
    plt.legend(loc='upper left')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.show()


# In[ ]:


vect = CountVectorizer(min_df=5, stop_words="english")
X_train = vect.fit_transform(train["content"])
X_test = vect.transform(test["content"])

model = LinearRegression()
model.fit(X_train, y_train)
print("Score: {:.2f}".format(model.score(X_test, y_test)))
draw_predicted(model, X_test, y_test)


# Predictions are not really good (and I'm not gonna lie, it's not going to stop). I fear that the fact that some tweets are so special, with too much retweets, or not enought, the model has real issues learning. For the rest of the kernel, we're going to keep only the tweets that have a logarithmic value between 8 and 11 (we need to keep some deviation so there is something to predict).
# 
# For the rest of the predictions, we're going to use LinearRegression, DecisionTreeRegressor and RandomForestRegressor.

# In[ ]:


common_tweets = past_years[8 < np.log(past_years["retweets"])]
common_tweets = common_tweets[np.log(common_tweets["retweets"]) < 11]

plt.plot(range(0, len(common_tweets["retweets"])), sorted(np.log(common_tweets["retweets"])))
plt.title("Sorted # of Retweets")
plt.ylabel("# of Retweets (logarithmic scale)")
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
plt.show()

train = common_tweets.sample(frac=0.8,random_state=200)
test = common_tweets.drop(train.index)
y_train = train["retweets"]
y_test = test["retweets"]


# In[ ]:


vect = CountVectorizer(min_df=5, stop_words="english")
X_train = vect.fit_transform(train["content"])
X_test = vect.transform(test["content"])

model = LinearRegression()
model.fit(X_train, y_train)
print("Score: {:.2f}".format(model.score(X_test, y_test)))
draw_predicted(model, X_test, y_test)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

vect = CountVectorizer(min_df=5, stop_words="english")
X_train = vect.fit_transform(train["content"])
X_test = vect.transform(test["content"])

tree = DecisionTreeRegressor(max_depth=5, random_state=180)
tree.fit(X_train, y_train)
print("Score: {:.2f}".format(tree.score(X_test, y_test)))
draw_predicted(model, X_test, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

vect = CountVectorizer(min_df=5, stop_words="english")
X_train = vect.fit_transform(train["content"])
X_test = vect.transform(test["content"])

tree = RandomForestRegressor(n_estimators=10, max_depth=100, random_state=180)
tree.fit(X_train, y_train)
print("Score: {:.2f}".format(tree.score(X_test, y_test)))
draw_predicted(model, X_test, y_test)


# Ok well, it is with a big disappointment that I tell you, we did not make it. Predicting number of retweets is too hard for us (me at least). I guess thre are no particular words that makes a tweet more "retweetable" than others (but the context and the sementics must help I imagine, that's not completly random).
# 
# But I won't leave you without anything ! I still have made a WordCloud of the top words used by Trump.

# In[ ]:


from wordcloud import WordCloud

vect = CountVectorizer(min_df=100, stop_words="english")
vect.fit(past_years["content"])
wc = WordCloud(width=1920, height=1080).generate_from_frequencies(vect.vocabulary_)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# Well that's all for me ! I hope you enjoyed the trip ! Have a good day, and take care of yourself :).
# 
# PS : I think that if there is no 'new' in the WordCloud, that because 'new' must be a stopword for the CountVectorizer.
