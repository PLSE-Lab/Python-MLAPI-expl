#!/usr/bin/env python
# coding: utf-8

# # **Sentiment analysis: Trump's tweets are mostly positive or negative?**
# 
# <img src="https://i.imgur.com/BoAqMXw.jpg">
# 
# # Table of contents
# 
# [<h3>1. Presentation of the data</h3>](#1)
# 
# [<h3>2. Quantity of tweets</h3>](#2)
# 
# [<h3>3. Are Trump's tweets mostly positive or negative?</h3>](#3)
# 
# [<h3>4. Number of retweets/favorites</h3>](#4)
# 
# [<h3>5. Does the positivity of a tweet influence the quantity of retweets/favorites?</h3>](#5)

# # 1. Presentation of the data<a class="anchor" id="1"></a>
# 
# Around 42.000 Tweets from Donald Trump between 2009 and 2020.
# 
# <strong><u>Data Content: </u></strong><br>
# <br><br>- <strong>id: </strong> Unique tweet id
# <br><br>- <strong>link: </strong>Link to tweet
# <br><br>- <strong>content: </strong>Text of tweet
# <br><br>- <strong>date: </strong>Date of tweet
# <br><br>- <strong>retweets: </strong>Number of retweets
# <br><br>- <strong>favorites: </strong>Number of favorites
# <br><br>- <strong>mentions: </strong>Accounts mentioned in tweet

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/trump-tweets/realdonaldtrump.csv")
df.head(5)


# In[ ]:


# Select only the columns, which will be used
df = df.drop(["id","link"], axis = 1)

df.describe()


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 15)
plt.show()


# # 2. Quantity of tweets<a class="anchor" id="2"></a>

# In[ ]:


df["date"] = pd.to_datetime(df["date"])
df["date"].apply(lambda x: x.year)

# Number of tweets by year
colors = []
for i in range(2020-2009+1):
    x = 0.7-0.06*i
    c = (x,x,0.5)
    colors.append(c)

bar = df["date"].apply(lambda x: x.year).value_counts().sort_index().plot.bar(figsize = (16,10), color = colors)
plt.title("Number of tweets by year\n", fontsize=20)
bar.tick_params(labelsize=14)
plt.axvline(8, 0 ,1, color = "grey", lw = 3)
plt.text(7.7, 8800, "President", fontsize = 18, color = "grey")
bar.tick_params(labelsize=18)
plt.show()

# Number of tweets (more details)
df["year_month"] = df["date"].apply(lambda x: str(x.year)+"-"+str(x.month))
df["year_month"] = pd.to_datetime(df["year_month"])
year_month = pd.pivot_table(df, values = "content", index = "year_month", aggfunc = "count")

bar = year_month.plot(figsize = (16,10))
plt.title("Number of tweets (more details)", fontsize=20)
plt.axvline(8, 0 ,1, color = "grey", lw = 3)
bar.tick_params(labelsize=18)
plt.legend("")
plt.xlabel("")
bar.get_yaxis().set_visible(False)
plt.show()


# The first year of his election, he was less active on Twitter but in the following years he increased his activity again.

# # 3. Are Trump's tweets mostly positive or negative?<a class="anchor" id="3"></a>
# 

# In[ ]:


# Calculate the polarity of the tweets of Trump with NLTK
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

df["polarity"] = df["content"].apply(lambda x: sentiment.polarity_scores(x))

df["pos"] = df["polarity"].apply(lambda x: x["pos"])
df["neg"] = df["polarity"].apply(lambda x: x["neg"])
df["compound"] = df["polarity"].apply(lambda x: x["compound"])

# Create the visualization
fig = plt.figure(figsize = (14,10))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Mean positivity/negativity of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Positivity plot
year_month = pd.pivot_table(df, values = "pos", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Negativity plot
year_month = pd.pivot_table(df, values = "neg", index = "year_month", aggfunc = "mean").apply(lambda x: -x)
ax.plot(year_month, lw = 5, color = "red")

# Add the "president" and "corona" lines
ax.legend(["pos","neg"], fontsize=18)
plt.axhline(0, 0 ,1, color = "black", lw = 1)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -0.18, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 0.39, "Corona", fontsize = 18, color = "orange")
ax.tick_params(labelsize=18)
plt.show()


# In[ ]:


# Create the visualization
fig = plt.figure(figsize = (14,6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Compound positivity/negativity of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Compound plot
df["year_month"] = df["date"].apply(lambda x: str(x.year)+"-"+str(x.month))
df["year_month"] = pd.to_datetime(df["year_month"])
year_month = pd.pivot_table(df, values = "compound", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5, color = "green")


# Add the "president" and "corona" lines
ax.legend(["pos","neg"], fontsize=18)
plt.axhline(0, 0 ,1, color = "black", lw = 1)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -0.18, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 0.7, "Corona", fontsize = 18, color = "orange")
plt.legend("")
plt.show()


# The tweets of Trump are in average quite positive beside of the end of the year 2011. Since he has been elected, no real change in the compound positivity/negativity can be seen.
# 
# I searched on Wikipedia to see if something special happened to him in the year 2011 and this is what I found:
# 
# Trump speculated about running for president in the 2012 election, making his first speaking appearance at the Conservative Political Action Conference (CPAC) in February 2011 and giving speeches in early primary states. In May 2011 he announced that he would not run.
# 
# Trump's presidential ambitions were generally not taken seriously at the time. Before the 2016 election, The New York Times speculated that Trump "accelerated his ferocious efforts to gain stature within the political world" after Obama lampooned him at the White House Correspondents' Association Dinner in April 2011.
# 
# In 2011 the then-superintendent of the New York Military Academy, Jeffrey Coverdale, ordered the then-headmaster of the school, Evan Jones, to give him Trump's academic records so that he could keep them secret, according to Jones. Coverdale said he had been asked to add to hand the records over to members of the school's board of trustees who were Mr. Trump's friends, but he refused to give the records to anyone and instead sealed Trump's records on campus. The incident reportedly happened days after Trump demanded the release of President Barack Obama's academic records.
# 
# <strong>Trump launched his political career in 2011 as a leading proponent of "birther" conspiracy theories alleging that Barack Obama, the first black U.S. president, was not born in the United States.</strong> In April 2011, Trump claimed credit for pressuring the White House to publish the "long-form" birth certificate, which he considered fraudulent, and later saying this made him "very popular". In September 2016, he acknowledged that Obama was born in the U.S. and falsely claimed that the rumors had been started by Hillary Clinton during her 2008 presidential campaign. <a href = "https://en.wikipedia.org/wiki/Donald_Trump">source</a>
# 
# 

# # 4. Number of retweets/favorites<a class="anchor" id="4"></a>

# In[ ]:


# Create the visualization
fig = plt.figure(figsize = (14,6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Quantity of retweets and favorites of Trump's tweets", fontsize=24)
ax.tick_params(labelsize=14)

# Monthly Average number of "retweets"
year_month = pd.pivot_table(df, values = "retweets", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Monthly Average number of "favorites"
year_month = pd.pivot_table(df, values = "favorites", index = "year_month", aggfunc = "mean")
ax.plot(year_month, lw = 5)

# Add the "president" and "corona" lines
ax.legend(["Retweets","Favorites"], fontsize=18)
plt.axvline("20-01-2017", 0 ,1, color = "grey", lw = 3)
plt.text("8-12-2016", -13000, "President", fontsize = 18, color = "grey")
plt.axvline("20-01-2020", 0 ,1, color = "orange", lw = 3)
plt.text("9-12-2019", 140000, "Corona", fontsize = 18, color = "orange")
ax.tick_params(labelsize=18)
plt.show()


# # 5. Does the positivity of a tweet influence the quantity of retweets/favorites?<a class="anchor" id="5"></a>
# 
# I wonder if a positive or a negative tweet is more likely to be retweets or added to favorite. Looking at the number of retweets/favorites is the easiest way to see if people are interested in what Trump posted on Twitter.

# In[ ]:


plt.figure(figsize = (10,8))
sns.heatmap(df[["retweets","favorites", "compound"]].corr(), annot = True, cmap="YlGnBu")
plt.title("Correlation between retweets, favorites and compound\n", fontsize = 14)
plt.show()


# There is a high correlation between retweets and favorites, but the correlation between compound and the two others is quite low. Let's visualize it in the scatterplot.

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot("compound","retweets", data = df, alpha = 0.5)
plt.title("Relation between retweets and compound (positivity/neg.)", fontsize = 15)
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot("compound","retweets", data = df, alpha = 0.01)
plt.title("Relation between retweets and compound (positivity/neg.)", fontsize = 15)
plt.show()


# The positivity of a tweet doesn't seem to really influence the quantity of retweets/favorites.
