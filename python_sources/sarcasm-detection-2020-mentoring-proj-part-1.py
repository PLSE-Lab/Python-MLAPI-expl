#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this project, we will examine a dataset comprised of comments from Reddit, some of which are sarcastic and some of which are not. This project is meant to be a basic introduction into machine learning methods and to help one understand the steps of a data science project. We will investigate the dataset and its features, train a basic machine learning model, and train a more complex NLP-based (Natural Language Processing) machine learning model.
# 
# Series:
# 1. Part 1: Exploring Data and Feature Engineering
# 2. [Part 2](https://www.kaggle.com/yastapova/sarcasm-detection-2020-mentoring-proj-part-2): Splitting Data and Building a Basic Machine Learning Model
# 3. [Part 3](https://www.kaggle.com/yastapova/sarcasm-detection-2020-mentoring-proj-part-3): Building a Text-Based Machine Learning Model
# 
# ## Goal
# The goal of the models is to detect sarcasm in Reddit comments. Therefore, we will be utilizing **Supervised Learning** for **Classification**. The resulting model will be a binary classifier because we have two classes: a label of 1 denotes a sarcastic comment and a label of 0 denotes a non-sarcastic comment.
# 
# ## Background
# A good introduction to the fundamentals of machine learning can be found in [this article](https://towardsdatascience.com/machine-learning-basics-part-1-a36d38c7916) by Javaid Nabi. I have also found a textbook by Dan Jurafsky and James H. Martin, called *Speech and Language Processing*, to be helpful when learning Natural Language Processing techniques. Currenlty, the [introduction](https://www.cs.colorado.edu/~martin/SLP/Updates/1.pdf) to the second edition can be found online, as well as the authors' [drafts](https://web.stanford.edu/~jurafsky/slp3/) of other chapters in the new 3rd edition.
# 
# ## Data
# The data was gathered by Mikhail Khodak, Nikunj Saunshi, and Kiran Vodrahalli for their paper "*[A Large Self-Annotated Corpus for Sarcasm](https://arxiv.org/abs/1704.05579)*" and is hosted [here](https://nlp.cs.princeton.edu/SARC/0.0/).
# 
# Citation:
# ```
# @unpublished{SARC,
#   authors={Mikhail Khodak and Nikunj Saunshi and Kiran Vodrahalli},
#   title={A Large Self-Annotated Corpus for Sarcasm},
#   url={https://arxiv.org/abs/1704.05579},
#   year=2017
# }
# ```
# 
# Below is a code block which contains Python libraries we will use in this project and which also lists the data files attached to this notebook. This code should be run before any other blocks.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualizations and charts

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Step 1: Examine the Dataset
# 
# A data science or machine learning project can only be as good as the data it's fed, so the first step of any project is to examine the dataset. We'll use the *train-balanced-sarcasm.csv* file provided. The next step that goes hand-in-hand with examining is cleaning the dataset, but since this one is already pre-cleaned we won't have to worry about that.
# 
# What is the goal of examining the data?
# 1. Understand what the features represent
# 2. Observe what type of data you are working with (structured vs. unstructured? text, numbers, dates, categories, etc?)
# 3. Determine if there are missing values
# 4. Identify any anomalies or outliers
# 5. Examine the target variable in particular and check if the classes are balanced or imbalanced
# 
# A good first step before we start taking averages and medians is to look at the unaltered dataset with your own eyes, just to see what it looks like and see if you can notice anything interesting immediately. To do so, let's load the CSV file into a dataframe and look at the top few rows.

# In[ ]:


data = pd.read_csv("/kaggle/input/sarcasm/train-balanced-sarcasm.csv")
data.head(10)


# I can immediately see that a large number of features contain text while other features contain numbers and datestamps. Based on some experience with Reddit, I can also see a few points of interest. I know that comments can have a score determined by how many people upvote or downvote the comment. I assume that is the meaning of the *score* column, and that *ups* and *downs* correspond to upvotes and downvotes, respectively.
# 
# Thus, I already have a question: how can one have a negative number of upvotes or downvotes? I would expect these numbers to be non-negative. I can also see that the sum of ups and downs does not always equal the score, which I would expect to be true. Finally, I see that many rows in this selection have -1 for both ups and downs, which may be a way to denote missing values. All in all, this tells me that these columns require more scrutiny and/or research.
# 
# Another point of interest is that *author* and *subreddit* both contain repeated values, which makes sense. Finally, I see that there are two date columns: *data* (which contains only year and month) and *created_utc* (which contains a full date and time stamp in what I assume is UTC time).
# 
# I also see that each row contains two long text fields: *comment* (which is the comment we are detecting sarcasm in) and *parent_comment* (which is the post to which our comment of interest is a reply).
# 
# The target variable *label* does not look interesting in this view, since I can only see 0s. However, I know that it is simply a binary column, so there isn't that much to learn about it at this stage.
# 
# Let's start by determining if we have any empty, null, or NaN data.

# In[ ]:


data.isna().sum(axis=0)


# Looks like 53 of the comments are blank. Since we'll be basing this project on text models, these will not be useful to us. Let's remove them before we continue.

# In[ ]:


empty_comments = data["comment"].isna()
empty_comments = data[empty_comments].index
data.drop(empty_comments, axis=0, inplace=True)


# Next, let's take a look at some of the comments and pull out a few examples of sarcastic ones. Here are the first 10 sarcastic comments, just so we can get a glimpse of what we're talking about:

# In[ ]:


for c in data[data["label"] == 1]["comment"][0:10]:
    print(c)


# Now that we have a clearer picture of what we're dealing with, let's continue by taking a glance at some column statistics.

# In[ ]:


data.describe()


# The *describe()* function only works on numeric columns unless you explicitly tell it to look at others. Here we can see some basic statistics on our score, ups, and downs features. We can see that the maximum values of score and ups are very high, but the maximum of downs is only 0. Likewise, the minimum values for score and ups get very low to -507, but the minimum for downs is only -1.
# 
# Curiously, we can also see that the maximum of score does not equal the maximum of ups. From the 25th and 75th percentile values, we can determine that half of the data is at or between the values of 0 and 4. Since the minimum and maximum are quite far from this range, this means that the distribution of the scores is likely to be very skewed. We can examine it in greater detail when we make charts in the next step.
# 
# The label column is also included and you can see from the mean of 0.5 that it is most likely balanced between the classes of 0 and 1. We can check this explicitly later.
# 
# Let's continue by looking in greater detail at the ups and downs to investigate whether -1s really do make up a large segment of the data.

# In[ ]:


data["downs"].value_counts()


# In[ ]:


data["ups"].value_counts()


# Here we have found that values for downs are only 0s or -1s, while ups have a much wider range. However, many ups are -1 as well. Interestingly, there are more -1s in ups than in downs, meaning that it is not always the case that ups and downs are both -1 in the same row.
# 
# Since ups has other negative values, let's take a look at how many of those rows are negative in general.

# In[ ]:


sum(data["ups"] < 0)


# Approximately 20% of the ups column is negative. We can also look at the row with the minimum value of ups.

# In[ ]:


i = data["ups"].idxmin()
data.loc[i, :]


# So what does this investigation of negative ups and downs tell us? Well, it tells me that the ups and downs columns seem unreliable. Since there is no explanation for what these columns mean on either the Kaggle page for this dataset or in the paper by the authors, we may want to consider not including these columns in our future models.
# 
# Finally, let's take a quick look at the label column to verify that the classes are indeed balanced.

# In[ ]:


data["label"].value_counts()


# # Step 2: Feature Interactions
# In the next step, we'll start looking at how features individaully relate to the target, as well as how features relate to each other. This will help us get a better understanding of the dataset as a whole.
# 
# We will also create some visualizations to help us in this task, using the ```matplotlib``` library. Charts and other visual elements are often clearer and easier to understand than raw numbers.
# 
# Finally, we will engage in some feature engineering, in which we will use the features we have to create new ones. This is a good way to add new information to our model that may not otherwise be considered. However, it may also make the model more complex if many new features are added. Kaggle has a tutorial on Feature Engineering [here](https://www.kaggle.com/learn/feature-engineering).
# 
# First, let's print the first few rows of the dataset again for easy reference.

# In[ ]:


data.head(15)


# Now let's start asking questions about the features. For example, I wonder if there are some subreddits that are more sarcastic than others.
# 
# To answer this question, I will calculate the percentage of sarcastic comments in each subreddit. But first, let's see how many different subreddits we have.

# In[ ]:


len(data["subreddit"].unique())


# Wow! We have almost 15,000 unique subreddits. I could calculate average sarcasm percentage for all of them, but that's too many to consider all at once. Besides, many of those subreddits are likely represented by very few comments, which wouldn't give us a good picture of how sarcastic those subreddits are.
# 
# Instead, let's limit our search to the top 25 most represented subreddits.

# In[ ]:


top_subreddits = data["subreddit"].value_counts()[0:25]
top_subreddits = list(top_subreddits.index)


# And once we have this list, let's filter our dataset and only calculate the averages for those top subreddits.

# In[ ]:


data[data["subreddit"].isin(top_subreddits)].groupby("subreddit").agg({"label" : "mean"})


# Here we can see that some subreddits seem much more sarcastic than others. The *worldnews* subreddit is highest among these with 64% sarcasm, whereas *AskReddit* is lowest with 40%.
# 
# Based on this data, can we conclude that *worldnews* and *atheism* are among the most sarcastic subreddits on the site? Is it true that if we logged on and picked a random post on r/worldnews, we would have a 64% chance of getting a sarcastic one? The correct answer is no, to both questions.
# 
# The reasoning lies in the fact that this dataset is taken from a sample of reddit comments. Not only that, but in order to create this balanced dataset, the authors even further sampled the data to even out the classes. In their paper, the authors state that in their original sample of comments, only 1% were sarcastic. Since this dataset has been manipulated in this way, it gives us a skewed view of comments and subreddits.
# 
# We cannot say from this dataset that *worldnews* and *atheism* are some of the most sarcastic subreddits on the site. We **can** say that they are some of the most sarcastic in our dataset. And it may be true that, as a whole, they tend to be more sarcastic than others. However, we cannot accept that the proportions we have calculated are representative of the whole subreddits.
# 
# Let's continue looking at how the features interact with the sarcasm label by moving on to score. Is there a difference in scores of sarcastic comments vs. non-sarcastic ones?

# In[ ]:


data.groupby("label").agg({"score" : "mean"})


# Looks like there's a difference, but it's so slight that it may not be significant. But averages do not paint a full picture of the data; what if we take a look at the distribution of scores? Let's continue on by making a histogram comparing the sarcastic scores and the non-sarcastic scores.

# In[ ]:


scores_sarc = data["score"][data["label"] == 1]
scores_non = data["score"][data["label"] == 0]
bins = list(range(-15, 16))
plt.hist(scores_sarc, bins=bins, alpha=0.5, label="sarcastic")
plt.hist(scores_non, bins=bins, alpha=0.5, label="non-sarcastic")
plt.xlabel("score")
plt.ylabel("frequency")
plt.legend(loc="upper right")
plt.show()


# In this histogram, I zoom in to the score range of -15 to +15 because we have very long, thin tails in the distribution of scores. We can examine those tails more closely as well, but I want to focus on the segment with the most values first. The distributions of scores for sarcastic and non-sarcastic comments seem to be very similar, but we can see that sarcastic comments are slightly more likely to get more extreme scores, while non-sarcastic comments are slightly more likely to get scores in the 1 to 3 range.
# 
# For our next visualization, let's see if the amount of sarcastic comments varies over time. Specifically, we'll use the *date* column. If we group by the *date*, we can take the average of the *label*, which would be equivalent to calculating the percentage of sarcastic comments. We also want to record a count of how many comments we have for each month, to ensure that the months are comparable.

# In[ ]:


by_month = data.groupby("date").agg({"label" : "mean", "comment" : "count"})
by_month


# Now let's turn this into a line graph with ```matplotlib```.

# In[ ]:


months = list(by_month.index)
label_pos = list(range(0, len(months), 6))
m_labels = [months[i] for i in label_pos]

plt.plot(months, by_month["label"])
plt.xlabel("year-month")
plt.ylabel("% sarcastic")
plt.xticks(label_pos, m_labels, rotation=45, ha="right")
plt.show()


# Interesting, the percentage of sarcastic comments jumps around a bit, but tends to decrease as time goes on. Is this reliable, or does the number of comments for each month drastically impact the results? Let's create another line graph to see the count of comments over time.

# In[ ]:


months = list(by_month.index)
label_pos = list(range(0, len(months), 6))
m_labels = [months[i] for i in label_pos]

plt.plot(months, by_month["comment"])
plt.xlabel("year-month")
plt.ylabel("# of comments")
plt.xticks(label_pos, m_labels, rotation=45, ha="right")
plt.show()


# Indeed, we see a large ramp up in the number of comments as time goes on. There are far more comments for 2015 and 2016 than there are before 2014. This reveals an interesting piece of information: as the number of comments increased over time, the percentage of sarcastic comments per month decreased.
# 
# Why would that be? My assumption is that though the label classes are balanced over the entire dataset, the authors did not bother to balance them over time, and possibly also not over other variables. Perhaps that was just difficult to do, but it reveals that using *date* as a feature in our models would cause them to base predictions on false trends.
# 
# Does that mean that *date* is entirely useless to us? Maybe not. We may be able to utilize some information from it if we tinker with it a bit, which brings us to our next step: Feature Engineering.

# # Step 3: Feature Engineering
# 
# As part of examining features more closely, we can also perform some **Feature Engineering**. Feature Engineering is when we create new features out of existing ones. It's a way to expose information that may otherwise go unnoticed, both by ourselves and our models.
# 
# For example, I wonder if sarcastic comments tend to be longer or shorter than non-sarcastic comments. Let's create a new feature called *comment_length* and check its relationship to *label*.

# In[ ]:


data["comment_length"] = data["comment"].apply(lambda x: len(x))
data.head()


# In[ ]:


data.groupby("label").agg({"comment_length" : "mean"})


# There seems to be no difference in the average length of sarcastic and non-sarcastic comments. What if we look at standard deviation?

# In[ ]:


data.groupby("label").agg({"comment_length" : "std"})


# Ah, from the standard deviations we can see that non-sarcastic comments vary in length a lot more than sarcastic ones do. This might prove useful to our future models.
# 
# Are there any other interesting features we can come up with? For starters, we can do the same thing for *parent_comment*. We can also count punctuation marks and use of capitalization. Additionally, we could look at the date columns.
# 
# What about that *date* column that seemed to be unhelpful? Let's continue engineering features by separating *date* into a *date_year* and a *date_month*.

# In[ ]:


data["date_year"] = data["date"].str[0:4]
data["date_month"] = data["date"].str[5:]
data["date_year"] = data["date_year"].astype("int64")
data["date_month"] = data["date_month"].astype("int64")
data.head()


# Now we can check if there are any months that tend to be more sarcastic than others. And remember to ensure that our month categories have balanced numbers of comments.

# In[ ]:


data.groupby("date_month").agg({"label" : "mean", "comment" : "count"})


# From these numbers, it seems that the ratio sarcastic and non-sarcastic comments stays fairly constant at 50/50 for most of the months, with a slight dip at the end of the year toward more non-sarcastic comments. We can also see the same issue here that we saw in the *date* column overall: we have far more comments (almost double!) for late months of the year compared to early months. This means that this trend may also be unreliable.
# 
# Now I wonder if we can see any trend in sarcasm throughout the days of the week. First, we'll have to engineer a new feature called *comment_day*. (This will take a few seconds to run.)

# In[ ]:


import datetime
date_format = "%Y-%m-%d %H:%M:%S"

def get_weekday(d):
    d = datetime.datetime.strptime(d, date_format)
    return d.strftime("%w")

data["comment_day"] = data["created_utc"].apply(lambda x: get_weekday(x))
data["comment_day"] = data["comment_day"].astype("int64")
data.head()


# Now let's see if there are any differences in amount of sarcastic comments per weekday. In this case, 0 indicates Sunday and 6 indicates Saturday.

# In[ ]:


data.groupby("comment_day").agg({"label" : "mean", "comment" : "count"})


# Looks like the portion of sarcastic comments stays fairly constant at 50/50 again, with a slight dip towards non-sarcastic on the weekends. In contrast to the months column we examined above, we can see that the number of comments stays relatively balanced across all the days of the week.
# 
# Now that we've closely examined the data and created a few new features, it's time to start preparing our data for training. I'll output our final data frame as a csv so that I can use it in the next notebook.

# In[ ]:


data.to_csv("sarcasm_prepped_data.csv", index=False)


# [\[Next\]](https://www.kaggle.com/yastapova/sarcasm-detection-2020-mentoring-proj-part-2) >> Part 2: Splitting Data and Building a Basic Machine Learning Model
