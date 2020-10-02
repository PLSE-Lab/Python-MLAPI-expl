#!/usr/bin/env python
# coding: utf-8

# # Shifterator text analysis on Animal Crossing reviews
# 
# I saw a tweet from [Ryan J. Gallagher](https://twitter.com/ryanjgallag) about his new Python package for people who hate wordclouds called "Shifterator". From his GitHub repo:
# 
# > _The Shifterator package provides functionality for constructing word shift graphs, vertical bart charts that quantify which words contribute to a pairwise difference between two texts and how they contribute. By allowing you to look at changes in how words are used, word shifts help you to conduct analyses of sentiment, entropy, and divergence that are fundamentally more interpretable._
# 
# I decided to try out this new package on this week's TidyTuesday dataset on [Animal Crossing](https://www.kaggle.com/jessemostipak/animal-crossing). I'm definitely more of an #rstats person where I'm a huge fan of Dr. Julia Silge and David Robinson's [TidyText package](https://www.tidytextmining.com/). But Shifterator looked intriguing enough for me to dust off my rusty Python skills (or lack thereof). Anyway, so apologies if my Python code looks terrible. Let me know in the comments what I should fix.

# In[ ]:


get_ipython().system('pip install shifterator')


# In[ ]:


# Import packages

import pandas as pd
import numpy as np
import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re

from shifterator import relative_shift as rs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# In[ ]:


# Load the review CSV
reviews = pd.read_csv("../input/animal-crossing/user_reviews.csv", encoding='utf-8')


# In[ ]:


reviews.head()


# Each review has the following columns: the date, a review, and the review text.
# 
# My research question will be: How do negative and positive reviews compare in the words they use?

# ## Inspect and prepare the data
# 
# The first thing I'm going to do is have a look at some of the data to get a bit of a feel for it.
# 
# Let's look at average review grades over time.

# In[ ]:


reviews['date'] = pd.to_datetime(reviews['date'])
reviews.index = reviews['date'] 

fig, ax = plt.subplots(figsize=(12, 8))

mean_daily_grades = reviews.resample('D', on='date').mean().reset_index('date')

# Plot horizontal bar graph
monthly_plot = sns.lineplot(data = mean_daily_grades,
                      x = 'date',
                      y = 'grade',
                      color="purple"
                      )

ax.set_title("Average daily grade")
x_dates = mean_daily_grades['date'].dt.strftime('%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.show()


# It looks like reviews started off pretty positively and declined steeply after which the average has been bouncing between 2 and almost 7. I could look at more distributions and things, but I won't for now. Mostly because I'd much prefer to use ggplot2 in R. :)

# In[ ]:


# Divide reviews into positive and negative based on the median grade for the dataset
median_grade = reviews.grade.median()

reviews.loc[reviews['grade'] <= median_grade, 'review_category'] = 'Negative' 
reviews.loc[reviews['grade'] > median_grade, 'review_category'] = 'Positive' 

reviews_neg = reviews[reviews['review_category'] == 'Negative']
reviews_pos = reviews[reviews['review_category'] == 'Positive']


# In[ ]:


texts = reviews['text'].tolist()
texts_neg = reviews_neg['text'].tolist()
texts_pos = reviews_pos['text'].tolist()


# I learned how to clean the review text data and calculate frequencies using [this tutorial](https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/calculate-tweet-word-frequencies-in-python/). The next few cells will clean and prepare the data by removing punctuation, stop words, change everything to lower case, etc so we can calculate frequencies.

# In[ ]:


# We will want to remove stop words
stop_words = set(stopwords.words('english'))


# In[ ]:


def remove_punctuation(txt):
    """Replace URLs and other punctuation found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with URLs and punctuation removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[ ]:


def clean_text(txt):
    """Removes punctuation, changes to lowercase, removes
        stopwords, removes "animal" and "crossing", and
        calculates word frequencies.

    Parameters
    ----------
    txt : string
        A text string that you want to clean.

    Returns
    -------
    Words and frequencies
    """
    
    tmp = [remove_punctuation(t) for t in txt]
    tmp = [t.lower().split() for t in tmp]
    
    tmp = [[w for w in t if not w in stop_words]
              for t in tmp]
    tmp = [[w for w in t if not w in ['animal', 'crossing']]
                     for t in tmp]
    
    tmp = list(itertools.chain(*tmp))
    tmp = collections.Counter(tmp)
        
    return tmp


# In[ ]:


# Clean up the review texts
clean_texts_neg = clean_text(texts_neg)
clean_texts_pos = clean_text(texts_pos)


# ## Plot data in a boring way
# 
# First, I thought it would be interesting to make more boring graphs of the data to compare to the cool ones with Shifterator.

# In[ ]:


# Dataframes for most frequent common words in positive and negative reviews
common_neg = pd.DataFrame(clean_texts_neg.most_common(15),
                             columns=['words', 'count'])
common_pos = pd.DataFrame(clean_texts_pos.most_common(15),
                             columns=['words', 'count'])


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
common_neg.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="red")

ax.set_title("Common Words Found in Negative Reviews")

plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
common_pos.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="green")

ax.set_title("Common Words Found in Positive Reviews")

plt.show()


# Yes, these plots are very boring. Common words between negative and positive reviews are pretty similar. For something a bit more nuanced I'd normally look to calculate something like tf-idf, but again, [I'd be a lot more at home doing that in R with TidyText](https://www.tidytextmining.com/tfidf.html). Tf-idf would tell you more about a words' relative importance in a corpus taking frequency into account.
# 
# Okay, thought you'd get away without seeing a word cloud? Not so fast. ;) Let's do one for good measure. I'll spare you and just plot one for the negative reviews.

# In[ ]:


# From https://www.kaggle.com/prakashsadashivappa/word-cloud-of-abstracts-cord-19-dataset
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    ).generate(str(texts_neg))


# In[ ]:


fig = plt.figure(
    figsize = (10, 8),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ## Create word shift graphs with Shifterator
# 
# Time to finally use the Shifterator package! We could compare negative and positive Animal Crossing reviews based on both frequency and sentiment (or other values) using this package, but I've only calculated frequencies, so we'll just try plotting that.
# 
# 
# ### Entropy shift
# 
# The first graph is an entropy shift graph. [See the GitHub repo for more details](https://github.com/ryanjgallagher/shifterator#entropy-and-kullback-leibler-divergence-shifts).

# In[ ]:


# Get an entropy shift
entropy_shift = rs.EntropyShift(reference=clean_texts_neg,
                                comparison=clean_texts_pos,
                                base=2)
entropy_shift.get_shift_graph() 


# It looks like the negative reviews are in purple and positive ones are in yellow. It looks like feedback about the whole "one island per Switch" dominates. Fortunately my husband doesn't play so we don't have to deal with that. I wonder if "fix", "ridiculous", "experience" refers to some of the goofy, clunky UX. At least I think it's pretty goofy and clunky. A lot of the words are nouns and verbs like "console", "family", "money", "fix", "save".
# 
# Among the positive reviews, there are more adjectives pulled out. For example, "best", "fun", "amazing", "relaxing, "perfect". I wonder what "bombing" refers to in the positive reviews?

# ### Jensen-Shannon divergence shifts
# 
# The second graph is an Jensen-Shannon divergence shift graph. [See the GitHub repo for more details](https://github.com/ryanjgallagher/shifterator#jensen-shannon-divergence-shifts).

# In[ ]:


# Get a Jensen-Shannon divergence shift
from shifterator import symmetric_shift as ss
jsd_shift = ss.JSDivergenceShift(system_1=clean_texts_neg,
                                 system_2=clean_texts_pos,
                                 base=2)
jsd_shift.get_shift_graph()


# Hmm, apart from the negative and positive reviews switching places, I don't have too much more to add to this plot. This analysis pulls out slightly different words and rankings.

# ## Conclusion
# 
# I hope this wasn't too painful to read for my Pythonista friends. I had fun checking out the new Shifterator package and I'm looking forward to digging into it in greater depth. It definitely seems much more promising than a word cloud, to say the least. If you fork this notebook and do something cool to extend and improve this analysis (especially if you add sentiment scores), let me know in the comments! Thanks again to Ryan Gallagher for [Shifterator](https://github.com/ryanjgallagher/shifterator).
