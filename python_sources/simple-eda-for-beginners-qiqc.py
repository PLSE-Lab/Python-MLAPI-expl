#!/usr/bin/env python
# coding: utf-8

# ### Notebook Objective:
# 
# Objective of this notebook is to explore the data and to build a simple baseline model.
# 
# ### Objective of the competition:
# 
# The objective was to predict whether a question asked on Quora is sincere or not.
# 
# An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
# 
# **Has a non-neutral tone**   
# - Has an exaggerated tone to underscore a point about a group of people
# - Is rhetorical and meant to imply a statement about a group of people
# 
# **Is disparaging or inflammatory**
#    - Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
#    - Makes disparaging attacks/insults against a specific person or group of people
#    - Based on an outlandish premise about a group of people
#    - Disparages against a characteristic that is not fixable and not measurable
# 
# **Isn't grounded in reality**
#    - Based on false information, or contains absurd assumptions
# 
# **Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers**

# In[ ]:


# Import important libraries
import os
import json
import string
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# #### Data Files

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Lets have a look at the shape of training and testing data

train_df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
print('Train shape :', train_df.shape)
print('Test shape',test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# There is no missing data....Got lucky there :P****

# ### Univariate Study
# Lets look at the distribution of dependent variable here :

# In[ ]:


# Count number of instances of each target in dataset
val_count = train_df['target'].value_counts()
print(val_count)


# In[ ]:


# CountPlot
sns.countplot(train_df['target'])


# In[ ]:


labels = [0,1]
values = [val_count[0], val_count[1]]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()


# * So there is about ~6% data that comprises of insincere questions and ~94% data that has sincere questions.

# ### Multivariate study
# Let us have a look at the most frequently occuring words in 'question_text' column :

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(train_df["question_text"], title="Word Cloud of Questions")


# There are a variety of words in there which is very confusing because words in there can be from sincere questions as well as insincere questions.

# **Lets seperate these 2 question types into 2 different dataframes** 

# In[ ]:


train_sincere_df = train_df[train_df['target'] == 0]
train_insincere_df = train_df[train_df['target'] == 1]


# In[ ]:


print("Sincere Questions df size : " + str(train_sincere_df.shape))
print("Insincere Questions df size : " + str(train_insincere_df.shape))


# **Now we have 2 different dataframes for each question type, lets analyze most occuring words - singly, in pair and in group of 3 - using ngram models for both dataframes separately**

# ### Frequently occuring words singly (Ngram = 1)

# In[ ]:


from collections import defaultdict

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train_sincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train_insincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')


# **Observations -**
# - Some words are common in both classes like 'people', 'will', etc.
# - Some words in sincere class are not frequent in insincere class like 'best', 'good'. etc.
# - Some words in insincere class are not frequent in sincere class like 'trump', 'women'. etc.

# ### Frequently occuring words in pair (Ngram = 2)

# In[ ]:


## custom function for ngram generation ##
def generate_ngrams(text, n_gram=2):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train_sincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train_insincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')


# ### Frequently occuring words in group of 3 (Ngram = 3)

# In[ ]:


## custom function for ngram generation ##
def generate_ngrams(text, n_gram=3):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train_sincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train_insincere_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')


# **Now, lets look at below questions to get a better idea of 'question_text' feature in both classes**
# - Average length of words in questions.
# - No. of stopwords in questions.
# - No. of words in questions.
# - No. of unique words in questions.

# In[ ]:


# Average length of words in questions
train_sincere_df["mean_word_len"] = train_sincere_df["question_text"].apply(lambda x : np.mean([len(c) for c 
                                                                                                in str(x).split()]))
train_insincere_df["mean_word_len"] = train_insincere_df["question_text"].apply(lambda x : np.mean([len(c) for c 
                                                                                                in str(x).split()]))

# No. of Stopwords in questions
train_sincere_df["num_stopwords"] = train_sincere_df["question_text"].apply(lambda x : len([word for word 
                                                                    in str(x).lower().split() if word in STOPWORDS]))
train_insincere_df["num_stopwords"] = train_insincere_df["question_text"].apply(lambda x : len([word for word 
                                                                    in str(x).lower().split() if word in STOPWORDS]))

# No. of words in questions
train_sincere_df["num_words"] = train_sincere_df["question_text"].apply(lambda x : len(str(x).split()))
train_insincere_df["num_words"] = train_insincere_df["question_text"].apply(lambda x : len(str(x).split()))

# No. of unique words in questions
train_sincere_df["num_unique_words"] = train_sincere_df["question_text"].apply(lambda x : len(set(str(x).split())))
train_insincere_df["num_unique_words"] = train_insincere_df["question_text"].apply(lambda x : len(set(str(x).split())))


# **Let's have a visual look at the distributions of the above computed results for a better idea**

# In[ ]:


## Truncate some extreme values for better visuals ##
train_sincere_df['num_words'].loc[train_sincere_df['num_words']>60] = 60 #truncation for better visuals
train_sincere_df['num_unique_words'].loc[train_sincere_df['num_unique_words']>10] = 10 #truncation for better visuals
train_sincere_df['num_stopwords'].loc[train_sincere_df['num_stopwords']>350] = 350 #truncation for better visuals
train_insincere_df['num_words'].loc[train_insincere_df['num_words']>60] = 60 #truncation for better visuals
train_insincere_df['num_unique_words'].loc[train_insincere_df['num_unique_words']>10] = 10 #truncation for better visuals
train_insincere_df['num_stopwords'].loc[train_insincere_df['num_stopwords']>350] = 350 #truncation for better visuals

df = [train_sincere_df, train_insincere_df]
train_df = pd.concat(df)

f, axes = plt.subplots(3, 1, figsize=(10,20))
sns.boxplot(x='target', y='num_words', data=train_df, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=12)
axes[0].set_title("Number of words in each class", fontsize=15)

sns.boxplot(x='target', y='num_unique_words', data=train_df, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_title("Number of unique words in each class", fontsize=15)

sns.boxplot(x='target', y='num_stopwords', data=train_df, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=12)
axes[2].set_title("Number of stopwords in each class", fontsize=15)
plt.show()


# **Observations -**
# - Insincere questions have more no. of words and are longer than sincere questions.
# - Since insincere questions are long, they have more no. of unique words.
# - No. of stopwords are more in insincere questions than in sincere questions.

# ### Now we will build a Baseline model
# To start with, let us just build a baseline model (Logistic Regression) with TFIDF vectors.

# In[ ]:


# Get the tfidf vectors #
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train_df['question_text'].values.tolist() + test_df['question_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['question_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['question_text'].values.tolist())


# Let's build the model now

# In[ ]:


train_y = train_df["target"].values

def runModel(train_X, train_y, test_X, test_y, test_X2):
    model = linear_model.LogisticRegression(C=5., solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:,1]
    pred_test_y2 = model.predict_proba(test_X2)[:,1]
    return pred_test_y, pred_test_y2, model

print("Building model.")
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break


# Getting the better threshold based on validation sample.

# In[ ]:


for thresh in np.arange(0.1, 0.201, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))


# **So we are getting a better F1 score for this model at 0.16.!**

# ## !!! That's all folks !!!
# If you liked this kernel, then please give it an upvote. 
# 
# Feedbacks are always welcome :)
