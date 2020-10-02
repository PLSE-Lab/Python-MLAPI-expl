#!/usr/bin/env python
# coding: utf-8

#    **Are you being serious!! (no really are you..?)**
#    
# 
# In this notebook I am going to look into the different techiques to see if we can predict whether or not a comment is being sarcastic.
# 
# For this I will be using the Sarcasm On Reddit data set.  The dataset I will be using will be approximately 1.3 million comments.  Approximately half of these are labeld as sarcastic (the user specified the \s flag in reddit to indicate they were being actually sarcastic) and the other hald is not sarcastic.
# 
# Aside from the intial comment, we also have the following features which we can use
# * subject being discussed
# * author of the original comment
# * follow up reply comment from another (unspecified) reddit user
# 
# 1. We will start by doing some initial statistical analysis to see if certain authors or subjects lean to being more sarcastic.
# We will then do some basic ngram analysis and wordBubbles to see if there are certain phrases that are very prevalent in sarcastic and non sarcastic comments.
# 
# 1. We will then train a model just using the comment to see how well we can predict if another comment is sarcastic enought.
# 
# 1. We will then see if we can improve that prediction by factoring in the author and the subject being discussed (which really are just weightings)
# 
# 1. Finally we will look into how well the model (trasined just on the comments) detetcts sarcasm in a non reddit context (assuming I can get some reliable data for this)
# 
# 
# 

# **Part one - reading in data and some basic statistical analysis**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First we use pandas to read in the balanced dataset into a panda dataframe
# Secondly we split that dataframe into two sets where one is sarcastic and one is not sarcastic

# In[ ]:


dataframe = pd.read_csv("../input/train-balanced-sarcasm.csv")


# In[ ]:


# allows us to take a look at some of the data
dataframe.head()


# In[ ]:


# tells you how many values in each column
dataframe.info()


# In[ ]:


# simply shows the number of rows and columns in your data set
dataframe.shape


# In[ ]:


# displays the total number of rows for each value of label (in this case 0 and 1)
dataframe['label'].value_counts()


# > Its balanced as we can see we have 505413 results assigned to both the sarcastic and non-sarcastic labels

# In[ ]:


# now we split these out into two sets, sarc and notsarc
notsarc = dataframe[dataframe['label'] == 0]
sarc = dataframe[dataframe['label'] == 1]


# In[ ]:


sarc['comment']


# In[ ]:


sarc['subreddit'].value_counts().head()


# In[ ]:


notsarc['subreddit'].value_counts().head()


# In[ ]:


dataframe['subreddit'].value_counts().head()


# In[ ]:





# Ok, just looking straight at the comments in this way doesn't take into account the volume of comments.  We can't conclude for example that AskReddit is the most sartcastic subject because it has the most comments.  We need to produce a table of percentage of comments for each subject that are sarcastic (we can then do the same for author as well)
# 

# In[ ]:


sub_df_subreddit = dataframe.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
sub_df_subreddit[sub_df_subreddit['sum'] > 1000].sort_values(by='mean', ascending=False).head(10)


# The above table is intersting.  I disregarded any subreddits with less than 1000 entries (the problem if sorting by mean sarcyness subjects with very low entries will come to the top naturally)
# 
# So it does seem that certain topics are more likely to be sarcastic.
# 
# Lets do the same for author

# In[ ]:


sub_df_author = dataframe.groupby('author')['label'].agg([np.size, np.mean, np.sum])
sub_df_author[sub_df_author['sum'] > 5].sort_values(by='mean', ascending=False).head(10)


# Hmmm, I'm having a hard time excepting the above.  Something is wrong, how can so many authors have exactly the same number of sarcastic and unsarcatic comments.  Unless it is in the way the balanced set was created.  Maybe they took an even number of both types of comments for each author.  In which case, author is useless.  I guess it is important to know and understand how the balanced data set was created.

# In[ ]:


print(len(dataframe[(dataframe['author'] == 'Biffingston') & (dataframe['label'] == 0)]))
print(len(dataframe[(dataframe['author'] == 'Biffingston') & (dataframe['label'] == 1)]))


# Ok so looking at the data for a frequent author we can see they have balanced it across author.  So author can be disregarded as feature to train sarcasm on 

# But what about ups, downs and score, do people like sarcasm, does that get voted up more.   We'll classify into 3 bins, those with a score of zero, those with a positve score, those with a negative score

# In[ ]:


print("score zero " , dataframe[dataframe['score'] == 0]['label'].describe()['mean'])
print("positive score " , dataframe[dataframe['score'] > 0]['label'].describe()['mean'])
print("negative score " , dataframe[dataframe['score'] < 0]['label'].describe()['mean'])


# **Now we move on to the machine learning part of this all.**
# 
# 

# In[ ]:


# some necessary imports
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


# there are some comments with null, these need to be dropped as it can cause issues
dataframe.dropna(subset=['comment'], inplace=True)


# Now we split our data into a training and validation set.
# 
# The model gets trained on the training set
# 
# The accuracy of the model then gets its accuracy measured by using the validation set

# In[ ]:


# lets split out the data into training and validation sets
# train_texts is what we train, y_train is the column ref and the label
# valid_texts is what we validate or test on, y_valid is the column ref and the label

# the default split is 75%training set, 25% testing set 



train_texts, valid_texts, y_train, y_valid =         train_test_split(dataframe['comment'], dataframe['label'], random_state=17)


# In[ ]:


# build bigrams, put a limit on maximal number of features
# and minimal word frequency
tf_idf = TfidfVectorizer(ngram_range=(1, 4), max_features=20000, min_df=4)
# multinomial logistic regression a.k.a softmax classifier
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 
                           random_state=17, verbose=1)
# sklearn's pipeline
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 
                                 ('logit', logit)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tfidf_logit_pipeline.fit(train_texts, y_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# this produces a a prediction an array the same size as valid texts, where 0 and 1 refer to the sarcasm predicton\nvalid_pred = tfidf_logit_pipeline.predict(valid_texts)')


# In[ ]:


valid_pred


# In[ ]:


accuracy_score(y_valid, valid_pred)


# In[ ]:


cm  = confusion_matrix(y_valid,valid_pred)
cm


# In[ ]:


tp_accuracy = cm[0][0]/(cm[0][0] + cm[0][1])
tn_accuracy = cm[1][1]/(cm[1][0]+cm[1][1])
print("true positive accuracy: " + str(tp_accuracy))
print("true negative accuracy: " + str(tn_accuracy))


# In[ ]:


def amISerious(model, text):
    result = model.predict([text])[0]
    if (result == 1):
        return "sarcastic"
    else:
        return "not sarcastic"
    


# In[ ]:


amISerious(tfidf_logit_pipeline,"whatever i didnt enjoy that film")


# In[ ]:


amISerious(tfidf_logit_pipeline,"yeah whatever, i didnt enjoy that film")


# In[ ]:





# In[ ]:




