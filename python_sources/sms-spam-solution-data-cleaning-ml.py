#!/usr/bin/env python
# coding: utf-8

# In this notebook we will use nltk librarie to analyse and clean the sms text body and then the random forest classifier algorithme to predict and classify the label of the sms based on the clean text.

# let's load the required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
from plotly import tools
import plotly.figure_factory as ff

import nltk
import re
import string


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's begin by importing the spam dataset, and showing the first 5 rows.

# In[ ]:


data=pd.read_csv('../input/spam.csv',delimiter=',',encoding='latin-1')


# In[ ]:


data.head()


# Drop down the unnecessary columns

# In[ ]:


data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)


# Let's show the 5 rows of our data again, using this time the plotly labrarie

# In[ ]:


py.iplot(ff.create_table(data.head()),filename='show_data')


# Let's change the columns name

# In[ ]:


data.columns=['label','text']
py.iplot(ff.create_table(data.head()),filename='show_data')


# In[ ]:


data.text[0]


# You can see that our text contains some punctuation characters so we need to deal with it

# To do that we'll import the string labrarie and using the punctuation attribute, you can see all the attributes of this labrarie using the dir() function

# In[ ]:


dir(string)


# In[ ]:


string.punctuation


# Let's create a function that will help us to remove the punctuation characters from the original text 

# In[ ]:


def remove_punctuation(text):
    new_text=''.join([char for char in text if char not in string.punctuation])
    return new_text


# We'll apply this function to create a new column 
# 

# In[ ]:


data['new_text']=data['text'].apply(lambda row : remove_punctuation(row))


# In[ ]:


data.head()


# In[ ]:


print(data.text[0])
data.new_text[0]


# You can see now that our text is out of punctuations

# In the second step we'll split each text into tokens so let's create another function

# In[ ]:


def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens 


# Let's apply this function and show our tokenized text

# In[ ]:


data['tokenized_text']=data['new_text'].apply(lambda row : tokenize(row.lower()))
data.head()


# The next step is to remove the stopwords because they does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence.

# Let's import the stop words and show the first 5 words in the list 

# In[ ]:


stopwords=nltk.corpus.stopwords.words('english')
stopwords[:5]


# So we need another function to remove the stop words

# In[ ]:


def remove_stopwords(text):
    clean_text=[word for word in text if word not in stopwords]
    return clean_text 


# In[ ]:


data['clean_text']=data['tokenized_text'].apply(lambda row : remove_stopwords(row))
data.head()


# Our final text preprocessing is the stemming step ,we'll use the PorterStemmer to do that. 

# In[ ]:


ps = nltk.PorterStemmer()


# In[ ]:


dir(ps)


# In the list above we are interested in the stem method to create a stemmed text 

# In[ ]:


def stemming(tokenized_text):
    stemmed_text=[ps.stem(word) for word in tokenized_text]
    return stemmed_text


# In[ ]:


data['stemmed_text']=data.clean_text.apply(lambda row : stemming(row))
data[['text','stemmed_text']].head()


# So finally we'll join the stemmed words to create our final text 

# In[ ]:


def get_final_text(stemmed_text):
    final_text=" ".join([word for word in stemmed_text])
    return final_text


# In[ ]:


data['final_text']=data.stemmed_text.apply(lambda row : get_final_text(row))
data.head()


# Now we'll doing some feature engineering it's a necessary preprocessing step for machine learning 

# In this data we have rows with texts so we need to transform each word in this texts into a feature column so let's do that ,we'll using TfidfVectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf_model=TfidfVectorizer()
tfidf_vec=tfidf_model.fit_transform(data.final_text)
tfidf_data=pd.DataFrame(tfidf_vec.toarray())
tfidf_data.head()


# We have now a data with 8026 columns each column represent a word 

# We deleted the punctuation before, but what if they have impact on our label ? so let's create a feature with percentage of punctuations.

# In[ ]:


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100


# In[ ]:


data['punct%'] = data['text'].apply(lambda x: count_punct(x))


# In[ ]:


bins = np.linspace(0, 100, 40)
plt.hist(data['punct%'], bins)
plt.title("Punctuation Distribution")
plt.show()


# The lenght of the text may too have an impact on our label? 

# In[ ]:


data['text_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))


# In[ ]:


bins = np.linspace(0, 250, 40)
plt.hist(data['text_len'],bins)
plt.title("text Length Distribution")
plt.show()


# In[ ]:


final_data=pd.concat([data['punct%'],data['text_len'],tfidf_data],axis=1)
final_data.head()


# Our final data have now 8028 columns

# It's time for machine learning, we'll build a classification model using RandomForestClassifer

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(final_data,data['label'],test_size=.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)


# In[ ]:


rf_prediction=rf_model.predict(X_test)


# In[ ]:


precision,recall,fscore,support = score(y_test,rf_prediction,pos_label='spam',average='binary')


# In[ ]:


print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((rf_prediction==y_test).sum() / len(rf_prediction),3)))


# You can use the model feature_importances_ attribute to check the features that have been very significant  in our model 

# In[ ]:


sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]


# To tune the model hyperparameters you can use the Grid Search methode

# Grid-search: Exhaustively search all parameter combinations in a given grid to determine the best model.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


rfg = RandomForestClassifier()
param = {'n_estimators': [10, 150, 300],
        'max_depth': [30, 60, 90, None]}

gs = GridSearchCV(rfg, param, cv=5, n_jobs=-1)
gs_fit = gs.fit(final_data, data['label'])


# In[ ]:


print(gs_fit.best_params_)
print(gs_fit.best_score_)

