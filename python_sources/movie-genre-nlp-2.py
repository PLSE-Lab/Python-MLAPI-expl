#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# In[ ]:


movie = pd.read_csv("/kaggle/input/movie-data/movie_metadata.csv", header=None)


# In[ ]:


movie.head()


# In[ ]:


movie.columns = ["movie_id", 1, "movie_name", 3, 4, 5 ,6, 7, "genre"]


# In[ ]:


movie.head(1)


# In[ ]:


plot = pd.read_csv("/kaggle/input/movie-data/plot_summaries.txt", sep = '\t', header=None)


# In[ ]:


plot.columns = ["movie_id", "plot"]


# In[ ]:


plot.head()


# In[ ]:


data = movie[["movie_id", "movie_name", "genre"]].merge(plot, on="movie_id")
data_test = pd.merge(plot, movie[["movie_id", "movie_name", "genre"]], on= "movie_id")


# In[ ]:


print(data.shape, data_test.shape)


# In[ ]:


data.head()


# In[ ]:


data[data["movie_name"] == "Narasimham"]


# In[ ]:


data["genre"][0]


# In[ ]:


list(json.loads(data["genre"][0]).values())


# In[ ]:


genre = []

for i in data["genre"]:
    #print(list(json.loads(i).values()))
    genre.append(list(json.loads(i).values()))


# In[ ]:


data["genre_new"] = genre


# In[ ]:


data.head(1)


# In[ ]:


data.dtypes


# In[ ]:


#type(str(data["genre_new"]))
data_new = data[~(data["genre_new"].str.len() == 0)]


# In[ ]:


data_new.shape


# In[ ]:


dumm = []

for i in genre:
    for j in i:
        #print(j)
        dumm.append(j)


# In[ ]:


all_genre = list(set(dumm))


# In[ ]:


var = sum(genre,[])


# In[ ]:


len(all_genre)


# In[ ]:


genre_new = nltk.FreqDist(dumm)


# In[ ]:


genre_new


# In[ ]:


type(genre_new)


# In[ ]:


len(genre_new.keys())


# In[ ]:


genre_df = pd.DataFrame.from_dict(genre_new, orient="index")


# In[ ]:


genre_df.columns = ["Count"]
genre_df.index.name = ["Genre"]


# In[ ]:


genre_df.head()


# In[ ]:


#genre_df["Genre"] = genre_df.index
#genre_df.reset_index()
del genre_df.index.name


# In[ ]:


genre_df = genre_df.reset_index()


# In[ ]:


genre_df.shape


# In[ ]:


genre_df.columns = ["Genre", "Count"]


# In[ ]:


genre_df.head(2)


# In[ ]:


plt.figure(figsize=(12,12))
sns.barplot(data=genre_df.sort_values("Count", ascending=False).loc[:20, :], x="Count", y="Genre")


# In[ ]:


def clean_text(text):
    
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    text = text.lower()
    
    return text


# In[ ]:


data_new["clean_plot"] = data_new["plot"].apply(lambda x : clean_text(x))


# In[ ]:


data_new.head(2)


# In[ ]:


def freq_plot(text):
    
    words = " ".join([x for x in text])
    words = words.split()
    fdist = nltk.FreqDist(words)
    return fdist


# In[ ]:


fdist = freq_plot(data_new["clean_plot"])


# In[ ]:


words_df = pd.DataFrame.from_dict(fdist, orient="index")


# In[ ]:


words_df = words_df.reset_index()


# In[ ]:


words_df.columns = ["Word","Count"]


# In[ ]:


words_df.head()


# In[ ]:


plt.figure(figsize=(12,12))
sns.barplot(data= words_df.sort_values(by="Count",ascending= False).iloc[:20, :], x = "Count", y= "Word")


# In[ ]:


nltk.download("stopwords")


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


stopwords = set(stopwords.words("english"))


# In[ ]:


def remove_stopwords(text):
    no_stop = []
    
    for i in text.split():
        if i not in stopwords:
            no_stop.append(i)
    return " ".join(no_stop)


# In[ ]:


data_new["clean_plot"] = data_new["clean_plot"].apply(lambda x : remove_stopwords(x))


# In[ ]:


data_new.head(2)


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[ ]:


multilabel_bina = MultiLabelBinarizer()
multilabel_bina.fit(data_new["genre_new"])


# In[ ]:


y = multilabel_bina.transform(data_new["genre_new"])


# In[ ]:


tfidf_vect = TfidfVectorizer(max_df= 0.8, max_features=10000)


# In[ ]:


data_new.shape


# In[ ]:


y.shape


# In[ ]:


xtrain, xval, ytrain, yval = train_test_split(data_new["clean_plot"], y, test_size = 0.2, random_state= 9)


# In[ ]:


tfidf_vect


# In[ ]:


xval.shape


# In[ ]:


xtrain_tfidf = tfidf_vect.fit_transform(xtrain)
xval_tfidf = tfidf_vect.transform(xval)


# In[ ]:


xtrain_tfidf.shape


# In[ ]:


xval_tfidf.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


logistic_mod = LogisticRegression()
onevsall = OneVsRestClassifier(logistic_mod)


# In[ ]:


onevsall.fit(xtrain_tfidf, ytrain)


# In[ ]:


y_pred = onevsall.predict(xval_tfidf)


# In[ ]:


y_pred[2]


# In[ ]:


multilabel_bina.inverse_transform(y_pred)[34]


# In[ ]:


print(classification_report(yval, y_pred))


# In[ ]:


y_pred_prob = onevsall.predict_proba(xval_tfidf)


# In[ ]:


t = 0.3
y_pred_new = (y_pred_prob >= t).astype(int)


# In[ ]:


print(classification_report(yval, y_pred_new))


# In[ ]:


def new_val(x):
    
    x = clean_text(x)
    x = remove_stopwords(x)
    x_vec = tfidf_vect.transform([x])
    x_pred = onevsall.predict(x_vec)
    
    return multilabel_bina.inverse_transform(x_pred)


# In[ ]:


for i in range(5): 
  k = xval.sample(1).index[0] 
  print("Movie: ", data_new['movie_name'][k], "\nPredicted genre: ", new_val(xval[k])), print("Actual genre: ",data_new['genre_new'][k], "\n")


# In[ ]:




