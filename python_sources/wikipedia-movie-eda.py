#!/usr/bin/env python
# coding: utf-8

# # Exploratry Data Analysis and Regression Model for  Wikipedia Movie Plots

# ![](https://boygeniusreport.files.wordpress.com/2016/03/movies-tiles.jpg?quality=98&strip=all)
# 
# <!--
# ![](https://i.ytimg.com/vi/KPX39PpS1sc/maxresdefault.jpg)
# ![](http://4.bp.blogspot.com/-Tz2h4J4VdPw/UvezidpNOeI/AAAAAAAAJck/oHfz4EHTg7o/s1600/2013_movies_wallpaper_by_z_designs-d5u94o3.jpg)
# -->

# ## 1. Import 

# ### 1.1. Import Library

# In[ ]:


# System
import os
from os import path

# Numerical Data
import numpy as np 
import pandas as pd

# Tools
import itertools

# NLP
import re

# Preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn import svm
from sklearn import metrics 

# Evaluation Matrics
from sklearn.metrics import f1_score

# Graph/ Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Image
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

# Input data
print(os.listdir("../input"))


# ## 2. Read Data

# In[ ]:


df = pd.read_csv("../input/wiki_movie_plots_deduped.csv")


# ### 2.1. Show Data Stats

# In[ ]:


df.head()


# In[ ]:


# df.describe()


# ## 3. Visualization

# In[ ]:


def get_params():
    params = {'legend.fontsize' : 'Large',
              'figure.figsize'  : figsize,
              'axes.labelsize'  : 'x-large',
              'axes.titlesize'  : 'xx-large',
              'xtick.labelsize' : 'Large',
              'ytick.labelsize' : 'Large'}
    return params

def count_plot(x, y, df=df, figsize=(18, 6)):
    # sns.set(style="ticks")
    sns.set(style="whitegrid")

    params = {'legend.fontsize': 'large',
              'figure.figsize' : figsize,
              'axes.labelsize' : 'x-large',
              'axes.titlesize' : 'xx-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    plt.rcParams.update(params)


    ax = sns.countplot(x=x, y=y, data=df)
    if y: plt.title(y)
    else: plt.title(x)
    plt.xticks(rotation=90)

    plt.show()


# ### 3.1. Showing Movie Count per Year for over 100 Years

# In[ ]:


# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(18, 30)
xcol = "Release Year"

params = get_params()

plt.rcParams.update(params)
plt.tick_params(labelsize=12)
sns.countplot(y=df[xcol], data=df)
plt.title("Movie Count Per "+ xcol)
# plt.xticks(rotation=90)
# plt.yticks(rotation=90)
plt.tight_layout()
plt.show()


# ### 3.2. Showing Movie Count per Country/Origin/Ethnicity

# In[ ]:


# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(18, 10)
xcol = "Origin/Ethnicity"

params = get_params()
plt.tick_params(labelsize=18)
plt.rcParams.update(params)
sns.countplot(x=df[xcol], data=df)
plt.title("Movie Count Per "+ xcol)

plt.xticks(rotation=90)
# plt.yticks(rotation=90)
# plt.tight_layout()
plt.show()


# ### 3.3. Showing Movie Count per Country/Ethnicity/Origin throughout over the past 100+ years

# In[ ]:


# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(20, 8)
xcol = "Origin/Ethnicity"
params = get_params()
plt.rcParams.update(params)
org = df[xcol].unique()
l = len(df[xcol])
con = []
for country in df[xcol].unique():
    c = df[df[xcol]==country]
    if len(c)>l*0.03:
        x = df[df[xcol]==country]["Release Year"].value_counts()
        sns.lineplot(x.index, x.values)
        con.append(country)
plt.legend(con)
plt.title("Movie Count Per "+ xcol)
# plt.tight_layout()
plt.show()


# ### 3.4  Movie Count  per Director

# In[ ]:


xcol = "Director"
df1 = pd.DataFrame({xcol:df[xcol]})
df1[xcol] = df1[xcol].apply(lambda x: re.sub("Directors: ", "", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub("Director: ", "", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub("[\(\)]", "", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" & ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" and ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub("/", ", ", x))

l = list()
for index, row in df1.iterrows():
    t = row[xcol].split(", ")
    l.extend([i for i in t if len(i.split(" "))>1])

df1 = pd.DataFrame({xcol:l})

c = df1[xcol].value_counts()
df1 = pd.DataFrame({xcol:c.index, "Count":c.values})
df1 = df1[df1[xcol]!="Unknown"]


# #### 3.4.1  Showing Movie Count for 30 Most Frequent Directors

# In[ ]:


n_show = 30
df2 = df1[df1["Count"]>n_show]
# print(len(df2))

# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(18, 30)
xcol = xcol

params = get_params()

plt.rcParams.update(params)
sns.barplot(x=df2["Count"], y=df2[xcol])
plt.title("Movie Count Per "+ xcol)
# plt.xticks(rotation=90)
# plt.yticks(rotation=90)

plt.show()


# ### 3.5  Movie Count  per Cast

# In[ ]:


xcol = "Cast"
df1 = pd.DataFrame({xcol:df[xcol]})
df1[xcol] = df1[xcol].fillna("None")
df1[xcol] = df1[xcol].apply(lambda x: re.sub("[\(\)]", "", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" & ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" and ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub("/", ", ", x))

l = list()
for index, row in df1.iterrows():
    t = row[xcol].split(", ")
    l.extend([i for i in t if len(i.split(" "))>1])

df1 = pd.DataFrame({xcol:l})

c = df1[xcol].value_counts()
df1 = pd.DataFrame({xcol:c.index, "Count":c.values})
df1 = df1[df1[xcol]!="None"]


# #### 3.5.1  Showing Movie Count for most frequent 50 Casts

# In[ ]:


n_show = 50
df2 = df1[df1["Count"]>n_show]
# print(len(df2))

# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(18, 30)
xcol = xcol

params = get_params()

plt.rcParams.update(params)
sns.barplot(x=df2["Count"], y=df2[xcol])
plt.title("Movie Count Per "+ xcol)
# plt.xticks(rotation=90)
# plt.yticks(rotation=90)
# plt.tight_layout()
plt.show()


# ### 3.6 Movie Count  per Genre

# In[ ]:


xcol = "Genre"
df1 = pd.DataFrame({xcol:df[xcol]})
df1[xcol] = df1[xcol].fillna("None")
df1[xcol] = df1[xcol].apply(lambda x: re.sub("[\(\)]", "", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" & ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub(" and ", ", ", x))
df1[xcol] = df1[xcol].apply(lambda x: re.sub("/", ", ", x))

l = list()
for index, row in df1.iterrows():
    t = row[xcol].split(", ")
    l.extend([i.strip() for i in t])

df1 = pd.DataFrame({xcol:l})

c = df1[xcol].value_counts()
df1 = pd.DataFrame({xcol:c.index, "Count":c.values})
df1 = df1[df1[xcol]!="None"]
df1 = df1[df1[xcol]!="unknown"]
df1[xcol] = df1[xcol].apply(lambda x: x.capitalize())


# #### 3.5.1  Showing Movie Count for most frequent 50 Genre

# In[ ]:


n_show = 50
df2 = df1[df1["Count"]>n_show]
# print(len(df2))

# sns.set(style="ticks")
sns.set(style="whitegrid")

figsize=(25, 28)
xcol = xcol

params = get_params()

plt.rcParams.update(params)
plt.tick_params(labelsize=18)
sns.barplot(x=df2["Count"], y=df2[xcol])
plt.title("Movie Count Per "+ xcol)
# plt.xticks(rotation=90)
# plt.yticks(rotation=90)

plt.show()


# In[ ]:


# !pip install wordcloud


# ### 3.6 Word Cloud

# In[ ]:


text = df["Plot"].str.cat(sep='. ')

stopwords = set(STOPWORDS)

wc = WordCloud(max_words=2000, stopwords=stopwords)

wc.generate(text)

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.show()


# ### Credit:
# 
# 1. http://4.bp.blogspot.com/-Tz2h4J4VdPw/UvezidpNOeI/AAAAAAAAJck/oHfz4EHTg7o/s1600/2013_movies_wallpaper_by_z_designs-d5u94o3.jpg
# 2. https://i.ytimg.com/vi/KPX39PpS1sc/maxresdefault.jpg
# 3. https://boygeniusreport.files.wordpress.com/2016/03/movies-tiles.jpg?quality=98&strip=all

# In[ ]:




