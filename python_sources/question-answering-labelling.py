#!/usr/bin/env python
# coding: utf-8

# **More to Come **
# 
# **please Upvote to Encourage **
# 
# ![](https://static2.proactiveinvestors.co.uk/thumbs/upload/News/Image/2019_08/672z311_1566568270_2019-08-23-14-51-10_743428f5cc30f464556b0ee235f06a3d.jpg)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# charts
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib
import squarify #TreeMap

# import graph objects as "go"

import plotly.graph_objs as go

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Reading Data**

# In[ ]:


train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")


# # **Sneak Peak in Data**

# In[ ]:


train.head()


# In[ ]:


test.head()


# # **Data From Eagle View**

# In[ ]:


train.info()


# In[ ]:


test.info()


# **Data distribution** - based on category

# In[ ]:


len(train["category"].unique())
sns.countplot(train["category"])


# **Data distribution** - length of Q&A

# In[ ]:


def lengthdataframe(data) :
    df_temp = [len(x) for x in data['question_title']]
    df_length = pd.DataFrame(data= df_temp, columns=["len_Question Title"])
    df_length['len_question_body'] = [len(x) for x in data['question_body']]
    df_length['len_answer'] = [len(x) for x in data['answer']]
    return df_length


# In[ ]:


def custom_distplot(dataframe_length) :
    fig, axes = plt.subplots(ncols=len(dataframe_length.columns), nrows=1,figsize=(15,3))
    for column, ax in zip(dataframe_length.columns, axes.flat):
        sns.distplot(dataframe_length[column],  ax=ax)
    plt.show()


# In[ ]:


dftrain_length_QA = lengthdataframe(train)
custom_distplot(dftrain_length_QA)


# In[ ]:


dftest_length_QA = lengthdataframe(test)
custom_distplot(dftest_length_QA)


# In[ ]:


dftrain_length_QA.describe()


# In[ ]:


dftest_length_QA.describe()


# # **Data distribution** - Based on host

# In[ ]:


train["host"].unique()


# In[ ]:


host = train.groupby("host").agg({'qa_id' : 'count'}).nlargest(20, 'qa_id')
host = host.sort_values(by='qa_id')
value=np.array(host)
#label=[df_dept.loc[x]["department"] for x in dept.index]
label= [x.split(".")[1]+ "(" +str(host.loc[x].qa_id)+")" if "meta." in x else str(x.split(".")[0])+ "(" +str(host.loc[x].qa_id)+")"  for x in host.index ]

# create a color palette, mapped to these values
cmap = matplotlib.cm.CMRmap
mini=min(value)
maxi=max(value)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(values)) for values in value]


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 10.0)
squarify.plot(sizes=value, label=label, alpha=.8)
plt.axis('off')
plt.show()


# # **Data distribution** - A look at output

# In[ ]:


categorical = train.columns[(train.dtypes.values == np.dtype('float64'))]
categorical


# In[ ]:


fig, axes = plt.subplots(ncols=int(len(categorical)/6), nrows=6,figsize=(25,25))
for column, ax in zip(categorical, axes.flat):
    sns.distplot(train[column],  ax=ax)
plt.show()


# In[ ]:


train[categorical].describe()


# **Correlation - Length of Q&A vs Output**

# In[ ]:


result = pd.concat([train[categorical], dftrain_length_QA], axis=1, sort=False)


# In[ ]:


f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(result.corr(method ='pearson'),annot=True,fmt= '.1f', ax=ax )


# # **Creating WordCloud Based on the Category**

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 
).generate(str(data))
    return wordcloud


# In[ ]:


categories = train["category"].unique()


# In[ ]:


fig, axes = plt.subplots(ncols=2, nrows=3,figsize=(30,25))
plt.axis('off')
for category, ax in zip(categories, axes.flat):
    wordcloud = show_wordcloud(train[train["category"]==category]['question_title'])
    ax.imshow(wordcloud)
    ax.title.set_text(category)
    ax.axis('off')
plt.subplots_adjust(wspace=0.05, hspace=0.01)


#    ![Thank You](https://tricksbystg.org/wp-content/uploads/2018/04/Thanks-gif-1.gif)
