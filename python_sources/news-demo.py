#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and setting up

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


import matplotlib.pyplot as py
import seaborn as sns
from nltk.corpus import stopwords as stop
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer


# # Preprocesing starts

# In[ ]:


df = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json", lines=True)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# So from the above command, we got the know that there are a total of 200853 records,and all the majoz news are about Politics having the headline "Sunday Roundup". There are in total 41 categories, 199344 headlines and 27993 authors.

# In[ ]:


df.info()


# In[ ]:


df = df[df['date']>pd.Timestamp(2018,1,1)]


# In[ ]:


df.shape


# In[ ]:


df = df[df['headline'].apply(lambda x:len(x.split())>5)]
df.shape


# in the above code we will remove all short headlines, as in future when we will remove stop words, they may become null
# In the above, we were able to see that we had 53 such headlines, so we removed all that data

# In[ ]:


df.sort_values('headline', inplace=True, ascending=False)
print(df.shape)
df_duplicated=df.duplicated('headline',keep=False)
print(df.shape)
df = df[~df_duplicated]
print(df.shape)


# In the above code, we first sorted the data in descending and then exactracted the duplicate data. And at the end, removed the duplicates by using '~'

# In[ ]:


# lets check if any of the cell is empty or has an ambigiious value
df.isna().sum()


# Hence, we see that no cell is empty

# In[ ]:


df.describe()


# So after doing pre-processing, we found out that we are left with 26 categories, 8453 headlines and 865 authors

# In[ ]:


index = df['category'].value_counts().index
index.shape


# In[ ]:


values = df['category'].value_counts().values
type(values)


# In[ ]:


py.bar(df['category'].value_counts().index, df['category'].value_counts(),width=0.8)
py.show()


# In[ ]:


news_per_month= df.resample('M', on= "date")['headline'].count()
news_per_month


# In[ ]:


py.figure()
py.title('Month wise distribution')
py.xlabel('Month')
py.ylabel('Number of articles')
py.bar(news_per_month.index.strftime('%b'), news_per_month, width=0.8)


# In[ ]:


sns.distplot(df['headline'].str.len(), hist=False)


# In[ ]:


df['day and month']= df['date'].dt.strftime("%a")+'_'+df['date'].dt.strftime('%b')


# In[ ]:


df.index= range(df.shape[0])
df_temp=df.copy()


# # NLTK

# In[ ]:


stop_words = stop.words('english')


# In[ ]:


for i  in range(len(df_temp["headline"])):
    string=""
    for word in df_temp["headline"][i].split():
        word = ("".join(e for e in word if e.isalpha()))
        word = word.lower()
        if not word in stop_words:
            string += word +" " 
    if i%500 == 0:
        print(i)
    df_temp.at[i,'headline']= string.strip()


# In[ ]:


from nltk.stem import WordNetLemmatizer
lemitizer = WordNetLemmatizer()


# In[ ]:


for i in range(len(df_temp['headline'])):
    string=""
    for w in df_temp['headline'][i]:
        string += lemitizer.lemmatize(w,pos='v')+" "
    print(string)
    df_temp.at[i,'headline'] += string.strip()


# In[ ]:


df_temp['headline'][0]


# # Training

# ## Bag of Words Algo

# In[ ]:


vectorize = CountVectorizer()
vectorize_features = vectorize.fit_transform(df_temp['headline'])
vectorize_features.shape


# In[ ]:


pd.set_option('display.max_colwidth', -1)

# to get the biggest possible headline to display


# In[ ]:


def bag_of_words_model(row_index, output_values):
    # to find the distance of the featutres of row_index corresponding to all the other rows
    couple_dist =  pairwise_distances(vectorize_features, vectorize_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:output_values]
    df1 = pd.DataFrame(
        {'publish_date':df['date'][indices].values, 
         'headline':df['headline'][indices].values,
         'euclidean distance':couple_dist[indices].ravel()
        }
    )
    print("headline: ",df['headline'][indices[0]])
    return (df1.iloc[1:,])

bag_of_words_model(133, 11)


# From the above, we were able to see that we are really not getting much accuracy, as 

# ## TFIDF Algorithm

# In[ ]:


vectorizer = TfidfVectorizer(min_df=0)
tfidf_headline_features = vectorize.fit_transform(df_temp['headline'])


# In[ ]:


def tfidf_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df1 = pd.DataFrame({'publish_date': df['date'][indices].values,
               'headline':df['headline'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',df['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    
    #return df.iloc[1:,1]
    return df1.iloc[1:,]

tfidf_based_model(133, 11)

