#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

import cufflinks as cf #importing plotly and cufflinks in offline mode  
import plotly.offline  
cf.go_offline()  
cf.set_config_file(offline=False, world_readable=True)

'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Netflix Visualization and Content-Based Recommendation system.**
# 
# Netflix is an application that keeps growing up the graph with its popularity, shows and content. This is an EDA or a story telling through its data along with a content-based recommendation system.
# 
# ![](https://media.foxbusiness.com/BrightCove/854081161001/201910/33/854081161001_6093736038001_6093735925001-vs.jpg)

# **Loading the dataset**

# In[ ]:


data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data['date_added'] = pd.to_datetime(data['date_added'])
data['Year_added'] = data['date_added'].dt.year
data['Month_added'] = data['date_added'].dt.month


# In[ ]:


data_TV = data[data['type']=='TV Show']
data_Movies = data[data['type']=='Movie']


# In[ ]:


data_TV['Year_added'].value_counts()


# In[ ]:


data_Movies['Year_added'].value_counts()


# In[ ]:


#data['listed_in'].unique()


# In[ ]:


#data['country'].unique()


# In[ ]:


data1=data[data['release_year']>=2000]
data1m=data1[data1['type']!="TV Show"]
data1t=data1[data1['type']!="Movie"]
data2m=pd.DataFrame(data1m['release_year'].value_counts()).reset_index()
data2m.rename(columns={'release_year':'count'},inplace=True)
movies=pd.DataFrame(['Movie']*data2m.shape[0],columns=['type'])
data2m=pd.concat([data2m,movies],axis=1)
data2t=pd.DataFrame(data1t['release_year'].value_counts()).reset_index()
data2t.rename(columns={'release_year':'count'},inplace=True)
Tv_shows=pd.DataFrame(['TV Shows']*data2t.shape[0],columns=['type'])
data2t=pd.concat([data2t,Tv_shows],axis=1)
data_final=pd.concat([data2m,data2t],ignore_index=True)
data_final.rename(columns={'index':'Release year'},inplace=True)
plt.figure(figsize=(16,6))

plt.title("Bargraph comparing the number of Movies and Tv shows from the year 2000 to 2020 ")

sns.barplot(x=data_final['Release year'],y=data_final['count'],hue=data_final['type'])


# In[ ]:


new_data = data[['type','listed_in','director','cast','country','rating','title','description']]
new_data.head()


# In[ ]:


get_ipython().system('pip install rake-nltk')
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# REMOVE NaN VALUES AND EMPTY STRINGS:
new_data.dropna(inplace=True)

blanks = []  # start with an empty list

col=['type','listed_in','director','cast','country','rating']
for i,col in new_data.iterrows():  # iterate over the DataFrame
    if type(col)==str:            # avoid NaN values
        if col.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

new_data.drop(blanks, inplace=True)


# In[ ]:


new_data.head(10)


# In[ ]:


new_data['Key_words/desc'] = ''

for i,n in new_data.iterrows():
    desc = n['description']
    r = Rake()
    r.extract_keywords_from_text(desc)
    score_for_keyword = r.get_word_degrees()
    n['Key_words/desc']=list(score_for_keyword.keys())
    
new_data.head(10)


# In[ ]:


new_data['cast'] = new_data['cast'].map(lambda x:x.split(',')[:3])
new_data['listed_in'] = new_data['listed_in'].map(lambda x:x.lower().split(','))
new_data['type'] = new_data['type'].map(lambda x:x.lower().split(','))
new_data['country'] = new_data['country'].map(lambda x:x.lower().split(','))
new_data['rating'] = new_data['rating'].map(lambda x:x.lower().split(','))
new_data['director'] = new_data['director'].map(lambda x:x.split(','))

new_data.drop('description',axis=1, inplace=True)

new_data.head(10)


# In[ ]:


for i,n in new_data.iterrows():
    n['cast'] = [x.lower().replace(' ','') for x in n['cast']]
    n['type'] = [x.lower().replace(' ','') for x in n['type']]
    n['rating'] = [x.lower().replace(' ','') for x in n['rating']]
    n['country'] = [x.lower().replace(' ','') for x in n['country']]
    n['director'] = ''.join(n['director']).lower()
    
new_data = new_data.set_index('title')
new_data.head(10)


# In[ ]:


new_data['bag_of_words'] = ''

cols = new_data.columns
for i,j in new_data.iterrows():
    words = ''
    for k in cols:
        if k!='director':
            words = words + ' '.join(j[k])+ ' '
        else:
            words = words + j[k] + ' '
            
    j['bag_of_words'] = words


# In[ ]:


new_data.head(10)


# In[ ]:


new_data.bag_of_words[1]


# In[ ]:


clean_data = new_data.drop(columns = [cols for cols in new_data.columns if cols!='bag_of_words'])
clean_data.head(10)


# In[ ]:


data[data.country=='India'][:5]


# In[ ]:


clean_data.loc['Article 15']['bag_of_words']


# In[ ]:


### Count vectorizer


# In[ ]:


count = CountVectorizer()
lol = count.fit_transform(clean_data['bag_of_words'])


# In[ ]:


#NLP

similarity_dekhna_hai = cosine_similarity(lol,lol)
similarity_dekhna_hai


# In[ ]:


similarity_dekhna_hai.shape


# In[ ]:


listy = pd.Series(clean_data.index)


# In[ ]:


listy[:5]


# In[ ]:


def recommendations(Title, cosine_sim = similarity_dekhna_hai):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = listy[listy == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(clean_data.index)[i])
        
    return recommended_movies


# In[ ]:


recommendations('Article 15')


# In[ ]:


recommendations('PK')


# In[ ]:


recommendations('3 Idiots')


# In[ ]:


recommendations('War Horse')


# In[ ]:




