#!/usr/bin/env python
# coding: utf-8

# **Introduction**

# Fun and finanace arnt friends and they are in most cases used paradoxically. thimgs like ...hard works pay with no or little fun, all day movies are the streanght of the weak, burn the candles overnight so as not to burn the volts at old age..... I am a strong fan of the sage. Statements like this are in most cases my watch word.  For further thought and reflections, are this words truely true?
# 
# cant we find furtune in fun? purpose in jokes, career in movies..... if we are to ponder over this kernel topic, we may at the end find an ace up our sleeve. As a friend rightly define popcorn movie has the movie we watch for entartainment but he has speedily forgoten that the fun and pleasure derived from the movie helps in better reflection to the cores of rightful defination of what life exactly is. We can think on these things 
# 
# Box office are places where ticket are sold and revenue predicted for a perticular movies. As i know, not only the box office but also cinemas, internet and so on.... Revenue are largly generated from internet this days. Let us use our analysis of data to predict movie revenue and help the industries.
# 
# Let us begin!!!!!!!!!!!!

# **About The Data**

# In this competition, you're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries.
# 
# We are predicting the worldwide revenue for 4398 movies in the test file.

# Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
import eli5
import shap
from catboost import CatBoostRegressor
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


# In[ ]:


import os
IS_LOCAL = False
if (IS_LOCAL):
    location = "../input/tmdb-box-office/"
else:
    location = "../input/"
os.listdir(location)


# In[ ]:


get_ipython().run_cell_magic('time', '', "xtrain = pd.read_csv(os.path.join(location, 'train.csv'))\nxtest = pd.read_csv(os.path.join(location, 'test.csv'))\nsam_sub = pd.read_csv(os.path.join(location, 'sample_submission.csv'))")


# We have 3000 train sample with 23 features and 4398 test samples with 22 feature

# In[ ]:


print("Xtrain: {}\nXtest: {}:".format(xtrain.shape, xtest.shape))


# In[ ]:


def show_head(data):
    return(data.head())


# In[ ]:


show_head(xtrain)
xtrain.columns


# In[ ]:


show_head(xtest)
xtest.columns


# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (total/data.isnull().count()*100)
    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    miss_column['Types'] = types
    return(np.transpose(miss_column))


# I noticed that we are having many missing values in this train file. The belonging to coliection column has about 80 percent missing values. This has already negate the column. If we fill the column we will be at the danger of incorrect data.
# 
# genres is good column we need to focus on because the type of genres can determine customers. Language is a good features too. Title and so on 
# 

# In[ ]:


missing_data(xtrain)


# **Data Exploration**
# 
# We will take the columns one after the other and do healing to the sick parts lol.....

# In[ ]:


missing_data(xtest)


# **Belong_to_Collection**
# 
# We will extract datas of the films that belongs to collection
# 
# The useful info here is the collection names and the fact that movie belongs to a collection. I will creat a column for the names of the collections and another for belonging to a collection. the column for belonging to a collection will be a binary classification 0 or 1. 0 for not belonging to a collection(empty) and 1 fpr belonging to collection.

# In[ ]:


dict_features = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_dict(df):
    for col in dict_features:
        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
xtrain = text_dict(xtrain)
xtest = text_dict(xtest)


# In[ ]:


for i, e in enumerate(xtrain['belongs_to_collection'][:5]):
    print(i, e)


# 2396 does not belong to any collection and no movie belongs to more than 1 collection. That makes it easy for us

# In[ ]:


xtrain['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()


# In[ ]:


xtrain['CollectionName'] = xtrain['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
xtrain['BelongCollection'] = xtrain['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

xtest['CollectionName'] = xtest['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
xtest['BelongCollection'] = xtest['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

xtrain = xtrain.drop(['belongs_to_collection'], axis=1)
xtest = xtest.drop(['belongs_to_collection'], axis=1)


# In[ ]:


show_head(xtrain)


# **Genres**

# A fim genres is a motion-picture category based om similarities in emotional responses. From defination, this column will show the level of how film will influence viewers base on their similarities. We should focus on this column.
# 
# From the look of things, it shows that a some of this films are much alike that the others. Those ones with number of genres will have more viewers I guess. Those films with 6, 7, 0 genres could be counted as outliers.
# 
# We should check on the list of the genres to know how many are they and to see if we can create columns for them. 

# In[ ]:


for i, e in enumerate(xtrain['genres'][:5]):
    print(i, e)


# In[ ]:


print('Number of Genres per movie')
xtrain['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()


# In[ ]:


GenresList = list(xtrain['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)


# From the wordcloud we see that DRAMA, COMEDY AND THRILLER often appear in the films. This is real because this three genres are most watch in reality. Any film that belongs to of this genres could have the potential of high revenue due to high number of customers.
# 
# I will create a column for top 10. This top 10 will have a great impact in out prediction.

# In[ ]:


plt.figure(figsize = (16, 12))
text = ' '.join([i for j in GenresList for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('NumberGenres')
plt.axis("off")
plt.show()


# I guess we should remove the AllGenres column to avoid noise and overfitting. I will remove it latter.

# In[ ]:


xtrain['GenresNumb'] = xtrain['genres'].apply(lambda x: len(x) if x != {} else 0)
xtrain['AllGenres'] = xtrain['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_genres = [m[0] for m in Counter([i for j in GenresList for i in j]).most_common(10)]
for g in top_genres:
    xtrain['genre_' + g] = xtrain['AllGenres'].apply(lambda x: 1 if g in x else 0)
    
xtest['GenresNumb'] = xtest['genres'].apply(lambda x: len(x) if x != {} else 0)
xtest['AllGenres'] = xtest['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_genres:
    xtest['genre_' + g] = xtest['AllGenres'].apply(lambda x: 1 if g in x else 0)

xtrain = xtrain.drop(['genres'], axis=1)
xtest = xtest.drop(['genres'], axis=1)


# In[ ]:


xtrain.columns
show_head(xtrain)


# **Production Company**
# 
# I believe this column is less important because many viewers knows absolute nothing about the company producing the movie.
# 
# Let us explore it first and extract some content then decision on it will be later things. Intrestingly the movies have more than one production company.

# In[ ]:


for i, e in enumerate(xtrain['production_companies'][:5]):
    print(i, e)
print('number of producing company')
xtrain['production_companies'].apply(lambda x: len(x) if x!= {} else 0).value_counts()


# In[ ]:


xtrain[xtrain['production_companies'].apply(lambda x: len(x) if x!= {} else 0)>7]

ProdCompList = list(xtrain['production_companies'].apply(lambda x: [i['name'] for i in x] if x!= {}
                                                         else []).values)

xtrain['ProdCompNumb'] = xtrain['production_companies'].apply(lambda x: len(x) if x!= {} else 0)
xtrain['AllProdComp'] = xtrain['production_companies'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_ProdComp = [m[0] for m in Counter([i for j in ProdCompList for i in j]).most_common(10)]
for g in top_ProdComp:
    xtrain['production_company_' + g] = xtrain['production_companies'].apply(lambda x: 1 if g in x else 0)
    


xtest['ProdCompNumb'] = xtest['production_companies'].apply(lambda x: len(x) if x!= {} else 0)
xtest['AllProdComp'] = xtest['production_companies'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_ProdComp = [m[0] for m in Counter([i for j in ProdCompList for i in j]).most_common(10)]
for g in top_ProdComp:
    xtest['production_company_' + g] = xtest['production_companies'].apply(lambda x: 1 if g in x else 0)
    
xtrain = xtrain.drop(['production_companies'], axis=1)
xtest = xtest.drop(['production_companies'], axis=1)


# In[ ]:


xtrain.columns


# **Production Countries**
# 
# The country of production may have impact depending on the social nature of the country and the language. We all know most movie seen by even we are 80% american movie. Let us see. some movies were produced in more than one countries. Most movies are produced in one country.

# In[ ]:


for i, e in enumerate(xtrain['production_countries'][:5]):
    print(i, e)
print('number of producing countries')
xtrain['production_countries'].apply(lambda x: len(x) if x!= {} else 0).value_counts()


# This confirms my guess. The USA top the list followed by Kingdom and france.
# 
# We will create column for the first five

# In[ ]:


ProdCountryList = list(xtrain['production_countries'].apply(lambda x: [i['name'] for i in x] if x!= {}
                                                         else []).values)
plt.figure(figsize = (16, 12))
text = ' '.join([i for j in ProdCountryList for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('NumberProdCountry')
plt.axis("off")
plt.show()


# In[ ]:


xtrain['ProdCountryNumb'] = xtrain['production_countries'].apply(lambda x: len(x) if x!= {} else 0)
xtrain['AllProdCountry'] = xtrain['production_countries'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_ProdCountry = [m[0] for m in Counter([i for j in ProdCountryList for i in j]).most_common(5)]
for g in top_ProdCountry:
    xtrain['production_country_' + g] = xtrain['production_countries'].apply(lambda x: 1 if g in x else 0)
    


xtest['ProdCountryNumb'] = xtest['production_countries'].apply(lambda x: len(x) if x!= {} else 0)
xtest['AllProdCountry'] = xtest['production_countries'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_ProdCountry = [m[0] for m in Counter([i for j in ProdCountryList for i in j]).most_common(5)]
for g in top_ProdCountry:
    xtest['production_country_' + g] = xtest['production_countries'].apply(lambda x: 1 if g in x else 0)
    
xtrain = xtrain.drop(['production_countries'], axis=1)
xtest = xtest.drop(['production_countries'], axis=1)


# In[ ]:


xtrain.columns


# **Spoken Language**
# 
# This also attracts our focus base on the country that mostly appear in our data.

# In[ ]:


for i, e in enumerate(xtrain['spoken_languages'][:5]):
    print(i, e)
    
print('Number of spoken languages')
xtrain['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()

LanguageList = list(xtrain['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

plt.figure(figsize = (16, 12))
text = ' '.join([i for j in LanguageList for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('NumberOfLanguage')
plt.axis("off")
plt.show()


# In[ ]:


xtrain['LangNumb'] = xtrain['spoken_languages'].apply(lambda x: len(x) if x!= {} else 0)
xtrain['AllLang'] = xtrain['spoken_languages'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_Lang = [m[0] for m in Counter([i for j in LanguageList for i in j]).most_common(10)]
for g in top_Lang:
    xtrain['spoken_language_' + g] = xtrain['spoken_languages'].apply(lambda x: 1 if g in x else 0)
    


xtest['LangNumb'] = xtest['spoken_languages'].apply(lambda x: len(x) if x!= {} else 0)
xtest['AllLang'] = xtest['spoken_languages'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_Lang = [m[0] for m in Counter([i for j in LanguageList for i in j]).most_common(10)]
for g in top_Lang:
    xtest['spoken_language_' + g] = xtest['spoken_languages'].apply(lambda x: 1 if g in x else 0)
    
xtrain = xtrain.drop(['spoken_languages'], axis=1)
xtest = xtest.drop(['spoken_languages'], axis=1)


# In[ ]:


xtrain.columns


# **Keywords**

# In[ ]:


for i, e in enumerate(xtrain['Keywords'][:5]):
    print(i, e)
    
print('Number of Keywords in films')
xtrain['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

KeywordsList = list(xtrain['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
plt.figure(figsize = (16, 12))
text = ' '.join(['_'.join(i.split(' ')) for j in KeywordsList for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top keywords')
plt.axis("off")
plt.show()


# In[ ]:


xtrain['KeywordsNumb'] = xtrain['Keywords'].apply(lambda x: len(x) if x!= {} else 0)
xtrain['AllKeywords'] = xtrain['Keywords'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_Keyword = [m[0] for m in Counter([i for j in KeywordsList for i in j]).most_common(40)]
for g in top_Keyword:
    xtrain['Keyword_' + g] = xtrain['Keywords'].apply(lambda x: 1 if g in x else 0)
    


xtest['KeywordsNumb'] = xtest['Keywords'].apply(lambda x: len(x) if x!= {} else 0)
xtest['AllKeywords'] = xtest['Keywords'].apply(lambda x: ''.join(sorted([i['name'] for i in x])) if x!= {} else '')
top_Keyword = [m[0] for m in Counter([i for j in KeywordsList for i in j]).most_common(40)]
for g in top_Keyword:
    xtest['Keyword_' + g] = xtest['Keywords'].apply(lambda x: 1 if g in x else 0)
    
xtrain = xtrain.drop(['Keywords','AllKeywords','AllLang','AllProdCountry','AllProdComp','AllGenres'], axis=1)
xtest = xtest.drop(['Keywords','AllKeywords','AllLang','AllProdCountry','AllProdComp','AllGenres'], axis=1)


# In[ ]:


xtrain.columns


# **Cast**

# In[ ]:


for i, e in enumerate(xtrain['cast'][:1]):
    print(i, e)

print('Number of casted persons in films')
xtrain['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

CastNameList = list(xtrain['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

CastGenderList = list(xtrain['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

CastCharacterList = list(xtrain['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)


# In[ ]:


xtrain['num_cast'] = xtrain['cast'].apply(lambda x: len(x) if x != {} else 0)
top_cast_names = [m[0] for m in Counter([i for j in CastNameList for i in j]).most_common(15)]
for g in top_cast_names:
    xtrain['cast_name_' + g] = xtrain['cast'].apply(lambda x: 1 if g in str(x) else 0)
xtrain['genders_0_cast'] = xtrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
xtrain['genders_1_cast'] = xtrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
xtrain['genders_2_cast'] = xtrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
top_cast_characters = [m[0] for m in Counter([i for j in CastCharacterList for i in j]).most_common(15)]
for g in top_cast_characters:
    xtrain['cast_character_' + g] = xtrain['cast'].apply(lambda x: 1 if g in str(x) else 0)
    
xtest['num_cast'] = xtest['cast'].apply(lambda x: len(x) if x != {} else 0)
for g in top_cast_names:
    xtest['cast_name_' + g] = xtest['cast'].apply(lambda x: 1 if g in str(x) else 0)
xtest['genders_0_cast'] = xtest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
xtest['genders_1_cast'] = xtest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
xtest['genders_2_cast'] = xtest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
for g in top_cast_characters:
    xtest['cast_character_' + g] = xtest['cast'].apply(lambda x: 1 if g in str(x) else 0)

xtrain = xtrain.drop(['cast'], axis=1)
xtest = xtest.drop(['cast'], axis=1)


# In[ ]:


xtrain.columns


# **Crew **

# In[ ]:


for i, e in enumerate(xtrain['crew'][:1]):
    print(i, e[:10])
    
print('Number of casted persons in films')
xtrain['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

list_of_crew_names = list(xtrain['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_of_crew_jobs = list(xtrain['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

list_of_crew_genders = list(xtrain['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

list_of_crew_departments = list(xtrain['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)


# In[ ]:


xtrain['num_crew'] = xtrain['crew'].apply(lambda x: len(x) if x != {} else 0)
top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
for g in top_crew_names:
    xtrain['crew_name_' + g] = xtrain['crew'].apply(lambda x: 1 if g in str(x) else 0)
xtrain['genders_0_crew'] = xtrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
xtrain['genders_1_crew'] = xtrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
xtrain['genders_2_crew'] = xtrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
top_cast_characters = [m[0] for m in Counter([i for j in CastCharacterList for i in j]).most_common(15)]
for g in top_cast_characters:
    xtrain['crew_character_' + g] = xtrain['crew'].apply(lambda x: 1 if g in str(x) else 0)
top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
for j in top_crew_jobs:
    xtrain['jobs_' + j] = xtrain['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
for j in top_crew_departments:
    xtrain['departments_' + j] = xtrain['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 
    
xtest['num_crew'] = xtest['crew'].apply(lambda x: len(x) if x != {} else 0)
for g in top_crew_names:
    xtest['crew_name_' + g] = xtest['crew'].apply(lambda x: 1 if g in str(x) else 0)
xtest['genders_0_crew'] = xtest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
xtest['genders_1_crew'] = xtest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
xtest['genders_2_crew'] = xtest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
for g in top_cast_characters:
    xtest['crew_character_' + g] = xtest['crew'].apply(lambda x: 1 if g in str(x) else 0)
for j in top_crew_jobs:
    xtest['jobs_' + j] = xtest['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
for j in top_crew_departments:
    xtest['departments_' + j] = xtest['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

xtrain = xtrain.drop(['crew'], axis=1)
xtest = xtest.drop(['crew'], axis=1)


# In[ ]:


show_head(xtrain)


# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(xtrain['revenue']);
plt.title('Distribution of revenue');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(xtrain['revenue']));
plt.title('Distribution of log of revenue');


# In[ ]:


xtrain['log_revenue'] = np.log1p(xtrain['revenue'])


# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(xtrain['budget']);
plt.title('Distribution of budget');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(xtrain['budget']));
plt.title('Distribution of log of budget');


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(xtrain['budget'], xtrain['revenue'])
plt.title('Revenue vs budget');
plt.subplot(1, 2, 2)
plt.scatter(np.log1p(xtrain['budget']), xtrain['log_revenue'])
plt.title('Log Revenue vs log budget');


# In[ ]:


xtrain['log_budget'] = np.log1p(xtrain['budget'])
xtest['log_budget'] = np.log1p(xtest['budget'])


# In[ ]:


xtrain['homepage'].value_counts().head()


# In[ ]:


xtrain['has_homepage'] = 0
xtrain.loc[xtrain['homepage'].isnull() == False, 'has_homepage'] = 1
xtest['has_homepage'] = 0
xtest.loc[xtest['homepage'].isnull() == False, 'has_homepage'] = 1


# In[ ]:


sns.catplot(x='has_homepage', y='revenue', data=xtrain);
plt.title('Revenue for film with and without homepage');


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=xtrain.loc[xtrain['original_language'].isin(xtrain['original_language'].value_counts().head(10).index)]);
plt.title('Mean revenue per language');
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=xtrain.loc[xtrain['original_language'].isin(xtrain['original_language'].value_counts().head(10).index)]);
plt.title('Mean log revenue per language');


# In[ ]:


plt.figure(figsize = (12, 16))
text = ' '.join(xtrain['original_title'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in titles')
plt.axis("off")
plt.show()


# In[ ]:


plt.figure(figsize = (12, 12))
text = ' '.join(xtrain['overview'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in overview')
plt.axis("off")
plt.show()


# In[ ]:


vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            min_df=5)

overview_text = vectorizer.fit_transform(xtrain['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, xtrain['log_revenue'])
eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')


# In[ ]:


print('Target value:', xtrain['log_revenue'][1000])
eli5.show_prediction(linreg, doc=xtrain['overview'].values[1000], vec=vectorizer)


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(xtrain['popularity'], xtrain['revenue'])
plt.title('Revenue vs popularity');
plt.subplot(1, 2, 2)
plt.scatter(xtrain['popularity'], xtrain['log_revenue'])
plt.title('Log Revenue vs popularity');


# In[ ]:


xtest.loc[xtest['release_date'].isnull() == True, 'release_date'] = '01/01/98'

def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year
    
xtrain['release_date'] = xtrain['release_date'].apply(lambda x: fix_date(x))
xtest['release_date'] = xtest['release_date'].apply(lambda x: fix_date(x))
xtrain['release_date'] = pd.to_datetime(xtrain['release_date'])
xtest['release_date'] = pd.to_datetime(xtest['release_date'])

def process_date(df):
    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    
    return df

xtrain = process_date(xtrain)
xtest = process_date(xtest)


# In[ ]:


d1 = xtrain['release_date_year'].value_counts().sort_index()
d2 = xtest['release_date_year'].value_counts().sort_index()
data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]
layout = go.Layout(dict(title = "Number of films per year",
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))


# to be continued......
# If you like my kernel please upvote and leave a comment
# 
# Many thanks to  Andrew Lukyanenko
# 
# https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation

# In[ ]:


d1 = xtrain['release_date_year'].value_counts().sort_index()
d2 = xtrain.groupby(['release_date_year'])['revenue'].sum()
data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='total revenue', yaxis='y2')]
layout = go.Layout(dict(title = "Number of films and total revenue per year",
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Count'),
                  yaxis2=dict(title='Total revenue', overlaying='y', side='right')
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))


# In[ ]:


d1 = xtrain['release_date_year'].value_counts().sort_index()
d2 = xtrain.groupby(['release_date_year'])['revenue'].mean()
data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='mean revenue', yaxis='y2')]
layout = go.Layout(dict(title = "Number of films and average revenue per year",
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Count'),
                  yaxis2=dict(title='Average revenue', overlaying='y', side='right')
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))


# this shows that films release on wednesday have higher revenue. This makes this column important in our prediction

# In[ ]:


sns.catplot(x='release_date_weekday', y='revenue', data=xtrain);
plt.title('Revenue on different days of week of release');


# **Runtime**

# This shows that films with more than one hour have high revenues but popular films are with short time. Comedy are popular and are short. This makes us have faith on this column

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
plt.hist(xtrain['runtime'].fillna(0) / 60, bins=40);
plt.title('Distribution of length of film in hours');
plt.subplot(1, 3, 2)
plt.scatter(xtrain['runtime'].fillna(0), xtrain['revenue'])
plt.title('runtime vs revenue');
plt.subplot(1, 3, 3)
plt.scatter(xtrain['runtime'].fillna(0), xtrain['popularity'])
plt.title('runtime vs popularity');


# **Status**

# In[ ]:


xtrain['status'].value_counts()


# In[ ]:


xtest['status'].value_counts()


# **Tagline**

# In[ ]:


plt.figure(figsize = (18, 16))
text = ' '.join(xtrain['tagline'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in tagline')
plt.axis("off")
plt.show()


# **Collections**

# In[ ]:


show_head(xtrain)


# So our films with collection has more income that is interesting

# In[ ]:


sns.boxplot(x='BelongCollection', y='revenue', data=xtrain);


# **Genres**

# In[ ]:


sns.catplot(x='GenresNumb', y='revenue', data=xtrain);
plt.title('Revenue for different number of genres in the film');


# In[ ]:


f, axes = plt.subplots(3, 5, figsize=(24, 12))
plt.suptitle('Violinplot of revenue vs genres')
for i, e in enumerate([col for col in xtrain.columns if 'genre_' in col]):
    sns.violinplot(x=e, y='revenue', data=xtrain, ax=axes[i // 5][i % 5]);


# **Production Companies**

# In[ ]:


f, axes = plt.subplots(6, 5, figsize=(24, 32))
plt.suptitle('Violinplot of revenue vs production company')
for i, e in enumerate([col for col in xtrain.columns if 'production_company' in col]):
    sns.violinplot(x=e, y='revenue', data=xtrain, ax=axes[i // 5][i % 5]);


# **Production Countries**

# In[ ]:


sns.catplot(x='ProdCountryNumb', y='revenue', data=xtrain);
plt.title('Revenue for different number of countries producing the film');


# **Cast**

# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(xtrain['num_cast'], xtrain['revenue'])
plt.title('Number of cast members vs revenue');
plt.subplot(1, 2, 2)
plt.scatter(xtrain['num_cast'], xtrain['log_revenue'])
plt.title('Log Revenue vs number of cast members');


# **Keywords**

# In[ ]:


f, axes = plt.subplots(6, 5, figsize=(24, 32))
plt.suptitle('Violinplot of revenue vs keyword')
for i, e in enumerate([col for col in xtrain.columns if 'keyword_' in col]):
    sns.violinplot(x=e, y='revenue', data=xtrain, ax=axes[i // 5][i % 5]);


# **Crew**

# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(xtrain['num_crew'], xtrain['revenue'])
plt.title('Number of crew members vs revenue');
plt.subplot(1, 2, 2)
plt.scatter(xtrain['num_crew'], xtrain['log_revenue'])
plt.title('Log Revenue vs number of crew members');


# **Modelling the Game**
# 
# 
# Now the boring and the mindless lifting is over.....Let us model some demons our of this sucker files

# In[ ]:


show_head(xtrain)


# In[ ]:


for col in ['original_language', 'CollectionName']:
    le = LabelEncoder()
    le.fit(list(xtrain[col].fillna('')) + list(xtest[col].fillna('')))
    xtrain[col] = le.transform(xtrain[col].fillna('').astype(str))
    xtest[col] = le.transform(xtest[col].fillna('').astype(str))


# In[ ]:


show_head(xtrain)


# In[ ]:


train_texts = xtrain[['title', 'tagline', 'overview', 'original_title']]
test_texts = xtest[['title', 'tagline', 'overview', 'original_title']]


# In[ ]:


for col in ['title', 'tagline', 'overview', 'original_title']:
    xtrain['len_' + col] = xtrain[col].fillna('').apply(lambda x: len(str(x)))
    xtrain['words_' + col] = xtrain[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    xtrain = xtrain.drop(col, axis=1)
    xtest['len_' + col] = xtest[col].fillna('').apply(lambda x: len(str(x)))
    xtest['words_' + col] = xtest[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    xtest = xtest.drop(col, axis=1)


# In[ ]:


show_head(xtrain)


# In[ ]:


xtrain.fillna(xtrain.mean(), inplace=True)
xtest.fillna(xtest.mean(), inplace=True)


# In[ ]:


ytrain = np.log1p(xtrain['revenue'])
xtrain = xtrain.drop(['id', 'revenue', 'homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue'], axis=1)
xtest = xtest.drop(['id', 'homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)


# In[ ]:



xtrain.shape


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(xtrain, ytrain, test_size=0.1, random_state=4, shuffle=True)


# **Random Forest Regression**
# 
# We will build our model first with RFC.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor as rfr

clf_rfr = rfr(n_estimators=1000, random_state=4)
clf_rfr.fit(x_train, y_train)


# In[ ]:


predictions = clf_rfr.predict(xtest)
pred = clf_rfr.predict(x_valid)


# Evaluating the clf itself

# In[ ]:


errors = abs(pred - y_valid)
print('Mean Absolute Error:',round(np.mean(errors), 2), 'degrees.')


# In[ ]:


mape = 100 * (errors/y_valid)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


sam_sub['revenue'] = np.expm1(predictions)
sam_sub.to_csv("RFC.csv", index=False)


# 

# In[ ]:





# In[ ]:




