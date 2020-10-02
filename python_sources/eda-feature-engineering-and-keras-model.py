#!/usr/bin/env python
# coding: utf-8

# 

# This kernel is dedicated for TMDB revenue prediction challenge.In this kernel i have done
# - Getting started with TMDB
# - Cleaning TMDB data
# - Exploratory data analysis of TMDB data
# - feature engineering
# - Keras model
# - model Evaluvation

# **if you like my kernel,please do consider upvoting it**

# ![](https://media.giphy.com/media/WZ4M8M2VbauEo/giphy.gif)

# ### problem statement.

# In a world... where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate."
# 
# In this competition, you're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release.

# ### Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn import preprocessing
from datetime import datetime
import ast
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.metrics import mean_squared_logarithmic_error
from sklearn.model_selection import KFold


# ###  Loading dataset

# In[ ]:





# In[ ]:


df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')


# ### Getting a basic ideas about the data

# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


df_train.info()


# In[ ]:


df_train.head(5)


# In[ ]:


df_train.describe(include='all')


# ### Missing values

# In[ ]:


df_train.isna().sum().sort_values(ascending=False)


# In[ ]:


missing=df_train.isna().sum().sort_values(ascending=False)
sns.barplot(missing[:8],missing[:8].index)
plt.show()


# There are many variable with large number of null values,we will inspect those variables first.

# In[ ]:


plt.style.use('seaborn')


# In[ ]:


import ast
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
dfx = text_to_dict(df_train)
for col in dict_columns:
       df_train[col]=dfx[col]


# #### belongs to collection

# 
# 
# 
# - belongs_to_collection - Contains the TMDB Id, Name, Movie Poster and Backdrop URL of a movie in JSON format. You can see the Poster and Backdrop Image like this: https://image.tmdb.org/t/p/original/. Example: https://image.tmdb.org/t/p/original//iEhb00TGPucF0b4joM1ieyY026U.jpg

# In[ ]:


df_train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()


# Only 604 films belong to some collections

# In[ ]:


collections=df_train['belongs_to_collection'].apply(lambda x : x[0]['name'] if x!= {} else '?').value_counts()[1:15]
sns.barplot(collections,collections.index)
plt.show()


# We can observe that james bond collection films,friday the 13th,         
# The pink panther stands first among the number of films released 
# in particular collection series

# ### Tagline

# - tagline : The tagline which was assosiated with the film

# In[ ]:


df_train['tagline'].apply(lambda x:1 if x is not np.nan else 0).value_counts()


# - A wordcloud using taglines

# In[ ]:


plt.figure(figsize=(10,10))
taglines=' '.join(df_train['tagline'].apply(lambda x:x if x is not np.nan else ''))

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(taglines)
plt.imshow(wordcloud)
plt.title('Taglines')
plt.axis("off")
plt.show()


# #### Keywords

# In[ ]:


keywords=df_train['Keywords'].apply(lambda x: ' '.join(i['name'] for i in x) if x != {} else '')
plt.figure(figsize=(10,10))
data=' '.join(words for words in keywords)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(data)
plt.imshow(wordcloud)
plt.title('Taglines')
plt.axis("off")
plt.show()


# ### Production companies

# - The most famous production companies and the number films released by each

# In[ ]:


x=df_train['production_companies'].apply(lambda x : [x[i]['name'] for i in range(len(x))] if x != {} else []).values
Counter([i for j in x for i in j]).most_common(20)


# ### Production countries

# In[ ]:


countries=df_train['production_countries'].apply(lambda x: [i['name'] for i in x] if x!={} else []).values
count=Counter([j for i in countries for j in i]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# - These are the countries in which most films are released.
#   USA stand first and way above 
#   from other countries in terms of number of films assosiated with countries. 

# ### Spoken languages

# - This indicates the number of languages spoken in a film

# In[ ]:


df_train['spoken_languages'].apply(lambda x:len(x) if x !={} else 0).value_counts()


# - we can see that in most films there is only one language spoken in it.
# - Most number of languages spoken in a film is 9.

# Now we will inspect the languages spoken

# In[ ]:


lang=df_train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in lang for i in j]).most_common(5)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# - As expected English comes first followed by French

# ### Genre

# - genres : Contains all the Genres Name & TMDB Id in JSON Format

# We will inspect which genre films are most common

# In[ ]:


genre=df_train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else [])
count=Counter([i for j in genre for i in j]).most_common(10)
sns.barplot([val[1] for val in count],[val[0] for val in count])


# - Drama is the most common genre followed by comedy and thrillers.

# In[ ]:


dfx = text_to_dict(df_test)
for col in dict_columns:
       df_test[col]=dfx[col]


# ### Revenue

# - This is our target variable.
# - We will inspect the distribution first.

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('skewed data')
sns.distplot(df_train['revenue'])
plt.subplot(1,2,2)
plt.title('log transformation')
sns.distplot(np.log(df_train['revenue']))
plt.show()


# - The target variable is skewed,so we will log transform it to obtain a standard distribution.

# In[ ]:


df_train['log_revenue']=np.log1p(df_train['revenue'])


# - Histogram 

# In[ ]:


plt.subplots(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_train['revenue'],bins=10,color='g')
plt.title('skewed data')
plt.subplot(1,2,2)
plt.hist(np.log(df_train['revenue']),bins=10,color='g')
plt.title('log transformation')
plt.show()


# In[ ]:


df_train['revenue'].describe()


# ### Budget

# In[ ]:


plt.subplots(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_train['budget']+1,bins=10,color='g')
plt.title('skewed data')
plt.subplot(1,2,2)
plt.hist(np.log(df_train['budget']+1),bins=10,color='g')
plt.title('log transformation')
plt.show()


# #### Revenue vs budget

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.scatterplot(df_train['budget'],df_train['revenue'])
plt.subplot(1,2,2)
sns.scatterplot(np.log1p(df_train['budget']),np.log1p(df_train['revenue']))
plt.show()


# 

# - We dont see any linear relationship among budget and revenue.

# In[ ]:


df_train['log_budget']=np.log1p(df_train['budget'])
df_test['log_budget']=np.log1p(df_train['budget'])


# ### Popularity

# In[ ]:


plt.hist(df_train['popularity'],bins=30,color='violet')
plt.show()


# ### Popularity vs revenue

# In[ ]:


sns.scatterplot(df_train['popularity'],df_train['revenue'],color='violet')
plt.show()


# ### Extracting date features

# In[ ]:


def date(x):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<19:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
df_train['release_date']=df_train['release_date'].fillna('1/1/90').apply(lambda x: date(x))
df_test['release_date']=df_test['release_date'].fillna('1/1/90').apply(lambda x: date(x))


# In[ ]:


#from datetime import datetime
df_train['release_date']=df_train['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
df_test['release_date']=df_test['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))


# In[ ]:


df_train['release_day']=df_train['release_date'].apply(lambda x:x.weekday())
df_train['release_month']=df_train['release_date'].apply(lambda x:x.month)
df_train['release_year']=df_train['release_date'].apply(lambda x:x.year)


# In[ ]:


df_test['release_day']=df_test['release_date'].apply(lambda x:x.weekday())
df_test['release_month']=df_test['release_date'].apply(lambda x:x.month)
df_test['release_year']=df_test['release_date'].apply(lambda x:x.year)


# ### Release day of week

# In[ ]:


day=df_train['release_day'].value_counts().sort_index()
sns.barplot(day.index,day)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='45')
plt.ylabel('No of releases')


# - We can see that most films are released on friday.
# - This might be because of some strategy .

# #### Is there any relation between release day and revenue?

# In[ ]:


sns.catplot(x='release_day',y='revenue',data=df_train)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')
plt.show()


# #### Is there any relation between runtime and revenue?

# In[ ]:


sns.catplot(x='release_day',y='runtime',data=df_train)
plt.gca().set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation='90')
plt.show()


# #### Which months yeilds the maximum revenue?

# In[ ]:


plt.figure(figsize=(10,15))
sns.catplot(x='release_month',y='revenue',data=df_train)
month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
plt.gca().set_xticklabels(month_lst,rotation='90')
plt.show()


# - The months of April,may and june yeilds maximum revenue.

# ### Year vs revenue

# In[ ]:


plt.figure(figsize=(15,8))
yearly=df_train.groupby(df_train['release_year'])['revenue'].agg('mean')
plt.plot(yearly.index,yearly)
plt.xlabel('year')
plt.ylabel("Revenue")
plt.savefig('fig')


# - Revenue from films seems increasing and decreasing throughout the years.
# - There is a steep increase in revenue after 2017 or so.

# ### Runtime

# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(np.log1p(df_train['runtime'].fillna(0)))

plt.subplot(1,2,2)
sns.scatterplot(np.log1p(df_train['runtime'].fillna(0)),np.log1p(df_train['revenue']))


# ### Homepage

# In[ ]:


df_train['homepage'].value_counts().sort_values(ascending=False)[:5]


# 
# #### Inspecting revenue ,budget ,popularity and runtime of each genre

# In[ ]:


genres=df_train.loc[df_train['genres'].str.len()==1][['genres','revenue','budget','popularity','runtime']].reset_index(drop=True)
genres['genres']=genres.genres.apply(lambda x :x[0]['name'])


# In[ ]:


genres=genres.groupby(genres.genres).agg('mean')


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.barplot(genres['revenue'],genres.index)

plt.subplot(2,2,2)
sns.barplot(genres['budget'],genres.index)

plt.subplot(2,2,3)
sns.barplot(genres['popularity'],genres.index)

plt.subplot(2,2,4)
sns.barplot(genres['runtime'],genres.index)


# ### Crew

# In[ ]:



crew=df_train['crew'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in crew for i in j]).most_common(15)


# - There are the most fomous and common crew members
# - Number of films in which they appeared is also shown.

# ### Cast

# In[ ]:


cast=df_train['cast'].apply(lambda x:[i['name'] for i in x] if x != {} else [])
Counter([i for j in cast for i in j]).most_common(15)


# - There are the actors that have appeared in most films

# ### Feature Engineering

# In reference with [kernel](https://www.kaggle.com/zero92/eda-tmdb-box-office-prediction) by B.H

# In[ ]:


def  prepare_data(df):
    df['_budget_runtime_ratio'] = (df['budget']/df['runtime']).replace([np.inf,-np.inf,np.nan],0)
    df['_budget_popularity_ratio'] = df['budget']/df['popularity']
    df['_budget_year_ratio'] = df['budget'].fillna(0)/(df['release_year']*df['release_year'])
    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']
    df['budget']=np.log1p(df['budget'])
    
    df['collection_name']=df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
    df['has_homepage']=0
    df.loc[(pd.isnull(df['homepage'])),'has_homepage']=1
    
    le=LabelEncoder()
    le.fit(list(df['collection_name'].fillna('')))
    df['collection_name']=le.transform(df['collection_name'].fillna('').astype(str))
    
    le=LabelEncoder()
    le.fit(list(df['original_language'].fillna('')))
    df['original_language']=le.transform(df['original_language'].fillna('').astype(str))
    
    df['_num_Keywords'] = df['Keywords'].apply(lambda x: len(x) if x != {} else 0)
    df['_num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)
    
    df['isbelongto_coll']=0
    df.loc[pd.isna(df['belongs_to_collection']),'isbelongto_coll']=1
    
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0 ,"isTaglineNA"] = 1 

    df['isOriginalLanguageEng'] = 0 
    df.loc[ df['original_language'].astype(str) == "en" ,"isOriginalLanguageEng"] = 1
    
    df['ismovie_released']=1
    df.loc[(df['status']!='Released'),'ismovie_released']=0
    
    df['no_spoken_languages']=df['spoken_languages'].apply(lambda x: len(x))
    df['original_title_letter_count'] = df['original_title'].str.len() 
    df['original_title_word_count'] = df['original_title'].str.split().str.len() 


    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    
    
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x : np.nan if len(x)==0 else x[0]['id'])
    df['production_countries_count'] = df['production_countries'].apply(lambda x : len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x : len(x))
    df['cast_count'] = df['cast'].apply(lambda x : len(x))
    df['crew_count'] = df['crew'].apply(lambda x : len(x))

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

    for col in  ['genres', 'production_countries', 'spoken_languages', 'production_companies'] :
        df[col] = df[col].map(lambda x: sorted(list(set([n if n in train_dict[col] else col+'_etc' for n in [d['name'] for d in x]])))).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    df.drop(['genres_etc'], axis = 1, inplace = True)
    
    cols_to_normalize=['runtime','popularity','budget','_budget_runtime_ratio','_budget_year_ratio','_budget_popularity_ratio','_releaseYear_popularity_ratio',
    '_releaseYear_popularity_ratio2','_num_Keywords','_num_cast','no_spoken_languages','original_title_letter_count','original_title_word_count',
    'title_word_count','overview_word_count','tagline_word_count','production_countries_count','production_companies_count','cast_count','crew_count',
    'genders_0_crew','genders_1_crew','genders_2_crew']
    for col in cols_to_normalize:
        print(col)
        x_array=[]
        x_array=np.array(df[col].fillna(0))
        X_norm=normalize([x_array])[0]
        df[col]=X_norm
    
    df = df.drop(['belongs_to_collection','genres','homepage','imdb_id','overview','id'
    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'
    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline', 'collection_id'
    ],axis=1)
    
    df.fillna(value=0.0, inplace = True) 

    return df

    


# In[ ]:


def get_json(df):
    global dict_columns
    result=dict()
    for col in dict_columns:
        d=dict()
        rows=df[col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']]=0
                else:
                    d[i['name']]+=1
            result[col]=d
    return result
    
    

    
train_dict=get_json(df_train)
test_dict=get_json(df_test)


# In[ ]:


df_train.shape


# In[ ]:


for col in dict_columns :
    
    remove = []
    train_id = set(list(train_dict[col].keys()))
    test_id = set(list(test_dict[col].keys()))   
    
    remove += list(train_id - test_id) + list(test_id - train_id)
    for i in train_id.union(test_id) - set(remove) :
        if train_dict[col][i] < 10 or i == '' :
            remove += [i]
    for i in remove :
        if i in train_dict[col] :
            del train_dict[col][i]
        if i in test_dict[col] :
            del test_dict[col][i]
                  
    

            


# ### Splitting train and test

# In[ ]:


df_test['revenue']=np.nan
all_data=prepare_data((pd.concat([df_train,df_test]))).reset_index(drop=True)
train=all_data.loc[:df_train.shape[0]-1,:]
test=all_data.loc[df_train.shape[0]:,:]
print(train.shape)


# In[ ]:


all_data.head()


# In[ ]:


train.drop('revenue',axis=1,inplace=True)


# In[ ]:


y=train['log_revenue']
X=train.drop(['log_revenue'],axis=1)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
kfold=KFold(n_splits=3,random_state=42,shuffle=True)


# In[ ]:


X.columns


# ### Keras model

# In[ ]:


from keras import optimizers

    
model=models.Sequential()
model.add(layers.Dense(356,activation='relu',kernel_regularizer=regularizers.l1(.001),input_shape=(X.shape[1],)))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(256,kernel_regularizer=regularizers.l1(.001),activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.rmsprop(lr=.001),loss='mse'
,metrics=['mean_squared_logarithmic_error'])


# In[ ]:



epochs=40


# In[ ]:


hist=model.fit(X_train,y_train,epochs=epochs,verbose=0,validation_data=(X_test,y_test))


# ### Mean absolute error

# In[ ]:



mae=hist.history['mean_squared_logarithmic_error']
plt.plot(range(1,epochs),mae[1:],label='mae')
plt.xlabel('epochs')
plt.ylabel('mean_abs_error')
mae=hist.history['val_mean_squared_logarithmic_error']
plt.plot(range(1,epochs),mae[1:],label='val_mae')
plt.legend()


# ### Loss

# In[ ]:


mae=hist.history['loss']
plt.plot(range(1,epochs),mae[1:],label='trraining loss')
plt.xlabel('epochs')
plt.ylabel('loss')
mae=hist.history['val_loss']
plt.plot(range(1,epochs),mae[1:],label='val_loss')
plt.legend()


# ### Making my submission

# In[ ]:


test.drop(['revenue','log_revenue'],axis=1,inplace=True)


# In[ ]:


y=np.expm1(model.predict(test))
df_test['revenue']=y
df_test[['id','revenue']].to_csv('submission.csv',index=False)


# **If you like my kernel please consider upvoting,Thank you !**
