#!/usr/bin/env python
# coding: utf-8

# # TMDB Box Office Movie's Revenue Prediction
# 
# ![](https://cdn.onebauer.media/one/empire-tmdb/films/284054/images/6ELJEzQJ3Y45HczvreC3dg0GV5R.jpg?quality=50&width=1800&ratio=16-9&resizeStyle=aspectfill&format=jpg)
# 
# ### My job to predict the international box office revenue for each movie in this given dataset

# ### Import all necesary libraries

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from collections import OrderedDict


# In[ ]:


# Load dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.nunique()


# ### Exploratory Data Analysis

# #### Data Analysis

# In[ ]:


train['revenue'].describe()


# In[ ]:


train['budget'].describe()


# In[ ]:


train.plot.scatter('budget','revenue')


# In[ ]:


# top ten movies with the costliest budgets.
budget = train.sort_values(by='budget', ascending=False)
print((budget.loc[:, "budget"]).head(10))


# In[ ]:


# Top ten movies with the highet revenue
revenue = train.sort_values(by='revenue', ascending=False)
print("Top ten movies with the highet revenue\n")
print((revenue.loc[:,"revenue"]).head(10))


# In[ ]:


#first removing features which are irrelevant for our prediction
train.drop(['imdb_id','poster_path'],axis=1,inplace=True)
test.drop(['imdb_id','poster_path'],axis=1,inplace=True)


# #### Missing or Null Values in the given dataset

# In[ ]:


#we have a lot of null values for homepage
#Converting homepage as binary
train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1

#Homepage v/s Revenue
sns.catplot(x='has_homepage', y='revenue', data=train);
plt.title('Movies revenues with and without homepage');


# In[ ]:


train = train.drop(['homepage'],axis =1)
test = test.drop(['homepage'],axis =1)


# In[ ]:


#Converting collections as binary
train['collection'] = 0
train.loc[train['belongs_to_collection'].isnull() == False, 'collection'] = 1
test['collection'] = 0
test.loc[test['belongs_to_collection'].isnull() == False, 'collection'] = 1

#collections v/s Revenue
sns.catplot(x='collection', y='revenue', data=train);
plt.title('Movies Revenue with and without collection');


# In[ ]:


#Collection too increaes the revenue
train=train.drop(['belongs_to_collection'],axis =1)
test=test.drop(['belongs_to_collection'],axis =1)


# ### Lanaguage

# #### Find the Most Profitable Movie languages wise

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
ax.tick_params(axis='both', labelsize=12)
plt.title('Original Language and Revenue', fontsize=20)
plt.xlabel('Revenue', fontsize=16)
plt.ylabel('Original Language', fontsize=16)
sns.boxplot(ax=ax, x='revenue', y='original_language', data=train, showfliers=False, orient='h')
plt.show()


# Some languages seem to attract greater audiences than others and end up generating more revenue. For example, the highest revenue movies are in English, Chinese and Turkish. ('en', 'zh' and 'tr'). Hindi ('hi') and Japanese ('ja') are not far behind.

# In[ ]:


#How language contributes to revenue
plt.figure(figsize=(15,11)) #figure size

#It's another way to plot our data. using a variable that contains the plot parameters
g1 = sns.boxenplot(x='original_language', y='revenue', 
                   data=train[(train['original_language'].isin((train['original_language'].sort_values().value_counts()[:10].index.values)))])
g1.set_title("Revenue by language", fontsize=20) # title and fontsize
g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # It's the way to rotate the xticks when we use variable to our graphs
g1.set_xlabel('Language', fontsize=18) # Xlabel
g1.set_ylabel('Revenue', fontsize=18) #Ylabel

plt.show()


# #### Most Common Languages

# In[ ]:


plt.figure(figsize = (12, 8))
text = ' '.join(train['original_language'])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=800).generate(text)
plt.imshow(wordcloud)
plt.title('Top Languages', fontsize=20)
plt.axis("off")
plt.show()


# The most common languages in the movie data seem to be English ('en'), French ('fr'), Russian ('ru'), Hindi ('hi') etc

# In[ ]:


#Taking only en and zh into consideration as they are the highest grossing
train['original_language'] = train['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))
test['original_language'] = test['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))


# ### Genre

# Visualize the relationship between the genre and revenue of the movie
# 

# In[ ]:


genres = []
repeated_revenues = []
for i in range(len(train)):
  if train['genres'][i] == train['genres'][i]:
      movie_genre = [genre['name'] for genre in eval(train['genres'][i])]
      genres.extend(movie_genre)
      repeated_revenues.extend([train['revenue'][i]]*len(movie_genre))
  
genre = pd.DataFrame(np.zeros((len(genres), 2)))
genre.columns = ['genre', 'revenue']
genre['genre'] = genres
genre['revenue'] = repeated_revenues


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
ax.tick_params(axis='both', labelsize=12)
plt.title('Genres and Revenue', fontsize=20)
plt.xlabel('revenue', fontsize=16)
plt.ylabel('genre', fontsize=16)
sns.boxplot(ax=ax, x=repeated_revenues, y=genres, showfliers=False, orient='h')
plt.show()


# It looks like some movie genres tend to earn more revenue than others on average. Animation and Adventure movies lead the way in terms of revenue, but Family and Fantasy are not far behind.

# In[ ]:


#adding number of genres for each movie
genres_count=[]
for i in train['genres']:
    if(not(pd.isnull(i))):
        
        genres_count.append(len(eval(i)))
        
    else:
        genres_count.append(0)
train['num_genres'] = genres_count


# In[ ]:


#Genres v/s revenue
sns.catplot(x='num_genres', y='revenue', data=train);
plt.title('Revenue for different number of genres in the film');


# In[ ]:


plt.figure(figsize = (12, 8))
text = ' '.join(genres)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=800).generate(text)
plt.imshow(wordcloud)
plt.title('Top Genres', fontsize=30)
plt.axis("off")
plt.show()


# The most common movie genres seem to be Drama, Comedy, Thriller, Action and Adventure.

# In[ ]:


#Adding genres count for test data
genres_count_test=[]
for i in test['genres']:
    if(not(pd.isnull(i))):
        
        genres_count_test.append(len(eval(i)))
        
    else:
        genres_count_test.append(0)
test['num_genres'] = genres_count_test


# In[ ]:


#Dropping genres
train.drop(['genres'],axis=1, inplace = True)
test.drop(['genres'],axis=1, inplace = True)


# #### Production Company

# In[ ]:


#Adding production_companies count for data
prod_comp_count=[]
for i in train['production_companies']:
    if(not(pd.isnull(i))):
        
        prod_comp_count.append(len(eval(i)))
        
    else:
        prod_comp_count.append(0)
train['num_prod_companies'] = prod_comp_count


# In[ ]:


#number of prod companies vs revenue
sns.catplot(x='num_prod_companies', y='revenue', data=train)
plt.title('Revenue for different number of production companies in the film')


# In[ ]:


#Adding production_companies count for  test data
prod_comp_count_test=[]
for i in test['production_companies']:
    if(not(pd.isnull(i))):
        
        prod_comp_count_test.append(len(eval(i)))
        
    else:
        prod_comp_count_test.append(0)
test['num_prod_companies'] = prod_comp_count_test


# In[ ]:


#number of prod companies vs revenue
sns.catplot(x='num_prod_companies', y='revenue', data=train);
plt.title('Revenue for different number of production companies in the film');


# In[ ]:


#Dropping production_companies
train.drop(['production_companies'],axis=1, inplace = True)
test.drop(['production_companies'],axis=1, inplace = True)


# #### Production Countries

# In[ ]:


#Adding production_countries count for  data
prod_coun_count=[]
for i in train['production_countries']:
    if(not(pd.isnull(i))):
        
        prod_coun_count.append(len(eval(i)))
        
    else:
        prod_coun_count.append(0)
train['num_prod_countries'] = prod_coun_count


# In[ ]:


#number of prod countries vs revenue
sns.catplot(x='num_prod_countries', y='revenue', data=train);
plt.title('Revenue for different number of production countries in the film');


# In[ ]:


#Adding production_countries count for  test data
prod_coun_count_test=[]
for i in test['production_countries']:
    if(not(pd.isnull(i))):
        
        prod_coun_count_test.append(len(eval(i)))
        
    else:
        prod_coun_count_test.append(0)
test['num_prod_countries'] = prod_coun_count_test


# In[ ]:


#Dropping production_countries
train.drop(['production_countries'],axis=1, inplace = True)
test.drop(['production_countries'],axis=1, inplace = True)


# In[ ]:


#handling overview
#mapping overview present to 1 and nulls to 0
train['overview']=train['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)
test['overview']=test['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)
sns.catplot(x='overview', y='revenue', data=train);
plt.title('Revenue for film with and without overview');


# In[ ]:


train = train.drop(['overview'],axis=1)
test = test.drop(['overview'],axis=1)


# #### Movie Cast

# In[ ]:


#cast
#Adding cast count for  data
total_cast=[]
for i in train['cast']:
    if(not(pd.isnull(i))):
        
        total_cast.append(len(eval(i)))
        
    else:
        total_cast.append(0)
train['cast_count'] = total_cast


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(train['cast_count'], train['revenue'])
plt.title('Number of cast members vs revenue');


# In[ ]:


#cast
#Adding cast count for  test data
total_cast=[]
for i in test['cast']:
    if(not(pd.isnull(i))):
        
        total_cast.append(len(eval(i)))
        
    else:
        total_cast.append(0)
test['cast_count'] = total_cast


# In[ ]:


#Dropping cast
train = train.drop(['cast'],axis=1)
test = test.drop(['cast'],axis=1)


# #### Crew

# In[ ]:


#crew
total_crew=[]
for i in train['crew']:
    if(not(pd.isnull(i))):
        
        total_crew.append(len(eval(i)))
        
    else:
        total_crew.append(0)
train['crew_count'] = total_crew


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(train['crew_count'], train['revenue'])
plt.title('Number of crew members vs revenue');


# In[ ]:


#Adding crew count for  test data
total_crew=[]
for i in test['crew']:
    if(not(pd.isnull(i))):
        
        total_crew.append(len(eval(i)))
        
    else:
        total_crew.append(0)
test['crew_count'] = total_crew


# In[ ]:


#Dropping crew
train = train.drop(['crew'],axis=1)
test = test.drop(['crew'],axis=1)


# In[ ]:


#Dropping original_title
train = train.drop(['original_title'],axis=1)
test = test.drop(['original_title'],axis=1)


# ### Check correlation between variables

# In[ ]:


col = ['revenue','budget','popularity','runtime']

plt.subplots(figsize=(10, 8))

corr = train[col].corr()

sns.heatmap(corr, xticklabels=col,yticklabels=col, linewidths=.5, cmap="Reds")


# In[ ]:


#budget and revenue are highly correlated
sns.regplot(x="budget", y="revenue", data = train)


# #### Release Date

# In[ ]:


#Check how revenue depends of day
train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])


# In[ ]:


release_day = train['release_date'].value_counts().sort_index()
release_day_revenue = train.groupby(['release_date'])['revenue'].sum()
release_day_revenue.index = release_day_revenue.index.dayofweek
sns.barplot(release_day_revenue.index,release_day_revenue, data = train,ci=None)
plt.show()


# In[ ]:


#adding day feature to the data

train['release_day'] = train['release_date'].dt.dayofweek 
test['release_day'] = test['release_date'].dt.dayofweek 


# In[ ]:


#filling nulls in test
test['release_day'] = test['release_day'].fillna(0)


# In[ ]:


train.drop(['release_date'],axis=1,inplace=True)
test.drop(['release_date'],axis=1,inplace=True)


# In[ ]:


#status
print("train data")
print(train['status'].value_counts())
print("test data")
test['status'].value_counts()


# In[ ]:


#Feature is irrelevant hence dropping
train.drop(['status'],axis=1,inplace =True)
test.drop(['status'],axis=1,inplace =True)


# In[ ]:


#keywords
Keywords_count=[]
for i in train['Keywords']:
    if(not(pd.isnull(i))):
        
        Keywords_count.append(len(eval(i)))
        
    else:
        Keywords_count.append(0)
train['Keywords_count'] = Keywords_count


# In[ ]:


#number of prod countries vs revenue
sns.catplot(x='Keywords_count', y='revenue', data=train);
plt.title('Revenue for different number of Keywords in the film');


# In[ ]:


Keywords_count=[]
for i in test['Keywords']:
    if(not(pd.isnull(i))):
        
        Keywords_count.append(len(eval(i)))
        
    else:
        Keywords_count.append(0)
test['Keywords_count'] = Keywords_count


# In[ ]:


#Dropping title and keywords
train = train.drop(['Keywords'],axis=1)
train = train.drop(['title'],axis=1)
test = test.drop(['Keywords'],axis=1)
test = test.drop(['title'],axis=1)


# #### Tagline 

# In[ ]:



train['isTaglineNA'] = 0
train.loc[train['tagline'].isnull() == False, 'isTaglineNA'] = 1
test['isTaglineNA'] = 0
test.loc[test['tagline'].isnull() == False, 'isTaglineNA'] = 1

#Homepage v/s Revenue
sns.catplot(x='isTaglineNA', y='revenue', data = train);
plt.title('Revenue for film with and without tagline');


# In[ ]:


train.drop(['tagline'],axis=1,inplace =True)
test.drop(['tagline'],axis=1,inplace =True)


# In[ ]:


#runtime has 2 nulls; setting it to the mean
#filling nulls in test
train['runtime'] = train['runtime'].fillna(train['runtime'].mean())
test['runtime'] = test['runtime'].fillna(test['runtime'].mean())


# #### Spoken languages

# In[ ]:


#adding number of spoken languages for each movie
spoken_count=[]
for i in train['spoken_languages']:
    if(not(pd.isnull(i))):
        
        spoken_count.append(len(eval(i)))
        
    else:
        spoken_count.append(0)
train['spoken_count'] = spoken_count


spoken_count_test=[]
for i in test['spoken_languages']:
    if(not(pd.isnull(i))):
        
        spoken_count_test.append(len(eval(i)))
        
    else:
        spoken_count_test.append(0)
test['spoken_count'] = spoken_count_test


# In[ ]:


#dropping spoken_languages
train.drop(['spoken_languages'],axis=1,inplace=True)
test.drop(['spoken_languages'],axis=1,inplace=True)


# In[ ]:


train.info()


# Let's check dataset

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train['budget'] = np.log1p(train['budget'])
test['budget'] = np.log1p(test['budget'])


# ### Prepare dataset for train the Model

# In[ ]:


y= train['revenue'].values
cols = [col for col in train.columns if col not in ['revenue', 'id']]
X= train[cols].values
y = np.log1p(y)


# ### Traning the model
# 
# 1. Linear Regression
# 2. Random Forest Regression

# ### Model1: Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
scores = cross_val_score(clf, X, y, scoring="neg_mean_squared_error", cv = 8)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean())


# ### Model2: Random Forest Regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, min_samples_split=5, random_state = 10,
                             n_estimators=500)
scores = cross_val_score(regr, X, y, scoring="neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean())


# ### # Testing the model

# In[ ]:


cols = [col for col in test.columns if col not in ['id']]
X_test= test[cols].values


# In[ ]:


regr.fit(X,y)
y_pred = regr.predict(X_test)


# In[ ]:


y_pred=np.expm1(y_pred)
pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_RF.csv', index=False)


# ![](https://i.pinimg.com/originals/42/44/d8/4244d86f56e6b8d25c8bd67732165021.gif)

# I hope this kernal is useful to you.
# 
# If find this notebook help you to learn, **Please Upvote**.

# ![](https://media.tenor.com/images/ba3ec917b6414b01fa85d33979336864/tenor.gif)
