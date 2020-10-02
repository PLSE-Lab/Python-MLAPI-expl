#!/usr/bin/env python
# coding: utf-8

# # Reading and Knowing Data

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


imdb=pd.read_csv('../input/IMDB-Movie-Data.csv')


# In[4]:


imdb.head()


# # Data Visualization

# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import seaborn as sb


# In[7]:


imdb.describe()


# In[8]:


movie_yearly_count = imdb['Year'].value_counts().sort_index(ascending=False)


# In[9]:


movie_yearly_count


# In[10]:


movie_yearly_count.plot(kind='bar')


# In[11]:


movies_comparisons = ['Revenue (Millions)', 'Metascore', 'Runtime (Minutes)', 'Votes','Year']


# In[12]:


for comparison in movies_comparisons:
    sb.jointplot(x='Rating', y=comparison, data=imdb, alpha=0.5, color='g', size=10, space=0)


# In[13]:


import itertools


# In[14]:


unique_genres = imdb['Genre'].unique()
individual_genres = []
for genre in unique_genres:
    individual_genres.append(genre.split(','))

individual_genres = list(itertools.chain.from_iterable(individual_genres))
individual_genres = set(individual_genres)

individual_genres


# In[15]:


print('Number of movies in each genre:')

for genre in individual_genres:
    current_genre = imdb['Genre'].str.contains(genre).fillna(False)
    plt.figure()
    plt.xlabel('Year')
    plt.ylabel('Number of Movies Made')
    plt.title(str(genre))
    imdb[current_genre].Year.value_counts().sort_index().plot(kind='bar', color='g')
 
    print(genre, len(imdb[current_genre]))
   


# In[16]:


plt.plot( 'Votes', 'Rating', data=imdb, linestyle='', marker='+', markersize=4)
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.title('Voting VS Rating', loc='left')


# # Feature Engineering

# Since I am predicting the rating of movies before the release, I will be using the features that are relevant before the movie is released. 
# <n> 
#     <li>Runtime</li>
#     <li>Year</li>
#     <li>Genre</li>
#     <li>Description</li>
#     <li>Metascore</li>
#     <li>Votes</li>

# ## Value_count

# In[17]:


imdb['Genre'].value_counts()


# In[18]:


imdb['Director'].value_counts()


# ## Data_mapping

# In[19]:


data=[imdb]
data_mapping={'Action,Adventure,Sci-Fi':0,       
'Drama':1,                         
'Comedy,Drama,Romance':2,          
'Comedy':3,                        
'Drama,Romance':4,                 
'Comedy,Drama':5,                  
'Animation,Adventure,Comedy ':6,   
'Action,Adventure,Fantasy':7,      
'Comedy,Romance':8,                
'Crime,Drama,Thriller':9,          
'Crime,Drama,Mystery':10,           
'Action,Adventure,Drama':11,        
'Action,Crime,Drama':12,            
'Horror,Thriller':13,               
'Drama,Thriller':14,                
'Biography,Drama':15,               
'Biography,Drama,History':16,       
'Action,Adventure,Comedy':17,       
'Adventure,Family,Fantasy':18,      
'Action,Crime,Thriller':19,         
'Action,Comedy,Crime':20,           
'Horror':21,                        
'Action,Adventure,Thriller':22,     
'Crime,Drama':23,                   
'Action,Thriller':24,                
'Animation,Action,Adventure':25,     
'Biography,Crime,Drama':26,          
'Thriller':27,                       
'Horror,Mystery,Thriller':28,        
'Biography,Drama,Sport':29}

for dataset in data:
    dataset['genre']=dataset['Genre'].map(data_mapping)


# In[20]:


common_value='0'
imdb['genre']=imdb['genre'].fillna(common_value)


# In[21]:


imdb.head(1)


# In[22]:


imdb['Runtime (Minutes)'].value_counts().sort_index(ascending=True)


# In[23]:


imdb['Metascore'].describe()


# In[ ]:





# In[ ]:





# In[24]:


data=[imdb]
for dataset in data:
    dataset['Runtime (Minutes)']=dataset['Runtime (Minutes)'].astype(int)
    dataset.loc[ dataset['Runtime (Minutes)'] <= 66, 'Runtime (Minutes)'] = 1
    dataset.loc[(dataset['Runtime (Minutes)'] > 66) & (dataset['Runtime (Minutes)'] <= 90), 'Runtime (Minutes)'] = 2
    dataset.loc[(dataset['Runtime (Minutes)'] > 90) & (dataset['Runtime (Minutes)'] <= 120), 'Runtime (Minutes)'] = 3
    dataset.loc[(dataset['Runtime (Minutes)'] > 120) & (dataset['Runtime (Minutes)'] <= 140), 'Runtime (Minutes)'] = 4
    dataset.loc[(dataset['Runtime (Minutes)'] > 140) & (dataset['Runtime (Minutes)'] <= 160), 'Runtime (Minutes)'] = 5
    dataset.loc[(dataset['Runtime (Minutes)'] > 160) & (dataset['Runtime (Minutes)'] <= 180), 'Runtime (Minutes)'] = 6
    dataset.loc[dataset['Runtime (Minutes)'] > 180, 'Runtime (Minutes)']=7


# In[25]:


imdb['Runtime (Minutes)'].describe()


# In[ ]:





# In[26]:


imdb=imdb.fillna(imdb.median())


# In[27]:


imdb.loc[:,['Runtime (Minutes)','Year','genre','Metascore','Votes']]


# # Modelling

# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[29]:


X=imdb.loc[:,['Runtime (Minutes)','Year','genre','Metascore','Votes']]
y=imdb['Rating']


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)


# In[31]:


model_lnr=LinearRegression()


# In[32]:


model_lnr.fit(X_train,y_train)


# In[33]:


y_predict=model_lnr.predict(X_test)


# In[34]:


r2_score(y_test, y_predict)


# In[35]:


model_lnr.predict([[120,2019,1,75,757074]])


# - Random Forrest

# In[36]:


from sklearn.ensemble import RandomForestRegressor


# In[37]:


model2_RN=RandomForestRegressor(n_estimators=13)


# In[38]:


model2_RN.fit(X_train,y_train)


# In[39]:


y_predict2=model2_RN.predict(X_test)


# In[40]:


r2_score(y_test,y_predict2)


# **KFold**

# In[41]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[42]:


kfold=KFold(n_splits=10, shuffle=True, random_state=0 )


# In[43]:


clf = RandomForestRegressor()
score = cross_val_score(clf, X, y, cv=kfold, n_jobs=1)
print(score)


# In[44]:


clf.fit(X_train,y_train)


# In[45]:


clf.predict([[2,120,2015,2,9]])


# In[46]:


clf2=LinearRegression()
score= cross_val_score(clf2, X, y, cv=kfold, n_jobs=1)


# In[47]:


clf2.fit(X_train,y_train)


# In[48]:


clf2.predict([[6,120,2015,2,9]])


# In[ ]:




