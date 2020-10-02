#!/usr/bin/env python
# coding: utf-8

# ## Analysing the IMDB 5000 Movie Dataset & predicting the imdb score. 
# #### In this notebook I have worked on this given dataset and perform the following tasks:
# ##### 1. Analysed which movie genre has the highest average IMDB score. 
# ##### 2. Analysed some other features of the dataset and using the correlations found out the top 5 numerical features that influence the Imdb score.
# ##### 3. Used 3 different machine learning algorithms to predict the imdb score based on these 5 numerical features. 

# ### Importing all the libraries needed

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None) 


# In[ ]:


#Loading the dataset
df = pd.read_csv('../input/imdb-5000-movie-dataset/movie_metadata.csv')
df.sample(5)


# In[ ]:


df.columns,df.shape


# ### Checking for Null values in the dataset

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(),cmap='Blues',cbar=False,yticklabels=False)


# In[ ]:


# Movie with the lowest Imdb rating is Documentary `Justin Bieber: Never Say Never`
df_low_imdb=df[df['imdb_score']==1.6]
df_low_imdb


# In[ ]:


# Movie with the highest Imdb rating is Comedy `Towering Inferno`
df_max_imdb=df[df['imdb_score']==9.5]
df_max_imdb


# ### Creating a histogram of all the features in dataframe. 

# In[ ]:


df.hist(bins=30,figsize=(15,15),color='g')


# ### Creating a new column in the dataframe which shows the number of different genres a movie has. 

# In[ ]:


df['genres_num'] = df.genres.apply(lambda x: len(x.split('|')))


# In[ ]:


df.head(2)


# In[ ]:


df['genres_num'].max()


# In[ ]:


df[df['genres_num']==8]


# ### Creating a new column in the dataframe which replaces '|' in genres column to ',' 

# In[ ]:


df['Type_of_genres'] = df.genres.apply(lambda x: x.replace('|',','))
df.head(2)


# ### Creating a new column in the dataframe which shows only the first genre type of a movie. This will help to sort the movies on  single genre type.

# In[ ]:


df['genres_first'] = df.genres.apply(lambda x: x.split('|')[0] if '|' in x else x)


# In[ ]:


df.head()


# ### Analysing which movie genre has the best imdb score.
# #### From the graph it's clear that Documentaries are rated highest on Imdb whereas Thriller are least rated

# In[ ]:


plt.figure(figsize=(14,10))
sns.boxplot(x='imdb_score',y='genres_first',data=df)


# ### Finding which numerical features have the most influence on Imdb score. 
# #### selecting the top 5 features for my predictive model. 

# In[ ]:


correlations = df.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)


# ### The top 5 features that I am selecting for building my predictive model are following: 

# In[ ]:


df_for_ML = df[['num_critic_for_reviews','duration','director_facebook_likes','num_voted_users','num_user_for_reviews']]


# ### Maikng a pivot table to find the average imdb score for different movie genre.

# In[ ]:


pd.pivot_table(df,index='genres_first', values='imdb_score').sort_values('imdb_score', ascending = False)
# There seems some outliers here. As we saw clearly above in our boxplot that Documentary are highest rated on an average. 


# ### Number of movies released every year. 

# In[ ]:


df.title_year.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(20,20),color='g')


# ### Number of movies as per different genre type.

# In[ ]:


df.genres_first.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(10,10),color='g')


# ### Number of movies as per countries. 

# In[ ]:


df.country.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(20,20),color='g')


# In[ ]:


df_for_ML.head(2)


# ### Plotting graphs of all the feautres I selected for my ML model against the imdb score

# In[ ]:


for i in df_for_ML.columns:
    axis = df.groupby('imdb_score')[[i]].mean().plot(figsize=(10,5),marker='o',color='g')


# ### filling up all the missing values with the mean of all the values in that particular feature

# In[ ]:


df_for_ML["num_critic_for_reviews"] = df_for_ML["num_critic_for_reviews"].fillna(df["num_critic_for_reviews"].mean())
df_for_ML["duration"] = df_for_ML["duration"].fillna(df["duration"].mean())
df_for_ML["director_facebook_likes"] = df_for_ML["director_facebook_likes"].fillna(df["director_facebook_likes"].mean())
df_for_ML["num_user_for_reviews"] = df_for_ML["num_user_for_reviews"].fillna(df["num_user_for_reviews"].mean())
df_for_ML["num_voted_users"] = df_for_ML["num_voted_users"].fillna(df["num_voted_users"].mean())


# In[ ]:


sns.heatmap(df_for_ML.isnull(),cmap='Blues',cbar=False,yticklabels=False)


# In[ ]:


df_for_ML.info()


# ### Splitting the data for training and testing.

# In[ ]:


from sklearn.model_selection import train_test_split
X = df_for_ML
y = df['imdb_score']
X.shape,y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Using the linear regression model for predicting the the imdb score.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


prec_lm=lm.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print('The mean squared error using Linear regression is: ',mean_squared_error(y_test,prec_lm))
print('The mean absolute error using Linear regression is: ',mean_absolute_error(y_test,prec_lm))


# In[ ]:


### Using the Xgboost model for predicting the the imdb score.


# In[ ]:


from xgboost import XGBClassifier
Xgb = XGBClassifier()
Xgb.fit(X_train,y_train)


# In[ ]:


prec_Xgb=Xgb.predict(X_test)


# In[ ]:


print('The mean squared error using the Xgboost model is: ',mean_squared_error(y_test,prec_Xgb))
print('The mean absolute error using the Xgboost model is: ',mean_absolute_error(y_test,prec_Xgb))


# ### Using the Random Forest model for predicting the the imdb score.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:


prec_rf=rf.predict(X_test)


# In[ ]:


print('The mean squared error using Random Forest model is: ',mean_squared_error(y_test,prec_rf))
print('The mean absolute error using Random Forest model is: ',mean_absolute_error(y_test,prec_rf))


# ### Conclusions:
# 
# #### The random forest model has perforemed the best with MSE of '0.935' and MAE of '0.711'
# #### The documentries are on a average highest rated on IMDB.
# #### Number of users that vote for a particular movie influence the Imdb score the most.

# In[ ]:




