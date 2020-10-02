#!/usr/bin/env python
# coding: utf-8

# # Analysing the Metadata on over 45,000 movies & predicting the Movie Profit.
# #### In this notebook I have worked on this movies meta dataset & credits dataset to perform the following tasks:
# ##### Cleaned the Movies_metadataset and merged it with credits.csv to get more information about a movie. For example: director name, actor name.
# ##### Picked out some important information from the combined dataset to create new columns like: country of production, production companies, spoken language, actor 1 name, actor 2 name, director name etc.
# ##### Analysed the combined dataset using some important visualizations. 
# ##### Using correlations found out the most important numerical features that influence the profit of movies the most. 
# ##### Used Linear regression & Random forest models to predict the Profit of movies based on these important numerical features. 

# ### Importing all the libraries needed

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib.inline', '')


# ### Importing the datasets.

# In[ ]:


df_credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
df_credits.head(1)


# In[ ]:


df_meta = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
df_meta.head(1)


# In[ ]:


df_meta.release_date.dtype


# ### Converting the date to datetime and using year to create a new column.

# In[ ]:


df_meta['release_date'] = pd.to_datetime(df_meta['release_date'],errors='coerce')
df_meta['year'] = df_meta['release_date'].dt.year


# ### Cleaning up the Id column in Movie_metadata & converting it to 'int' & merging the 2 dataframes.

# In[ ]:


df_meta['id'] = df_meta['id'].apply(lambda x: x.replace('-','0'))
df_meta['id'] = df_meta['id'].astype(int)


# In[ ]:


merged_data = pd.merge(df_meta,df_credits,on='id')


# In[ ]:


merged_data.sample(1)


# ### Picking up only the important features from the dataframe to use it further.

# In[ ]:


data_to_use = merged_data.drop(['belongs_to_collection','homepage','homepage','poster_path','tagline','video','adult','imdb_id','status'],axis=1)


# In[ ]:


data_to_use.head(1)


# ### Looking for null values in the new dataframe.

# In[ ]:


plt.figure(figsize=(8,5))
sns.heatmap(data_to_use.isnull(),cmap='Blues',cbar=False,yticklabels=False)


# ### By using the ast library retrieving the genres list and important features from cast and crew columns and displaying them in readable format. 

# In[ ]:


import ast
data_to_use['genres'] = data_to_use['genres'].map(lambda x: ast.literal_eval(x))
data_to_use['cast'] = data_to_use['cast'].map(lambda x: ast.literal_eval(x))
data_to_use['crew'] = data_to_use['crew'].map(lambda x: ast.literal_eval(x))


# ### replacing all the nan values with unknown and cleaning up the important features further. 

# In[ ]:


data_to_use['production_companies'] = data_to_use['production_companies'].replace(np.nan,'unknown')
data_to_use['production_countries'] = data_to_use['production_countries'].replace(np.nan,'unknown')
data_to_use['spoken_languages'] = data_to_use['spoken_languages'].replace(np.nan,'unknown')


# In[ ]:


data_to_use['production_company_name_only_first'] = data_to_use['production_companies'].apply(lambda x: x.split(',')[0])


# In[ ]:


data_to_use['production_company_name_only_first'] = data_to_use['production_company_name_only_first'].apply(lambda x: x.split(':')[-1])


# In[ ]:


data_to_use.head(1)


# In[ ]:


data_to_use['production_company_name_only_first1'] = data_to_use['production_company_name_only_first'].apply(lambda x: x.replace('[]','Unknown'))


# In[ ]:


data_to_use.drop(['production_company_name_only_first'],inplace=True,axis=1)


# In[ ]:


data_to_use['production_countries_name'] = data_to_use['production_countries'].apply(lambda x: x.split(':')[-1])


# In[ ]:


data_to_use['production_countries_name_only'] = data_to_use['production_countries_name'].apply(lambda x : x[:-2])


# In[ ]:


data_to_use.head(1)


# In[ ]:


data_to_use.spoken_languages[1112]


# In[ ]:


data_to_use['spoken_languages_only'] = data_to_use['spoken_languages'].apply(lambda x: x.split(',')[-1])


# In[ ]:


data_to_use['spoken_languages_only1'] = data_to_use['spoken_languages'].apply(lambda x: x.split(':')[-1])


# In[ ]:


data_to_use['spoken_languages_only12'] = data_to_use['spoken_languages_only1'].apply(lambda x : x[:-2])


# In[ ]:


data_to_use.head(1)


# ### Displaying some important features like actor names, director names etc in direct readable format in the dataframe.

# In[ ]:


def make_genresList(x):
    gen = []
    st = " "
    for i in x:
        if i.get('name') == 'Science Fiction':
            scifi = 'Sci-Fi'
            gen.append(scifi)
        else:
            gen.append(i.get('name'))
    if gen == []:
        return np.NaN
    else:
        return (st.join(gen))


# In[ ]:


data_to_use['genres_list'] = data_to_use['genres'].map(lambda x: make_genresList(x))


# In[ ]:


def get_actor1(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == []:
        return np.NaN
    else:
        return (casts[0])


# In[ ]:


data_to_use['actor_1_name'] = data_to_use['cast'].map(lambda x: get_actor1(x))


# In[ ]:


def get_actor2(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == [] or len(casts)<=1:
        return np.NaN
    else:
        return (casts[1])


# In[ ]:


data_to_use['actor_2_name'] = data_to_use['cast'].map(lambda x: get_actor2(x))


# In[ ]:


def get_actor3(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == [] or len(casts)<=2:
        return np.NaN
    else:
        return (casts[2])


# In[ ]:


data_to_use['actor_3_name'] = data_to_use['cast'].map(lambda x: get_actor3(x))


# In[ ]:


def get_directors(x):
    dt = []
    st = " "
    for i in x:
        if i.get('job') == 'Director':
            dt.append(i.get('name'))
    if dt == []:
        return np.NaN
    else:
        return (st.join(dt))


# In[ ]:


data_to_use['director_name'] = data_to_use['crew'].map(lambda x: get_directors(x))


# ### This dataframe I am going to use for my Movie recommendation notebook later on. 

# In[ ]:


data_to_use.head(1)


# ### Picking up all the important columns for exploratory data analysis.

# In[ ]:


data_for_analyses = data_to_use.drop(['genres','id','original_language','production_companies','production_countries','release_date','spoken_languages','cast','crew','production_countries_name','spoken_languages_only','spoken_languages_only1'],axis=1)


# In[ ]:


data_for_analyses.head(2)


# In[ ]:


# converting budget column to int.
data_for_analyses['budget'] = data_for_analyses['budget'].astype(int)


# In[ ]:


data_for_analyses.describe()


# In[ ]:


data_for_analyses[data_for_analyses['revenue']==2787965087.0]


# ### Making a new column called as profit = revenue - budget. I am going to use this feature for building my predictive model.

# In[ ]:


data_for_analyses['Profit'] = data_for_analyses['revenue'] - data_for_analyses['budget']


# In[ ]:


data_for_analyses.head(1)


# ### Looking at this graph we can see in 2014 there were the most number of movies produced. 

# In[ ]:


data_for_analyses.year.value_counts(dropna=False).sort_index().plot(kind='barh',color='g',figsize=(20,25))


# In[ ]:


data_for_analyses['popularity'] = data_for_analyses['popularity'].astype(float)


# In[ ]:


data_for_analyses_clean = data_for_analyses.dropna(how='any')


# In[ ]:


plt.figure(figsize=(8,5))
sns.heatmap(data_for_analyses_clean.isnull(),cmap='Blues',cbar=False,yticklabels=False)


# ### Making correlations to find which numerical features influence the Profit feature most. 

# In[ ]:


correlations = data_for_analyses_clean.corr()
f,ax = plt.subplots(figsize=(10,6))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)


# ### Maikng a new dataframe for my machine learning model

# In[ ]:


df_for_ML = data_for_analyses_clean[['budget','popularity','revenue','runtime','vote_average','vote_count','year','Profit']]


# ### Plotting the histograms of all the features in my Machine learning dafaframe

# In[ ]:


df_for_ML.hist(bins=30,figsize=(15,8),color='g')


# ### Visualising the relations between vote average and other features. 

# In[ ]:


for i in df_for_ML.columns:
    axis = data_for_analyses_clean.groupby('vote_average')[[i]].mean().plot(figsize=(10,5),marker='o',color='g')


# In[ ]:


df_for_ML.head(1)


# ### Splitting my machine learning dataset into train and test. 

# In[ ]:


from sklearn.model_selection import train_test_split
X = df_for_ML.drop('Profit',axis=1)
y = df_for_ML['Profit']
X.shape,y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Using the Linear Regression model to predict the Profit of a movie based on selected features.

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


prec_lm=lm.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print('The mean squared error using Linear regression is: ',mean_squared_error(y_test,prec_lm))
print('The mean absolute error using Linear regression is: ',mean_absolute_error(y_test,prec_lm))


# ### Using the random Forest model to predict the Profit of a movie based on selected features.

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
# #### The random forest model has perforemed the best with MSE of '8177714132974.215' and MAE of '271274.95012048195'
# #### The budget of the movie has a big influence on the profit of a movie.
# #### Movies with higher budget tent get better vote average.
