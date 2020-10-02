#!/usr/bin/env python
# coding: utf-8

# <font size="6">To predict a movie's worldwide box office revenue

# <font size="3" >We are provided with 7398 movies and a variety of metadata obtained from The Movie Database (TMDB). Movies are labeled with id. Data points include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries.
# 
# We are going to predict the worldwide revenue for 4398 movies in the test file.

# <font size ="3"><b>Importing Necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize 
import itertools  # Iterating tools
import re  # Regular Expressions


# <font size ="3"><b>Reading train data and test data

# In[ ]:


train_data = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
test_data = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")


# <font size ="3"><b> Getting information of the loaded data

# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# <font size ="3"><b> Shape of test data and train data

# In[ ]:


train_data.shape , test_data.shape


# <font size ="3"><b> Let's clean the data by removing columns which are less important and finding the null values

# In[ ]:


pd.DataFrame(train_data.isnull().sum()).T


# now, we calculate  percentage to these missing values for both the train and test set

# In[ ]:


perc_traindata = ((pd.DataFrame(train_data.isnull().sum()).T)/len(train_data))*100
perc_traindata


# Representation of these missing values % in graph

# In[ ]:


sns.set(rc={'figure.figsize':(12,8)})
perc_traindata.plot.bar()
plt.xlabel("Columns with missing train data")
plt.ylabel("Percentage of missing train data")


# If NAN values are more than 60 , we drop that column 

# In[ ]:


train_data.drop(columns=['belongs_to_collection','homepage'],axis = 1 ,inplace=True)


# Some columns from which 'Prediction of Revenue' doesn't affect. <br>
# Listed as : 'imdb_id', 'poster_path','tagline', 'overview', 'original_title','Keywords' ,'crew'

# In[ ]:


train_data.drop(columns=['imdb_id', 'poster_path','tagline', 'overview', 'original_title','Keywords' ,'crew'],axis = 1 ,inplace=True)


# In[ ]:


#Checking again the shape of train data
train_data.shape


# In[ ]:


train_data.dtypes


# In[ ]:


#Test data null values
pd.DataFrame(test_data.isnull().sum()).T


# In[ ]:


perc_testdata = ((pd.DataFrame(test_data.isnull().sum()).T)/len(test_data))*100
perc_testdata


# In[ ]:


sns.set(rc={'figure.figsize':(12,8)})
perc_testdata.plot.bar()
plt.xlabel("Columns with missing test data")
plt.ylabel("Percentage of missing test data")


# In[ ]:


test_data.drop(columns=['belongs_to_collection','homepage'],axis = 1 ,inplace=True)
test_data.drop(columns=['imdb_id', 'poster_path','tagline', 'overview', 'original_title','Keywords' ,'crew'],axis = 1 ,inplace=True)


# In[ ]:


#Checking again the shape of test data
test_data.shape


# In[ ]:


test_data.dtypes


# <font size ="3"><b> Let's first extract the data from the columns which have JSON Objects <br>
# 1. GENRE

# In[ ]:


#extract only genres column from the train dataset and create new dataset of it which contains only genres column
new=train_data.loc[:,["genres"]]
#fill allna with "None"
new["genres"]=train_data["genres"].fillna("None");
new["genres"].head(5)


# In[ ]:


#extract genre function which will take input as a row [{'id': 35, 'name': 'Comedy'}] and returns
# array of each genre name e.g. ['Comedy'] and if there the row is empty it will return ['None']
def extract_genres(row):
    if row == "None":
        return ['None']
    else:
        results = re.findall(r"'name': '(\w+\s?\w+)'", row)
        return results

#apply extract_genres function on genres column of new dataset    
new["genres"] = new["genres"].apply(extract_genres)
new["genres"].head(10) 


# In[ ]:


#declare a dictionary 
genres_dict = dict()

# loop through all the rows of genres column and set count of the genre
for genre in new["genres"]:
    for elem in genre:
        if elem not in genres_dict:
            genres_dict[elem] = 1
        else:
            genres_dict[elem] += 1


# In[ ]:


#generate data from from dictionary which includes count of each genre
genres_df = pd.DataFrame.from_dict(genres_dict, orient='index')


# In[ ]:


genres_df.columns = ["number_of_movies"]
#sort by number of movies descending 
genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)
#plot bar chart
genres_df.plot.bar()


# <b>DRAMA and COMEDY is the most common genre

# <font size ="3"><b>2. SPOKEN LANGUAGE

# In[ ]:


train_data['spoken_languages']


# In[ ]:


new=train_data.loc[:,["spoken_languages"]]
#fill allna with "None"
new["spoken_languages"]=train_data["spoken_languages"].fillna("None");
new["spoken_languages"].head(5)


# In[ ]:


def extract_spoken(row):
    if row == "None":
        return ['None']
    else:
        results = re.findall(r"'name': '(\w+\s?\w+)'", row)
        return results

#apply extract_genres function on genres column of new dataset    
new["spoken_languages"] = new["spoken_languages"].apply(extract_spoken)
new["spoken_languages"].head(10) 


# In[ ]:


#declare a dictionary 
genres_dict = dict()

# loop through all the rows of genres column and set count of the genre
for genre in new["spoken_languages"]:
    for elem in genre:
        if elem not in genres_dict:
            genres_dict[elem] = 1
        else:
            genres_dict[elem] += 1


# In[ ]:


#generate data from from dictionary which includes count of each genre
language_df = pd.DataFrame.from_dict(genres_dict, orient='index')
language_df


# <font size ="3"><b> Extract a new dataframe related to movie budget, revenue, runtime

# In[ ]:


import time
import datetime

movietime = train_data.loc[:,["title","release_date","budget","runtime","revenue"]]
movietime.dropna()

movietime.release_date = pd.to_datetime(movietime.release_date)
movietime.loc[:,"Year"] = movietime["release_date"].dt.year
movietime.loc[:,"Month"] = movietime["release_date"].dt.month
movietime.loc[:,"Day_of_Week"] = (movietime["release_date"].dt.dayofweek)
movietime.loc[:,"Quarter"]  = movietime.release_date.dt.quarter 

movietime = movietime[movietime.Year<2018]
movietime.head(6)


# GRAPH BETWEEN REVENUE AND YEAR 

# In[ ]:


data_plot = movietime[['revenue', 'Year']]
money_Y = data_plot.groupby('Year')['revenue'].sum()

money_Y.plot(figsize=(15,8))
plt.xlabel("Year of release")
plt.ylabel("revenue")
plt.xticks(np.arange(1970,2020,5))

plt.show()


# MONTH OF RELEASE AND REVENUE

# In[ ]:


f,ax = plt.subplots(figsize=(18, 10))
plt.bar(movietime.Month, movietime.revenue, color = 'blue')
plt.xlabel("Month of release")
plt.ylabel("revenue")
plt.show()


# <font size="5"> LINEAR REGRESSION

# <font size ="3"><b>Since, budget and popularity would be the main aspect to predict the revenue.   

# In[ ]:


x = train_data[['popularity', 'runtime', 'budget']] #independent variables
y= train_data['revenue'] #dependent variable


# In[ ]:


from sklearn.metrics import mean_squared_log_error as msle
from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[ ]:


from sklearn.model_selection import train_test_split
x, X_test, y, y_test = train_test_split(x, y, test_size=0.30, random_state=1)


# In[ ]:


x.describe()


# In[ ]:


x.isnull().sum()


# In[ ]:


#we fill the null values in the train data
x.median()


# In[ ]:


medianFiller = lambda t: t.fillna(t.median())
x = x.apply(medianFiller,axis=0)


# In[ ]:


x.isnull().sum()


# In[ ]:


model = reg.fit(x,y)
model


# In[ ]:


y_pred = reg.predict(x)
for idx, col_name in enumerate(x.columns):
    print("The coefficient for {} is {}".format(col_name, model.coef_[0]))


# In[ ]:


intercept = model.intercept_
print("The intercept for our model is {}".format(intercept))


# In[ ]:


#The score (R^2) for in-sample and out of sample
model.score(x, y)


# In[ ]:





# The Linear Regression gives us an accuracy of 61% <br>
# <br>
# <font size=3>Model of Test_Data

# In[ ]:



x_test = test_data[['popularity', 'runtime', 'budget']]
x_test.isnull().sum()
#pred = reg.predict(x_test)


# In[ ]:


x_test.median()


# In[ ]:


medianFiller = lambda t: t.fillna(t.median())
x_test = x_test.apply(medianFiller,axis=0)


# In[ ]:


pred = reg.predict(x_test)
pred


# In[ ]:





# In[ ]:




