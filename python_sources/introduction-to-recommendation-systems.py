#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation system with Movielens Dataset
# A Recommender System can be defined as an algorithm that performs information filtering by trying to provide the most relevant and accurate items that fit a user's needs and interests from a large pool of Information. Recommendation systems could be used to recommend Movies, Products, Friends (on social media) and even articles, the list is endless. Compaines such as Facebook,Amazon, Netflix, Linkedin use recommendation systems to filter content.
# 
# ## Types of RS
# 1. Collaborative filtering recommender systems
# 2. Content-based recommender systems
# 3. Hybrid recommender systems
# 4. Context-aware recommender systems
# 
# For this example, we will focus on an introduction to building a Recommendation system using the Collaborative filtering:User-based collaborative filtering Approach. User-based Collaborative filtering is an algorithm which is used to make automatic predictions about a user's interests by compiling preferences from several simliar users.
# 
# For this introduction, we won't dive int the technicality of extracting features using Natural Language processing. We will focus on these features:
# * User ID
# * Movie ID
# * User's Rating for a specific Movie
# * Tag
# 
# So, lets import our dataset of rated movies and see what our data looks like.

# In[ ]:


#import "Movie Lens Small Latest Dataset" dataset

import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

links= pd.read_csv('../input/movie-lens-small-latest-dataset/links.csv')
movies=pd.read_csv('../input/movie-lens-small-latest-dataset/movies.csv')
ratings=pd.read_csv('../input/movie-lens-small-latest-dataset/ratings.csv')
tags=pd.read_csv('../input/movie-lens-small-latest-dataset/tags.csv')

dataset=movies.merge(ratings,on='movieId').merge(tags,on='movieId').merge(links,on='movieId')
dataset.head()


# Next, we drop the columns we would not be using in this tutorial and view a summary of out table.

# In[ ]:


to_drop=['title','genres','timestamp_x','timestamp_y','userId_y','imdbId','tmdbId']

dataset.drop(columns=to_drop,inplace=True)
print(dataset.describe())
dataset.head()


# In[ ]:


dataset=pd.get_dummies(dataset) #encode the catergorical data

print(dataset)
dataset.isnull().sum()# check the number of missing data cells


# From the output above, we dont need to handle missing data since the dataset is complete. So the next step is to divide our datset into training and tests. We also need to seperate our features from our labels.
# 
# For this problem set, our label would be the "ratings" score.

# In[ ]:


#Divide data into test,train and validation
#train dataset
train_dataset = dataset.sample(frac=0.9,random_state=0) #90% of the dataset
test_dataset=dataset.drop(train_dataset.index) #10% of the Dataset


#seperate labels
train_labels = train_dataset.pop('rating')
test_labels = test_dataset.pop('rating')


# # Build Model
# We are going to handle this as a regression problem because even though the possible range of ratings is enclosed in values 0-5, our dataset summary above shows we have ratings such as "3.5" in our dataset.
# 
# We will be building a Regression Model using the Random Forest Algorithm.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#scale the features
sc = StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.fit_transform(test_dataset)

#train model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(train_dataset, train_labels)

#predict ratings on test data
predicted_labels = regressor.predict(test_dataset)


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Evaluating Model Performance 

# In[ ]:



actual_labels=np.array(test_labels)

print('Mean Absolute Error:', metrics.mean_absolute_error(actual_labels, predicted_labels))
print('Mean Squared Error:', metrics.mean_squared_error(actual_labels, predicted_labels))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(actual_labels, predicted_labels)))
print('Mean Average Percentage Error: ',mean_absolute_percentage_error(actual_labels,predicted_labels))


# In[ ]:


print('Size of Test Labels',actual_labels.size)
print('Size of Predicted Labels',predicted_labels.size)

#create a new dataframe
predicted_movies=pd.DataFrame({'Actual':actual_labels,'Predicted':predicted_labels}).reset_index()


# Below is a Table showing Predicted User Ratings Vs the Actual ratings given by those users.

# In[ ]:


#print Table of Test Dataset
predicted_movies.head(20)


# # Visualize Prediction Results
# 
# 1. Histogram showing distribution of Prediction Errors

# In[ ]:


difference=actual_labels-predicted_labels
plt.hist(difference,normed=True,color='orange',bins=15,alpha=0.8)


# 2. Line plot showing first 50 predictions in Test Dataset. A graph of **Actual** ( *in Blue*) vs **Predicted** (*in Orange*) User rating on movies.

# In[ ]:


sns.set(style="darkgrid")
mapping=predicted_movies['Actual'][:50].plot(kind='line',x='Datetime',y='Volume BTC',color='blue',label='Actual')
predicted_movies['Predicted'][:50].plot(kind='line',x='Datetime',y='Volume BTC',color='orange',label='Predicted',ax=mapping)
mapping.set_xlabel('Users')
mapping.set_ylabel('Ratings')
mapping.set_title('Regression Graph show Actual vs Predicted ratings')
mapping=plt.gcf()
mapping.set_size_inches(20,12)


# 3. Joint plot showing KDE graph of Actual vs Predicted Ratings.

# In[ ]:


sns.jointplot(predicted_movies['Actual'],predicted_movies['Predicted'],predicted_movies,kind='kde')


# ## Feedback
# If you liked this notebook, have suggestions, recommendations or need clarifications,drop your comments  below. If this was helpful an upvote would be appreciated.
# 
# ### Contact Me:
# Feel free to contact me via samtheo1597@gmail.com, +2348151475929
