#!/usr/bin/env python
# coding: utf-8

# # Goodreads Rating Prediction Using Linear Regressio

# The goal of this analysis is to predict book rating and to understand what are the important factors that make a book more popular than others. The sections of this analysis include:
# 1. Data Exploration
# 2. Data Cleaning
# 3. Data Visualisation
# 4. Data Preprocessing
# 5. Machine Learning Model

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'darkgrid')
sns.set_palette('deep')


# # Data Exploration

# In[ ]:


# read the data
data = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)


# # Data Exploration

# In[ ]:


# show the first few rows
data.head(5)


# In[ ]:


# check basic features and data types
data.info()


# The dataset contains both numerical and categorical data types.

# In[ ]:


# check no. of records
len(data)


# # Data Cleaning

# In[ ]:


# check for doublications
data.duplicated().any()


# Let's use heatmap to visualise above result!

# In[ ]:


sns.heatmap(data.isnull(), cmap='viridis')


# No duplicated or missing values, that makes things a little easier.

# # Data Visualisation

# In[ ]:


# ratings distribution
sns.kdeplot(data['average_rating'], shade = True)
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Frequency')


# In[ ]:


# top 5 languages
data['language_code'].value_counts().head(5).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# Over 95% of books are in English, which means this variable is nearly constant!

# In[ ]:


# number of books per rating
sns.barplot(data['average_rating'].value_counts().head(15).index, data['average_rating'].value_counts().head(15))
plt.title('Number of Books Each Rating Received\n')
plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.xticks(rotation=45)


# In[ ]:


# highest rated books
popular_books = data.nlargest(10, ['ratings_count']).set_index('title')['ratings_count']
sns.barplot(popular_books, popular_books.index)


# In[ ]:


# highest reviewed books
highest_reviews = data.nlargest(10, ['text_reviews_count'])
sns.barplot(highest_reviews['text_reviews_count'], highest_reviews['title'])


# In[ ]:


# top 10 books under 200 pages for busy book lovers
under200 = data[data['# num_pages'] <= 200]
top10under200 = under200.nlargest(10, ['ratings_count'])
sns.barplot(top10under200['ratings_count'], top10under200['title'], hue=top10under200['average_rating'])
plt.xticks(rotation=15)


# In[ ]:


# top 10 longest books
longest_books = data.nlargest(10, ['# num_pages']).set_index('title')
sns.barplot(longest_books['# num_pages'], longest_books.index)


# In[ ]:


# top languages
data['language_code'].value_counts().plot(kind='bar')
plt.title('Most Popular Language')
plt.ylabel('Counts')
plt.xticks(rotation = 90)


# In[ ]:


# top published books
sns.barplot(data['title'].value_counts()[:15], data['title'].value_counts().index[:15])
plt.title('Top Published Books')
plt.xlabel('Number of Publications')


# In[ ]:


# authors with highest rated books
plt.figure(figsize=(10, 5))
authors = data.nlargest(5, ['ratings_count']).set_index('authors')
sns.barplot(authors['ratings_count'], authors.index, ci = None, hue = authors['title'])
plt.xlabel('Total Ratings')


# In[ ]:


# authors with highest publications
top_authors = data['authors'].value_counts().head(9)
sns.barplot(top_authors, top_authors.index)
plt.title('Authors with Highest Publication Count')
plt.xlabel('No. of Publications')


# In[ ]:


# visualise a bivariate distribution between ratings & no. of pages
sns.jointplot(x = 'average_rating', y = '# num_pages', data = data)


# In[ ]:


# visualise a bivariate distribution between ratings & no. of reviews
sns.jointplot(x = 'average_rating', y = 'text_reviews_count', data = data)


# # Data Preprocessing

# Data preprocessing is the conversion of data into machine-readable form can be interpreted, analysed and used by machine learning algorithms. In this analysis we will apply anomaly detection and feature engineering techniques.

# ### 1) Anomaly Detection

# In[ ]:





# The main goal of this section is to remove extreme outliers (abnormal distance from other values) from features, this will have a positive impact on the accuracy of the model.

# In[ ]:


# find no. of pages outliers
sns.boxplot(x=data['# num_pages'])


# Above plot shows points between 1,000 to 6,000, these are outliers as there are not included in the box of other observation i.e no where near the quartiles.

# In[ ]:


# remove outliers from no. of pages 
data = data.drop(data.index[data['# num_pages'] >= 1000])


# In[ ]:


# find ratings count outliers
sns.boxplot(x=data['ratings_count'])


# Above plot shows points between 1,000,000 to 5,000,000 are outliers.

# In[ ]:


# remove outliers from ratings_count
data = data.drop(data.index[data['ratings_count'] >= 1000000])


# Above plot shows points between 20,000 to 80,000 are outliers

# In[ ]:


# remove outliers from text_reviews_count
data = data.drop(data.index[data['text_reviews_count'] >= 20000])


# ### 2) Feature Engineering

# Feature engineering is the process of selecting and transforming variables when creating a predictive model. Many machine learning algorithms require that their input is numerical and therefore categorical features such as title, authors and language code must be transformed into numerical features before we can use any of these algorithms.

# In[ ]:


# encode title column
le = preprocessing.LabelEncoder()
data['title'] = le.fit_transform(data['title'])


# In[ ]:


# encode authors column
data['authors'] = le.fit_transform(data['authors'])


# In[ ]:


# encode language column
enc_lang = pd.get_dummies(data['language_code'])
data = pd.concat([data, enc_lang], axis = 1)


# # Machine Learning Model

# The aim of this section is to come up with a model for predicting the book ratings. We'll use linear regression to build a model that predicts book ratings. Linear regression algorithm is a basic predictive analytics technique. There are two kinds of variables in a linear regression model:
# 1. The __input__ or __predictor variable__ is the variable(s) that help predict the value of the output variable. It is commonly referred to as __X__.
# 2. The __output variable__ is the variable that we want to predict. It is commonly referred to as __Y__.

# In[ ]:


# divide the data into attributes and labels
X = data.drop(['average_rating', 'language_code', 'isbn'], axis = 1)
y = data['average_rating']


# Attributes are the independent variables whilst labels are dependent variables whose values are to be predicted.

# In[ ]:


# split 80% of the data to the training set and 20% of the data to test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)


# test_size variable is where we actually specify the proportion of the test set. Now to train our algorithm, we need to import LinearRegression class instantiate it, and call the fit() method along with the training data.

# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# Next step is to use the test data to check accurately our algorithm predicts the percentage score.

# In[ ]:


predictions = lr.predict(X_test)


# Now compare the actual output values for X_test with the predicted values.

# In[ ]:


pred = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': predictions.tolist()}).head(25)
pred.head(10)


# In[ ]:


# visualise the above comparison result
pred.plot(kind='bar', figsize=(13, 7))


# Though the model is not very precise, the predicted percentages are close to the actual ones.

# In[ ]:


# evaluate the performance of the algorithm
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

