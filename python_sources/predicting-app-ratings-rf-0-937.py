#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import time
import datetime

import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


data = pd.read_csv("../input/googleplaystore.csv")


# ## Data Exploration
# 
# Exploring our data

# In[ ]:


# Print first few rows from data

data.head()


# In[ ]:


# Print last few rows from data

data.tail()


# In[ ]:


# Shape of data 

data.shape


# In[ ]:


# Basic information of data

data.info()


# Looks like there are missing values in "Rating", "Type", "Content Rating", "Current Ver" and " Android Ver".

# In[ ]:


# Checking null values

data.isnull().sum()


# In[ ]:


# Types of data

data.dtypes


# In[ ]:


# Describing our data

data.describe()


# In[ ]:


# Columns present in our data

data.columns


# ## Data Cleaning
# 
# *   Removing null values
# *   Filling missing values
# *   Remove certain characters from the string and convert it into usable format.
# 
# 
# 
# 
# 

# In[ ]:


# The best way to fill missing values might be using the median instead of mean

data['Rating'] = data['Rating'].fillna(data['Rating'].median())


# In[ ]:


# Lets convert all the versions in the format number.number to simplify the data
# We have to clean all non numerical values & unicode charachters 

replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
for i in replaces:
    data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
for j in regex:
    data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

data['Current Ver'] = data['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)
data['Current Ver'] = data['Current Ver'].fillna(data['Current Ver'].median())


# In[ ]:


# Count the number of unique values in category column 

data['Category'].unique()


# In[ ]:


# Check the record  of unreasonable value which is 1.9
i = data[data['Category'] == '1.9'].index
data.loc[i]


# It's obvious that the first value of this record is missing (App name) and all other values are respectively propagated backward starting from "Category" towards the "Current Ver"; and the last column which is "Android Ver" is left null. It's better to drop the entire recored instead of consider these unreasonable values while cleaning each column!
# 
# 

# In[ ]:


# Drop this bad column
data = data.drop(i)


# In[ ]:


# Removing NaN values
data = data[pd.notnull(data['Last Updated'])]
data = data[pd.notnull(data['Content Rating'])]


# ## Plotting
# 
# 1.) Count of application according to Category

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Category', data=data, palette='Set1')
plt.xticks(rotation=90)
plt.xlabel('Categories', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.title('Count of application according to category', fontsize=15, color='r')


# Seems like family and games category have quite large number 
# 
# 2.) Let's look at the Rating distribution

# In[ ]:


from pylab import rcParams

rcParams['figure.figsize'] = 11.7, 8.27
g = sns.kdeplot(data.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)


# 3.) Let's check out the percentage of free apps in data

# In[ ]:


labels =data['Type'].value_counts(sort = True).index
sizes = data['Type'].value_counts(sort = True)


colors = ["lightblue","orangered"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 5,5
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=150,)

plt.title('Percent of Free App in data', size = 15)
plt.show()


# Majority of the apps are free
# 
# 4.) Category Distribution
# 
# *   Wordcloud represents categories with highest number of active apps 
# 
# 

# In[ ]:


from wordcloud import WordCloud
wordcloud1 = WordCloud(max_font_size=350, collocations=False, max_words=33, width=1600, height=800, background_color="white").generate(' '.join(data['Category']))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# 5.) Content Distribution

# In[ ]:


plt.figure(figsize=(10,8))
ax = sns.countplot(y='Content Rating', data=data)


# ## Categorical Data Encoding
# 
# We need to convert categorical features (strings) to numerical features (numbers). 
# 
# Many machine learning algorithms can support categorical values without further manipulation but there are many more algorithms that do not. We need to make all data ready for the model, so we will convert categorical variables (variables that stored as text values) into numerical variables. 

# In[ ]:


# App values encoding

LE = preprocessing.LabelEncoder()
data['App'] = LE.fit_transform(data['App'])


# In[ ]:


# Category features encoding

CategoryList = data['Category'].unique().tolist() 
CategoryList = ['cat_' + word for word in CategoryList]
data = pd.concat([data, pd.get_dummies(data['Category'], prefix='cat')], axis=1)


# In[ ]:


# Genres features encoding

LE = preprocessing.LabelEncoder()
data['Genres'] = LE.fit_transform(data['Genres'])


# In[ ]:


# Content Rating features encoding

LE = preprocessing.LabelEncoder()
data['Content Rating'] = LE.fit_transform(data['Content Rating'])


# In[ ]:


# Type encoding

data['Type'] = pd.get_dummies(data['Type'])


# In[ ]:


# Last Updated encoding

data['Last Updated'] = data['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))


# In[ ]:


# Price cleaning

data['Price'] = data['Price'].apply(lambda x : x.strip('$'))


# In[ ]:


# Installs cleaning

data['Installs'] = data['Installs'].apply(lambda x : x.strip('+').replace(',', ''))


# In[ ]:


# Convert kbytes to Mbytes

k_indices = data['Size'].loc[data['Size'].str.contains('k')].index.tolist()

converter = pd.DataFrame(data.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))

data.loc[k_indices,'Size'] = converter


# In[ ]:


# Size cleaning

data['Size'] = data['Size'].apply(lambda x: x.strip('M'))
data[data['Size'] == 'Varies with device'] = 0
data['Size'] = data['Size'].astype(float)


# ## Building Machine Learning Model

# In[ ]:


features = ['App', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
features.extend(CategoryList)

X = data[features]
y = data['Rating']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


# #### Decision Tree Regression
# 
# Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.

# In[ ]:


from sklearn import tree


# In[ ]:


clf = tree.DecisionTreeRegressor(criterion='mae', max_depth=5, min_samples_leaf=5, random_state=42)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


accuracy = clf.score(X_test, y_test)
accuracy


# #### Random Forest Regression Model
# 
# The RandomForestRegressor class of the sklearn.ensemble library is used to solve regression problems via random forest. The most important parameter of the RandomForestRegressor class is the n_estimators parameter. This parameter defines the number of trees in the random forest.
# 

# In[ ]:


model = RandomForestRegressor(n_estimators = 200, n_jobs=-1, random_state=10)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


acc = model.score(X_test, y_test)
acc


# In[ ]:


Pred = model.predict(X_test)
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, Pred)


# In[ ]:


'Mean Squared Error:', metrics.mean_squared_error(y_test, Pred)


# In[ ]:


'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Pred))


# ## Feature Selection
# 
# Feature selection is a process where you automatically select those features in your data that contribute most to the prediction variable or output in which you are interested. 
# 
# Three benefits of performing feature selection before modeling your data are:
# 
# *  **Reduces Overfitting:** Less redundant data means less opportunity to make decisions based on noise.
# *  **Improves Accuracy:** Less misleading data means modeling accuracy improves.
# *  **Reduces Training Time:** Less data means that algorithms train faster.

# ### Cross-Validation
# 
# ### Splitting into training and CV for Cross-Validation

# In[ ]:


X = data.loc[:,['Reviews', 'Size', 'Installs', 'Content Rating']]

x_train, x_cv, y_train, y_cv = train_test_split(X, data.Rating)

clf = tree.DecisionTreeRegressor(criterion='mae', max_depth=5, min_samples_leaf=5, random_state=42)


# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


ac = clf.score(x_cv, y_cv)
ac


# **Thanks for reading this kernel till here. If you like it, please upvote this kernel. This will help me to write more.**
# 
# **If you have doubts please leave a comment.**
# 
# **Thanks!!**
