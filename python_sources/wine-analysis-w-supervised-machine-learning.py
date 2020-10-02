#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Initial Overview of Data
# I plan to parse the data and view it in its most raw form.

# In[ ]:


init_data = pd.read_csv("../input/winemag-data_first150k.csv")
print("Length of dataframe before duplicates are removed:", len(init_data))
init_data.head()


# ### Drop Duplicates
# I need to drop the duplicates from the data. This is taken from https://www.kaggle.com/carkar/classifying-wine-type-by-review, analysis by CarKar.

# In[ ]:


parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))
parsed_data.head()


# ### Exploratory Analysis

# In[ ]:


price_variety_df = parsed_data[['price','variety']]
price_variety_df.info()


# In[ ]:


fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='variety', y='price', data=price_variety_df)
plt.xticks(rotation = 90)
plt.show()


# To me, this means that there are a large number of varieties that are somewhat unique, in that they are lower priced and come in lower quantities. To that end, I will focus on wines that cost more than $100 for the preliminary analysis.

# In[ ]:


price_variety_clipped = price_variety_df[price_variety_df['price'] > 100]
price_variety_clipped.shape


# In[ ]:


fig2, ax2 = plt.subplots(figsize=(30,10))
sns.boxplot(x='variety', y='price', data=price_variety_clipped)
plt.xticks(rotation = 90)
plt.show()


# This plot shows leads me to believe that there is a large variation of price in terms of the price of each variety of these wines.

# In[ ]:


## find counts of each type of wine
variety_counts = price_variety_clipped['variety'].value_counts().to_frame()
## standard deviations
variety_std = price_variety_clipped.groupby('variety').std()
variety_counts.index.name = 'variety'
## merge the dataframes
variety_counts = variety_counts.sort_index()
variety_std = variety_std.join(variety_counts)
## show a plot detailing it
regression_plot = sns.regplot(x='variety', y='price', data=variety_std, color="#4CB391")
regression_plot.set(xlabel='Count', ylabel='Standard Deviation')
plt.show()


# We can see that there is some correlation between the total quantity of the variety of wine and the standard deviation in the price. There is a general positive trend, but the regression does not fit the data well at all.

# #### Price and Points Correlation

# In[ ]:


parsed_data_clipped = parsed_data[parsed_data['price'] > 100]
parsed_data_clipped.shape


# In[ ]:


## Determine if there is a correlation between the quantity of the points and the price
sns.pairplot(data=parsed_data_clipped[['points', 'price']])


# In[ ]:


sns.jointplot(x='points', y='price', data=parsed_data_clipped, kind="hex", color="#4CB391")


# We can see that there *is* some correlation between the points that a wine gets and the price of it, but it is not a tell all sign of an expensive wine. 

# #### Price and Count Correlation

# In[ ]:


fig3, ax3 = plt.subplots(figsize=(30,10))
sns.countplot(x='variety', data=price_variety_clipped)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


## find the mean price of wine from each country
mean_country_price = parsed_data_clipped.groupby('country')['price'].mean().to_frame()
country_counts = parsed_data_clipped['country'].value_counts().to_frame()
mean_count_df = country_counts.join(mean_country_price)
mean_count_plot = sns.regplot(x='country', y='price', data=mean_count_df, color="#2F32B7")
mean_count_plot.set(xlabel='Count', ylabel='Mean Price')
plt.show()


# Again, there seems to be some correlation, but it is not indicative of the price.

# ## Try using a Decision Tree
# This seems like a perfect use of this classification technique. Maybe there are some intuitions that we can pick up by running supervised models on it.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


# In[ ]:


## create the x and y columns
x_unsplit = parsed_data[['country', 'points', 'price', 'province', 'region_1']]
y_unsplit = parsed_data['variety']
## one hot encode the data
x_unsplit = pd.get_dummies(x_unsplit, columns=['country', 'province', 'region_1'])
y_unsplit = pd.get_dummies(y_unsplit, columns=['variety'])
print(x_unsplit.shape, y_unsplit.shape)
## get the training and testing data
X_train, X_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, random_state=1, train_size=0.65)


# In[ ]:


X_train_35 = X_train.fillna({"price": 35.})
X_test_35 = X_test.fillna({"price":35.})
clf.fit(X_train_35, y_train)


# In[ ]:


y_predictions = clf.predict(X_test_35)


# In[ ]:


from sklearn.metrics import accuracy_score
dt_acc = accuracy_score(y_test,y_predictions)
print(dt_acc)


# ## Accuracy: 62.60%
# Considering that the values which were originally NaN were filled with a completely arbitrary value, we can say that the decision tree is a step in the right direction.
# Next, we can see if using the mean of the price of each variety will make the model more applicable.

# In[ ]:


parsed_data_mean = parsed_data
parsed_data_mean['price'] = parsed_data_mean.groupby('variety').transform(lambda x: x.fillna(x.mean()))


# In[ ]:


## create the x and y columns
x_unsplit_mean = parsed_data_mean[['country', 'points', 'price', 'province', 'region_1']]
y_unsplit_mean = parsed_data_mean['variety']
## one hot encode the data
x_unsplit_mean = pd.get_dummies(x_unsplit_mean, columns=['country', 'province', 'region_1'])
y_unsplit = pd.get_dummies(y_unsplit_mean, columns=['variety'])
print(x_unsplit_mean.shape, y_unsplit_mean.shape)
## get the training and testing data
X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(x_unsplit_mean, y_unsplit_mean, random_state=1, train_size=0.65)


# In[ ]:


clf.fit(X_train_mean, y_train_mean)


# In[ ]:


y_predictions_mean = clf.predict(X_test_mean)


# In[ ]:


dt_acc_mean = accuracy_score(y_test_mean,y_predictions_mean)
print(dt_acc_mean)


# ## Accuracy: 44.70%
# Maybe using the mean does not help the classifier due to the fact that there is a large variation in the price data.

# # Try an Ensemble Method
# If we continue to use the 35 arbitrary insert, the accuracy may be improved if we use a random forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()


# In[ ]:


forest_clf.fit(X_train_35, y_train)
forest_predictions = forest_clf.predict(X_test_35)
rf_acc = accuracy_score(y_test,forest_predictions)


# In[ ]:


print(rf_acc)


# ## Accuracy: 60.89%
# Cool.

# In[ ]:




