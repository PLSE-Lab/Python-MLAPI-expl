#!/usr/bin/env python
# coding: utf-8

# **The goal of this kernel is to go through the steps of exploring the data, tidy the data, highlight some interesting facts,  and put in place a simple machine learning model through labeling instead of one-hot encoding. My ML is just a toy example to get started on Kaggle**

# ** Import packages**

# In[ ]:


# loading the packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# **Loading my data**

# In[ ]:


# loading my data
df = pd.read_csv('../input/winemag-data_first150k.csv', index_col = 0)                


# **Visual Inspection**

# In[ ]:


df.head(3)


# In[ ]:


df.columns


# In[ ]:


# number rows and columns:
df.shape


# In[ ]:


# check out how many values are missing
df.info()


# **By observing the results of above code, we notice that any column with less than 150,930 rows needs to be corrected.**

# In[ ]:


df1 = df.dropna(subset=['country'])


# **Identifying Top 10 countries**

# In[ ]:


# frequency counts - Top 10 countries
df1.country.value_counts(dropna=False).head(10)


# **Identifying Top 10 price points**

# In[ ]:


#frequency counts - Top 10 price points
df1.price.value_counts(dropna=False).head(10)


#  **There are a lot of missing price points.  Let's fix this by inserting the average price point.**

# In[ ]:


# as noted in the price point value counts, there are almost 14,000 missing values. 
mean_price = df1.price.mean()
df1['price'] = df1['price'].fillna(mean_price)
df1.price.value_counts().head(5)


# **Notice that the #1 price point is now a float (decimal). We'll convert it to integer later on. **

# In[ ]:


# frequency counts - Top 5 alloted points
df1.points.value_counts(dropna=False).head(5)


# **Quick view of the statistical summary**

# In[ ]:


df1.describe()


# In[ ]:


# let's have a look at the columns to ensure the price column has been fixed.
df1.info()


# **Now we have the same amount of entries in the price column than the points column. But there is still some cleaning to do in this data.**

# **The column region_2 is missing lots of values. Since this column won't be crucial for our exploratory purposes let's delete it.**

# In[ ]:


df1 = df1.drop(['region_2'], axis = 1)


# In[ ]:


df1.info()


# **Since the #1 price point is showing as float (decimals)... convert it integer.**

# In[ ]:


df1.price = df1.price.astype(int)


# In[ ]:


df1['price'].value_counts(dropna = False).head(5)


# **What country sells the most expensive bottle of wine, how many review points it got and what it is the variety? **

# In[ ]:


most_expensive = df1.groupby('country')[['price', 'points', 'variety']].max().nlargest(5, ['price'])
most_expensive


# **There is a correlation between points and price. 
# Let 's see it by 'points'  :**

# In[ ]:


best_rated = df1.groupby('country')[['points', 'price', 'variety']].max().nlargest(5, ['points'])
best_rated


# **Best rapport between 'points' vs 'price' per bottle is from Argentina. **

# **Visualizing the distribution for the column 'points'**

# In[ ]:


# let's see the distribution for the column 'points':
import matplotlib.pyplot as plt
_ = plt.hist(df1['points'])
plt.ylabel('# of units')
plt.xlabel('Points')
plt.show()


# **Basic statistics...**

# In[ ]:


price = df1.price
mean_price = np.mean(price)
median_price= np.median(price)
std_dev = np.std(price)
print('mean price =', mean_price, 'median price =', median_price, 'standard deviation =', std_dev)


# **Compute an empirical cumulative distribution function(ECDF). This will give us a full view of how pricing is distributed** 

# In[ ]:


x = np.sort(df1['price'])
y = np.arange(1, len(x) + 1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('price')
_ = plt.ylabel('ECDF') 
plt.margins(0.02)
plt.show()


# **Above graph shows that more than 95% of prices are less than 77 dlls, this can be confirmed by computing the percentiles**

# In[ ]:


# confirm above graph by computing percentiles
np.percentile(df1['price'], [25, 50, 75, 90, 95])


# **Ok time to do some machine learning...**

# In[ ]:


country_labels, country_uniques = pd.factorize(df1.loc[:, 'country'])


# In[ ]:


country_labels


# In[ ]:


province_labels, province_uniques = pd.factorize(df1.loc[:, 'province'])
winery_labels, winery_uniques = pd.factorize(df1.loc[:, 'winery'])
variety_labels, variety_uniques = pd.factorize(df1.loc[:, 'variety'])


# In[ ]:


# add columns with the country, provicne and winery labels just created. 
df1['country_labels'] = country_labels
df1['province_labels'] = province_labels
df1['winery_labels'] = winery_labels
df1['variety_labels'] = variety_labels
df1.head(3)


# In[ ]:


df1 = df1.loc[:, ['country_labels', 'country','province_labels', 'province', 'winery_labels', 
                           'winery', 'variety', 'variety_labels', 'points', 'price']]
df1.head()


# In[ ]:


#create a mapping 
variety_name = dict(zip(df1.variety_labels.unique(), df1.variety.unique()))


# In[ ]:


# create train_test split
from sklearn.model_selection import train_test_split

X = df1[['country_labels', 'province_labels', 'winery_labels','points', 'price',]]
y = df1['variety_labels']
# 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


#create the classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)


# In[ ]:


# train the classifier
knn.fit(X_train, y_train)


# In[ ]:


#estimate the accuracy of the classifier
knn.score(X_test, y_test)


# In[ ]:


#using the KNN classifier model to classify new objects and find out the variety name!
#let's apply it with country_label = 3, province_label = 4, winery_label = 9, points 95 and price = 250

variety_prediction = knn.predict([[3, 4, 9, 95, 250]])
variety_name[variety_prediction[0]]


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))


# **Any comments would be appreciated ** 

# 
