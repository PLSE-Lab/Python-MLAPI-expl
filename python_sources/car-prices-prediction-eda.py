#!/usr/bin/env python
# coding: utf-8

# ### In this kernel I will try to explain what empirical strategies I used to fill in the missing data. Sometimes the explanations may be too detailed. So let's jump into it.

# I used some plots from Eli Gertz's kernel (https://www.kaggle.com/eligertz/used-cars-price-prediction )

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns


# ## Let's read our dataset

# In[ ]:


df = pd.read_csv('../input/craigslist-carstrucks-data/craigslistVehicles.csv')


# In[ ]:


# First 5 rows of our data
df.head()


# In[ ]:


#Delete some columns
df = df.drop(columns=['image_url', 'lat', 'long', 'city_url', 'desc', 'city', 'VIN'])


# In[ ]:


#Find and delete duplicates
df.drop_duplicates(subset='url')
df.shape


# In[ ]:


df[df.isnull().sum(axis=1) < 9].shape, df[df.isnull().sum(axis=1) >= 9].shape


# In[ ]:


#Let's leave lines with less than 9 missing values
df = df[df.isnull().sum(axis=1) < 9]
df.shape


# In[ ]:


#let's take a look how many missing values we have in our dataset
df.isnull().sum()


# In[ ]:


df[df.price == 0].shape


# Price can't be 0, so I deleted all rows with 0 price

# In[ ]:


df = df[df.price != 0]
df.shape


# In[ ]:


plt.figure(figsize=(8, 8))
sns.boxplot(y= 'price', data=df)


# As we can see there are a lot of unreasonably high prices(above 100k)

# In[ ]:


#delete data with prices above 100k
df = df[df.price < 100000]
df.shape


# In[ ]:


plt.figure(figsize=(8, 10))
sns.boxplot(y= 'price', data=df)


# In[ ]:


plt.figure(figsize=(15, 13))
year_plot = sns.countplot(x = 'year', data=df)
year_plot.set_xticklabels(year_plot.get_xticklabels(), rotation=90,fontsize=8);


# Most of the cars in our dataset have been manufactured since 1985. Let's keep cars with year above the year of 1985.

# In[ ]:


df = df[df.year > 1985]
df.shape


# In[ ]:


plt.figure(figsize=(15, 13))
ax = sns.countplot(x = 'year', data=df, palette='Set1')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10);


# In[ ]:


df.odometer.quantile(.999)


# Remove extremely high odometer values.

# In[ ]:


df = df[~(df.odometer > 500000)]
df.shape


# In[ ]:


plt.figure(figsize = (8, 12))
sns.boxplot(y = 'odometer', data = df[~(df.odometer > 500000)])


# In[ ]:


sns.set(style="ticks", color_codes='palette')
sns.pairplot(df, hue= 'condition');


# ## Let's do something with missing data

# This is where we start filling in the missing data. I have used empirical strategies that may be wrong.

# In[ ]:


df.isnull().sum()


# In[ ]:


mean= df[df['year'] == 2010]['odometer'].mean()
mean


# Odometer is a counter of how many kilometers or miles a car has traveled. Some cars are used more, some less. But the starting point for each car is the year in which it was produced. So we find the average value for a particular year, and fill in all the missing values for that year with this value.

# In[ ]:


df.odometer = df.groupby('year')['odometer'].apply(lambda x: x.fillna(x.mean()))


# In[ ]:


df.odometer.isnull().sum()


# In[ ]:


df['condition'].isnull().sum()


# In[ ]:


df.loc[(df['year'] >= 2019)]['condition'].isnull().sum()


# I decided to mark all cars produced in 2019 or later with the label " New"

# In[ ]:


df.loc[df.year>=2019, 'condition'] = df.loc[df.year>=2019, 'condition'].fillna('new')


# In[ ]:


df.loc[(df['year'] >= 2019)]['condition'].isnull().sum()


# In[ ]:


df['condition'].unique()


# I made the assumption that the condition of the car depends on the number of kilometers traveled.

# In[ ]:


excellent_odo_mean = df[df['condition'] == 'excellent']['odometer'].mean()
good_odo_mean = df[df['condition'] == 'good']['odometer'].mean()
like_new_odo_mean = df[df['condition'] == 'like new']['odometer'].mean()
salvage_odo_mean = df[df['condition'] == 'salvage']['odometer'].mean()
fair_odo_mean = df[df['condition'] == 'fair']['odometer'].mean()
print('excelent {}, good {}, like_new {}, salvage {}, fair {}'.format(excellent_odo_mean, good_odo_mean,
                                                                like_new_odo_mean, salvage_odo_mean,
                                                                fair_odo_mean))


# In[ ]:


df.loc[df['odometer'] <= like_new_odo_mean, 'condition'] = df.loc[df['odometer'] <= like_new_odo_mean, 'condition'].fillna('like new')
df.loc[df['odometer'] >= fair_odo_mean, 'condition'] = df.loc[df['odometer'] >= fair_odo_mean, 'condition'].fillna('fair')
df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'] = df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'].fillna('excellent')
df.loc[((df['odometer'] > excellent_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'] = df.loc[((df['odometer'] > excellent_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'].fillna('good')
df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'].fillna('salvage')


# In[ ]:


df.isnull().sum()


# In[ ]:


df['cylinders'].unique()


# In[ ]:


df['cylinders'].value_counts().head()


# In[ ]:


df['cylinders'].isnull().sum()


# Most cars have four, six or eight cylinders, although some have three, five or ten(https://itstillruns.com/determine-many-cylinders-3374.html ). So I decide to use 6 cylinders to fill missing data.

# In[ ]:


df['cylinders'] = df['cylinders'].fillna(df['cylinders'].value_counts().index[0])


# The remaining missing values in next 4 columns are filled with the most common value

# In[ ]:


df['transmission'] = df['transmission'].fillna(df['transmission'].value_counts().index[0])
df['title_status'] = df['title_status'].fillna(df['title_status'].value_counts().index[0])
df['fuel'] = df['fuel'].fillna(df['fuel'].value_counts().index[0])
df['size'] = df['size'].fillna(df['size'].value_counts().index[0])


# In[ ]:


df = df.dropna(subset=['make'])


# The remaining missing values are filled with the label 'Unknown'

# In[ ]:


df = df.fillna('Unkown')


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.drop(columns=['url'])


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# For our algorithm, we have to transform categorical features for further use. I decide to use LabelEncoder. Here is an article about it ( https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621 )

# In[ ]:


labels = ['manufacturer', 'make', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 
          'drive', 'size', 'type', 'paint_color']
les = {}

for l in labels:
    les[l] = LabelEncoder()
    les[l].fit(df[l])
    tr = les[l].transform(df[l]) 
    df.loc[:, l + '_feat'] = pd.Series(tr, index=df.index)

labeled = df[ ['price'
                ,'odometer'
                ,'year'] 
                 + [x+"_feat" for x in labels]]


# In[ ]:


labeled.head()


# In[ ]:


X = labeled.drop(columns=['price'])
y = labeled['price']
print(X.shape, y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
print(X_train.shape, X_test.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# The scale of the odometer is much larger than that of other features. So I decided to bring all the data to a single scale using MinMaxScaler. Here is an article about it ( https://www.quora.com/Minmaxscaler-vs-Standardscaler-Are-there-any-specific-rules-to-use-one-over-the-other-for-a-particular-application )

# In[ ]:


scaler = MinMaxScaler()  
scaler.fit(X_train)    
X_train_normed = pd.DataFrame(scaler.transform(X_train))
X_test_normed = pd.DataFrame(scaler.transform(X_test))


# ## Let's build our model

# I chose RandomForestRegressor as an algorithm.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rdf = RandomForestRegressor()
rdf.fit(X_train_normed, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error as MSE


# In[ ]:


y_pred = rdf.predict(X_test_normed)
rmse2 = np.sqrt(MSE(y_test, y_pred))
print("RMSE = {:.2f}".format((rmse2)))


# In[ ]:


accuracy = rdf.score(X_test_normed,y_test)
print(accuracy*100,'%')


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [100]}


# In[ ]:


clf = GridSearchCV(rdf, parameters, cv=5, n_jobs= -1)


# In[ ]:


clf.fit(X_train_normed, y_train)


# In[ ]:


clf.best_estimator_


# In[ ]:


y_pred = clf.best_estimator_.predict(X_test_normed)
rmse2 = np.sqrt(MSE(y_test, y_pred))
print("RMSE = {:.2f}".format(rmse2))
accuracy = clf.score(X_test_normed,y_test)
print(accuracy*100,'%')


# So I got an accuracy of 86.6% and Root Mean Square Error (RMSE) of 4071.35
