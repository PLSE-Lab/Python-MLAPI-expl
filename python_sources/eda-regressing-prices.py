#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Data Cleaning

# In[ ]:


df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', index_col='id')
print(df.shape)
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df[df['number_of_reviews'] == 0].isna().sum()[['last_review', 'reviews_per_month']]


# In[ ]:


df['reviews_per_month'].fillna(0, inplace=True)
df['last_review'] = pd.to_datetime(df['last_review'])


# So the only meaningful missing values are for 'last_review' and 'reviews_per_month'. We can see those are missing because there are no reviews for the property yet. 
# 
# We can fill the NaN values for 'reviews_per_month' with 0, but can't do much for 'last_review' as it's a date.
# 
# The only other columns with NaN values are 'name' and 'host_name'. Not terribly important for any analysis, so just leave them alone.

# ### Feature Engineering

# In[ ]:


df.drop(['host_name', 'name'], axis=1, inplace=True)


# In[ ]:


df['last_review_year'] = df['last_review'].dt.year
df['last_review_month'] = df['last_review'].dt.month
df['last_review_day'] = df['last_review'].dt.day


# No really value from these columns. The 'host_name' is usually a first name, and already have a better unique identifier for hosts with 'host_id'.
# 
# The 'name' column is just the title they gave for their listing. Completely subjective and not a lot to gain from including it.

# ### Exploration

# In[ ]:


neighbourhoods = df['neighbourhood_group'].unique()

fig = plt.figure(figsize=(8,20))
axes = fig.subplots(nrows=len(neighbourhoods), ncols=1)

for i, neighbourhood in enumerate(neighbourhoods):
    axes[i].title.set_text(neighbourhood)
    sns.scatterplot(x='longitude',y='latitude', data=df[df['neighbourhood_group'] == neighbourhood], ax=axes[i])
    
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(12,10))
plt.title('Manhattan')
sns.scatterplot(x='longitude',y='latitude', data=df[df['neighbourhood_group'] == 'Manhattan'], hue='neighbourhood')
plt.tight_layout()


# Brooklyn, Manhattan, and Queens are very dense with listings. The Bronx and Staten Island not so much.

# In[ ]:


X_cols = ['room_type', 'host_id', 'price', 'minimum_nights', 'number_of_reviews', 
          'calculated_host_listings_count', 'availability_365', 
          'last_review', 'last_review_year', 'last_review_month', 'last_review_day']
# cols = ['room_type', 'host_id', 'price']
fig = plt.figure(figsize=(20,30))
axes = fig.subplots(nrows=len(X_cols) // 2 + 1, ncols=2)

j = 0
for i, col in enumerate(X_cols):
    df[col].value_counts()[:10].sort_values(ascending=True).plot(kind='barh', title=col, ax=axes[i // 2, j])
    j = 1 if j == 0 else 0

    plt.tight_layout()


# Lot to get from this.
# 
# 1. Room type - almost no shared rooms being listed.
# 2. Host ID - two people are listing disproportionately more than others. Might be apartment listing companies.
# 3. Price - doesn't decrease linearly suprisingly.
# 4. Minimum nights - doesn't increase linearly suprisingly. You see a linear trend for [1,3], but then jumps to 30.
# 5. Number of reviews - has the linear increase expected.
# 6. Host listing count - see an expected linear trend, except for that random 327. Circles back to host ID and that these are probably companies renting out apartments.
# 7. Availability 365 - no trend whatsover. Not all surprising.
# 8. Last review - didn't expect to see any day have a disproportionate amount of reviews, but surprisingly there are a few days.
# 9. Last review year - listings have been getting more popular every year.
# 10. Last review month - summer is the hot time for travelers.
# 11. Last review day - People make the most reviews on the 1st or 30th of the month. 

# In[ ]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


sns.pairplot(df[X_cols])


# No strong relationships found.

# In[ ]:


min_nights = df['minimum_nights'].value_counts()[:20]
plt.figure(figsize=(12,10))
sns.violinplot(x='last_review_year',
            y='price', 
#             hue='last_review_year',
            data=df[df['price'] < 1000])


# Distribution or prices have stayed relatively consistent. The lower range has increased since 2011 though, which makes sense assuming inflation of New York rent prices.

# ### Regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


# In[ ]:


fig = plt.figure()
axes = fig.subplots(nrows=1, ncols=2)

axes[0].title.set_text('Prices')
axes[1].title.set_text('Log(1 + Prices)')

df['price_log'] = np.log1p(df['price'])

sns.distplot(df['price'], ax=axes[0])
sns.distplot(df['price_log'], ax=axes[1])


# In[ ]:


sns.distplot(df[(df['price_log'] <= 7.5) & (df['price_log'] >= 2.5)]['price_log'])


# In[ ]:


df.columns


# In[ ]:


df['neighbourhood_group'].value_counts()


# In[ ]:


# Feature Encoding #
# encode string columns to int cols
lbl_make = LabelEncoder()
df['neighbourhood_group_int'] = lbl_make.fit_transform(df['neighbourhood_group'])
df['room_type_int'] = lbl_make.fit_transform(df['room_type'])

df.dropna(inplace=True)


# In[ ]:


X_cols = [x for x in df.columns if x not in ['host_id', 'price', 'price_log', 'neighbourhood', 'neighbourhood_group', 'room_type', 'last_review', 'price_bin']]
X = df[X_cols]
y = df['price_log'].values

X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LassoCV(cv=10, random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r_2 = r2_score(y_test, y_pred)
adj_r_2 = 1 - (1 - r_2) * ( (df.shape[0] - 1) / (df.shape[0] - len(X_cols) - 1) )

print(mse, r_2, adj_r_2)


# In[ ]:


clf.alpha_, clf.coef_


# In[ ]:


X_cols_small = [col for col, x in zip(X_cols, clf.coef_) if abs(x) >= 1e-3]
X_cols_small


# In[ ]:


# PCA #
covar_matrix = PCA(n_components = len(X_cols))
covar_matrix.fit(X)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features


# In[ ]:


plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(97, 100.5)
plt.xticks(np.arange(0, len(X_cols)+2, 1))
plt.style.context('seaborn-whitegrid')


plt.plot(var)


# Just two variables is enough to explain 95% of the variation in the predictors.

# In[ ]:


# PLS #
arr_r_2 = []

for n_comp in range(1, len(X_cols_small)):
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train, y_train)
    r_2 = r2_score(y_test, pls.predict(X_test))
    arr_r_2.append(r_2)

idx_max = np.argmax(arr_r_2)
print(f'{idx_max}: {round(arr_r_2[idx_max] * 100, 2)}%')


# So the best we can do with PLS is 47% of the variation in prices explained. Not too bad, there's definitely a relationship.

# In[ ]:


pls = PLSRegression(n_components=4)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test)

fig = plt.figure(figsize=(12,10))
axes = fig.subplots(nrows=2, ncols=2)

axes[0, 0].title.set_text('Log Real Prices')
axes[0, 1].title.set_text('Log Predicted Prices')
axes[1, 0].title.set_text('Real Prices')
axes[1, 1].title.set_text('Predicted Prices')

sns.distplot(y_test, ax=axes[0, 0])
sns.distplot(y_pred, ax=axes[0, 1])

sns.distplot(np.exp(y_test), ax=axes[1, 0])
sns.distplot(np.exp(y_pred), ax=axes[1, 1])

Predicted prices ignore the outliers for the most part.
# In[ ]:




