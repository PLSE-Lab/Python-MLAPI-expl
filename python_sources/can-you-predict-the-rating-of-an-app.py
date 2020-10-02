#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# One of my friends suggested that I look into this dataset to see if we can predict the rating of an app with given data. In addition, I wanted to do a simple hypothesis testing to see if games perform better than other categories of apps. To try and predict the ratings, I decided to use simple regressions. Also, since the full version of the data has more data points, I ended up using it over the 32k data. 
# 
# In the end, what I found was that the given data is not enough. General data such as the category or the number of reviews are not good predictors. I assume that better predictors would be data that evaluates user experience. For example, average number of times the user opens the app a day or a week, average number of minutes spent in the app per day, etc. When it comes to the hypothesis testing, I found that games do not really perform better than games. 

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats import weightstats as stests

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv('/kaggle/input/google-playstore-apps/Google-Playstore-Full.csv')
data.head()


# # Initial data cleaning

# I dropped columns that I thought were unnecessary and renamed 'Content Rating' to 'CR,' trying to avoid having spaces in the names of the columns.
# 
# The name of the app might play a role in the rating, but I assume it was minimal, hence why I removed it. The same logic applies to when the app was last updated and min/max versions required.

# In[ ]:


data = data.drop(columns = ['App Name', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Last Updated', 'Minimum Version', 'Latest Version'])
data = data.rename(columns={'Content Rating':'CR'})


# I did basic exploration of the values for Installs, Price, Reviews, and Rating. I found that with Installs that I probably should just convert it to an int value. In Price, I found couple of string that I got rid of; I also decided to treat the Price as 'Free' for 0 and 'Paid' for 1. With Reviews, I just converted the values to float. With Rating, I found that there were about 10-15 values that basically were words, so I only got values with numbers in them. 

# In[ ]:


data = data[data.Size.str.contains('\d')]
data.Size[data.Size.str.contains('k')] = "0."+data.Size[data.Size.str.contains('k')].str.replace('.','')
data.Size = data.Size.str.replace('k','')
data.Size = data.Size.str.replace('M','')
data.Size = data.Size.str.replace(',','')
data.Size = data.Size.str.replace('+','')
data.Size = data.Size.astype(float)

data = data[data.Installs.str.contains('\+')]
data.Installs = data.Installs.str.replace('+','')
data.Installs = data.Installs.str.replace(',','')
data.Installs.astype(int)

data.Price = data.Price.str.contains('1|2|3|4|5|7|8|9').replace(False, 0)

data = data[data.applymap(np.isreal).Reviews]
data.Reviews = data.Reviews.astype(float)

data = data[data.Rating.str.contains('\d') == True]
data.Rating = data.Rating.astype(float)


# When looking at categories provided initially, I immediately noticed that categories were basically divided into Games and Non-Games. Thus, I decided to split the dataset into two.

# In[ ]:


data.Category.unique()


# In[ ]:


data.Category = data.Category.fillna('Unknown')
games = data[data.Category.str.contains('GAME', regex=False)]
other = data[~data.Category.str.contains('GAME', regex=False)]


# Basic cleaning of outliers.

# In[ ]:


z_Rating = np.abs(stats.zscore(games.Rating))
games = games[z_Rating < 3]
z_Reviews = np.abs(stats.zscore(games.Reviews))
games = games[z_Reviews < 3]

z_Rating2 = np.abs(stats.zscore(other.Rating))
other = other[z_Rating2 < 3]
z_Reviews2 = np.abs(stats.zscore(other.Reviews))
other = other[z_Reviews2 < 3]


# Initial hypothesis testing before doing more exploration. As I mentioned above, I wanted to see which categories perform better than the others. It seems that games generally perform worse than other categories.

# In[ ]:


games_mean = np.mean(games.Rating)
games_std = np.std(games.Rating)

other_mean = np.mean(other.Rating)
other_std = np.std(games.Rating)

print('Games mean and std: ', games_mean, games_std)
print('Other categories mean and std: ', other_mean, other_std)

ztest, pval = stests.ztest(games.Rating, other.Rating, usevar='pooled', value=0, alternative='smaller')
print('p-value: ', pval)


# # EDA

# ## Games

# By doing basic EDA and plotting simple graphs, the first thought that I had was that there is not much of a correlation between Rating and other independent variables. In the vase of Reviews vs Rating, we can see that, generally, number of reviews does not really affect the rating.
# 
# Something interesting that I noticed was that there are not a lot of Casino games, but they perform on average better than other categories. Also, it seems that not only puzzles are popular, but also people rate puzzles higher than other games.

# In[ ]:


f, ax = plt.subplots(3,2,figsize=(10,15))

games.Category.value_counts().plot(kind='bar', ax=ax[0,0])
ax[0,0].set_title('Frequency of Games per Category')

ax[0,1].scatter(games.Reviews[games.Reviews < 100000], games.Rating[games.Reviews < 100000])
ax[0,1].set_title('Reviews vs Rating')
ax[0,1].set_xlabel('# of Reviews')
ax[0,1].set_ylabel('Rating')

ax[1,0].hist(games.Rating, range=(3,5))
ax[1,0].set_title('Ratings Histogram')
ax[1,0].set_xlabel('Ratings')

d = games.groupby('Category')['Rating'].mean().reset_index()
ax[1,1].scatter(d.Category, d.Rating)
ax[1,1].set_xticklabels(d.Category.unique(),rotation=90)
ax[1,1].set_title('Mean Rating per Category')

ax[2,0].hist(games.Size, range=(0,100),bins=10, label='Size')
ax[2,0].set_title('Size Histogram')
ax[2,0].set_xlabel('Size')

games.CR.value_counts().plot(kind='bar', ax=ax[2,1])
ax[2,1].set_title('Frequency of Games per Content Rating')

f.tight_layout()


# Some categorical encoding.

# In[ ]:


games_dum = pd.get_dummies(games, columns=['Category','CR','Price'])


# I got this code from [here](https://www.geeksforgeeks.org/exploring-correlation-in-python/). As I predicted above, we can see that there is no correlation between Rating and other independent variables.

# In[ ]:


corrmat = games_dum.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# ## Other Categories

# After doing some EDA for Games, I figured that I should do the same for other categories. I decided that I get top 17 categories here, since in games I only had 17 types. I went through the same process for EDA here, so I think there is not much of an explanation needed here.

# In[ ]:


other = other[other.Category.map(other.Category.value_counts() > 3500)]


# In[ ]:


f, ax = plt.subplots(3,2,figsize=(10,15))

other.Category.value_counts().plot(kind='bar', ax=ax[0,0])
ax[0,0].set_title('Frequency of Others per Category')

ax[0,1].scatter(other.Reviews[other.Reviews < 100000], other.Rating[other.Reviews < 100000])
ax[0,1].set_title('Reviews vs Rating')
ax[0,1].set_xlabel('# of Reviews')
ax[0,1].set_ylabel('Rating')

ax[1,0].hist(other.Rating, range=(3,5))
ax[1,0].set_title('Ratings Histogram')
ax[1,0].set_xlabel('Ratings')

d = other.groupby('Category')['Rating'].mean().reset_index()
ax[1,1].scatter(d.Category, d.Rating)
ax[1,1].set_xticklabels(d.Category.unique(),rotation=90)
ax[1,1].set_title('Mean Rating per Category')

ax[2,0].hist(other.Size, range=(0,100),bins=10, label='Size')
ax[2,0].set_title('Size Histogram')
ax[2,0].set_xlabel('Size')

other.CR.value_counts().plot(kind='bar', ax=ax[2,1])
ax[2,1].set_title('Frequency of Others per Content Rating')

f.tight_layout()


# In[ ]:


other_dum = pd.get_dummies(other, columns=['Category','CR','Price'])


# In[ ]:


corrmat = other_dum.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# # Regression

# Even if I found that there is no correlation between Rating and other independent variables, I still decided to run regressions, just for the sake of it.

# In[ ]:


y = games_dum.Rating
X = games_dum.drop(columns=['Rating'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# By looking at the R squared value, we see that running a linear regression is basically not helpful at all.

# In[ ]:


reg = LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test,pred)

print('MAE: ', mae)
print('RMSE: ', np.sqrt(mse))
print('R2: ', r2)


# Same with polynomial degrees. I was curious to see if it would give me better results.

# In[ ]:


d = range(4)
for degree in d:
    poly = PolynomialFeatures(degree=degree)
    Xpoly = poly.fit_transform(X)
    Xpoly_test = poly.fit_transform(X_test)

    polyreg = LinearRegression()
    polyreg.fit(Xpoly, y)
    predpoly = polyreg.predict(Xpoly_test)

    mae2 = mean_absolute_error(y_test, predpoly)
    mse2 = mean_squared_error(y_test, predpoly)
    r2poly = r2_score(y_test,pred)
    
    print('Degree: ', degree)
    print('MAE: ', mae2)
    print('RMSE: ', np.sqrt(mse2))
    print('R2: ', r2poly)


# # Conclusion
# 
# As we can see, number of reviews, size, cost, and category of an app are not good predictors. After a certain point, most of the apps perform the same way. 
# 
# One suggestion I would have to improve the model is to add more independent variables that would be helpful. I assume looking into user experience data, such as average number of times users open the app during the day/the week, average amount of minutes spent on the app, etc., is going to be useful.
