#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import math
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, plot_importance


# Reading the data from a csv

# In[ ]:


df = pd.read_csv(path, low_memory=False, nrows=178915)


# In[ ]:


# Cleaning the outliers from clicks

df = df[df['clicks'] <= 2.0]
df = df[df['impressions'] <= 38.5]


# In[ ]:


# Creating profit feature, calculated by decreasing the cost from the revenues
df['profit'] = df['revenue'] - df['cost']


# Creating profit_per_impr feature which is the profits divided by the impressions
df['profit_per_impression'] = df.profit / df.impressions


# Adding ctr feature, which is #clicks / #impressions to measure the success of the ad
df['ctr'] = df.clicks / df.impressions


# Adding cpi feature - cost per impression 
df['cpi'] = df.cost / df.impressions


# Adding the cpm feature -> (cost / impressions) * 1000
df['cpm'] = df.cpi * 1000


# Adding rev_per_impr feature, i.e. revenue per impression
df['rev_per_impr'] = df.revenue / df.impressions

# Creating a feature that holds the weekday of the event based on the date, using Python library called datetime
# Turning the dates to type datetime
df['eDate'] = [datetime.strptime(x, '%m/%d/%Y') for x in list(df['eDate'])]
# Add a column for day of the week, a number represents a day, 0 is monday, 1 is Tuesday and so on
df['Day_Of_Week'] = list(map(datetime.weekday, df['eDate']))


# # 1. The RTB Problem
# The problem is that we don't know for sure what will be the revenue from every impression we buy. It is not a deterministic system where we know exactly what is going to happen giving the history, so we have to model. And because it is real time and very very fast, we need algorithms for scaling, so we trade human accuracy and "control" for algo's speed and scale. 
# When the accuracy of the estimation is low the targeting will be wrong and we will usually price the impression the wrong way and loose money. This is a problem we need to handle.

# # 2. Exploring The Data And Some Insights

# In[ ]:


df.head(60)


# In[ ]:


df.describe()


# In[ ]:


print df.impressions.median()
print df.clicks.median()
print df.clicks.quantile(0.95)
print df.impressions.quantile(0.95)


# In[ ]:


df.sort_values(by='productId').head()


# In[ ]:


df.info()


# Like we can see, we have some missing values (NULL) in 'publisherCategory', 'advertiserCategory' and 'advMaturity' features - I will take care of that when I'll model

# In[ ]:


# Plotting the distribution of imprssions
sns.distplot(df['rate'])
plt.legend(['rate'])
plt.show()


# In[ ]:


# Plotting the distribution of imprssions
df_c = df[df['clicks'] < 40]
sns.distplot(df_c['clicks'])
plt.legend(['clicks'])
plt.show()


# In[ ]:


print sum(df.profit)


# In[ ]:


# Counting how many observations we have for each devise
df.groupby('deviceType').count()['eDate']


# Like we can see, we have 170,094 of the devices are mobile type, and 8,821 are tablets

# In[ ]:


df.eDate.unique()


# We look at three dates, 4/1, 4/2 and 4/3 of 2018

# In[ ]:


countries = df.country.unique()
print countries
print 'Number of countries is:', len(countries)


# In[ ]:


print 'There are',  df.productId.unique().size, 'unique apps.'


# In[ ]:


# Plotting the distribution of profit
sns.distplot(df['profit'])
plt.legend(['profit'])
plt.show()
print "Total profit mean:", df['profit'].mean()
print "Total profit median:", df['profit'].median()
print 'Max profit is:', max(df['profit'])
print 'Min profit is:', min(df['profit'])
print "std:", df['profit'].std()
print

# Now we look for profit for each devise
df_mobile = df[df['deviceType'] == 'MOBILE']
df_tablet = df[df['deviceType'] == 'TABLET']

# For mobile
print "Avg profit for mobiles:", df_mobile.profit.mean()
print "The median for mobiles is:", df_mobile.profit.median()
print
# For tablet
print "Avg profit for tablets:", df_tablet.profit.mean()
print "The median for tablets is:", df_tablet.profit.median()


# In[ ]:


print "There are", len(df.publisherCategory.unique()), "publisher categories compared to", len(df.advertiserCategory.unique()), "advertiser categories"


# In[ ]:


# Impressions Analysis 
num_of_imp = df.impressions.sum()
num_of_clicks = df.clicks.sum()
print "Total number of impressions is", num_of_imp, "total number of clicks is", num_of_clicks, "and there are", float(num_of_imp)/float(num_of_clicks), "impressions per click"


# In[ ]:


# Let's take the 5 highest publisher categories by avg ctr 
print 'Top 5 by ctr:'
print df.groupby('publisherCategory').mean().sort_values(by='ctr', ascending=False).head(5)


# In[ ]:


# Lets see how many clicks and impressions we get from each network

# How many networks do we have?
df.networkType.unique()

# Count by type
count_by_network = df.groupby('networkType').count()['eDate']
print 'Count of impressions by network:'
print count_by_network
count_3G = count_by_network[0]
count_WIFI = count_by_network[3]
print

sum_impressions_by_network = df.groupby('networkType').sum()['impressions']
print 'Sum of impressions by network:'
print sum_impressions_by_network
print 

print 'Impressions per connection for 3G:', sum_impressions_by_network[0] / count_3G
print 'Impressions per connection for WIFI:', sum_impressions_by_network[3] / count_WIFI
print 'Impressions per connection for WIFI:', sum_impressions_by_network[1] / count_by_network[1]
print 'Impressions per connection for WIFI:', sum_impressions_by_network[2] / count_by_network[2]


# In[ ]:


# Lets see for some adv category - GAME_ROLE_PLAYING the count of the publisher category
df_current = df[df['advertiserCategory'] == 'GAME_ROLE_PLAYING']
sns.countplot(x="publisherCategory", data=df_current)
plt.show()


# Like we can expect, for each category (I ran more for myself) the distribution is far from even, some adv categories go well with other publisher categories and some don't. 

# In[ ]:


# Let's see how many impressions and clicks we get each day of the week (we only have data on Sunday=6, Monday=0, Tuesday=1)

# Grouping by the day of the week and sum
df_day = df.groupby("Day_Of_Week").sum()

# Impressions
bar_test = plt.bar([0, 1, 6], df_day['impressions'], 1/1.5, color="blue")
plt.ylabel('Impressions')
plt.xlabel('Day Of The Week')    
plt.show()

# Clicks
bar_test = plt.bar([0, 1, 6], df_day['clicks'], 1/1.5, color="red")
plt.ylabel('clicks')
plt.xlabel('Day Of The Week')    
plt.show()


# # 3. Additionl Data I Would Want
# * There are features that I needed (like profit or ctr) that I calculated above.  
# * Additional data I would find useful that it is not in our database is time of the day (might has an impact on the numbers of cicks), from which website did the user come from (browsing history in general, gives better targeting) and personal data about the user (if it's even possible to have and use, also for targeting).

# # 4. US campaign. Who Are The Best Publishers?

# In[ ]:


# Creating a new table only with US users
df_us = df[df.country == 'US']


# In[ ]:


# Looking for the most profitible publisher category for users from the US
print df_us.groupby('publisherCategory').sum().sort_values(by='profit', ascending=False).head(5)['profit']


# Like we can see, the most profitiable publisher category for US users is Personalization (3063.65) and after that is GAME_CARD (229.26)

# In[ ]:


# Looking for the top publisher category by average profits for users from the US
print df_us.groupby('publisherCategory').mean().sort_values(by='profit', ascending=False).head(5)['profit']


# Like we can see, the top publisher category by average profit for US users is Personalization (18.56) and after that is Shopping (0.71). GAME_CARD is third (0.39).
# 
# Until now, the best publisher category in terms of profits is **Personalization**

# In[ ]:


# Looking for the top publisher categories by average ctr for users from the US
print df_us.groupby('publisherCategory').mean().sort_values(by='ctr', ascending=False).head(5)['ctr']


# By average ctr the top categories are in the game domain when GAME_ADVENTURE is number one (0.175) when after comes GAME_CASINO with (0.104)

# Now let's say we have confidence that our campaign will produce great ctr, where is the largest US traffic proportion? let's find the publisher categories with the highest number of impressions:

# In[ ]:


# Looking for the top publisher categories by avergae number of impressions for users from the US
print df_us.groupby('publisherCategory').mean().sort_values(by='impressions', ascending=False).head(5)[['impressions', 'profit', 'ctr']]


# Like we can see, the top publisher category by avg number of impressions is Education but, it is not profitiable. On the other hand, the two categories right after, Shopping and Personalization, are profitiable. This strengthes the conclusion the we should focus on the **Personalization** category.   

# In[ ]:


# Now let's find the most profitiable app (productId)
df_us.groupby('productId').sum().sort_values(by='profit', ascending=False).head(10)


# It's very important noting that it also depends on more stuff such as the **category of the ad**. In the question it was referred to to campaigns in plural so I put less emphasize on that, but given an ad category, I would output the best publisher categories with regards to that. For example let's take 'Games' as the ad category: 

# In[ ]:


df_us_games = df_us[df_us['advertiserCategory'] == 'Games']  # Filtering for Games adv category only
df_us_games.groupby('publisherCategory').sum().sort_values(by='profit', ascending=False).head(5)['profit']


# Like we can see, the most profitiable publisher category for american people when the ad is in the game category is News.

# **To conclude,** it depends on the requested data that we recieve, but when we only look for the best publisehrs for US users, we will see which features such as publisher category, date, device and so on work best for american users using metrics like profits but also clicks or traffic. 

# # 5. Negative ROI, why?
# ROI is negative when cost is greater than the revenue. Why is that? Possible reasons:
# * **Wrong pricing** - Can be caused by weak analysis and modeling. If we bid for a single impression, wrong estimation of ctr will cause wrong estimation of revenue and wrong pricing that might cause us to pay more than we should (or less than we should and then loose a porfitiable impression at the acution)
# * Technical or design preblems 

# # 6. Monitoring The Revenue
# I would run a script (we can scheduele the script to run automatically and permanetly in a time we set and even send the result to our emails). 
# 
# For example if we want to see every day yesterday's total revenue and the database server is mysql, we can use mySqlLdb (an interface for connecting python to mysql server) and extract what we want using the following query: SELECT SUM(revenue) FROM RTB_table WHERE eDate = yesterday. 
# 
# The yesterday date would be calculated with datetime library as follows: yesterday = datetime.now() - timedelta(days=1)
# 
# We used the revenue and eDate metrics. 

# # 7. Modeling
# In this section I will suggest an algorithm to estimate the best bid based on historical data.
# 
# Background:
# Given a new request, or an impression offered, we need to find the bid amount that will give us the highest profit for this impression. I will build a model that given the request data will estimate the ctr and from that we will calculte the revenue. According to the expected revenue we can place the right bid.
# First step will be to estimate how many clicks I will get for this impression, or more accurately, the click-through rate (ctr). After estimate that, I can multiply my prediction with the given rate and receive the predicted revenue. 

# **Preprocessing the data:**

# In[ ]:


# Fill the missing values
most_freq = df['publisherCategory'].value_counts().idxmax()
df['publisherCategory'] = df['publisherCategory'].fillna(most_freq)

most_freq_adv = df['advertiserCategory'].value_counts().idxmax()
df['advertiserCategory'] = df['advertiserCategory'].fillna(most_freq_adv)

most_freq_mat = df['advMaturity'].value_counts().idxmax()
df['advMaturity'] = df['advMaturity'].fillna(most_freq_adv)


# I had an idea of filling the missing values by creating a list of triples for every row: (publisherCategory, advertiserCategory, advMaturity) and lets say I have a missing value in advMaturity, I'll check what is publisherCategory and advertiserCategory and look what is the most frequent triple with these two and by that fill the third but I had a memory error when trying to do it so I took the more simple approach. The idea is that these categories are connected, you expect to see some categories go with a certain group that some aren't. For example you would expect to see 'GAME SPORT' ad category with 'sports' publisher category.

# In[ ]:


# Changing these features to type Categories, in order to make them readable for the model
df['country'] = df['country'].astype('category').cat.codes
df['OS'] = df['OS'].astype('category').cat.codes
df['networkType'] = df['networkType'].astype('category').cat.codes
df['deviceType'] = df['deviceType'].astype('category').cat.codes
df['publisherCategory'] = df['publisherCategory'].astype('category').cat.codes
df['advertiserCategory'] = df['advertiserCategory'].astype('category').cat.codes
df['advMaturity'] = df['advMaturity'].astype('category').cat.codes


# In[ ]:


df.head()


# In[ ]:


# Choosing the attributes I want to model with
df_to_model = df[['ctr', 'country', 'OS', 'networkType', 'deviceType', 'publisherCategory', 'advertiserCategory', 'advMaturity', 
                 'Day_Of_Week']]
print len(df_to_model)
print df_to_model.head()

# Spliting the data to train and test sets
x=df_to_model.drop('ctr',axis=1).values
y=df_to_model['ctr'].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.0,random_state=21)
y_train = np.array(y_train)
y_test = np.array(y_test)


# I chose to use knn regressor (http://www.saedsayad.com/k_nearest_neighbors_reg.htm) because I needed a regression algorithm (the output is a real number) and the knn worked best from the ones I tried. I choose k=30 after test looping through a range of k's and 30 gave me the best result. 

# **Training:**

# In[ ]:


# knn regressor
neigh = KNeighborsRegressor(n_neighbors=1, weights='distance')
neigh.fit(x, y) 


# **testing the algo's validity**

# I will do it with the score function that return the coefficient of determination (R^2) that gives us a measure of the error, or, how well the model works. The training score is for the data set that we train the model on, the test score is for the portion of the dataset that the model didn't "see" while training and it is for testing the model on completely new data. Also, I put some exapmles for the predicted revenues next to the actual revenues in order to compare with our eyes how close they are.  

# In[ ]:


neigh_score_train = neigh.score(x, y)
print('Training score: ',neigh_score_train)
neigh_score_test = neigh.score(x_test, y_test)
print('Testing score: ',neigh_score_test)


# In[ ]:


prediction_arr = neigh.predict(np.array(df[['country', 'OS', 'networkType', 'deviceType', 'publisherCategory', 
                                            'advertiserCategory', 'advMaturity', 'Day_Of_Week']]))


# What is the score of the whole dataset:
neigh_score = neigh.score(x, y)
print "score of the whole dataset:", neigh_score

df['predicted_ctr'] = prediction_arr
df['pred_revenue'] = df.predicted_ctr * df.impressions * df.rate
print df.head(20)[['pred_revenue', 'revenue']]


# **How we would bid:**
# Let's say I bid for the impression at index 18. I estimated that the revenue will be 0.643, I take a safe range and pricing according to 0.5. Let's say I want 30% margins, so I place a bid of 0.5*1.3 = 0.65 (research showed that in second price option the best strategy is to bid the amount that we actually want to pay). I'll do it for every impression. As more data flows into the system we can improve the accuracy of the model by keep training it. 

# I used nothing but Python and its libraries mentioned at the top of the page. All the code is here in this file

# # lightGBM Modeling

# In[ ]:


df_train = df[:120000]
df_test = df[120000:]

df_train_set, df_val_set = train_test_split(df_train, test_size=0.25, random_state=1989)

features = ['country', 'OS', 'networkType', 'deviceType', 'publisherCategory', 
                                            'advertiserCategory', 'advMaturity', 'Day_Of_Week']


# In[ ]:


lgb = LGBMRegressor(n_estimators=10**4, max_depth=9, colsample_bytree=0.7, subsample=0.9, learning_rate=0.5)
lgb.fit(df_train_set[features], df_train_set['ctr'], 
        eval_set=[(df_train_set[features], df_train_set['ctr']), 
                  (df_val_set[features], df_val_set['ctr'])], 
        verbose=100, early_stopping_rounds=10)


# In[ ]:


plot_importance(lgb)


# We will take out device type and OS

# In[ ]:


df_train['ctr_pred'] = lgb.predict(df_train[features])
df_test['ctr_pred'] = lgb.predict(df_test[features])


# In[ ]:


print (df_train['ctr'] - df_train['ctr_pred']).abs().mean()
print (df_test['ctr'] - df_test['ctr_pred']).abs().mean()


# In[ ]:


features2 = ['country', 'publisherCategory', 'advertiserCategory', 'advMaturity']


# In[ ]:


lgb = LGBMRegressor(n_estimators=10**4, max_depth=9, colsample_bytree=0.7, subsample=0.9, learning_rate=0.5)
lgb.fit(df_train_set[features2], df_train_set['ctr'], 
        eval_set=[(df_train_set[features2], df_train_set['ctr']), 
                  (df_val_set[features2], df_val_set['ctr'])], 
        verbose=100, early_stopping_rounds=10)


# In[ ]:


df_train['ctr_pred'] = lgb.predict(df_train[features2])
df_test['ctr_pred'] = lgb.predict(df_test[features2])

# Negative cannot happen, so every negative prediciton map it to zero
df_train['ctr_pred'] = [n if n >= 0 else 0 for n in df_train['ctr_pred']]
df_test['ctr_pred'] = [n if n >= 0 else 0 for n in df_test['ctr_pred']]


# In[ ]:


print (df_train['ctr'] - df_train['ctr_pred']).abs().mean()
print (df_test['ctr'] - df_test['ctr_pred']).abs().mean()


# In[ ]:


sns.distplot(df_train['ctr'])
sns.distplot(df_train['ctr_pred'])

plt.legend(['train_ctr', 'train_ctr_pred'])


# In[ ]:


sns.distplot(df_test['ctr'])
sns.distplot(df_test['ctr_pred'])

plt.legend(['test_ctr', 'test_ctr_pred'])


# In[ ]:


plot_importance(lgb)


# In[ ]:


df_pred_actual = df_test[['ctr', 'ctr_pred']]
df_pred_actual.head(30)


# # Results Summary
# I used another algorithms called lightGBM that supposed to be more accurate and has a built in function that let me know which features are important and which are noise. 
# I evaluated the accuracy with calculating the average error of the predictions. I also plotted it. 
# 

# # Lior Katz
