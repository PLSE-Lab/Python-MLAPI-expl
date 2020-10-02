#!/usr/bin/env python
# coding: utf-8

# ## Purpose of this notebook
# #### 1. Exploratory data analysis on the Zomato ratings for Restaurants in Bangalore dataset for the purpose of inference.
# #### 2. Regression using Random Forest Regressor to predict the average cost for given parameters like locality, restaurant type, rating, votes and restaurant category.
# 
# Thank you to the person who provided the zomato ratings dataset on kaggle. Also, really appreciate the effort of all those who previously posted kernels based on this dataset. Thanks for the inspiration.
# 
# Any suggestions and feedback are most welcome. Kindly upvote if you like it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


# In[ ]:


data = pd.read_csv("../input/zomato.csv")
display_all(data.head()) 


# In[ ]:


display_all(data.describe(include='all'))


# ### Data Cleanup
# * Delete columns that are not important.
# * Rename columns to have string with lower case and no special characters.

# In[ ]:


del data['url']
del data['address']
del data['phone']
del data['location']
del data['book_table']

data.rename(columns={'approx_cost(for two people)':'cost_for_two','listed_in(city)':'locality', 'listed_in(type)':'category'},inplace=True)


# Data Cleanup (contd.)
# * Remove rows with NA or NULLS

# In[ ]:


((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)

data.rate = data.rate.replace("NEW",np.nan)
data.dropna(how='any',inplace=True)
((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)


# Data Cleanup (contd.)
# * Some restaurants are listed under multiple blocks in Koramangala. Hence, remove the block number and have the locality as Koramangala instead. 
# * Restaurants in Kalyan Nagar and Kammanahalli also have a similar overlap. Convert all to have the locality as Kammanahalli.
# * The field rest_type has inconsistent entries. For example, rest_type = 'Casual Dining, Bar' and rest_type = 'Bar, Casual Dining'. Convert these to have a unique entry.
# * Some dessert restaurants in Brigade Road are wrongly marked as Pubs and Bars. Correct the entries.

# In[ ]:


# Convert rate and cost_for_two to float
data.rate = data.rate.astype(str)
data.rate = data.rate.apply(lambda x:x.replace('/5',''))
data.rate = data.rate.astype(float)

data.cost_for_two = data.cost_for_two.astype(str)
data.cost_for_two = data.cost_for_two.apply(lambda x:x.replace(',',''))
data.cost_for_two = data.cost_for_two.astype(float)

data.loc[data['locality'].str.contains('Koramangala'), 'locality'] = 'Koramangala'
data.loc[data['locality'].str.contains('Kalyan Nagar'), 'locality'] = 'Kammanahalli'


# Some restaurants on brigade road are wrongly marked as category=Pubs and bars, correcting the category to desserts.
# There may be many such records in the dataset but for now lets clean up only these
data.loc[(~(data['rest_type'].str.contains('Pub|Bar')) & (data['category'].str.contains('Pub')) & (data['locality']=='Brigade Road'))
          , 'category'] = 'Desserts'

# standardize the data in rest_type
vals_to_replace = {'Bar, Casual Dining':'Casual Dining, Bar'
                   ,'Cafe, Casual Dining':'Casual Dining, Cafe'
                   ,'Pub, Casual Dining':'Casual Dining, Bar'
                   ,'Casual Dining, Pub':'Casual Dining, Bar'
                   ,'Pub':'Bar'
                   ,'Pub, Bar':'Bar'
                   ,'Bar, Pub':'Bar'
                   ,'Microbrewery, Bar':'Microbrewery'
                   ,'Microbrewery, Pub':'Microbrewery'
                   ,'Pub, Microbrewery':'Microbrewery'
                   ,'Microbrewery, Casual Dining':'Casual Dining, Microbrewery'
                   ,'Lounge, Casual Dining':'Casual Dining, Lounge'
                   ,'Sweet Shop, Quick Bites':'Quick Bites, Sweet Shop'
                   ,'Cafe, Quick Bites':'Quick Bites, Cafe'
                   ,'Beverage Shop, Quick Bites':'Quick Bites, Beverage Shop'
                   ,'Bakery, Quick Bites':'Quick Bites, Bakery'
                   ,'Desert Parlor, Quick Bites':'Quick Bites, Desert Parlor'
                   ,'Food Court, Quick Bites':'Quick Bites, Food Court'
                   ,'Bar, Quick Bites':'Quick Bites, Bar'
                   ,'Cafe, Lounge':'Lounge, Cafe'
                   ,'Cafe, Bakery':'Bakery, Cafe'
                   ,'Cafe, Dessert Parlor':'Dessert Parlor, Cafe'
                   ,'Cafe, Beverage Shop':'Beverage Shop, Cafe'
                   ,'Bar, Lounge':'Lounge, Bar'
                   ,'Fine Dining, Lounge':'Lounge, Fine Dining'
                   ,'Microbrewery, Lounge':'Lounge, Microbrewery'}

data['rest_type'] = data['rest_type'].replace(vals_to_replace)
data.rest_type.sort_values().unique()


# ### Save cleaned up file to feather
# * Remove Duplicate restaurants within a locality to get unique restaurants. This dataset will be used for all plotting purposes. It is also the input feed to the models.
# * Save this dataset to feather so that it can be accessed effciently.

# In[ ]:


data_unique = data.drop_duplicates(subset=['locality', 'name'], keep='first')

data_unique.reset_index(inplace=True)
os.makedirs('tmp', exist_ok=True)
data_unique.to_feather('tmp/zomato-unique')


# ### Get the cleaned up file from feather

# In[ ]:


data_unique = pd.read_feather('tmp/zomato-unique')
data_unique.head()


# ### Plot Graphs
# * Plot density, histogram and boxplot of cost_for_two to see how the values are distributed.
# * Inference: The cost_for_two ranges from a minimum of Rs. 40 to a maximum of Rs. 6000. The median is at Rs. 600. 

# In[ ]:


fig, axes = plt.subplots(3,1,sharex=True, figsize=(20,14))
data_unique['cost_for_two'].plot.density(ax=axes[0], color = "#3399cc")
sns.boxplot(ax=axes[1], x=data_unique['cost_for_two'], color = "#3399cc")
data_unique['cost_for_two'].plot.hist(ax=axes[2], bins=50, color = "#3399cc")

data_unique['cost_for_two'].describe()


# #### Plot the number of restaurants per locality:
# * Inference: Koramangala and BTM top the list with more than 800 restaurants while most localities on an average have around 400 restaurants.
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax = sns.countplot(x="locality", data=data_unique, order=data_unique['locality'].value_counts().index, color = "#3399cc")
ax.set_title("Count Per Locality")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.show()


# #### Plot histogram of the ratings value
# * Inference: 
# * * Most ratings are in the range of 3.5 to 4.5
# * * The lowest ratings are below 2.5

# In[ ]:


data_unique['rate'].plot.hist(bins=50, color = "#3399cc")


# ### Now lets find out which 10 restaurants have been rated the highest in each locality 
# * Lets select only those restaurants in the price range of Rs. 400 to Rs. 800 (This pirce range accounts for 50% of our data values).
# * Within each locality the top 10 restaurants would be those with the highest number of votes and rating.
# * Remove rows with duplicate restaurant names within the same locality. These duplicates may have crept into the data because of multiple categories assigned to a single restaurant. 
# 

# In[ ]:


data_select = data[
                   (data['cost_for_two'] >= 400) & (data['cost_for_two'] <= 800)
#                           & (data_top_all['category'].str.contains('Buffet'))
#                           & (data_top_all['rest_type'].str.contains('Food Truck'))
#                           & (data_top_all['cuisines'].str.contains('North Indian'))
#                           & (data_top_all['online_order'].str.contains('Yes'))
                 ]

data_top = data_select.sort_values(['locality', 'votes', 'rate', ],ascending=False)
data_top.drop_duplicates(subset=['locality', 'name'], keep='first', inplace=True)
data_top10 = data_top.groupby('locality').head(10)
data_top10[["locality", "name", "rest_type", "category", "cost_for_two", "votes", "rate"]]


# #### Of the top 10 rated restaurants in each locality lets find out how many are restaurant chains(restaurants with presence in multiple localities)
# * Inference: 54 restaurant chains

# In[ ]:


pd.concat(g for _, g in data_top10[["name", "locality", "cost_for_two", "votes", "rate"]].groupby("name") if len(g) > 1)


# #### Crosstab of restaurant names vs locality to find out which restaurant chains have the highest presence in the top 10 per locality.
# * Inference(from the counts in column 'All' at the extreme right):
# * * Empire Restaurant in 20 locations
# * * Onesta in 19 locations
# * * Meghana Foods in 13 locations
# * * Paradise in 11 locations
# 

# In[ ]:


pd.crosstab(data_top10.name, data_top10.locality, margins=True)


# ### Restaurants in each locality with the lowest ratings
# * Select records with rating < 2.5 (from the histogram of the rating we have seen that there are minimal records with rate < 2.5)
# * In each locality the restaurants with the lowest rating would be those with a high number of votes but rating < 2.5
# * Remove duplicates entries within each locality

# In[ ]:


data_bot = data_select.sort_values(['locality', 'votes', 'rate', ],ascending=[False,False,True])
data_bot = data_bot[data_bot['rate'] < 2.5]
data_bot.drop_duplicates(subset=['locality', 'name'], keep='first', inplace=True)
data_bot10 = data_bot.groupby('locality').head(10)
data_bot10[["locality", "name","rest_type", "category", "cost_for_two", "votes", "rate"]]


# Are there any restaurant chains among these low rated restaurants?

# In[ ]:


pd.concat(g for _, g in data_bot10[["name", "locality", "votes", "rate"]].groupby("name") if len(g) > 1)


# Are there any common names between the top 10 rated and the bottom rated restaurants?
# * Inference: No common names in the 2 lists

# In[ ]:


pd.merge(data_top10, data_bot10, on='name')


# ### Top 10 rated restaurants in all of Bangalore with cost_for_two in the range of Rs.400 to Rs.800
# * Inference: 6 restaurants out of the top 10 are in Koramangala

# In[ ]:


data_top_all = data_select.sort_values(['votes', 'rate', ],ascending=False)             
data_top_all.drop_duplicates(subset=['name'], keep='first', inplace=True)
data_top_all[["name", "cuisines", "locality", "cost_for_two", "votes", "rate"]].head(10)


# ### Lowest rated restaurants in all of Bangalore with cost for two in the range of Rs.400 - 800

# In[ ]:


data_bot_all = data_select.sort_values(['votes', 'rate', ],ascending=[False,True])
data_bot_all = data_bot_all[data_bot_all['rate'] < 2.5]
data_bot_all.drop_duplicates(subset=['name'], keep='first', inplace=True)
data_bot_all[["name", "cuisines", "locality", "cost_for_two", "votes", "rate"]].head(10) 


# ### Predict cost for two at a restaurant. What input features should we use? Lets plot the features and find out.
# 
# #### Plot the average cost_for_two per locality
# * Inference: Brigade Road, Church Street, Lavelle Road, MG Road and Residency Road are among the priciest localities to eat out with the average cost_for_two around Rs. 1000 (the black vertical line at the top of each bar represents the 95% confidence interval).

# In[ ]:


# Cost for two vs locality
fig, ax = plt.subplots(figsize=(15,5))
ax = sns.barplot(ax=ax, x='locality', y='cost_for_two', data=data_unique, color = "#3399cc")
ax.set_title("Cost_for_two Per Locality")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.show()


# #### Plot cost_for_two per type of restaurant (Pub, Casual Dining, Quick bites, Cafe, Lounge etc.)
# * Inference: At the pricier end are Microbreweries, Clubs, Lounge Bars and Fine Dining Restaurants. Quick bites, Kiosks, Dessert shops, Mess are at the lower end of the price range.

# In[ ]:


# Cost for two vs restaurant type
fig, ax = plt.subplots(figsize=(20,10))
ax = sns.barplot(ax=ax, x='rest_type', y='cost_for_two', data=data_unique, color = "#3399cc")
ax.set_title("Cost_for_two Per Restaurant Type")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.show()


# #### Plot pivot of cost_for_two for various categories of restaurants (Buffet, Cafes, Desserts, Drinks & Nightlife, Delivery, Dine-out) per locality
# * Inference: 
# * * Drinks and Nightlife, Dine-out and Buffet are among the pricier categories across most localities.
# * * All categories of restaurants are priced low at Banashankari

# In[ ]:


data_unique.pivot_table('cost_for_two', columns=['category'], index='locality' ).plot(kind='bar',figsize=(20,10))


# #### Plot rate vs cost_for_two
# 
# Inference:
# * With the exception of a few cases, mostly higher the cost_for_two, higher is the rating

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
ax = sns.barplot(ax=ax, x='rate', y='cost_for_two', data=data_unique, color = "#3399cc")
ax.set_title("Cost_for_two vs rating")
plt.show()


# #### Plot cost for two vs online_order (yes/no)
# 
# Inference:
# * Food at restaurants with online_order=No is more expensive

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))
ax = sns.barplot(ax=ax, x='online_order', y='cost_for_two', data=data_unique, color = "#3399cc")
ax.set_title("Cost_for_two vs online_order (Yes/No)")
plt.show()


# ## Predict cost_for_two using RandomForestRegressor on the data
# * Columns to use will be rest_type, locality, category, rate, votes, online_order, cost_for_two
# 
# Inference:
# * One hot encoding gives slightly better accuracy than categorical conversion of string variables
# * A noticable increase in accuracy of the model after setting the input to random sequence using df.sample(frac=1).reset_index(drop=True)
# * Train, test accuracy = [0.98, 0.88] 

# ### Get the cleaned up file from feather. To re-test model run code from here onwards.

# In[ ]:


data_unique = pd.read_feather('tmp/zomato-unique')
display_all(data_unique.head())


# Columns 'locality', 'rest_type', 'category', 'rate', 'votes', 'online_order' impact the values in 'cost_for_two'. Therefore select only those columns in the dataset which is input to our model

# In[ ]:


# Select columns and set the rows in random order before feeding to the model

data_unique = data.drop_duplicates(subset=['locality', 'name'], keep='first')
data_unique = data_unique[['locality', 'rest_type', 'category', 'rate', 'votes', 'online_order', 'cost_for_two']]
data_unique = data_unique.sample(frac=1).reset_index(drop=True)


# #### View the feature correlation (numeric features only)
# 
# Notice that cost_for_two is correlated with rate and votes

# In[ ]:


sns.pairplot(data_unique, diag_kind='kde', plot_kws={'alpha':0.2})


# #### One hot encode string data before feeding it to the model

# In[ ]:


#1 - one hot encode string data
data_unique = pd.get_dummies(data_unique)
display_all(data_unique.head(10))


# #### Set up train and test data 

# In[ ]:


X = data_unique.drop(['cost_for_two'], axis=1)
y = data_unique.cost_for_two


# In[ ]:


# Split into train and test datasets
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 2500
n_trn = len(data_unique)-n_valid
X_train, X_valid = split_vals(X,n_trn)
y_train, y_valid = split_vals(y,n_trn)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# #### Random Forest Regressor 

# In[ ]:


# Random Forest with default number of estimators
m = RandomForestRegressor(random_state=0, n_jobs=-1)
m.fit(X_train, y_train)
res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]
print(res)


# Lets see how the predictions are calculated

# In[ ]:


preds = np.stack([tree.predict(X_valid) for tree in m.estimators_])
preds.shape, preds[:,0], np.mean(preds[:,0]), y_valid


# #### Lets visualize a single tree with max_depth = 5. Double click on the chart to zoom it.

# In[ ]:


m = RandomForestRegressor(random_state=0, n_estimators=1, max_depth=5, n_jobs=-1)
m.fit(X_train, y_train)
res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]
print(res)


# In[ ]:


from sklearn.datasets import *
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image  
import graphviz
import pydot


tree = m.estimators_[0]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X_train.columns, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Show graph
Image(graph.create_png())
# Create PDF
#graph.write_pdf("iris.pdf")
# Create PNG
#graph.write_png("iris.png")


# #### Lets see what happens if we create a deeper tree (omit specify max_depth) 
# 
# Inference:
# * The accuracy improves

# In[ ]:


m = RandomForestRegressor(random_state=0, n_estimators=1, n_jobs=-1)
m.fit(X_train, y_train)
res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]
print(res)


# #### How many trees will give us the best accuracy? (code for this may take a minute or two to run)
# 
# Inference:
# * Lets choose number of trees = 150, from the plot below, beyond this the accuracy does not get any better

# In[ ]:


# Try different numbers of n_estimators 
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    m.set_params(n_estimators=n)
    m.fit(X_train, y_train)
    scores.append(m.score(X_valid, y_valid))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


# Random Forest with 150 estimators
m = RandomForestRegressor(random_state=0, n_estimators=150,n_jobs=-1)
m.fit(X_train, y_train)
res = [m.score(X_train,y_train), m.score(X_valid,y_valid)]
print(res)


# #### Plot feature importance

# In[ ]:


# Calculate feature importances
importances = m.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_train.columns[i] for i in indices]

plt.figure(figsize=(15,15))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]),importances[indices], color = "#3399cc")
plt.yticks(range(X.shape[1]), names)

plt.show()


# #### Feature importance in listed sorted order

# In[ ]:


names


# In[ ]:




