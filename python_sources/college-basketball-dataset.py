#!/usr/bin/env python
# coding: utf-8

# # What makes certain Colleges better than others at Basketball?
# 
# This dataset provided to us by Andrew Sundberg is fantastically prepared for us, so there's little we need to do in terms of data cleaning.
# 
# However, while finding ensuring a clean and robust dataset is a large part of a data scientists role, another just-as-important part is the art of answering a relevant question.
# 
# We could use the dataset to find lots of historical trivia, but while it might be good to know, it's not something that I would consider particularly relevant. Instead, we'll be trying to find out which features correlate with success.
# 
# In short, what makes good College Basketball teams good?

# # Import Libraries
# 
# All the libraries used in this notebook a fairly standard - sometimes you don't need state of the art tech to do what you need

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import polyfit


# # Load and Process Data
# 
# We load the data and look at the column names - the Kaggle dataset description contains good metadata about what each one means

# In[ ]:


raw_data = pd.read_csv("../input/college-basketball-dataset/cbb.csv")
raw_data.columns


# For this analysis I want to estimate performance from raw Basketball statistics (such as Effective Field Goal Percentage), and not estimated statistics such as BARTHAG. We drop these from the dataset before continuing

# In[ ]:


data_subset = raw_data.copy()
data_subset.drop(['ADJOE','ADJDE','BARTHAG','ADJ_T','WAB','POSTSEASON','SEED'],axis=1,inplace=True)


# We also want to create our target column - Win percentage. Once we have it, we don't need to the two original columns any more

# In[ ]:


data_subset['Win%'] = data_subset['W']/data_subset['G']

data_subset.drop(['W','G'],axis=1,inplace=True)


# # Exploratory Analysis
# 
# We can see that shooting categories are most highly correlated with Win%. There is also a clear order - Effective Field Goal % is first, followed by 2 point %, followed by 3 point %. Free-Throw rate is the relevant statistic with the lowest correlation of all.

# In[ ]:


data_subset.corr()['Win%'].sort_values()[:-1] #This removes Win%, which would otherwise be 100% correlated with itself


# A visual comparison of the two highest and lowest scoring features shows the clear correlation present in one and not the other. However, despite this, there is still clear variance remains in the EFG_O graph, meaning that it alone cannot explain team success

# In[ ]:


data_subset.plot.scatter(x = 'EFG_O',y='Win%');


# In[ ]:


data_subset.plot.scatter(x = 'FTR',y='Win%');


# Given teams don't play in the same conferences, it is reasonable to expect this to have a significant effect on performance too. The B12 and ACC are the highest, the MEAC and SWAC are the lowest

# In[ ]:


data_subset.groupby(['CONF'])['Win%'].mean().sort_values(ascending=False).plot(kind='bar', figsize = (10,7));


# Finally, we can also use the data to find the best and worst seasons of all time:
# 
# In 2015, San Jose State and Grambling State lost all their conference games, while in the same year Kentucy recorded a 97.4% win rate. A quick look at the columns for shooting statistics shows a very clear story about what differentiates good teams from the bad

# In[ ]:


data_subset[data_subset['Win%'] == data_subset['Win%'].min()]


# In[ ]:


data_subset[data_subset['Win%'] == data_subset['Win%'].max()]


# # Feature Selection
# 
# We have a lot of features to make our analysis from, but some of them (such as the shooting figures) are highly similar. It would make more sense if we found underlying features that describe these various similar ones.
# 
# Thankfully, a technique called Principal Components Analysis allows us to do just this, by finding the combinations of features that explain the most variance of the target column.

# In[ ]:


data_subset['YEAR'] = data_subset['YEAR'].astype(str)

dummy_df = pd.get_dummies(data_subset)


# For Principal Components Analysis to be reliable, the data must first be Standardised. SKLearn comes with useful tools to do this for us, and we can apply the PCA model to our new dataset.

# In[ ]:


standard_df = pd.DataFrame(StandardScaler().fit_transform(dummy_df), columns = dummy_df.columns)

standard_df = standard_df.drop('Win%',axis=1)

pca = PCA(n_components=3)

pca_df = pd.DataFrame(pca.fit_transform(standard_df))

pca_df.columns = ['Feature1','Feature2','Feature3']

pca_df.head()


# The first component is broadly connected to Offensive Statistics

# In[ ]:


abs(pd.Series(pca.components_[0],index = standard_df.columns)).sort_values(ascending=False)[:5]


# The second component seems more connected to Defensive Statistics

# In[ ]:


abs(pd.Series(pca.components_[1],index = standard_df.columns)).sort_values(ascending=False)[:5]


# And the final seems linked to Conference choice

# In[ ]:


abs(pd.Series(pca.components_[2],index = standard_df.columns)).sort_values(ascending=False)[:5]


# # Final Analysis

# We'll use good-old fashioned Linear Regression to make predictions from our principal components.

# In[ ]:


regmodel = LinearRegression()

regmodel.fit(X = pca_df, y = dummy_df['Win%'])

outputs = regmodel.predict(pca_df)


# We can then plot a graph of predicted v actual Win % for each team in each season

# In[ ]:


# Fit with polyfit
b, m = polyfit(dummy_df['Win%'], outputs, 1)

plt.plot(dummy_df['Win%'], outputs, '.', alpha = 0.4)
plt.plot(dummy_df['Win%'], b + m * dummy_df['Win%'], '-')
plt.show()


# # Notable Results
# 
# Given our parameters, we can see which teams over-or-underperformed their seasons results, and by how much, and see how we correlate with the BARTHAG statistics

# In[ ]:


results_df = data_subset[['TEAM','YEAR','Win%']].copy()

results_df['prediction'] = outputs

results_df['difference'] = results_df['Win%'] - results_df['prediction']


# Find the top 5 underperformers and the top 5 overperformers

# In[ ]:


results_df.sort_values('difference')[:5]


# In[ ]:


results_df.sort_values('difference',ascending=False)[:5]


# These are some pretty big variations! Let's look at some individual cases

# In[ ]:


data_subset[data_subset.index == 367]


# Their performances given their Effective Field Goal performance on Offense and Defence mean that the 2018 Portland State team should have had a Win% well below 50%, yet they managed to win 64.5% of their games!

# In[ ]:


EFG_O_view = data_subset.groupby(round(data_subset['EFG_O'],0))['Win%'].mean()
EFG_D_view = data_subset.groupby(round(data_subset['EFG_D'],0))['Win%'].mean()
print(EFG_O_view[EFG_O_view.index == 49])
print(EFG_D_view[EFG_D_view.index == 55])


# In[ ]:


data_subset[data_subset.index == 292]


# In[ ]:


print(EFG_O_view[EFG_O_view.index == 48])
print(EFG_D_view[EFG_D_view.index == 57])


# Meanwhile, the 2016 Oklahoma State team should have had a win percentage between 35%-45%, but only barely scraped into the 30% mark. Perhaps playing in the Big 12 means that normally good performance isn't always good enough

# # Conclusions
# 
# This notebook is not complete, and there will be more to come. However, what should be clear from it so far is the large amount of variance present in sports. It's what keeps us watching. Indeed, even though we know which features are mostly correlated with success, and which are not, there are still many cases where the top-level figures don't tell the whole story! Maybe your next notebook will help illuminate the reasons why...
