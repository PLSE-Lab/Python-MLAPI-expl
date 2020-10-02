#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


# Read in dataset
df = pd.read_csv('/kaggle/input/disease-data/Disease.csv')


# In[ ]:


# Take a glance at the data we are working with
df.head()


# In[ ]:


# Unnamed: 0 seems to be indentical to the index, so let's drop that from the dataframe
df.drop('Unnamed: 0', axis=1, inplace=True)


# We see that, fortunately for us, all the datatypes are numerical, allowing us a lot of possible features to run regression models with. We also see that there are no null values.
# 
# PS: I have no idea what any of these features mean...

# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


# Let's take a quick glance at how each column correlates with the others
df.corr()


# I think it makes the most sense to try to predict the Disease Severity based on these markers. Here we'll find features with the strongest correlation to Disease Severity by converting every value to an absolute value.
# 
# We can tinker with nlargest for the final output. Looping over each combination of features is doable for this dataset, but would become a more expensive operation with another. Here we get the 21 (all of them) best correlated features to Disease Severity, convert them to a list and drop the top one, because the label will always have a perfect correlation with itself.

# In[ ]:


abs_df = df.corr().abs()
best_chance_features = abs_df.nlargest(21, 'Disease Severity').index.to_list()[1:]


# In[ ]:


# I'll make a pairplot with a few good features in seaborn to visualize things more easily
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
sns.set(palette='Set1')
sns.pairplot(df[['Disease Severity','PLEKHM1','PSMB6','BX440400', 'LOC440104','PHKG1']], diag_kind="kde")


# Let's import what we'll need for running linear regression, splitting the dataset into training and testing parts, and evaluating the performance of the model with metrics.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Let's define a function that loops over our best_chance_features, establishes our X features and y label, splits our data into training and testing data, instantiates, fits, and predicts for our testing data, and finally returns a dictionary with keys as the feature columns and values as the RMSE associated with them being the features.

# In[ ]:


def feature_looper(best_chance_features):
    
    scores = {}
    
    for i in range(len(best_chance_features)):
    
        feature_cols = best_chance_features[:i+1]
        
        X = df[feature_cols]
        y = df['Disease Severity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        lreg = LinearRegression()
        lreg.fit(X_train,y_train)
        y_pred = lreg.predict(X_test)

        scores[str(feature_cols)] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    return scores


# In[ ]:


scores = feature_looper(best_chance_features)

# This will tell us which features gave us the best result!
for k,v in scores.items():
    if v == min(scores.values()):
        print("N_largest should be {}".format(len(k.split(','))))
        print(k)
        print("RMSE: {}".format(v))


# In[ ]:


# If your CPU can handle it, you can loop through every possible combination of some subset of features instead.
# We may see better results doing this. I've filtered by making sure to include one particularly good feature

from itertools import combinations

combos=[]
for i in range(15, len(best_chance_features)+1):
    for subset in combinations(best_chance_features, i):
        # Because PLEKHM1 has such a high correlation with Disease Severity
        # I think all good models will have that feature
        if 'PLEKHM1' in subset:
            combos.append(list(subset))
len(combos)


# In[ ]:


def feature_looper_combos(best_chance_features):
    
    scores = {}
    
    for i in combos:
    
        feature_cols = i

        X = df[feature_cols]
        y = df['Disease Severity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        lreg = LinearRegression()
        lreg.fit(X_train,y_train)
        y_pred = lreg.predict(X_test)

        scores[str(feature_cols)] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    return scores


# In[ ]:


scores_combos = feature_looper_combos(best_chance_features)

for k,v in scores_combos.items():
    if v == min(scores_combos.values()):
        print(k)
        print("RMSE: {}".format(v))


# When we are convinced we won't find a much better RMSE, we should retrain the model with the entired dataset including the selected features. However, the dataset is very small, so we should not expect very good numbers at this point.
