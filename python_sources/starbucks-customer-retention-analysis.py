#!/usr/bin/env python
# coding: utf-8

# # Don't go to Starbucks Everyday

# Till this date, Starbucks remain as the largest coffeehouse. Labelled as premium coffee brand, some financial experts said that people especially millenials should stop habit of buying Starbucks if they don't want to go broke. 

# # Starbucks customer remain loyal

# 
# This dataset was a part of my project assignment for subject Customer Behaviour Analysis. Due to time contraints, we only manage to get a bit more than 100 respondents and all respondents residing in Malaysia
# 
# The objective of this study is to understand the behaviour of customer that will remain loyal to Starbucks. What makes them continue buying Starbucks

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
starbucks = pd.read_csv('/kaggle/input/starbucks-customer-retention-malaysia-survey/Starbucks satisfactory survey encode cleaned.csv')
starbucks.columns 


# In[ ]:


starbucks.head()


# In[ ]:


print("Starbucks data set dimensions : {}".format(starbucks.shape))


# In[ ]:


# review the target class size
# 0 yes will continue buying starbucks, 1 no will not continue buying starbucks
starbucks.groupby('loyal').size()


# We can see that the data is imbalance. More respondents are loyal Starbucks customer

# In[ ]:


#delete Id column since we won't need this
starbucks = starbucks.drop('Id', axis=1)


# In[ ]:


#starbucks.groupby('loyal').hist(figsize=(20, 20))


# ## Feature Engineering
# 
# We are going to use Chi Square to pick top 10 attributes

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = starbucks.iloc[:,0:21]  #independent columns
y = starbucks.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# We can see that priceRate give the highest influence to retain Starbucks Customer

# In[ ]:


feature_names = ['priceRate', 'membershipCard', 'spendPurchase', 'productRate', 'status', 'visitNo', 'timeSpend', 'ambianceRate','location','method']
X = starbucks[feature_names]
y = starbucks.loyal


# In[ ]:


#import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


#initialize
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


# In[ ]:


#evaluation method - train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = starbucks.loyal, random_state=0)


# In[ ]:


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# From above results we can see that Gaussian Naive Bayes give the highest accuracy
