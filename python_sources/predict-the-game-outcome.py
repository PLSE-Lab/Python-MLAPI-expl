#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


# In[ ]:


data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")


# In[ ]:


X = data.drop( labels=["blueWins"], axis=1 )
y = data["blueWins"]


# In[ ]:


X.corr()


# In[ ]:


regModel = LinearRegression()
regModel.fit(X, y)


# In[ ]:


print( "R^2="+str(regModel.score(X, y)) )


# We have bad linear regression model. Now we shall try to filter a list ofparameters to do linear model better.

# In[ ]:


corrTable = X.corr()
cols = X.columns


# In[ ]:


newCols = []
for i in cols:
    for j in cols:
        if( i!=j and abs(corrTable[i][j])<0.01 ):
            newCols.append(j)
newCols = list( pd.Series(newCols).unique() )
print( len(newCols) )


# In[ ]:


X = data[newCols]
regModel = LinearRegression()
regModel.fit(X, y)
print( "R^2="+str(regModel.score(X, y)) )


# After filtering we see model is worse in every situation. **So we can't predict win of blue using linear regression built in this data.** So let's try to build another model for predicting.

# In[ ]:


from sklearn import tree
treeRegressor = tree.DecisionTreeRegressor(
    min_samples_leaf=100,
    max_leaf_nodes=2
)
treeRegressor.fit(X, y)


# In[ ]:


print( "R^2="+str(treeRegressor.score(X, y)) )


# In[ ]:


from sklearn import tree
treeClassifier = tree.DecisionTreeClassifier(
    min_samples_leaf=100,
    max_leaf_nodes=2
)
treeClassifier.fit(X, y)
print( "R^2="+str(treeClassifier.score(X, y)) )


# As we see, **decision tree is better than linear regression model**. Now let's try to build logistic regression model.

# In[ ]:


tree.plot_tree(treeClassifier)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression()
logRegModel.fit(X, y)
print( "R^2="+str(logRegModel.score(X, y)) )


# As we see, **the best model**, built by us, **for predicting wining of blue command is classification tree**.
