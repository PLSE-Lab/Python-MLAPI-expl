#!/usr/bin/env python
# coding: utf-8

# This is part of a [larger project](https://github.com/maxims94/electric-mobility-study).

# # Model training

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import neighbors
from sklearn import tree
from sklearn import linear_model
import graphviz
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('../input/dataset2.csv',index_col='country')
df.head()


# ## Split data 

# Norway is an extreme outlier and will therefore not be considered part of the training data. It has an exceptionally high market share of EVs while only having a relatively small population.

# In[ ]:


df = df.drop('Norway',axis=0)
X = df.drop('max_ev_p',axis=1)
y = df['max_ev_p']

train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)


# ## Unlabeled data

# In[ ]:


new_X = pd.DataFrame(columns=X.columns)
new_X = new_X.append(pd.Series([0,0,0,0,0],index=X.columns,name='zero'))
new_X = new_X.append(pd.Series([0,0,0,0,100],index=X.columns,name='worst'))
new_X = new_X.append(pd.Series([1,0,100000,100,0],index=X.columns,name='best'))
new_X = new_X.append(pd.Series([1,0.56,40000,67,66],index=X.columns,name='country1'))
new_X = new_X.append(pd.Series([0,0.28,55000,78,34],index=X.columns,name='country2'))

new_X


# ## Helper functions 

# In[ ]:


def add_prediction(df,pred):
    df2 = df.copy()
    df2['pred'] = pred
    return df2


# ## Model 1: KNN

# KNN is a natural choice. If two countries are similar in wealth and politics, they can expected to have a similar market share of electric vehicles.

# In[ ]:


for n in [1,2,3,4,5,10,15]:
    knn = neighbors.KNeighborsRegressor(n_neighbors=n)
    knn.fit(train_X,train_y)
    pred = knn.predict(test_X)
    print(mean_squared_error(test_y, pred))


# ### Use n = 2

# In[ ]:


knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(train_X,train_y)
pred = knn.predict(test_X)
print(mean_squared_error(test_y, pred))
add_prediction(df.loc[test_X.index], pred)


# In[ ]:


add_prediction(new_X, knn.predict(new_X))


# ## Model 2: Decision trees

# In[ ]:


for n in [1,2,3,4,5,10]:
    dtree = tree.DecisionTreeRegressor(random_state=0,max_depth=n)
    dtree.fit(train_X,train_y)
    dtree_pred = dtree.predict(test_X)
    print(n, mean_squared_error(test_y,dtree_pred))


# ### max_depth = 2 is best 

# In[ ]:


dtree = tree.DecisionTreeRegressor(random_state=0,max_depth=2)
dtree.fit(train_X,train_y)
dtree_pred = dtree.predict(test_X)
    
add_prediction(df.loc[test_X.index], dtree_pred)


# ### Visualize decision tree for n_neighbors = 2

# In[ ]:


plt.figure(figsize=(3,2))
graphviz.Source(tree.export_graphviz(dtree, feature_names=X.columns))
#graphviz.Source(tree.plot_tree(dtree))


# In[ ]:


add_prediction(new_X, dtree.predict(new_X))


# ## Model 3: Random forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=0,n_estimators=100)
rf.fit(train_X,train_y)
rf_pred = rf.predict(test_X)

print(mean_squared_error(rf_pred, test_y))

add_prediction(df.loc[test_X.index],rf_pred)


# ## Model 4: Linear Regression

# Some of the features are strongly correlated (e.g. GDP per capita and social progress index), which causes problems with linear regression models. Therefore, we will only train the model on a subset of features.

# In[ ]:


reg = linear_model.LinearRegression()

columns = ['ssm','ppp','nat_p']

train_X_reg = train_X[columns].copy()
test_X_reg = test_X[columns].copy()
new_X_reg = new_X[columns].copy()

reg.fit(train_X_reg,train_y)
reg_pred = reg.predict(test_X_reg)

print("MSE:", mean_squared_error(test_y,reg_pred))

print(reg.coef_,train_X_reg.columns)

add_prediction(df.loc[test_X.index], reg.predict(test_X_reg))


# In[ ]:


add_prediction(new_X_reg, reg.predict(new_X_reg))


# # Choice of model
# 
# ## Model 1: KNN
# 
# * The sample size is too small for KNN to work well
# * For example, the predicted share should not drop after a certain GDP per capita has been reached merely because there is a country with a high GDP per capita that has a low EV market share
# 
# ## Model 2,3: Decision trees
# 
# * The relationships between the predictor and target variables are roughly linear and don't depend on other predictor variables
# * e.g. a higher gini index means a lower market share, higher GDP per capita is correlated with a higher market share
# * Therefore, decision trees, which model non-linear relationships, are not a natural choice here
# 
# ## Model 4: Regression (Winner!)
# 
# A very simple model with many upsides:
# * It is competitive when it comes to MSE (it has 11, while random forests have 12)
# * Its predictions are **consistent with our intuition** (e.g. higher GDP per capita always means higher EV market share) and, thus, are **easily understandable**!
# * Since it is linear, **it is very robust** (e.g. the predicted value does not suddenly drop after a certain level of GDP per capita has been reached)
# * **Its predictions on unlabeled data make sense!** (Which is the end goal!)
# 
# It still has some minor problems:
# * Sometimes produces negative values (e.g. see the 'worst' unlabeled sample above)
# * It is bad at predicting non-European countries (e.g. it predicts China to have a negative EV share, even though it is one of the leaders in the industry!)
#     * But this is because it was trained mostly on European countries!
# * Cannot model complex relationships by nature
# 
# However, the upsides prevail (especially the last point) and for these reasons, **we will pick the regression model!**

# In[ ]:




