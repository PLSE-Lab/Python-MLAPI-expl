#!/usr/bin/env python
# coding: utf-8

# # Santander Value Prediction
# 
# In this notebook I will show you an approach in how to handle a dataset with several features. I will implement two techniques which are:
# 
# * PCA
# * Selection of K best
# * Random Forest With Cross Validation
# 
# Using both techniques I'll try to show a way in how to extract important features from this dataset. I use Extreme Gradient Boost for better results.
# 
# ### Important!
# ##### For future work I will try to apply permutation for each feature column and show you how the MAE is improved or worsen

# # 1. Loading libraries

# In[ ]:


"""Handle data"""
import numpy as np
import pandas as pd

"""Metrics"""
from sklearn.metrics import mean_absolute_error

"""Feature Selection"""
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV

"""Regressors"""
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1.1 Reading csv and dropping garbage

# In[ ]:


train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('test.csv')


# In[ ]:


#test_ID = test['ID']
y_train = train['target']
#y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
#test.drop("ID", axis = 1, inplace = True)


# Obtaining name of columns with only one value, which are going to be dropped from train and test datasets

# In[ ]:


columns_one_value = [element for element, ele in train.items() 
                     if (pd.Series(train[element], name=element)).nunique() == 1]


# In[ ]:


train = train.drop(columns_one_value, axis=1)
#test = test.drop(columns_one_value, axis=1)
train = train.round(16)
#test = test.round(16)


# I proceed removing columns with only one value.

# In[ ]:


colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            
train.drop(colsToRemove, axis=1, inplace=True) 
#test.drop(colsToRemove, axis=1, inplace=True) 


# In[ ]:


print("Shape train: ", train.shape)
#print("Shape test: ", test.shape)


# # 2. Feature Extraction

# This is one of the most <b>important</b> things we should priorize. There are many ways in how to extract the most important features. In this case I will proceed with this methodology:
# 
# * Using PCA I will look at what is the best number of features
# * Having the "best number of features" I will proceed to extract the features with the method "selection of K best" tested with a random forest.
# * After that, I will apply Random Forest with Croos Validation based on XGBoost

# ## 2.1 Dimensionality Reduction

# In[ ]:


pca = PCA()
pca.fit(train)


# In[ ]:


# Plotting to visualize the best number of elements
plt.figure(1, figsize=(9, 8))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('Number of Feautres')
plt.ylabel('Variance Ratio')


# So as we can see, the best number of features is in the range between [50-100] aprox. I will proceed to choose the best 100 features.

# ## 2.2 Selecting the K best Features

# In[ ]:


ytrain = np.array(y_train)
ytrain = ytrain.astype('int')


# In[ ]:


# Initialize SelectKBest function
X = SelectKBest(chi2, k=100).fit_transform(train, ytrain)


# In[ ]:


X.shape


# ## 2.3 Testing with Random Forest

# In this part I try to test how is the performance with just 100 features using Random Forest

# In[ ]:


RandForest_K_best = RandomForestRegressor()      
RandForest_K_best.fit(X, ytrain)


# In[ ]:


ypred = RandForest_K_best.predict(X)


# In[ ]:


mae = mean_absolute_error(ytrain, ypred)
print("MAE with 100 features: ", mae)


# ## 2.4 Applying Random Forest with Cross Validation based on XGBoost

# Having a dataset with 100 features, I will try to extract the most important features from this, so I will apply a more sophisticated method based on applying Croos Validation through a regressor, in this case I will use XGBoost.

# In[ ]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.1, max_depth=3, n_estimators=300) 
# Initialize the RFECV function setting 3-fold cross validation
rfecv = RFECV(estimator=xg_reg, step=1, cv=3)


# In[ ]:


rfecv = rfecv.fit(X, y_train)


# In[ ]:


print('Best number of features :', rfecv.n_features_)


# In[ ]:


# Plotting the best features with respect to the Cross Validation Score
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Score of Selected Features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# As we can see, the best score is obtained for 57 features. So lets modify our dataset and apply a regressor.

# In[ ]:


Xnew = X[:,rfecv.support_]
print(Xnew.shape)


# ## 2.5 Apply XGBoost

# In[ ]:


xg_reg.fit(Xnew ,ytrain)


# In[ ]:


ypred = xg_reg.predict(Xnew)
mae = mean_absolute_error(ytrain, ypred)
print("MAE with 57 features: ", mae)


# ## 3. Conclusion
# 
# We can observe the follows results:
# 
# * <b>MAE - RandomForest</b> with 100 features: 3658356.481798614
# * <b>MAE - XGBoost</b> with 57 features: 4853294.47253908
# 
# Further we could apply:
# * <b>Permutation</b> to see how could be the model behavior for every feature. 
# * <b>Kolmogorov Smirnov</b> to test the null hypothesis for every distribution of each feature.

# In[ ]:




