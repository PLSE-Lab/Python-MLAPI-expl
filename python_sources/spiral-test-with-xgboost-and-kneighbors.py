#!/usr/bin/env python
# coding: utf-8

# # Interpolating and Extrapolating with XGBoost and KNeighbors
# 
# You might find discussions about certain machine learning optimizers being good at **interpolating** but not so good at **extrapolating**. XGBoost is a very popular optimizationg algorithm among kagglers. Here we apply XGBoost
# to this [**simple data set**](https://www.kaggle.com/pliptor/a-visual-and-intuitive-traintest-pattern). It is a classification problem with three classes of points on a plane. We first run the classification using the KNeiborsClassifier and then compare with XGBoost and draw conclusions.
# 
# Note there are already a few samples of scripts for this dataset for various classifiers including neural networks written in R. This is a sample script using Python and sklearn. 
# 

# # Load data and split point coordinates and target values
# 

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
train = pd.read_csv("../input/train.csv");
print(train.head())
trainnp = train.values
X_train = trainnp[:,1:] 
y_train = trainnp[:,0] 


# # KNeiborsClassifier decision regions
# 
# Let's see how KNeiborsClassifier does with this data set

# In[ ]:


from mlxtend.plotting import plot_decision_regions # convenient library for plotting decision regions
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier
knclassifier = KNeighborsClassifier(n_neighbors=3, p=2)

# train the classifier
knclassifier.fit(X_train, y_train)


# In[ ]:


plot_decision_regions(X_train, y_train.astype(int), knclassifier)


# We see the KNeighborsClassifier does a good job drawing the boundaries for each class. We also see very good **interpolation**. We intepret interpolated point as points that are near or between points of the same class. **Extrapolation** is also good. We interpret extrapolated points as points away from any point from the trainig set. One could argue though that the extrapolation is failing if the arms were to continue the spiral outwards. However, we will se XGBoost does a terrible extrapolation job.  

# # XGBoost
# 
# Now let's try with xgboost. We'll be running with pretty much the default parameters. 

# In[ ]:


import xgboost as xgb

params = {
    'booster':'gbtree',
    'colsample_bylevel':1,
    'colsample_bytree':1,
    'gamma':0, 
    'learning_rate':0.1, 
    'max_delta_step':0,
    'max_depth':3,
    'min_child_weight':1,
    'n_estimators':100,
    'objective':'multi:softprob',
    'random_state':2018,
    'reg_alpha':0, 
    'reg_lambda':1,
    'seed':2018,
    'subsample':1}

xgb_clf = xgb.XGBClassifier(**params)
xgb_clf = xgb_clf.fit(X_train, y_train, verbose=True)


# In[ ]:


xgb_clf = xgb_clf.fit(X_train, y_train)

plot_decision_regions(X_train, y_train.astype(int), xgb_clf)


# # Conclusions
# 
# It seems XGBoost has no issues with **interpolation**. However, **extrapolation** is not nearly as smooth compared with that of KNeiborsClassifier. It might be improved by adjusting parameters but in my attempts it is not easy. This might be indication that dealing of unseen data that is not a result of **interpolation** may not be really well handled by XGBoost. I thought that's something to be kept in mind.This is not to say that XGBoost is bad. On the contrary, it does pretty well in many competitions. However it may just be because the unseen data typically don't deviate much from the train data.
