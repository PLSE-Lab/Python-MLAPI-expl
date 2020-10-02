#!/usr/bin/env python
# coding: utf-8

# # Import the datasets, ensemble, tree and r2 score form sklearn

# In[ ]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import tree


# # Load the boston dataset and split it into training and testing dataset

# In[ ]:


(X,y) = datasets.load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)


# # Create a base estimator. A stump regressor in this case

# In[ ]:


stump = tree.DecisionTreeRegressor(max_depth=1)


# # Create an AdaBoost regressor with the weak learner - stump in this case

# In[ ]:


AdaRegressor = ensemble.AdaBoostRegressor(base_estimator = stump, loss="linear", n_estimators=10, random_state=20)
AdaRegressor.fit(X_train, y_train)


# # Predict the test data and calculate the r2 score

# In[ ]:


y_predict = AdaRegressor.predict(X_test)
r2_score(y_test, y_predict)


# 
