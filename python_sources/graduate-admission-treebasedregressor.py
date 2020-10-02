#!/usr/bin/env python
# coding: utf-8

# **Application of tree based regressor using graduate admission data!
# **
# 
# Outline:
# * Data analysis
# * Decision Tree Regressor
# * Random Forest Regressor
# * AdaBoost Regressor
# * GradientBoost Regressor
# * ExtremeGradientBoost Regressor
# 
# 
# 
# **Data Content:**
# The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are : 1. GRE Scores ( out of 340 ) 2. TOEFL Scores ( out of 120 ) 3. University Rating ( out of 5 ) 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 5. Undergraduate GPA ( out of 10 ) 6. Research Experience ( either 0 or 1 ) 7. Chance of Admit ( ranging from 0 to 1 )
# 
# **Acknowledgements:**
#  The dataset is owned by Mohan S Acharya.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('../input/Admission_Predict.csv')
data.info()
data.head(5)


# In[3]:


corr_matrix = data.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})
plt.show()


# Looking at correlation matrix above and below blocks, most of features are highly related to chance of admission (drop serial no coloumn).

# In[4]:


corr_matrix["Chance of Admit "].sort_values(ascending=False)


# In[5]:


X,Y = data.iloc[:,1:-1].values,data.iloc[:,-1].values


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state = 33)


# In[7]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=33)
dt_reg.fit(x_train,y_train)


# One can use r2 score for regressor performance evaluation, but I used MSE for my analysis.
# 
# For deceision tree regressor, it overfits since gap between train error and test error is huge!

# In[8]:


y_train_predic = dt_reg.predict(x_train)
y_test_dt = dt_reg.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print("Decision Tree train_error is:",mean_squared_error(y_train,y_train_predic))
print("Decision Tree test_error is:",mean_squared_error(y_test,y_test_dt))


# RandomForest is a good ensemble learning which is bagging of decision tree regressor.

# In[9]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=500,random_state=33,bootstrap=True,n_jobs=-1)
rf_reg.fit(x_train,y_train)
y_test_rf = rf_reg.predict(x_test)
print("RandomForest train_error is:",mean_squared_error(y_train,rf_reg.predict(x_train)))
print("RandomForest test_error is:",mean_squared_error(y_test,y_test_rf))


# Let's try different boosting ensemble regressors. First, we look at AdaBoost.
# 
# Here, I set the random hyperparameters. We need to do grid search to tune the parameters, for this example I did not use it.

# In[10]:


from sklearn.ensemble import AdaBoostRegressor
ada_reg = AdaBoostRegressor(n_estimators=500,learning_rate=0.5,random_state=33)
ada_reg.fit(x_train,y_train)
y_test_ada = ada_reg.predict(x_test)
print("AdaBoost test_error is:",mean_squared_error(y_test,y_test_ada))


# Another popular boosting technique is Gradient Boosting, but it is prone to overfitting. If we dont monitor its learning path (i.e. testing error behavior ), it performs worse than other methods! Below we find optimal solution by monitoring testing error. N_estimator is how many trees we want use in decision tree, if it is too large you overfitt data. Below figure helps us see how error changes as we use different number of trees.

# In[11]:


from sklearn.ensemble import GradientBoostingRegressor
grbt_reg = GradientBoostingRegressor(max_depth=2,n_estimators=500,random_state=33)
grbt_reg.fit(x_train,y_train)

errors = np.zeros((500,1))
i = 0
for y_pred in grbt_reg.staged_predict(x_test):
    errors[i] = mean_squared_error(y_test,y_pred)
    i = i + 1
    #print(y_pred)

best_n_estimator = np.argmin(errors)

plt.plot(errors)
plt.xlabel('number of trees');plt.ylabel('RMSE');plt.show()

grbt_reg_best = GradientBoostingRegressor(max_depth=2,n_estimators=best_n_estimator)
grbt_reg_best.fit(x_train,y_train)
y_test_gbrt = grbt_reg_best.predict(x_test)

print("GBR test_error is:",mean_squared_error(y_test,y_test_gbrt))


# So far our best model is from, gradient boost regressor so lets look how it performs on test data!
# 
# As you can see, most of prediction errors for instances are centered around which validates our results.

# In[12]:


plt.hist(y_test-y_test_gbrt,bins=20);
plt.xlabel('Prediction Error');plt.ylabel('Frequency');plt.show()


# There is a similar gradient boosting method called Extreme gradient boosting (Xgb) which is an optimization of random forest. And if one can tune parameters well, it gives pretty good result as well. That's right use grid search to find best parameters.

# In[13]:


import xgboost as xgb
xgb_reg = xgb.XGBRegressor(random_state=33,num_parallel_tree=500,learning_rate=0.05,early_stopping_rounds=10,max_depth=2)
xgb_reg.fit(x_train,y_train)
y_test_xgb = xgb_reg.predict(x_test)

print("GBR test_error is:",mean_squared_error(y_test,y_test_xgb))


# In[14]:


plt.hist(y_test-y_test_xgb,bins=20);
plt.xlabel('Prediction Error');plt.ylabel('Frequency');plt.show()

