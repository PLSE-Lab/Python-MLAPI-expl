#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


re = pd.read_csv('../input/RE.csv')


# In[ ]:


re.head()


# In[ ]:


re.shape


# In[ ]:


re.columns


# In[ ]:


re1 = re.drop(['No', 'X1 transaction date'],1)


# In[ ]:


re1.head()


# In[ ]:


re1.isnull().sum()


# In[ ]:


re1.info()


# In[ ]:


re1.describe()


# In[ ]:


re1['X4 number of convenience stores'].unique()


# In[ ]:


re1.skew()


# In[ ]:


re1.kurt()


# Kurtosis > 3 is said to be platykurtic. Here only the distance from the metro station is said to be platykuric and rest all are leptokurtic.

# In[ ]:


print(np.median(re1['X2 house age']))
print(np.median(re1['X3 distance to the nearest MRT station']))
print(np.median(re1['X4 number of convenience stores']))
print(np.median(re1['X5 latitude']))
print(np.median(re1['X6 longitude']))
print(np.median(re1['Y house price of unit area']))


# In[ ]:


np.mean(re1)


# In all the cases except for latitude and longitude median < mean hence the data is positively skewed.

# In[ ]:


import seaborn as sns
sns.pairplot(re1, diag_kind = 'kde')


# The pair plot clearly explains the measure of skewness and the measure of peakedness of each of the features.
# 
# Kurtosis > 3 is said to be platykurtic. Here only the distance from the metro station is said to be platykuric and rest all are leptokurtic.

# 2.	Split the input data into dependent and independent variables

# In[ ]:


X = re1.drop('Y house price of unit area',1)
y = re1['Y house price of unit area']


# 3.	Data split into test and train (70, 30 ratio)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100) 


# 4.	Apply Bagging classifier and train the model

# Bagging classifier cannot be applied as the target variable is continuous. So bagging regressor is applied.

# In[ ]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()


# In[ ]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 100)


# In[ ]:


dt = DecisionTreeRegressor(max_depth = 4, random_state = 100)
br = BaggingRegressor(base_estimator = dt, n_estimators = 100, random_state = 100)


# In[ ]:


rf = RandomForestRegressor()
br_fit = br.fit(X_train,y_train)


# In[ ]:


br_fit


# In[ ]:


br1 = BaggingRegressor(base_estimator = rf, random_state = 100)


# In[ ]:


br1_fit = br1.fit(X_train, y_train)


# 5.	Predict the model with test dataset

# In[ ]:


y_pred = br_fit.predict(X_test) #decision tree


# In[ ]:


y_pred = br1_fit.predict(X_test)#random forest


# 6.

# In[ ]:


# DECISION TREE BASE ESTIMATOR


# In[ ]:


print(br.score(X_train,y_train)) 


# In[ ]:


print(br.score(X_test,y_test))


# In[ ]:


# RANDOM FOREST BASE ESTIMATOR


# In[ ]:


print(br1.score(X_train,y_train))
print(br1.score(X_test,y_test))


# In[ ]:


# USING KFOLD
from sklearn.model_selection import cross_val_score
res_br=cross_val_score(br,X_train,y_train.ravel(),cv=kfold)
print(np.mean(res_br))


# In[ ]:


from sklearn.model_selection import cross_val_score
res_br1=cross_val_score(br1,X_train,y_train.ravel(),cv=kfold)
print(np.mean(res_br1))


# INFERENCES:
# 
# The accuracy for bagging regressor with Random forest as the base estimator is more than that with the decision tree.
# Even on applying kfold cross validation regressor with random forest as the base estimator shows high accuracy which indicates that random forest handles most of the problems like overfitting, bias and variance errors more effectively and efficiently when compared to that of a decision tree

# In[ ]:




