#!/usr/bin/env python
# coding: utf-8

# **Predicting the cost of treatment with a limited data. **

# In[ ]:


import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas_profiling

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/insurance.csv')


# Checking the basic structure of the data;

# In[ ]:


data.info()


# In[ ]:


data.head()


# ## Evaluate the DATA
# 
# <b>pandas_profiling</b> library is another great tool to understand the basic structure of the data.

# In[ ]:


pandas_profiling.ProfileReport(data)


# First thing first, we need to seperate our target variable from data. In this case the <u>target</u> variable is <b>"charges"</b>. Also we can create another dataframe which contains all variables except target(charges) variable with a name of data_predictors.

# In[ ]:


target = pd.DataFrame(data =data.iloc[:,-1:], index=range(len(data)))
data_predictors = pd.DataFrame(data =data.iloc[:,:-1], index=range(len(data)))


# In[ ]:


data_predictors.head()


# In[ ]:


target.head()


# Also we need to seperate categorical and continious variable because of the different preprocessing pathways. We can create <b>'data_num'</b> variable which represents all continious variables in the dataset. Alse we can clean up the categorical variables with <b>pandas.DataFrame.drop</b> attribute.

# In[ ]:


categorical_variables = ["sex", "smoker", "region"]

data_num = data_predictors.copy()
for i in categorical_variables:
    data_num = data_num.drop(i, axis=1)


# In[ ]:


data_num.head()


# It is time to handle categorical variables

# 

# In[ ]:


ohe = OneHotEncoder()

sex = OrdinalEncoder().fit_transform(data.iloc[:,1:2])
sex = ohe.fit_transform(sex).toarray()

smoker = OrdinalEncoder().fit_transform(data.iloc[:,4:5])
smoker = ohe.fit_transform(smoker).toarray()

region = OrdinalEncoder().fit_transform(data.iloc[:,5:6])
region = ohe.fit_transform(region).toarray()


# In[ ]:


sex_df = pd.DataFrame(data=sex, index=range(len(sex)), columns=['female', 'Male'])
smoker_df = pd.DataFrame(data=smoker, index=range(len(smoker)), columns=['non-smoker', 'smoker'])
region_df = pd.DataFrame(data=region, index=range(len(region)), columns=['sw', 'se', 'nw', 'ne'])


# Final dataframe consists all variables

# In[ ]:


final_df = pd.concat([data_num, sex_df, smoker_df, region_df], axis=1)


# Time to split data...

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(final_df,target,test_size=0.2, random_state=0)


# Scaling the variables

# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# Fit the data with DecisionTreeRegressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

tree_reg = DecisionTreeRegressor()


# In[ ]:


tree_reg.fit(X_train,y_train)


# Calculate Mean Square Error of the DecisionTreeRegressor model. 

# In[ ]:


charge_predictions = tree_reg.predict(X_train)
tree_mse = mean_squared_error(y_train, charge_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# Visulaize the results. Looks good..

# In[ ]:


tree_predict_df = pd.DataFrame(data=tree_reg.predict(X_test), index=range(len(tree_reg.predict(X_test))))
tree_actual_df = pd.DataFrame(data=y_test.values, index=range(len(y_test.values)))

tree_result_df = pd.concat([tree_predict_df, tree_actual_df], axis=1)
tree_result_df = pd.DataFrame(tree_result_df.values, columns=["predicted","actual"])

tree_result_df.plot(kind="scatter", x="predicted", y="actual", color='b', figsize=(5,5))


# Let's try LinearRegression

# In[ ]:


from sklearn.linear_model import LinearRegression


lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)


# In[ ]:


lin_predict_df = pd.DataFrame(data=lin_reg.predict(X_test), index=range(len(lin_reg.predict(X_test))))

lin_actual_df = pd.DataFrame(data=y_test.values, index=range(len(y_test.values)))


# Calculate Mean Square Error of the LinearRegression model.

# In[ ]:


charge_predictions = lin_reg.predict(X_train)
lin_mse = mean_squared_error(y_train, charge_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# DecisionTreeRegressor model had better results than LinearRegression.

# Visulaize the LinearRegression. 

# In[ ]:


lin_result_df = pd.concat([lin_predict_df, lin_actual_df], axis=1)
lin_result_df = pd.DataFrame(lin_result_df.values, columns=["predicted","actual"])
lin_result_df.plot(kind="scatter", x="predicted", y="actual", color='b', figsize=(5,5))


# Evaluate the both models. Tree Regression has better predictions.

# In[ ]:


import matplotlib.pyplot as plt
#import seaborn as sns

tree_result_df.plot(kind="scatter", x="predicted", y="actual", color='b',title="Tree Regression")

lin_result_df.plot(kind="scatter", x="predicted", y="actual", color='b',  title="Lineer Regression")

