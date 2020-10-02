#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Importing Data

# In[ ]:


df = pd.read_csv("../input/concrete-dataset/Concrete_Data.csv")


# In[ ]:


df.head()


# We want to measure the Compressive Strength of the Concrete, therefore, we'll drop this feature from the dataset.

# # Checking Data and Renaming Columns

# In[ ]:


df.columns


# In[ ]:


df_new = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'BFS',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly_Ash',
       'Water  (component 4)(kg in a m^3 mixture)':'Water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarser_agg',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine_agg',
       'Age (day)':'Days',
       'Concrete compressive strength(MPa. megapascals)':'Comp_str'})


# In[ ]:


df_new.head()


# In[ ]:


df_new.columns


# In[ ]:


df_new.describe()


# Now let's check the columns before we start applying some visualiations

# # Checking Missing Values

# In[ ]:


df_new.isnull().sum()


# No missing values. Proceeding to visuals

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Data Visualization

# In[ ]:


mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

f, ab = plt.subplots(figsize=(15,10))
sns.heatmap(df_new.corr(), annot=True, mask=mask)


# Initially, we can see a great correlation in the features, Cement, Superplsticizer and Days. We'll surely use them as features for our models.

# In[ ]:


sns.regplot(x='Cement',y='Comp_str', data=df_new)


# It seems is a positive relationshio, let' check more features

# Since this variable have many values in 0, let's try to run a jointplot

# In[ ]:


sns.jointplot(x='BFS',y='Comp_str', kind='kde',data=df_new)


# Data here seems to have more variance. Most of our data have no BFS (Please refer to this link for more details on its use: https://theconstructor.org/concrete/blast-furnace-slag-cement/23534/)
# 
# It seems that this feature is more important in certain cases on the industry. We see that with values of 200, the Comp_str decreases a little, which is expected. For smaller values than 100 the comp_str is more disperse, so it may be related to the proportin of the other features. 

# In[ ]:


sns.jointplot(x='Fly_Ash',y='Comp_str',  kind='kde', data=df_new)


# Same thing as the BFS data, but more concentrated.

# In[ ]:


sns.regplot(x='Water',y='Comp_str', data=df_new)


# In[ ]:


sns.jointplot(x='Superplasticizer',y='Comp_str',kind='kde',data=df_new)


# Similar to BFS and Fly Ash. The use of superpasticizes aims to reduce the use of Water in concrete (This can be seen in their correlation which is very negative -0.66), which would decrease the Comp_str even more.

# In[ ]:


sns.regplot(x='Coarser_agg',y='Comp_str', data=df_new)


# In[ ]:


sns.regplot(x='Fine_agg',y='Comp_str', data=df_new)


# The effect of Coarser and Fina Aggregators seems to have a small negative impact when incresing their percentage. We'll use this feature.

# In[ ]:


sns.jointplot(x='Days',y='Comp_str',  kind='kde', data=df_new)


# It looks like that, with more days, our concrete becomes a bit stronger.

# # Linear Regression

# Initially. We'll use a Linear regression to try to fit the model, since this seems the reasonable approach for this case.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


# In[ ]:


features = ['Cement', 'BFS', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarser_agg',
       'Fine_agg', 'Days']
targets = ['Comp_str']

LR = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(df_new[features], df_new[targets], test_size=0.20, random_state=42)


# In[ ]:


model = LR.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15,10))
x = np.linspace(0,80,1000)
y=x
plt.scatter(y_test, y_pred)
plt.plot(x,y)

print("The R^2 for the test data in this Linear Regression is: ", model.score(X_test,y_test))

print("The R^2 for the training data in this Linear Regression is: ", model.score(X_train,y_train))

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

print("The Root Mean Squared Error for this is: ",RMSE )


# The test dataset contain a slight higher value of R^2. Since the values are so close together, I believe the model didn't overfit. It would be alarming if the test dataset had a considerable higer value.

# # Exploring Further ML Models with GridSearchCV

# We'll now check more models, like DecisionTreeRegressor, RandomForestRegressor and SVR. But for the first time, I'm going to use a GridSearchCV to check the best hyperparameters to the models.

# In[ ]:


from sklearn import tree, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


params_SVR = [{'C':[1,10, 100, 250, 500, 750, 1000], 'max_iter':[3000, 4000, 5000, 6000]}]
params_DTR = [{'max_depth':[5,6,7,8]}]
params_RFR = [{'n_estimators':[200, 250, 300, 350, 400, 450, 500, 550, 600]}]

SVR = svm.LinearSVR()
DTR = tree.DecisionTreeRegressor()
RFR = RandomForestRegressor()


# In[ ]:


grid_SVR = GridSearchCV(SVR, params_SVR, cv=3, scoring='r2')
grid_DTR = GridSearchCV(DTR, params_DTR, cv=3, scoring='r2')
grid_RFR = GridSearchCV(RFR, params_RFR, cv=3, scoring='r2')


# In[ ]:


model_SVR = grid_SVR.fit(X_train, y_train.values.ravel())
model_DTR = grid_DTR.fit(X_train, y_train)
model_RFR = grid_RFR.fit(X_train, y_train)


# We can see here that the SVR fails for every amount of iterations, therefore, we'll not use for predictions.

# In[ ]:


print(grid_SVR.best_params_)
print(grid_DTR.best_params_)
print(grid_RFR.best_params_)


# In[ ]:


SVR_1 = svm.SVR(C=100, max_iter=400)
model_SVR1 = SVR_1.fit(X_train,y_train)
y_SVR1 = model_SVR1.predict(X_test)

DTR_1 = tree.DecisionTreeRegressor(max_depth=8)
model_DTR1 = DTR_1.fit(X_train,y_train)
y_DTR1 = model_DTR1.predict(X_test)

RFR_1 = RandomForestRegressor(n_estimators=300, criterion='mse')
model_RFR_1 = RFR_1.fit(X_train,y_train)
y_RFR1 = model_RFR_1.predict(X_test)


# # Final Graph and Data

# In[ ]:


from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15,10))
x = np.linspace(0,80,1000)
y=x
plt.scatter(y_test, y_SVR1)
plt.plot(x,y)

print("The R^2 for the test data in this SVR Regression is: ", model_SVR1.score(X_test,y_test))

print("The R^2 for the training data in this SVR Regression is: ", model_SVR1.score(X_train,y_train))

RMSE = np.sqrt(mean_squared_error(y_test,y_SVR1))

print("The Root Mean Squared Error for this is: ",RMSE )


# We can see that the SVR model didn't fit the data too well. We'll check the Decision Tree Regressor.

# In[ ]:


from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15,10))
x = np.linspace(0,80,1000)
y=x
plt.scatter(y_test, y_DTR1)
plt.plot(x,y)

print("The R^2 for the test data in this Decision Tree Regression is: ", model_DTR1.score(X_test,y_test))

print("The R^2 for the training data in this Decision Tree Regression is: ", model_DTR1.score(X_train,y_train))

RMSE = np.sqrt(mean_squared_error(y_test,y_DTR1))

print("The Root Mean Squared Error for this is: ",RMSE )


# By increasing the value of max_depth the results on the traing are better, but this is not necessarily true for predictions, because the method will overfit on data. After checking smaller values of max_depth, like 5 and 6, the data seemed more like classification, not well distributed as it is now. For higher values like 10 and 11, the data fitted too well, and I believe with higher values overfit will begin to be a problem. Currently the data is overfit to the training data

# In[ ]:


from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15,10))
x = np.linspace(0,80,1000)
y=x
plt.scatter(y_test, y_RFR1)
plt.plot(x,y)

print("The R^2 for the test data in this Random Forest Regression is: ", model_RFR_1.score(X_test,y_test))

print("The R^2 for the training data in this Random Forest Regression is: ", model_RFR_1.score(X_train,y_train))

RMSE = np.sqrt(mean_squared_error(y_test,y_RFR1))

print("The Root Mean Squared Error for this is: ",RMSE )


# As the data for the Tree Regressor may overfit because of it's depth, we now added a Random Forest Regressor, which seems to perform good on both training and test dataset!

# In[ ]:




