#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Here we are going to predict the height and weight of the person. We are taking height and gender is the input attributes and weight going to output attribute.
# 
# Table Of Content:
# 
# *   **1. Feature Enineering/Data Pre Processing**
#   
#  *      1(a). Import Dataset
#  *      1(b). Describing Descriptive Statistics
#  *      1(c). Visualising Descriptive Statistics
#  *      1(d). Checking Null or Empty Values (Data Cleaning)
#  *      1(e). Label Encoder/One Hot Encoder
#  *      1(f). Handle Outliers
#  *      1(g). Feature Split
#  *      1(h). Resample Evaluate performance model
#      
# *    **2. Modeling**
#    
#  *      2(a). Regression Models Without Feature Scale.
#  *      2(b). Regression Models With Feature Scale.
#  *      2(c). Regularisation Tuning For Top 2 Regression Algorithms.
#  *      2(d). Ensemble and Boosting Regression Algorithms With Feature Scale.
#  *      2(e). Regularisation Tuning For Top 2 Ensemble and Boosting Regression Algorithms.
#  *      2(f). Compare All 4 Tunned Algorithms And Selecting The Best Algorithm
#  *      2(g). Fit and Predict The Best Algorithm.
#  *      2(h). Accuracy Of An Algorithm.

# # 1. Feature Enineering/Data Pre Processing
# 
# # 1(a). Import Dataset

# In[ ]:


# Importing the Dataset
import pandas as pd
dataset = pd.read_csv("../input/weight-height/weight-height.csv")


# # 1(b). Describing Descriptive Statistics

# In[ ]:


# Displaying the head and tail of the dataset
dataset.head()


# In[ ]:


dataset.tail()


# In[ ]:


# Displaying the shape and datatype for each attribute
print(dataset.shape)
dataset.dtypes


# In[ ]:


# Displaying the describe of each attribute
dataset.describe()


# As per above min value is clear that height and weight not starting from zero. it is starting around 50+...

# # 1(c). Visualising Descriptive Statistics

# In[ ]:


# Histogram Visualisation For Height Attribute with distribution plot

import seaborn as sb
sb.distplot(dataset['Height'])


# Wowww Perfect we got good normal distribution...Now we are going to check weight attribute.

# In[ ]:


sb.distplot(dataset['Weight'])


# As per above plot weight is not a normal distribution but it is almost simlar to normal..
# 
# If we want best prediction we need to apply feature scale then only we will get better accuracy.

# In[ ]:


# Checking the correlation between input and output attributes.
corr_value=dataset.corr()
sb.heatmap(corr_value,square=True)


# # 1(d). Checking Null or Empty Values (Data Cleaning)
# 
# Checking the null or empty values and applying data cleaning to our dataset.

# In[ ]:


# Displaying the Null or empty values 
dataset.info()


# In[ ]:


# Displaying the Null or empty values sum
dataset.isna().sum()


# We dont have any missing values so now safely we can go ahead...

# # 1(e). Label Encoder/One Hot Encoder
# 
# Encoding the categorical value into numerical value.

# In[ ]:


# encoding gender column
dataset['Gender'].unique()


# In[ ]:


dataset['Gender']=dataset['Gender'].map({'Male':0,'Female':1})
dataset['Gender'].unique()


# In[ ]:


# Displaying first 5 rows
dataset.head()


# # 1(f). Handle Outliers

# In[ ]:


# Checking the outliers with each input attribute to output attribute.

plt.plot(dataset['Gender'],dataset['Weight'])
plt.title("Checking Outliers")
plt.xlabel("Gender")
plt.ylabel('Weight')
plt.show()


# We dont have any outliers as per above plot

# In[ ]:


# Checking the outliers with Height input attribute and Weight output attribute.

plt.plot(dataset['Height'],dataset['Weight'])
plt.title("Checking Outliers")
plt.xlabel("Height")
plt.ylabel('Weight')
plt.show()


# So finally we dont have any outliers in our dataset, So now we can safely go ahead.

# # 1(g). Feature Split
# 
# Splitting the input and output attributes

# In[ ]:


y=dataset['Weight'].values
x=dataset.drop(['Weight'],axis=1)


# # 1(h). Resample Evaluate performance model
# 
# Now we are going to split the input and output attribute as training and test set to evaluate model performance..

# In[ ]:


# Splitting dataset into train and test split.

train_size=0.80
test_size=0.20
seed=5
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)


# # 2. Modeling
# 
# #  2(a). Regression Models Without Feature Scale.

# In[ ]:


# Spot Checking and Comparing Algorithms Without Feature Scale
models=[]
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
models.append(('linear_reg',LinearRegression()))
models.append(('knn',KNeighborsRegressor()))
models.append(('SVR',SVR()))
models.append(("decision_tree",DecisionTreeRegressor()))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
error='neg_mean_squared_error'
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()


# We got accuracy very highly now we are going to applying feature scale to same code and lets check how well it performed.
# 
# # 2(b). Regression Models With Feature Scale.

# In[ ]:


# Create Pipeline with Standardization Scale and models
# Standardize the dataset
from sklearn.pipeline import Pipeline
from sklearn. preprocessing import MinMaxScaler
pipelines=[]
pipelines.append(('scaler_lg',Pipeline([('scaler',MinMaxScaler()),('lg',LinearRegression())])))
pipelines.append(('scale_KNN',Pipeline([('scaler',MinMaxScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('scale_SVR',Pipeline([('scaler',MinMaxScaler()),('SVR',SVR())])))
pipelines.append(('scale_decision',Pipeline([('scaler',MinMaxScaler()),('decision',DecisionTreeRegressor())])))

# Evaluate Pipelines
predictions=[]
names=[]
for name, model in pipelines:
    fold=KFold(n_splits=10,random_state=5)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg='%s : %f (%f)'%(name,result.mean(),result.std())
    print(msg)
    
#Visualize the compared algorithms
fig=plt.figure()
fig.suptitle("Algorithms Comparisions")
plt.boxplot(predictions)
plt.show()


# We didn't get good accuracy because we dont have complex values we have only gender and height as input attributes.
# 
# Linear Regression Accuracy : -100.963799 (3.484180)
# 
# KNN Regressor accuracy: : -121.553066 (4.666317)
# 
# SVR accuracy :-104.947244 (3.553198)
# 
# Decision Tree : : -200.952203 (7.381938)
# 
# 
# # 2(c). Regularisation Tuning For Top 2 Regression Algorithms.
# 
# Top 2 algorithms Linear regression and SVR so now we are applying tuning to this algorithms.
# 

# In[ ]:


# SVR Tuning
import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=MinMaxScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
kernel=['linear','poly','rbf','sigmoid']
c=[0.2,0.4,0.6,0.8,1.0]
param_grid=dict(C=c,kernel=kernel)
model=SVR()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


# Linear Regression Algorithm tuning


import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=MinMaxScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
param_grid=dict()
model=LinearRegression()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# # 2(d). Ensemble and Boosting Regression Algorithms With Feature Scale.
# 
# Ensemble and boosting algorithms

# In[ ]:


# Ensemble and Boosting algorithm to improve performance


# Boosting methods
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Ensemble Bagging methods
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
ensembles=[]
ensembles.append(('scaledAB',Pipeline([('scale',MinMaxScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('scaledGBR',Pipeline([('scale',MinMaxScaler()),('GBR',GradientBoostingRegressor())])))
ensembles.append(('scaledRF',Pipeline([('scale',MinMaxScaler()),('rf',RandomForestRegressor(n_estimators=10))])))
ensembles.append(('scaledETR',Pipeline([('scale',MinMaxScaler()),('ETR',ExtraTreesRegressor(n_estimators=10))])))
ensembles.append(('scaledRFR',Pipeline([('scale',MinMaxScaler()),('RFR',RandomForestRegressor(n_estimators=10))])))
# Evaluate each Ensemble Techinique
results=[]
names=[]
for name,model in ensembles:
    fold=KFold(n_splits=10,random_state=5)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    results.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    
# Visualizing the compared Ensemble Algorithms
fig=plt.figure()
fig.suptitle('Ensemble Compared Algorithms')
plt.boxplot(results)
plt.show()


# Ada Boost Regression Algorithm : -110.605590 (5.951022)
# 
# Gradient Boosting Regression Algorithm : -102.850162 (3.451911)
# 
# We are going to apply tuning to this algorithms
# 
# # 2(e). Regularisation Tuning For Top 2 Ensemble and Boosting Regression Algorithms.

# In[ ]:


# GradientBoostingRegressor Tuning

import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=MinMaxScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
learning_rate=[0.1,0.2,0.3,0.4,0.5]
n_estimators=[5,10,15,20,25,30,40,50,100,200]
param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)
model=GradientBoostingRegressor()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


# AdaBoostRegressor Tuning

import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=MinMaxScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
learning_rate=[0.1,0.2,0.3,0.4,0.5]
n_estimators=[5,10,15,20,25,30,40,50,100,200]
param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)
model=AdaBoostRegressor()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# After Applying tuning to those algorithm we are accuracy like wise
# 
# GradientBoostingRegressor Tuning :-102.648845 using {'learning_rate': 0.1, 'n_estimators': 50} 
# 
# AdaBoostRegressor Tuning : -109.112158 using {'learning_rate': 0.5, 'n_estimators': 200} 

# # 2(f). Compare All 4 Tunned Algorithms And Selecting The Best Algorithm
# 
# 1. SVR Tuning -104.664036 using {'C': 1.0, 'kernel': 'linear'} 
# 2. Linear Regression algorithm -100.963799 using {} 
# 3. GradientBoostingRegressor Tuning :-102.648845 using {'learning_rate': 0.1, 'n_estimators': 50} 
# 4. AdaBoostRegressor Tuning : -109.112158 using {'learning_rate': 0.5, 'n_estimators': 200} 

# # 2(g). Fit and Predict The Best Algorithm.

# In[ ]:


# Finalize Model
# we will finalize the gradient boosting regression algorithm and evaluate the model for house price predictions.

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
scaler=MinMaxScaler().fit(x_train)
scaler_x=scaler.transform(x_train)
model=GradientBoostingRegressor(random_state=5,n_estimators=50,learning_rate=0.1)
model.fit(scaler_x,y_train)

#Transform the validation test set data
scaledx_test=scaler.transform(x_test)
y_pred=model.predict(scaledx_test)


# # 2(h). Accuracy Of An Algorithm.

# In[ ]:


# Accuracy of algorithm
from math import sqrt
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("rmse",rmse)
r2=r2_score(y_test,y_pred)
print("mse",mse)
print("r2_score",r2)


# Perfect we got root mean square error around 10 we done..
# 
# If any questions please let me know...
