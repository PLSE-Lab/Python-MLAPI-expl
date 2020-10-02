#!/usr/bin/env python
# coding: utf-8

# ## Welcome to exploration and analysis of the auto mpg data set.
# Welcome to this ipython notebook created for exploration and analysis of the Auto- MPG data-set from UCI Machine Learning Library. The data-set is fairly standard on kaggle but can be accessed separately from the UCI Machine Learning Repository along with many other interesting data-sets. Check http://archive.ics.uci.edu/ml/index.php for more.
# 
# This notebook aims primarily to demonstrate use of pandas and seaborn for exploration and visualization of the data-set along with use of scikit learn library to build regression models to predict the Miles Per Gallon(MPG) using the factors provided in the data-set

# ### About DataSet

# This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original". 
# 
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#These are libraries for visualization 
import seaborn as sns
import matplotlib.pyplot as plt 

#Setting instances
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### Importing Data file

# In[ ]:


data = pd.read_csv('../input/auto-mpg.csv',index_col='car name', )


# In[ ]:


data.head()


# #### Shape of data

# In[ ]:


data.shape 


# #### Attributes data types

# In[ ]:


data.dtypes


# #### Looking for null values

# In[ ]:


data.isnull().sum()


# There is no null values in the data set....

# #### Casting horsepower to flot

# In[ ]:


data.shape


# #### Duplicates Values

# In[ ]:


data.duplicated().sum()


# #### Unique values in different Columns

# In[ ]:


data.cylinders.unique()


# cylinder column has **5** unique values

# In[ ]:


data['model year'].unique()


# the model of autos are between 70 and 82

# In[ ]:


data.origin.unique()


# The origin has three different values i.e. **1** for **USA**, **2** for **Europe** , **3** for **Japan**

# In[ ]:


data.horsepower.unique()


# there is **?** which represents missing value in horsepower column...

# In[ ]:


data = data[data.horsepower != '?']


# In[ ]:


#Validting th changes in horsepower column
print('?' in data.horsepower)


# In[ ]:


#Shape of data  after changes...
data.shape


# In[ ]:


data.dtypes


# #### Casting Hosrepower as float

# In[ ]:


data.horsepower = data.horsepower.astype('float')
data.dtypes


# ### Features Engineering

# First, i would like to add **Power to Weight** ratios column, because it has great impact in general on mileage.

# In[ ]:



data['PW_ratio']= (data.horsepower / data.weight)


# #### Creating Displacemnt in CC

# In[ ]:


data['DispCC']=data['displacement']* 16.3871


# #### Calculating Engine displacement in liters

# In[ ]:


data['DispLitr']= data['DispCC']/1000


# #### Shape of data after feature engineering

# In[ ]:


data.shape


# #### Discriptive statistics 

# In[ ]:


data.describe()


# ## EDA
# 

# In[ ]:


sns.distplot(data['mpg']);
plt.title('MPG Distribution in Data')
plt.show()


# In[ ]:


sns.distplot(data['acceleration'], hist=True, kde=False, color='red')
plt.title('Acceleration Distribution in Data')
plt.show()


# In[ ]:


plt.figure(figsize=(12,4))
sns.boxplot(x='cylinders',y='displacement', data=data)
plt.show()


# In[ ]:


data['displacement'].value_counts().plot(kind='hist');
plt.xlabel('Displacement in Cu Inches')
plt.title('Displacement Distribution')
plt.show()


# In[ ]:


data['DispCC'].value_counts().plot(kind='hist', color='Green')
plt.xlabel('Displacement CC')
plt.title('Displacement in CC Distribution')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
data['acceleration'].value_counts().head(20).plot(kind='hist', title='Acceleraion of top 20 Cars');


# In[ ]:


plt.figure(figsize=(14,5))
sns.scatterplot(x='DispLitr', y='mpg', data=data)
plt.title('Mpg against Displacement in Litres wrt Countries')
plt.show()


# In[ ]:


sns.boxplot(x='cylinders',y='mpg',data=data);
plt.title('Relation between Cylinders and MPG')
plt.show()


# In[ ]:


sns.boxplot(x='model year',y='mpg',data=data);
plt.title('Relation between Model Year and MPG')
plt.show()


# In[ ]:


sns.scatterplot(x='PW_ratio', y='mpg', data=data)
plt.title('Power to Weight ratio comparison with miles per Gallon')
plt.show()


# In[ ]:


sns.scatterplot(x='horsepower', y='mpg', data=data);
plt.title('Relation between Cylinders, Horsepower and MPG')
plt.show()


# In[ ]:


sns.scatterplot(x='acceleration', y='mpg', data=data);
plt.title('Relation between Acceleration and MPG')
plt.show()


# In[ ]:


sns.scatterplot(x='displacement', y='mpg',size='weight', hue='weight', data=data)
plt.title('Relation between Acceleration, Weight and MPG')
plt.show()


# In[ ]:


sns.pairplot(data, height = 2.0,hue ='origin')
plt.title('Comparison between car by manufacturing Countries')
plt.show()


# As i mentioned before.... it is country wise comparison of carss... 1 for USA, 2 for Europe, 3 for Japan 

# In[ ]:


cor = data.corr()
plt.figure(figsize=(12,6))
sns.heatmap(cor, annot=True)
plt.title('Correlation in Auto-MPG Data')
plt.show()


# ### Applying Machine Learning Algorithm i.e Random Forrest in this Case

# First importing Libraries 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor


# #### Creating Dependant and Independant Variables

# In[ ]:


X = data.drop('mpg', axis=1)
Y = data[['mpg']]
print(X.shape, Y.shape)


# #### Normalizing Data

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X)
X_ = scaler.transform(X)
X = pd.DataFrame(data=X_, columns = X.columns)
X.head()


# #### Train test spliting of Data

# In[ ]:


xtrain,xtest, ytrain, ytest = train_test_split(X, Y,test_size = 0.3, random_state=30, shuffle= True)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


# #### Applying Random Forrest

# In[ ]:


rf_rgr = RandomForestRegressor(criterion='mse', max_depth=5, random_state=30)


# In[ ]:


rf_rgr.fit(xtrain, ytrain)


# In[ ]:


rf_rgr.predict(xtest)


# In[ ]:


print(ytest.head(), rf_rgr.predict(xtest)[0:5])


# In[ ]:


r2_score(ytest, rf_rgr.predict(xtest))


# In[ ]:


mean_squared_error(ytest, rf_rgr.predict(xtest))


# #### Feature Importance

# In[ ]:


features_tuple=list(zip(X.columns,rf_rgr.feature_importances_))


# In[ ]:


feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])
feature_imp=feature_imp.sort_values("Importance",ascending=False)


# In[ ]:


plt.figure(figsize=(12,4))
sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='g')
plt.xlabel("Auto MPG Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.title("Random Forest Classifier - Features Importance")


# ### Tunning Random Forest Regressor with GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid1 = {"n_estimators" : [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}

RF = RandomForestRegressor(random_state=30)
# Instantiate the GridSearchCV object: logreg_cv
RF_cv1 = GridSearchCV(RF, param_grid1, cv=5,scoring='r2',n_jobs=4)

# Fit it to the data
RF_cv1.fit(xtrain,ytrain)

#RF_cv1.cv_results_, 
RF_cv1.best_params_, RF_cv1.best_score_


# In[ ]:


param_grid2 = {"n_estimators" : [72,75,78,81,84,87,90],
           "max_depth" : [5,6,7,8,9,10,11,12,13,14,15],
           "min_samples_leaf" : [1,2,3,4]}

RF = RandomForestRegressor(random_state=30)
# Instantiate the GridSearchCV object: logreg_cv
RF_cv2 = GridSearchCV(RF, param_grid2, cv=5,scoring='r2',n_jobs=4)

# Fit it to the data
RF_cv2.fit(xtrain,ytrain)

#RF_cv2.grid_scores_, 
RF_cv2.best_params_, RF_cv2.best_score_


# ### Tunned Random Forest

# In[ ]:


RF_tuned = RF_cv2.best_estimator_


# In[ ]:


RF_tuned.fit(xtrain, ytrain)


# In[ ]:


pred = RF_tuned.predict(xtest)


# In[ ]:


print(ytest.head(), pred[0:5])


# In[ ]:


r2_score(ytest, pred)


# In[ ]:


mean_squared_error(ytest, pred)

