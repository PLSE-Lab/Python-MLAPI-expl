#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
dataset.head()


# # Analysing the Dataset

# In[ ]:


dataset.shape


# In[ ]:


print(dataset['Fuel_Type'].unique())
print(dataset['Seller_Type'].unique())
print(dataset['Transmission'].unique())
print(dataset['Owner'].unique())


# # Checking NaN value present in out dataset or not

# In[ ]:


#check missing null values
dataset.isnull().sum()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.columns


# In[ ]:


final_dataset=dataset[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset['Current_Year']=2020


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset['no_of_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset.corr()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.pairplot(final_dataset)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
corrmat = final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(15,15))
# plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


final_dataset.head()


# In[ ]:


# X is independent features and Y is dependent features
X=final_dataset.drop('Selling_Price',axis=1)
Y = final_dataset['Selling_Price']


# # checking which feature is important by the help of ExtraTreeRegressor algo

# In[ ]:


## feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,Y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)


# In[ ]:


X_train.shape


# # Now, Its time to build our model by the help of Random Forest Algorithm

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# Hyperparameter tuning by the help of Randomized searchcv

# In[ ]:


#Hyperparameter tuning in Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


# use the random grid to search for best heperparameters
rf = RandomForestRegressor()


# In[ ]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


rf_random.fit(X_train,Y_train)


# In[ ]:


predictions = rf_random.predict(X_test)


# In[ ]:


predictions


# In[ ]:


sns.distplot(Y_test-predictions)


# In[ ]:


plt.scatter(Y_test,predictions)


# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

cv = ShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
cross_val_score(RandomForestRegressor(),X,Y,cv=cv)


# # <<<<<<<<<<<<<<<<-------If you like this approach!!!!! Pleaseeee Upvote------->>>>>>>>>>>>>>>>>>>

# In[ ]:




