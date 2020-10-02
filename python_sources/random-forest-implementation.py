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


#reading the dataset
df=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")


# In[ ]:


#seeing the top 5 records
df.head()


# In[ ]:


#Checking no.of rows and columns of dataset
df.shape


# In[ ]:


#Checking the unique values of categorial variable 
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[ ]:


#checking for null values in the dataset
df.isnull().sum()


# In[ ]:


#Numerical Statistics
df.describe()


# In[ ]:


#Checking the columns of datasets
df.columns


# In[ ]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset['current_year'] = 2020


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset['no_years'] = final_dataset['current_year'] - final_dataset['Year']


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset.drop(['Year'], axis=1, inplace=True)


# In[ ]:


final_dataset.drop(['current_year'], axis=1, inplace=True)


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset=pd.get_dummies(final_dataset, drop_first=True)


# In[ ]:


final_dataset.head()


# In[ ]:


final_dataset.corr()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(final_dataset)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


corrmat = final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


model = ExtraTreesRegressor()
model.fit(X,y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[ ]:


#Hyperparameter Tuning

#number of tress in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop = 1200, num=12)]

#Number of features to consider at every split
max_features = ['auto', 'sqrt']

#maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,30,num=6)]

#min number of samples required to split a node
min_samples_split = [2,5,10,15,100]

#min number of smples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


rf = RandomForestRegressor()


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[ ]:


rf_random.fit(X_train,y_train)


# In[ ]:


predictions=rf_random.predict(X_test)


# In[ ]:


predictions


# In[ ]:


sns.distplot(y_test-predictions)


# In[ ]:


plt.scatter(y_test, predictions)


# In[ ]:




