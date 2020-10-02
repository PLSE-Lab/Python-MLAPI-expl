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


df = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


finalDataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[ ]:


finalDataset.head()


# In[ ]:


finalDataset['CurrYear'] = 2020


# In[ ]:


finalDataset['carAge'] = finalDataset['CurrYear'] - finalDataset['Year']


# In[ ]:


finalDataset.head()


# In[ ]:


finalDataset.drop(['Year'], axis=1, inplace = True)


# In[ ]:


finalDataset.head()


# In[ ]:


finalDataset.drop(['CurrYear'], axis =1, inplace=True)


# In[ ]:


finalDataset.head()


# In[ ]:


finalDataset = pd.get_dummies(finalDataset, drop_first=True)


# In[ ]:


finalDataset.head()


# In[ ]:


finalDataset.corr()


# In[ ]:


import seaborn as sns
sns.pairplot(finalDataset)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# plot in heatmap

corrmat = finalDataset.corr()
topCorrFeatures = corrmat.index
plt.figure(figsize = (10,7))
#plot heatmap
g = sns.heatmap(finalDataset[topCorrFeatures].corr(), annot=True, cmap="PiYG")


# In[ ]:



finalDataset.head()


# In[ ]:


x = finalDataset.iloc[:,1:]     #independent feature
y = finalDataset.iloc[:,0]      #dependent feature
x.head()


# In[ ]:


#feature importance --> to know the important feature
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


#Plot to visualise better
featImportance = pd.Series(model.feature_importances_, index=x.columns)
featImportance.plot(kind='barh')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
x_train.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forestRegressor = RandomForestRegressor()


# In[ ]:



import numpy as np
#number of tree
n_estimators = [int(x) for x in np.linspace(start =100, stop = 1500, num=15)]
print(n_estimators)


# In[ ]:


#number of feature to consider at every split
maxFeatures = ['auto', 'sqrt']
#maximum levels
maxDepth = [int(x) for x in np.linspace(5,30, num =6)]

#minimum number of sample to split a node
minSampleSplit = [2,5,10,15,100]

#minumum sample at each leaf node
minSampleLeaf = [1,2,5,10]


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
#randomiszedSearchCv is pretty much fast than GridSearchCV


# In[ ]:


#create random Grid
rndmGrid = {'n_estimators' : n_estimators,
           'max_features' : maxFeatures,
           'max_depth': maxDepth,
           'min_samples_split' : minSampleSplit,
           'min_samples_leaf': minSampleLeaf}
print(rndmGrid)


# In[ ]:


#Create base model
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf,
                              param_distributions=rndmGrid,
                              scoring='neg_mean_squared_error',
                              n_iter= 10,
                              cv=5, verbose=2,
                              random_state=42,
                              n_jobs=1)


# In[ ]:


rf_random


# In[ ]:


rf_random.fit(x_train, y_train)


# In[ ]:


predictions = rf_random.predict(x_test)
predictions


# In[ ]:


sns.distplot(y_test-predictions)


# In[ ]:


plt.scatter(y_test, predictions)


# In[ ]:


import pickle
file = open('random_forestRegression_model.pkl', 'wb')


pickle.dump(rf_random, file)


# In[ ]:





# In[ ]:




