#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Reading Datasets and Overview

# Read Data

# In[ ]:


data_iris = pd.read_csv('/kaggle/input/iris-dataset-with-outliers/Iris_with_outliers.csv')
print('Read')


# Overview, First Look

# In[ ]:


data_iris.info()


# In[ ]:


data_iris.columns


# Dropping unnecessary columns

# In[ ]:


data_iris.drop(labels=data_iris.columns[0], axis=1, inplace=True)
print("dropped")


# In[ ]:


data_iris.head()


# In[ ]:


data_iris.info()


# In[ ]:


data_iris.describe()


# In[ ]:


data_iris.groupby('Species').agg(["min","max","std","mean"])


# Checking There is NaN values or not.

# In[ ]:


data_iris.isnull().values.any()


# In[ ]:


data_iris.isna().sum()


# If there was NaN values, I could assign the average values with the sample script below.

# In[ ]:


for column in data_iris.columns[1:-1]:
    data_iris[column].fillna(value=data_iris[column].mean(), inplace=True)


# ## Visualizing Data

# In[ ]:


sns.scatterplot(data=data_iris, x="Id",y="SepalLengthCm",hue="Species")


# In[ ]:


sns.pairplot(data = data_iris, hue="Species", markers=["o","s","d"]);


# In[ ]:


sns.pairplot(data = data_iris, kind="reg", hue="Species");


# ## Outlier Detection

# 3 Sigma Methodology

# ![1_IdGgdrY_n_9_YfkaCh-dag.png](attachment:1_IdGgdrY_n_9_YfkaCh-dag.png)

# In[ ]:


data_iris.shape


# In[ ]:


for column in data_iris.columns[1:-1]:
    for specy in data_iris["Species"].unique():
        Specy_type=data_iris[data_iris["Species"]==specy]
        Selected_column=Specy_type[column]
        avg = Selected_column.mean()
        std = Selected_column.std()
        upper_lmt = avg + (3 * std) 
        lower_lmt= avg - (3 * std)
        outliers=Selected_column[((Selected_column > upper_lmt) | (Selected_column< lower_lmt))].index # picking outliers' indeces
        data_iris.drop(index=outliers, inplace=True) # dropping outliers
        print(column,specy,outliers)               


# IQR - Interquartile Range

# ![Screenshot_1.png](attachment:Screenshot_1.png)

# In[ ]:


for column in data_iris.columns[1:-1]:
    for specy in data_iris["Species"].unique():
        Specy_type = data_iris[data_iris["Species"] == specy]
        Selected_column = Specy_type[column]
        q1 = Selected_column.quantile(0.25) # for select first quartile
        q3 = Selected_column.quantile(0.75) # for select third quartile
        iqr = q3 - q1 # this is interquartile range
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr        
        outlierss = Selected_column[(Selected_column > upper_limit) | (Selected_column < lower_limit)].index # picking outliers' indeces
        print(outlierss)
        data_iris.drop(index = outlierss, inplace=True) # dropping outliers
        
        
        


# In[ ]:


data_iris.to_csv("updated_data.csv")


# ## Modelling

# Difference between One-Hot Encoding and Label Encoding
# 
# ![onehotencoding.jpg](attachment:onehotencoding.jpg)

# Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("/kaggle/working/updated_data.csv")
data.head()


# In[ ]:


data.drop(data.columns[0:2], axis=1, inplace=True)
data.head()
print("dropped columns")


# In[ ]:


data.head()


# In[ ]:


labenc= LabelEncoder()
data["Species"] = labenc.fit_transform(data["Species"]) # transforming Species column into label encoding format


# In[ ]:


data.head() # check this out


# Small controls before modelling

# In[ ]:


data.isna().sum()


# In[ ]:


data.dtypes


# Building the Model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split( data.iloc[:,0:-1] , data.iloc[:,-1] , test_size=0.2 )


# In[ ]:


import xgboost as xgb 


# In[ ]:


xgb_clsfr = xgb.XGBClassifier(objective="multiclass:softmax", num_class=3)


# In[ ]:


xgb_clsfr.fit(x_train,y_train)


# In[ ]:


predictions = xgb_clsfr.predict(x_test)
predictions


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


confusion_matrix(y_test,predictions)

