#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import re

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from math import sqrt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


d=pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')


# In[ ]:


d.head()


# In[ ]:


d.describe().T


# In[ ]:


d.info()


# In[ ]:


d.isnull().sum()


# In[ ]:


d['society'].shape


# In[ ]:


d['size'].unique()


# In[ ]:


d.corr()


# In[ ]:


sns.pairplot(d)


# In[ ]:


sns.distplot(d['price'])


# In[ ]:


d.select_dtypes(exclude=['object']).describe()


# In[ ]:


corr=d.corr()


# In[ ]:


sns.heatmap(corr)


# In[ ]:


from collections import Counter
Counter(d['total_sqft'])


# In[ ]:


d.shape


# In[ ]:


#preprocessing the total sqft cols as it has vivid entries
def preprocess_total_sqft(my_list):
    if len(my_list) == 1:
        
        try:
            return float(my_list[0])
        except:
            strings = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']
            split_list = re.split('(\d*.*\d)', my_list[0])[1:]
            area = float(split_list[0])
            type_of_area = split_list[1]
            
            if type_of_area == 'Sq. Meter':
                area_in_sqft = area * 10.7639
            elif type_of_area == 'Sq. Yards':
                area_in_sqft = area * 9.0
            elif type_of_area == 'Perch':
                area_in_sqft = area * 272.25
            elif type_of_area == 'Acres':
                area_in_sqft = area * 43560.0
            elif type_of_area == 'Cents':
                area_in_sqft = area * 435.61545
            elif type_of_area == 'Guntha':
                area_in_sqft = area * 1089.0
            elif type_of_area == 'Grounds':
                area_in_sqft = area * 2400.0
            return float(area_in_sqft)
        
    else:
        return (float(my_list[0]) + float(my_list[1]))/2.0


# In[ ]:


d['total_sqft'] = d.total_sqft.str.split('-').apply(preprocess_total_sqft)


# In[ ]:


#converting the categorical to numerical data - area_type
d.area_type.value_counts()


# In[ ]:


replace_area_type = {'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3}
d['area_type'] = d.area_type.map(replace_area_type)


# In[ ]:


#converting the categorical to numerical data - availabilty
d.availability.value_counts()


# In[ ]:


def replace_availabilty(my_string):
    if my_string == 'Ready To Move':
        return 0
    elif my_string == 'Immediate Possession':
        return 1
    else:
        return 2


# In[ ]:


d['availability'] = d.availability.apply(replace_availabilty)


# In[ ]:


#converting NaN in location
d['location'].isnull().sum()


# In[ ]:


d['location'] = d['location'].fillna('No Location')


# In[ ]:


#converting the categorical to numerical data - size
Counter(d['size'])


# In[ ]:


le = LabelEncoder()
le.fit(d['size'].astype('str').append(d['size'].astype('str')))
d['size'] = le.transform(d['size'].astype('str'))


# In[ ]:


d.head()


# In[ ]:


#converting the NaNs to other - society
d['society'] = d['society'].fillna('Other')


# In[ ]:


le.fit(d['society'].append(d['society'].fillna('Other')))
d['society'] = le.transform(d['society'])


# In[ ]:


#converting the categorical to numerical data - location
Counter(d['location'])


# In[ ]:


le.fit(d['location'].append(d['location'].fillna('other')))
d['location']=le.transform(d['location'])


# In[ ]:


#converting NaNs in bath
d['bath'].isna().sum()


# In[ ]:


#missing values are filled by grouping the rows based on location and taking the mean of the column 'bath' in that location.
col_bath=d.groupby('location')['bath'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


col_bath.isna().sum()


# In[ ]:


col_bath[~col_bath.notnull()]


# In[ ]:


#col 1775 has nan even after transformation
col_bath = col_bath.fillna(col_bath.mean())


# In[ ]:


#finally its resolved
col_bath.isnull().sum()


# In[ ]:


d['bath']=col_bath


# In[ ]:


#missing values are filled by grouping the rows based on location and taking the mean of the column 'balcony' in that location.
d['balcony'].isnull().sum()


# In[ ]:


col_balcony=d.groupby('location')['balcony'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


col_balcony.isna().sum()


# In[ ]:


#col 45 has nan even after transformation
col_balcony = col_balcony.fillna(col_balcony.mean())


# In[ ]:


col_balcony.isnull().sum()


# In[ ]:


d['balcony']=col_balcony


# In[ ]:


d.head()


# In[ ]:


#preprocessing for building ML models
x=d.drop('price',axis=1)
y=d['price']


# In[ ]:


d.info()


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)


# In[ ]:


#Linear Regression
lr=LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


lpred=lr.predict(x_test)
print(lpred)


# In[ ]:


lrrmse=np.sqrt(np.mean((y_test-lpred)**2))
lrrmse


# In[ ]:


#Decision Tree
dt=DecisionTreeRegressor()


# In[ ]:


dt.fit(x_train,y_train)


# In[ ]:


dtpred=dt.predict(x_test)
print(dtpred)


# In[ ]:


dtrmse=np.sqrt(np.mean((y_test-dtpred)**2))
dtrmse


# In[ ]:


#Random Forest
rf=RandomForestRegressor()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


rfpred=rf.predict(x_test)
print(rfpred)


# In[ ]:


rfrmse=np.sqrt(np.mean((y_test-rfpred)**2))
rfrmse


# In[ ]:




