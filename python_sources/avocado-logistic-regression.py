#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


avocado = pd.read_csv('../input/avocado.csv')


# In[ ]:


avocado.shape


# In[ ]:


avocado.head()


# In[ ]:


avocado.columns


# In[ ]:


avocado.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


avocado.describe()


# In[ ]:


avocado.info()


# In[ ]:


def low_cardinality_cols(dataframe):
    low_card_cols = [cname for cname in dataframe.columns if (dataframe[cname].nunique()<55 and dataframe[cname].dtype=='object')]
    return (low_card_cols)
                                                              


# In[ ]:


low_cardinality_cols(avocado)


# In[ ]:


def cols_with_missing_values(dataframe):
    cols_missing_data = [cname for cname in dataframe.columns 
                        if dataframe[cname].isnull().any()]
    return (cols_missing_data)


# In[ ]:


cols_with_missing_values(avocado)


# In[ ]:


avocado['region'].unique() # we have a total column, we can delete those records


# In[ ]:


avocado[ avocado['region'] == 'TotalUS'].head()


# In[ ]:


avocado = avocado[ avocado['region'] != 'TotalUS']


# In[ ]:


# checking if the records are removed


# In[ ]:


avocado[ avocado['region'] == 'TotalUS']


# In[ ]:


# adding new columns
avocado['small Hass'] = avocado['4046']
avocado['large Hass'] = avocado['4225']
avocado['extra large Hass'] = avocado['4770']


# In[ ]:


avocado.columns


# In[ ]:


# removing the number columns
avocado.drop(['4046','4225','4770'],axis=1,inplace=True)


# In[ ]:


avocado.columns


# In[ ]:


# get the values for the region column
region_dummies =   pd.get_dummies(data=avocado['region'])


# In[ ]:


# similar for the year column
year_dummies =   pd.get_dummies(data=avocado['year'])


# In[ ]:


# join the dataframes on index
avocado =   avocado.join(other=region_dummies,on=region_dummies.index,how='inner')


# In[ ]:


avocado.drop('key_0',axis=1,inplace=True)


# In[ ]:


avocado  = avocado.join(other=year_dummies,on=year_dummies.index,how='inner')


# In[ ]:


# check the new shape
avocado.shape


# In[ ]:


# check the new columns
avocado.columns


# In[ ]:


# create the feature
X = avocado.drop(['key_0','Total Volume','Total Bags','Date', 'year','type','region'],axis=1)
y = avocado['type']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


predictions = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:




