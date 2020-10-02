#!/usr/bin/env python
# coding: utf-8

# # Indian Trade Data regression
# By : Hesham Asem
# 
# ____
# 
# so we have here 2 data files , for import & export for a specific indian company , between 2010 & 2018
# 
# lets use them to build a regression model , so we can expect the value using other features . 
# 
# data file : 
# https://www.kaggle.com/lakshyaag/india-trade-data#2018-2010_export.csv
# 
# _____
# 
# first to import the libraries 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid")


# then to read the data

# In[ ]:


idata = pd.read_csv('../input/india-trade-data/2018-2010_import.csv') 
edata = pd.read_csv('../input/india-trade-data/2018-2010_export.csv') 


# since we have 2 files , we'll need to handle them separately 

# ____
# 
# # Needed Functions . 
# 
# as usual , we'll build here important funcstion which will be helpful in data processing . 
# 
# and you can see that some functions have the arg (data) , since we'll have to repeat our steps on idata & edata

# In[ ]:


def max_counts( feature , number, data, return_rest = False ) : 
    counts = data[feature].value_counts()
    values_list = list(counts[:number].values)
    rest_value =  sum(counts.values) - sum (values_list)
    index_list = list(counts[:number].index)
    
    if return_rest : 
        values_list.append(rest_value )
        index_list.append('rest items')
    
    result = pd.Series(values_list, index=index_list)
    if len(data[feature]) <= number : 
        result = None
    return result


# In[ ]:


def series_pie(series) : 
    plt.pie(series.values,labels=list(series.index),autopct ='%1.2f%%',labeldistance = 1.1,explode = [0.05 for i in range(len(series.values))] )
    plt.show()


# In[ ]:


def series_bar(series) : 
    plt.bar(list(series.index),series.values )
    plt.show()


# In[ ]:


def make_label_encoder(original_feature , new_feature,data) : 
    enc  = LabelEncoder()
    enc.fit(data[original_feature])
    data[new_feature] = enc.transform(data[original_feature])
    data.drop([original_feature],axis=1, inplace=True)


# _____
# 
# # Import Data
# 
# let's have a look to import data

# In[ ]:


idata.head()


# few features , what is the dimension ? 

# In[ ]:


idata.shape


# so we'll need to start in data processing

# _____
# 
# # Data Processing
# 

# let's start with HSCode feature , how many unique values in it

# In[ ]:


len(idata['HSCode'].unique())


# let's have a look to top values for it

# In[ ]:


idata['HSCode'].value_counts()[:10]


# ______
# 
# so it looks that this feature is very correlated to the other feature Commodity . 
# 
# so how many are them 

# In[ ]:


len(idata['HSCode'].unique())


# & how many unqiue values for Commodity ? 

# In[ ]:


len(idata['Commodity'].unique())


# cool , lets get the Commodity data , for only specific HSCode value

# In[ ]:


new_data = idata[idata['HSCode']==5]['Commodity']
new_data


# what are the unique values for it ? 

# In[ ]:


new_data.unique()


# and to be more sure , we'll mak for block for HSCode values , to get unique values for Commodity data 

# In[ ]:


for x in range(idata['HSCode'].max()): 
    new_data = idata[idata['HSCode']==x]['Commodity']
    print(len(new_data.unique()))
    print('---------------')


# so it's clear that those two features are very correlated , so we can drop one of them , let's drop commodity

# In[ ]:


idata.drop(['Commodity'],axis=1, inplace=True)


# _____
# 
# how data looks like now ? 

# In[ ]:


idata.head()


# let's work with country feature

# In[ ]:


idata['country'].unique()


# how many country we have here 

# In[ ]:


len(idata['country'].unique())


# what are the most repeated countries 

# In[ ]:


counts = idata['country'].value_counts()


# In[ ]:


counts[:10]


# so we'll need to get the max countries to graph them , let's say we'll get max 8 countries

# In[ ]:


max_countries = max_counts('country',8,idata)


# In[ ]:


max_countries


# ____
# 
# let's pie chart it

# In[ ]:


series_pie(max_countries)


# also we can bar them 

# In[ ]:


series_bar(max_countries)


# so we can apply labelencoder to them

# In[ ]:


make_label_encoder('country','country_code',idata)


# _____
# 
# now how data looks like 

# In[ ]:


idata.head()


# we ned to check if we have any nulls

# In[ ]:


idata.info()


# so value feature got more than 14K null value , & we cannot fill them with mean or median , since this is the output , so it will mislead the training , we'll have to drop all these rows with any value less than or equal 0

# In[ ]:


new_data  = idata[idata['value'] > 0]


# now how fata looks like 

# In[ ]:


new_data.shape


# In[ ]:


new_data.info()


# cool , now we ready for training
# 
# ________
# 
# 

# 
# _____
# 
# 
# 
# # Building the Model
# 
# 
# let's first define X & y

# In[ ]:


X = new_data.drop(['value'], axis=1, inplace=False)
y = new_data['value']


# what are the dimensions ? 

# In[ ]:


X.shape


# In[ ]:


y.shape


# split it

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# _____
# 
# let's  use Gradient Boosting Regressor , with 1000 estimators & 10 depth & 0.1 learning rate

# In[ ]:


GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=10,learning_rate = 0.1 ,random_state=33)
GBRModel.fit(X_train, y_train)


# now how is accuracy ? 

# In[ ]:


print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))


# great work , & even we avoided OF , since test accuracy is high , now let's make prediction 

# In[ ]:


y_pred = GBRModel.predict(X_test)
print('Predicted Value for GBRModel is : ' , y_pred[:10])


# ______
# 
# ______
# 
# # Export Data
# 
# 
# now , we'll have to repeat almost all steps to export data , instead of import data

# In[ ]:


len(edata['HSCode'].unique())


# In[ ]:


edata['HSCode'].value_counts()[:10]


# In[ ]:


len(edata['HSCode'].value_counts())


# In[ ]:


edata.drop(['Commodity'],axis=1, inplace=True)


# In[ ]:


edata.head()


# In[ ]:


edata['country'].unique()


# In[ ]:


len(edata['country'].unique())


# In[ ]:


counts = edata['country'].value_counts()


# In[ ]:


counts[:10]


# In[ ]:


list(counts[:10].index)


# In[ ]:


max_countries = max_counts('country',8,edata)
max_countries


# In[ ]:


series_pie(max_countries)


# In[ ]:


series_bar(max_countries)


# In[ ]:


make_label_encoder('country','country_code',edata)


# In[ ]:


edata.head()


# In[ ]:


edata.info()


# In[ ]:


new_data  = edata[edata['value'] > 0]


# In[ ]:


new_data.shape


# In[ ]:


new_data.info()


# In[ ]:


X = new_data.drop(['value'], axis=1, inplace=False)
y = new_data['value']


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# In[ ]:


GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=10,learning_rate = 0.1 ,random_state=33)
GBRModel.fit(X_train, y_train)


# In[ ]:


print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))


# In[ ]:


y_pred = GBRModel.predict(X_test)
print('Predicted Value for GBRModel is : ' , y_pred[:10])


# In[ ]:




