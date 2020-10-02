#!/usr/bin/env python
# coding: utf-8

# # Reading Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


trainDF=pd.read_csv('../input/train.csv')


# In[ ]:


trainDF.head(1)


# # Extracting information from **belongs_to_collection** Column

# In[ ]:


trainDF.belongs_to_collection=trainDF.belongs_to_collection.fillna('[{}]')

belongs_to_collectionList=[]
for index,row in trainDF.belongs_to_collection.iteritems():
    belongs_to_collectionStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            belongs_to_collectionStr=belongs_to_collectionStr+';'+dic['name'] 
    belongs_to_collectionStr=belongs_to_collectionStr.strip(';') # trim leading ;
    belongs_to_collectionList.append(belongs_to_collectionStr)
    
tempDF=pd.DataFrame(belongs_to_collectionList,columns=['belongs_to_collection'])
trainDF.belongs_to_collection=tempDF['belongs_to_collection']


# In[ ]:


trainDF.belongs_to_collection.value_counts().head()


# # Extracting information from **Genres** column

# In[ ]:


trainDF.genres=trainDF.genres.fillna('[{}]')

genresList=[]
for index,row in trainDF.genres.iteritems():
    genresStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            genresStr=genresStr+';'+dic['name'] 
    genresStr=genresStr.strip(';') # trim leading ;
    genresList.append(genresStr)
    
tempDF=pd.DataFrame(genresList,columns=['genres'])
trainDF.genres=tempDF['genres']


# In[ ]:


trainDF.genres.head(10) # semicolon seperated values


#  # Extracting information from **production_companies** column

# In[ ]:


trainDF.production_companies[0] # Actual value


# In[ ]:


trainDF.production_companies=trainDF.production_companies.fillna('[{}]')

production_companiesList=[]
for index,row in trainDF.production_companies.iteritems():
    production_companiesStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            production_companiesStr=production_companiesStr+';'+dic['name'] 
    production_companiesStr=production_companiesStr.strip(';') # trim leading ;
    production_companiesList.append(production_companiesStr)
    
tempDF=pd.DataFrame(production_companiesList,columns=['production_companies'])
trainDF.production_companies=tempDF['production_companies']


# In[ ]:


trainDF.production_companies.head() # Extracted Value


# #  Extracting information from  **production_countries** column

# In[ ]:


trainDF.production_countries[56] # actual values


# In[ ]:


trainDF.production_countries=trainDF.production_countries.fillna('[{}]')

production_countriesList=[]
for index,row in trainDF.production_countries.iteritems():
    production_countriesStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            production_countriesStr=production_countriesStr+';'+dic['name'] 
    production_countriesStr=production_countriesStr.strip(';') # trim leading ;
    production_countriesList.append(production_countriesStr)
    
tempDF=pd.DataFrame(production_countriesList,columns=['production_countries'])
trainDF.production_countries=tempDF['production_countries']


# In[ ]:


trainDF.production_countries.head()


# #  Extracting information from  **release_date** column

# In[ ]:


dateSplit=trainDF.release_date.str.extract('([0-9]+)/([0-9]+)/([0-9]+)')
dateSplit.columns=['ReleaseMonth','ReleaseDate','ReleaseYear']


# if year is between 00 to 19.. Then it could be between 2000 to 2019
# 
# Else year is between 20 to 99 then it could be between 1920 to 1999

# In[ ]:


dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']='19'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']

dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']='20'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']


# In[ ]:


trainDF=pd.concat([trainDF,dateSplit],axis=1)


# In[ ]:


trainDF.head()


# In[ ]:




