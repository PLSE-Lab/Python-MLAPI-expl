#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Predicting the category of crimes in San Francisco
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas.tools.rplot as rplot
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Load the training data and test
dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")
#Resolution eliminate too many null data and be very significant for our study
#dftrain[dftrain.Resolution =='NONE']
#dftrain.Resolution.value_counts()


# In[ ]:


#Exploration data
dftrain.info()


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.shape


# In[ ]:


dftrain.Category.value_counts()
#category = dftrain.pop("Category")
#category.describe()


# In[ ]:


dftrain.Category.value_counts().plot(kind = 'bar')


# In[ ]:


#Number of crimes per district
dftrain.PdDistrict.value_counts()


# In[ ]:


#Total number of suicides in a particular district
dftrain[(dftrain.Category =='SUICIDE') & (dftrain.PdDistrict =='SOUTHERN')].PdDistrict.value_counts()


# In[ ]:


fig, axs = plt.subplots(1,2)
dftrain[(dftrain.Category =='SUICIDE') ].PdDistrict.value_counts().plot(kind ='barh', ax=axs[0], title = 'Suicides')
dftrain[(dftrain.Category =='FAMILY OFFENSES')].PdDistrict.value_counts().plot(kind='bar', ax=axs[1], title = 'Family offenses', color = 'g')


# In[ ]:


dftrain.PdDistrict.value_counts().plot(kind = 'barh',title = 'Crimes by districts')


# In[ ]:


fig, axs = plt.subplots(1,2)
dftrain[dftrain.Category == 'DRUG/NARCOTIC'].PdDistrict.value_counts().plot(kind = 'bar', ax=axs[0], title = 'Drugs')
dftrain[dftrain.Category == 'ASSAULT'].PdDistrict.value_counts().plot(kind = 'bar', ax=axs[1],color = 'g', title = 'Assault')


# In[ ]:


#Specific category of crime district
dftrain[dftrain.PdDistrict == 'TENDERLOIN'].Category.value_counts()


# In[ ]:


dftrain[dftrain.PdDistrict == 'TENDERLOIN'].Category.value_counts().plot(kind = 'bar', title = 'Category of crimes in tenderloin')


# In[ ]:


dftrain[dftrain.Category == 'DRUG/NARCOTIC'].PdDistrict.value_counts().plot(kind = 'barh', title = 'Drugs in districts')

