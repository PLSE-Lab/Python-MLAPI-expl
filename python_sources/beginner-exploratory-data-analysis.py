#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#loading data into variable
data = pd.read_csv('../input/MetObjects.csv')


# In[ ]:


data.head()#shows first 5 rows


# ### intial observation

# we have more null values in data

# ### Some stats

# In[ ]:


data.info()


# This data.info() gives you the **column values, no.of.rows, Memory size** and their **data types**. you can even get the **data types** by typing data.dtypes.

# ### observation:-
# - we have 43 columns and 448203, ranging from 0 to 448202.
# - we have more than half of the columns with missing values.
# - 2 columns have boolean, 3 columns have integer, 38 columns have object data type.

# In[ ]:


data.shape


# this says we have 448203 rows and 43 columns.

# In[ ]:


data.columns


# Here we have list of the columns used in dataset.

# In[ ]:


data.describe()


# this describing the columns with int data type with **count,Mean(average),Min(minimum),Max(maximum),Std(Standard-deviation) and percentiles**.

# ### Missing values

# In[ ]:


data.isna().sum()


# This shows the **sum of missing values** in Dataset for each column

# In[ ]:


#view the missing values in ascending order
data.isna().sum().sort_values(ascending = False)


# In[ ]:


#lets view the columns that have more than 50% missing/NULL values.
data.isna().sum().sort_values(ascending = False).head(26)


# ### Observation:-
# - so we have **26 columns** as shown above which **have morethan 50% of data missing**, so we can drop them
# - we can **drop them**, becuase we can not impute missing values as more than half of data is missing, so if we **impute it might show effect on Accuracy and on our model**.

# In[ ]:


data.isna().sum().sort_values(ascending = False).tail(17)


# so in these 17 columns, columns like **Dimensions,Classification,Title,Objectdate,Medium,Objectname,Creditline** have missing values where we **can impute**, and columns like **Department, Objectid, ispublicdomain, ishighlight, repository, metadatadate, objectbegindate, objectenddate, linkresource, objectnumber** have no missing values. 

# In[ ]:





# ### Data visualizations

# univariate analysis

# ### Department

# In[ ]:


data['Department'].value_counts()


# In[ ]:


fig= plt.figure(figsize=(18, 9))
sns.countplot(data['Department'])


# In[ ]:


data['Department'].value_counts()[:20].plot(kind='barh')


# ### Observations
# - we can rank them like below
#   - 1st - Drawings and Prints
#   - 2nd - European Sculpture and Decorative Arts
#   - 3rd - Asian Art
#   - 4th - Photograpbs
#   - 5th - Costume institute
#  
# So most of them are from **Drawsings and Prints**.

# ### ObjectID

# In[ ]:


data['Object ID'].value_counts().sum()


# This says that **we have uniques values** in this columns as it is **ID**, so we have **no duplicates**.

# ### IsPublicDomain

# In[ ]:


data['Is Public Domain'].value_counts()


# - so we have **202199** in **Public domain**.
# - **246004** not in public domian

# ### Repository

# In[ ]:


data['Repository'].value_counts()[:20].plot(kind='barh')


# we have the only repository is **Metropolitan Museum of Art** with all data.

# ### Objectname

# In[ ]:


data['Object Name'].value_counts()[:10].plot(kind = 'barh')


# we have most of the object names are 
# - from **Print** keeping Photograph,drawing next.

# ### Culture

# In[ ]:


data['Culture'].value_counts()[:10].plot(kind = 'bar')


# In[ ]:


data['Culture'].value_counts()[:10].plot(kind = 'area')


# In[ ]:


data['Culture'].value_counts()[:10].plot(kind = 'pie')


# from all the above graphs, you can say that
# - we have **highest** art from **American**.
# - **second** is from **French**
# - **3rd** is from ***japan**

# ### Artist Role

# In[ ]:


data['Artist Role'].value_counts()[:10].plot(kind = 'pie')


# Most of the artists play a role of 
#  - Artist -1st
#  - Publisher - 2nd

# ### Artist Display Name

# In[ ]:


data['Artist Display Name'].value_counts().head(5).plot(kind = 'bar')


# we have more of the Art from the Artist **Walker Evans**, with **Kinney brothers** in second palce.

# ### Country

# In[ ]:


data['Country'].value_counts().head(15)


# In[ ]:


data.Country.value_counts()[:10]


# In[ ]:


data['Country'].value_counts()[:20].plot(kind='barh')


# So its clear that we have **most of the art** from in rankings is 
# - Egypt
# - Unites States
# - Iran

# ### Heatmap

# Bivariate analysis

# In[ ]:


# Let's check the correlation between the variables 
plt.figure(figsize=(20,10)) 
sns.heatmap(data.corr(), annot=True)


# so here we can see that we have more **correlation between the variables** are
# - Object beign Date & object End Date.

# ## Final observation
# 
# - most of the ART are from **Drawsings and Prints**.
# - ObjectID is **unique**(no duplicates)
# - we have 202199 art works in **Public** domain.
# - Most of the art work is in **Print** format.
# - we have the highest art work from **American** and **French**.
# - Most of the art work is purely from **Artists**.
# - we have more of the Art from the **Artist Walker Evans**, with **Kinney brothers in second palce**.
# - we have most of the art from the country **Egypt**.

# In[ ]:




