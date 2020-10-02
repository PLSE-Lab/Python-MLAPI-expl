#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


claim = pd.read_csv('../input/claims-data/ClaimsData.csv',index_col = 0)


# In[ ]:


claim.head()


# In[ ]:


# claim.loc[(claim.index > 50)].iloc[:,0:11].groupby('age').sum().plot.bar(stacked=True)
# claim.loc[(claim.index > 50)].iloc[:,0:11].groupby('age').sum().plot.line()
#claim.loc[(claim.index > 50)].iloc[:,0:11].groupby('age').sum().plot.hist()
# claim.loc[(claim.index > 50)].iloc[:,0:11].groupby('age').sum().plot.area()







# Coronary artery disease (CAD), also known as ischemic heart disease (IHD), involves the reduction of blood flow to the heart muscle due to build-up of plaque in the arteries of the heart.

# In[ ]:


a = claim.loc[(claim.index > 0)].iloc[:,0:11].groupby('age').sum() 
a.describe()
#a.plot.scatter(x= a['ihd'], y = a['cancer']) 
len(a)


# In[ ]:


a.head()


# In[ ]:


insurance = pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


insurance.head()


# In[ ]:


insurance.head(1000).plot.scatter(x = 'age' , y = 'charges' )


# In[ ]:


insurance.plot.hexbin(y = 'age' , x = 'charges',gridsize=15 )


# We can see charges increase as the age increase

# In[ ]:


insurance[['age','charges']].groupby('age').sum().plot.line()


# We can see the charges increasing over a period of time.

# In[ ]:


insurance.select_dtypes(exclude = 'object').plot.bar(stacked=True)


# In[ ]:


claim.loc[(claim.index > 30) & (claim.index < 60)].groupby('age').sum().iloc[:,0:11].plot.bar(stacked = True)


# In[ ]:


claim.groupby('age').sum().plot.line(legend= False)
claim.groupby('age').sum().plot.area()


# In[ ]:




