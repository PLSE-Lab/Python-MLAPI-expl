#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/852k-used-car-listings/tc20171021.csv', error_bad_lines=False)


# In[ ]:


display(df.head())
display(df.info())


# In[ ]:


del df['Vin']
df.head()


# In[ ]:


print("We can see that there is no missing data")
df.info()


# In[ ]:


print("The number of rows is %d and columns is %d" % df.shape)


# In[ ]:


print("The following columns are numerical: Price, Year, Mileage")
print("The following columns are nominal (We have no binomial or ordinal categorical): City, State, Vin, Make, Model")


# In[ ]:


count_by_make = df.groupby('Make')['Id']                   .count()                   .reset_index(name='count')                   .sort_values(['count'], ascending=False)

fig, ax = plt.subplots(figsize=(6,18))
sns.barplot(ax=ax, x='count', y='Make', data=count_by_make)


# In[ ]:


display(type(df.groupby('Make')['Id'].count()))
display(type(df.groupby('Make')['Id'].count()                         .reset_index(name='count')))


# In[ ]:


df[df['Make'] == 'Porsche'].sort_values('Price', ascending=False)


# In[ ]:


df.head()


# In[ ]:




