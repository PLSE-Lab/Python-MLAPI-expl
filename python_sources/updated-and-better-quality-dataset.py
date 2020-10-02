#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')


# **Display top 5 rows of the Data**

# In[ ]:


data.head()


# * First Change the data's index and its name

# In[ ]:


data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
data.set_index('id', inplace=True)
data.head()


# Lets Process the floor column

# In[ ]:


floor = data['floor']
floor.replace('-',0, inplace=True)
data.head()


# Lets change animal data
# * acept - 1
# * not acept - 0

# In[ ]:


animal = data['animal']
animal.replace({'acept':1, 'not acept':0}, inplace = True)
data.head()


# Lets change furniture data
# * furnished - 1
# * not furnished - 0

# In[ ]:


furnished = data['furniture']
furnished.replace({'furnished':1, 'not furnished':0}, inplace = True)
data.head()


# Time to remove 'R$' from prices, also remove ,(comma) from amount like 1,201 to 1201
# * hoa
# * rent amount
# * property tax
# * fire insurance
# * total
# 

# In[ ]:


hoa = data['hoa']
hoa = hoa.str[2:].apply(lambda x : x.replace(',',''))
#Some data are 'm info' and 'cluso'
i_index = hoa[hoa.str.contains('m')].index
c_index = hoa[hoa.str.contains('c')].index


# In[ ]:


hoa[i_index] = 0
hoa[c_index] = 0
data['hoa'] = hoa


# In[ ]:


data['hoa'].astype('int64')
data.head()


# In[ ]:


rent = data['rent amount']
rent = rent.str[2:].apply(lambda x : x.replace(',',''))
#Some data are 'm info' and 'cluso'
i_index = rent[rent.str.contains('m')].index
c_index = rent[rent.str.contains('c')].index


# In[ ]:


rent[i_index] = 0
rent[c_index] = 0
data['rent amount'] = rent
data['rent amount'].astype('int64')
data.head()


# In[ ]:


prop = data['property tax']
prop = prop.str[2:].apply(lambda x : x.replace(',',''))
#Some data are 'm info' and 'cluso'
i_index = prop[prop.str.contains('m')].index
c_index = prop[prop.str.contains('c')].index


# In[ ]:


prop[i_index] = 0
prop[c_index] = 0
data['property tax'] = prop
data['property tax'].astype('int64')
data.head()


# In[ ]:


fire = data['fire insurance']
fire = fire.str[2:].apply(lambda x : x.replace(',',''))
#Some data are 'm info' and 'cluso'
i_index = fire[fire.str.contains('m')].index
c_index = fire[fire.str.contains('c')].index


# In[ ]:


fire[i_index] = 0
fire[c_index] = 0
data['fire insurance'] = fire
data['fire insurance'].astype('int64')
data.head()


# In[ ]:


total = data['total']
total = total.str[2:].apply(lambda x : x.replace(',',''))
#Some data are 'm info' and 'cluso'
i_index = total[total.str.contains('m')].index
c_index = total[total.str.contains('c')].index


# In[ ]:


total[i_index] = 0
total[c_index] = 0
data['total'] = total
data['total'].astype('int64')
data.head()


# Finally Our Data is much more cleaner and operatable.
# Now we can export this data to new updated csv for furthur processing

# In[ ]:


new_csv = data.to_csv('updated_brasilian_housing_to_rent.csv')

