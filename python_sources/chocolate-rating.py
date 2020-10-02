#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().system('ls')


# In[ ]:


data = pd.read_csv('../input/flavors_of_cacao.csv')
data.head()


# In[ ]:


data['Cocoa\nPercent'] = data['Cocoa\nPercent'].apply(lambda x: float(x.split('%')[0]))/100
data.head()


# In[ ]:


data.columns[data.isnull().any()].tolist


# In[ ]:


print(data.corr())


# Chocolates Ratings do not have significant correlation with any other parameters

# In[ ]:


plt.hist(data['Rating'])


# Most Chocolates have ratings between 3.0 and 4.0

# In[ ]:


plt.hist(data['Cocoa\nPercent'])


# Most chocolates have 70%  of cocoa

# In[ ]:


plt.plot(data['Rating'], data['Cocoa\nPercent'],'o')


# This shows no correlation between the amount of cocoa to the ratings of the chocolates

# In[ ]:


data['Specific Bean Origin\nor Bar Name'][data['Rating']==5]


# Chuao and Toscano Black got the bean for highest rated chocolates

# In[ ]:


data['Company\xa0\n(Maker-if known)'][data['Rating']==5]


# Amedei got the 5 star rated chocolates. Let's see what average rating they got.

# In[ ]:


amedei=data['Rating'][data['Company\xa0\n(Maker-if known)']=='Amedei']
print(amedei)
print('Average rating for Amedei is: ',amedei.mean())


# In[ ]:


data[data['Specific Bean Origin\nor Bar Name'].isin(['Chuao', 'Toscano Black'])]


# The 5 rating for the chocolate is probably because the company and the place of the bean.
# Amedei which produces chocolates from Chuao and Toscano Black produces the highest rated chocolates. Best beans are grown in Venezuela.

# In[ ]:


chuao=data['Rating'][data['Specific Bean Origin\nor Bar Name']=='Chuao']
toscano=data['Rating'][data['Specific Bean Origin\nor Bar Name']=='Toscano Black']
print('Average rating for chocolates produced from Chuao bean is: ',chuao.mean())
print('Average rating for chocolates produced from Toscano Black bean is: ',toscano.mean())


# Generally, chocolates produced from Toscano Black bean has average higher rating than that from Chuao bean.

# In[ ]:


data['Company\xa0\n(Maker-if known)'][data['Rating']<2]


# These companies produce the lowest rated (<2) chololates.

# In[ ]:




