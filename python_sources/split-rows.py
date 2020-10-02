#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.DataFrame({
   'Cuisine': ['001', '002', '003', '004', '005'],
   'Bill': ['100', '200', '300', '400', '500'],
        'City': ['Mumbai,Bangalore', 'Pune,Mumbai,Delhi', 'Mumbai,Bangalore', 'Mumbai,Pune', 'Bangalore'],
        'Locality': ['Beach,Centre', 'Centre', 'Beach,Centre', 'Beach,Centre', 'Centre,Bazaar'] 
   })


# In[ ]:


df


# In[ ]:


reshaped = (df.set_index(df.columns.drop('City',1).tolist())
   .City.str.split(',', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'City'})
   .loc[:, df.columns]
)


# In[ ]:


reshaped


# In[ ]:


reshaped1 = (reshaped.set_index(reshaped.columns.drop('Locality',1).tolist())
   .Locality.str.split(',', expand=True)
   .stack()
   .reset_index()
   .rename(columns={0:'Locality'})
   .loc[:, reshaped.columns]
)


# In[ ]:


reshaped1

