#!/usr/bin/env python
# coding: utf-8

# # Google PlayStore Study ::
# ## (((( Cleaning Data ))))

# ## - Importing Libraries ::

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## - Data configrations ::
# 

# In[ ]:


df_google= pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


df_google.info()


# ## - Cleaning the data:

#  ### Create heate map to indecate the nan values in all dataset:

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )


# ## Fill nan value with the mean of col:

# In[ ]:


df_google=df_google.fillna(value=df_google['Rating'].mean())


# ## Cheack the heat map and save it in external csv file

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )
df=df_google.to_csv('Goog_out1.csv')


# ## Prepration for Model ::
# ### Transfer Object to numric :
# ### (Reviews, Size, Installs, Price)

# ## convert Reviews to string then replase M with 000000 :

# In[ ]:


df_google['Reviews']=df_google['Reviews'].str.replace('M','000000').astype(float)


# In[ ]:


df_google.to_csv('Goog_out1.csv')
df= pd.read_csv('Goog_out1.csv')
df
#Save it


# ## I found a statment in the col that I want to make it a float so i remove it with 0.0

# In[ ]:


df_google[df_google['Size']=='Varies with device']=0.0


# ## Replase all useless values for the KING Float 

# In[ ]:


df_google['Size']=df_google['Size'].str.replace('M','000000')
df_google['Size']=df_google['Size'].str.replace('+','')
df_google['Size']=df_google['Size'].str.replace(',','')


# In[ ]:


df_google['Size']=df_google['Size'].str.replace('k','000').astype(float)


# In[ ]:


df_google.to_csv('Goog_out1.csv')
df= pd.read_csv('Goog_out1.csv')
df
#Save it


# ## I found another damm statment in the data col that I want to make it a float so i remove it with 0.0
# ## Replase all useless values for the KING Float 

# In[ ]:


df_google[df_google['Installs']=='Free']=0.0
df_google['Installs']=df_google['Installs'].str.replace('+','')
df_google['Installs']=df_google['Installs'].str.replace(',','')
df_google['Installs']=df_google['Installs'].astype(float)


# In[ ]:


df_google.to_csv('Goog_out1.csv')
df= pd.read_csv('Goog_out1.csv')
df.info()


#  # **FINALLY !**

# In[ ]:




