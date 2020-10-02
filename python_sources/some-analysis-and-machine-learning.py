#!/usr/bin/env python
# coding: utf-8

# # Texture database 

# This sediment database contains location, description, and texture of samples taken by numerous marine sampling programs. Most of the samples are from the Atlantic Continental Margin of the United States, but some are from as diverse locations as Lake Baikal, Russia, the Hawaiian Islands region, Puerto Rico, the Gulf of Mexico, and Lake Michigan. The database presently contains data for over 27,000 samples, which includes texture data for approximately 3800 samples taken or analyzed by the Atlantic Continental Margin Program (ACMP), a joint U.S. Geological Survey/Woods Hole Oceanographic Institution project conducted from 1962 to 1970. As part of the ACMP, some historical data from samples collected between 1955 and 1962 were also incorporated into the dataset.

# ### In this kernel i will: 

#     - Clean up missing data 
#     - Find which location who has the most findings
#     - Analyse Georges bank
#     - Find correlations using different heatmaps

# In[ ]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read the data
tex = pd.read_csv('../input/texture.csv',index_col=0)


# In[ ]:


# Visualizing the missing data with heatmap
sns.heatmap(tex.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# First we have to clean up all the missing data. Now we don't have any data that we can plug in so instead i will replace the NaN with floats with value of 0.

# In[ ]:


# Filling the NaN values with 0.00
clean = tex.fillna(0.00)
clean.head()


# In[ ]:


# Check the columns.
clean.info()


# Now all the missing data is set as 0, which basically isnt a value, so it wont affect any models. 

# In[ ]:


sns.heatmap(clean.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Top Lithology 

# In[ ]:


# Value counts for the Lithology 
top = clean['LITHOLOGY'].value_counts().head(10)
top.head()


# As we can se above, there is a lot of data collected for Massachusetts and Georges bank. I want to analyse the data for Georges Bank and Gulf of Mexico.

# In[ ]:


clean ['AREA'].value_counts().head(10)


# First on the list is Georges Bank. 
# 
# NOTE: print ut ett bilde og fortell basics om stein derifra. Finn hva som er gjennomsnittstemp. flest type stein osv.  

# # George Bank visualizations

# In[ ]:


# Singeling out the Georges Bank columns. 
Georges = clean[clean['AREA']=='GEORGES BANK']


# # Georges Bank Correlations 
#  

# In[ ]:


# Finding correlations.
sns.heatmap(Georges.corr(method='pearson'))


# Lets check the info to see what floats we want to pick out, or the columns that looks interesting to check out.

# In[ ]:


#Here is a better visuals for latitude and longitude.
sns.jointplot(x='LATITUDE',y='LATITUDE',
              data=Georges,kind='reg',
              color='b')
sns.set(style='white',color_codes=True)


# In[ ]:


# More detailed correlations with scikit 
from sklearn.preprocessing import LabelEncoder
labe = LabelEncoder()
dic = {}

labe.fit(Georges.MONTH_COLL.drop_duplicates())
dic['MONTH_COLL'] = list(labe.classes_)
Georges.MONTH_COLL = labe.transform(Georges.MONTH_COLL)


# In[ ]:


cor = ['LATITUDE','LONGITUDE','DEPTH_M','T_DEPTH','B_DEPTH']


# In[ ]:


kor = np.corrcoef(Georges[cor].values.T)


# In[ ]:


sns.set(font_scale=1.5)
map = sns.heatmap(kor,cbar=True,
                  cmap="YlGnBu",
                  annot = True, 
                  square= True,
                  fmt = '.1f',
                  annot_kws = {'size':10}, 
                 yticklabels = cor,
                 xticklabels = cor)


# # Analisis of the Gulf 

# In[ ]:


clean ['AREA'].value_counts().head(10)


# In[ ]:


gulf = clean[clean['AREA']=='GULF OF MEXICO']


# In[ ]:


gulf.head()


# In[ ]:


sns.heatmap(gulf.corr())


# In[ ]:


#the correlations look the same for both gulf and Georges bank
sns.jointplot(x='LATITUDE',y='LATITUDE',
              data=gulf,kind='reg',
              color='b')
sns.set(style='white',color_codes=True)


# In[ ]:


# We see that some of the correlation values are different. 
from sklearn.preprocessing import LabelEncoder
labe = LabelEncoder()
dic = {}

labe.fit(gulf.MONTH_COLL.drop_duplicates())
dic['MONTH_COLL'] = list(labe.classes_)
gulf.MONTH_COLL = labe.transform(gulf.MONTH_COLL)

cor = ['LATITUDE','LONGITUDE','DEPTH_M','T_DEPTH','B_DEPTH']
kor = np.corrcoef(gulf[cor].values.T)
sns.set(font_scale=1.5)
map = sns.heatmap(kor,cbar=True,
                  cmap="YlGnBu",
                  annot = True, 
                  square= True,
                  fmt = '.1f',
                  annot_kws = {'size':10}, 
                 yticklabels = cor,
                 xticklabels = cor)

