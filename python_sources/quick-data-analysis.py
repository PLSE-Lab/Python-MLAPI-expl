#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/Data_Entry_2017.csv')
print(len(df.index))
df.head()


# In[ ]:


#drop unused columns
df = df[['Image Index','Finding Labels']]

#create new columns for each decease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia',
                  'Nodule','Pneumothorax','Atelectasis','Pleural_Thickening',
                  'Mass','Edema','Consolidation','Infiltration','Fibrosis',
                  'Pneumonia','No Finding']

for pathology in pathology_list :
    df[pathology] = df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
    
df['Multi_Finding'] = df['Finding Labels'].apply(lambda x: 1 if x.find('|') > -1 else 0)

df.head()


# In[ ]:


data1 = pd.melt(df,
             id_vars=['Image Index'],
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]
data1.head()


# In[ ]:


data1_grouped = data1.groupby('Category')['Count'].sum()
data1_grouped


# In[ ]:


import matplotlib.pyplot as plt

data1_grouped.plot(kind='bar')
plt.show()


# In[ ]:




