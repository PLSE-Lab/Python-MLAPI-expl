#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns


# In[ ]:


df_mush = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


df_mush.head()


# In[ ]:


df_mush.describe().T


# In[ ]:


# https://www.kaggle.com/fedi1996/boston-crime-analysis-with-plotly

import plotly.express as px

habitat_counts = df_mush['habitat'].value_counts()
values = habitat_counts.values
categories = pd.DataFrame(data=habitat_counts.index, columns=["habitat"])
categories['values'] = values

fig = px.treemap(categories, path=['habitat'], values=values, height=700,
                 title="Counts by Habitat", 
                 color_discrete_sequence = px.colors.sequential.RdBu)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# In[ ]:


df_mush[df_mush['class']=='e'].describe().T


# In[ ]:


df_mush[df_mush['class']=='p'].describe().T


# In[ ]:


df_mush.isnull().sum()


# In[ ]:


df_mush.info()


# In[ ]:


#https://github.com/santosjorge/cufflinks/issues/185
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[ ]:


import cufflinks as cf
cf.set_config_file(offline=True)


# In[ ]:


df_mush.columns


# In[ ]:


for col in df_mush.columns[1:]:
    pd.crosstab(df_mush['class'], df_mush[col], margins=True, normalize=True).iplot(kind='bar', title = 'Counts of mushroom class in '+col)
# pd.crosstab(df_mush['class'], df_mush['cap-shape'], margins=True, normalize=True).iplot(kind='bar', title = 'Counts of mushroom class in cap-shape')
# pd.crosstab(df_mush['class'], df_mush['cap-surface'], margins=True, normalize=True).iplot(kind='bar')


# In[ ]:


for col in df_mush.columns[1:]:
    pd.crosstab(df_mush[col], df_mush['class'], normalize=True).iplot(kind='bar', title ='Types of ' + col)


# In[ ]:


#https://stackoverflow.com/questions/12286607/making-heatmap-from-pandas-dataframe
#https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe
from IPython.display import display, HTML


# In[ ]:


for col in df_mush.columns[1:]:
    df_c = pd.crosstab(df_mush['class'], df_mush[col], normalize=True)
    style = df_c.style.background_gradient(cmap='Blues')
    display(style)


# In[ ]:


for col in df_mush.columns[1:]:
    df_c = pd.crosstab(df_mush[col], df_mush['class'], normalize=True)
    style = df_c.style.background_gradient(cmap='Blues')
    display(style)


# In[ ]:


dum = pd.get_dummies(df_mush, prefix=df_mush.columns)


# In[ ]:


dum.corr()['class_e'].nlargest(5)


# In[ ]:


dum.corr()['class_e'].nsmallest(5)


# In[ ]:


dum.corr()['class_p'].nlargest(5)


# In[ ]:


dum.corr()['class_p'].nsmallest(5)

