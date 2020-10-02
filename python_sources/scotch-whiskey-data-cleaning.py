#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import re
import itertools
import warnings
warnings.filterwarnings('ignore')
import os

raw= pd.read_csv("../input/scotch_review.csv")


# In[ ]:


raw.head(10)


# ********#To figure out the name, age and concentration********

# In[ ]:


#To figure out the name, age and concentration

df=raw.name.str.split(',',expand=True)

df.head(3)


# **Renaming columns**

# In[ ]:


df.columns=['name','a','b','c','d']
df.head()


# ** Extracting the concentration % of Whiskey to a single column****

# In[ ]:



df.a = df.a.apply(lambda x: np.nan if not '%' in str(x) else x)
df.b = df.b.apply(lambda x: np.nan if not '%' in str(x) else x)
df.c = df.c.apply(lambda x: np.nan if not '%' in str(x) else x)
df.d = df.d.apply(lambda x: np.nan if not '%' in str(x) else x)
df['concentration'] = [df.a[i] if df.a[i] is not np.nan else df.b[i] if df.b[i] is not np.nan else df.c[i] if df.c[i] else df.d[i] if df.d[i] is not np.nan else df.a[i] for i in range(df.shape[0])]


# In[ ]:


df.reindex().head()


# **Creating a new table with name and concentration**

# In[ ]:


df_a=pd.concat([df.name,df.concentration],axis=1)
df_a.sample(5)


# **Lets clean the price column**

# In[ ]:


df_p=raw.price.str.replace("[({',$qwertyuioplkjhgfdsazxcvbnm%:]", " ")
df_price=df_p.convert_objects(convert_numeric=True)
df_price.sample()


# **Removing % sign and changing the dtype of column concentration**

# In[ ]:


df_b=df_a.concentration.str.replace("[({'qwertyuioplkjhgfdsazxcvbnm%:]", " ")
df_b.head()


# In[ ]:


df_c=df_b.convert_objects(convert_numeric=True)
df_c.head()


# In[ ]:


df_col_merged =pd.concat([df_a.name, df_c, df_price, raw.category, raw.iloc[:,3],raw.description], axis=1)

df_col_merged.info()


# In[ ]:


final_df=df_col_merged.dropna()
final_df


# In[ ]:


final_df.loc[final_df['price'].idxmax()]



# **Frequency distribution of Whiskey conc %**

# In[ ]:


GSW=final_df.loc[final_df['category'] == 'Grain Scotch Whisky', 'name':'category']
BMSW=final_df.loc[final_df['category'] == 'Blended Malt Scotch Whisky', 'name':'category']
SGW=final_df.loc[final_df['category'] == 'Single Grain Whisky', 'name':'category']
SMS=final_df.loc[final_df['category'] == 'Single Malt Scotch', 'name':'category']
BSW=final_df.loc[final_df['category'] == 'Blended Scotch Whisky', 'name':'category']


# In[ ]:




fig, ax = plt.subplots(2, 3, figsize=(12, 10))
sns.distplot(final_df['concentration'],axlabel="Frequency distribution of Concentration", color="r", kde = True, ax=ax[0][0])
#sns.distplot(smw['concentration'],  hist=False,kde = True,rug_kws={"color": "g"}, ax=ax[0][1])
sns.distplot(GSW['concentration'],axlabel="GSW", kde = True, ax=ax[0][1])
sns.distplot(BMSW['concentration'], axlabel="BMSW",    kde = True, ax=ax[0][2])
sns.distplot(SGW['concentration'],axlabel="SGW",  kde = True, ax=ax[1][0])
sns.distplot(SMS['concentration'], axlabel="SMS", kde = True, ax=ax[1][1])
sns.distplot(BSW['concentration'],label='BSW' ,axlabel="BSW", kde = True, ax=ax[1][2])




# **Lets analyze if there is any relation b/w concentration and price**

# In[ ]:


whiskey_type=final_df.category.unique()
whiskey_type


# In[ ]:


final_df.groupby('category').mean()

