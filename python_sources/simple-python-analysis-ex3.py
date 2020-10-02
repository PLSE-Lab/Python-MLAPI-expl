#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns

cocoa_df = pd.read_csv('../input/flavors_of_cacao.csv')
cocoa_df.head()


# In[ ]:


cocoa_df.describe()


# In[ ]:


cocoa_df.columns


# # Where is the best and worst chocolate produced?

# In[ ]:


mean_by_country = cocoa_df.groupby(["Company\nLocation"])['Rating'].mean()


# In[ ]:


mean_sorted = mean_by_country.sort_values(ascending=False)
top_5 = mean_sorted[:5]
bottom_5 = mean_sorted[-5:]


# In[ ]:


top_bottom_5 = pd.concat([top_5, bottom_5])


# In[ ]:


top_bottom_5.plot('barh')


# On average, the best chocolate is produced in Chile; the worst chocolate is produced in India.

# # Where are the best and worst beans grown?

# In[ ]:


mean_by_country = cocoa_df.groupby(["Broad Bean\nOrigin"])['Rating'].mean()


# In[ ]:


mean_sorted = mean_by_country.sort_values(ascending=False)
top_5 = mean_sorted[:5]
bottom_5 = mean_sorted[-5:]


# In[ ]:


top_bottom_5 = pd.concat([top_5, bottom_5])


# In[ ]:


top_bottom_5.plot('barh')


# On average, the best beans come from Peru; the worst come from Ghana & Madagascar

# * # Is there a correlation between cocoa percentage and rating?

# In[ ]:


cocoa_df['Cocoa\nPercent'] = cocoa_df['Cocoa\nPercent'].apply(lambda x: float(x.replace('%', '')))


# In[ ]:


sns.jointplot(data=cocoa_df, x='Cocoa\nPercent', y='Rating', size=7, kind='reg')


# There doesn't seem to be much correlation between cocoa percentage and the overall rating

# # How much chocolate was tested (by country)?

# In[ ]:


count_by_country = cocoa_df.groupby(["Company\nLocation"]).size()


# In[ ]:


count_by_country = count_by_country.sort_values(ascending=False)[:10]


# In[ ]:


count_by_country.plot('barh')


# In[ ]:




