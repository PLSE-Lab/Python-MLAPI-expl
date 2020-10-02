#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install plotly-geo


# In[ ]:


import plotly.figure_factory as ff

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/icu-beds-by-county-in-the-us/data-FPBfZ.csv')


# In[ ]:


fip_codes_df = pd.read_csv('https://raw.githubusercontent.com/kjhealy/fips-codes/master/county_fips_master.csv', encoding = "ISO-8859-1")


# In[ ]:


fip_codes_df['county_name'] = fip_codes_df['county_name'].apply(lambda x: x.replace(' County', ''))


# In[ ]:


fip_codes = fip_codes_df.set_index('county_name')['fips'].to_dict()


# In[ ]:


df['county_fips_code'] = df['County'].apply(lambda x: fip_codes.get(x))


# In[ ]:


df_county = df[pd.notnull(df['county_fips_code'])]


# In[ ]:


colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
endpts = list(np.linspace(1, 12, len(colorscale) - 1))
fips = df_county['county_fips_code'].tolist()
values = df_county['ICU Beds'].tolist()

fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='ICU Beds per County',
    legend_title='Residents Aged 60+ Per Each ICU Bed'
)

fig.layout.template = None
fig.show()


# In[ ]:


icu_by_state = df.groupby('State')[['ICU Beds']].sum()


# In[ ]:


plt.figure(figsize=(15, 10))
ax = sns.barplot(x='ICU Beds', y='State', data=icu_by_state.reset_index().sort_values(by=['ICU Beds'], ascending=False))


# In[ ]:




