#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates
# 
# <img src="data:image/jpeg;base64,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"/>
# 
# <br>
# 
# ## Key facts regarding suicides
# * Close to 800 000 people die due to suicide every year.
# * For every suicide there are many more people who attempt suicide every year. A prior suicide attempt is the single most important risk factor for suicide in the general population.
# * Suicide is the third leading cause of death in 15-19-year-olds.
# * 79% of global suicides occur in low- and middle-income countries.
# * Ingestion of pesticide, hanging and firearms are among the most common methods of suicide globally.

# # Importing important libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
import plotly.figure_factory as ff
import plotly.express as px


# In[ ]:


data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.country.value_counts()[:5]


# ## A animation chart describing the number of suicides in top countries

# In[ ]:


# don't parse dates while running this code
perc = data.loc[:,["year","country",'suicides_no']]
perc['total_suicides'] = perc.groupby([perc.country,perc.year])['suicides_no'].transform('sum')
perc.drop('suicides_no', axis=1, inplace=True)
perc = perc.drop_duplicates()
perc = perc[(perc['year']>=1990.0) & (perc['year']<=2012.0)]
perc = perc.sort_values("year",ascending = False)

top_countries = ['Mauritius','Austria','Iceland','Netherlands',"Republic of Korea"] 
perc = perc.loc[perc['country'].isin(top_countries)]
perc = perc.sort_values("year")
fig=px.bar(perc,x='country', y="total_suicides", animation_frame="year", 
           animation_group="country", color="country", hover_name="country")
fig.show()


#  ## Pie chart describing age-group distribution of people commiting suicide

# In[ ]:


data.age.value_counts().plot(kind='pie',shadow=True,startangle=90,explode=(0,0.1,0,0,.2,0),figsize=(15,10))


# In[ ]:


data=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv",parse_dates=["year"],index_col="year")
data.head()


# In[ ]:


data['gdp_per_capita ($)'][:'2000'].plot(figsize=(15,10),legend=True,color='r')
data['gdp_per_capita ($)']['2000':].plot(figsize=(15,10),legend=True,color='g')
plt.legend(["Before 2000","After 2000"])
