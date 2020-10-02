#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# In[ ]:


notices = pd.read_csv('../input/procurement-notices.csv')


# In[ ]:


notices.info()


# In[ ]:


notices['Notice Type'].unique()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
notices.groupby(["Publication Date", "Notice Type"])["Project ID"].count().unstack().plot(ax = ax)


# In[ ]:


notices['Publication Date'] = pd.to_datetime(notices['Publication Date'])
notices['Deadline Date'] = pd.to_datetime(notices['Deadline Date'])
notices.info()


# In[ ]:


notices[(notices['Deadline Date'] > pd.Timestamp.today()) | (notices['Deadline Date'].isnull())].count().ID


# In[ ]:


current_calls = notices[(notices['Deadline Date'] > pd.Timestamp.today())]
calls_by_country = current_calls.groupby('Country Name').size()

iplot([go.Choropleth(
    locationmode='country names',
    locations=calls_by_country.index.values,
    text=calls_by_country.index,
    z=calls_by_country.values
)])


# In[ ]:


ax = current_calls.groupby('Deadline Date').size().plot.line()
ax.set_title('Deadline Dates Distribution')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




