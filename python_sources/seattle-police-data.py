#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import plotly.plotly as py
from datetime import timedelta, date
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import os
print(os.listdir("../input"))


# In[ ]:


police = pd.read_csv('../input/crisis-data.csv')
police.tail()


# In[ ]:


police['Reported'] = pd.to_datetime(police['Occurred Date / Time'])
police['Reported Date'] = pd.to_datetime(police['Reported Date'])
police.set_index('Reported Date')


# In[ ]:


police.info()


# In[ ]:


police.sort_values(by='Reported Date').tail()


# In[ ]:


police['Subject Veteran Indicator'].value_counts()


# In[ ]:


police['Use of Force Indicator'].value_counts()


# In[ ]:


police.describe()


# In[ ]:


police['Year'] = police['Reported Date'].dt.year
police['Month'] = police['Reported Date'].dt.month
police['Hour'] = police['Reported'].dt.hour
police.tail()


# In[ ]:


police['Year'].value_counts()


# In[ ]:


police = police.drop(police[(police.Year == 1900)].index)


# In[ ]:


annual = police.groupby(['Year']).size()
annual.plot.bar(x='Year')


# In[ ]:


month = police.groupby(['Month']).size()
month.plot.bar(x='Month')


# In[ ]:


hour = police.groupby(['Hour']).size()
hour.plot.bar(x='Hour')


# In[ ]:


police['Initial Call Type'].value_counts().head(10)


# In[ ]:


month = pd.DataFrame(month)
month[0]


# In[ ]:


data1 = [go.Scatter(x=month.index, y=month[0])]

layout = dict(title = "Number of Calls by Month",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

fig = dict(data = data1, layout = layout)
iplot(fig)


# In[ ]:


yesdate = date.today() - timedelta(2)
yesterday = police[police['Reported Date'] == yesdate]
call_type = yesterday['Initial Call Type'].value_counts()
call_type = pd.DataFrame(call_type)
call_type


# In[ ]:


data2 = [go.Bar(x=call_type.index, y=call_type['Initial Call Type'])]

layout = dict(title = "Type of Calls " + str(yesdate),
              xaxis= dict(title= 'Date',ticklen= 0, dtick=1, automargin=True))

fig = dict(data = data2, layout = layout)
iplot(fig)

