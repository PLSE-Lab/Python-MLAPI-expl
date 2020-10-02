#!/usr/bin/env python
# coding: utf-8

# # Corona
# - I use log10(Cases)

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df = df.groupby(['ObservationDate',"Country/Region"]).sum().reset_index()

# log10 here 
df['daily_existing']=np.log10(df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff())
colors="Oranges"
fig = px.choropleth(df,locations="Country/Region",locationmode="country names",                     color="daily_existing",animation_frame="ObservationDate",                     color_continuous_scale=colors, hover_name="Country/Region",                    range_color=(0.1,5))
fig.update_layout(title_text="Remaining cumulative cases per country",title_x=0.5)
fig.update_layout(transition = {'duration': 30})
fig.show()


# # Nepal earthquake
# - Occured on UTC 2015-04-25 06:11:25
# - I chose dataset from 04-25 06h to 04-27 06h (UTC). To see aftershocks.

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import datetime

df = pd.read_csv("/kaggle/input/eq20150425/query.csv")
df.index=pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H::00')
df = df.iloc[::-1].reset_index(drop=True)

colors="Oranges"#"Rainbow"
fig = px.scatter_geo(df,lat="latitude",lon="longitude",                     color="mag",animation_frame="time",                     color_continuous_scale=colors, hover_name="mag",                    range_color=(1,6))

fig.show()


# In[ ]:



# df = pd.read_csv("/kaggle/input/eq20150425/query.csv")
# df = df.groupby(['ObservationDate',"Country/Region"]).sum().reset_index()

# # log10 here 
# df['daily_existing']=np.log10(df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff())
# colors="Oranges"
# fig = px.choropleth(df,locations="Country/Region",locationmode="country names",\
#                      color="daily_existing",animation_frame="ObservationDate",\
#                      color_continuous_scale=colors, hover_name="Country/Region",\
#                     range_color=(0.1,5))
# fig.update_layout(title_text="Remaining cumulative cases per country",title_x=0.5)
# fig.update_layout(transition = {'duration': 30})
# fig.show()

