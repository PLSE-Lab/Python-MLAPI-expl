#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing packages and libraries
import pandas as pd
import plotly.express as px


# In[ ]:


#loading data set to work space
df = pd.read_csv("../input/owidcoviddata/covid-data.csv")


# In[ ]:


df


# In[ ]:


#head of the data set
df.head()


# In[ ]:


#shape of the data set
df.shape


# In[ ]:


#showing minimum date and maximum date of the dataset
print(df.date.min())
print(df.date.max())


# In[ ]:


#data cleaning
#dropping unwanted rows and arranging rows
df = df[df.location!='World']
#sorting data by date
df=df.sort_values(by=['date'])


# In[ ]:


#visualizing data
df


# In[ ]:


#visualizing dataset
fig = px.choropleth(df,locations = "iso_code",
                     color="new_cases",
                    hover_name="location",
                    animation_frame="date",
                    title="New Covid Cases",
                    color_continuous_scale=px.colors.sequential.PuRd)
fig["layout"].pop("updatemenus")
fig.show()


# In[ ]:




