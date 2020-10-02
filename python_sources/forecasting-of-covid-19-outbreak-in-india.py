#!/usr/bin/env python
# coding: utf-8

# ## First Analyse the present condition in India
# 
# First COVID-19 case reported in India on 30th january 2020 when s student arrived Kerala from Wuhan. <br>
# It's 27th March morning and india has its 934 death with 21632 total confirmed cases due to COVID-19. fresh cases from all states/Union Territory have been reported by the union Ministry of Healt & Family welfare. <br>
# 
# Goal:-
# We need a strong model that predicts how the virus could spread across differnet  states/UTs. The goal of this task is to build a model that predicts the spread of the virus in the next 30 days.

# In[ ]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
import plotly.express as px


# #### Reading the Dataset

# In[ ]:


Cases=pd.read_csv('../input/covid_19_india.csv',index_col=[1])
Cases.tail(4)


# In[ ]:


Cases.index=pd.to_datetime(Cases.index,dayfirst=True)


# In[ ]:


Cases.drop(['Sno','Time',],axis='columns',inplace=True)


# In[ ]:


Cases.rename({'State/UnionTerritory':'State/UTs','Cured':'Recovered'},axis='columns',inplace=True)


# In[ ]:


Cases.tail(5)


# In[ ]:


Cases['ActiveCases']=Cases['Confirmed']-(Cases['Recovered']+Cases['Deaths'])


# In[ ]:


Cases.drop({'ConfirmedIndianNational','ConfirmedForeignNational'},axis='columns',inplace=True)


# In[ ]:


Cases.tail(4)


# In[ ]:


print('Earlist Entry :',Cases.index.min())
print('Last Entry    :',Cases.index.max())
print('Total Day     :',Cases.index.max()-Cases.index.min())


# In[ ]:


data_today=Cases[Cases.index=='2020-04-28']


# In[ ]:


data_today


# In[ ]:


fig = px.pie(data_today[data_today["Confirmed"]>100], values="Confirmed", names="State/UTs", title="Number of confirmed Cases by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


fig = px.pie(data_today[data_today["Confirmed"]>100], values="Deaths", names="State/UTs", title="Number of Deaths by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


fig = px.pie(data_today[data_today["Confirmed"]>100], values="Recovered", names="State/UTs", title="Number of Recovered by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


date_wise = Cases.groupby(['Date','State/UTs','Confirmed'])['Recovered','Deaths','ActiveCases'].sum().reset_index().sort_values('Confirmed',ascending=False)


# In[ ]:


date_wise.head(4)


# In[ ]:


fig = px.bar(date_wise,height=500,x='Date',y='Confirmed',hover_data =['State/UTs','ActiveCases','Deaths'],color='Confirmed')
fig.show()


# In[ ]:


fig = px.bar(date_wise,height=500,x='Date',y='Deaths',hover_data =['State/UTs','ActiveCases','Deaths'],color='Confirmed')
fig.show()


# In[ ]:


Cases.plot()


# ## Work in Progress....
# 
# #### if u like don't forget to upvote

# In[ ]:




