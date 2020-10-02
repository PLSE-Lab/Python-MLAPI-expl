#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline  import iplot
import plotly as py
import plotly.tools as tls


# In[ ]:


print(py.__version__)
#### Plotly can work online and offline
py.offline.init_notebook_mode(connected =True)


# In[ ]:


dataDf = pd.read_csv('../input/lozpdata-assignment-week01-data-visualizaition/train.csv')
dataDf['Province_State'] =dataDf['Province_State'].fillna('NA')
dataDf.isnull().sum()
filteredDf = dataDf.loc[(dataDf.ConfirmedCases > 0) | (dataDf.Fatalities > 0)]


# In[ ]:


print(type(filteredDf))


# In[ ]:


filteredDf.iplot(kind='scatter',x= 'Country_Region',y='ConfirmedCases',title ='Country_Wise_Confirmed_Cases',xTitle  ='Country Name',yTitle='Number of Cases',theme='solar')


# In[ ]:


filteredDf.iplot(kind='scatter',x= 'Country_Region',y='Fatalities',title ='Country_Wise_Death',xTitle  ='Country Name',yTitle='Number of Deaths',theme='white',bargap=1)


# In[ ]:


### Plotting the Deaths Day Wise

confirmedCf =  dataDf.loc[(dataDf.ConfirmedCases > 0)]

#### Plotting how the confirmed cases increases day on day basis , we will be plotting line graph
confirmedCf['Date'] = pd.to_datetime(confirmedCf['Date'])
firstDate = list(confirmedCf['Date'].sort_values())
firstDate =firstDate[0]
def assignFIrstDay(x):
    
    return firstDate
confirmedCf['firstDateofCase']  ='NA'
confirmedCf['firstDateofCase'] =confirmedCf['firstDateofCase'].apply(assignFIrstDay)


# In[ ]:


confirmedCf['Number_of_Day'] = confirmedCf['Date']  - confirmedCf['firstDateofCase'] 
confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].astype('string')
confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].str.replace('days 00:00:00.000000000','')
confirmedCf['Number_of_Day'] =confirmedCf['Number_of_Day'].astype('int')


# In[ ]:


confirmedCf.iplot(kind='bar',x ='Number_of_Day',y = 'ConfirmedCases' ,title ='Confirmed_Cases_Day_Wise',xTitle  ='Country Name',yTitle='Number of Deaths',theme='solar')


# In[ ]:


confirmedCf.iplot(kind='bar',x ='Date',y = 'Fatalities' ,title ='Fatalities_Date_Wise',xTitle  ='Day',yTitle='Number of Deaths')


# In[ ]:



usData = filteredDf.loc[(filteredDf.Country_Region == 'US')]
usData.columns
### Visualizing the Data Area Wise


# In[ ]:


usData.Province_State.nunique()


# In[ ]:


get_ipython().system('pip install bubbleplot')


# In[ ]:


from bubbly.bubbly import bubbleplot


# In[ ]:


figure = bubbleplot(dataset=usData, x_column='ConfirmedCases', y_column='Fatalities', 
 bubble_column='Date', color_column='Province_State',
    x_title="ConfirmedCases", y_title="Fatalities", title='Confirmed Cases and Fatalities State Wise for USA',
     height=650)

iplot(figure, config={'scrollzoom': True})


# In[ ]:





# In[ ]:




