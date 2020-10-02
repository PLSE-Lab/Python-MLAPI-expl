#!/usr/bin/env python
# coding: utf-8

# This is where I ask my questions and answer my questions..

# In[ ]:


import numpy as np #our math guy
import pandas as pd#our tabular data guy
import matplotlib.pyplot as plt #our plot guy
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# In[ ]:


data_athlete=pd.read_csv("../input/olympic-games/athletes.csv")
data_country=pd.read_csv("../input/olympic-games/countries.csv")


# In[ ]:


#data-athletes     
data_athlete.head()#11 columns


# In[ ]:


data_athlete['Final_Score'] = 3*data_athlete['gold'] +(2 * data_athlete['silver'])+data_athlete['bronze']


# In[ ]:


data_athlete.info()
data_athlete.describe()
#Inference-1:some extra and few  less values


# #inference02 from describe is that max no of gold won by a player is 5,silver is 2,bronze
# #inference03 from decribe it can be noticed the mean height of all player is 1.76 cm which is pretty high,
# #but it includes both men and women ,we need to find mean values for both sperately
# #inference04 from decribe it can be said that average of player was 72 kg..again the same question
# #What is the mean weight of males and females seprately??
# #max height:221 cm who is that dude find out pls??must be a basketballer
# #max weight:170 kg who is that dude??
# #min height:121 cm who is she??
# #min weight:31kg who is she??

# In[ ]:


data_country.head()#question which country won how man medals,who was the most medal scoring medallists is no of medals won is related to gdp or population??


# In[ ]:


data_country.info()
data_country.describe()


# In[ ]:


#there are a lot of missing value in this country table
#mean gdp=12882unit per capita,min=277unit per capita,max=101449 unit per capita 
#can you seem how gdp average gdp increases 25%->1781,50%->5233,75%->15494,max->101449....ofcouse i know luxembourg has only few people making its gdp too high


# In[ ]:


fig=px.scatter(data_athlete,x='height',y='weight',color='sex',size='gold',hover_data=['nationality','name'])
fig.show()


# ![](https://i.imgur.com/gwGOLOf.png)

# #code
# ```python
# fig=px.scatter(data_athlete,x='height',y='weight',color='gold',size='gold',hover_data=['nationality','name'])
# fig.show()
# ```

# In[ ]:


#result:as weight increase ,height increase
IFrame('https://i.imgur.com/DRfO77T.png',height=400,width=1200)


# #code
# ```python
# fig=px.scatter(data_athlete,x='nationality',y='height',color='sex',hover_data=['nationality','name'])
# fig.show()
# ```

# In[ ]:


#result:average of women's heights is smaller than average men's heights in almost all countries
IFrame('https://i.imgur.com/jNRTsBK.png',height=500,width=1200)


# #code
# ```python
# fig=px.scatter(data_athlete,x='nationality',y='weight',color='sex',hover_data=['nationality','name'])
# fig.show()
# ```

# In[ ]:


#result-,title="Women weight's average is lower as compared to men as they also have smaller heights so also smaller weights"
from IPython.display import HTML,IFrame
IFrame('https://i.imgur.com/5BKx7sd.png',height=500,width=1200)


# #code
# ```python
# fig=px.scatter(data_athlete,x='nationality',y='Final_Score',color='gold',size='gold',hover_data=['nationality','name'])
# fig.show()
# ```

# In[ ]:


#result-most medal final score(3-gold,2-silver,1-bronze...final_score=3*gold+2*silver+bronze)value 
#top player-michael phleps,katie ledecky,simone biles,katinka hosszu,danuta kozak,usain bolt,elaine thompson,simon manuel,ryan murphy,jason kenny and others
IFrame('https://imgur.com/2uHtJ6m.png',height=700,width=1200)


# In[ ]:




