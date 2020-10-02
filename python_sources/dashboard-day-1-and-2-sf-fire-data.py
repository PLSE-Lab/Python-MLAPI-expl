#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all the necessary packages
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()


# In[ ]:


#read the data
data = pd.read_csv("../input/fire-department-calls-for-service.csv")


# Since the dataframe is huge, 10000 records are sampled from it. As the purpose is the visualization of important features, working with a smaller dataframe retains the meaningful characteristics present in the data and makes the job simpler. 
# 
# The dataframe now contains only 4 columns of relevant data.

# In[ ]:


df = data.sample(10000)
print(df.head())

df = df[['Response DtTm', 'On Scene DtTm', 'Box', 'Number of Alarms']]


# Since the dataframe is huge, 10000 records are sampled from it. As the purpose is the visualization of important features, working with a smaller dataframe retains the meaningful characteristics present in the data and makes the job simpler. 
# 
# The dataframe now contains only 4 columns of relevant data.

# In[ ]:


print(df.count())

df = df.dropna().reset_index()

print(df.count())


# The data is cleaned. All the null values are dropped. This reduces the length of the dataframe to 7396.

# In[ ]:


box = df.groupby('Box').size()

plt.scatter(box.index, box)
plt.xlabel("Box No.")
plt.ylabel("Count of incidents")
plt.title("Scatterplot showing counts of incidents in each box")
plt.show()


# The total no. of incidents that occurred in each box is plotted.  

# In[ ]:


# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=box.index, y=box, mode='markers')]

# specify the layout of our figure
layout = dict(title = "Scatterplot showing counts of incidents in each box",
              xaxis= dict(title= 'Box No.'), yaxis= dict(title= 'Count of incidents')  )

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


print(np.mean(box))

risk = box[box > 5]
print(len(risk))


# On an average, the no. of incidents  in each box stays below 5.
# 
# The boxes where the no. of incidents that took place is more than 10 are marked as high risk.

# One useful feature to note will be the time taken to reach on scene after getting the information.
# 
# Reducing that time difference, specially in high risk areas will improve the safety. 

# In[ ]:


df['Time Diff'] = 0

for i in range(len(df)):
    dt1 = parser.parse(df['On Scene DtTm'][i])
    dt2 = parser.parse(df['Response DtTm'][i])
    td = (dt1 - dt2).total_seconds()
    df['Time Diff'][i] = td/60
    


# Those fire incidents where the no. of alarms is 1 are considered as they account for more than 90% of the data.
# 
# 

# In[ ]:


d_ = df[df["Number of Alarms"] == 1]

plt.scatter(d_["Box"],d_["Time Diff"])
plt.plot(risk, "ro")
plt.xlabel("Box No.")
plt.ylabel("Time taken to respond(in mins.)")
plt.title("Scatterplot showing time taken to respond(in mins.) vs box no.")
plt.show()


# In[ ]:


trace1 = go.Scatter(
   x = d_["Box"],
   y = d_["Time Diff"],
   mode = 'markers', 
   name = 'low risk boxes' 
   )

trace2 = go.Scatter(
   x = risk.index,
   y = risk,
   mode = 'markers',
   name = 'high risk boxes' 
   )

layout = dict(title = "Scatterplot showing time taken to respond(in mins.) vs box no.",
              xaxis= dict(title= 'Box No.'), yaxis= dict(title= 'Time taken to respond(in mins.)')  )

data = [trace1, trace2]
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:




