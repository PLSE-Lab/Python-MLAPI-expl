#!/usr/bin/env python
# coding: utf-8

# # Dashboarding with Jupyter Notebooks
# 
# By [Elloa B. Guedes](http://www.github.com/elloa)
# 
# * Working with data from [NY Bus Breakdown and Delays](https://www.kaggle.com/new-york-city/ny-bus-breakdown-and-delays)
# 

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# to use the csvvalidator package, you'll need to 
# install it. Turn on the internet (in the right-hand
# panel; you'll need to have phone validated your account)

import sys
get_ipython().system('{sys.executable} -m pip install csvvalidator')


# Considering the data available in this dataset, I've decided to dashboard the following information:
# 
# - Number of bus breakdowns per school-year
# - Proportion of reasons for bus breakdowns per year
# - Number of students affected daily
# 
# The following attributes are important in our context:
# 
# 1. **School_Year**: Indicates the school year the record refers to. The DOE school year starts in September every year.  
# 2. **Reason**: Reason for delay as entered by staff employed by reporting bus vendor. User chooses from the following categories:  
#   2.1 Accident  
#   2.2 Delayed by School
#   2.3 Flat Tire  
#   2.4 Heavy Traffic  
#   2.5 Mechanical Problem  
#   2.6 Other   
#   2.7 Problem Run  
#   2.8 Weather Conditions   
#   2.9 Won't Start  
# 3. **Number_Of_Students_On_The_Bus**: Number of students on the bus at the time of the incident as estimated by the staff employed
# 4. **Occurred_On**: timestamp

# In[ ]:


df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
df.head(10)


# In[ ]:


df = df[["School_Year","Reason","Number_Of_Students_On_The_Bus","Occurred_On"]]
df.head(7)


# There is some problem on data:
# 
# 1. _Missing values_: If any entry is missing, the example is going to be droped
# 2. _Convert to time series_: Values of Occured on

# In[ ]:


df.dropna(inplace=True)
df.head(12)


# In[ ]:


df['Occurred_On'] = pd.to_datetime(df['Occurred_On'])
df.head(15)


# In[ ]:


len(df)


# In[ ]:


### Validating

# import everything from the csvvalidator package
from csvvalidator import *

# Specify which fields (columns) your .csv needs to have
# You should include all fields you use in your dashboard
field_names = ("School_Year","Reason","Number_Of_Students_On_The_Bus","Occurred_On")

# create a validator object using these fields
validator = CSVValidator(field_names)

# write some checks to make sure specific fields 
# are the way we expect them to be
validator.add_value_check("School_Year", # the name of the field
                          str, 
                          'EX1', # code for exception
                          'School_Year invalid'# message to report if error thrown
                         )
validator.add_value_check("Reason", 
                          # check for a date with the sepcified format
                          str, 
                          'EX2',
                          'Reason'
                         )
validator.add_value_check('Number_Of_Students_On_The_Bus',
                          # makes sure the number of units sold is an integer
                          int,
                          'EX3',
                          'Number_Of_Students_On_The_Bus invalid'
                         )
validator.add_value_check("Occurred_On", 
                          str,
                          'EX4', 
                          'Occurred_On" invalid')

results = validator.validate(df)


# In[ ]:


lines_remove = []
for di in results:
    lines_remove.append(di['row'])
    
df.drop(df.index[lines_remove],inplace=True)
    


# In[ ]:


len(df)


# ## Visualizing: Number of bus-breakdown per school-year
# 
# 1. Organizing data
# 2. Formating beautiful plot

# In[ ]:


newdf = pd.DataFrame(df["School_Year"].value_counts())
newdf.rename(index=str, columns={"School_Year": "Bus Breakdowns"},inplace=True)
newdf.index.name = "Year"
newdf.sort_index(ascending=True,inplace=True)
newdf


# In[ ]:


from matplotlib.pyplot import figure

newdf.plot.bar(align='center', alpha=0.8,color='blue')
plt.title("Counting the number of bus breakdowns per school year")
plt.show()


# ## Visualizing with plotly

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# copying df to insert new index and make manipulation easier
newdf2 = newdf.copy()
newdf2.reset_index(level=0, inplace=True)

data = [
    go.Bar(
        x=newdf2['Year'], # assign x as the dataframe column 'x'
        y=newdf2['Bus Breakdowns']
    )
]

# specify the layout of our figure
layout = dict(title = "Number of Bus Breakdowns per School Year",
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)


# # Visualizing: Proportion of reasons for bus breakdowns per year
# 
# 1. Organizing data
# 2. Plotting data

# In[ ]:


df2 = df[["School_Year","Reason"]]


for ano in df2["School_Year"].unique():
    f, axes = plt.subplots(figsize=(8,8))
    dados = df2.loc[df["School_Year"] == ano]
    dados = pd.DataFrame(dados["Reason"].value_counts())
    
    total = sum(dados["Reason"])
    novo = [x/total for x in dados["Reason"]]
    dados["Fraction"] = novo

    axes.pie(dados["Fraction"],labels=dados.index, autopct='%.2f')
    plt.title("Year "+ str(ano))
    plt.show()
    plt.close('all')
    
    
    


# ## Visualizing pie chart with plotly

# In[ ]:


## organizing data
df3 = df[["School_Year","Reason"]]
df3.reset_index(level=0, inplace=True)




year = "2015-2016"
df3.loc[df3["School_Year"]== year]

dados = pd.DataFrame(df3["Reason"].value_counts())

total = sum(dados["Reason"])
novo = [x/total for x in dados["Reason"]]
dados["Fraction"] = novo
dados.reset_index(level=0, inplace=True)
dados["Reason"] = dados["index"]
dados.drop(["index"],axis=1,inplace=True)
dados.head()


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()



data = [
    go.Pie(
        labels=dados["Reason"],
        values=dados["Fraction"]
    )
]

# specify the layout of our figure
layout = dict(title = "Reasons of Bus Breakdowns")

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)




# # Visualizing: Number of students affected daily
# 
# 1. Organizing data
# 2. Set interval: Days in 2018
# 3. Visualize data

# In[ ]:


interest = df.loc[df['School_Year'] == '2018-2019']
interest.set_index(pd.to_datetime(interest["Occurred_On"]),inplace=True)
interest.drop(["School_Year","Reason","Occurred_On"],axis = 1, inplace= True)
interest.head(10)


# In[ ]:


interest = interest['Number_Of_Students_On_The_Bus'].resample('D').sum()
interest.head(10)


# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(interest)
plt.gcf().autofmt_xdate()
plt.title("Students affected per day in 2018-2019")
plt.show()


# ## Visualizing with plotly

# In[ ]:


## organizing data
newdf = interest.to_frame()
newdf.reset_index(level=0, inplace=True)
newdf.head(4)


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

trace1 = go.Scatter(
    x = newdf["Occurred_On"],
    y = newdf["Number_Of_Students_On_The_Bus"],
    mode = 'lines+markers',
    name = 'lines+markers'
)

# specify the layout of our figure
layout = dict(title = "Students Affected by Day")

# create and show our figure
fig = dict(data = [trace1], layout = layout)
iplot(fig)


# 

# 
