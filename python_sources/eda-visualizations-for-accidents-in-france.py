#!/usr/bin/env python
# coding: utf-8

# # Exploring Vehicle Accidents in France
# 
# * This notebook represents the exploration and visualization of the dataset Accidents in France [Dataset](https://www.kaggle.com/ahmedlahlou/accidents-in-france-from-2005-to-2016). 
# * This data set consists of all the recorded accidents in France since 2005 to 2016. 
# 
# 
# **Contents**
# 
# **1. Reading Data**  
# **2. Cleaning the Data**   
# **3. Exploration**    
# **3.1 Exploration based on date of accidents**  
# * Is the number of Accidents per year decreasing ? (from 2005 to 2016)
# * Which months have higher frequency of Accidents ?  
# * Which Day-of-the-Month is most safe to drive ?  
# * Time series of all accidents from 2005 to 2016
# * Time series for all accidents in each year
# 
# **3.2 Exploration based on roads where accidents occured**  
# * Which types of roads are high risk?
# * Which type of road gradient is high risk?
# 
# **3.3 Exploration based on people involved in the accidents**  
# * What was the condition of the people after the accident?
# * What was the age distribution of the people involved?
# * What was the sex distribution of the people involved?
# 
# **3.4 Exploration based on use of safety equipment**
# * What was the distribution of Safety Equipment used?
# * Did use of Safety Eqipment impact condition of people after the accident?
# 
# ## Hover the cursor on plots for more details.
# 
# 

# In[ ]:


# import the required libraries 

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import numpy as np

import plotly.plotly as py
from plotly import tools
import pandas as pd
import string, os, random
import calendar
import numpy as np
globalcolors = ['092a35', 'a2738c', '645c84', '427996', '658525', '404b69', '0f4471', '0f4471', '0f4471', '0f4471']
init_notebook_mode(connected=True)
punc = string.punctuation


# ## 1. Read the dataset
# 
# Read the dataset using pandas read_csv function. Then we join all the dataframes using *Num_Acc* attribute(de facto Accident ID) as the key to create one aggregate dataframe.

# In[ ]:


df1 = pd.read_csv('../input/caracteristics.csv', low_memory = False, encoding = 'latin-1')
df2 = pd.read_csv('../input/vehicles.csv', low_memory = False)
df3 = pd.read_csv('../input/places.csv', low_memory = False)
df4 = pd.read_csv('../input/users.csv', low_memory = False)

from functools import reduce
accidents = reduce(lambda left, right: pd.merge(left, right, on = "Num_Acc"), [df1, df2, df3, df4])


# ### 1.1 Glimpse the dataset

# In[ ]:


print("Rows: ", accidents.shape[0], "Columns: ", accidents.shape[1])


# In[ ]:


accidents.head()


# In[ ]:


accidents.columns.values


# ## 2. Cleaning the Dataset
# 
# We identify the percentage of NaN values in the dataset and eliminate the columns where the majority of the values are NaN.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total NaN Values', 'Percentage of NaN Values'])

missing_data(accidents)


# In[ ]:


accidents = accidents.drop(['v2', 'v1', 'long', 'lat', 'pr1', 'pr', 'gps'], axis = 1)
missing_data(accidents)


# ###  Is the number of accidents per year decreasing ?

# In[ ]:


# function to aggregate and return keys and values
def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values

x1, y1 = create_stack_bar_data('an', accidents)

for i in range(len(x1)):
    x1[i] += 2000

#x1 = x1[:-1]
#y1 = y1[:-1]
color1 = ['092a35']*9
color2 = ['a2738c']*3
color1.extend(color2)
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="year count", marker = dict(color = color1))
layout = dict(height=400, title='Year wise Number of Accidents in France', legend=dict(orientation="h"), 
              xaxis = dict(title = 'Year'), yaxis = dict(title = 'Number of Accidents'))
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);


# > - Number of accidents in France have steadily decreased from 2005 to 2013, but an increasing trend is observed from 2013 to 2016. 
# > - In this dataset, 2005 had the largest number of accidents equal to 374561. 
# 
# ### Which months have higher frequency of accidents ?

# In[ ]:


x2, y2 = create_stack_bar_data('mois', accidents)
#mapp = {}
#for m,v in zip(x2, y2):
#    mapp[m] = v
xn = [calendar.month_name[int(x)] for x in (x2)]
vn = y2

trace1 = go.Bar(x=xn, y=vn, opacity=0.75, name="month", marker=dict(color=globalcolors[5]))
layout = dict(height=400, title='Month Wise Number of Accidents in France', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')


# > - June, July, September, October have the highest number of accidents, while February has the lowest.
# > - On an average about 296,164 accidents occur every month in France.
# > - October has the highest number of accidents (with about 334,884 incidents) than any other month
# > - Weather in France during September and October is cold and wet whereas, June and July form the peak tourist season
# 
# ### Which Day-of-the-Month is not not safe ?

# In[ ]:


x1, y1 = create_stack_bar_data('jour', accidents)
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="monthday", marker=dict(color=globalcolors[4]))
layout = dict(height=400, title='Nummber of Accidents per Day', legend=dict(orientation="h") );

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')


# > - Number of accidents per month-day is mostly uniform. 31st is lowest beacuse only 7 months have 31 days.
# > - On an average about 114,644 accidents occur every day in France.
# 
# ### Plotting the time series of all accidents from 2005 to 2013 

# In[ ]:


accidents.an += 2000
dates = pd.to_datetime(accidents.an*10000+accidents.mois*100+accidents.jour,format='%Y%m%d')
accidents.an -= 2000
aggregated = dates.value_counts().sort_index()
x_values = aggregated.index.tolist()
y_values = aggregated.values.tolist()
x1,y1 = x_values, y_values

#x1, y1 = create_stack_bar_data('jour', accidents)
trace1 = go.Scatter(x=x1, y=y1, opacity=0.75, name="monthday", marker=dict(color='092a35'), line = dict(
        width = 0.6))
layout = dict(height=400, title='Time Series of Accidents from 2005 to 2016', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')


# ### Plotting the time series of accidents for each year

# In[ ]:


accidents.an += 2000
dates = pd.to_datetime(accidents.an*10000+accidents.mois*100+accidents.jour,format='%Y%m%d')
accidents.an -= 2000
traces = []
for key, grp in dates.groupby(dates.dt.year):
    #print(grp)
    aggregated = grp.dt.month.value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    x1,y1 = x_values, y_values
    x1 = [calendar.month_name[int(x)] for x in (x1)]


    

#x1, y1 = create_stack_bar_data('jour', accidents)
    trace1 = go.Scatter(x=x1, y=y1, opacity=0.75, line = dict(
        width = 1.5), name = str(key), marker = dict(color = np.random.randn(500)*key), mode = 'lines', 
                       text = str(key))
    layout = dict(height=400, title='Time Series of Accidents for each Year', legend=dict(orientation="h"));
    traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig, filename='stacked-bar')


# > - A sharp rise is observed in the months of June, July, September and October.
# > - Sharp drops are observed in February and August.
# > - December, 2006 has the highest number of accidents at 36,648.
# > - February, 2013 has the lowest number of accidents at 15,605.
# 
# ### Which types of roads are high-risk ?

# In[ ]:


x1, y1 = create_stack_bar_data('catr', accidents)
x1 = ['Highway', 'National Road', 'Departmental Road', 'Communal Way', 'Off-Public Network', 'Parking Lot', 'Other']
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Category", marker=dict(color='092a35'))

x2, y2 = create_stack_bar_data('circ', accidents)
x2 = ['Unknown','One Way', 'Bidirectional', 'Separated Carriageways', 'Variable Assignment Channels']
trace2 = go.Bar(x = x2, y = y2, opacity = 0.75, marker=dict(color='a2738c'), name = "Traffic Flow")

x3, y3 = create_stack_bar_data('prof', accidents)
x3 = ['Unknown', 'Dish','Slope', 'Hill-Top', 'Hill-Bottom']
trace3 = go.Bar(x = x3, y = y3, opacity = 0.75, marker=dict(color='645c84'), name = "Road Gradient")


fig = tools.make_subplots(rows = 3, cols = 1)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
layout = dict(height=900, title='Accidents by Type of Road');
fig.layout.update(layout)
#fig['layout'].update(height=800,title='Accidents by Type of Road')
iplot(fig, filename='stacked-bar')


# > - Communal Ways and Departmental Roads are riskiest with 1.6 million and 1.1 million accidents each.
# > - Biderectional Roads are by far the riskiest with over 2.1 million accidents

# In[ ]:


keydict = {1:'Highway', 2:'National Road', 3:'Departmental Road', 4:'Communal Way', 5:'Off-Public Network', 6:'Parking Lot', 9:'Other'}
roadtype = accidents[['catr','circ']]
traces = []
for key, grp in roadtype.groupby(roadtype.catr):
    aggregated = grp.circ.value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    x1,y1 = x_values, y_values
    x1 = ['Unknown','One Way', 'Bidirectional', 'Separated Carriageways', 'Variable Assignment Channels']


    

#x1, y1 = create_stack_bar_data('jour', accidents)
    trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name = keydict[key], 
                        marker = dict(color = globalcolors[int(key)-1]))
    layout = dict(height=400, title='Distribution of Accidents based on Type of Road', legend=dict(orientation="h"));
    traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig)


# ### What was the condition of people involved in the accidents ? 

# In[ ]:


keydict = {1:'Driver', 2:'Passenger', 3:'Pedestrian', 4:'Pedestrian in Motion'}
people = accidents[['catu','grav']]
traces = []
for key, grp in people.groupby(people.catu):
    aggregated = grp.grav.value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    x1,y1 = x_values, y_values
    x1 = ['Unscathed','Killed', 'Hospitalized', 'Light Injury']


    

#x1, y1 = create_stack_bar_data('jour', accidents)
    trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name = keydict[key],  
                        marker = dict(color = globalcolors[key-5]))
    layout = dict(height=400, title='Condition of People involved in the Accidents', legend=dict(orientation="h"));
    traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig)


# ### Distribution of People involved in accidents by Age

# In[ ]:


ageusers = accidents[['an_nais', 'catu']]
ageusers['age'] = 2016 - ageusers.an_nais

keydict = {1:'Driver', 2:'Passenger', 3:'Pedestrian', 4:'Pedestrian in Motion'}
traces = []
for key, grp in ageusers.groupby(ageusers.catu):
    if(key < 4):
    #aggregated = grp.an_nais.value_counts().sort_index()
        x1 = grp.age.values
    #y_values = aggregated.values.tolist()
    #x1,y1 = x_values, y_values
    #x1 = ['Driver','Passenger', 'Pedestrian', 'Pedestrian in Motion']


    
    
#x1, y1 = create_stack_bar_data('jour', accidents)
        trace1 = go.Histogram(x=x1, opacity=0.5, name = keydict[key], 
                        marker = dict(color = globalcolors[key-1]))
        layout = dict(height=400, title='Distribution of People involved in Accidents by Age', 
                  legend=dict(orientation="h"), barmode = 'overlay');
        traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig)


# ### Distribution of people invloved in accidents by Sex

# In[ ]:


keydict = {1:'Male', 2:'Female'}
people = accidents[['catu','sexe']]
traces = []
for key, grp in people.groupby(people.sexe):
    aggregated = grp.catu.value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    x1,y1 = x_values, y_values
    x1 = ['Driver','Passenger', 'Pedestrian', 'Pedestrian in Motion']


    

#x1, y1 = create_stack_bar_data('jour', accidents)
    trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name = keydict[key],  
                        marker = dict(color = globalcolors[key-5]))
    layout = dict(height=400, title='Distribution of people involved in accidents by Sex', legend=dict(orientation="h"));
    traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig)


# ### What was the Safety Equipment used by the people involved ?

# In[ ]:


safety = accidents[['secu', 'grav']]
safety = safety.dropna()
safety['equipment'] = (safety.secu/10).astype(int)
safety.secu = (safety.secu - safety.equipment*10).astype(int)


x1, y1 = create_stack_bar_data('equipment', safety)
x1 = ['Belt', 'Helmet', "Children's Device", 'Reflective Equipment', "Other"]
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, marker=dict(color='092a35'))
layout = dict(height=400, title='Distribution of Safety Equipment', legend=dict(orientation="h") );

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')


# ### Did Safety Equipment have any impact on the condition of the people involved ?

# In[ ]:


keydict = {1:'Unscathed', 2:'Killed', 3: 'Hospitalized', 4: 'Light Injury'}
traces = []
for key, grp in safety.groupby(safety.grav):
    if (key != 0):
        count = safety.secu.count()
        #print(count)
        aggregated = (grp.secu.value_counts()).sort_index()
        x_values = aggregated.index.tolist()
        y_values = (aggregated.values/safety.secu.value_counts().sort_index().values*100).tolist()
        x1,y1 = x_values[1:], y_values[1:]
        x1 = ['Equipment Present','Equipment Absent', 'Not Determined']


    

#x1, y1 = create_stack_bar_data('jour', accidents)
        trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name = keydict[key],  
                        marker = dict(color = globalcolors[key-2]))
        layout = dict(height=400, title='Relationship between Safety Equipment and Severity of Accident', 
                      legend=dict(orientation="h"), barmode = 'stack', yaxis = dict(title = 'Percentage'),
                      xaxis = dict(title = 'Safety Equipment'));
        traces.append(trace1)
fig = go.Figure(data= traces, layout=layout)
iplot(fig)


# > - 1.9% of the people were killed and 17.8% were hospitalized if Safety Equipment was used.
# > - 10.7% of the people were killed and 35.8% were hospitalized if Safety Equipment was not used.

# In[ ]:




