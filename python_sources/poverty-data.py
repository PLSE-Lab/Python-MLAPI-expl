#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# **Load Data**

# In[2]:


Aug_Data = pd.read_csv("../input/mpidata-augmented/MPIData_augmented.csv")
Aug_Data.head()


# In[3]:


Aug_Data.shape


# In[4]:


Aug_Data['Sub-national region'].head()


# In[5]:


Aug_Data.info()


# In[6]:


Aug_Data['soil_clay'].describe()


# In[7]:


Aug_Data.describe()


# In[8]:


Aug_Data['PercPoverty'].nunique(), # Number of unique entries


# In[9]:


Aug_Data.nunique(), # Number of unique entries for features


# In[10]:


len(Aug_Data.index)-Aug_Data.count(), # Number of missing entries


# In[11]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

trace0 = go.Box(y=Aug_Data["soil_clay"].values.tolist())
trace1 = go.Box(y=Aug_Data["soil_sand"].values.tolist())
clay_data = [trace0]
sand_data = [trace1]
py.iplot(clay_data)
py.iplot(sand_data)


# In[12]:


corr = Aug_Data[["precipitation",'PercPoverty', 'DepvrIntensity', 
                'popDensity',  'TimeToCity',"MPI_Region",
       'AvgNightLight', 'LandClassification', 'Elevation', 'Temperature',
       'Evaporation', 'Modis_LAI', 'Modis_EVI', 'Conflicts_total']].corr()

trace = go.Heatmap(z=corr.values.tolist(),
                   x=corr.columns,
                   y=corr.columns)
data=[trace]
py.iplot(data, filename='labelled-heatmap')


# In[13]:


from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)

trace0 = go.Histogram(
    x=Aug_Data["precipitation"].values.tolist(),name = "Precipitation"
)
trace1 = go.Histogram(
    x=Aug_Data["MPI_Region"].values.tolist(),name = "MPI"
)
trace2 = go.Histogram(
    x=np.log(Aug_Data["MPI_Region"]+0.0001).values.tolist(),name = "MPI (Log)"
)

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('Precipitation', 'MPI', 'MPI (Log)'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)

fig['layout'].update(title='Precipitation and MPI Plots')

py.iplot(fig)


# In[14]:


# Create a trace
trace = go.Scatter(
    x = Aug_Data["Conflicts_total"].values.tolist(),
    y = Aug_Data["MPI_Region"].values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')


# > ## Modelling
# 1. Using the features you selected in Part 4 of Task 1, fit a linear regression model
# 2. Use the model to generate predictions on the test set
# 3. Calculate and present the accuracy of the model on the test-set
# 4. Using the estimated parameters, summarize the relationship between the features and the output

# In[15]:


import statsmodels.api as sm

Aug_Data_no_na = Aug_Data[["MPI_Region","precipitation","Modis_EVI"]].dropna()
y = Aug_Data_no_na["MPI_Region"]
X = Aug_Data_no_na[["precipitation","Modis_EVI"]]/1000

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[16]:


y = np.log(Aug_Data_no_na["MPI_Region"]+.0001) # We added .0001 to the MPI_Region because there are zeroes in the data and log(0) is undefined
X = Aug_Data_no_na[["precipitation","Modis_EVI"]]/1000

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[17]:


# Create a trace
res = predictions - y

trace = go.Scatter(
    y = res.values.tolist(),
    x = predictions.values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')


# In[18]:


New_Aug_Data_no_na = Aug_Data[["MPI_Region","Conflicts_total", "popDensity", "TimeToCity","precipitation","Evaporation","Modis_LAI","Modis_EVI"]].dropna()
y = np.log(New_Aug_Data_no_na["MPI_Region"]+.0001) # We added .0001 to the MPI_Region because there are zeroes in the data and log(0) is undefined
X = New_Aug_Data_no_na[["Conflicts_total", "popDensity", "TimeToCity","precipitation","Evaporation","Modis_LAI","Modis_EVI"]]/1000

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[19]:


# Create a trace
res = predictions - y

trace = go.Scatter(
    y = res.values.tolist(),
    x = predictions.values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')

