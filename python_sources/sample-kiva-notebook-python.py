#!/usr/bin/env python
# coding: utf-8

# # Task 1
# 
# #### 1. Provide a summary of your interpretation of the problem statement.
# 
# The goal of this challenge is to develop a solution that would help Kiva better prioritize it's investment initiatives and allocate funding to those who are in most need.
# 
# #### 2. Provide a summary of your interpretation of what a potential solution could be.
# 
# #### 3. Identifying data sources:	
# 
# a. Identify primary data sources and provide a summary of what they contain
# 
# * **kiva_loans.csv**: Information relating to the loan including purpose, region and borrower information
# 
# * **kiva_mpi_region_locations.csv**: Localized poverty information
# 
# * **loan_theme_ids.csv**: Loan themes
# 
# * **loan_themes_by_region.csv**: Loan theme information augmented with location data
# 
# 
# b. In addition to the data provided by Kaggle, create a list of data that you think would be useful to solving the problem, and identify if this data is available in the external data sources.
# 
# * **Precipitation**: I know that precipitation is important for agriculture so this would be important. This data is available [here](https://www.kaggle.com/reubencpereira/spatial-data-repo/data)
# 
# #### 4. Choose the following
# 
# a. An feature that represents poverty or economic activity (Y) that you will intend to use
# 
# I'm going to choose the multinational poverty index *MPI* as my indicator of poverty.
# 
# b. A set of features (X) that you think would be useful in explaining Y
# 
# I believe that precipitation, EVI and population density could explain some of the variability in MPI.
# 
# # Task 2
# 
# #### 1. Identify the data sets that you will need for your analysis based on the features you selected in part 4 of Task 1
# 
# I will need the **kiva_loans.csv** and **kivaData_augmented.zip** for this analysis.
# 
# #### 2. Using either Python or R, import the data into your kernel
# 

# In[1]:


import pandas as pd

#Import data and save it as MPData
MPIData = pd.read_csv("../input/kiva-augmented-data/MPIData_augmented.csv")


# #### 3. Conduct a inspection of each imported table. This will include answering the following questions
# 
# a. What's the dimension of each data set?

# In[ ]:


#(rows,columns)
MPIData.shape


# b. What types of variables are contained in each data set?
# 
# c. How much of the data is missing?

# In[2]:


#Object = String, float* or int* = numeric
MPIData.info()


# d. Are there any invalid items or outliers?

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import numpy as np


trace0 = go.Box(
    y=MPIData["precipitation"].values.tolist()
)
data = [trace0]
py.iplot(data)


# # Task 3
# 
# #### 1. Create visualizations to display the distributions of Y and features X
# 
# #### 2. Visualize the relationships between Y and each feature in X
# 

# In[ ]:



corr = MPIData[["precipitation",'PercPoverty', 'DepvrIntensity', 
                'popDensity',  'TimeToCity',"MPI_Region",
       'AvgNightLight', 'LandClassification', 'Elevation', 'Temperature',
       'Evaporation', 'Modis_LAI', 'Modis_EVI', 'Conflicts_total']].corr()

trace = go.Heatmap(z=corr.values.tolist(),
                   x=corr.columns,
                   y=corr.columns)
data=[trace]
py.iplot(data, filename='labelled-heatmap')


# In[ ]:



trace0 = go.Histogram(
    x=MPIData["precipitation"].values.tolist(),name = "Precipitation"
)
trace1 = go.Histogram(
    x=MPIData["MPI_Region"].values.tolist(),name = "MPI"
)
trace2 = go.Histogram(
    x=np.log(MPIData["MPI_Region"]+0.0001).values.tolist(),name = "MPI (Log)"
)
data = [trace0]
py.iplot(data)
data = [trace1]
py.iplot(data)
data = [trace2]
py.iplot(data)


# In[ ]:



# Create a trace
trace = go.Scatter(
    x = MPIData["precipitation"].values.tolist(),
    y = MPIData["MPI_Region"].values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')


# # Task 4
# 
# 
# #### 1. Using the features you selected in Part 4 of Task 1,  fit a linear regression model 
# 
# #### 2. Use the model to generate predictions on the test set
# 
# #### 3. Calculate and present the accuracy of the model on the test-set
# 
# #### 4. Using the estimated parameters, summarize the relationship between the features and the output
# 

# In[3]:


import statsmodels.api as sm

MPIData_no_na = MPIData[["MPI_Region","AvgNightLight","Evaporation"]].dropna()
y = MPIData_no_na["MPI_Region"]
X = MPIData_no_na[["AvgNightLight","Evaporation"]]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[6]:


import numpy as np

y = np.log(MPIData_no_na["MPI_Region"]+.0001)
X = MPIData_no_na[["AvgNightLight","Evaporation"]]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:


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

