#!/usr/bin/env python
# coding: utf-8

# Inspired by [Pedro Lealdino Filho's ](http://https://twitter.com/pedrolealdino)articles on Medium, [How to visualize high frequency financial data using Plotly and R,](http://https://towardsdatascience.com/how-to-visualize-high-frequency-financial-data-using-plotly-and-r-97171ae84be1) and [The easy way to predict stock prices using machine learning](http://https://medium.com/@pedrolealdino/the-stupidly-easy-way-to-predict-stock-prices-using-machine-learning-dbb65873cac8), which show us how to plot candlesticks using plotly in R, and how to make predictions using H2O.ai (also in R), I decided to replicate his steps in Python, hoping that market enthusiats more versed in Python can benefit from Pedro's contributions.

# First we load all the libraries required.

# In[ ]:


import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np
import h2o
from h2o.automl import H2OAutoML


# Assuming we have downloaded the asset's data into a csv file called 'input.csv', we read the data from the file. When I downloaded the data from Metatrader, the csv file was enconded in utf-16, so when I tried to load the info, it appeared in hex. That's why I added the 'encoding = "utf-16" ' parameter.

# In[ ]:


file = open('../input/input.csv', 'r', encoding = "utf-16") #so the data won't be imported in utf-16

data = []
for line in file:
    
    data1 = line.split(',')
    if len(data1) == 7:
        data.append([datetime.strptime(data1[0], '%Y.%m.%d %H:%M'),         
                    float(data1[1]),
                    float(data1[2]), float(data1[3]),
                    float(data1[4]), int(data1[5]),
                    int(data1[6])])

file.close()


# In[ ]:





# We move the data to panda dataframe and add the names of the columns.

# In[ ]:


df = pd.DataFrame(data)
df.columns = ['DATE', 'OPEN', 'HIGH',
              'LOW', 'CLOSE', 'TICKVOL', 'VOL'
             ]


# Now it's time to use the plotly library to create the candlestick chart.

# In[ ]:


fig = go.Figure(data=[go.Candlestick(x=df['DATE'],
                open=df['OPEN'], high=df['HIGH'],
                low=df['LOW'], close=df['CLOSE'])
                     ])

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# At a first glance the candlesticks are barely perceived. You can click into the graph and select a section in order to zoom in. However, just for the sake of illustration I decided to plot the last rows of the dataframe.

# In[ ]:


cola = df.tail()


# In[ ]:


fig = go.Figure(data=[go.Candlestick(x=cola['DATE'],
                open=cola['OPEN'], high=cola['HIGH'],
                low=cola['LOW'], close=cola['CLOSE'])
                     ])

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# Now we proceed with the code for Pedro's post "The easy way to predict stock prices using machine learning"

# First we create a column named "SHIFTED" in our dataframe that will have the closing price for the next row.

# In[ ]:


# Let's add a new column "shifted" to the dataframe and populate it with the 
# closing value for the posterior row

shifted = df["CLOSE"][1:].tolist()
shifted.append(np.nan) # adds a NaN value so the list will be of same length as df
df["SHIFTED"] = shifted


# Since the last row will have an NaN value in the "SHIFTED" column, we get rid of it (if there where any other rows with NaN values, they would also be deleted).

# In[ ]:


df = df.dropna()


# And since the "DATE" and "TICKVOL" vaues will not be used to train the model, we get rid of them as well.

# In[ ]:


df = df.drop(['DATE', 'TICKVOL'], axis = 1)


# Now we export the contents of our dataframe to a csv called salida.csv.

# In[ ]:


df.to_csv(r'salida.csv', index = False)


# This is the moment to initiate the H2O instance. This was a bit tricky. 
# First, bear in mind that H2O is not supported on Python 3.7 yet, so I had to create another environment with Python 3.5 in order to install the library.
# 
# Also, in order to initialize the H2O instance you must have a Java JRE installed (I know that for most users this is a given, but I had recently acquired a new laptop and had not needed to install Java).
# 
# Furthermore, the H2O instance uses a .jar file that may be blocked by your firewall, which was my case. I had to locate the .jar file, doubleclick on it and explicitly let my firewall recognize it as a safe file.

# In[ ]:


h2o.init(nthreads = -1, max_mem_size = "16g")


# Once the H2O instance is running, we import the file we had previously exported using h2o's import_file method/function/whatever.

# In[ ]:


info_df = h2o.import_file("salida.csv")


# As Pedro's article does, we preview the contents of the frame created.

# In[ ]:


info_df.describe(chunk_summary=True)


# And now we split the data into training and testing sets.

# In[ ]:


parts = info_df.split_frame(ratios=[.8])
train = parts[0]
test = parts[1]


# We decide which will be our predicted y value ("SHIFTED") and which w ill be our predictors x (the other values, OPEN, HIGH, CLOSE, LOW, VOL). 

# In[ ]:


y = "SHIFTED"
x = info_df.columns
x.remove(y)


# Now we take advantage of H2O's Automodel feature, which tests different models and decides which performs better.

# In[ ]:


automodel = H2OAutoML(max_models=20, seed=1)
automodel.train(x=x, y=y, training_frame=train)


# And let's see the leader's results.

# In[ ]:


automodel.leader


# Finally, lets make some predictions using the leader model and our test set.

# In[ ]:


predictions = automodel.leader.predict(test)


# Finally, let's see those predictions.

# In[ ]:


predictions


# In[ ]:




