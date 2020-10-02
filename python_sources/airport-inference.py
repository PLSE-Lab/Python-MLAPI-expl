#!/usr/bin/env python
# coding: utf-8

# In this kernel, I will try to 1) Visualize the location of airports worldwide and 2) compare this with population density to see which population is best served.       

# ## Imports, reads, and basic housekeeping

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.graph_objs import *

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# In[ ]:


names = ['id',
'Name' ,
'City' ,
'Country',
'Airport ID',
'IATA' ,
'lat', 
'lon' ,
'Altitude',
'Timezone' ,
'DST' ,
'Tz' ,
'Type',
'Source']


# In[ ]:


stations = pd.read_csv('../input/airports-extended.csv', names = names)


# In[ ]:


stations.head(2)


# Subset so that we only have stations of each type to differentiate.

# In[ ]:


airports = stations[stations['Type'] =='airport']
train_stations = stations[stations['Type'] =='station']
ferries = stations[stations['Type'] =='port']
other = stations[stations['Type'] =='unknown']


# In[ ]:


airports.shape


# In[ ]:


train_stations.shape


# There are over 7K airports but only 1422 train stations. I don't imagine that represents the global truth...just what we have here in this OpenFlights data. Anyways, let's do a quick search regarding geographic distribution.

# In[ ]:


airports.Country.value_counts().head()


# In[ ]:


# Find counts of airports by country.
airports_country_counts = airports.Country.value_counts()
airports_country_counts_df = pd.DataFrame(airports_country_counts).reset_index()
airports_country_counts_df.columns = ['Country', 'Count']
airports_country_counts_df.tail()


# In[ ]:


# Do the same for train stations:
train_country_counts = train_stations.Country.value_counts()
train_country_counts_df = pd.DataFrame(train_country_counts).reset_index()
train_country_counts_df.columns = ['Country', 'Count']
train_country_counts_df.head()


# ## Station counts by country

# In[ ]:


number_countries_to_show = 10

trace1 = go.Bar(
    x=airports_country_counts_df[0:number_countries_to_show].Country,
    y=airports_country_counts_df[0:number_countries_to_show].Count,
    name='Airports'
)
trace2 = go.Bar(
    x=airports_country_counts_df[0:number_countries_to_show].Country,
    y=train_country_counts_df[0:number_countries_to_show].Count,
    name='Train stations'
)

data = [trace1, trace2]
layout = go.Layout(
    title = 'Airports and Train Station Counts <br> (by top 10 airport nations)',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)


# As expected, most stations are in the US (it's a big and rich nation). For fun, let's keep comparing transit modes and create a variable of the ratio of airports : train stations and see if we find any outliers.

# In[ ]:



counts_df = pd.merge(train_country_counts_df, airports_country_counts_df, 
             left_on='Country', 
             right_on = 'Country',
             how='inner')
counts_df.columns = ['Country','Trains','Airports']
counts_df['train/plane_ratio'] = counts_df['Trains']/counts_df['Airports']
counts_df.sort_values(by='train/plane_ratio')[0:10]                                                            


# In[ ]:


trace1 = go.Bar(
    x=counts_df.Country,
    y=counts_df['train/plane_ratio'],
    name='Ratio'
)


data = [trace1]
layout = go.Layout(
    title = 'Airports and Train Station Counts <br> (by top 10 airport nations)',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)


# Kinda cool...small countries have high ratios. But this again makes me question the data. I think we are not looking at all train stations, but we are looking at all airports. :( for incomplete data. 

# ## Geo-plot stations by transit type

# In[ ]:


conditions = [
    (stations['Type'] == 'airport'),
    (stations['Type'] == 'station'),
    (stations['Type'] == 'port'),
    (stations['Type'] == 'unknown')]

choices = ['blue', 'red', 'green','black']
stations['color'] = np.select(conditions, choices, default='black')


# In[ ]:


data = [ dict(
        type = 'scattergeo',
#         locations = country_counts_df['Country'],
        lat = stations.lat,
        lon = stations.lon,
#         z = airports['lon'],
        text = stations['Country']+stations['Name'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = True,
        reversescale = True,
        marker = dict(
            color = stations.color
        )
         )]

layout = dict(
    title= 'Locations of stations, colored by type <br> (zoom and scroll to adjust view)',
    geo = dict(
        scope='usa',
        showframe = True,
        showcoastlines = True,
        showcountries=True,
        projection = dict(
            type = 'equirectangular'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )


# To note...it's pretty cool thatthe red lines clearly track a line from West to East across the American plains, seemingly indicating one main train line. This (sensing a pattern from location) makes me wonder about the black (unknown) dots. Do you think we can do machine learning and identify correctly which type of station each black dot is? Let's see.

# ## Predict Unknown stations

# Since most stations in this dataset are airports, the imbalanced nature of the data would lead us to naively believe that all unknown stations are also airports.  But that's likely not 100% true. Let's see if an ML model can be less naive. 

# Let's first make training, validation, and prediction (unknown labels) set. 

# In[ ]:


# drop color variable from before:
stations = stations.drop('color', 1)
# split into known and unknown station type
known = stations[stations['Type'] !='unknown']
prediction = stations[stations['Type'] =='unknown']


# In[ ]:


known.head()


# It's probably also unfair to have OurAirports be the source, since that'll be too easy to predict airports in that way. Similarly, many of the unknown station types have 'User" as source. We'll strip source from the data. In fact, let's strip away everything other than just lat, lon, and elevation and see what happens?!

# In[ ]:


training_parent = known[['lat','lon','Altitude','Type']]
prediction_parent = prediction[['lat','lon','Altitude']]


# Split off validation data:

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(training_parent[['lat','lon','Altitude']], 
                                                    training_parent[['Type']],
                                                    test_size=0.33, random_state=42)


# In[ ]:


print('Train shape is: ',x_train.shape,' and the shape of the test set is: ',x_test.shape) 


# In[ ]:


# convert to np array
np_x_train = x_train.values
np_y_train = y_train.values


# In[ ]:


# train
model = XGBClassifier()
model.fit(np_x_train, np_y_train.ravel())
print(model)


# In[ ]:


# predict
y_pred = model.predict(x_test.values)


# In[ ]:


y_pred_df = pd.Series(y_pred)


# In[ ]:


print('Predictions:',y_pred_df.value_counts())
print('Actual test set distribution:', y_test.Type.value_counts())


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# Even though this seems high, the training set had mostly airports, so always predicting airport in the test set would have resulted in (2594 / 3061 =) 84.6%, so actually our model isn't really learning anything other than guessing. This probably makes sense. I think we'd need to parse out some more features for the decision tree to then find anything meaningful. 

# In[ ]:


# predict on the unknowns
prediction_final = model.predict(prediction_parent.values)


# In[ ]:


fps = pd.Series(prediction_final)
end_result = prediction.reset_index()
end_result['prediction'] = fps
end_result.head(50)


# In[ ]:


end_result[end_result['prediction'] == 'station'].head()


# Do any of these stations seem like train stations instead of airports? E.g. did we pick right?

# ## Other to dos:
# 
# - Find centroids? Are these places really transit hubs?

# In[ ]:





# In[ ]:




