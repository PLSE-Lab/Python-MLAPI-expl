#!/usr/bin/env python
# coding: utf-8

# # Daily Weather Data Analysis
# 
# In this notebook, using decision tree we are trying to predict whether it will rain at for particular day at 3pm or not based on daily weather data of 9am.
# 
# 

# In[ ]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


data = pd.read_csv('../input/daily_weather.csv')


# In[ ]:


data.columns


# 
# 
# * **number:** unique number for each row
# * **air_pressure_9am:** air pressure averaged over a period from 8:55am to 9:04am (*Unit: hectopascals*)
# * **air_temp_9am:** air temperature averaged over a period from 8:55am to 9:04am (*Unit: degrees Fahrenheit*)
# * **air_wind_direction_9am:** wind direction averaged over a period from 8:55am to 9:04am (*Unit: degrees, with 0 means coming from the North, and increasing clockwise*)
# * **air_wind_speed_9am:** wind speed averaged over a period from 8:55am to 9:04am (*Unit: miles per hour*)
# * ** max_wind_direction_9am:** wind gust direction averaged over a period from 8:55am to 9:10am (*Unit: degrees, with 0 being North and increasing clockwise*)
# * **max_wind_speed_9am:** wind gust speed averaged over a period from 8:55am to 9:04am (*Unit: miles per hour*)
# * **rain_accumulation_9am:** amount of rain accumulated in the 24 hours prior to 9am (*Unit: millimeters*)
# * **rain_duration_9am:** amount of time rain was recorded in the 24 hours prior to 9am (*Unit: seconds*)
# * **relative_humidity_9am:** relative humidity averaged over a period from 8:55am to 9:04am (*Unit: percent*)
# * **relative_humidity_3pm:** relative humidity averaged over a period from 2:55pm to 3:04pm (*Unit: percent *)
# 

# In[ ]:


data.head()


# In[ ]:


data[data.isnull().any(axis=1)]


# We will not need to number for each row so we can clean it.

# In[ ]:


del data['number']


# Now let's drop null values using the *pandas dropna* function.

# In[ ]:


before_rows = data.shape[0]
print(before_rows)


# In[ ]:


data = data.dropna()


# In[ ]:


after_rows = data.shape[0]
print(after_rows)


# <p style="font-family: Arial; font-size:1.25em;color:purple; font-style:bold"><br>
# How many rows dropped due to cleaning?<br><br></p>

# In[ ]:


before_rows - after_rows


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold">
# Convert to a Classification Task <br><br></p>
# Binarize the relative_humidity_3pm to 0 or 1.<br>
# Here higher humidity level is taken anything more than 25%, but it depends from city to city. For example 25% humidity in New Delhi is quite high and therefore more posibility of rain but for city like Mumbai 60% is anual average humidity level. So for Mumbai something like 85% or more can cause rain.
# 

# In[ ]:


clean_data = data.copy()
humidity_level=24.99
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > humidity_level)*1
print(clean_data['high_humidity_label'])


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Target is stored in 'y'.
# <br><br></p>
# 

# In[ ]:


y=clean_data[['high_humidity_label']].copy()
y


# In[ ]:


clean_data['relative_humidity_3pm'].head()


# In[ ]:


y.head()


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# 
# Use 9am Sensor Signals as Features to Predict Humidity at 3pm
# <br><br></p>
# 

# In[ ]:


morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']


# In[ ]:


X = clean_data[morning_features].copy()


# In[ ]:


X.columns


# In[ ]:


y.columns


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# Perform Test and Train split
# <br><br></p>
# 
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# In[ ]:


print(type(X_train),
type(X_test),
type(y_train),
type(y_test))


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# Fit on Train Set
# <br><br></p>

# In[ ]:


humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)


# In[ ]:


type(humidity_classifier)


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# Predict on Test Set 
# <br><br></p>

# In[ ]:


predictions = humidity_classifier.predict(X_test)


# In[ ]:


predictions[:10]


# In[ ]:


y_test['high_humidity_label'][:10]


# <p style="font-family: Arial; font-size:1.75em;color:purple; font-style:bold"><br>
# Measure Accuracy of the Classifier
# <br><br></p>

# In[ ]:


accuracy_score(y_true = y_test, y_pred = predictions)

