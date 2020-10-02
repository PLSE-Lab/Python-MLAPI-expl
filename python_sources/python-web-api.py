#!/usr/bin/env python
# coding: utf-8

# **What is an API?**
# 
# An API (Application Programming Interface) is something that any particular website can design to this thing called an API to give out their data and allow your web application to communicate with that data. Facebook, Twitter, Yelp, and many other services rely and have their own API's.
# 
# With APIs, applications talk to each other without any user knowledge or intervention.
# 
# When we want to interact with an API in Python (like accessing web services), we get the responses in a form called JSON.

# **What is JSON?**
# 
# JSON (JavaScript Object Notation) is a compact, text based format for computers to exchange data and is once loaded into Python just like a dictionary.
# 
# JSON data structures map directly to Python data types, which makes this a powerful tool for directly accessing data.
# 
# This makes JSON an ideal format for transporting data between a client and a server.

# **JSON vs Dictionary**
# 
# It is apples vs. oranges comparison:
# 
# JSON is a data format (a string).
# 
# Python dictionary is a data structure (in-memory object).

# **How to Query a JSON API in Python**
# 
# * Import Library Dependencies
# * Create Query URL (which contains the url, apikey, and query filters)
# * Perform a Request Call & Convert to JSON
# * Extract Data from JSON Response (Query it)

# **TASK**
# 
# Get weather information from the city of Los Angeles

# **Create an API Key**
# 
# Just register on the Sign up page https://home.openweathermap.org/users/sign_up
# 
# Get your unique API key on your personal page

# In[ ]:


## Step 1. Import Library Dependencies
# Dependencies
import requests as req
import json
import pandas as pd
##print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Create Query URL**
# 
# The "?" syntax is used in the begginning of our query string so we can start building a filterized version of our data
# 
# The "&" syntax is used to perform our different types of queries, in this case one for city and units

# In[ ]:


# A. Get our base URL for the Open Weather API
base_url = "http://api.openweathermap.org/data/2.5/weather"

# B. Get our API Key 
key = "c703c966f9be8a0c4869b86832a0898f"

# C. Get our query (search filter)
query_city = "Los Angeles"
query_units = "metric"

# D. Combine everything into our final Query URL
query_url = base_url + "?apikey=" + key + "&q=" + query_city + "&units=" + query_units

# Display our final query url
query_url


# **Perform a Request Call & Convert to JSON**
# 
# **Step 1**. **Perform a Request Call** : Using our req.get() method, we'll get back a response from our Weather Map API with the filtered queries.

# In[ ]:


# Perform a Request Call on our search query
response = req.get(query_url)
response


# **Step 2**. **Convert to JSON**: Then just call the .json() at the end to convert it into a JSON format (aka dictionary)

# In[ ]:


response = response.json()
response


# In[ ]:


# Using json.dumps() allows you to easily read the response output
print(json.dumps(response, indent=4, sort_keys=True))


# **Extract Data from JSON Response**
# 
# Finally, we're able to access our JSON object and extract information from it just as if it was a Python Dictionary.
# 
# A JSON object contains a key-value pair.

# In[ ]:


# Extract the temperature data from our JSON Response
temperature = response['main']['temp']
print ("The temperature is " + str(temperature))

# Extract the weather description from our JSON Response
weather_description = response['weather'][0]['description']
print ("The description for the weather is " + weather_description)


# **How to Perform Multiple API Calls**

# In[ ]:


# A. Get our base URL for the Open Weather API
base_url = "http://api.openweathermap.org/data/2.5/weather"

# B. Get our API Key 
key = "c703c966f9be8a0c4869b86832a0898f"

# C. Create an empty list to store our JSON response objects
weather_data = []

# D. Define the multiple cities we would like to make a request for
cities = ["London", "Paris", "Las Vegas", "Stockholm", "Sydney", "Hong Kong"]

# E. Read through each city in our cities list and perform a request call to the API.
# Store each JSON response object into the list
for city in cities:
    query_url = base_url + "?apikey=" + key + "&q=" + city
    weather_data.append(req.get(query_url).json())


# In[ ]:


# Now our weather_data list contains 6 different JSON objects for each city
# Print the first city (London) 
weather_data


#  **Extract Multiple Queries and Store in Pandas DataFrame**
# 
# **Using For Loop**

# In[ ]:


# Create an empty list for each variable
city_name = []
temperature_data = []
weather_description_data = []

# Extract the city name, temperature, and weather description of each City
for data in weather_data:
    city_name.append(data['name'])
    temperature_data.append(data['main']['temp'])
    weather_description_data.append(data['weather'][0]['description'])

# Print out the list to make sure the queries were extracted 
print ("The City Name List: " + str(city_name))
print ("The Temperature List: " + str(temperature_data))
print ("The Weather Description List: " + str(weather_description_data))


#  **Extract Multiple Queries and Store in Pandas DataFrame**
#  
#  **Using List Comprehension**

# In[ ]:


# Extract the city name, temperature, and weather description of each City
city_name = [data['name'] for data in weather_data]
temperature_data = [data['main']['temp'] for data in weather_data]
weather_description_data = [data['weather'][0]['description'] for data in weather_data]

# Print out the list to make sure the queries were extracted 
print ("The City Name List: " + str(city_name))
print ("The Temperature List: " + str(temperature_data))
print ("The Weather Description List: " + str(weather_description_data))


# **Create a dictionary to store your data**

# In[ ]:


# Create a dictionary containing our newly extracted information
weather_data = {"City": city_name, 
                "Temperature": temperature_data,
                "Weather Description": weather_description_data}


# **Convert your dictionary into a Pandas DataFrame**

# In[ ]:


# Convert our dictionary into a Pandas Data Frame
weather_data = pd.DataFrame(weather_data).reset_index()
weather_data

