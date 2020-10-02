#!/usr/bin/env python
# coding: utf-8

# A common Federal business development data analysis requires a list of unique identifiers for government contractors falling into a specific category. The U.S. Federal Government's [System for Award Managment](http://sam.gov) maintains information about entities doing business with the government and serves as an authoritative data source. 
# 
# This kernal explores using the Application Program Interface (API) published by the [General Services Administration](http://gsa.gov) for importing entity registration data from the [System for Award Managment](http://sam.gov). The [SAM API](https://gsa.github.io/sam_api/sam/index.html) allows for search parameters and returns Javascript Object Notation (JSON) formatted data.
# # Prerequisites
# * Queries require a unique API key. Obtain one at https://api.data.gov.
# 

# # Build the Query URL
# The following code creates the query URL. Customize the query by: 
# 1. Updating the **myAPIKey** string with your unique API key. 
# 2. Editing the **qterms** string according to the [SAM Search API](https://gsa.github.io/sam_api/sam/search.html) direction.
# 
# To display the query URL, run the code by clicking inside the code cell and pressing Shift+Return. 
# 
# For testing purposes, click the URL to view the results.

# In[ ]:


baseURL = "https://api.data.gov/sam/v3/registrations?qterms="
myAPIKey = "ENTER YOUR API KEY HERE"
qterms = "GSA"
queryURL = baseURL + qterms + "&api_key=" + myAPIKey
print(queryURL)


# # Running the Query
# The following code saves the output from the URL query to **response**. 

# In[ ]:


import requests
from requests.exceptions import HTTPError

try:
    response = requests.get(queryURL)
    response.raise_for_status()
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')  
except Exception as err:
    print(f'Other error occurred: {err}')  
else:
    print('Successfully retreived data from ' + queryURL)


# # Viewing the Retrieved Data

# In[ ]:


response.json()


# # Creating a List of Unique Identifiers
# The **duns** and **duns_plus4** values represent the unique identifiers needed for querying other Federal government systems. 

# In[ ]:


dunsNumbers = []
t = response.json()
for i in range(len(t['results'])):
    dunsNumbers.append(t['results'][i]['duns'] + t['results'][i]['duns_plus4'])
dunsNumbers


# **dunsNumbers** now contains the data needed for further queries. 
