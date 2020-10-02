#!/usr/bin/env python
# coding: utf-8

# This kernel is for the benefit of users who have followed the link from the recent Data Cleaning exercise and wished to map the missing San Francisco ZIP codes to the listed addresses. Unfortunately, I have found this data to be unavailable and have reason to believe, as shown below, that no public record was accessible when the data was collated. However I was thorough in making sure it wasn't hidden amongst the other data so my brief kernel may be useful if you ever need to be certain that geographical data is missing from a future source you're using.
# 
# You can directly download the data from OpenAddresses at the following link: http://results.openaddresses.io/sources/us/ca/san_francisco
# 
# Any corrections, suggestions or comments are appreciated.
# 
# To begin with, let's load in the data, get a list of columns and view the first few rows:

# In[1]:


import numpy as np 
import pandas as pd 

df = pd.read_csv('../input/ca.csv') #place California data in our DataFrame, df
print(df.columns)
df.head()


# Since we're looking for the San Francisco data in this case, let's review the CITY column: 

# In[2]:


cities = df.CITY.value_counts()
cities[:15] #top city regions by number of addresses


# Already this is unusual. San Francisco is the largest Californian city by population after Los Angeles but there are seemingly no data for its addresses. Perhaps, for whatever reason, San Francisco is named in the DISTRICT or REGION columns instead of CITY. Examining their contents:

# In[3]:


print('REGION value counts: \n', df.REGION.value_counts(), '\n')
print('DISTRICT value counts: \n', df.DISTRICT.value_counts())


# The REGION column only contains some superfluous clarification that the addresses are in California. The DISTRICT column is empty. These are of no help.
# 
# It is possible, albeit unlikely, that the San Francisco data has been mislabelled under 'UNASSIGNED' in the CITY column. To check this,  the average GPS coordinates of these addresses should lie in the city itself. Testing this hypothesis:

# In[4]:


test = df[df.CITY == 'UNASSIGNED']
long = test.LON.mean()
lat = test.LAT.mean()

print('Average UNASSIGNED location is at Longitude: %f Latitude: %f' % (long, lat))


# According to Google Maps this location lies midway between Sacramento and Reno, fairly central to California state, but not in San Francisco. This is unsurprising: the 'UNASSIGNED' values presumably correspond to smaller, rural locales across the whole of California that do not fall within any particular city limits. The average latitude and longitude are therefore reasonably central, with a slight Northern bias (perhaps since Southern California is more arid and has a smaller rural population) but they don't correspond to a specific location, and certainly not San Francisco.
# 
# As a final check, I obtained a list of San Francisco ZIP codes from this local government website: 
# 
# https://data.sfgov.org/Geographic-Locations-and-Boundaries/San-Francisco-ZIP-Codes/srq6-hmpi/data
# 
# If any of the addresses, albeit not labelled accurately, lie in these ZIP codes then they are indeed in San Francisco:
# 

# In[5]:


SF_ZIP = ['94102', '94104', '94103', '94105', '94108', '94107', '94110', '94109', '94112', '94111', '94115', '94114', '94117', '94116', 
          '94118', '94121', '94123', '94122', '94124', '94127', '94126', '94129', '94131', '94133', '94132', '94134', '94139', '94143',
          '94146', '94151', '94159', '94158', '94188', '94177']

SF_addresses = df[df['POSTCODE'].isin(SF_ZIP)]
print(SF_addresses)


# This returns an empty DataFrame.  The San Francisco data can therefore be presumed absent; no addresses in the California dataset are listed under ZIP codes belonging to San Francisco. This is presumably by design, and would suggest the data has been purposefully either not collected, or not published. It is possible that at the time this dataset was prepared, San Francisco either had data privacy laws restricting public access to this information or the data simply had not been collected yet.
# 
# I hope this kernel has been useful to you!
# 
# Again, the latest data can be found at http://results.openaddresses.io/sources/us/ca/san_francisco
