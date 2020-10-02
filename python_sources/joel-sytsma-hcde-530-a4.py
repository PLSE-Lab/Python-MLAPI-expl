#!/usr/bin/env python
# coding: utf-8

# ## A4: Manipulating Data
# This assignment is focused on working with dictionaries and manipulating data. Follow the instructions for each step below. After each step, insert your Code Cell with your solution if necessary (in some steps there will be some code provided for you).The assignment is in two parts (A and B). Part A focuses on data manipulation and using dictionaries. Part B focuses on retrieving data to update the dictionary.
# 
# ### Submission
# When you have finished, submit this homework by sharing your Kaggle notebook. Click Commit in the upper right of your Kaggle notebook screen. Click on Open Version to view it. Make sure to set Sharing permissions to public. Then copy the URL for that version. To submit on Canvas, click Submit Assignment and paste the link into the URL submission field.

# # PART A
# ### Step 1: Accessing values at a specified key in a dictionary.
# 
# We have created a list of cities to keep track of what current populations are and potentially identify which cities are growing fastest, what the weather is like there, etc. Add code to print the number of residents in Chiago the our `citytracker` list. (the value associated with key 'Chicago' in the dictionary dinocount). Hint: this is just one simple line of code. Add the code below.

# In[ ]:


citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}


# ### Step 2: Incrementing the value of a dictionary at a key.
# 
# Write code to increment the number of residents in Seattle by 17,500 (that happened in one month in Summer 2018!). In other words, add 17,500 to the existing value of cities at key 'Seattle').  Then, print out the number of residents in the Seattle.

# In[ ]:


citytracker['Seattle']=citytracker['Seattle']+17500 #I'm redefining Seattle to add 17500 to the number associated with the Seattle key

print(citytracker['Seattle'])


# ### Step 3: Adding an entry to a dictionary. 
# 
# Our list of cities just got bigger. (What could go wrong?) Write code to insert a new key, 'Los Angeles' into the dictionary, with a value of 45000. Verify that it worked by printing out the value associated with the key 'Los Angeles'

# In[ ]:


citytracker['Los Angeles'] = 45000 #setting a new key for the city tracker dictionary and defining it's value to 45000

print(citytracker['Los Angeles'])


# ### Step 4: Concatenating strings and integers. 
# 
# Write code that creates a string that says 'Denver: X', where X is the number of residents extracted from the `citytracker` dictionary.  Print the string. Hint: you will need to use the + string concatenation operator in conjunction with str() or another string formatting instruction.

# In[ ]:


print("Denver:", citytracker['Denver']) # printing the word Denver, followed by the value associated with the Denver key


# ### Step 5: Iterating over keys in a dictionary.  
# 
# Write code that prints each city (key), one line at a time, using a for loop.

# In[ ]:


for city in citytracker: #setting the city variable to inherit the key for each loop of the citytracker dictionary
    print(city) #printing the variable we've assigned the key value to


# ### Step 6: iterating over keys to access values in a dictionary. 
# 
# Write code that prints each city (key), followed by a colon and the number of residents (e.g., Seattle : 724725), one line at a time using a for loop.

# In[ ]:


for city in citytracker: ##setting the city variable to inherit the key for each loop of the citytracker dictionary
    print(city, ":", citytracker[city]) #added code so that the variable that city currently is referencing calls that value in each iteration


# ### Step 7: Testing membership in a dictionary.
# 
# Write code to test whether 'New York' is in the `citytracker` dictionary.  If the test yields `true`, print `New York: <x>`, where `<x>` is the current population. If the test yields false, it should print "Sorry, that is not in the Coty Tracker. Do the same thing for the key 'Atlanta'.

# In[ ]:


if 'New York' in citytracker: #querying citytracker keys for the "New York" key
    print ('New York',":", citytracker['New York']) #if it's true then it would print this. New york isn't there, so it won't print this
else:
    print("Sorry, that is not in the Coty Tracker") #setting up and else statement to print if New York doesn't appear
    
if 'Atlanta' in citytracker: #same logic as New York, but for Atlanta
    print ('Atlanta',":", citytracker['Atlanta'])
else:
    print("Sorry, that is not in the Coty Tracker")
    


# ### Step 8: Default values
# 
# We have a list of potential cities (in the list potentialcities) and we want to check whether the city exists within the City Tracker. If it is, we want to print `city: #`, where city is the city name and # is the population. If the city is not in the dictionary, it should print zero. Add to the code below to do this. *Hint: you can use default values here to make this take less code!*
# 

# In[ ]:


potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee', 'Chicago'] #I added Chicago to the list, just to make sure my code below worked

for city in potentialcities: #looping through the potentialcities list and assigning the value in each index the variable "city"
    if city in citytracker: #creating a conditional statement to see if the index value is a key in the citytracker dictionary
        print(city,":", citytracker[city]) #if "city" is in citytracker, then print it's appropriate value
    else: #if city is not in the dictionary
        print("0") #then print 0
        


# ### Step 9: Printing comma separated data from a dictionary.
# 
# You may have worked with comma separated values before: they are basically spreadsheets or tables represented as plain text files, with each row represented on a new line and each cell divided by a comma. Print out the keys and values of the dictionary stored in `citytracker`. The keys and values you print should be separated only by commas (there should be no spaces). Print each `key:value` pair on a different line. *Hint: this is almost identical to Step 6*

# In[ ]:


for city in citytracker: #same as problem 6
    x=citytracker[city] #setting variable x to be the value to the pair that the city key calls. Doing this because the format function doesn't accept dictionaries.
    print(city+",{}".format(x)) #printing the city variable plus ","" and the text that was placed in the x. I removed the spaces using format.


# ### Step 10: Saving a dictionary to a CSV file
# Write key and value pairs from `citytracker` out to a file named 'popreport.csv'. *Hint: the procedure is very close to that of Step 9.* You should also include a header to row describe each column, labeling them as "city" and "pop".

# In[ ]:


import os
import csv #importing the functionality to write a CSV

with open('popreport.csv', 'w') as f: #using the open function to write to the file popreport.cvv. I"m going to put what I want to write in the variable f.
    for city in citytracker: #looping through the citytracker dictionary to find each key value
        x=citytracker[city] #setting up the x variable to represent the value of each key that is found
        print(city+",{}\n ".format(x)) #just keeping track of each loop
        f.write(city+",{}\n".format(x)) #writing each loop to f. the format function removes spaces and \n starts a new line after each loop.
    



### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # PART B
# In this part, you will use API keys to access weather information for cities in the `citytracker` dictionary. You need to get the weather for each item and store it in the same (or a new) dictionary. You then print out the data and format it to be pretty (whatever that means to you) and finally, write it out as a json file with a timestamp.
# 
# **You will need to enable Internet connections in the sidebar to the right of your notebook. If you get a `Connection error` in the Console, it is because your notebook can't access the internet.**
# 
# ### Step 1: Accessing APIs to retrieve data
# First, you will need to request an API Secret Key from OpenCage (https://opencagedata.com/api) and add it to your Kaggle notebook in the Add-ons menu item. Once you have the Secret Key, you attach it to this notebook (click the checkbox) so you can make the API call. Make sure the **Label** for your key in your Kaggle Secrets file is what you use in your code below.
# 
# You will also an API Secret Key from DarkSky (https://darksky.net/dev). Attach it to this notebook and use it in the code. Make sure you have created different labels for each key and use them in the code below.
# 
# Finally, make sure to install the `opencage` module in this notebook. Use the console at the bottom of the window and type `pip install opencage`. You should receive a confirmation message if it installs successfully/
# 
# Then try running the code cells below to see the output. Once the code sucessfully works for Seattle (which has been provided for you below), try typing in different cities instead to see the results to make sure it is working.
# 
# ### Step 2: Retreiving values for each city in your dictionary
# Now try to get information for all of the cities in your `citytracker` dictionary. You can print the information out to make sure it is working. Store the results of `getForecast` for each city in your dictionary.
# 
# ### Step 3: Writing the datafile
# Save the results of your work as a JSON formatted output file in your Kaggle output folder and Commit your notebook. Make sure to make it publc and submit the resulting URL in Canvas.
# 

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("openCage") # make sure this matches the Label of your key
key1 = secret_value_0

from opencage.geocoder import OpenCageGeocode
geocoder = OpenCageGeocode(key1)
for query in citytracker:
    results = geocoder.geocode(query)
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
    print (query,"Lat: %s, Lon: %s" % (lat, lng))


# In[ ]:


# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("darkSky") # make sure this matches the Label of your key

import urllib.error, urllib.parse, urllib.request, json, datetime

def safeGet(url):
    try:
        return urllib.request.urlopen(url)
    except urllib2.error.URLError as e:
        if hasattr(e,"code"):
            print("The server couldn't fulfill the request.")
            print("Error code: ", e.code)
        elif hasattr(e,'reason'):
            print("We failed to reach a server")
            print("Reason: ", e.reason)
        return None

def getForecast(lat="lat",lng="lng"):
    # https://api.darksky.net/forecast/[key]/[latitude],[longitude]
        key2 = secret_value_0
        url = "https://api.darksky.net/forecast/"+key2+"/"+lat+","+lng
        return safeGet(url)

cityweather = {}

for query in citytracker: #put everything in a for loop
    results = geocoder.geocode(query)
    lat = str(results[0]['geometry']['lat'])
    lng = str(results[0]['geometry']['lng'])
    print(query)
    getForecast(lat,lng)
    data = json.load(getForecast(lat,lng))
    current_time = datetime.datetime.now() 
    
    #all the variables, consider using the .update
    print("Retrieved at: %s" %current_time)
    print(data['currently']['summary'])
    print("Temperature: " + str(data['currently']['temperature']))
    print(data['minutely']['summary'])
    
    cityweather[query] = {'current time':current_time, 'temperature': data['currently']['temperature'],'mini forecast':data['minutely']['summary']}
with open('jsonfile.json', 'w') as json_file:
        json.dump(citytracker, json_file)
        

