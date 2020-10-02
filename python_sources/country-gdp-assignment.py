#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import Libraries
import numpy as np


# In[ ]:


#getting Data From File
fileCountryWithGDP = open("../input/Countries with GDP.txt","r")
listCountryWithGDP = fileCountryWithGDP.readlines()
print(listCountryWithGDP)


# In[ ]:


# there are four lines in files and second and fourth lines are of our concern
strCountries = listCountryWithGDP[1]
strCountriesGDP =listCountryWithGDP[3]


# In[ ]:


print(strCountries)


# In[ ]:


print(strCountriesGDP)


# In[ ]:


# Convert string into numpy array 
listCountries = strCountries.split(",") # Converting String to list
arrCountries = np.array(listCountries) # Converting List to numpy 
print(arrCountries)


# In[ ]:


#same process for country
listCountriesGDP = strCountriesGDP.split(",")
arrCountriesGDP = np.array(listCountriesGDP)
print(arrCountriesGDP)


# In[ ]:


# Now we have two numpy array one for countries and another for GDP
# lets check whether both have same elements or not to check gdp corresponding to the country
arrCountriesGDP.size == arrCountries.size


# In[ ]:


# Both datasets have equal number of elements now we have accurate data so we can proceed further 
# now lets clean our data values inside array lets filter countries array first
arrCountries


# In[ ]:


# Countries name have extra single quotes and a \n is also there with the last element
arrCountries = np.chararray.replace(arrCountries,"\n","")
arrCountries = np.chararray.replace(arrCountries,"'","")
print(arrCountries)


# In[ ]:


arrCountriesGDP


# In[ ]:


#now Comes the main part 

# find and print the name of country with highest GDP
#typecasting of GDP to float as we need to 
arrCountriesGDP = arrCountriesGDP.astype('float')
arrCountries[arrCountriesGDP.argmax()]


# In[ ]:


#find and print the name of country with lowest GDP
arrCountries[arrCountriesGDP.argmin()]


# In[ ]:


#print out text and input values iteratively
for country in arrCountries:
    print(country)


# In[ ]:


#print out the entire list of countries with their GDPs
for index,country in enumerate(arrCountries):
    print("Country : ",country.strip(),"| GDP: ",arrCountriesGDP[index])


# In[ ]:


# Print the highest GDP value, lowest GDP value, mean GDP value, standardized GDP value and sum of all GDP values
max(arrCountriesGDP)


# In[ ]:


min(arrCountriesGDP)


# In[ ]:


arrCountriesGDP.mean()


# In[ ]:


arrCountriesGDP.std()


# In[ ]:


sum(arrCountriesGDP)

