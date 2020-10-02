#!/usr/bin/env python
# coding: utf-8

# **World Indicators Mini Project**

# The world indicators dataset provides a multitude of data from around the world for various indicators. In this project, I want to see if CO2 emissions into the environment have a relationship with fertility rates in a country. 
# I chose the United States and Europe + Baltics to compare fertiltity and CO2 emissions. 

# The first step is to import the necessary libraries. 
# We use pandas for manipulating data and dataframes, and matplotlib for data visualization. 

# In[2]:


#Import Libraries 

import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# Next we will load in the dataset provided to us by Kaggle. 

# In[3]:


#Load in dataset from Kaggle 

data = pd.read_csv('../input/Indicators.csv')


# The world indicators dataset is HUGE. In order to evaluate the data we actually want, we're going to have to pull it from our current data frame, labeled as 'data' above, and utilize filters and masks to essentially create a new dataframe of the information we want to look at. 
# 
# The data can be messy in the dataframe, so I'm also going to drop some columns we don't need before storing it in a final value.

# In[16]:


# select CO2 emissions for the United States

#Filtering the information we would like to use 

filterIndicator1 = 'CO2 emissions \(metric'
filterIndicator2 = 'USA'

#Creating mask for the filters 

mask1 = data['IndicatorName'].str.contains(filterIndicator1) 
mask2 = data['CountryCode'].str.contains(filterIndicator2)

#Storing the masks in a new value 
USCarbon = data[mask1 & mask2]

#Preview the new dataframe (nice for seeing how you would like to clean it up if at all)
USCarbon.head()

#Drop unnecessary columns 

USCarbonFinal = USCarbon.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])

#View final dataframe

USCarbonFinal.head(10)


# Now we'll do the same thing for the European data.

# In[5]:


# select CO2 emissions for Central Europe and the Baltics
filterIndicator3 = 'CO2 emissions \(metric'
filterIndicator4 = 'CEB'

mask3 = data['IndicatorName'].str.contains(filterIndicator3) 
mask4 = data['CountryCode'].str.contains(filterIndicator4)

#create dateframe from filter masks
EuroCarbon = data[mask3 & mask4]

#Drop unwanted columns from dataframe
EuroCarbonFinal = EuroCarbon.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])

#View first 10 columns in dataframe post drop 
EuroCarbonFinal.head(10)


# We can see just from eyeballing the two datasets that the European carbon emissions are notably lower than that of their US counterpart. 
# Let's see how the two look when compared graphically. 

# In[6]:


#Create a graph to compare the two countries Carbon Emissions 

#X & Y values
x = USCarbon['Year']
y = USCarbon['Value']  
Y = EuroCarbon['Value']

#Creating our plot using matplotlib function 

plt.plot(x, y, color='r', label='US')
plt.plot(x, Y, color='b', label='Euro')
plt.xlabel('Years')
plt.ylabel('Carbon Emissions')
plt.title('US vs. Euro Carbon Emissions')

#Add a legend
plt.gca().legend(('US','Euro'))
plt.show()


# The graph provides an easy to decipher comparison of the two CO2 emissions. Both appear to be increasing carbon emissions as years pass, but Europe has a slower rate and emits less all together. 

# We can now do the same thing for the fertility rates for both country codes using the same methods as above. 

# In[7]:


#Now let's look at fertility rates per country 

#US Fertility Rate 

filterIndicator5 = 'SP.DYN.TFRT.IN'
filterIndicator6 = 'USA'

mask5 = data['IndicatorCode'].str.contains(filterIndicator5) 
mask6 = data['CountryCode'].str.contains(filterIndicator6)

# Storing masks 
USFertility = data[mask5 & mask6]

#Drop unwanted Columns
USFertilityFinal = USFertility.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])

#View first 10 columns
USFertilityFinal.head(10)


# In[8]:


#Fertility rates in Europe/Baltics

filterIndicator7 = 'SP.DYN.TFRT.IN'
filterIndicator8 = 'CEB'

mask7 = data['IndicatorCode'].str.contains(filterIndicator7) 
mask8 = data['CountryCode'].str.contains(filterIndicator8)

# Create a new dataframe for the European Fertility data 
EuroFertility = data[mask7 & mask8]

#Drop unwanted Columns
EuroFertilityFinal = EuroFertility.drop(columns=['CountryName','CountryCode','IndicatorName','IndicatorCode'])

#View first 10 columns
EuroFertilityFinal.head(10)


# Just from looking at the graphs, it appears that both countries fertility rates are moving down, but neither appears to be incredibly drastic from the preview of the datasets. 
# 
# We can gain a better understanding of what these comparisons look like grahically. 

# In[9]:


#Create a graph to compare the two countries fertility rates over the years.


#X & Y values 

x = USFertility['Year']
y = USFertility['Value']
Y = EuroFertility['Value']

#Creating a graph using matplotlib 

plt.plot(x, y, color='r', label='US')
plt.plot(x, Y, color='b', label='Euro')
plt.xlabel('Years')
plt.ylabel('Fertility')
plt.title('US vs. Euro Fertility Rates')

#Add a legend
plt.gca().legend(('US','Euro'))
plt.show()


# As mentioned above, both countries appear to have decreasing fertility rates. 
# Europe starts out with lower rates and gets even lower, where as the US starts out significantly higher and has some dramatic drops before leveling back out a bit. 
# 
# But could there be any link to carbon emissions? 

# Let's get a little bit more information about our datasets by using the describe function.

# In[ ]:


#Is there a correlation between CO2 Emissions and fertility rates? 

#First let's get some more information from our datasets 

#US Carbon Emissions Data Set

#Mean, Std, min, max, counts (should be the same)
USCarbonFinal.describe()

#Count : Year = 52, Value = 52
#Mean : Value = 19.303472
#Standard Deviation : Value = 1.564525
#min : Value = 15.681256
#max : Value = 22.510582


#Euro Carbon Data Set

#Mean, Std, min, max, counts (should be the same)
EuroCarbonFinal.describe()

#Count : Year = 52, Value = 52
#Mean : Value = 8.237914
#Standard Deviation : Value = 1.805031
#min : Value = 5.114244
#max : Value = 11.285238


# In[ ]:


#US Fertility Data Set

#Mean, Std, min, max, counts (should be the same)
USFertilityFinal.describe()

#Count : Year = 54, Value = 54
#Mean : Value = 2.158602
#Standard Deviation : Value = 0.482290
#min : Value = 1.738000
#max : Value = 3.654000


#Euro Fertility Data Set

#Mean, Std, min, max, counts (should be the same)
EuroFertilityFinal.describe()

#Count : Year = 54, Value = 54
#Mean : Value = 1.893536
#Standard Deviation : Value = 0.432010
#min : Value = 1.251015
#max : Value = 2.498618


# The describe function allows us to see information like the mean, standard deviation, and miniumum and maximums. 
# All of this data is represented graphically and in our data frames already, but this is another way to take a quick glance at how our data compares to each other. 
# 

# The next thing I'm going to do is merge my like dataframes together.
# I want all of my US data in one dataframe and my Europe data in another.
# This way I can see visually how the numbers change for each variable over the years.
# 
# Because there's some extra data in this dataframe, I'm going to use my drop columns command again, as well as the reoder and rename command. This makes it easier to read the dataframe and understand exactly what you're looking at. 

# In[10]:


#Merge Dataframes together  

USMerge = USCarbon.merge(USFertility, on='Year', how='inner')

#Drop Columns
USMerge1 = USMerge.drop(columns=['CountryName_x','CountryName_y','IndicatorCode_x',
                                     'CountryName_y','CountryCode_y','IndicatorCode_y',
                                    'IndicatorName_x', 'IndicatorName_y'])

#Rename Columns 
USMerge1.rename(columns={'CountryCode_x': 'CountryCode','Value_x' : 'CO2_Value',
                          'Value_y':'Fertility_Value'}, inplace=True)
#preview
USMerge1.head()

#Reorder Columns so years is not in the middle of the data values 
USMergeFinal = USMerge1[['CountryCode', 'Year', 'CO2_Value', 
                             'Fertility_Value']]

#Final Dataframe
USMergeFinal.head()


# The new US dataframe allows me to see all in one place how my data shifts over the years. 
# Now let's do the same thing for the European data. 

# In[11]:


#Merge Dataframes together 

#European Datasets 

EuroMerge = EuroCarbon.merge(EuroFertility, on='Year', how='inner')

#Drop Columns
EuroMerge1 = EuroMerge.drop(columns=['CountryName_x','CountryName_y','IndicatorCode_x',
                                     'CountryName_y','CountryCode_y','IndicatorCode_y',
                                    'IndicatorName_x', 'IndicatorName_y'])

#Rename Columns 
EuroMerge1.rename(columns={'CountryCode_x': 'CountryCode','Value_x' : 'CO2_Value',
                          'Value_y':'Fertility_Value'}, inplace=True)

#preview data 

EuroMerge1.head()

#Reorder Columns so years is not in the middle of the data values 
EuroMergeFinal = EuroMerge1[['CountryCode', 'Year', 'CO2_Value', 
                             'Fertility_Value']]
#Final View of data

EuroMergeFinal.head()


# 
# 
# We gained some great insights from further exploring our data and evaulating our dataframes. 
# As a final step, we will create two more side by side graphs to show both countries carbon and fertility rates side by side. 
# 
#  
# 
# 
# 

# In[12]:


#Final Graphs 

#x & y values
x1 = USCarbon['Year']
y1 = USCarbon['Value']
Y1 = EuroCarbon['Value']

#Carbon
plt.subplot(1, 2, 1)
plt.plot(x1, y1, color='r', label='US')
plt.plot(x1, Y1, color='b', label='Euro')
plt.title('Fertility & Carbon Emissions')
plt.ylabel('Carbon Emissions')

####

#x & y values
x2 = USFertility['Year']
y2 = USFertility['Value']
Y2 = EuroFertility['Value']

#Fertility
plt.subplot(1, 2, 2)
plt.plot(x2, y2, color='r', label='US')
plt.plot(x2, Y2, color='b', label='Euro')
plt.xlabel('Years')
plt.ylabel('Fertility Rates')

plt.gca().legend(('US','Euro'))
plt.show()


# Without doing a deeper analysis using pearsons r for correlation, it's difficult to make a conclusive decision about if CO2 is or isn't impacting fertility rates for women. 
# However, based on the graphs alone, there does not appear to be reason to expect correlation or causation. Fertility rates do appear to be decreasing, but so do carbson emissions.
# Something worth noting is that the carbon emission spikes are around the same time period as big drops in fertility rates. 
# Whether or not this is from carbon emissions is impossible to tell just from these graphs and this data alone. There could be a multitude of reasons for what we see being represented on the graphs that aren't carbon related (or just carbon alone for that matter.) 
