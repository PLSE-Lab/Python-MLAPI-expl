#!/usr/bin/env python
# coding: utf-8

# # Analysis of US Mass Shootings (1966 - 2017)

# In[ ]:


from IPython.display import Image
Image("../input/us-shootings-pic/m1.jpg" ,width="800" )


# ### Hi all, this is the first time I've submitted a kernel to Kaggle. As you can probably see, I had to edit it many times! Please enjoy my analysis. Any advice on how to make it better is appreciated. Leave your comments below!
# 
# ### Before I begin my analysis of the mass shootings data, I realize this maybe a touchy subject for some. I offer my condolences to the family members of the victims and condemn any kind of killings.
# 
# ###### The dataset I'll be working with is a modified version of the 'ver 5 Mass Shootings' data set. The changes I made are:
# * Changed the date format to (DD/MM/YY)
# * Added a 'Year' and 'Month' column in Excel
# * Replaced 'null' values in the 'Cause' column with 'unknown'
# * Replaced 'null' value in the 'Target' column with 'unknown'

# ### Importing the required libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Loading the data

# In[ ]:


data = pd.read_csv('../input/modified-dataset/Mass_Shootings.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# #### I'll get rid of the columns I won't be needing
# 

# In[ ]:


del data['S#']
del data['Policeman Killed']
del data['Gender']
del data['Employeed (Y/N)']
del data['Employed at']


# In[ ]:


data.columns


# #### The features avilable are
# * **S#** - Just the index
# * **Title** - A case name that the 'shooting' has been assigned
# * **Location** - The city and the state the shooting occured
# * **Date** - The date of the shooting
# * **Incident Area** - School, Church, Parks, etc.
# * **Open/Close Location** - Describes whether the shooting occured in an open or closed or an open and closed space
# * **Target** - The victims of the shooting
# * **Cause** - Shooting cause e.g. Racial, Terrorism, Psychotic outbreak, etc.
# * **Summary** - A brief summary of the event
# * **Fatalities** - Death count (also counts the shooter, if they were killed/ commited suicide afterwards)
# * **Injured** - Injury count
# * **Total victims** - injury plus the death count (Excludes the perpetrator if he/she is killed)
# * **Age** - Age of the shooter. Data is only available for events which had a single shooter. (More on this later)
# * **Mental_Health_Issues** - States whether the shooter had mental health issue or not, or whether the case is unclear or unknown
# * **Race** - Race of the shooter
# * **Latitude** - self-explanatory
# * **Longitude** - self-explanatory

# In[ ]:


len(data)


# So, we've got 323 records

# ### What is a 'mass shooting'?

# According to the Investigative Assistance for Violent Crimes Act of 2012, signed into law on Jan 2013, a mass shooting is defined as a shooting resulting in at least 3 victims, excluding the perp. For the purpose of this analysis, I'll stick with this definition. 'Mass shootings' does not have a broadly accepted definiton.
# 
# 

# ### I'm curious about the ages of the shooters. Please note, in this data set, the age of the shooter was available if the shooting had a single shooter only. 

# In[ ]:


data['Age'].describe() 


# Out of 189 shooters, the youngest shooter was only 11 years old whereas the eldest shooter was 70. The mean age is 31.
# 

# ### Let's do a point plot to visualize total victim count in the last 50 years

# In[ ]:


years = data['Year'].unique().tolist()
years = sorted(years)
total_victims_groupedby_year = data.groupby('Year')['Total victims'].sum()


sns.set_style("whitegrid")
plt.figure(figsize=(20,5))
plt.title('US mass shootings victim count from 1966 - 2017', fontsize = 18)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Total victims', fontsize = 15)
sns.pointplot(years, total_victims_groupedby_year) #Excluded '2018' as only 2 mass shootings occured in this year


# There seems to be a high between 2013 and 2017. Let's take a closer look. I also hope this number drops significantly in 2018!

# In[ ]:


sns.pointplot(years[37:42], total_victims_groupedby_year[37:42])
plt.xlabel('Year')


# Just by looking at the output above, it seems that in 2017, there has been seven times as many victims of mass shootings as there was back in 2013. I'll do the math, just to be sure.

# In[ ]:


mask_13 = data['Year'] == 2013
mask_17 = data['Year'] == 2017
data_13 = data[mask_13]; data_17 = data[mask_17]

data_17['Total victims'].sum() / data_13['Total victims'].sum()


# There, almost 6.6 times as much!

# ### I'm curious to know the 'causes' behind this rise. 

# Below, I've created a copy of the dataframe that contains all the details from 2013-2017. I'll be doing some analysis on it.

# In[ ]:


data_13to17 = data[(data['Year'] == 2013) | ((data['Year'] == 2014)) | ((data['Year'] == 2015)) | ((data['Year'] == 2016)) |
    ((data['Year'] == 2017))]
plt.figure(figsize=(10,6))
sns.countplot(y = data_13to17['Cause'], order = data_13to17['Cause'].value_counts().index)


# So, from 2013 to 2017, the motive behind 'most' of the shootings is **unknown**. But, there has been many **psychotic outbreak** and **terrorism** related shootings during this time.

# ### How does the number of mass shootings in the last 5 years compare to mass shootings over the last 50 years?
# 

# In[ ]:


len(data_13to17) / len(data) * 100


# More than fifty percent of the total number of mass shootings in the last 50 years (1966-2017), took place between 2013 and 2017!

# ### How many mental health related issues ?

# In[ ]:


sns.countplot(y = data_13to17['Mental_Health_Issues'], order = data_13to17['Mental_Health_Issues'].value_counts().index)


# Although most shooters have an uknown mental health status, there has been some shooters with mental health related issues.

# ## Deadliest shootings

# In[ ]:


Image("../input/us-shootings-pic/m2.jpg" ,width="800" )


# ### Let's take a look at the top 15 devastating (highest death count) shootings in the last 50 years and see whether our data matches with the picture above.

# In[ ]:


top15_mass_shootings = data[['Title', 'Location', 'Year', 'Date','Cause','Mental_Health_Issues', 'Incident Area', 'Injured', 'Fatalities', 'Total victims']].sort_values(
    'Fatalities', ascending = False)[:15]
top15_mass_shootings


# The data is legit!

# ### What about the most deadliest shooting?

# In[ ]:


top15_mass_shootings[:1]


# ##### The most devastating shooting took place on the 1st of October 2017 at Las Vegas, NV. The shooting caused about 59 deaths and 527 injuries with a total of 585 victims. That is over 5 times over the victim count compared to the second most devastating shooting.
# 
# ##### Here's a brief summary 
# 
# "Stephen Craig Paddock, opened fire from the 32nd floor of Manadalay Bay hotel at Last Vegas concert goers for no obvious reason. He shot himself and died on arrival of law enforcement agents. He was 64".
# 
# The cause and mental health status of the shooter is 'unknown'.

# ### The causes for the top 15 mass shootings (according to fatality count)

# In[ ]:


plt.figure(figsize=(10,6))


sns.countplot(top15_mass_shootings['Cause'], order = top15_mass_shootings['Cause'].value_counts().index,
            palette='RdBu')
plt.title('Causes for the top 15 deadliest shootings', fontsize = 13)


# Terrorism stands out as the number one reason for the 15 most deadliest shootings.

# ### Let's rank the shootings by victim count

# In[ ]:


top15_mass_shootings_v = data[['Title', 'Location', 'Year', 'Date','Cause','Mental_Health_Issues', 'Incident Area', 'Injured', 'Fatalities', 'Total victims']].sort_values(
    'Total victims', ascending = False)[:15]
top15_mass_shootings_v


# Do you see it? The Las Vegas shooting had five times as many victims than the Orlando Night Club massacre, which has the second most victim count.

# ### Now, let's take a look at the reasons behind the shootings over the last 50 years.

# In[ ]:


plt.figure(figsize=(10,6))

sns.countplot(y = data['Cause'], order = data['Cause'].value_counts().index)


# The motive for most shootings is not known. Psychotic outbreaks, terrorism, anger and frustation are also some of the top causes.

# ### What about the shooter(s) 'race'?

# In[ ]:


plt.figure(figsize=(10,6))

sns.countplot(y = data['Race'], order = data['Race'].value_counts().index, palette='Blues_r')


# Do the results surprise you?

# ### Here's all the victims of the shootings over the last 50 years

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(y = data['Target'], order = data['Target'].value_counts().index)


# It seems, on most cases, it's random. However, family members, co-workers and students have been targeted plenty of times as well.

# ### I'm also curious to know the amount of mental health related incidents over the last 50 years. Let's do a plot first

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Mental_Health_Issues'], palette='Greens_r', order = data['Mental_Health_Issues'].value_counts().index)


# Mental health status of most shooters is unknown, but there's a high amount of shooters with mental health issues. Let's check what percentage of shooters did have mental health related issues.

# In[ ]:


percentage_of_mental_health_shootings = (data.Mental_Health_Issues == 'Yes').sum() / (data['Mental_Health_Issues'].count())
percentage_of_mental_health_shootings * 100


# There, almost 33 percent of the shooters had mental health issues. Should there be a law to restrict access to guns for people with mental health issues?

# ### Mass shootings by month

# In[ ]:


plt.figure(figsize=(10,6))

sns.countplot(y = data['Month'], order = data['Month'].value_counts().index)


# It seems most mass shootings took place in February over the last 50 years. We're in 2018 and there has already been a case of a mass shooting in the US on valentines day. 

# ### Below is an interactive Power BI report with the US states and  total victim count

# In[ ]:


from IPython.display import IFrame
powerBiEmbed = 'https://app.powerbi.com/view?r=eyJrIjoiMDMyMmJhMDItZWY3YS00Mzc1LWFiMDUtNzY4YjUwOWFmM2FhIiwidCI6IjZlODUyZjhkLTNlNGItNDRkZC04M2RhLTAyM2M5OGY3ZjdhYSJ9'
IFrame(powerBiEmbed, width=800, height=600)


# In[ ]:


Image("../input/us-shootings-pic/m3.jpg" ,width="800")


# ### Summary
# * More than 50 percent of the total mass shootings in the US from 1966 to 2017 took place between 2013 and 2017. The motive behind most of these shootings is unknown. Psychotic outbreaks, followed by terrosim and anger are some of the leading known causes.
# * Terrorism has been the number one reason behind 15 of US's most deadliest shootings.
# * 33 percent of all shooters seemed to have mental health issues. There's still a large portion of shooters whos status is yet to be known. US might need to take 'mental health' problems more seriously.
# 
# ### Some interesting facts
# * Most mass shootings seem to take place in February. We've had one this year(2018) as well.
# * When ranked according to total victim count, the Las Vegas shooting ranks the highest. In addition to this, it has 5 times as many victims when compared to the second ranked shooting by victim count.
# * From available data, the youngest shooter was only 12 years old.
