#!/usr/bin/env python
# coding: utf-8

# **Historical Plane Crashes**
# 
# 
# The aim of this project is to have a better understanding of the data. We explore the data and try to find some substantial patterns. 
# 
# In order to have a strcutural approach in the analysis - we ask ourselves questions during the exploration phase and try to logically explain the observed trends. 
# 
# ***Note: we will not try to generalise the answers, but only explain what is the data trying to show us.***
# 
# 
#  Let us start by getting set up with our favorite libraries 

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 # line only for colab 
# from google.colab import files
# uploaded = files.upload()


# In[ ]:


# colab
# import io
# df2 = pd.read_csv(io.BytesIO(uploaded['Airplane_Crashes_and_Fatalities_Since_1908.csv']))

# kaggle
df2 = pd.read_csv("../input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv")


# jupyter 
# import os
# # os.getcwd()
# df2 = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")


# # Data types and missing values

# Let us look at the datatypes in each column before proceeding to look at the data:

# In[ ]:


df2.dtypes


# Checking for NA values in data

# In[ ]:


#@title Checking NA values in each column
a = [(i, df2[i].isna().sum()) for i in df2.columns]
labels, ys = zip(*a)
 
plt.figure(figsize=[8,12])
plt.barh(labels, ys , height=0.8)
plt.title("Missing values in each Column")

plt.rcParams.update({'font.size': 22})

# for a in zip(*a):
#   print(a)


# In[ ]:


df2.head(3)


# # Quick observations
# 
# - Right off the bat we can see that there some missing values present. 
# - In the "**Location**" column, we can find information about countries - maybe we can extract some info from there. 
# - Another interesting approach is to extract *Military vs non-Military* information from the "**Operator**" column
# - There is also a possibilty of applying regex on "**Summary**" column and extracting information about the crashes.
# - Route column appears to be be inconsistent intuitively - but we can still try to get useful information

# # Birds-eye Approach
# 
# It is always good to start from a "higher-view" and then gradually move into deeper analysis. For the sake of simplicity and making our analysis fruitful, let us define two characteristics of the the data - 
# 1. ***Dimensions***: are variables in the context of analysis
# 2. ***Measures***: the change in these values, changes the "measure" 
# 
# You can observe that this approach is similar to the coordinate geometry analogy (dimensions and values at a coordinate).
# 
# Logically speaking, we can "measure" 2 things from the data :
# 
# 1. *Crashes*
# 2. *Fatalities*
# 
# Now these measures can be a function of one or many dimensions. In the first part of the analysis, we will look at the following dimensions individually:
# 
# 1. *Is Military?*
# 2. *Country*
# 3. *Operator*
# 4. *Type*
# 5. *Route*
# 
# In the first part, we explore basic questions such as: ***Is there a pattern between Military flights and fatalities?***. This way we start off from a birds-eye view and move towards deeper patterns between different dimensions.
# 
# In the higher levels, we can explore the relationship between groups of dimensions agaisnt the measures. In higher levels, we can answer questions/ hypothesis such as : ***Is there a pattern between Military plane and Operators in terms of fatalities per crash?***

# # "Is Military?" Column 
# 
#  **Make a new column for military or non-military Operators**

# **Defining a function for "apply operation"**
# 
# We will define a fucntion and then use the DataFrame.apply method():

# In[ ]:


logic = df2['Operator'].str.contains("Military")

df2["Is Military?"] = logic


# **Quick glance at the data**

# In[ ]:


df2.head(3)


# ## How many military planes?
# 

# In[ ]:


df2['Is Military?'].value_counts()


# In[ ]:




df2['Is Military?'].value_counts().plot(kind='bar')


# ~85% crashes are non-military - maybe we can explore and see if there are eny more interesting patterns

# # Creating "Country" column from "Location" column
# 
# Data analysis always involves wrangling and churning existing data to give a new meaningful perspective.
# 
# We can separate the Location by "," and have the last entry as country. 
# 
# Here, we also observe that we have states from USA - we can group them as one later.

# We will define a function which splits the string of "Location" and take the last value:

# In[ ]:


#defininf function for split

def split_country(x):
  a = x.split(",")[-1]
  return a.replace(" ", "")


#df2['Location'].apply(split_country) 


# ## Why removing NaN values is important for apply method
# 
# If we remove the comment from the above code - we get an error, this is because there are missing value present inside df\["Location"]. Hence, let us quickly analyse to see the extent of missing values inside "Location":

# In[ ]:


df2['Location'].isna().value_counts()


# there are 20 missing values inside the "Location" column - let us look at the data and decide what is the best approach:

# In[ ]:


df2[df2['Location'].isna()==True]


# **Decision**
# 
# as the missing values are very small part of data - we can remove the missing values and continue analysis of "Location" of plane crashes.
# 
# Dropping NA:

# In[ ]:


df2.dropna(inplace=True, subset=['Location'])


# quick verification:

# In[ ]:


df2['Location'].isna().value_counts()


# ## Applying "Country" logic:

# In[ ]:


df2['Country'] = df2['Location'].apply(split_country)


# In[ ]:


df2.head(3)


# ## Applying "USA" logic

# As said before - we see a lot of US states in the data - we can still further group these into single "USA" variable. 
# 
# We copy paste the list of states from internet and then apply Series.isin operation to assign 'USA' value.

# In[ ]:


x = "Alabama, Alaska, AmericanSamoa, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, DistrictofColumbia, Florida, Georgia, Guam, Hawaii, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, Minor OutlyingIslands, Mississippi, Missouri, Montana, Nebraska, Nevada, NewHampshire, NewJersey, NewMexico, NewYork, NorthCarolina, NorthDakota, NorthernMarianaIslands, Ohio, Oklahoma, Oregon, Pennsylvania, PuertoRico, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, U.S.VirginIslands, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming"
x=x.split(", ")
x=pd.Series(x)


# In[ ]:


indices = df2['Country'].isin(x)


df2.loc[indices ,'Country'] = 'USA'
df2.head()


# # Extracting information about year of crash
# 
# We can populate year from the "Date" column separately and investigate any chronological patterns in the data.

# In[ ]:


df2['Date'] = pd.to_datetime(df2['Date'])
df2["Year"] = df2['Date'].apply(lambda x: x.year)
df2.head()


# # **Level 1 analysis**

# ##**Military** 

# In[ ]:


#@title Military vs non-military crashes

df2['Is Military?'].value_counts().plot(kind='barh', title='Military vs non-military crashes', figsize=(10,10), fontsize=12)


# In[ ]:


#@title Fatalities per crash
# df2['Is Military?'].value_counts()

df2.groupby(['Is Military?'])['Fatalities'].mean().plot(kind='barh', title = 'Fatalities per crash', figsize=(10,10), fontsize=12)


# ### **Observations**
# 
# We observe that ***fatalities per crash (FPC)*** for military crashes is *higher than* non-military crashes. We can explore further and understand why this is the case. 
# 
# ***Do the composition of non-military have higher percentage of commercial flights?*** Because, intuitively, passenger flights would have a higher FPC ratio when compared to military flights. 

# ##**Country**

# In[ ]:


#@title Top 15 countries by crash locations
df2['Country'].value_counts()[:15].plot(kind='barh',  figsize=(20,10), rot=20, fontsize=22, title="Top 15 countries by crash locations")


# In[ ]:


#@title Top 15 locations by FPC
df2.groupby(['Country'], sort=True)['Fatalities'].mean().sort_values(ascending=False)[:15].plot(kind='barh', figsize=(20,10), rot=20, fontsize=22, title="Top 15 locations by FPC")
# g.sort_index()


# We see that on the country level, the ***countries with high number of crashes do not necessarily correspond to high FPC***. 
# 
# In order to get a clearer picture, we will ***calculate the FPC for top 20 countries by no. of crashes.***
# 
# Selcting top 20 countries by crashes:

# In[ ]:


subc = df2['Country'].value_counts()[:20]
subc.head()


# In[ ]:


subfpc = df2.groupby(['Country'], sort=True)['Fatalities'].mean()
subfpc.shape


# Calculating the FPC by countries and only selecting the required countries.

# In[ ]:


logic = subfpc.index.isin(subc.index)
logic
subfpc[logic]


# Joing this data and viewing together

# In[ ]:


subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)


# **Inner join and plot:**

# In[ ]:


df = subc.join(subfpc[logic], how="inner")
df.rename(columns={'Country':'Crashes', 'Fatalities':'FPC'}, inplace=True)
df.plot(kind='barh', figsize=(20,10), fontsize=20, rot=30)


# To better analyse the data, we can separate the FPC and crashes axis. By doing so we get the following:

# In[ ]:


#@title Top 20 Crash locations by no.of crashes
subc = df2['Country'].value_counts()[:20]

subfpc = df2.groupby(['Country'])['Fatalities'].mean().sort_values(ascending=False)


logic = subfpc.index.isin(subc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subc.join(subfpc[logic], how="inner")
df.rename(columns={'Country':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 20 Crash locations by no.of crashes")

df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1, label="FPC")
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0, label="Crashes")
# plt.legend(loc="upper right")

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.show()


# **Observations**
# 
# - ***The number of crashes in USA are the highest, but the FPC is not the highest***. We need to investigate and see what is the reason here?
# - Looking at USSR (and incidently also Russia) we can see that ***both crashes and FPC are relatively higher***. We need to dive deeper and investigate further.

# In a similar way - we can slice ***top-20 FPC countries*** and see how the crashes compare:

# In[ ]:


#@title Top 20 Crash locations by FPC
subc = df2['Country'].value_counts()

subfpc = df2.groupby(['Country'])['Fatalities'].mean().sort_values(ascending=False)[:20]


logic = subc.index.isin(subfpc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subfpc.join(subc[logic], how="inner")
df.rename(columns={'Country':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 20 Crash locations by FPC")
df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1)
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0)



ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.legend()
plt.show()


# ### **Observations**
# 
# We observe that:
# - ***large number of crashes do no necessary mean large FPC values***. In such cases, the flight may be carrying large number of people aboard (like long haul passenger planes). 
# - ***Further investigate EastGermany and Canary islands***, because the no.of crashes are higher than other locations in the group.
# 
# - USA has most number of crashes, but this might be because ***USA has the most busiest airports*** (from wikipedia).
# 

# ### **Higher level investigation ideas:**
# 
# 
# 
# why usa has higher crashes?
# 
# - is it military or commercial?
# 
# - is something strange with no.of crashes or FPC?
# 
# - is there any cluster about the years in which the crashes occur
# 
# Canary islands and east germany have quite significant number of crashes 
# 
# - see why so?
# 
# - passenger planes, but any more information?

# ##**Operator**

# We can repeat the same excercise as conducted in "Country" and see FPC for top 20 operators by highes number of crashes.

# In[ ]:


#@title Top 10 Operators by no.of crashes
subc = df2['Operator'].value_counts()[:10]

subfpc = df2.groupby(['Operator'])['Fatalities'].mean().sort_values(ascending=False)


logic = subfpc.index.isin(subc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subc.join(subfpc[logic], how="inner")
df.rename(columns={'Operator':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 10 Operators by no.of crashes")

df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1, label="FPC")
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0, label="Crashes")
# plt.legend(loc="upper right")

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.show()


# In[ ]:


#@title Top 10 Operators by FPC
subc = df2['Operator'].value_counts()

subfpc = df2.groupby(['Operator'])['Fatalities'].mean().sort_values(ascending=False)[:10]


logic = subc.index.isin(subfpc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subfpc.join(subc[logic], how="inner")
df.rename(columns={'Operator':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 10 Operators by FPC")

df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1, label="FPC")
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0, label="Crashes")
# plt.legend(loc="upper right")

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.show()


# In[ ]:


# df2.groupby(['Operator'])['Fatalities'].mean().sort_values(ascending=False)[:20]
# df2[df2['Operator']=="Pan American World Airways / KLM"]


# ### **Observations**
# 
# 
# ***1.*** ***The FPC for passenger/commercial operators such as American Airlines, PANAM and Aeroflot is higher than the Military Operators.***
# 
# There are some key points to note before generalising.
# - Commercial and passenger planes might have higher frequency of crashes because the ***frequency of total flights is higher***. (We do not have this data available in the dataset, but it would have given a much clearer picture)
# 
# - At the same time, ***passenger planes carry much more people***. So, even if the crashes are fewer compared to total flights, the FPC increase because of the number of people on board.
# 
# - We can verify this by looking at "US Aerial mail service" and "Aeropostale". Even though the number of crashes are in top-15 figures, the FPC value is lesser because the ***number of people on board are fewer***.
# 
# 
# 
# 
# ***2.*** ***High FPC operators generally have single digit crashes in contrast to high FPC***. 
# This intuitively makes sense because one crash of a large passenger plane will result in large number of fatalities. For example, the ***Pan American World Airways / KLM*** disaster included two boeing 747s and resulted in 583 fatalities.

# **Further analysis**
# 
# - We need to dive deeper and verify the hypothesis about FPC correlating with passenger planes.
# 
# - We also need to dive deeper and analyse Aeroflot, because it looks equally dangerous as US Air force in terms of both FPC and no.of crashes.

# ##**Type**

# In[ ]:


#@title Top 10 Airliners by no. of crashes
subc = df2['Type'].value_counts()[:10]

subfpc = df2.groupby(['Type'])['Fatalities'].mean().sort_values(ascending=False)


logic = subfpc.index.isin(subc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subc.join(subfpc[logic], how="inner")
df.rename(columns={'Type':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1)
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0)


ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.legend()
plt.title("Top 10 Airliners by no. of crashes")
plt.show()


# In[ ]:


#@title Top 10 Airliners by highest FPC
subc = df2['Type'].value_counts()

subfpc = df2.groupby(['Type'])['Fatalities'].mean().sort_values(ascending=False)[:10]


logic = subc.index.isin(subfpc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subfpc.join(subc[logic], how="inner")
df.rename(columns={'Type':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 24})

fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1)
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0)

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.legend()
plt.title("Top 10 Airliners by highest FPC")
plt.show()



# ### **Observations**
# 
# - Here too, We can observe that in high FPC cases, the number of crashes are usually single digit. This means the ***flight was carrying large number of people aboard***. In such scenarios, just one crash results in high number of fatalities.
# 
# - Investgate ***Douglas DC-3*** deeper, as the number of crashes are higher than other airliners.

# ## **Route**

# In[ ]:


#@title Top 10 Routes by no.of crashes
subc = df2['Route'].value_counts()[:10]

subfpc = df2.groupby(['Route'])['Fatalities'].mean().sort_values(ascending=False)


logic = subfpc.index.isin(subc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subc.join(subfpc[logic], how="inner")
df.rename(columns={'Route':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 22})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 10 Routes by no.of crashes")

df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1, label="FPC")
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0, label="Crashes")
# plt.legend(loc="upper right")

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.show()


# In[ ]:


#@title Top 10 Routes by FPC
subc = df2['Route'].value_counts()

subfpc = df2.groupby(['Route'])['Fatalities'].mean().sort_values(ascending=False)[:10]


logic = subc.index.isin(subfpc.index)

subc = pd.DataFrame(subc)
subfpc = pd.DataFrame(subfpc)



df = subfpc.join(subc[logic], how="inner")
df.rename(columns={'Route':'Crashes', 'Fatalities':'FPC'}, inplace=True)
# df.plot(kind='barh', figsize=(20,10), fontsize=11, rot=30, title="Top 20 Flight type by FPC")
plt.rcParams.update({'font.size': 24})


fig = plt.figure(figsize=(20,15)) # Create matplotlib figure



ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twiny() # Create another axes that shares the same x-axis as ax.
# plt.figure(figsize=(20,20))


width = 0.4
plt.title(" Top 10 Routes by FPC")

df.FPC.plot(kind='barh', color='orange', ax=ax, width=width, position=1, label="FPC")
df.Crashes.plot(kind='barh', color='blue', ax=ax2, width=width, position=0, label="Crashes")
# plt.legend(loc="upper right")

ax.set_xlabel('FPC', color='orange')
ax.tick_params(axis='x', colors='orange')
ax2.set_xlabel('Crashes', color='blue')
ax2.tick_params(axis='x', colors='blue')

plt.show()


# ## **Year**

# Let us see how the number of ***crashes*** and ***FPC*** (Fatalities per crash) evolve over the years:

# In[ ]:


#@title Crashes vs FPC over the Years


a = df2.groupby('Year').size()
a=pd.DataFrame(a)
a.rename(columns={0: 'Crashes'}, inplace=True)

b = df2.groupby('Year')['Fatalities'].mean()
b=pd.DataFrame(b)
b.rename(columns={'Fatalities': 'FPC'}, inplace=True)


# a=a.reset_index()
# b=b.reset_index()

df = b.join(a, how='inner', on='Year')


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

color = 'tab:blue'
ax.set_ylabel('Crashes', color=color)
ax.tick_params(axis='y', labelcolor=color)
ax.plot(df.index, df['Crashes'], color=color)

ax1 = ax.twinx()

color = 'tab:red'
ax1.set_ylabel('FPC', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(df.index, df['FPC'], color=color)


plt.title("Crashes vs FPC over the Years")
plt.show()


# ***Observations***:
# 
# - We see that ***both crashes are FCP are in a downward trend since around 1980s.***
# - We can investigate the difference betwee ***1960's - 80's*** and ***1980 - onwards***, to see why the crashes and FPC was upward and then downwards.

# In[ ]:


#@title Military vs Non-Military crashes


a = df2[df2["Is Military?"]==True].groupby('Year').size()
a=pd.DataFrame(a)
a.rename(columns={0: 'Military crashes'}, inplace=True)

b = df2[df2["Is Military?"]==False].groupby('Year').size()
b=pd.DataFrame(b)
b.rename(columns={0: 'Non-Military crashes'}, inplace=True)
b


# a=a.reset_index()
# b=b.reset_index()

df = b.join(a, how='inner', on='Year')


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

color = 'tab:red'
ax.set_ylabel('Military crashes', color=color)
ax.tick_params(axis='y', labelcolor=color)
ax.plot(df.index, df['Military crashes'], color=color)

ax1 = ax.twinx()

color = 'tab:green'
ax1.set_ylabel('Non-Military crashes', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(df.index, df['Non-Military crashes'], color=color)


plt.title("Military vs Non-Military crashes")
plt.show()


# Obervations:
# 
# - We actually see something interesting: ***between 1970's - 80s***, the crashes for Military were going down and non-military were going up. We can try seeing if there is something to explore here.
# 
# - After ***80's***, for both Military and non-Military, ***the number of crashes are in a steady decline***.
# 
# - We can investigate the spike around ***1944/45.***

# In[ ]:


#@title Military vs Non-Military FPC


a = df2[df2["Is Military?"]==True].groupby('Year')['Fatalities'].mean()
a=pd.DataFrame(a)
a.rename(columns={'Fatalities': 'Military FPC'}, inplace=True)

b = df2[df2["Is Military?"]==False].groupby('Year')['Fatalities'].mean()
b=pd.DataFrame(b)
b.rename(columns={'Fatalities': 'Non-Military FPC'}, inplace=True)
b


# a=a.reset_index()
# b=b.reset_index()

df = b.join(a, how='inner', on='Year')


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

color = 'tab:red'
ax.set_ylabel('FPC')
ax.tick_params(axis='y')
ax.plot(df.index, df['Military FPC'], color=color, label="Military")

# ax1 = ax.twinx()

color = 'tab:green'
ax.plot(df.index, df['Non-Military FPC'], color=color, label="Non-Military")


plt.title("Military vs Non-Military FPC")
plt.legend()
plt.show()



# Observations:
# 
# - We can check the spikes at 1935, 1990 and 2002 for Military operators.
# 
# - Apart fromt the spikes, the FPC for both Military and non-Military Operators ***are in a downward trend since ~1985.***
