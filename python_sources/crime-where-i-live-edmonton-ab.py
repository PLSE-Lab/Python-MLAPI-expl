#!/usr/bin/env python
# coding: utf-8

# # CRIME WHERE I LIVE (Edmonton AB) - MY FIRST KERNEL
# 
# **Purpose: **
# 
# The purpose of this Kernel is to analyze Reported Crime Incidents where I live (Edmonton, Alberta, Canada), merge the dataset with Edmonton population figures and calculate a crime rate for the various incident types that we can visualize, analyze and report on.
# 
# **Procedure:**
# 
# **OBTAIN DATA**
# 1. Obtain Edmonton Crime Statistics (Identify Source)
# 2. Obtain Edmonton Population Statistics (Identify Source)
# 
# **ANALYZE + PREPARE DATA**
# 3. Analyze + Clean Crime Dataset + Filter Crime dataset to group by year (int) , type of incident (str) and calculate the sum of reported incidents (int). 
# 4. Analyze + Clean Population Dataset Filter Population dataset to show year (int) and population (int).
# 5. Merge datasets on year (int)
# 
# **CREATE VISUALIZATIONS + REPORT ANALYSIS**
# 6. Prepare visualizations + share thoughts

# In[ ]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt


# Let's import the datasets. I downloaded these from the edmonton open dataportal. Websites are included below. 

# In[ ]:


# EDMONTON CRIME DATA - https://dashboard.edmonton.ca/dataset/EPS-Neighbourhood-Criminal-Incidents/xthe-mnvi/data
df1 = pd.read_csv('../input/EPS_Neighbourhood_Criminal_Incidents.csv')

# EDMONTON POPULATION STATS - https://data.edmonton.ca/dataset/Edmonton-Population-History/frjf-2vsa
df2 = pd.read_csv('../input/Edmonton_Population_History.csv')


# Let's take a quick peak at the Crime data we will be using and make some notes about what we might need to do to fix it. 

# In[ ]:


df1.head()


# Our crime data set has a great deal of information around the types of reported incidents, the neighbourhood they occurred in and the number of incidents organized by year, quarter and month. 
# 
# Our columns are:
# 
# **Neighbourhood Description (Occurrence)** - The neighbourhood in Edmonton AB where the incident took place. 
# 
# **UCR Violation Type Group (Incident)** - The type of incident (Assault, Break and Enter, Robbery, Sexual Assault, Theft FROM Vehicle, Theft OF Vehicle, Homicide, Theft Over $5000). 
# 
# **Incident Reported Year** - the Year in which the incident took place. 
# 
# **Incident Reported Quater** - the quarter (1, 2, 3 or 4) that the incident took place. 
# 
# **Incident Reported Month** - the month(1 - 12, 1 being January and so forth) when the incident took place. 
# 
# **'# Incidents** - the total number of reported incidents for that type, year, quarter and month.
# 
# The column names are a bit odd and will be difficult to continually type, so the first step will be to rename the columns into something that will be easier to type and use. I came up with the following new names (feel free to use whatever you like):
# 
# Neighbourhood Description (Occurrence) = Neighbourhood_Name
# 
# UCR Violation Type Group (Incident) - Violation_Type
# 
# Incident Reported Year - Year
# 
# Incident Reported Quater - Quarter
# 
# Incident Reported Month - Month
# 
# '# Incidents - Number_Incidents
# 

# In[ ]:


df1.columns = ['Neighbourhood_Name', 'Violation_Type', 'Year', 'Quarter', 'Month', 'Number_Incidents']
df1.head()


# Now we have some easy to type column names we can use to group, filter and merge the data on.
# 
# Let's make some tables using pandas and see what we can come up with.

# In[ ]:


df1.groupby('Neighbourhood_Name').Number_Incidents.sum().sort_values(ascending=False)


# Pretty cool...we can order the neighbourhoods of Edmonton by the areas with the most reported incidents since 2009. Downtown and Oliver are in the lead and from what I know, those are the Urban Centers of the City of Edmonton so it makes sense that they would have the highest number of reported incidents. 
# 
# **Neighbourhood not entered:**
# 
# There is quite a large sum of incidents where the neighbourhood has not been entered...This could have serious implications if we are analyzing data by neighbourhood or if the data is also missing a large piece of reported incidents. Maybe we should dig into this more to see if it might affect our analysis on crime for the City of Edmonton as a whole. 
# 
# Let's try to isolate the Not Entered Neighbourhood data and then make a decision as to if it will affect our overall goal of calculating crime rate by type of crime and year for the City of Edmonton as a whole. We can use the pandas loc function to locate and display anything with a neighbourhood name equal to not entered. 

# In[ ]:


df1.copy().loc[df1['Neighbourhood_Name'] == 'Not Entered']


# It appears that although neighbourhood name is not always entered, the rest of the columns seem to be intact and have complete data. The type of assault, year, quarter, month and reported incidents when filtered don't have any unusual unique values or headings that were unexpected. 
# 
# Since I am not analyzing crime on a neighborhood by neighborhood basis I will include the data with neighborhood = Not Entered in the calculation of the number of incidents sum that we will use later. The final dataframe I create from this data to merge with population will NOT include neighborhood information. 

# **BUILDING OUR CRIME DATAFRAME**
# 
# In order to build this dataframe we have to know what we need to include. Our goal is to calculate incident rate by type of crime and year. To do this, we need to group our crime data by type of incident and year, have the data show the sum of total reported incidents for that specific year and incident type, then merge population figures from our separate dataset based on the Year column. 

# In[ ]:


incident_year = pd.Series.unique(df1['Year'])  # A series of all the unique years included in our dataset. This will be our index or row labels. 
violation_type = pd.Series.unique(df1['Violation_Type'])  # A series of all the unique Violation types. This will be our column labels. 

#GEnerate the empty dataframe that we will be graphing
df3 = pd.DataFrame(columns=violation_type, index=incident_year)
df3.head(10)


# In[ ]:


# For each value (i) in incident_year, fill the location of year = i, sum the number of incidents based on the violation_type.
for i in incident_year:
    df3.loc[i] = pd.Series(df1.groupby(['Year', 'Violation_Type']).Number_Incidents.sum()[i])

df3.head(10)


# Now let's add a Year column for merging with population (happens after we clean the population dataset) and rename the dataframe to crime_clean to use later in merging (You can call it whatever you like).

# In[ ]:


df3['Year'] = pd.Series.unique(df1['Year'])
crime_clean = df3
crime_clean.head(10)


# **POPULATION DATASET**
# 
# Now we can begin to work on the population dataset. Let's take a quick peak at it. 

# In[ ]:


df2.head()


# This looks pretty great. We have some easy to use columns and our column name for merging already matches our Crime Dataframe...but wait a minute. That year data looks pretty intense. We don't really need a lot of that information...in fact...the only thing we need is the 4 characters after the second slash that represent the year. We should check the type to see if that column is a datetime object. Python and Pandas have a lot of great resources for parsing the year out of such a long date format.

# In[ ]:


df2.dtypes


# Check this out:

# In[ ]:


df4 = df2
# apply - removes characters 6 through 10 from the year column and changes type to int
df4['Year'] = df2.Year.apply(lambda x: x[6:10]).astype(int)
df4.head(10)


# Great...now we have an integer year that we can use to merge. Let's clean up this table include only the columns we want in our final population_clean dataframe. 

# In[ ]:


population_clean = df4[['Year', 'Population']].sort_values(by='Year', ascending=False)
population_clean.head(10)


# Looks pretty great. We have a year column to merge on. Wait a minute...some of the years appear to be missing. Let's think about this. Why would we be missing population data? 
# 
# Well, it is probably an expensive endeavour to go out and perform a census so its likely some of the years won't have population figures. We will find out post merge where our data is missing and we can go to alternative sources to fill in that information. Before we can see this, we need to merge our datasets and check what is missing.
# 
# We will call our merged dataset crime_population.

# In[ ]:


# Merge our crime_clean database with population_clean on the Year column, how=outer means that null values will be filled with NaN and not ignored from the dataset. 
# This ensures we can check and fill in missing population figures from alternative sources. 
crime_population = crime_clean.merge(population_clean, on='Year', how='outer')
crime_population.head(15)


# The dataset from the Edmonton Data Portal on Crime indicated that the figures are not updated until the following quarter after quarter end. This means our current year data for 2018 crime stats is incomplete (at the time of writing we are in q4 2018). As such we cannot rely on the 2018 crime stats if we are to calcualte a crime rate on the city as a whole. This will be called our max_year.
# 
# We do not have crime stats for any year prior to 2009 but we have a lot of population stats. This is our minimum year. 
# 
# We must limit our analysis to include the time frame between minimum year (INCLUSIVE) and maximum year (NOT INCLUSIVE) as this is the only period in which we have valid data. 

# In[ ]:


# This code sets and checks the min_year and max_year.
min_year = pd.Series.unique(df1['Year']).min()
max_year = pd.Series.unique(df1['Year']).max()
print(min_year)
print(max_year)


# In[ ]:


# This will eliminate any line of data with a year less than the min year, or only include data that is equal to or greater than the minimum year. 
crime_population = crime_population.copy().loc[crime_population['Year'] >= int(min_year)]
crime_population.head(10)


# In[ ]:


# This will eliminate any line of data that is greater than or equal to our max year (since we don't have complete data for 2018 we would want that out)
# We will also set our index to equal the year so that we can get ready to graph based on the year. 
crime_population = crime_population.copy().loc[crime_population['Year'] < int(max_year)].set_index('Year')
crime_population.head(10)


# Great. Now we can see specifically that we are missing population figures for three of the years in our analysis, 2011, 2015 and 2017.
# 
# Ideally, we would want to go to an independent source for htis information but that may not always be possible or plausible. Population figures from census as well as estimates are normally provided in the Annual Report of a City, so that would be a great primary source to include in our analysis. Where this value is not available or does not make sense, we would establish another criteria, perhaps (prior year population + subsequent year population) divided by 2, to come up with an average population value we could use.
# 
# The City of Edmonton Annual Reports can all be found here https://www.edmonton.ca/city_government/facts_figures/coe-annual-reports.aspx
# 

# **2017 Population:**
# 
# 2017 Population - From City of Edmonton 2017 Annual Report is estimated at  932546. We will use this figure in our analysis. 
# 
# We can use the loc to update the value for year 2017, in the population column to 932546

# In[ ]:


crime_population.loc[2017, 'Population'] = 932546
crime_population.head(10)


# **2015 Population:**
# 
# Annual report indicates 877926 which is same as 2014. Data was likely not collected on this year and it is unlikely the population, which has grown each year remained the same as 2014.
# 
# For 2015 I have applied the average (2016 pop + 2014 pop) / 2.
# 
# We again use the loc to update the 2015 population value as follows:

# In[ ]:


crime_population.loc[2015, 'Population'] = int((crime_population.loc[2016, 'Population'] +
                                                crime_population.loc[2014, 'Population']) / 2)
crime_population.head(10)


# We only have one missing value now and that is 2011. 
# 
# **2011 Population:**
# 
# Annual report indicates 812201. 
# 
# Using our average formula, the population value would be calculated as 807,409 (prior year population + subsequent year population) / 2.
# 
# I will use the value indicated in the City of Edmonton Annual Report of 812,201 as that is an independent source. 

# In[ ]:


crime_population.loc[2011, 'Population'] = 812201
crime_population.head(10)


# Now we have a dataframe that shows us the total incidents by type in the year, as well as the population.
# 
# The standard analysis of crime is that as population increases, so should crime. This is why crime rate is a much better value vs. total reported incidents because the crime states are normalized over a population (usually, shown as a rate per 100,000 people). So let's say our homicide rate is 5. That means that for every 100,000 people living in Edmonton in a particular year, there are 5 homicides. If we had 500,000 people living in Edmonton, the rate would indicate that we would have 25 homicides for that year (as an example). 
# 
# Crime Rate per 100,000 is calculated as (Number of Incidents / Population) * 100,000.
# 
# For our final dataframe, we will divide every cell by the population and multiply by 100,000. We will call this dataframe df_rate. If this formula is successful, we should expect to see the population column equal 100,000 for every year. We can even test a few of the observations and incident types by manually calculating. 

# In[ ]:


df_rate = crime_population.div(crime_population.Population, axis='index') * 100000
df_rate.head(10)


# Great. Now we don't want to include a straight line of population of 100,000. It doesn't provide us any great information so let's drop it from the table. 

# In[ ]:


df_rate = df_rate.drop('Population', 1)
df_rate.head(10)


# Now that we have a dartaframe of crime rates  by incident type and year, we can start to graph these things to see how they look and list out some findings. Let's use a line chart so we can see a general shape of how each crime rate moves. 

# In[ ]:


df_rate.plot.line(
        figsize=(20, 10),
        fontsize=16
)

plt.title("INCIDENT RATE PER 100,000 (Edmonton, AB)", fontsize=15)
plt.xlabel('YEAR', fontsize=13)
plt.ylabel('INCIDENT RATE', fontsize=13)
plt.show()


# Neato but there is a lot going on in this graph and most of the lines look flat. This happens when we map large ranges together. From our table above, our homicide rates were around 2-3 per 100,000 people and our theft from vehicle rates ranged from 700 to 1400 per 100,000.
# 
# With such a large range it is no wonder some of our lines look flat and difficult to interpret. We should graph each incident type on its own to see if there are any similarities. 
# 
# That may seem tedious but luckily for us we can use a for statement to iterate through our violation or incident types and generate a graph of the crime rate for that particular incident type. 

# In[ ]:


incident_type = pd.Series.unique(df1['Violation_Type'])

for t in incident_type:
    df_rate[t].plot.line(
        figsize=(10, 5),
        fontsize=16
    )
    plt.title(t.upper() + " INCIDENT RATE PER 100,000 (Edmonton, AB)", fontsize=15)
    plt.xlabel('YEAR', fontsize=15)
    plt.ylabel('INCIDENT RATE', fontsize=15)
    plt.show()
    


# **ANALYZE GRAPHS AND REPORT THOUGHTS**
# 
# Anyting involving theft or break and enter (excluding robberies) seems to follow a general U shape curve, decreasing for a period of 5 years and increasing again going into 2017. I wonder if that curve matches with conditions in the economy or with the price of Oil, since our economy here in Alberta is closely tied to oil. For part 2 of this analysis we could begin finding that data and mapping it to see if the trends are similar. Edmonton Police could then alternate resources in times of economic hardship to prevent some of these incidents. We could expand our analysis to go into the quarterly figures to see if there are increased thefts at different times of year. Are there more incidents in the summer when the weather overnight does not drop to freezing? Are there more incidents around the holidays when people scramble and times can be tough? These analysis can help Police make resource allocation decisions at different times and could lead to more success in preventing incidents. 
# 
# I wonder what kind of policy decisions the Edmonton Police Service made before and after the ups and downs related to theft. I remember some campaigns about locking your cars and keeping valuables out of sight. Maybe those had an impact in the years? As a next step we could write some language processing functions to look for policy changes to reduce theft, break and enter or robberies. We could focus on the years prior to the large drops in the rate and see if there was a specific initiative that led to such drops and then again any policy changes prior to the increases to see if there was a specific policy that helped curb some of these incidents.
# 
# It would appear that theft seems to be heading towards homes and vehicles in current years with robberies decreasing from 200 to the 120-140 range and remainign stagnant there since 2011, while other theft categories and break and enter follow that same U trend, increasing in most recent years. What policy changes have remained in place around robberies vs. what policy changes have been made around other theft categories? Comparing these could give us insight as to why rates have increased. 
# 
# Sexual assault rates have followed an upwards trend. This could be a result of increased incidents of sexual assaults or increased reporting of sexual assaults, something that  has been a known difficulty for victims to do for many reasons. We could again scan policy changes or news articles to see if the Edmonton Police Service has had any initiatives around this. What changes in Edmonton culture have happened in the last 10 years that could be contributing factors? There could be news articles or information from advocacy groups that may help shed some light. Has victim support been positive or negative in the City? Has it led to more or less incident reports?
# 
# What happened to the homicide rate in 2011 to create such a huge spike? We could scan news articles to see if there was some kind of gang war that would have driven homicides up in that particular year as they generally remain stagnant year over year.
# 
# It feels like the only conclusion I can reliably come to is that more analysis is definitely needed. I will do my best to dig deeper into each of the rate graphs and try to find more information about the various categories and certain intiatives that may have led to decreases or increases in the rates.
# 
# It would also make sense to group some of these incident types together. Grouping the theft categories together could provide us insight into types of incidents that could have strong correlations. Do theft of vehicle, theft from vehicle and theft over $5000 follow the same general trends? More analysis required and more to come on this.
# 
# This being my first ever Kernal I would love to hear suggestions on improving code or better explaining my code/thoughts. Thank you to those who read it and for those of you on the learning path as I am, I hope I was able to help you learn something.
# 
# If you are completely brand new, Kaggle Learning (https://www.kaggle.com/learn/overview) is about as good as it gets for free knowledge specifically focused on data science.
# 
# To learn Python I completed the Python Masterclass taught by Tim Buchalka (https://www.udemy.com/share/1000dOBUEZd1lWQ3w=/ ).
# 
# Best of luck to you all. 
# 
# Regards,
# 
# Ricardos Moussallem
