# %% [markdown]
# # Crimes in Montgomery County
# 
# This project aims to explore of crimes in Montgomery Count, Maryland.
# 
# The data used is downloaded and saved in the "Crime.csv" file of this repository, but can also be accessed on the Montgomery County open data website: (https://data.montgomerycountymd.gov/). Each row in the data is a crime reported by a law enforcement officer and contains the following information:
# 
# 1. **Incident ID**: Police Incident Number
# 2. **Offence Code**: Offense_Code is the code for an offense committed within the incident as defined by the National Incident-Based Reporting System (NIBRS) of the Criminal Justice Information Services (CJIS) Division Uniform Crime Reporting (UCR) Program.
# 3. **CR Number**: Police Report Number
# 4. **Dispatch Date / Time**: The actual date and time an Officer was dispatched
# 5. **NIBRS Code**: FBI NBIRS codes
# 6. **Victims**: Number of Victims
# 7. **Crime Name1**: Crime against Society/Person/property or Other
# 8. **Crime Name2**: Describes the NBIRS_CODE
# 9. **Crime Name3**: Describes the OFFENSE_CODE
# 10. **Police District Name**: Name of District (Rockville, Wheaton etc.)
# 11. **Block Address**: Address in 100 block level
# 12. **City**: City
# 13. **State**: State
# 14. **Zip Code**: Zip code
# 15. **Agency**: Assigned Police Department
# 16. **Place**: Place description
# 17. **Sector**: Police Sector Name, a subset of District
# 18. **Beat**: Police patrol area subset within District
# 19. **PRA**: Police patrol area, a subset of Sector
# 20. **Address Number** House or Bussines Number
# 21. **Street Prefix** North, South, East, West
# 22. **Street Name** Street Name
# 23. **Street Suffix** Quadrant(NW, SW, etc)
# 24. **Street Type** Avenue, Drive, Road, etc
# 25. **Start_Date_Time**: Occurred from date/time
# 26. **End_Date_Time**: Occurred to date/time
# 27. **Latitude**: Latitude
# 28. **Longitude**: Longitude
# 29. **Police District Number**: Major Police Boundary
# 30. **Location**: Location
# 

# %% [markdown]
# Let me first import libraries and package that I will be using

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%matplotlib inline
sns.set(style="white")

# %% [markdown]
# # Step-I: Data Cleaning
# First thing we do is read the file using _read_csv()_ function, next we will check on the size of the data, the existence of null entries, along with birds eye view of information about the data. We do this using _shape()_,_isnull()_, _info()_ functions respectively as is done in the following cells. We learned that the dataset has 202094 rows and 30 columns. The names of the columns can be accessed using the _columns()_ function. The description of the names of the columns is given in the introduction part.

# %% [code]
mg_crimes = pd.read_csv("../input/Crime.csv")
mg_crimes.head()

# %% [code]
# checking size of the dataset
mg_crimes.shape

# %% [code]
# checking columns
mg_crimes.columns

# %% [code] {"scrolled":true}
# Preliminary infromation about the dataset
mg_crimes.info()

# %% [markdown]
# We note here that there are some missing entries on the data, to get the exact number of missing information let us call the _isnull()_ function.

# %% [code] {"scrolled":false}
# missing data for each column
mg_crimes.isnull().sum()

# %% [code]
print(f"Ratio of missing entries for column named 'Dispatch Date / Time': {mg_crimes['Dispatch Date / Time'].isnull().sum()/len(mg_crimes)}")
print(f"Ratio of missing entries for column named 'End_Date_Time': {mg_crimes['End_Date_Time'].isnull().sum()/len(mg_crimes)}")

# %% [markdown]
# How to replace the missing values is the question for analysis, I have to options: one replacing the missing data by average and two dropping the missing information.

# %% [markdown]
# # Step-II: Analyzing Crime Patterns
# ## a) When do crimes happen?
# 
# One of the first questions I would like to raise is when do crimes most likely occur?, is there a difference in number of crimes between days of the week?, time of the day?, and month of the year? To answer this questions it is necessary to convert the time format into datetime format using the pandas _to_datetime()_ function.This parsing will help us extract the day, date, and time from the columns _Dispatch Date/ Time_, _End_Date_Time_, and _Start_Date_Time_.  
# As part of taking care of the missing data obtained above, we can also make a comparison of information from these there columns and see if there is a gap in results obtained.

# %% [markdown]
# ### Crime day (dispatch time)
# To identify the day of the week with highest frequency of crimes, let us extract the day of the week from the "Dispatch Date / Time" column. To achive this, we will use the _day_name()_ function. 

# %% [code]
# Using dispatch time identify days of week
dispatch_time = pd.to_datetime(mg_crimes["Dispatch Date / Time"])
dispatch_week = dispatch_time.dt.day_name()
print(dispatch_week.value_counts())
ax = sns.barplot(x=dispatch_week.value_counts().index,y=dispatch_week.value_counts(),color='blue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
ax.set(ylim=(0, 26000),xlabel='Day of the week',ylabel='Frequency')
ax.set(ylabel='Frequency')
ax.set(xlabel='Day of the week')
plt.title("Crime frequency by day of week",fontsize= 16)
plt.xticks(fontsize=16)
plt.figure(figsize=(12,10))

# %% [code]
((22723-16856)/((22723+16856)*0.5))*100

# %% [markdown]
# We see that crimes are high during weekdays. Saturday, Sunday, Monday sees less crime and then crimes starts to climb up on Tuesday. The highest crime day is on Friday the lowest being on Sunday,the percentage difference between the highest and the lowest is about 30%.
# 
# Is it because weekends most people are staying at home, going to church, visit families?

# %% [markdown]
# ### Crime time (dispatch time)
# What time of the day do crimes frequently occur?

# %% [code]
# Using dispatch time identify time of the day
dispatch_hour = dispatch_time.dt.hour
print(dispatch_hour.value_counts())
ax = sns.barplot(x=dispatch_hour.value_counts().index,y=dispatch_hour.value_counts(), color='red')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 10000))
ax.set(xlim=(-1, 24))
ax.set(ylabel='Frequency')
ax.set(xlabel='Hours')
plt.title("Crime frequency by hour of day",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# We see that the crimes are low early in the morning and peaks up around 3pm with a little dip around 2pm (why?) and continues to steadily decrease towards midlnight.
# Is it because the late afternoon (3-4pm) is high trafiic time, when people are out in the street?

# %% [markdown]
# ### Crime month (dispatch time)

# %% [code]
# Using dispatch time identify month of the year
dispatch_month = dispatch_time.dt.month
print(dispatch_month.value_counts())
ax = sns.barplot(x=dispatch_month.value_counts().index,y=dispatch_month.value_counts(), color='green')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 15000))
ax.set(xlim=(-1, 12))
ax.set(ylabel='Frequency')
ax.set(xlabel='Months')
plt.title("Crime frequency by month of year",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# With exception of February and March the crimes seem to occur almost uniformly in each month. 
# Is it because these months come after holidays, or weather is cold during these months?

# %% [markdown]
# ### Crime day (End time)

# %% [code]
# Using end time identify days of week
end_time = pd.to_datetime(mg_crimes["End_Date_Time"])
end_week = end_time.dt.day_name()
print(end_week.value_counts())
ax = sns.barplot(x=end_week.value_counts().index,y=end_week.value_counts(),color='blue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
ax.set(ylim=(0, 16000),xlabel='Day of the week',ylabel='Frequency')
ax.set(ylabel='Frequency')
ax.set(xlabel='Day of the week')
plt.title("Crime frequency by day of week",fontsize= 16)
plt.xticks(fontsize=16)
plt.figure(figsize=(12,10))

# %% [markdown]
# We still see that occurence of crimes during weekends is lower than the weekdays, and Friday is highest crime day. 

# %% [markdown]
# ### Crime time (End time)

# %% [code]
# Using end time identify time of the day
end_hour = end_time.dt.hour
print(end_hour.value_counts())
ax = sns.barplot(x=end_hour.value_counts().index,y=end_hour.value_counts(), color='red')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 9000))
ax.set(xlim=(-1, 24))
ax.set(ylabel='Frequency')
ax.set(xlabel='Hours')
plt.title("Crime frequency by hour of day",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# Highest crime recorded during midnight(why?), morning times (8-10am and at noon) see high crime occurence. 

# %% [markdown]
# ### Crime month (End time)

# %% [code]
# Using end time identify month of the year
end_month = end_time.dt.month
print(end_month.value_counts())
ax = sns.barplot(x=end_month.value_counts().index,y=end_month.value_counts(), color='green')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 10000))
ax.set(xlim=(-1, 12))
ax.set(ylabel='Frequency')
ax.set(xlabel='Months')
plt.title("Crime frequency by month of year",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# Lowest crime months are now seem to be April, May, June

# %% [markdown]
# ### Crime day (Start time)

# %% [code]
# Using start time identify days of week
start_time = pd.to_datetime(mg_crimes["Start_Date_Time"])
start_week = start_time.dt.day_name()
print(start_week.value_counts())
ax = sns.barplot(x=start_week.value_counts().index,y=start_week.value_counts(),color='blue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
ax.set(ylim=(0, 36000),xlabel='Day of the week',ylabel='Frequency')
ax.set(ylabel='Frequency')
ax.set(xlabel='Day of the week')
plt.title("Crime frequency by day of week",fontsize= 16)
plt.xticks(fontsize=16)
plt.figure(figsize=(12,10))

# %% [markdown]
# We still see that occurence of crimes during weekends is lower than the weekdays, and Friday is highest crime day. 

# %% [markdown]
# ### Crime time (Start time)

# %% [code]
# Using start time identify time of the day
start_hour = start_time.dt.hour
print(start_hour.value_counts())
ax = sns.barplot(x=start_hour.value_counts().index,y=start_hour.value_counts(), color='red')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 15000))
ax.set(xlim=(-1, 24))
ax.set(ylabel='Frequency')
ax.set(xlabel='Hours')
plt.title("Crime frequency by hour of day",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# Highest crime recorded during midnight(why?), afternoon times see high crime occurence. 

# %% [markdown]
# ### Crime month (Start time)

# %% [code]
# Using start time identify month of the year
start_month = start_time.dt.month
print(start_month.value_counts())
ax = sns.barplot(x=start_month.value_counts().index,y=start_month.value_counts(), color='green')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 20000))
ax.set(xlim=(-1, 12))
ax.set(ylabel='Frequency')
ax.set(xlabel='Months')
plt.title("Crime frequency by month of year",fontsize= 16)
plt.xticks(fontsize=15)
plt.figure(figsize=(12,10))

# %% [markdown]
# Lowest crime months are now seem to be April, May, June

# %% [markdown]
# Comparing dispatcher, end, and start times:
# We have obtained on cleaning data stage that we have zero missing entries for start_date_time column and it will be used as 'standard' data against which we contrast the results obtained from dispatch and end time calculations.

# %% [markdown]
# ## Crime by year
# 
# Let us see if there is a trend on crime rate by year.  

# %% [code]
# Using start time identify crime by year
start_year = start_time.dt.year
print(start_year.value_counts())
ax = sns.barplot(x=start_year.value_counts().index,y=start_year.value_counts())
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(ylim=(0, 60000))
ax.set(xlim=(-1, 7))
ax.set(ylabel='Frequency')
ax.set(xlabel='Year')
plt.title("Crime frequency by year",fontsize= 16)
plt.xticks(fontsize=15)
plt.savefig('crime_year.png')
plt.figure(figsize=(20,10))

# %% [markdown]
# We see here that the the crime count of year 2016 and 2020 are small, of course the year 2020 is still in progress so that is why crimes are low for this year. But the crime record of 2016 needs further investigation, but I suspect it is because of missing values.  

# %% [markdown]
# # Can we combine all info in one plot?

# %% [code]
df_proj1=dispatch_week.value_counts()
df_proj2=end_week.value_counts()
df_proj3=start_week.value_counts()
groups = [df_proj1,df_proj2,df_proj3]
group_labels = ['Dispatch', 'End time','Start time']
colors_list = ['blue', 'red', 'green']

# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [df_proj1, df_proj2, 
     df_proj3],
    axis=1).plot.bar(title='Comparison of weekly crimes',grid=False,width=0.8,figsize=(20, 8), color=colors_list).legend(bbox_to_anchor=(1.1, 1))
plt.ylabel('Frequency',fontsize=25)
plt.xlabel('Days of week',fontsize=25)
plt.xticks(fontsize=20)
plt.savefig('crime_day.png')
plt.figure(figsize=(12,8))

# %% [code]
df_proj1=dispatch_hour.value_counts()
df_proj2=end_hour.value_counts()
df_proj3=start_hour.value_counts()
groups = [df_proj1,df_proj2,df_proj3]
group_labels = ['Dispatch', 'End time','Start time']
colors_list = ['blue', 'red', 'green']

# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [df_proj1, df_proj2, 
     df_proj3],
    axis=1).plot.bar(title='Comparison of hourly crimes',grid=False,width=0.8,figsize=(20, 8), color=colors_list).legend(bbox_to_anchor=(1.1, 1))
plt.ylabel('Frequency',fontsize=25)
plt.xlabel('Hour',fontsize=25)
plt.xticks(fontsize=20)
plt.savefig('crime_hour.png', dpi=100)
plt.figure(figsize=(12,10))

# %% [code]
df_proj1=dispatch_month.value_counts()
df_proj2=end_month.value_counts()
df_proj3=start_month.value_counts()
groups = [df_proj1,df_proj2,df_proj3]
group_labels = ['Dispatch', 'End time','Start time']
colors_list = ['blue', 'red', 'green']

# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [df_proj1, df_proj2, 
     df_proj3],
    axis=1).plot.bar(title='Comparison of crimes by month',grid=False,width=0.8,figsize=(20, 8), color=colors_list).legend(bbox_to_anchor=(1.1, 1))
plt.ylabel('Frequency',fontsize=25)
plt.xlabel('Months',fontsize=25)
plt.xticks(fontsize=20)
plt.savefig('crime_month.png', dpi=100)
plt.figure(figsize=(12,10))

# %% [markdown]
# ## b) Where do crimes happen?
# 
# The second question I would like to ask is where do crimes most likely be comitted?, is there a difference in number of crimes between cities? if so which cities are safe and which aren't? To answer these questions we need to seek information of location from the data set. In doing so we see that we have different columns that yields information about location, i.e Police District Name, Block Address, Zip Code, Sector, Beat, Latitude, Longitude, Police District Number, Location, Address Number.  
# Based on the _isnull()_ assessment I made at the data cleaning stage, I chose the Police District Name column to do analysis, as this column has zero null entry.
# We need to drop rows with null values and then hroup the crime occurence by Police District Name to areas with the highest crime frequencies. .

# %% [code]
# high crime area Police District Name
mg_crimes_filtered = mg_crimes.dropna(axis=0, subset=['Police District Name'])
mg_crimes_area = mg_crimes_filtered.groupby('Police District Name')['Police District Name'].count()
mg_crimes_sorted = mg_crimes_area.sort_values(ascending=False, inplace=False)
print(mg_crimes_sorted)
ax = sns.barplot(x=mg_crimes_sorted.index,y=mg_crimes_sorted, color="blue")
ax.set_ylabel('Crimes')
ax.set_xlabel('Police District')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Crime occurence by area",fontsize= 16)
plt.xticks(fontsize=15)
plt.savefig('crime_bydistrict.png', dpi=100)
plt.figure(figsize=(12,10))

# %% [markdown]
# We see that Silver Spring is where the highest crime occurence is registered, with in this area we can further refine to identify the cities associated with the highest crimes

# %% [markdown]
# ## c) What type of crimes?

# %% [markdown]
# Information related to the types of crimes involved are provided in columns 'Crime Name1','Crime Name2','Crime Name3'. As these columns have same amount of missing data, i.e 71, there is no preference to chose one from the other based on null analysis. But looking at the description and a view on the columns I found the third column 'Crime Name3' to be somewhat broader and thus I use it to make my analysis. As a future endeveaor it will be good to make further analysis based on 'Crime Name1' and 'Crime Name2' and make a comparison.
# The question I try to answer at this stage is: which types of crimes are commonly occuring? In order to identify the types of crimes I need to pivote table 'Crime Name3' and count the frequency of 'Incident ID'.

# %% [code]
# Identifying the most recurring types of crimes
crime_types = mg_crimes.pivot_table(index='Crime Name3',values=['Incident ID'],aggfunc='count')
crime_types.rename(columns={'Incident ID':'Frequency'}, inplace=True)
crime_types.reset_index(inplace=True)
sorted_crime_types_top= crime_types.sort_values(by='Frequency', ascending=False).head()
sorted_crime_types_bottom= crime_types.sort_values(by='Frequency', ascending=True).head()

# %% [code]
print(f"Top five types of crimes")
sorted_crime_types_top


# %% [code] {"scrolled":true}
print(f"Least five types of crimes ")
sorted_crime_types_bottom

# %% [markdown]
# ### Relation between crime types and district and area?

# %% [code]
crimes_district = mg_crimes.pivot_table(index=['Police District Name', 'Crime Name3'], values=['Incident ID'], aggfunc='count')
crimes_district.rename(columns={'Incident ID':'Count'}, inplace=True) # Renaming column
crimes_district.reset_index(inplace=True) # Removing indexes

idx = crimes_district.groupby(['Police District Name'])['Count'].transform(max) == crimes_district['Count']
crimes_district[idx]

# %% [code]
# Ordering top to bottom
crimes_district.sort_values(by=['Count'], ascending=False)[idx]

# %% [markdown]
# ### Which of places/areas (street,parking, mall etc?) do we frequently see crimes

# %% [code]
crimes_place = mg_crimes.pivot_table(index=['Place'], values=['Incident ID'], aggfunc='count')
crimes_place.rename(columns={'Incident ID':'Count'}, inplace=True) 
crimes_place.reset_index(inplace=True) 

top_places = crimes_place[crimes_place['Count']>1000].sort_values(by=['Count'], ascending=False)
top_places

# %% [code]

ax = top_places.plot.barh (x='Place',y='Count',figsize=(10, 6), color='red')
plt.xlabel('Frequency') # add to x-label to the plot
plt.ylabel('Crime places') # add y-label to the plot
plt.title('Crime places by frequency') # add title to the plot
plt.savefig('crime_types.png', dpi=100)

# %% [markdown]
# 
# # Conclusions
# 
# * For dispatch time analysis, three factors were used: hour, day of the week and month. According to the hours chart, most crimes occur during the morning (7am to 12pm). The beginning of the week was also the period most frequently (Monday, Tuesday and Wednesday). The analysis of the months shows that February and March, have lesser crime incidents.
# * Most of the crimes committed are indicated to be committed in and around Silver Spring district.
# * We also observed that the most crimes occur around residential places, and least crimes are commited in places like Bar/Club.
# * The most common crime is Larceny followed by drugs/marijuana possession.
# * The least crimes are crimes such as: homicide, damage property, etc.

# %% [markdown]
# # Further exploration needed on...
# 
# From the data it is still possible to dig deep and explore more parameters of which I cite the following ideas :
# 
# * Is it possible to determine the longest and shortest police response time? Does the response time vary from district to district?
# * It will be insightful and helpful to classify the types of crimes by whether they are violent or not.
# * Furthermore, a relevant information could be to know in which places there are more occurrences of crimes and in which shift these crimes are most common.
# * Further analysis based on year is needed to answer the low crime level of year 2016.
# * Will we arrive at different conclusion if we make analysis based on population density instead of district?

# %% [markdown]
# This notebook was created by [Dawit H. Hailu](https://www.linkedin.com/in/dawit-h-hailu-ph-d-4b7b8787/)

# %% [code]
