#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the pandas library as pd
import pandas as pd

try:
    #In order not to use(try except): Set warn_bad_lines to issue warnings about bad records
    # Read 'police.csv' into a DataFrame named ri
    ri = pd.read_csv('../input/dataset-of-traffic-stops-in-rhode-island/police.csv')
                     
    # Examine the head of the DataFrame
    print(ri.head())

except pd.io.common.CParserError:
    print("Your data contained rows that could not be parsed.")


# In[ ]:


Before beginning your analysis, it's important that you familiarize yourself with the dataset.
In this exercise, you'll read the dataset into pandas, examine the first few rows,
and then count the number of missing values.


# In[ ]:


# Count the number of missing values in each column
print(ri.isnull().sum())


# In[ ]:


Often, a DataFrame will contain columns that are not useful to your analysis. 
Such columns should be dropped from the DataFrame, to make it easier for you to focus on the remaining columns.

I'll drop the county_name column because it only contains missing values, 
and I'll drop the state column because all of the traffic stops took place in one state (Rhode Island).
Thus, these columns can be dropped because they contain no useful information.
The number of missing values in each column has been printed to the console for you.


# In[ ]:


print(ri.shape)

# Drop the 'county_name' column
ri.drop('county_name', axis=1, inplace=True)

# Examine the shape of the DataFrame (again)
print(ri.shape)


# In[ ]:


Dropping rows
When i know that a specific column will be critical to your analysis,
and only a small fraction of rows are missing a value in that column, it often makes sense to remove those rows from the dataset.

The driver_gender column will be critical to many of your analyses. 
Because only a small fraction of rows are missing driver_gender, we'll drop those rows from the dataset.


# In[ ]:


# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=['driver_gender'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame (again)
print(ri.shape)


# In[ ]:


# 2-second step is convert the types of columns to the suitable type


# In[ ]:


Fixing a data type
We saw that the is_arrested column currently has the object data type. 
we'll change the data type to bool, which is the most suitable type for a column containing True and False values.

Fixing the data type will enable us to use mathematical operations on the is_arrested column that would not be possible otherwise.


# In[ ]:


print(ri.dtypes)


# In[ ]:


# Change the data type of 'is_arrested' to 'bool'
ri['is_arrested'] = ri.is_arrested.astype('bool')

# Check the data type of 'is_arrested' 
print(ri.is_arrested.dtype)


# In[ ]:


Combining object columns
Currently, the date and time of each traffic stop are stored in separate object columns: stop_date and stop_time.

I'll combine these two columns into a single column, and then convert it to datetime format.


# In[ ]:


# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = ri.stop_date.str.cat(ri.stop_time , sep=' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame
print(ri.dtypes)


# In[ ]:


# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns
print(ri.columns)


# In[ ]:


Examining traffic violations
Before comparing the violations being committed by each gender,
you should examine the violations committed by all drivers to get a baseline understanding of the data.

I'll count the unique values in the violation column, and then separately express those counts as proportions.


# In[ ]:


# Count the unique values in 'violation'
print(ri.violation.value_counts())
print('')
print(ri.violation.value_counts().sum())
print('')
# Express the counts as proportions
print(ri.violation.value_counts(normalize=True))


# In[ ]:


Comparing violations by gender
The question we're trying to answer is whether male and female drivers tend to commit different types of traffic violations.


# In[ ]:


# Create a DataFrame of female drivers
female = ri[ri.driver_gender=='F']

# Create a DataFrame of male drivers
male = ri[ri.driver_gender=='M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize=True))
print('')

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize=True))


# In[ ]:


Comparing speeding outcomes by gender
When a driver is pulled over for speeding, many people believe that gender has an impact 
on whether the driver will receive a ticket or a warning.

First, i'll create two DataFrames of drivers who were stopped for speeding: one containing females and the other containing males.

Then, for each gender, i'll use the stop_outcome column to calculate what percentage of stops resulted in a "Citation" (meaning a ticket) versus a "Warnin


# In[ ]:


# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender=='F')& (ri.violation=='Speeding' )]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender=='M')& (ri.violation=='Speeding' )]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize=True))

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize=True))


# In[ ]:


Calculating the search rate
During a traffic stop, the police officer sometimes conducts a search of the vehicle.
I'll calculate the percentage of all stops in the ri DataFrame that result in a vehicle search, also known as the search rate.


# In[ ]:


# Check the data type of 'search_conducted '
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize=True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())


# In[ ]:


# Calculate the search rate for both groups simultaneously
print(ri.groupby('driver_gender').search_conducted.mean())


# In[ ]:


Adding a second factor to the analysis
Even though the search rate for males is much higher than for females,
it's possible that the difference is mostly due to a second factor.

For example, you might hypothesize that the search rate varies by violation type,
and the difference in search rate between males and females is because they tend to commit different violations.

You can test this hypothesis by examining the search rate for each combination of gender and violation.
If the hypothesis was true, you would find that males and females are searched at about the same rate for each violation.
Find out below if that's the case!


# In[ ]:


# Calculate the search rate for each combination of gender and violation
print(ri.groupby(['driver_gender','violation']).search_conducted.mean())


# In[ ]:


# Reverse the ordering to group by violation before gender
print(ri.groupby(['violation','driver_gender']).search_conducted.mean())


# In[ ]:


ri.search_type.unique()


# In[ ]:


Counting protective frisks
During a vehicle search, the police officer may pat down the driver to check if they have a weapon.
This is known as a "protective frisk."

I'll first check to see how many times "Protective Frisk" was the only search type.
Then,i'll use a string method to locate all instances in which the driver was frisked.


# In[ ]:


# Count the 'search_type' values
print(ri.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)

# Check the data type of 'frisk'
print(ri.frisk.dtype)

# Take the sum of 'frisk'
print(ri.frisk.sum())


# In[ ]:


ri['inventory'] = ri.search_type.str.contains('Inventory', na=False)
ri.inventory.sum()


# In[ ]:


Comparing frisk rates by gender
I'll compare the rates at which female and male drivers are frisked during a search.
get_ipython().set_next_input('Are males frisked more often than females, perhaps because police officers consider them to be higher risk');get_ipython().run_line_magic('pinfo', 'risk')

Before doing any calculations, it's important to filter the DataFrame to only include the relevant subset of data,
namely stops in which a search was conducted.


# In[ ]:


# Create a DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())


# In[ ]:


#Time series analysis 


# In[ ]:


ri.head()


# In[ ]:


Calculating the hourly arrest rate
When a police officer stops a driver, a small percentage of those stops ends in an arrest.
This is known as the arrest rate. In this exercise, you'll find out whether the arrest rate varies by time of day.

First, you'll calculate the arrest rate across all stops in the ri DataFrame. 
Then, you'll calculate the hourly arrest rate by using the hour attribute of the index. The hour ranges from 0 to 23, in which:

0 = midnight
12 = noon
23 = 11 PM


# In[ ]:


# Calculate the overall arrest rate
print(ri.is_arrested.mean())

# Calculate the hourly arrest rate
# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()
print(hourly_arrest_rate)


# In[ ]:


I'll create a line plot from the hourly_arrest_rate object.
A line plot is appropriate in this case because you're showing how a quantity changes over time.

This plot should help me to spot some trends that may not have been obvious when examining the raw numbers!


# In[ ]:


# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()


# In[ ]:



# Calculate the annual rate of drug-related stops
print(ri.drugs_related_stop.resample('A').mean())


# In[ ]:


# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()

# Display the plot
plt.show()


# In[ ]:


Plotting drug-related stops
In a small portion of traffic stops, drugs are found in the vehicle during a search. 
I'll assess whether these drug-related stops are becoming more common over time.

The Boolean column drugs_related_stop indicates whether drugs were found during a given stop.
I'll calculate the annual drug rate by resampling this column,
and then I'll use a line plot to visualize how the rate has changed over time.


# In[ ]:


# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()
# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate,annual_search_rate], axis='columns')
# Create subplots from 'annual'
annual.plot(subplots=True)
plt.show()


# In[ ]:


# Create a frequency table of driver race and driver gender
table = pd.crosstab(ri.driver_race,ri.driver_gender)
print(table)


# In[ ]:


# Create a bar plot of 'table'
table.plot(kind='bar')
table.plot(kind='bar',stacked=True)
# Display the plot
plt.show()


# In[ ]:


ri.is_arrested.dtype


# In[ ]:


search_rate=ri.groupby('violation').search_conducted.mean()
print(search_rate)


# In[ ]:


#to make it more easier to read 
search_rate.sort_values()


# In[ ]:


search_rate.sort_values().plot(kind='bar')
plt.show()


# In[ ]:


search_rate.sort_values().plot(kind='barh')
plt.show()


# In[ ]:




