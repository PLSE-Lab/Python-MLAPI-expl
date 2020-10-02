#!/usr/bin/env python
# coding: utf-8

# Here I am going to analyze the Crime Incidents reports of Boston. The libraries used by me in this analysis is Python libraries which includes Pandas and Matplotlib.
# 

# In[ ]:




import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for doing Exploratory data analysis(EDA)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


crime = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding = 'unicode_escape') #create a data frame 
offense_codes=pd.read_csv('/kaggle/input/crimes-in-boston/offense_codes.csv', encoding = 'unicode_escape') #create a data frame 


# First I am loading the csv file into a pandas dataframe. For reading a file use read_csv function from pandas library.

# In[ ]:


crime.head()


# head() function is used to show first 5 rows of the dataframe. You can see any number of rows by just passing that number as an argument to head() function.

# In[ ]:


crime.columns


# dataframe.columns is used to see the name of the columns of the dataframe so that we can easily use them in analysis.
# Here we will see the list of columns the dataframe has.

# In[ ]:


crime.info()


# I am using dataframe.info method to see what is the data type of each column and how many non null entries they have and if datatype conversion required or neccessary for doing analysis.
# Here we have clean data and no data type change is required.

# In[ ]:


crime_by_district=crime.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)
crime_by_district


# In[ ]:


crime_by_district.plot(kind='bar')


# As in the above plot we can see B2 district has highest crime incidents or it is the most unsafe district to live in Boston, whereas A15 has the lowest crime incidents in Boston.
# 

# In[ ]:


crime_by_offenseCodeGroup=crime.groupby('OFFENSE_CODE_GROUP')['INCIDENT_NUMBER'].count()
plt.figure(figsize=(20,10))
crime_by_offenseCodeGroup.plot(kind='bar')


# From above graph, we can see that most of the incidents are registered against the 'Motor vehicle Accident response' and the second most crime is 'Larceny'.

# In[ ]:


crime_by_year=crime.groupby('YEAR')['INCIDENT_NUMBER'].count()
plt.figure(figsize=(20,10))
crime_by_year.plot(kind='bar' )


# The crime rate was high in 2018 has less crime incidents from 2016 & 2017 but it is more than in  2015. 

# In[ ]:


crime_by_month=crime.groupby('MONTH')['INCIDENT_NUMBER'].count()
plt.figure(figsize=(20,10))
crime_by_month.plot(kind='bar' )


# As we can see June , July and August have more crime incidents than the other months. 

# In[ ]:


crime_by_yearmonth=crime.groupby(['YEAR','MONTH'])['INCIDENT_NUMBER'].count()

crime_by_yearmonth.unstack(level=0).plot(kind='bar', subplots=True)


# Same can be stated as above graph that over 4 years June , july and August have more incidents than the other months in the year.

# In[ ]:


crime_by_week=crime.groupby('DAY_OF_WEEK')['INCIDENT_NUMBER'].count()
plt.figure(figsize=(20,10))
crime_by_week.plot(kind='bar' )


# Weekends(Saturday and Sunday) have less crime incidents than weekdays.  

# In[ ]:


crime_by_hour=crime.groupby('HOUR')['INCIDENT_NUMBER'].count()
plt.figure(figsize=(20,10))
crime_by_hour.plot(kind='bar' )


# 4PM to 6 PM is the most unsafe hour in Boston, most of the crime happened at this time.

# In[ ]:


crime.columns
crime.groupby('STREET')['INCIDENT_NUMBER'].count().sort_values(ascending= False).iloc[:10].plot(kind='barh')


# Washington street is the most unsafe street in the boston.

# In[ ]:


crime[(crime['STREET'] == 'WASHINGTON ST' )].groupby('HOUR')['INCIDENT_NUMBER'].count().plot(kind='barh')


# In washington street also, the most unsafe hour is 4PM to 6 PM.

# In[ ]:


crime[(crime['STREET'] == 'BLUE HILL AVE' )].groupby('HOUR')['INCIDENT_NUMBER'].count().plot(kind='barh')


# Blue Hill Ave the second most unsafe street also shows that the 4PM to 6 PM is the most unsafe hour.

# In[ ]:


crime['SHOOTING'].value_counts()


# There are 1019 shooting incidents in Boston.

# In[ ]:


crime[crime['SHOOTING'] == 'Y'].groupby('YEAR')['INCIDENT_NUMBER'].count().plot(kind='bar')


# In[ ]:


crime[crime['SHOOTING'] == 'Y'].groupby('STREET')['INCIDENT_NUMBER'].count().sort_values(ascending=False).iloc[:10].plot(kind='bar')


# From here also we can see washington street is not safe.

# **Conclusion**
# 
# By using pandas , I have analysed and visualized the Boston crime data and it turned out that B2 dristict has more crime incidents and Washington street is the most unsafe street in Boston. Incidents have higher count in weekdays than the weekends and June, July and August months have seen more incidents than other months and most of the incidents happened in 4PM to 6PM. 
