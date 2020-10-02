#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


import os
destdir = '../input/'
files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ]


# In[ ]:


files


# In[ ]:


#df2014 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2014.csv')
#df2015 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2015.csv')
df2016 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2016.csv', nrows = 100000)
#df2017 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2017.csv')
#df2018 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2018.csv')


# ## Take a look into the 2016 data

# In[ ]:


df2016.head(n=2)


# In[ ]:


df2016.shape


# So in the 2016 dataset there are about 10.6 million entries for parking ticket, and each entry has 51 columns.
# 
# Lets take a look at the number of unique values for each column name...

# In[ ]:


d = {'Unique Entry': df2016.nunique(axis = 0),
        'Nan Entry': df2016.isnull().any()}
pd.DataFrame(data = d, index = df2016.columns.values)


# As it turns out, the last 11 columns in this dataset has no entry. So we can ignore those columns, while carrying out any visualization operation in this dataframe.
# 
# Also if the entry does not have a **Plate ID** it is very hard to locate those cars. Therefore I am going to drop those rows as well.

# In[ ]:


drop_column = ['No Standing or Stopping Violation', 'Hydrant Violation',
               'Double Parking Violation', 'Latitude', 'Longitude',
               'Community Board', 'Community Council ', 'Census Tract', 'BIN',
               'BBL', 'NTA',
               'Street Code1', 'Street Code2', 'Street Code3','Meter Number', 'Violation Post Code',
                'Law Section', 'Sub Division', 'House Number', 'Street Name']
df2016.drop(drop_column, axis = 1, inplace = True)


# In[ ]:


drop_row = ['Plate ID']
df2016.dropna(axis = 0, how = 'any', subset = drop_row, inplace = True)


# Check if there is anymore rows left without a **Plate ID**.

# In[ ]:


df2016['Plate ID'].isnull().any()


# In[ ]:


df2016.shape


# # Create a sample data for visualization

# The cleaned dataframe has 10624735 rows and 40 columns. 
# 
# But this is still a lot of data points. I does not make sense to use all of them to get an idea of distribution of the data points. So for visualization I will use only 0.1% of the whole data. Assmuing that the entries are not sorted I pick my 0.1% data points from the main dataframe at random.

# In[ ]:


mini2016 = df2016.sample(frac = 0.1, replace = False)


# In[ ]:


mini2016.shape


# My sample dataset has about 10K data points, which I will use for data visualization. Using the whole dataset is unnecessary and time consuming.

# ## Barplot of 'Registration State'

# In[ ]:


x_ticks = mini2016['Registration State'].value_counts().index
heights = mini2016['Registration State'].value_counts()
y_pos = np.arange(len(x_ticks))
fig = plt.figure(figsize=(15,14)) 
# Create horizontal bars
plt.barh(y_pos, heights)
 
# Create names on the y-axis
plt.yticks(y_pos, x_ticks)
 
# Show graphic
plt.show()


# In[ ]:


pd.DataFrame(mini2016['Registration State'].value_counts()/len(mini2016)).nlargest(10, columns = ['Registration State'])


# You can see from the barplot above: in our sample ~77.67% cars are registered in state : **NY**. After that 9.15% cars are registered in state : **NJ**, followed by **PA**, **CT**, and **FL**.

# ## How the number of tickets given changes with each month?

# In[ ]:


month = []
for time_stamp in pd.to_datetime(mini2016['Issue Date']):
    month.append(time_stamp.month)
m_count = pd.Series(month).value_counts()

plt.figure(figsize=(12,8))
sns.barplot(y=m_count.values, x=m_count.index, alpha=0.6)
plt.title("Number of Parking Ticket Given Each Month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.ylabel("No. of cars", fontsize=16)
plt.show();


# So from the barplot above **March** and **October** has the highest number of tickets!

# ## How many parking tickets are given for each violation code?

# In[ ]:


violation_code = mini2016['Violation Code'].value_counts()

plt.figure(figsize=(16,8))
f = sns.barplot(y=violation_code.values, x=violation_code.index, alpha=0.6)
#plt.xticks(np.arange(0,101, 10.0))
f.set(xticks=np.arange(0,100, 5.0))
plt.title("Number of Parking Tickets Given for Each Violation Code", fontsize=16)
plt.xlabel("Violation Code [ X5 ]", fontsize=16)
plt.ylabel("No. of cars", fontsize=16)
plt.show();


# ## How many parking tickets are given for each body type?

# In[ ]:


x_ticks = mini2016['Vehicle Body Type'].value_counts().index
heights = mini2016['Vehicle Body Type'].value_counts().values
y_pos = np.arange(len(x_ticks))
fig = plt.figure(figsize=(15,4))
f = sns.barplot(y=heights, x=y_pos, orient = 'v', alpha=0.6);
# remove labels
plt.tick_params(labelbottom='off')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('Car models [Label turned off due to crowding. Too many types.]', fontsize=16);
plt.title('Parking ticket given for different type of car body', fontsize=16);


# In[ ]:


df_bodytype = pd.DataFrame(mini2016['Vehicle Body Type'].value_counts() / len(mini2016)).nlargest(10, columns = ['Vehicle Body Type'])


# Top 10 car body types that get the most parking tickets are listed below : 

# In[ ]:


df_bodytype


# In[ ]:


df_bodytype.sum(axis = 0)/len(mini2016)


# Top 10 vehicle body type includes 93.42% of my sample dataset.

# ## How many parking tickets are given for each vehicle make?

# Just for the sake of changing the flavor of visualization this time I will make a logplot of car no. vs make. In that case we will be able to see much smaller values in the same graph with larger values.

# In[ ]:


vehicle_make = mini2016['Vehicle Make'].value_counts()

plt.figure(figsize=(16,8))
f = sns.barplot(y=np.log(vehicle_make.values), x=vehicle_make.index, alpha=0.6)
# remove labels
plt.tick_params(labelbottom='off')
plt.ylabel('log(No. of cars)', fontsize=16);
plt.xlabel('Car make [Label turned off due to crowding. Too many companies!]', fontsize=16);
plt.title('Parking ticket given for different type of car make', fontsize=16);

plt.show();


# In[ ]:


pd.DataFrame(mini2016['Vehicle Make'].value_counts() / len(mini2016)).nlargest(10, columns = ['Vehicle Make'])


# ## Insight on violation time

# In the raw data the **Violaation Time** is in a format, which is non-interpretable using standard **to_datatime** function in pandas. We need to change it in a useful format so that we can use the data. After formatting we may replace the old **Violation Time ** column with the new one.

# In[ ]:


timestamp = []
for time in mini2016['Violation Time']:
    if len(str(time)) == 5:
        time = time[:2] + ':' + time[2:]
        timestamp.append(pd.to_datetime(time, errors='coerce'))
    else:
        timestamp.append(pd.NaT)
    

mini2016 = mini2016.assign(Violation_Time2 = timestamp)
mini2016.drop(['Violation Time'], axis = 1, inplace = True)
mini2016.rename(index=str, columns={"Violation_Time2": "Violation Time"}, inplace = True)


# So in the new **Violation Time** column the data is in **Timestamp** format.

# In[ ]:


hours = [lambda x: x.hour, mini2016['Violation Time']]


# In[ ]:


# Getting the histogram
mini2016.set_index('Violation Time', drop=False, inplace=True)
plt.figure(figsize=(16,8))
mini2016['Violation Time'].groupby(pd.TimeGrouper(freq='30Min')).count().plot(kind='bar');
plt.tick_params(labelbottom='on')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('Day Time', fontsize=16);
plt.title('Parking ticket given at different time of the day', fontsize=16);


# ## Parking ticket vs county

# In[ ]:


violation_county = mini2016['Violation County'].value_counts()

plt.figure(figsize=(16,8))
f = sns.barplot(y=violation_county.values, x=violation_county.index, alpha=0.6)
# remove labels
plt.tick_params(labelbottom='on')
plt.ylabel('No. of cars', fontsize=16);
plt.xlabel('County', fontsize=16);
plt.title('Parking ticket given in different counties', fontsize=16);


# ## Unregistered Vehicle?

# In[ ]:


sns.countplot(x = 'Unregistered Vehicle?', data = mini2016)


# In[ ]:


mini2016['Unregistered Vehicle?'].unique()


# ## Vehicle Year

# In[ ]:


pd.DataFrame(mini2016['Vehicle Year'].value_counts()).nlargest(10, columns = ['Vehicle Year'])


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(x = 'Vehicle Year', data = mini2016.loc[(mini2016['Vehicle Year']>1980) & (mini2016['Vehicle Year'] <= 2018)]);


# ## Violation In Front Of Or Opposite

# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(x = 'Violation In Front Of Or Opposite', data = mini2016);


# In[ ]:


# create data
names = mini2016['Violation In Front Of Or Opposite'].value_counts().index
size = mini2016['Violation In Front Of Or Opposite'].value_counts().values
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.figure(figsize=(8,8))
from palettable.colorbrewer.qualitative import Pastel1_7
plt.pie(size, labels=names, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:




