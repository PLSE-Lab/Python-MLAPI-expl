#!/usr/bin/env python
# coding: utf-8

# # Objective

# To analyze Uber and Lyft datasets.

# In[ ]:


# let's import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

get_ipython().run_line_magic('config', "Inlinebackground.figureFormat='retina'")
sns.set(font_scale=1.5)
pd.options.display.max_rows = 200
pd.options.display.max_columns = 200


# In[ ]:


# let's load the datasets
# cab data
cab_data = pd.read_csv(r'cab_rides.txt', encoding='utf-16')
# weather data
weather_data = pd.read_csv(r'weather.txt', encoding='utf-16')


# In[ ]:


cab_data.head(3)


# In[ ]:


weather_data.head(3)


# In[ ]:


cab_data.info() # basic descr.


# ## Atrribute information

# **Dependent variables**:       
# 
#     Distance - distance between source and destination             
#     Cab_type - Uber or Lyft                 
#     Time_stamp - epoch time when data was queried              
#     Destination - destination of the ride              
#     Source - the starting point of the ride              
#     Surge_multiplier - the multiplier by which price was increased, default 1              
#     Id - unique identifier              
#     Product_id - uber/lyft identifier for cab-type              
#     Name - Visible type of the cab eg: Uber Pool, UberXL              
# 
# **Target variable**:               
# 
#     Price - price estimate for the ride in USD

# ### Data Type      
# **Object**
# 
#     - Cab type
#     - Destination
#     - Source
#     - Id
#     - Product Id
#     - Name
#     
# **Numeric**       
#     - Distance
#     - Time stamp
#     - Price
#     - Surge Multiplier

# ### Feature category        
# **Categorical**       
#     - Cab type
#     - Destination
#     - Source
#     - Product Id
#     - Name
#     - Id
#     - Surge Multiplier
#     
# **Continuous**     
#     - Distance
#     - Time stamp
#     - Price

# # Exploratory Data Analysis

# In[ ]:


# let's impute the unix epoch time to standard date time format
cab_data['time_stamp'] = pd.to_datetime(cab_data['time_stamp'], unit='ms')
cab_data['date'] = cab_data['time_stamp'].dt.date  # extract date
cab_data['hour'] = cab_data['time_stamp'].dt.hour  # extract hour

cab_data.drop('time_stamp', axis=1, inplace=True)  # drop time_stamp feature

# before doing EDA, let's split the dataset into Uber and Lyft
uber = cab_data[cab_data['cab_type']=='Uber']
lyft = cab_data[cab_data['cab_type']=='Lyft']

cab_data.head(3)


# ## Univariate analysis

# ### Continuous variables

# **Distance**

# In[ ]:


overall = cab_data['distance'].describe() # measure of central tendency
overall


# In[ ]:


lyft_distance = lyft['distance'].describe()
uber_distance = uber['distance'].describe()


# In[ ]:


df = pd.DataFrame({'Overall': overall.values,
                  'Lyft': lyft_distance.values,
                  'Uber': uber_distance.values}, index= ['Count', 'Mean', 'Std. Dev.', 'Min', '25%', '50%', '75%', 'Max'])
df


# In[ ]:


# df.to_csv(r'C:\Users\gokul\Downloads\distance_metrics.csv')


# In[ ]:


def calculate_mop(**kwargs):
    """ function to calculate and display the measures of dispersion."""
    for name, df in kwargs.items():
        print(name, '\n')
        print(f'Standard deviation:     {df.std()}')
        print(f'Skewness:               {df.skew()}')
        print(f'Kurtosis:               {df.kurtosis()}\n')


# In[ ]:


calculate_mop(Lyft= lyft['distance'], Uber= uber['distance'])


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.distplot(lyft['distance'], ax=ax1, kde=True)
ax1.set_title('Distribution of distance in Lyft', fontsize=20)
ax1.set_ylim(0, 0.6)
a = sns.distplot(uber['distance'], ax=ax2)
ax2.set_title('Distribution of distance in Uber', fontsize=20)
ax2.set_ylim(0, 0.6)


# In[ ]:


# a.figure.savefig(r'C:\Users\gokul\Downloads\distance.jpg')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
sns.boxplot(lyft['distance'], ax=ax1)
ax1.set_title('Lyft', fontsize=15)
ax1.set_xlim(0, 8)
sns.boxplot(uber['distance'], ax=ax2)
ax2.set_title('Uber', fontsize=15)


# In[ ]:


# lyft[lyft['distance']<0.3] # can we remove records below 0.3 as cancellation records


# In[ ]:


# uber[uber['distance']<0.25].sort_values(by='distance', ascending=False).head(30)


# From the above graphs, we can see that most of the rides are in the range of approximately 0.5 to 5.5 miles. 
# The distribution is slightly right skewed in the both Lyft and Uber.
# Distance in Lyft is more dispersed than Uber.           
# Both the data contains outliers, due to certain weather conditions riders have to travel extra distance than usual,
# and occassionally riders tend to travel long distances.

# **Price (Target variable)**

# In[ ]:


overall = cab_data['price'].describe()
overall # measure of central tendency


# In[ ]:


uber_price = uber['price'].describe()
uber_price


# In[ ]:


lyft[lyft.price<2.9].shape


# In[ ]:


lyft_price = lyft['price'].describe()
lyft_price


# In[ ]:


uber.price.sum(), lyft.price.sum()


# In[ ]:


df = pd.DataFrame({'Overall': overall.values,
                  'Lyft': lyft_price.values,
                  'Uber': uber_price.values}, index= ['Count', 'Mean', 'Std. Dev.', 'Min', '25%', '50%', '75%', 'Max'])
df


# In[ ]:


# df.to_csv(r'C:\Users\gokul\Downloads\metrics.csv')


# In[ ]:


calculate_mop(Lyft= lyft['price'], Uber= uber['price']) # measure of dispersion


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
a = sns.distplot(lyft['price'], ax=ax1)
ax1.set_title('Distribution of price in Lyft', fontsize=20)
ax1.set(xlabel='Price')
ax1.set_ylim(0, 0.12)
b =sns.distplot(uber[~uber['price'].isnull()]['price'], ax=ax2)
ax2.set_title('Distribution of price in Uber', fontsize=20)


# In[ ]:


# a.figure.savefig(r'C:\Users\gokul\Downloads\price.jpg')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
sns.boxplot(lyft['price'], ax=ax1)
ax1.set_title('Lyft', fontsize=15)

sns.boxplot(uber[~uber['price'].isnull()]['price'], ax=ax2)
ax2.set_title('Uber', fontsize=15)
ax2.set_xlim(0, 100)


# **Outliers**       
# These outliers are due to use of high-end cars and high surge multipliers. So, we decided to keep it.

# In[ ]:


lyft[(lyft['price']>40)].head(3)


# In[ ]:


uber[(uber['price']>40)].head(3)


# In[ ]:


# a = uber[uber['price']<40].groupby(by=['source', 'destination']).median()#.head(10)
# a


# In[ ]:


# a.to_csv(r'C:\Users\gokul\Downloads\tab.csv')


# In[ ]:


uber[uber['price']>40].groupby(by=['source', 'destination']).mean().head(10)


# **Outliers handling**      
# We could see that, some rides higher price than usual for different car models. From our analysis, we came to know,
# duration of the trip also has impact on the price, and we do not posses data regarding the
# duration of the trip, so we cannot remove these outliers.

# The price distribution is right skewed, from the boxplot we could see the outliers present in the data.         
# On average, the price range varies from 5 to 40 US dollars,           
# Presence of outliers is due to factors such as use of luxury or premium cars for rides, travelling in high traffic city 
# and bad weather conditions.

# **Categorical variables**

# **Cab type**

# In[ ]:


cab_data['cab_type'].value_counts() # frequency count


# In[ ]:


cab_data['cab_type'].value_counts(normalize=True) # percentage of values


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot('cab_type', data=cab_data)
plt.title('Frequency of Uber and Lyft data', fontsize=15)


# The dataset contains relatively high proportion of Uber data, with both having records more than 300,000 data points.

# **Car model**

# In[ ]:


lyft['name'].value_counts() # frequency count


# In[ ]:


uber['name'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['name'], ax=ax1)
ax1.set_title('Frequency count of car models in Lyft', fontsize=20)
sns.countplot(uber['name'], ax=ax2)
ax2.set_title('Frequency count of car models  in Uber', fontsize=20)


# From the frequency plot, we could almost all the car models are used in similar frequency.

# **Source**

# In[ ]:


lyft['source'].value_counts() # frequency count


# In[ ]:


uber['source'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['source'], ax=ax1)
ax1.set_title('Frequency count of different source location in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)
# ax1.set_ylim(0, 25000)
sns.countplot(uber['source'], ax=ax2)
ax2.set_title('Frequency count of different source location  in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)


# **Destination**

# In[ ]:


lyft['destination'].value_counts()


# In[ ]:


uber['destination'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['destination'], ax=ax1)
ax1.set_title('Frequency count of different destination location in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)
ax1.set_ylim(0, 30000)
sns.countplot(uber['destination'], ax=ax2)
ax2.set_title('Frequency count of different destination location  in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)


# **Product Id**

# In[ ]:


lyft['product_id'].value_counts()


# In[ ]:


uber['product_id'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['product_id'], ax=ax1)
ax1.set_title('Frequency count of different Product names in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)
sns.countplot(uber['product_id'], ax=ax2)
ax2.set_title('Frequency count of different Product names in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)


# In[ ]:





# **Surge multiplier**

# In[ ]:


lyft['surge_multiplier'].value_counts() # frequency count


# In[ ]:


uber['surge_multiplier'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['surge_multiplier'], ax=ax1)
ax1.set_title('Frequency count of different surge multipliers in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)
ax1.set_ylim(0, 350000)
sns.countplot(uber['surge_multiplier'], ax=ax2)
ax2.set_title('Frequency count of different surge multipliers in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)
ax2.set_ylim(0, 350000)


# From the above charts, we see that, there is a variety of surge mulipliers in Lyft, whereas in Uber, there is only one 
# surge multiplier.     
# This has increased the number of riders in Uber, compared to Lyft.

# **Hour**

# In[ ]:


lyft['hour'].value_counts()


# In[ ]:


uber['hour'].value_counts()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.countplot(lyft['hour'], ax=ax1)
ax1.set_title('Frequency count of different destination location in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)
ax1.set_ylim(0, 17500)
sns.countplot(uber['hour'], ax=ax2)
ax2.set_title('Frequency count of different destination location  in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)
ax2.set_ylim(0, 17500)


# Hour has almost similar distribution for both Uber and Lyft.  This is due to fact that cab riders has the option to choose
# between Lyft and Uber for their customers.       
# We can see that there is high usage in the hours from 10 to evening 7, this makes sense as this the office hours,
# where people likely to travel frequently.

# 

# ## Bi-variate Analysis

# ### Continuous & Continuous

# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))
sns.scatterplot(lyft['distance'], lyft['price'], ax=ax1)
ax1.set_title('Price vs Distance in Lyft', fontsize=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)
sns.scatterplot(uber['distance'], uber['price'], ax=ax2)
ax2.set_title('Price vs Distance in Uber', fontsize=20)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)
ax2.set_ylim(0, 100)


# In[ ]:


lyft['distance'].corr(lyft['price'])


# In[ ]:


uber['distance'].corr(uber['price'])


# Distance is one of the important factors, which drives the price of the rides.  We could see that there is a positive 
# correlation in the graph, with the presence of outliers as we saw before, because of the use of luxury car models and 
# bad weather conditions.

# ### Categorical & Categorical:

# **Cab type vs Source**

# In[ ]:


a= pd.crosstab(cab_data['cab_type'], cab_data['source'])
a


# In[ ]:


pd.crosstab(cab_data['cab_type'], cab_data['source'], normalize=True)


# In[ ]:


plt.figure(figsize=(10,7))
pd.crosstab(cab_data['cab_type'], cab_data['source']).plot.bar(stacked=True, figsize=(20,7), rot=0)


# **Cab type vs Destination**

# In[ ]:


pd.crosstab(cab_data['cab_type'], cab_data['destination'])


# In[ ]:


pd.crosstab(cab_data['cab_type'], cab_data['destination'], normalize=True)


# In[ ]:


plt.figure(figsize=(10,7))
pd.crosstab(cab_data['cab_type'], cab_data['destination']).plot.bar(stacked=True, figsize=(20,7), rot=0)


# In[ ]:





# **Cab type vs Surge multiplier**

# In[ ]:


pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier'])


# In[ ]:


pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier'], normalize=True)


# In[ ]:


plt.figure(figsize=(10,7))
pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier']).plot.bar(stacked=True, figsize=(20,7), rot=0)


# In[ ]:





# **Car model vs Source**

# In[ ]:


pd.crosstab(lyft['name'], lyft['source'])


# In[ ]:


plt.figure(figsize=(10,7))
pd.crosstab(lyft['name'], lyft['source']).plot.bar(stacked=True, figsize=(20,7), rot=0)


# In[ ]:


cab_data['source'].nunique(), cab_data['destination'].nunique()


# In[ ]:


pd.crosstab(uber['name'], uber['source'])


# In[ ]:





# **Car model vs Surge multiplier**

# In[ ]:


pd.crosstab(lyft['name'], lyft['surge_multiplier'])


# We can see that, the surge multiplier doesn't vary for different car models in Lyft, except for few exceptions
# in Lyft XL, which is a high-end car services of Lyft.

# In[ ]:


pd.crosstab(lyft['name'], lyft['surge_multiplier'], normalize=True)


# In[ ]:


plt.figure(figsize=(10,7))
pd.crosstab(lyft['name'], lyft['surge_multiplier']).plot.bar(stacked=True, figsize=(20,7), rot=0)


# In[ ]:


pd.crosstab(uber['name'], uber['surge_multiplier'])


# In[ ]:





# **Source vs Destination**

# In[ ]:


pd.crosstab(uber['source'], uber['destination'])


# This table shows that, we cannot hope to go to any place within Boston city, 
# As we can see that, these destinations are within 3 miles from source. Are people in Massachussettes, choose to skip cabs 
# within 3 miles, we need to analyze further to make any claims.

# In[ ]:





# ## Weather dataset

# In[ ]:


weather_data.info() # basic info


# **Attribute Information**
# 
# Location - Location name              
# Clouds              
# Pressure - pressure in mb              
# Rain - rain in inches for the last hr              
# Time_stamp - epoch time when row data was collected              
# Humidity - humidity in %              
# Wind - wind speed in mph

# ### Data Type      
# **Object**
# 
#     - location
#     
# **Numeric**       
#     - temp
#     - clouds
#     - pressure
#     - rain
#     - time stamp
#     - humidity
#     - wind

# ### Feature category        
# **Categorical**       
#     - location
#     
# **Continuous**     
#     - temp
#     - time stamp
#     - clouds
#     - pressure
#     - rain
#     - humidity
#     - wind

# ### Univariate Analysis

# In[ ]:


# let's impute the unix epoch time to standard date time format
weather_data['time_stamp'] = pd.to_datetime(weather_data['time_stamp'], unit='s')
weather_data['date'] = weather_data['time_stamp'].dt.date
weather_data['hour'] = weather_data['time_stamp'].dt.hour

weather_data.drop('time_stamp', axis=1, inplace=True)

weather_data.head(3)


# **Continuous**

# **Temperature**

# In[ ]:


weather_data['temp'].describe()


# In[ ]:


calculate_mop(Temperature = weather_data['temp'])


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(weather_data['temp'])
plt.title('Distribution of Temperature', fontsize=15)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data['temp'])
plt.title('Distribution of Temperature', fontsize=15)


# In[ ]:


weather_data[weather_data.temp < 26].date.value_counts()


# In[ ]:


weather_data[weather_data.temp > 53].date.value_counts()


# According to this website https://www.timeanddate.com/weather/usa/boston/historic?month=12&year=2018,
# the data is legitimate. We can keep the outliers.

# In[ ]:


weather_data[weather_data['temp']<27].shape


# In[ ]:


weather_data[weather_data['temp']>52.5].shape


# Temperature is slighty left skewed, and it makes sense, because this data is collected around the month of november,
# although occassionally we can see high temperature as well in locations such as Financial district, Boston university
# and Back bay.

# In[ ]:


weather_data[weather_data.temp>53].location.value_counts()


# **Clouds**

# In[ ]:


weather_data['clouds'].describe()


# In[ ]:


calculate_mop(Clouds=weather_data['clouds'])


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(weather_data['clouds'])
plt.title('Distribution of Clouds', fontsize=15)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data['clouds'])
plt.title('Distribution of Clouds', fontsize=15)


# **Pressure**

# In[ ]:


weather_data['pressure'].describe()


# In[ ]:


calculate_mop(Pressure=weather_data['pressure'])


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(weather_data['pressure'])
plt.title('Distribution of Pressure', fontsize=15)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data['clouds'])
plt.title('Distribution of Clouds', fontsize=15)


# **Rain**

# In[ ]:


weather_data['rain'].describe()


# In[ ]:


calculate_mop(Rain=weather_data['rain'])


# In[ ]:


plt.figure(figsize=(8,5))
a = sns.distplot(weather_data[~weather_data['rain'].isnull()]['rain'])
plt.title('Distribution of Rain', fontsize=15)


# In[ ]:


# a.figure.savefig(r'C:\Users\gokul\Downloads\rain.jpg')


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data[~weather_data['rain'].isnull()]['rain'])
plt.title('Distribution of Rain', fontsize=15)


# In[ ]:


weather_data[weather_data['rain']>0.13].date.value_counts()


# **Outliers handling**

# According to this website https://www.timeanddate.com/weather/usa/boston/historic?month=12&year=2018, 
# there was rain only on specific days at particular hours.     
# So, we've decided not to remove these outliers.

# Rain is right skewed, it is understandable as this data is collected for 17 days, only in few days, there was rain
# in Boston city.

# **Humidity**

# In[ ]:


weather_data['humidity'].describe()


# In[ ]:


calculate_mop(Humidity = weather_data['humidity'])


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(weather_data['humidity'])
plt.title('Distribution of Humidity', fontsize=15)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data['humidity'])
plt.title('Distribution of Humidity', fontsize=15)


# **Wind**

# In[ ]:


weather_data['wind'].describe()


# In[ ]:


calculate_mop(Wind=weather_data['wind'])


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(weather_data['wind'])
plt.title('Distribution of Wind', fontsize=15)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(weather_data['wind'])
plt.title('Distribution of Wind', fontsize=15)


# 

# **Category**

# **Location**

# In[ ]:


weather_data['location'].value_counts()


# In[ ]:





# ### Bi-variate analysis

#     - temp
#     - time stamp
#     - clouds
#     - pressure
#     - rain
#     - humidity
#     - wind

# **Continuous & Continuous**

# In[ ]:


sns.pairplot(weather_data[['temp', 'clouds', 'rain', 'pressure', 'humidity', 'wind']])


# **Categorical & Continuous**

# In[ ]:


sns.boxplot(weather_data['temp'], y=weather_data['location'], orient='h')


# In[ ]:


sns.boxplot(weather_data['rain'], y=weather_data['location'], orient='h')


# Since the data is collected within Boston city, the rain has effect on all the locations within the city,
# among those locations such as Financial District, Haymarket Square and North end have experienced high rainfall.

# In[ ]:


sns.boxplot(weather_data['clouds'], y=weather_data['location'], orient='h')


# In[ ]:


sns.boxplot(weather_data['pressure'], y=weather_data['location'], orient='h')


# In[ ]:


sns.boxplot(weather_data['humidity'], y=weather_data['location'], orient='h')


# In[ ]:


sns.boxplot(weather_data['wind'], y=weather_data['location'], orient='h')


# In[ ]:


weather_data['date'].value_counts().sort_index()


# In[ ]:





# ### Missing Value Treatment

# #### Cab rides dataset

# In[ ]:


nrows, ncols = cab_data.shape
print(f'Cab ride dataset contains {nrows} rows and {ncols} columns.')


# In[ ]:


mv  = cab_data.isnull().sum().sum()
prop = round(((mv/cab_data.shape[0]) * 100),3)
print(f'Cab ride dataset contains {mv} missing values, which is {prop} % of whole data.')


# In[ ]:


cab_data.isnull().sum()


# The missing values are present in the price (target) column.

# Let's check the missing values are occuring at random.

# In[ ]:


# let's check the cab type
cab_data[cab_data['price'].isnull()]['cab_type'].value_counts() 


# We can see that, the missing values are present only in the Uber data. Let's also check the car type and model.

# In[ ]:


cab_data[cab_data['cab_type']=='Uber'].name.value_counts()


# In[ ]:


cab_data[cab_data['price'].isnull()]['name'].value_counts() # car model


# We have checked the Uber official website for different car models, and the missing values are less than 10%
# of the original data, we have decided to drop the records containing missing values in the target price feature.

# In[ ]:


# let's drop those records
cab_data.dropna(how='any', inplace=True)
nrows, ncols = cab_data.shape
print(f'Now the dataset contains {nrows} rows and {ncols} columns.')

uber = cab_data[cab_data['cab_type']=='Uber']
lyft = cab_data[cab_data['cab_type']=='Lyft']


# In[ ]:


cab_data.isnull().sum().sum() # check for missing values


# In[ ]:


# cab_data.to_csv('C:\Users\gokul\Downloads\cabs.csv')


# In[ ]:





# **Weather dataset**

# In[ ]:


nrows, ncols = weather_data.shape
print(f'Cab ride dataset contains {nrows} rows and {ncols} columns.')


# In[ ]:


mv  = weather_data.isnull().sum().sum()
prop = round(((mv/weather_data.shape[0]) * 100),3)
print(f'Cab ride dataset contains {mv} missing values, which is {prop} % of whole data.')


# In[ ]:


weather_data.isnull().sum()


# We can see that, in the 'rain' feature 85.75 % of the data is missing. After checking the weather conditions from the 
# official website of Boston and through observation, we can infer that,      
# The missing values are due to unobserved input variable, that is there was no rain observed on those particular hour.        
# 
# So, we have decided to impute the missing values with zero, which denotes no rain.

# In[ ]:


# let's impute the missing values in the 'rain' column with 0
weather_data['rain'].fillna(0, inplace=True)


# In[ ]:


weather_data.isnull().sum().sum() # check for missing values


# ### Base Model

# In[ ]:


# weather data supposed to contain 1 record per hour, since it has more than one values for few hours, 
# we took groupby average
weather_data = weather_data.groupby(['location','date', 'hour']).mean()
weather_data.reset_index(inplace=True)


# In[ ]:


merged_data = pd.merge(cab_data, weather_data, how='left', left_on=['source', 'date', 'hour'],
        right_on=['location', 'date', 'hour'])


# In[ ]:


merged_data.info()


# We could see that there is null values in the data.

# In[ ]:


merged_data[merged_data.temp.isnull()].groupby(['source', 'date', 'hour']).mean().head(6)


# Weather data doesn't have records for this particular dates and hours, let's impute these values with the previous values.

# In[ ]:


df1 = weather_data.loc[
    (weather_data['date']==datetime.date(2018, 11, 28)) &
    (weather_data['hour']==0)]

df2 = weather_data.loc[
    (weather_data['date']==datetime.date(2018, 12, 4)) &
    (weather_data['hour']==5)]
df3 = weather_data.loc[
    (weather_data['date']==datetime.date(2018, 11, 28)) &
    (weather_data['hour']==2)]
df4 = weather_data.loc[
    (weather_data['date']==datetime.date(2018, 12, 4)) &
    (weather_data['hour']==7)]


lookup = pd.concat([df1, df2, df3, df4])
lookup = lookup.groupby(['hour', 'location', 'date']).mean().reset_index()
df5 = weather_data.loc[
    (weather_data['date']==datetime.date(2018, 12, 18)) &
    (weather_data['hour']==18)]

lookup = pd.concat([lookup, df5])
lookup['hour'] += 1
lookup.reset_index(inplace=True)


# In[ ]:


weather_data = pd.concat([weather_data, lookup], ignore_index=True) 


# In[ ]:


weather_data.shape


# In[ ]:


cab_data = pd.merge(cab_data, weather_data, how='left',
                left_on=['source', 'date', 'hour'],
                right_on=['location', 'date', 'hour'])


# In[ ]:


cab_data.info()


# In[ ]:


cab_data.drop('index', axis=1, inplace=True)


# In[ ]:


cab_data.shape, cab_data.drop_duplicates().shape


# In[ ]:





# In[ ]:


# drop unnecessary features
cab_data = cab_data.drop(['id', 'product_id', 'location', 'date'], axis=1)


# In[ ]:


corr_m = cab_data.corr()


# In[ ]:


x = np.tri(corr_m.shape[0],k=-1)


# In[ ]:


plt.figure(figsize=(15,10))
a = sns.heatmap(corr_m, annot=True, mask=x)


# In[ ]:


# a.figure.savefig(r'C:\Users\gokul\Downloads\corr.jpg')


# In[ ]:


# Initial data preparation


# In[ ]:


data = cab_data.drop(['price', 'surge_multiplier'], axis=1) # we are dropping surge multiplier, to avoid data leak
labels = cab_data['price'].copy()


# In[ ]:


# model building libraries

# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


uber = cab_data[cab_data['cab_type']=='Uber']
uber.reset_index(inplace=True)
uber.drop('index', axis=1, inplace=True)
lyft = cab_data[cab_data['cab_type']=='Lyft']
lyft.reset_index(inplace=True)
lyft.drop('index', axis=1, inplace=True)


# In[ ]:


uber.drop('cab_type', axis=1, inplace=True)
lyft.drop('cab_type', axis=1, inplace=True)


# In[ ]:


lyft_data = lyft.copy() # backups
uber_data = uber.copy()


# In[ ]:


uber_data.head()


# In[ ]:


uber.info()


# **Categorical columns encoding**

# **One Hot encoding**

# In[ ]:


ohe = OneHotEncoder()
car_type = pd.DataFrame(ohe.fit_transform(uber[['name']]).toarray(), columns=sorted(list(uber['name'].unique())))
source = pd.DataFrame(ohe.fit_transform(uber[['source']]).toarray(), 
                       columns=['src_'+loc for loc in sorted(list(uber['source'].unique()))])
destination = pd.DataFrame(ohe.fit_transform(uber[['destination']]).toarray(), 
                           columns=['dest_'+loc for loc in sorted(list(uber['destination'].unique()))])


# In[ ]:


ohe = OneHotEncoder()
lyft_car_type = pd.DataFrame(ohe.fit_transform(lyft[['name']]).toarray(), columns=sorted(list(lyft['name'].unique())))
lyft_source = pd.DataFrame(ohe.fit_transform(lyft[['source']]).toarray(),
                           columns=['src_'+loc for loc in sorted(list(lyft['source'].unique()))])
lyft_destination = pd.DataFrame(ohe.fit_transform(lyft[['destination']]).toarray(),
                                columns=['dest_'+loc for loc in sorted(list(lyft['destination'].unique()))])


# In[ ]:


uber = pd.concat([uber, car_type, source, destination], axis=1)
uber.drop(['name', 'source', 'destination'], axis=1, inplace=True)


# In[ ]:


lyft = pd.concat([lyft, lyft_car_type, lyft_source, lyft_destination], axis=1)
lyft.drop(['name', 'source', 'destination'], axis=1, inplace=True)


# **Label Encoding**

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


uber_le = uber_data.copy()
lyft_le = lyft_data.copy()

lb = LabelEncoder()

uber_le['name'] = lb.fit_transform(uber_data['name'])
uber_le['source'] = lb.fit_transform(uber_data['source'])
uber_le['destination'] = lb.fit_transform(uber_data['destination'])

lyft_le['name'] = lb.fit_transform(lyft_le['name'])
lyft_le['source'] = lb.fit_transform(lyft_le['source'])
lyft_le['destination'] = lb.fit_transform(lyft_le['destination'])


# In[ ]:


uber_leX = uber_le.drop(['price', 'surge_multiplier'], axis=1)
uber_ley = uber_le['price'].copy()

lyft_leX = lyft_le.drop(['price', 'surge_multiplier'], axis=1)
lyft_ley = lyft_le['price'].copy()


# In[ ]:


uber_X = uber.drop(['price', 'surge_multiplier'], axis=1)
uber_y = uber['price'].copy()


# In[ ]:


lyft_X = lyft.drop(['price', 'surge_multiplier'], axis=1)
lyft_y = lyft['price'].copy()


# In[ ]:


uber_leX.shape


# In[ ]:


lyft_leX.shape


# In[ ]:


import statsmodels.api as sm


# **Uber base model**

# In[ ]:


x_constant = sm.add_constant(uber_X)
uber_model = sm.OLS(uber_y, x_constant).fit()
uber_model.summary()


# We could see that Wind doesn't have significant impact on the price column.

# **Lyft base model**

# In[ ]:


x_constant = sm.add_constant(lyft_X)
lyft_model = sm.OLS(lyft_y, x_constant).fit()
lyft_model.summary()


# We could see that all the features are significant in the Lyft model, according to the p-values.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(uber_X, uber_y, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[ ]:


train_pred = lin_reg.predict(X_train)
print(f'Train score {np.sqrt(mean_squared_error(y_train, train_pred))}')

predicted = lin_reg.predict(X_test)
print(f'Test score {np.sqrt(mean_squared_error(y_test, predicted))}')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(lyft_X, lyft_y, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[ ]:


train_pred = lin_reg.predict(X_train)
print(f'Train score {np.sqrt(mean_squared_error(y_train, train_pred))}')

predicted = lin_reg.predict(X_test)
print(f'Test score {np.sqrt(mean_squared_error(y_test, predicted))}')


# ### Hypothesis Testing

# **Price and Source**

# In[ ]:


import statsmodels.api         as     sm
from   statsmodels.formula.api import ols
 
mod = ols('price ~ source', data = uber_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('price ~ source', data = lyft_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# **Price and Destination**

# In[ ]:


mod = ols('price ~ destination', data = uber_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('price ~ destination', data = lyft_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# **Price and Name**

# In[ ]:


mod = ols('price ~ name', data = uber_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:


mod = ols('price ~ name', data = lyft_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# **Price and Cab type**

# In[ ]:


mod = ols('price ~ cab_type', data = cab_data).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)


# In[ ]:





# ## Feature Selection

# ### 1. Correlation

# In[ ]:


lyft_data.info()


# **Lyft**

# In[ ]:


plt.figure(figsize=(15,10))
corr_m = lyft_data.corr()
x = np.tri(corr_m.shape[0],k=-1)
sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds, mask=x)
plt.show()


# In[ ]:


corr_m['price'].abs().sort_values(ascending=False)[1:]


# From the correlation, we can see distance is moderately correlated,           
# followed by pressure, hour and wind. (we can ignore surge multiplier, as it leaks info about price.)          
# Rain and temperature are not significant.

# **Uber**

# In[ ]:


plt.figure(figsize=(15,10))
corr_m = uber_data[['distance', 'destination', 'source', 'price','name', 'hour', 'temp', 'clouds', 'pressure', 'rain', 'humidity',
       'wind']].corr()
x = np.tri(corr_m.shape[0],k=-1)
sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds, mask=x)
plt.show()


# In[ ]:


corr_m['price'].abs().sort_values(ascending=False)[1:]


# From the correlation, we can see distance is moderately correlated,           
# followed by pressure, hour and wind.          
# Rain and clouds are not significant.

# In[ ]:


uber1_X = uber_X.copy()
uber1_y = uber_y.copy()


# In[ ]:


lyft1_X = lyft_X
lyft1_y = lyft_y


# ### 2. Backward Elimination

# **Uber**

# In[ ]:


#Backward Elimination
cols = list(uber1_X.columns)
pmax = 1
counter=0
while (len(cols)>0):
    p= []
    counter+=1

    X_1 = uber1_X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(uber1_y,X_1).fit()
#     print(counter)
#     print(len(pd.Series(model.pvalues.values)))
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
#         print('inside')
        cols.remove(feature_with_p_max)
    else:
        break
    print(feature_with_p_max)
#     print(len(cols))
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


len(selected_features_BE)


# In[ ]:


uber2 = uber1_X[selected_features_BE]


# In[ ]:


uber2_X = uber2
uber2_y = uber_data['price'].copy()


# In[ ]:


x_constant = sm.add_constant(uber2_X)
uber_model = sm.OLS(uber2_y, x_constant).fit()
uber_model.summary()


# In[ ]:





# **Lyft**

# In[ ]:


#Backward Elimination
cols = list(lyft1_X.columns)
pmax = 1
counter=0
while (len(cols)>0):
    p= []
    counter+=1
    X_1 = lyft1_X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(lyft1_y,X_1).fit()
#     print(counter)
#     print(len(pd.Series(model.pvalues.values)))
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
#         print('inside')
        cols.remove(feature_with_p_max)
    else:
        break
    print(feature_with_p_max)
#     print(len(cols))
selected_features_BE = cols
print('\n',selected_features_BE)


# In[ ]:


len(selected_features_BE)


# In[ ]:


lyft2 = lyft1_X[selected_features_BE]


# In[ ]:


lyft2_X = lyft2
lyft2_y = lyft_data['price'].copy()


# In[ ]:


x_constant = sm.add_constant(lyft2_X)
lyft_model = sm.OLS(lyft2_y, x_constant).fit()
lyft_model.summary()


# In[ ]:





# In[ ]:





# ### 3. Step Forward Selection

# **Uber**

# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[ ]:


# Build RF classifier to use in feature selection
clf = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(uber1_X, uber1_y, test_size = 0.3, random_state = 0)


# Build step forward feature selection
sfs1 = sfs(clf,k_features = 38,forward=True,
           floating=False, scoring='r2',
           verbose=2,
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)


# In[ ]:





# **Lyft**

# In[ ]:


# Build RF classifier to use in feature selection
clf = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(lyft1_X, lyft1_y, test_size = 0.3, random_state = 0)


# Build step forward feature selection
sfs1 = sfs(clf,k_features = 38,forward=True,
           floating=False, scoring='r2',
           verbose=2,
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)


# In[ ]:





# ### 4. Lasso

# In[ ]:


from sklearn.linear_model import LassoCV


# **Uber**

# In[ ]:


reg = LassoCV()
reg.fit(uber1_X, uber1_y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(uber1_X,uber1_y))
coef = pd.Series(reg.coef_, index = uber1_X.columns)
coef


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:





# **Lyft**

# In[ ]:


reg = LassoCV()
reg.fit(lyft1_X, lyft1_y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(lyft1_X, lyft1_y))
coef = pd.Series(reg.coef_, index = lyft1_X.columns)
coef


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:





# ### 5. Using VIF

# In[ ]:


## Building of simple OLS model.
X_constant = sm.add_constant(uber1_X)
model = sm.OLS(uber1_y, X_constant).fit()
predictions = model.predict(X_constant)
model.summary()


# In[ ]:


### calculating the vif values as multicollinearity exists

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(uber1_X.values, j) for j in range(1, uber1_X.shape[1])]
vif


# In[ ]:


# removing collinear variables
# function definition

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        print(output.columns)
    return(output)


# In[ ]:


## passing X to the function so that the multicollinearity gets removed.
train_out = calculate_vif(uber1_X)


# In[ ]:


## includes only the relevant features.
train_out.head()


# In[ ]:


len(train_out.columns)


# **Select Features**

# In[ ]:


uber_X = uber_X.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1) #onehot encoded
lyft_X = lyft_X.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1)


# In[ ]:


uber_leX = uber_leX.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1) # label encoded
lyft_leX = lyft_leX.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1)


# In[ ]:


uber_leX.head()


# In[ ]:


lyft_leX.head()


# ### Feature transformation

# #### Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

uber_std = pd.DataFrame(sc.fit_transform(uber_X[['distance', 'hour', 'pressure', 'rain']]), 
                        columns=['distance', 'hour', 'pressure', 'rain'])

lyft_std = pd.DataFrame(sc.fit_transform(lyft_X[['distance', 'hour', 'pressure', 'rain']]),
                        columns=['distance', 'hour', 'pressure', 'rain'])

uber_X = uber_X.drop(['distance', 'hour', 'pressure', 'rain'], axis=1)
lyft_X = lyft_X.drop(['distance', 'hour', 'pressure', 'rain'], axis=1)

uber_X = pd.concat([uber_std, uber_X], axis=1)
lyft_X = pd.concat([lyft_std, lyft_X], axis=1)


# ### Model Building

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV


# ### 1. Linear Regression

# **Uber** (with weather conditions)

# In[ ]:


X_trainu, X_testu, y_trainu, y_testu = train_test_split(uber_X, uber_y, test_size=0.3, random_state=42)


# In[ ]:


lin_reg_uber = LinearRegression()
lin_reg_uber.fit(X_trainu, y_trainu)

# print(f'Train score : {lin_reg_uber.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, lin_reg_uber.predict(X_trainu)))}')
predicted = lin_reg_uber.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')


# In[ ]:


train_cv = cross_val_score(LinearRegression(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(LinearRegression(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

l_reg_uber = {}
l_reg_uber['Train'] = round(train_rmse, 4)
l_reg_uber['Test'] = round(test_rmse, 4)
l_reg_uber


# **Lyft** (with weather conditions)

# In[ ]:


X_trainl, X_testl, y_trainl, y_testl = train_test_split(lyft_X, lyft_y, test_size=0.3, random_state=42)


# In[ ]:


lin_reg_lyft = LinearRegression()
lin_reg_lyft.fit(X_trainl, y_trainl)

# print(f'Train score : {lin_reg_lyft.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, lin_reg_lyft.predict(X_trainl)))}')
predicted = lin_reg_lyft.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')


# In[ ]:


train_cv = cross_val_score(LinearRegression(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(LinearRegression(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

l_reg_lyft = {}
l_reg_lyft['Train'] = round(train_rmse, 4)
l_reg_lyft['Test'] = round(test_rmse, 4)
l_reg_lyft


# In[ ]:





# ### 2. Ridge

# **Uber** (with weather conditions)

# In[ ]:


ridge_reg = Ridge(random_state=42)
ridge_reg.fit(X_trainu, y_trainu)

ridge_reg_predict = ridge_reg.predict(X_testu)

# print(f'Train score : {ridge_reg.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, ridge_reg.predict(X_trainu)))}')
predicted = ridge_reg.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

np.sqrt(np.abs(cross_val_score(Ridge(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')))


# In[ ]:


train_cv = cross_val_score(Ridge(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(Ridge(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

r_reg_uber = {}
r_reg_uber['Train'] = round(train_rmse, 4)
r_reg_uber['Test'] = round(test_rmse, 4)
r_reg_uber


# **Hyperparameter Tuning**

# In[ ]:


lambdas=np.linspace(1,100,100)
params={'alpha':lambdas}
grid_search=GridSearchCV(Ridge(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')
grid_search.fit(X_trainu,y_trainu)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')
predicted = model.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)


# **Lyft** (with weather conditions)

# In[ ]:


ridge_reg = Ridge(random_state=42)
ridge_reg.fit(X_trainl, y_trainl)

ridge_reg_predict = ridge_reg.predict(X_testl)

# print(f'Train score : {ridge_reg.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, ridge_reg.predict(X_trainl)))}')
predicted = ridge_reg.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

# cross_val_score(Ridge(), X_trainl, y_trainl, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(Ridge(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(Ridge(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

r_reg_lyft = {}
r_reg_lyft['Train'] = round(train_rmse, 4)
r_reg_lyft['Test'] = round(test_rmse, 4)
r_reg_lyft


# **Hyperparameter Tuning**

# In[ ]:


lambdas=np.linspace(1,100,100)
params={'alpha':lambdas}
grid_search=GridSearchCV(Ridge(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')
grid_search.fit(X_trainl,y_trainl)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')
predicted = model.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

# cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1)


# ### 3. Lasso

# **Uber** (with weather conditions)

# In[ ]:


lasso_reg = Lasso(random_state=42)
lasso_reg.fit(X_trainu, y_trainu)

lasso_reg_predict = lasso_reg.predict(X_testu)

# print(f'Train score : {lasso_reg.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, lasso_reg.predict(X_trainu)))}')
predicted = lasso_reg.predict(X_testu)
# print(np.sqrt(mean_squared_error(y_trainu, lasso_reg.predict(X_trainu))))
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(Lasso(), X_trainu, y_trainu, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(Lasso(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(Lasso(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

la_reg_uber = {}
la_reg_uber['Train'] = round(train_rmse, 4)
la_reg_uber['Test'] = round(test_rmse, 4)
la_reg_uber


# **Hyperparameter Tuning**

# In[ ]:


lambdas=np.linspace(1,100,100)
params={'alpha':lambdas}
grid_search=GridSearchCV(Lasso(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')
grid_search.fit(X_trainu,y_trainu)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')
predicted = model.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)


# **Lyft** (with weather conditions)

# In[ ]:


lasso_reg = Lasso(random_state=42)
lasso_reg.fit(X_trainl, y_trainl)

# print(f'Train score : {lasso_reg.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, lasso_reg.predict(X_trainl)))}')
predicted = lasso_reg.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

# cross_val_score(Lasso(random_state=42), X_trainl, y_trainl, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(Lasso(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(Lasso(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

la_reg_lyft = {}
la_reg_lyft['Train'] = round(train_rmse, 4)
la_reg_lyft['Test'] = round(test_rmse, 4)
la_reg_lyft


# **HyperParameter Tuning**

# In[ ]:


lambdas=np.linspace(1,100,100)
params={'alpha':lambdas}
grid_search=GridSearchCV(Lasso(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')
grid_search.fit(X_trainl,y_trainl)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')
predicted = model.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

np.sqrt(np.abs(cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_absolute_error')))


# In[ ]:





# ### 4. Elastic Net

# **Uber** (with weather conditions)

# In[ ]:


elastic_reg = ElasticNet(random_state=42)
elastic_reg.fit(X_trainu, y_trainu)

elastic_reg_predict = elastic_reg.predict(X_testu)

# print(f'Train score : {elastic_reg.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, elastic_reg.predict(X_trainu)))}')
predicted = elastic_reg.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(ElasticNet(), X_trainu, y_trainu, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(ElasticNet(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(ElasticNet(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

el_reg_uber = {}
el_reg_uber['Train'] = round(train_rmse, 4)
el_reg_uber['Test'] = round(test_rmse, 4)
el_reg_uber


# **Hyperparameter Tuning**

# In[ ]:


# parametersGrid = {"alpha": [ 0.001, 0.01, 0.1, 1, 10, 100],
#                   "l1_ratio": np.arange(0.2, 1.0, 0.1)}
params={'alpha':lambdas}

grid_search=GridSearchCV(ElasticNet(),param_grid=params,cv=10,scoring='r2')
grid_search.fit(X_trainu,y_trainu)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainu, y_trainu)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')
predicted = model.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)


# **Lyft** (with weather conditions)

# In[ ]:


elastic_reg = ElasticNet(random_state=42)
elastic_reg.fit(X_trainl, y_trainl)

elastic_reg_predict = elastic_reg.predict(X_testl)

# print(f'Train score : {elastic_reg.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, elastic_reg.predict(X_trainl)))}')
predicted = elastic_reg.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

cross_val_score(ElasticNet(), X_trainl, y_trainl, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(ElasticNet(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(ElasticNet(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

el_reg_lyft = {}
el_reg_lyft['Train'] = round(train_rmse, 4)
el_reg_lyft['Test'] = round(test_rmse, 4)
el_reg_lyft


# **HyperParameterTuning**

# In[ ]:


# parametersGrid = {"alpha": [ 0.001, 0.01, 0.1, 1, 10, 100],
#                   "l1_ratio": np.arange(0.2, 1.0, 0.1)}
params={'alpha':lambdas}

grid_search=GridSearchCV(ElasticNet(),param_grid=params,cv=10,scoring='r2')
grid_search.fit(X_trainl,y_trainl)
grid_search.best_estimator_


# In[ ]:


model = grid_search.best_estimator_

# print(f'Train score : {model.score(X_trainl, y_trainl)}')
print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')
predicted = model.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

# cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1)


# In[ ]:





# ### 5. KNN

# **KNN taking long time, so dropping the model.**

# ### 5. Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_curve


# **Uber**

# In[ ]:


X_trainu, X_testu, y_trainu, y_testu = train_test_split(uber_leX, uber_ley, test_size=0.3, random_state=42)


# In[ ]:


dtree = DecisionTreeRegressor()

dtree.fit(X_trainu, y_trainu)

train_pred = dtree.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = dtree.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1)


# In[ ]:


max_depth = range(1,20)
train_results = []
test_results = []
for n in max_depth:
    dt = DecisionTreeRegressor(max_depth=n)
    dt.fit(X_trainu, y_trainu)
    train_pred = dt.predict(X_trainu)
    rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
    train_results.append(rmse)
    y_pred = dt.predict(X_testu)
    ts_rmse = np.sqrt(mean_squared_error(y_testu, y_pred))
    test_results.append(ts_rmse)


# In[ ]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depth, train_results, 'b', label='Train RMSE')
line2, = plt.plot(max_depth, test_results, 'r--', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE score')
plt.xlabel('Tree depth')
plt.show()


# Let's choose 15 as max depth

# In[ ]:


dtree = DecisionTreeRegressor(max_depth=15)

dtree.fit(X_trainu, y_trainu)

train_pred = dtree.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = dtree.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

# cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(DecisionTreeRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

dt_reg_uber = {}
dt_reg_uber['Train'] = round(train_rmse, 4)
dt_reg_uber['Test'] = round(test_rmse, 4)
dt_reg_uber


# **Lyft**

# In[ ]:


X_trainl, X_testl, y_trainl, y_testl = train_test_split(lyft_leX, lyft_ley, test_size=0.3, random_state=42)


# In[ ]:


dtree = DecisionTreeRegressor(max_depth=15)

dtree.fit(X_trainl, y_trainl)

train_pred = dtree.predict(X_trainl)

tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))
print(f'Train score : {tr_rmse}')
predicted = dtree.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

# cross_val_score(DecisionTreeRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1)


# In[ ]:


train_cv = cross_val_score(DecisionTreeRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(DecisionTreeRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

dt_reg_lyft = {}
dt_reg_lyft['Train'] = round(train_rmse, 4)
dt_reg_lyft['Test'] = round(test_rmse, 4)
dt_reg_lyft


# **HyperParameter tuning**

# In[ ]:


param_grid = {'max_depth': np.arange(3, 30),
             'min_samples_split': np.arange(.1,1.1,.1),
             'min_samples_leaf': np.arange(.1,.6,.1)}


# In[ ]:


grid_srch_dtree = tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10,scoring='neg_mean_squared_error')
grid_srch_dtree.fit(X_trainu, y_trainu)
grid_srch_dtree.best_estimator_


# In[ ]:





# ### 6. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
# from sklearn.cross


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_trainu, y_trainu)

train_pred = rf.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = rf.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(RandomForestRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(RandomForestRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(RandomForestRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

rf_reg_uber = {}
rf_reg_uber['Train'] = round(train_rmse, 4)
rf_reg_uber['Test'] = round(test_rmse, 4)
rf_reg_uber


# **Hyper Parameter Tuning**

# In[ ]:


param_grid = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
              'max_features' : list(range(1,X_trainu.shape[1])),
              'max_depth': np.arange(3, 30),
             'min_samples_split': np.arange(.1,1.1,.1),
             'min_samples_leaf': np.arange(.1,.6,.1)}


# In[ ]:


grid_srch_rf = tree = GridSearchCV(RandomForestRegressor(), param_grid, cv=10,scoring='neg_mean_squared_error')
grid_srch_rf.fit(X_trainu, y_trainu)
grid_srch_rf.best_estimator_


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_trainl, y_trainl)

train_pred = rf.predict(X_trainl)

tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))
print(f'Train score : {tr_rmse}')
predicted = rf.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(RandomForestRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(RandomForestRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(RandomForestRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

rf_reg_lyft = {}
rf_reg_lyft['Train'] = round(train_rmse, 4)
rf_reg_lyft['Test'] = round(test_rmse, 4)
rf_reg_lyft


# ### 7. Boosting

# **Ada Boost**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


abr = AdaBoostRegressor(random_state=42)

abr.fit(X_trainu, y_trainu)

train_pred = abr.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = abr.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(AdaBoostRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(AdaBoostRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(AdaBoostRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

abr_reg_uber = {}
abr_reg_uber['Train'] = round(train_rmse, 4)
abr_reg_uber['Test'] = round(test_rmse, 4)
abr_reg_uber


# In[ ]:


abr = AdaBoostRegressor(random_state=42)

abr.fit(X_trainl, y_trainl)

train_pred = abr.predict(X_trainl)

tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))
print(f'Train score : {tr_rmse}')
predicted = abr.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(AdaBoostRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(AdaBoostRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(AdaBoostRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

abr_reg_lyft = {}
abr_reg_lyft['Train'] = round(train_rmse, 4)
abr_reg_lyft['Test'] = round(test_rmse, 4)
abr_reg_lyft


# In[ ]:





# **Gradient Boosting**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# **Uber**

# In[ ]:


X_trainu.head()


# In[ ]:


gbr = GradientBoostingRegressor(random_state=42)

gbr.fit(X_trainu, y_trainu)

train_pred = gbr.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = gbr.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(GradientBoostingRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(GradientBoostingRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(GradientBoostingRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

gbr_reg_uber = {}
gbr_reg_uber['Train'] = round(train_rmse, 4)
gbr_reg_uber['Test'] = round(test_rmse, 4)
gbr_reg_uber


# **Lyft**

# In[ ]:


gbr = GradientBoostingRegressor(random_state=42)

gbr.fit(X_trainl, y_trainl)

train_pred = gbr.predict(X_trainl)

tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))
print(f'Train score : {tr_rmse}')
predicted = gbr.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(GradientBoostingRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(GradientBoostingRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(GradientBoostingRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

gbr_reg_lyft = {}
gbr_reg_lyft['Train'] = round(train_rmse, 4)
gbr_reg_lyft['Test'] = round(test_rmse, 4)
gbr_reg_lyft


# **Xg Boosting**

# In[ ]:


from xgboost import XGBRegressor


# **Uber**

# In[ ]:


xbr = XGBRegressor(random_state=42)

xbr.fit(X_trainu, y_trainu)

train_pred = xbr.predict(X_trainu)

tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))
print(f'Train score : {tr_rmse}')
predicted = xbr.predict(X_testu)
rmse = np.sqrt(mean_squared_error(y_testu, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(XGBRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(XGBRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(XGBRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

xbr_reg_uber = {}
xbr_reg_uber['Train'] = round(train_rmse, 4)
xbr_reg_uber['Test'] = round(test_rmse, 4)
xbr_reg_uber


# **Lyft**

# In[ ]:


xbr = XGBRegressor(random_state=42)

xbr.fit(X_trainl, y_trainl)

train_pred = xbr.predict(X_trainl)

tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))
print(f'Train score : {tr_rmse}')
predicted = xbr.predict(X_testl)
rmse = np.sqrt(mean_squared_error(y_testl, predicted))
print(f'Test score : {rmse}')

cv = cross_val_score(XGBRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
print(np.sqrt(np.abs(cv)))


# In[ ]:


train_cv = cross_val_score(XGBRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(XGBRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

xbr_reg_lyft = {}
xbr_reg_lyft['Train'] = round(train_rmse, 4)
xbr_reg_lyft['Test'] = round(test_rmse, 4)
xbr_reg_lyft


# In[ ]:





# **Cat Boost**

# Cat boost was designed to handle categorical values in the data automatically, it prevents overfitting, and has less prediction 
# time, because it builds symmetric trees.

# In[ ]:


from catboost import CatBoostRegressor


# **Uber**

# In[ ]:


X = uber_data.drop(['surge_multiplier', 'price', 'humidity', 'clouds', 'temp', 'wind'], axis=1)
y = uber_data['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features_indices = np.where(X.dtypes != np.number)[0]
categorical_features_indices


# In[ ]:


model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', verbose=400)
model.fit(X_train, y_train,cat_features=[1,2,3,4],eval_set=(X_test, y_test),plot=True)


# In[ ]:


train_cv = cross_val_score(CatBoostRegressor(), X_train, y_train, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(CatBoostRegressor(), X_test, y_test, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

cbr_reg_uber = {}
cbr_reg_uber['Train'] = round(train_rmse, 4)
cbr_reg_uber['Test'] = round(test_rmse, 4)
cbr_reg_uber


# **Lyft**

# In[ ]:


X = lyft_data.drop(['surge_multiplier', 'price', 'humidity', 'clouds', 'temp', 'wind'], axis=1)
y = lyft_data['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features_indices = np.where(X.dtypes != np.number)[0]
categorical_features_indices


# In[ ]:


model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=[1,2,3,4],eval_set=(X_test, y_test),plot=True)


# In[ ]:


train_cv = cross_val_score(CatBoostRegressor(), X_train, y_train, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
train_rmse = np.sqrt(np.abs(train_cv)).mean()

test_cv = cross_val_score(CatBoostRegressor(), X_test, y_test, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')
test_rmse = np.sqrt(np.abs(test_cv)).mean()

xbr_reg_lyft = {}
xbr_reg_lyft['Train'] = round(train_rmse, 4)
xbr_reg_lyft['Test'] = round(test_rmse, 4)
xbr_reg_lyft


# **Uber results**

# In[ ]:


final_results = pd.DataFrame([l_reg_uber, r_reg_uber, la_reg_uber, el_reg_uber, dt_reg_uber,
                              rf_reg_uber, abr_reg_uber, gbr_reg_uber, xbr_reg_uber, cbr_reg_uber],
                            index=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression',
                                  'Decision Tree', 'Random Forest', 'Ada Boost', 'Gradient Boost', 'Xg Boost',
                                  'Cat Boost'])
final_results


# **Lyft results**

# In[ ]:


final_results = pd.DataFrame([l_reg_lyft, r_reg_lyft, la_reg_lyft, el_reg_lyft, dt_reg_lyft,
                              rf_reg_lyft, abr_reg_lyft, gbr_reg_lyft, xbr_reg_lyft, cbr_reg_lyft],
                            index=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression',
                                  'Decision Tree', 'Random Forest', 'Ada Boost', 'Gradient Boost', 'Xg Boost',
                                  'Cat Boost'])
final_results


# In[ ]:




