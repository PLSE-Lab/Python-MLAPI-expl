#!/usr/bin/env python
# coding: utf-8

# # CraigsList Car and Truck Price Estimator
# 
# 

# # Getting Started
# 
# Dataset contains following features which will be used for analysis and prediction.
# 
# 1.	URL: Link to Listing
# 2.	City: Craigslist Region
# 3.	Price:  Price of the vehicle
# 4.	Year: Year of Manufacturing
# 5.	Manufacturer: Manufacturer of Vehicle
# 6.	Make: Model of Vehicle
# 7.	Condition: Vehicle condition
# 8.	Cylinder: Number of Cylinders
# 9.	Fuel: Type of Fuel required
# 10.	Odometer: Miles traveled
# 11.	Title_status: Title status (e.g. clean, missing, etc.)
# 12.	transmission: Type of transmission
# 13.	vin: Vehicle Identification Number
# 14.	drive: Drive of vehicle
# 15.	size: Size of vehicle
# 16.	type: Type of vehicle
# 17.	Paint_color: Color of vehicle
# 18.	image_url: Link to image
# 19.	lat: Latitude of listing
# 20.	Long: Longitude of listing
# 21.	county_fips : Federal Information Processing Standards code
# 22.	county_name: County of listing
# 23.	state_fips: Federal Information Processing Standards code
# 24.	state_code: 2 letter state code
# 25.	state_name: State name
# 26.	Weather: Historical average temperature for location in October/November
# 
# As mentioned before each listing can be identified by unique URL. We will go through following steps to get our regression model.
# 
# __1. Data import and analysis (Visualizations)__
# 
# __2. Data pre-processing and cleaning__
# 
# __3. Data transformation (feature engineering)__
# 
# __4. Modelling__
# 
# __5. Evaluation using matrices__
# 

# # Data load
# 
# Data is already made available through kaggle repository. This data is downloaded and used as it is directly.

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

cars = pd.read_csv("craigslistVehiclesFull.csv", header=0)

cars.head()


# # Data Analysis and Exploratory Visualization
# 
# We will be using pandas and matplotlib functionality to describe data.

# In[ ]:


cars.describe()


# In[ ]:


cars.shape


# ### Check Null counts in different columns
# 
# We have to determine action on various based on null values in columns. Since This check will be required periodically after formatting different columns lets create function and call it once

# In[ ]:


def checkNullableCounts(data):
    null_columns=data.columns[data.isnull().any()]
    print(data[null_columns].isnull().sum())
    
checkNullableCounts(cars)


# ### Price 
# 
# From the values obtained above for nullable column counts we can see that price is available on all columns and we are trying to predict price hence all the data points can be used for data analysis and refinement

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_carprice(cars):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    sns.boxplot(y='price', data=cars)
    plt.xlabel('Price Distribution boxplot')
    plt.subplot(122)
    sns.distplot(cars['price'], kde=False)
    plt.xlabel('Price Distribution')

plot_carprice(cars)


# ### Filter data on Price
# 
# It appears that there are lot of values in data which are outside regular values. Lets assume that minimum price for any car or truck that is available for resale is 100 and maximum 100K. This means that we would not be targetting any high end cars and truck from manufacturers like Ferrari, porsche or Lamborghini
# 
# From both the plots above we can confirm that there are some outliers in price column which should be removed.
# It can be assumed that we are only looking at cars and trucks which are not luxirious and hence the price should not exceed 100k. Hence dropping data for cars which have price tags above 100K.

# In[ ]:


cars_rawdata = cars.drop(cars[(cars['price'] > 100000) | (cars['price'] < 100)].index)
plot_carprice(cars_rawdata)


# # Data Features Cleanup
# 
# ## Manufacturer distribution
# Let's look at count of records for different manufacturers

# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.countplot(x='manufacturer',data=cars_rawdata,order=cars_rawdata['manufacturer'].value_counts().index);
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation= 90);


# ## City Statistics
# 
# Lets plot city counts distribution.

# In[ ]:


plt.figure(figsize=(20,10))
ax = sns.countplot(x='city',data=cars_rawdata,order=cars_rawdata['city'].value_counts().index);
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10);


# Based on graph represented above it looks like there is variable number of cars available in different cities and from nullability information it is confirmed that city information is available on all data points.
# 
# Since we are trying to predict price of cars available in different cities we will plot distribution of mean values of cars for various cities and rank cities lowest when mean is lowest.

# In[ ]:


##Return dataframe with rank related dataframe that has unique values of category, rank based on mean price for category in
##Ascending order i.e. lowest mean price category gets lowest rank (Little bias but based on real data)
def rankColumnInDataframe(cars, column):
    retDF = cars.groupby(column)['price'].mean().sort_values().to_frame()
    retDF[column]=retDF.index
    retDF=retDF.reset_index(drop=True)
    retDF['rank']=retDF.index+1
    return retDF

cities = rankColumnInDataframe(cars_rawdata, 'city')
cities.head()


# In[ ]:


plt.figure(figsize=(10,6))
g = sns.PairGrid(cities)
g = g.map(plt.scatter)


# Based on distribution plotted above we can notice that mean prices by cities are distributed evenly and hence to encode these cities we can use their rank.

# ## Location Information
# 
# City information is available on all the data points however we have more detailed feature on dataset which provides information about location from which listing was posted
# columns which provide this information are:
# - county_fips 
# - county_name 
# - state_fips  
# - state_code
# - latitude
# - longitude
# 
# Weather is also a location dependent feature however following columns have missing values or nulls (As obtained before)
# 
# |Column          | Null count |
# |----------------|------------|
# |county_fips     |  58833     |
# |county_name     |  58833     |
# |state_fips      |  58833     |
# |state_code      |  58833     |
# |weather         |  59428     |
# 
# Lets check weather information correlation with price using box and count plots
# 

# In[ ]:


def plotCountAndPriceForColumn(cars, column):
    f=plt.figure(figsize=(25,15))
    f.add_subplot(2,1,1)
    plt.xticks(fontsize=10)
    plt.title(column+' Histogram')
    sns.countplot(cars[column],palette=("magma"))
    f.add_subplot(2,1,2)
    plt.title(column +' vs Price')
    sns.boxplot(x=cars[column], y=cars['price'], palette=("magma"))


# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'weather')


# Based on count and price graphs for weather pointers we can conclude that mean effect of weather on car prices is very negligible and in approximately 2k-3k range. This observation also holds true for distinct values of weather where count of records is high i.e mean price at 59 degrees and 45 degrees is very close where count of cars is high. Hence weather information can be ignored. This can be considered if data is available on all data points and for future studies.
# 
# Lets try and see distribution of data points based on latitude and longitude
# 
# ### Latitude and Longitude
# 
# Distribution of Data over regions by sampling data for 10000 data points
# Plot using gmplot

# In[ ]:


cars_map = cars_rawdata.sample(n=10000, random_state=1)
latitude_list = cars_map['lat'].tolist()
longitude_list = cars_map['long'].tolist()

import gmplot
  
gmap3 = gmplot.GoogleMapPlotter(35, -100, 4, apikey=<your google maps key>) 
  
# scatter method of map object  
# scatter points on the google map 
gmap3.scatter( latitude_list, longitude_list, '#FF0000', size = 60, marker = False ) 

gmap3.draw("map.html")

from IPython.display import IFrame

display(IFrame(src='map.html', width=1000, height=600))


# From above location maps we can see that our dataset is distributed unevenly and area closer to east cost and west coast has more listings. This is done for only 10k customer and when plotted repeatedly it's noticed that some of the data points are over alask and in pacific ocean. Given these facts we can conclude that relying on latitude and longitude for accuracy of location is not advisable. Hence for location we will only use city location and drop other columns
# 
# ## Drop Insignifacant Columns
# 
# Following columns will be dropped from dataset:
# - Image_url: 
#     - Unique for each data point
#     - Unavailable for download
#     - Doesn't provide any significant information.
# - All location columns except City
#     - county_fips
#     - latitude
#     - longitude
#     - county_name 
#     - state_fips 
#     - state_code 
#     - weather    
# 
# 

# In[ ]:


cars_rawdata = cars_rawdata.drop(['image_url', 'county_fips', 'lat', 'long', 'county_name',
                                  'state_name','state_fips', 'state_code', 'weather'],axis=1)


# In[ ]:


print('Shape of cars raw dataset after dropping columns: {}'.format(cars_rawdata.shape))
print('Data types of columns in dataframe:')
cars_rawdata.dtypes


# # Categorical Columns
# 
# Following columns have missing values which needs to be analyzed in order to decide what to do with unknowns
# - drive
# - paint_color
# - type
# - size
# - transmission
# - title_status
# - fuel
# - condition
# 
# Lets replace missing values with common value 'NA' to analyze trend in data

# In[ ]:


cars_rawdata['drive_na'] = cars_rawdata.drive.fillna('NA')
cars_rawdata['paint_color_na'] = cars_rawdata.paint_color.fillna('NA')
cars_rawdata['type_na']= cars_rawdata['type'].fillna('NA')
cars_rawdata['size_na']= cars_rawdata['size'].fillna('NA')
cars_rawdata['transmission_na'] = cars_rawdata.transmission.fillna('NA')
cars_rawdata['title_status_na']=cars_rawdata.title_status.fillna('NA')
cars_rawdata['cylinders_na']=cars_rawdata.cylinders.fillna('NA')
cars_rawdata['fuel_na']=cars_rawdata.fuel.fillna('NA')
cars_rawdata['condition_na']=cars_rawdata.condition.fillna('NA')


# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'drive_na')


# Based on drive related box and countplot it can be seen that if drive information is not available on large number of data points however when value is not provided mean price falls between front wheel drive and rear wheel drive. 4 wheel drive cars are expensive. We are going to assign 'NA' for data points where this information is not available and encode  category similar to city where mean price if high for category, it'll have higher rank. However before we do this, glance at URL and make column data (in other iterations) shows that sometimes drive information is hidden in them. We want to extract this information for data points where this information is missing so we can accurately predict price of car with appropriate information
# 
# Lets check price boxplot and countplots for following columns so that we can identify if same function similat to drive and cities can be applied to these categorical columns
#     - paint_color
#     - type
#     - size
#     - transmission
#     - title
#     - cylinders
#     - fuel
#     - condition

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'paint_color_na')


# Based on pattern it's confirmed that we can apply similar encoding to paint_color as drive and city i.e. default will be 'NA' and ranking will be based on mean price for color. Colors would not be available in URL or make hence we don't need to refine this column more
# 
# Lets check count and price patterns for size

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'size_na')


# We can definitely encode this column similar to drive and cost but would not find this in URL or make hence skipping refinement of this column
# 
# Lets move on to type categorical column

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'type_na')


# This information can be encoded based on mean prices and information could be found in URL and make hence refinement will be required. Null values will be deafulted to 'NA' and encoding will be done based on mean price of the type
# Checking transmission columns patterns:

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'transmission_na')


# Above plots conclude that most of the vehicles are automatic but surprisingly data points where this information is not provided have higher mean prices hence we cannot encode them to 'automatic'. Information would not be available in URL or make hence skipping on data refinement however it can be encoded based on mean price (Might result in some bias where values are not provided)
# 
# Lets check patterns for title

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'title_status_na')


# Based on above plots we can conclude that most of the vehicles have clean status however vehicles with lien status are expensive hence encoding them based on mean price should help model in predictive better prices
# 
# Lets check pattern for fuel column

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'fuel_na')


# Most of the data points i.e. vehicles are gas based but it's interesting enough to see that mean price of diesel vehicles is higher than gas. We will use price based encoding and 'NA' as default value however we are not going to find this information in URL or make hence we will skip refinement of this column
# 
# Lets checke patterns for condition column

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'condition_na')


# We will use mean price based encoding given variance based on values and 'NA' as default value however we are not going to find this information in URL or make hence we will skip refinement of this column.
# 
# Lets check pattern for cylinder column

# In[ ]:


plotCountAndPriceForColumn(cars_rawdata, 'cylinders_na')


# It could be concluded that vehicles with higher cylinder numbers would have higher price and hence the mean price for respective category. We will encode this category similar to drive and city i.e. based on mean price. Default for this category will be 'NA'.
# 
# Based on all the above observations and discussions lets default following columns to NA and create dataframe to encode columns
# - paint_color
# - transmission
# - title_status
# - cylinder
# - size
# - fuel
# - condition

# In[ ]:


# Assign previously created NA valued columns to categorical columns
cars_rawdata['paint_color'] = cars_rawdata['paint_color_na']
cars_rawdata['size'] = cars_rawdata['size_na']
cars_rawdata['transmission'] = cars_rawdata['transmission_na']
cars_rawdata['title_status']= cars_rawdata['title_status_na']
cars_rawdata['cylinders']= cars_rawdata['cylinders_na']
cars_rawdata['fuel']= cars_rawdata['fuel_na']
cars_rawdata['condition']= cars_rawdata['condition_na']

# drop *_na named columns
cars_rawdata = cars_rawdata.drop(['paint_color_na', 'size_na', 'transmission_na', 'title_status_na', 'cylinders_na',
                                 'fuel_na','condition_na' ],axis=1)

#create dataframe to encode columns values for rank
paint_colors = rankColumnInDataframe(cars_rawdata, 'paint_color')
sizes = rankColumnInDataframe(cars_rawdata, 'size')
transmission_types = rankColumnInDataframe(cars_rawdata, 'transmission')
title_statuses = rankColumnInDataframe(cars_rawdata, 'title_status')
cylinders = rankColumnInDataframe(cars_rawdata, 'cylinders')
fuel_types = rankColumnInDataframe(cars_rawdata, 'fuel')
conditions = rankColumnInDataframe(cars_rawdata, 'condition')

# check sample of ranked dataframe
cylinders.head()


# Since we are going to search for drive and type related values in URL and make we should lists with distinct values of them and append/remove known values in list such as append awd for drive and remove 'NA' from both the lists

# In[ ]:


def returnUniqueValues(cars, column):
    return cars[column].unique()

drive_list = returnUniqueValues(cars_rawdata, 'drive_na').tolist()
type_list = returnUniqueValues(cars_rawdata, 'type_na').tolist()

drive_list.append('awd')
drive_list.append('4x4')
drive_list.remove('NA')
type_list.remove('NA')

print("Drive Values: ")
print(drive_list)
print("Type Values:")
print(type_list)
cars_rawdata = cars_rawdata.drop(['drive_na', 'type_na'],axis=1)


# ### Odometer
# 
# Odometer reading provides us insights into condition of the engine. This will be one of the most important feature required on each data point to estimate price. From previous nullability check we have seen that many data points have nulls on odometer information but based on google search and information from insurance companies it appears that average number of kms covered by vehicle is around 15-20K. We will create a function that will return odometer reading if unavailable on data based on age of the car

# In[ ]:


def odometerReading(year):
    return 15000* (2018-year)


# # URL
# 
# ## Format
# 
# URL of every datapoint contains description of the vehicle which is sometimes missing from make, year or model columns.
# Format of url contains :
# 1. City https://marshall.craigslist.org/cto/d//
# 2. Craigslist site and cto tag which stands for car and truck listing
# 3. Description of the listing (2010-dodge-challenger-se)
# 4. html page number (6717448841.html)
# 
# We are going to drop all other parts of url except description and split description using character '-'.

# In[ ]:


listing_url = [x.split('/')[-2].split('-') for x in cars_rawdata['url']]


# In[ ]:


listing_url[:10]


# We can see that every other url description has a year information followed by manufacturer information in some cases. We also noticed during initial analysis that there are following number of records has that nullable values
# - year               6315
# - manufacturer     136414
# 
# We can extract this missing information from URL description and as discussed before make can also contain this information. Lets check data in make column

# In[ ]:


# Remove duplicates from list
def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=False)))


# In[ ]:


import re

makelist = returnUniqueValues(cars_rawdata,'make')
print ("Unique values in make column :" + str(len(makelist)))
print("Sample of unique values for Make:")
makeList = [re.sub('[^A-Za-z0-9]+', '-', str(x)) for x in makelist]
##Clear '-' character if string starts with it
makeList = sorted([x[1:] if x.startswith('-') else x for x in makeList ])

# Split string into array elements - separator : '-'
makeList = [x.split('-') for x in makeList]

makeList = sort_and_deduplicate(makeList)
print(makeList[:100])


# ### Make
# 
# We can see from above makelist that manufacturer and year information could be available in make hence we will make sure to extract this information and refine make to use limited information. We might also find information about drive, type. Based on value lists created before we will try to extract as much information as possible
# 
# ### Manufacturer list
# 
# We have drive_list and type_list from previous functions, let's make manufacturer list using same function returnUniqueValues and remove null value

# In[ ]:


manufacturers_list = returnUniqueValues(cars_rawdata, 'manufacturer')
manufacturers_list = manufacturers_list[pd.notnull(np.array(manufacturers_list))]
manufacturers_list


# In[ ]:


def returnYear(x):
    try:
        if x.isdigit():
            if len(x)==4 and bool(re.match("((19[0-9][0-9])|(200[0-9])|(201[0-5]))",x)):
                return x
            if len(x)==2 and bool(re.match("([0-1][0-9])",x)):
                return x
    except:
        return None

def searchInColumns(column, columnValue):
    year= None
    manufacturer= None
    make = None
    drive = None
    retType = None
    returnValue = None
    try:
        if column=='url':
            returnValue = columnValue.split('/')[-2].split('-')
        elif column =='make':
            returnValue = re.sub('[^A-Za-z0-9]+', '-', str(columnValue))
            returnValue = returnValue[1:] if returnValue.startswith('-') else returnValue
            returnValue = returnValue.split('-')   
        make = list(returnValue)
        for x in returnValue:
            if returnYear(x):
                year = returnYear(x)
                make.remove(x)
            if x in manufacturers_list:
                manufacturer = x
                make.remove(x)
            if x in drive_list:
                drive = x
                make.remove(x)
            if x in type_list:
                retType = x
                make.remove(x) 
        make = ' '.join(make[:2])
        return year, manufacturer, make, drive, retType
    except Exception as e:
        print("Exception occured: {}".format(e))
        return year, manufacturer, make, drive, retType


# In[ ]:


searchInColumns('url', 'https://marshall.craigslist.org/cto/d/2010-dodge-challenger-se/6717448841.html')


# In[ ]:


searchInColumns('make', 'patriot high altitude')


# ### Year
# 
# Year is very important attribute on each data point to estimate price as it shows how old vehicle is. If year information is not available ater this cleanup if we are missing any values in this year column then we should drop data point.

# ### VIN
# 
# [VIN](https://driving-tests.org/vin-decoder/) (Vehicle identification number) is a unique code that is assigned to every motor vehicle when it's manufactured. The VIN is a 17-character string of letters and numbers without intrvening spaces or the letters Q, I and O which are avoided to avoid confusion with number 0 and 1. Each section of the VIN provides a specific piece of information about the vehicle including year, country, factory of manufacture; the make and model. VINs are usually printed in a single line and can be identified in several places on vehicle.
# 
# All the vehicle made after 1980 have 17 character long vin whereas the one manufactured before 1980 have 11 characters.
# 
# We will be extracting manufacturer and year information wherever vin is available to standardize information. After this activity since VIN is unique in each data point we will drop the information

# In[ ]:


from vininfo import Vin
##Get manufacturer and year from VIN
def getManufacturerModelYearFromVIN(VIN):
    try:
        manufacturer = None
        year = None
        model = None
        vin = Vin(VIN)
        if vin.manufacturer and vin.manufacturer != 'UnsupportedBrand':
            manufacturer = vin.manufacturer.lower()
        if vin.years:
            year= vin.years[0]
        return manufacturer, year
    except:
        return None, None


# In[ ]:


getManufacturerModelYearFromVIN('WDDNG79X97A124434')
#getManufacturerModelYearFromVIN('1FAFP55UO4G113464')


# ## Refine Dataset
# 
# We now have all the required functions to refine dataset. Lets apply following functions:
# - searchInColumns
# - odometerReading
# 
# While applying function we are going to make sure that year information if unavailable then will be extracted form url first instead of make. 
# 
# This will result in refinment of year, manufacturer, make, drive and type of vehicle

# In[ ]:


import math
def refineDataset(cars):
    returnDF = None
    try:
        returnDF = cars
        for i, row in returnDF.iterrows():
            year1, manufacturer1, make1, drive1, retType1 = searchInColumns('url', row['url'])
            year2, manufacturer2, make2, drive2, retType2 = searchInColumns('make', row['make'])
            ## if vin is avaible get this information
            if row['vin']:
                vin_manufacturer, vin_year = getManufacturerModelYearFromVIN(row['vin'])
            if math.isnan(row['year']):
                returnYear = None
                if year1:
                    returnYear = year1
                elif year2:
                    returnYear = year2
                elif vin_year:
                    returnYear = vin_year
                returnDF.at[i, 'year'] = returnYear
            if row['manufacturer'] is None:
                returnManufacturer = None
                if manufacturer1:
                    returnManufacturer = manufacturer1
                elif manufacturer2:
                    returnManufacturer = manufacturer2
                elif manufacturer_year:
                    returnManufacturer = vin_manufacturer
                returnDF.at[i, 'manufacturer'] = returnManufacturer
            # Assign refined value of make if available instead of one from url
            if row['make']:
                returnDF.at[i, 'make'] = make2 if make2 else make1
            if row['drive'] is None:
                returnDF.at[i, 'drive'] = drive1 if drive1 else drive2
            if row['type'] is None:
                returnDF.at[i, 'type'] = retType1 if retType1 else retType2
            if math.isnan(row['odometer']) or row['odometer'] <= float(100):
                returnDF.at[i, 'odometer'] = odometerReading(row['year'])
        return returnDF
    except:
        print("Exception occured: {}".format(e))
        return returnDF


# In[ ]:


cars_rawdata.head(10)


# In[ ]:


cars_refined = refineDataset(cars_rawdata)
cars_refined.head(10)


# ## Drop Vin and URL
# 
# These columns would not be used anymore for refining data and also doesn't have significance given they only have unique values, we will drop these 2 columns

# In[ ]:


cars_refined = cars_refined.drop(['vin','url' ],axis=1)


# In[ ]:


cars_refined.head()


# ## Manufacturer Standardization
# 
# Now as we have refined dataset, lets looks at Null count and effect of refinement on count of records for each manufacturer vs raw dataset.
# After looking at count plot we will standardize manufacturer as some of them are written in different way howeve mean the same for example: vw and volkswagen are same.
# We will perform this standardization exercise for drive list as well

# In[ ]:


checkNullableCounts(cars_refined)


# Based on nullability count we are going to take following action on columns
# 1. Year, odometer - Drop data points where information is missing
# 2. manufacturer, drive, type - replace Nulls with NA

# In[ ]:


cars_refined['drive'] = cars_refined.drive.fillna('NA')
cars_refined['manufacturer'] = cars_refined.manufacturer.fillna('NA')
cars_refined['type'] = cars_refined.type .fillna('NA')


# In[ ]:


cars_refined= cars_refined.dropna(axis=0, subset=['year','odometer'])
checkNullableCounts(cars_refined)


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.countplot(x='manufacturer',data=cars_refined,order=cars_refined['manufacturer'].value_counts().index);
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation= 90);


# There is definitely change in number of records in NA category and very little change in numbber of records at each category
# 
# ## Standardization function
# 
# It will be used against manufacturer and drive values to make sure only valida values are available in these columns

# In[ ]:


def standardizeUsingDict(columnValue, allowedValues):
    temp = columnValue
    try:
        if allowedValues[columnValue]:
            temp=allowedValues[columnValue]
        return temp
    except:
        return temp
    
def standardizeManufacturer(manufacturer):
    company={'aston':'aston-martin', 'chev':'chevrolet','harley':'harley-davidson', 'land rover' :'landrover', 
            'mercedes':'mercedes-benz', 'vw': 'volkswagen', 'alfa':'alfa-romeo', 'mercedesbenz':'mercedes-benz'}
    return standardizeUsingDict(manufacturer, company)
    

def standardizeDrive(drive):
    company={'awd':'4wd', '4x4':'4wd'}
    return standardizeUsingDict(drive, company)


# In[ ]:


cars_refined['manufacturer'] = cars_refined['manufacturer'].apply(standardizeManufacturer)


# In[ ]:


cars_refined['drive'] = cars_refined['drive'].apply(standardizeDrive)


# ## Encoding Columns
# 
# Previously we created dataframe with ranks for different categorical columns. Now we will do the same exercise for mean price for following columns and apply ranks on each column so we can start predicting prices
# - Columns Targeted
# 1. Manufacturer
# 2. Drive
# 3. Type
# 4. Make

# In[ ]:


manufacturers = rankColumnInDataframe(cars_refined, 'manufacturer')
drive_ranks = rankColumnInDataframe(cars_refined, 'drive')
type_ranks = rankColumnInDataframe(cars_refined, 'type')
make = rankColumnInDataframe(cars_refined, 'make')


# ## Encoding dataset by calculated ranks

# In[ ]:


def rankUsingDict(columnValue, allowedValues):
    temp = columnValue
    try:
        if allowedValues[columnValue]:
            temp=allowedValues[columnValue]
        return temp
    except:
        return temp

cities_dict = cities.set_index('city').T.to_dict('records')[1]

def rankCityFromDataframe(value):
    return rankUsingDict(value, cities_dict)


# In[ ]:


cars_refined['city'] = cars_refined['city'].apply(rankCityFromDataframe)


# In[ ]:


### Converting all ranking dataframes to dicts for faster processing 

manufacturers_dict = manufacturers.set_index('manufacturer').T.to_dict('records')[1]
make_dict = make.set_index('make').T.to_dict('records')[1]
drive_ranks_dict = drive_ranks.set_index('drive').T.to_dict('records')[1]
type_ranks_dict = type_ranks.set_index('type').T.to_dict('records')[1]
conditions_dict = conditions.set_index('condition').T.to_dict('records')[1]
fuel_types_dict = fuel_types.set_index('fuel').T.to_dict('records')[1]
cylinders_dict = cylinders.set_index('cylinders').T.to_dict('records')[1]
title_statuses_dict = title_statuses.set_index('title_status').T.to_dict('records')[1]
transmission_types_dict = transmission_types.set_index('transmission').T.to_dict('records')[1]
sizes_dict = sizes.set_index('size').T.to_dict('records')[1]
paint_colors_dict = paint_colors.set_index('paint_color').T.to_dict('records')[1]


# In[ ]:


def rankManufacturerFromDataframe(value):
    return rankUsingDict(value, manufacturers_dict)
def rankMakeFromDataframe(value):
    return rankUsingDict(value, make_dict)
def rankDriveFromDataframe(value):
    return rankUsingDict(value, drive_ranks_dict)
def rankTypeFromDataframe(value):
    return rankUsingDict(value, type_ranks_dict)
def rankConditionFromDataframe(value):
    return rankUsingDict(value, conditions_dict)
def rankFuelTypeFromDataframe(value):
    return rankUsingDict(value, fuel_types_dict)
def rankCylindersFromDataframe(value):
    return rankUsingDict(value, cylinders_dict)
def rankTitleStatusFromDataframe(value):
    return rankUsingDict(value, title_statuses_dict)
def rankTransmissionFromDataframe(value):
    return rankUsingDict(value, transmission_types_dict)
def rankSizeFromDataframe(value):
    return rankUsingDict(value, sizes_dict)
def rankPaintColorFromDataframe(value):
    return rankUsingDict(value, paint_colors_dict)


# In[ ]:


cars_refined['manufacturer'] = cars_refined['manufacturer'].apply(rankManufacturerFromDataframe)
cars_refined['make'] = cars_refined['make'].apply(rankMakeFromDataframe)
cars_refined['drive'] = cars_refined['drive'].apply(rankDriveFromDataframe)
cars_refined['type'] = cars_refined['type'].apply(rankTypeFromDataframe)
cars_refined['condition'] = cars_refined['condition'].apply(rankConditionFromDataframe)
cars_refined['fuel'] = cars_refined['fuel'].apply(rankFuelTypeFromDataframe)
cars_refined['cylinders'] = cars_refined['cylinders'].apply(rankCylindersFromDataframe)
cars_refined['title_status'] = cars_refined['title_status'].apply(rankTitleStatusFromDataframe)
cars_refined['transmission'] = cars_refined['transmission'].apply(rankTransmissionFromDataframe)
cars_refined['size'] = cars_refined['size'].apply(rankSizeFromDataframe)
cars_refined['paint_color'] = cars_refined['paint_color'].apply(rankPaintColorFromDataframe)


# Visualize converted data to get better insights into what has been changed in dataset

# In[ ]:


cars_prediction = cars_refined
cars_prediction.head(10)


# In[ ]:


cars_final = cars_prediction[['price','year','manufacturer','make','cylinders','city','title_status','transmission','paint_color','drive','size','type','condition','fuel', 'odometer']]


# # Save processed dataset.
# 
# We could avoid loading raw dataset and refining it by saving it for future use

# In[ ]:


cars_final.to_csv('Cars_processed_dataset.csv')


# In[ ]:


plt.figure(figsize = (30, 25))
sns.heatmap(cars_final.corr(), annot = True, cmap="YlGnBu")
plt.show()


# ## Select Features and columns to be predicted

# In[ ]:


price = cars_final['price']
features = cars_final.drop('price', axis=1)


# In[ ]:


# Import train_test_split
from sklearn.model_selection import train_test_split
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    price, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    #cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    cv_sets = ShuffleSplit(n_splits=10, test_size = 0.20, random_state = 0).split(X)

    # TODO: Create a decision tree regressor object
    #regressor = DecisionTreeRegressor()
    regressor = xgb.XGBRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    #params = {'max_depth':[10,50,100]}
    params = {'max_depth':[10,25,50],
              'objective':['reg:squarederror'],
              'colsample_bytree' : [0.3], 
              'learning_rate': [0.1],
              'alpha' : [10], 
              'n_estimators':[30, 50]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid


# In[ ]:


reg = fit_model(X_train, y_train)

best_model = reg.best_estimator_

# Produce the value for 'max_depth'
#print("Parameter 'max_depth' is {} for the optimal model.".format(best_model.get_params()['max_depth']))


# In[ ]:


print("Parameters for the optimal model : {}".format(best_model.get_params()))


# In[ ]:


import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 50, alpha = 10, n_estimators = 50)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[ ]:


print("R2 score for trained model based on results: " + str(performance_metric(y_test, preds)))


# # Conclusion
# 
# By using Xgboost we are able to reach benchmark i.e. 
# - RMSE: 5007.23 
# - R2: 0.7866
# 
# Decision trees provided:
# - RMSE : 9007
# - R2: 0.37
# 
# Which is very low compared to Xgboost

# In[ ]:




