#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> Road Safety Analysis </h1>

# In this notebook I will present the process a Data Scientist / Analyst should follow in order to extract useful information from a dataset. As an example I will use the given Acc.csv file for Accidents in the United Kingdom for 2017. The analysis is split in the mandatory steps for creating meaningful insights.

# <h3> Importing the data </h3>

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


AccidentsDF = pd.read_csv('../input/Acc.csv')


# ---

# <h3> Initial Exploration of the Dataset </h3>

# In[ ]:


# Full dataframe
AccidentsDF


# In[ ]:


# Number of rows
len(AccidentsDF)


# In[ ]:


# See Columns' names
AccidentsDF.columns


# In[ ]:


# Number of Columns
len(AccidentsDF.columns)


# In[ ]:


# Top rows
AccidentsDF.head()


# In[ ]:


# Bottom rows
AccidentsDF.tail()


# In[ ]:


# Find the null values from the dataset and get the column in which each null value exists
AccidentsDF_Null = AccidentsDF.isnull().sum()
AccidentsDF_Null[AccidentsDF_Null > 0].sort_values(ascending=False)


# In the following steps I will not use these columns, therefore there is no need to handle these Null values.

# In[ ]:


# Information on the columns
AccidentsDF.info()


# In[ ]:


# Get statistics on the columns
AccidentsDF.describe()


# In[ ]:


AccidentsDF.describe().transpose()


# Above the basic statistics of our Dataframe are presented  but most of them are meaningless as the specific attributes are recorded from Python in a fault data type. For example, the attribute Road_Type should be category and not integer as is obvious after the describe() function. 

# ---

# <h3> Renaming columns of the Dataframe </h3>

# In[ ]:


AccidentsDF.columns


# In[ ]:


AccidentsDF.columns = ['AccidentIndex', 'LocationEastingOSGR', 'LocationNorthingOSGR',
       'Longitude', 'Latitude', 'PoliceForce', 'AccidentSeverity',
       'NumberOfVehicles', 'NumberOfCasualties', 'Date', 'DayOfWeek',
       'Time', 'LocalAuthorityDistrict', 'LocalAuthorityHighway',
       '1stRoadClass', '1stRoadNumber', 'RoadType', 'SpeedLimit',
       'JunctionDetail', 'JunctionControl', '2ndRoadClass',
       '2ndRoadNumber', 'PedestrianCrossingHumanControl',
       'PedestrianCrossingPhysicalFacilities', 'LightConditions',
       'WeatherConditions', 'RoadSurfaceConditions',
       'SpecialConditionsAtSite', 'CarriagewayHazards',
       'UrbanOrRuralArea', 'PoliceOfficerAtScene',
       'AccidentLocationLSOA']


# In[ ]:


AccidentsDF.head()


# ---

# <h3> Subsetting the Dataframe </h3>

# I will select only the columns that I will use for the analysis.

# In[ ]:


AccDF = AccidentsDF[['AccidentIndex', 'PoliceForce', 'AccidentSeverity', 'NumberOfVehicles',
                   'Date', 'DayOfWeek', 'RoadType', 'SpeedLimit',
                   'JunctionDetail', 'LightConditions', 'WeatherConditions', 'RoadSurfaceConditions',
                   'SpecialConditionsAtSite', 'CarriagewayHazards', 'UrbanOrRuralArea']]


# In[ ]:


AccDF.columns


# In[ ]:


len(AccDF.columns)


# In[ ]:


AccDF.head()


# ---

# <h3> Changing the name of the needed elements </h3>

# In order to understand what the specific elements represent in each column of the dataset, I referred to the given metadata excel file and I changed these values.

# In[ ]:


AccDF['PoliceForce'] = AccDF['PoliceForce'].astype(str)

AccDF['PoliceForce'] = AccDF['PoliceForce'].replace({
        '1' : 'Metropolitan Police','3' : 'Cumbria','4' : 'Lancashire',
        '5' : 'Merseyside','6' : 'Greater Manchester','7' : 'Cheshire',
        '10' : 'Northumbria','11' : 'Durham','12' : 'North Yorkshire',
        '13' : 'West Yorkshire','14' : 'South Yorkshire','16' : 'Humberside',
        '17' : 'Cleveland','20' : 'West Midlands','21' : 'Staffordshire',
        '22' : 'West Mercia','23' : 'Warwickshire','30' : 'Derbyshire',
        '31' : 'Nottinghamshire','32' : 'Lincolnshire','33' : 'Leicestershire',
        '34' : 'Northamptonshire','35' : 'Cambridgeshire','36' : 'Norfolk',
        '37' : 'Suffolk','40' : 'Bedfordshire','41' : 'Hertfordshire',
        '42' : 'Essex','43' : 'ThamesValley','44' : 'Hampshire',
        '45' : 'Surrey','46' : 'Kent','47' : 'Sussex',
        '48' : 'City Of London','50' : 'Devon And Cornwall','52' : 'Avon And Somerset',
        '53' : 'Gloucestershire','54' : 'Wiltshire','55' : 'Dorset',
        '60' : 'North Wales','61' : 'Gwent','62' : 'South Wales','63' : 'Dyfed Powys',
        '91' : 'Northern','92' : 'Grampian','93' : 'Tayside','94' : 'Fife',
        '95' : 'Lothian And Borders','96' : 'Central','97' : 'Strathclyde',
        '98' : 'Dumfries And Galloway' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['AccidentSeverity'] = AccDF['AccidentSeverity'].astype(str)

AccDF['AccidentSeverity'] = AccDF['AccidentSeverity'].replace({'1' : 'Fatal',
                                                               '2' : 'Serious',
                                                               '3' : 'Slight'})


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['DayOfWeek'] = AccDF['DayOfWeek'].astype(str)

AccDF['DayOfWeek'] = AccDF['DayOfWeek'].replace({'1' : 'Sunday','2' : 'Monday', '3' : 'Tuesday',
                                                 '4' : 'Wednesday', '5' : 'Thursday', '6' : 'Friday',
                                                 '7' : 'Saturday'})                                                               


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['RoadType'] = AccDF['RoadType'].astype(str)

AccDF['RoadType'] = AccDF['RoadType'].replace({
        '1' : 'Roundabout','2' : 'One Way', '3' : 'Dual Carriageway',
        '6' : 'Single Carriageway', '7' : 'Slip Road', '9' : 'Unknown',
        '12' : 'One Way / Slip Road', '-1' : 'Data Missing'})


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['JunctionDetail'] = AccDF['JunctionDetail'].astype(str)

AccDF['JunctionDetail'] = AccDF['JunctionDetail'].replace({
        '0' : 'Not Junction Within 20 Meters',
        '1' : 'Roundabout','2' : 'Mini Roundabout', '3' :'T Junction',
        '5' : 'Slip Road', '6' : 'Croosroads', '7' : 'More than 4 Arms',
        '8' : 'Private Drive / Entrance', '9' : 'Other Junction', '-1' : 'Data Missing' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['LightConditions'] = AccDF['LightConditions'].astype(str)

AccDF['LightConditions'] = AccDF['LightConditions'].replace({
        '1' : 'Daylight','4' : 'Darkness Lights Lit', '5' : 'Darkness Lights Unlit',
        '6' : 'Darkness No Lighting', '7' : 'Darkness Lighting Unknown','-1' : 'Data Missing'})


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['WeatherConditions'] = AccDF['WeatherConditions'].astype(str)

AccDF['WeatherConditions'] = AccDF['WeatherConditions'].replace({
        '1' : 'Fine No Winds','2' : 'Raining No Winds', '3' : 'Snowing No Winds',
        '4' : 'Fine With Winds', '5' : 'Raining With Winds','6' : 'Snowing With Winds',
        '7' : 'Fog or Mist', '8' : 'Other', '9' : 'Unknown', '-1' : 'Data Missing' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['RoadSurfaceConditions'] = AccDF['RoadSurfaceConditions'].astype(str)

AccDF['RoadSurfaceConditions'] = AccDF['RoadSurfaceConditions'].replace({
        '1' : 'Dry','2' : 'Wet or Damp', '3' : 'Snow',
        '4' : 'Frost or Ice', '5' : 'Flood Over 3cm','6' : 'Oil or Diesel',
        '7' : 'Mud', '-1' : 'Data Missing' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['SpecialConditionsAtSite'] = AccDF['SpecialConditionsAtSite'].astype(str)

AccDF['SpecialConditionsAtSite'] = AccDF['SpecialConditionsAtSite'].replace({ '0' : 'None',
        '1' : 'Auto Traffic Signal Out','2' : 'Auto Traffic Signal Defective', '3' : 'Road Sign',
        '4' : 'Roadworks', '5' : 'Road Surface Defective','6' : 'Oil or Diesel',
        '7' : 'Mud', '-1' : 'Data Missing' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['CarriagewayHazards'] = AccDF['CarriagewayHazards'].astype(str)

AccDF['CarriagewayHazards'] = AccDF['CarriagewayHazards'].replace({ '0' : 'None',
        '1' : 'Vehicle Load On Road','2' : 'Other Object On Road', '3' : 'Previous Accident',
        '4' : 'Dog On Road', '5' : 'Other Animal On Road','6' : 'Pedestrian In Carriageway',
        '7' : 'Animal In Carriageway', '-1' : 'Data Missing' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF['UrbanOrRuralArea'] = AccDF['UrbanOrRuralArea'].astype(str)

AccDF['UrbanOrRuralArea'] = AccDF['UrbanOrRuralArea'].replace({ 
        '1' : 'Urban','2' : 'Rural', '3' : 'Unallocated' })


# In[ ]:


AccDF.head()


# In[ ]:


AccDF.info()


# I will convert some attributes from object to category.

# In[ ]:


AccDF.AccidentSeverity = AccDF.AccidentSeverity.astype('category')
AccDF.CarriagewayHazards = AccDF.CarriagewayHazards.astype('category')
AccDF.DayOfWeek = AccDF.DayOfWeek.astype('category')
AccDF.JunctionDetail = AccDF.JunctionDetail.astype('category')
AccDF.LightConditions = AccDF.LightConditions.astype('category')
AccDF.PoliceForce = AccDF.PoliceForce.astype('category')
AccDF.RoadSurfaceConditions = AccDF.RoadSurfaceConditions.astype('category')
AccDF.RoadType = AccDF.RoadType.astype('category')
AccDF.SpecialConditionsAtSite = AccDF.SpecialConditionsAtSite.astype('category')
AccDF.WeatherConditions = AccDF.WeatherConditions.astype('category')
AccDF.UrbanOrRuralArea = AccDF.UrbanOrRuralArea.astype('category')


# In[ ]:


AccDF.info()


# ---

# <h3> Basic Operations with the dataset </h3>

# In[ ]:


AccDF.describe().transpose()


# One of the important steps in order to have a general picture of the dataset is to extract a basic statistical information from the numeric attributes.
# The mean,standard deviation, min, max and the Quartiles are shown in the above table, for the two numeric attributes NumberOfVehicles and SpeedLimit.

# ---

# <h3> Filtering the Dataset </h3>

# Now, the dataset is cleaned and ready for the analysis. I will implement the analysis by answering some queries on the dataset in order to gain insight from the results.

# ****<h3> Find the percentage of all the accidents that are Fatal and occur on Saturday </h3>

# In[ ]:


FatAccSat = AccDF[(AccDF.DayOfWeek == 'Saturday') & (AccDF.AccidentSeverity == 'Fatal')]


# In[ ]:


FatNumAccSat = float(len(FatAccSat))
print(FatNumAccSat)


# In[ ]:


NumAcc = float(len(AccDF))
print(NumAcc)


# In[ ]:


Rate = (FatNumAccSat/NumAcc)*100
print(Rate)


# So, 0.216% of the Accidents that occur on Saturday are Fatal.

# ****<h3> Find the number of accidents that happened in Greater Manchester and occured when it was snowing </h3>

# In[ ]:


GrMan = AccDF[AccDF.PoliceForce == 'Greater Manchester']
float(len(GrMan)) #Number of accidents in Greater Manchester


# In[ ]:


SnowAcc = AccDF[(AccDF.WeatherConditions == 'Snowing No Winds') | 
                          (AccDF.WeatherConditions == 'Snowing With Winds')]
float(len(SnowAcc)) #Number of accidents when it was snowing


# In[ ]:


GrManSnowAcc = AccDF[(AccDF.PoliceForce == 'Greater Manchester') &
                     ((AccDF.WeatherConditions == 'Snowing No Winds') | 
                     (AccDF.WeatherConditions == 'Snowing With Winds'))]
float(len(GrManSnowAcc)) #Number of accidents in Greater Manchester when it was snowing


# So, 25 accidents happened in Greater Manchester when it was snowing.

# <h3> Find the number and the percentage of accidents that occured at urban area with speed greater than 30 </h3>

# In[ ]:


UrbAcc = AccDF[AccDF.UrbanOrRuralArea == 'Urban']
float(len(UrbAcc)) #Number of accidents in Urban area


# In[ ]:


HighSpeedAcc = AccDF[AccDF.SpeedLimit > 30]
float(len(HighSpeedAcc)) #Number of accidents with speed higher than 30


# In[ ]:


UrbHighSpeedAcc = AccDF[(AccDF.UrbanOrRuralArea == 'Urban') & (AccDF.SpeedLimit > 30)]
float(len(UrbHighSpeedAcc)) #Number of accidents in Urban area with speed higher than 30


# In[ ]:


Percentage = (float(len(UrbHighSpeedAcc))/float(len(UrbAcc)))*100
print(Percentage)


# So, 10% of the accidents that happened in urban area were due to the fact that the driver had been exceeding the speed limit of 30 miles per hour in these areas.

# ---

# <h3> Analysis using Visualizations </h3>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 8,4
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


AccDF.head()


# <h3> Distribution of the SpeedLimit attribute </h3>

# In[ ]:


plot1 = sns.distplot(AccDF.SpeedLimit, bins = 17)


# From this graph it is obvious that most of the accidents happened in the speed limit of 30 miles per hour.
# Also there is a significant number of accidents with speed greater than 60 miles per hour.

# Notice: The shape of distribution is as such because the SpeedLimit attribute should be categorical and not numeric, as shown from this plot. But, I handled it like numeric for the extraction of other statistical information from the dataset. 

# <h3> Boxplot of the AccidentSeverity and SpeedLimit of the accidents </h3>

# In[ ]:


plot2 = sns.boxplot(data = AccDF, x = 'AccidentSeverity', y = 'SpeedLimit')


# From this plot it is obvious that the Fatal accidents have big interquartile range and therefore an accident can be fatal at any speed.
# Moreover, the slight accidents occur in low speed with some outliers.

# <h3> Stacked Histograms for the severity of accidents and the speed limit </h3>

# In[ ]:


plt.hist(AccDF[AccDF.AccidentSeverity == 'Fatal'].SpeedLimit, label = 'Fatal')
plt.hist(AccDF[AccDF.AccidentSeverity == 'Slight'].SpeedLimit, label = 'Slight')
plt.hist(AccDF[AccDF.AccidentSeverity == 'Serious'].SpeedLimit, label = 'Serious')
plt.legend()
plt.show()


# We have the same results as the previous plot.

# <h3> Plot of the NumberOfVehicles and the SpeedLimit on specific DayOfWeek </h3>

# In[ ]:


plot3 = sns.lmplot(data = AccDF, x = 'SpeedLimit', y = 'NumberOfVehicles',                    fit_reg=False, hue = 'DayOfWeek')


# It is clear that the accidents with the most vehicles included happened in the speed limit of 40 and 50 miles per hour on Sundays, which makes sense as on that day most of the people return from weekend trips.

# <h3> Barplot of the day when accidents happened </h3>

# In[ ]:


plot4 = sns.countplot(data = AccDF, x = 'DayOfWeek', hue = 'AccidentSeverity')


# It is obvious that the most accidents occured on Fridays and were labeled as Slight.

# <h3> Barplot of the frequency of the Accident Severity of accidents </h3>

# In[ ]:


plot5 = sns.countplot(data = AccDF, y = 'AccidentSeverity', hue = 'RoadSurfaceConditions')


# So, most of the accidents are Slight and happened on Dry surface.
