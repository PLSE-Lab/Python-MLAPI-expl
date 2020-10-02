#!/usr/bin/env python
# coding: utf-8

# <font color='red' align='left'>Work in progress...</font>
# <br><font color='green' align='left'>please upvote if you find notebook useful</font>
# # PROBLEM STATEMENT
# 
# In this competition ask is to develop accurate models of metered building energy usage in the following areas:
# <br>chilled water, electric, hot water and stream meters.
# <br>
# <br>
# 
# 
# 
# <img align="left" src="https://github.com/rahul2712/ASHRAE/blob/master/evaluation_matrix.png?raw=true"></img>
# 
# 
# 
# 

# # SOLUTION APPROACH 1
# - <b>Step1:</b> Analyzing all features individually and along with target features <font color='green'><b>DONE</b></font>
# - <b>Step2:</b> Creating join between train and building metadata. Further join with weather train data. <font color='green'><b>DONE</b></font>
# - <b>Step3:</b> Handle all NULL values and convert all categorical features into numeric equivalents.<font color='green'><b>DONE</b></font>
# - <b>Step4:</b> Replacing NULL values in weather datasets (train/test) with average of train & test values for a given week.<font color='green'><b>DONE</b></font>
# - <b>Step5:</b> Create and save 52 (meter,primary_use combination) x 2 (iterations) = 104  models one for every meter type (0,1,2,3) & primary_use (Education,Office,etc.). Both train and test data have same number of combinations. <font color='green'><b>DONE</b></font>
# - <b>Step6:</b> No advanced feature engineering only basic feature engineering, e.g. extracting hour or day from <i>timestamp</i> feature or creating aggregates <font color='green'><b>DONE</b></font>
# - <b>Step7:</b> Preparing test data in same lines as training data, read saved models one by one, predict and submit results <font color='green'><b>DONE</b></font>
# - <font color='blue'>Result <b>Cross Validation score : 0.60438 and Public score: 1.42 </b>(still don't know reason for such a big difference). Requesting experts to comment and guide </font>
# 
# # FURTHER IMPROVEMENTS
# - Replacing null values under weather data with weekly means. Other alternate is replacing with interpolated values. We will first try weekly mean as we know interpolated values are giving better results.
# - Increasing training iterations from 6000 to 10000. As for number of models early stopping is not reached. Therefore, there is a scope of improvement.

# In[ ]:


#IMPORTING REQUIRED LIBRARIES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import datetime

from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import pickle

import gc
gc.enable()


import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # DATASET OVERVIEW

# In[ ]:


#DATASET VIEW
path1="/kaggle/input/ashrae-energy-prediction/"
path="/kaggle/input/ashrae-eda-and-104-models/"
data_files=list(os.listdir(path1))
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df_files.sort_values('File_Name'))


# ![ERD](https://github.com/rahul2712/ASHRAE/blob/master/ASHRAE.png?raw=true)

# # DATA READING STRATEGY
# 1. Will read only train, weather_train and building_metadata to start with.
# 2. As test set is big in size will be reading it after model training.
# 3. Not try to reduce datasize as per my understanding by reducing datasize sometimes we loose important information. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "#READING TRAIN DATASET\nprint('READING TRAIN DATASET...')\ndf_train=pd.read_csv(path1+'train.csv')\n\nprint('READING WEATHER TRAIN DATASET...')\ndf_weather_train=pd.read_csv(path1+'weather_train.csv')\n\nprint('READING WEATHER TEST DATASET...')\ndf_weather_test=pd.read_csv(path1+'weather_test.csv')\n\nprint('READING BUILDING METADATA...')\ndf_building_metadata=pd.read_csv(path1+'building_metadata.csv')\n\nprint('DATA READING COMPLETE')")


# In[ ]:


#All FUNCTIONS

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        elif 'datetime64[ns]' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(df_fa[col].max())+'/'+str(df_fa[col].min())
        df.at[col,'Sample_values']=list(df_fa[col].unique())
    display(df_fa.head())       
    return(df.fillna('-'))

#FUNCTION FOR READING DICTIONARY ITEMS AND HANDLING KEYERROR
def get_val(x,col):
    try:
        y=x[col]
    except:
        y=np.nan
    return(y)

#FUNCTION FOR CALCULATING RMSE
def rmse(y,pred):
    return(mean_squared_error(y,pred)**0.5)


# # FEATURE SUMMARY TRAIN SET

# In[ ]:


#CONVERTING timestamp TO DATATIME FIELD
df_train['timestamp']=pd.to_datetime(df_train['timestamp'])
#FEATURE SUMMARY FOR TRAIN DATASET
feature_summary(df_train)


# # UNDERSTANDING TRAIN FEATURES
# <table align=left >
#     <tr>
#         <th  bgcolor="cyan"><b>FEATURE NAME</b></th>
#         <th  bgcolor="cyan"><b>FEATURE DESCRIPTION</b></th>
#         <th  bgcolor="cyan"><b>ADDITIONAL INFORMATION</b></th> 
#     </tr>
#     <tr>
#         <td>building_id</td>
#         <td>a unique buildig identifier</td>
#         <td>Data contains 1449 unique builidings</td>
#     </tr>
#     <tr>
#         <td>meter</td>
#         <td>meter type</td>
#         <td>four different meter types. we can create models for each meter type</td>
#     </tr>
#     <tr>
#         <td>timestamp</td>
#         <td>timestamp for meter reading</td>
#         <td>we have readings for year 2016, starting from 2016-01-01 to 2016-12-31</td>
#     </tr>
#     <tr>
#         <td>meter_reading</td>
#         <td>target feature</td>
#         <td>target data is highly skewed</td>
#     </tr>
# </table>

# # UNDERSTANDING TARGET FEATURE <i>meter_reading</i>
# 1. Max value is 21904700.0
# 1. Min value is 0.0
# 1. Mean value is 2117.12
# 1. Only 8.01% observations have meter_reading greater or equal to 1000 units.
# 1. Only 1.15% observations have meter_reading greater or equal to 5000 units.
# 1. Only 0.47% observations have meter_reading greater or equal to 10,000 units.
# 
# 

# In[ ]:


#PLOT FOR METER READING BY DATES
plt.figure(figsize=(50,10))
plt.title("METER READING BY DATES",fontsize=40,color='b')
plt.xlabel("Dates",fontsize=40,color='b')
plt.ylabel("Meter Reading",fontsize=40,color='b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.plot(df_train['timestamp'],df_train['meter_reading'],color='green',linewidth=3)

plt.show()


# In[ ]:


#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE
pie_labels=['METER READING LESS THAN 1000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<1000.0].count()),
            'METER READING GREATER AND EQUAL TO 1000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=1000.0].count())            
           ]
pie_share=[df_train['meter_reading'][df_train.meter_reading<1000.0].count()/df_train['meter_reading'].count(),
           df_train['meter_reading'][df_train.meter_reading>=1000.0].count()/df_train['meter_reading'].count()
          ]
figureObject, axesObject = plt.subplots(figsize=(6,6))
pie_colors=('orange','grey')
pie_explode=(.15,.15)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)
axesObject.axis('equal')
plt.title('OBSERVATION % WITH METER READING LESS THAN 1000 UNITS AND OTHERWISE',color='blue',fontsize=12)
plt.show()


# In[ ]:


#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE
pie_labels=['METER READING LESS THAN 5000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<5000.0].count()),
            'METER READING GREATER AND EQUAL TO 5000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=5000.0].count())            
           ]
pie_share=[df_train['meter_reading'][df_train.meter_reading<5000.0].count()/df_train['meter_reading'].count(),
           df_train['meter_reading'][df_train.meter_reading>=5000.0].count()/df_train['meter_reading'].count()
          ]
figureObject, axesObject = plt.subplots(figsize=(6,6))
pie_colors=('orange','grey')
pie_explode=(.50,.25)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)
axesObject.axis('equal')
plt.title('OBSERVATION % WITH METER READING LESS THAN 5000 UNITS AND OTHERWISE',color='blue',fontsize=12)
plt.show()


# In[ ]:


#PIE CHART CHECKING DISTRIBUTION OF TARGET FEATURE
pie_labels=['METER READING LESS THAN 10,000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading<10000.0].count()),
            'METER READING GREATER AND EQUAL TO 10,000 UNITS : '+str(df_train['meter_reading'][df_train.meter_reading>=10000.0].count())            
           ]
pie_share=[df_train['meter_reading'][df_train.meter_reading<10000.0].count()/df_train['meter_reading'].count(),
           df_train['meter_reading'][df_train.meter_reading>=10000.0].count()/df_train['meter_reading'].count()
          ]
figureObject, axesObject = plt.subplots(figsize=(6,6))
pie_colors=('orange','grey')
pie_explode=(.50,.25)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)
axesObject.axis('equal')
plt.title('OBSERVATION % WITH METER READING LESS THAN 10,000 UNITS AND OTHERWISE',color='blue',fontsize=12)
plt.show()


# # DATA CATEGORIZATION BY <i>meter</i> TYPE
# 1. Meter type 0 is 59.66% of total training observations
# 1. Meter type 1 is 20.69% of total training observations
# 1. Meter type 0 is 13.40% of total training observations
# 1. Meter type 0 is 6.25% of total training observations
# 

# In[ ]:


#PIE CHART SHOWING DATA CATEGORIZATION BY METER TYPE
pie_labels=['METER TYPE 0 : '+str(df_train['meter'][df_train.meter==0].count()),
            'METER TYPE 1 : '+str(df_train['meter'][df_train.meter==1].count()),
            'METER TYPE 2 : '+str(df_train['meter'][df_train.meter==2].count()),
            'METER TYPE 3 : '+str(df_train['meter'][df_train.meter==3].count())
           ]
pie_share=[df_train['meter'][df_train.meter==0].count()/df_train['meter'].count(),
           df_train['meter'][df_train.meter==1].count()/df_train['meter'].count(),
           df_train['meter'][df_train.meter==2].count()/df_train['meter'].count(),
           df_train['meter'][df_train.meter==3].count()/df_train['meter'].count()
          ]
figureObject, axesObject = plt.subplots(figsize=(6,6))
pie_colors=('blue','orange','grey','green')
pie_explode=(.05,.05,.15,.05)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=45)
axesObject.axis('equal')
plt.title('TRAIN DATA CATEGORIZATION BY METER TYPE',color='blue',fontsize=12)
plt.show()


# In[ ]:


#FEATURE SUMMARY BY METER TYPE
print('FEATURE SUMMARY METER TYPE 0')
display(feature_summary(df_train[df_train.meter==0]))


# In[ ]:


#PLOT METER READING BY DATES FOR METER TYPE 0 
plt.figure(figsize=(50,10))
plt.title("METER READING BY DATES FOR METER TYPE 0",fontsize=40,color='b')
plt.xlabel("Dates",fontsize=40,color='b')
plt.ylabel("Meter Reading",fontsize=40,color='b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.plot(df_train['timestamp'][df_train.meter==0],df_train['meter_reading'][df_train.meter==0],color='blue',linewidth=3)

plt.show()


# In[ ]:


#FEATURE SUMMARY BY METER TYPE
print('FEATURE SUMMARY METER TYPE 1')
display(feature_summary(df_train[df_train.meter==1]))


# In[ ]:


#PLOT METER READING BY DATES FOR METER TYPE 1 
plt.figure(figsize=(50,10))
plt.title("METER READING BY DATES FOR METER TYPE 1",fontsize=40,color='b')
plt.xlabel("Dates",fontsize=40,color='b')
plt.ylabel("Meter Reading",fontsize=40,color='b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.plot(df_train['timestamp'][df_train.meter==1],df_train['meter_reading'][df_train.meter==1],color='orange',linewidth=3)

plt.show()


# In[ ]:


#FEATURE SUMMARY BY METER TYPE
print('FEATURE SUMMARY METER TYPE 2')
display(feature_summary(df_train[df_train.meter==2]))


# In[ ]:


#PLOT METER READING BY DATES FOR METER TYPE 2 
plt.figure(figsize=(50,10))
plt.title("METER READING BY DATES FOR METER TYPE 2",fontsize=40,color='b')
plt.xlabel("Dates",fontsize=40,color='b')
plt.ylabel("Meter Reading",fontsize=40,color='b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.plot(df_train['timestamp'][df_train.meter==2],df_train['meter_reading'][df_train.meter==2],color='grey',linewidth=3)

plt.show()


# In[ ]:


#FEATURE SUMMARY BY METER TYPE
print('FEATURE SUMMARY METER TYPE 3')
display(feature_summary(df_train[df_train.meter==3]))


# In[ ]:


#PLOT METER READING BY DATES FOR METER TYPE 3
plt.figure(figsize=(50,10))
plt.title("METER READING BY DATES FOR METER TYPE 3",fontsize=40,color='b')
plt.xlabel("Dates",fontsize=40,color='b')
plt.ylabel("Meter Reading",fontsize=40,color='b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.plot(df_train['timestamp'][df_train.meter==3],df_train['meter_reading'][df_train.meter==3],color='green',linewidth=3)

plt.show()


# # UNDERSTANDING <i>building_id</i>
# <font color='red'>Work in prgress...</font>

# # UNDERSTANDING <i>timestamp</i>
# <font color='red'>Work in prgress...</font>

# # FEATURE SUMMARY BUILDING METADATA SET

# In[ ]:


#FEATURE SUMMARY FOR BUILDING METADATA DATASET
feature_summary(df_building_metadata)


# # UNDERSTANDING BUILDING METADATA FEATURES
# <table align=left >
#     <tr>
#         <th  bgcolor="cyan"><b>FEATURE NAME</b></th>
#         <th  bgcolor="cyan"><b>FEATURE DESCRIPTION</b></th>
#         <th  bgcolor="cyan"><b>ADDITIONAL INFORMATION</b></th> 
#     </tr>
#     <tr>
#         <td>site_id</td>
#         <td>a unique site identifier</td>
#         <td>Data contains 16 unique sites</td>
#     </tr>
#     <tr>
#         <td>building_id</td>
#         <td>a unique buildig identifier</td>
#         <td>Data contains 1449 unique builidings</td>
#     </tr>
#     <tr>
#         <td>primary_use</td>
#         <td>primary use of the building</td>
#         <td>feature have string values, can be conveted into dummies</td>
#     </tr>
#     <tr>
#         <td>square_feet</td>
#         <td>building carpet area in square feets</td>
#         <td>on null values</td>
#     </tr>
#     <tr>
#         <td>year_built</td>
#         <td>year in which building was build</td>
#         <td>feature has 774 null values, can be filled with most occuring value</td>
#     </tr>
#     <tr>
#         <td>floor_count</td>
#         <td>number of floors in the building</td>
#         <td>1094 null values, can be filled with mean</td>
#     </tr>
# </table>

# # FEATURE SUMMARY WEATHER TRAIN SET

# In[ ]:


#CONVERTING timestamp TO DATATIME FIELD IN WEATHER TRAIN DATASET AND EXTRACTING OTHER TIME FEATURES
df_weather_train['timestamp']=pd.to_datetime(df_weather_train['timestamp'])
df_weather_train['month']=df_weather_train.timestamp.dt.month
df_weather_train['year']=df_weather_train.timestamp.dt.year
df_weather_train['day']=df_weather_train.timestamp.dt.day
df_weather_train['hour']=df_weather_train.timestamp.dt.hour
df_weather_train['week_day']=df_weather_train.timestamp.apply(lambda x:x.weekday())
df_weather_train['week']=df_weather_train.timestamp.apply(lambda x:x.isocalendar()[1])
#FEATURE SUMMARY FOR WEATHER TRAIN DATASET
feature_summary(df_weather_train)


# # UNDERSTANDING WEATHER TRAIN FEATURES
# <table align=left >
#     <tr>
#         <th  bgcolor="cyan"><b>FEATURE NAME</b></th>
#         <th  bgcolor="cyan"><b>FEATURE DESCRIPTION</b></th>
#         <th  bgcolor="cyan"><b>ADDITIONAL INFORMATION</b></th> 
#     </tr>
#     <tr>
#         <td>site_id</td>
#         <td>a unique site identifier</td>
#         <td>Data contains 16 unique sites</td>
#     </tr>
#     <tr>
#         <td>timestamp</td>
#         <td>timestamp for weather conditions</td>
#         <td>we have readings for year 2016, starting from 2016-01-01 to 2016-12-31</td>
#     </tr>
#     <tr>
#         <td>air_temperature</td>
#         <td>air temperature for given site and timestamp</td>
#         <td>have 55 null values, can be filled with mean or we can predict temperature using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>cloud_coverage</td>
#         <td>cloud coverage for given site and timestamp</td>
#         <td>have 69173 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>dew_temperature</td>
#         <td>dew_temperature for given site and timestamp</td>
#         <td>have 113 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>precip_depth_1_hr</td>
#         <td>precipitation for given site and timestamp</td>
#         <td>have 50289 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>sea_level_pressure</td>
#         <td>sea level pressure for given site and timestamp</td>
#         <td>have 10618 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>wind_direction</td>
#         <td>wind direction for given site and timestamp</td>
#         <td>have 6268 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
#     <tr>
#         <td>wind_speed</td>
#         <td>wind speed for given site and timestamp</td>
#         <td>have 304 null values, can be filled with mean or we can predict cloud coverage using other features in train weather table</td>
#     </tr>
# </table>

# # FEATURE SUMMARY WEATHER TEST SET

# In[ ]:


get_ipython().run_cell_magic('time', '', "#CONVERTING timestamp TO DATATIME FIELD IN WEATHER TEST DATASET AND EXTRACTING OTHER TIME FEATURES\ndf_weather_test['timestamp']=pd.to_datetime(df_weather_test['timestamp'])\ndf_weather_test['month']=df_weather_test.timestamp.dt.month\ndf_weather_test['year']=df_weather_test.timestamp.dt.year\ndf_weather_test['day']=df_weather_test.timestamp.dt.day\ndf_weather_test['hour']=df_weather_test.timestamp.dt.hour\ndf_weather_test['week_day']=df_weather_test.timestamp.apply(lambda x:x.weekday())\ndf_weather_test['week']=df_weather_test.timestamp.apply(lambda x:x.isocalendar()[1])\n#FEATURE SUMMARY FOR WEATHER TRAIN DATASET\nfeature_summary(df_weather_test)")


# # APPENDING WEATHER TRAIN AND TEST TO CALCULATE MEANS FOR MISSING VALUES

# In[ ]:


#HORIZONTALLY APPENDING WEATHER TRAIN AND TEST
df_weather=pd.concat([df_weather_train,df_weather_test],axis=0,ignore_index=True)
#FEATURE SUMMARY FOR COMBINED WEATER DATASET
feature_summary(df_weather)


# In[ ]:


#CALCULATING MEANS FOR SITE ID AND WEEK
df_calc_means=df_weather[['site_id','week','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
            'sea_level_pressure','wind_direction','wind_speed']].groupby(['site_id','week']).mean().reset_index()
cols=['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
            'sea_level_pressure','wind_direction','wind_speed']

#ROUNDING OFF MEANS TO 2 DECIMAL PLACES
df_calc_means[cols]=df_calc_means[cols].round(2)

#REMOVING NULL VALUES FROM CALCULATED MEANS DATAFRAME
df_calc_means['cloud_coverage'].replace(np.nan,round(df_calc_means['cloud_coverage'].mean(),2),inplace=True)
df_calc_means['precip_depth_1_hr'].replace(np.nan,round(df_calc_means['precip_depth_1_hr'].mean(),2),inplace=True)
df_calc_means['sea_level_pressure'].replace(np.nan,round(df_calc_means['sea_level_pressure'].mean(),2),inplace=True)

print('FEATURE SUMMARY FOR CALCULATED MEANS BY SITE ID AND WEEK')
feature_summary(df_calc_means)


# # JOINING TRAIN SETS
# 1. First Join Train set with Building Metadata on building_id to populate building related information. Lets call joined set as df_train_BM
# 2. Second Join between df_train_BM and df_weather_train on site_id and timestamp. Lets call joined set as df_train_BMW

# # JOINING BUILDING METADATA WITH TRAIN SET

# In[ ]:


#JOINING TRAIN SET AND BUILDING METADATA
df_train_BM=pd.merge(df_train,df_building_metadata,how='left',on='building_id')
feature_summary(df_train_BM)


# # UNDERSTANDING BUILDING <i>primary_use</i> FEATURE
# <i>primary_use</i> feature has 16 different categories

# In[ ]:


get_ipython().run_cell_magic('time', '', "#UNDERSTANDING BUILDING PRIMARY_USER FEATURE\npu_ls=list(df_train_BM['primary_use'].unique())\ndf_pu=pd.DataFrame(pu_ls,columns=['primary_use'])\ndf_pu['% Distribution']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['primary_use'][df_train_BM.primary_use==x].count()/\n                                                   df_train_BM['primary_use'].count(),4)*100)\ndf_pu['Number_of_observations']=df_pu['primary_use'].apply(lambda x:df_train_BM['primary_use'][df_train_BM.primary_use==x].count())\ndf_pu['Avg_consumption']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['meter_reading'][df_train_BM.primary_use==x].mean(),2))\ndf_pu['Avg_sq_feet']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['square_feet'][df_train_BM.primary_use==x].mean(),2))\ndf_pu['Consumption_per_sq_feet']=df_pu['primary_use'].apply(lambda x:round(df_train_BM['meter_reading'][df_train_BM.primary_use==x].sum()/\n                                                                          df_train_BM['square_feet'][df_train_BM.primary_use==x].sum(),4))\ndisplay(df_pu)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n#PIE CHART SHOWING DATA CATEGORIZATION BY METER TYPE\npie_labels=[]\npie_share=[]\n\nfor pu in pu_ls:\n    pie_labels.append(pu+' : '+str(df_train_BM['primary_use'][df_train_BM.primary_use==pu].count()))\n    pie_share.append(df_train_BM['primary_use'][df_train_BM.primary_use==pu].count()/df_train_BM['primary_use'].count())                  \n    \nfigureObject, axesObject = plt.subplots(figsize=(15,15))\n\npie_explode=(.1,.1,.1,.1,.89,.89,.89,.1,.99,.99,.99,.99,.99,.99,.99,.99)\naxesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',startangle=45)\naxesObject.axis('equal')\nplt.title('TRAIN DATA CATEGORIZATION BY PRIMARY USE OF BUILDING',color='blue',fontsize=12)\nplt.show()")


# # UNDERSTANDING RELATIONSHIP BETWEEN <i>primary_use</i> AND <i>meter</i> FEATURES

# In[ ]:


#UNDERSTANDING RELATIONSHIP BETWEEN primary_use AND meter FEATURES
df=df_train_BM[['building_id','meter','primary_use']].groupby(['meter','primary_use']).count().reset_index()
df.columns=['meter_type','primary_use','observation_count']
display(df)


# # REPLACING NULL VALUES FOR <i>year_built</i> AND <i>floor_count</i> FEATURES
# 1. Replacing NULL year_built by mode of year_built
# 1. Replacing NULL floor_count by mean of floor_count
# 1. There can be other ways replacing NULL values. We will be exploring those while improving our model
# 

# In[ ]:


#VALUES REPLACING NULL VALUES
print('Mean for floor_count is:',round(df_train_BM['floor_count'].mean(),0))
print('Mode for year_built is:',df_train_BM['year_built'].mode()[0])

#REPLACING NULL VALUES
df_train_BM['floor_count'].replace(np.nan,round(df_building_metadata['floor_count'].mean(),0),inplace=True)
df_train_BM['year_built'].replace(np.nan,df_building_metadata['year_built'].mode()[0],inplace=True)

#FEATURE SUMMARY AFTER REPLACING NULL VALUES FOR FEATURES floor_count AND year_built
print('Feature summary after replacing NULL values')
feature_summary(df_train_BM)


# # CONVERTING <i> primary_use</i> FEATURE TO DUMMIES

# In[ ]:


# #CREATING DUMMIES FOR pirmary_use FEATURE
# df_train_BMF=pd.concat([df_train_BM,pd.get_dummies(df_train_BM['primary_use'],prefix='pu')],axis=1)
# df_train_BMF.drop('primary_use',axis=1,inplace=True)

# #FEATURE SUMMARY POST DUMMY CREATION
# print('FEATURE SUMMARY AFTER CREATING DUMMIES')
# feature_summary(df_train_BMF)


# In[ ]:


del df_train
gc.collect()


# # JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TRAIN SET

# In[ ]:


get_ipython().run_cell_magic('time', '', "# JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TRAIN SET ON site_id AND timestamp\ncols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',\n            'sea_level_pressure','wind_direction','wind_speed']\ndf_train_BMW=pd.merge(df_train_BM,df_weather_train[cols],how='left',on=['site_id','timestamp'])\nfeature_summary(df_train_BMW)")


# In[ ]:


#CLEARING DATAFRAMES
del df_train_BM
gc.collect()


# # SOME BASIC FEATURE ENGINEERING
# - Separating month, day of the week, day, hour from timestamp

# In[ ]:


get_ipython().run_cell_magic('time', '', "#EXTRACTING INFORMATION FROM timestamp FEATURE\ndf_train_BMW['month']=df_train_BMW.timestamp.dt.month\ndf_train_BMW['day']=df_train_BMW.timestamp.dt.day\ndf_train_BMW['hour']=df_train_BMW.timestamp.dt.hour\ndf_train_BMW['week_day']=df_train_BMW.timestamp.apply(lambda x:x.weekday())\ndf_train_BMW['week']=df_train_BMW.timestamp.apply(lambda x:x.isocalendar()[1])\n\n#GARBAGE COLLECTION\ngc.collect()")


# In[ ]:


#FEATURE SUMMARY FOR NEW FEATURES
lfe=['month','day','hour','week_day','week']
print('FEATURE SUMMARY FOR GENERATING CALCULATED FEATURES')
feature_summary(df_train_BMW[lfe])


# # REPLACING NULL VALUES WITH CALCULATED MEANS
# 
# 1. Following features have NULL values <i>air_temperature, cloud_coverage, dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_direction,wind_speed </i>
# 1. NULL values will be replaced with the means calculated for <i>site_id</i> and <i>week</i> as climatic features are highly influenced by time and location
# 
# <font color='red' align='left'>Work in progress...</font>

# In[ ]:


get_ipython().run_cell_magic('time', '', "#REPLACING NULL VALUES IN WEATHER RELATED FIELDS\nfor i in range(0,df_calc_means.shape[0]):\n    print('replaceing null for site_id: ',df_calc_means.iloc[i,].site_id,' ; week ',df_calc_means.iloc[i,].week,' ; count ',i)\n    df_train_BMW[['site_id','week','air_temperature']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].air_temperature,inplace=True)\n    df_train_BMW[['site_id','week','cloud_coverage']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].cloud_coverage,inplace=True)\n    df_train_BMW[['site_id','week','dew_temperature']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].dew_temperature,inplace=True)\n    df_train_BMW[['site_id','week','precip_depth_1_hr']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].precip_depth_1_hr,inplace=True)\n    df_train_BMW[['site_id','week','sea_level_pressure']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].sea_level_pressure,inplace=True)\n    df_train_BMW[['site_id','week','wind_direction']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].wind_direction,inplace=True)\n    df_train_BMW[['site_id','week','wind_speed']][(df_train_BMW.site_id==df_calc_means.iloc[i,].site_id) & (df_train_BMW.week==df_calc_means.iloc[i,].week)].replace(np.nan,df_calc_means.iloc[i,].wind_speed,inplace=True)")


# In[ ]:


#FEATURE SUMMARY POST REPLACING WEATHER NULL VALUES
feature_summary(df_train_BMW)


# # BUILDING PREDICTOR

# In[ ]:


def ASHRAE_predict_lgb(X,y,i): 
    

    params = {'num_leaves': 31,
              'objective': 'regression',
              'learning_rate': 0.1,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": 0.1,
              "feature_fraction": 0.9,
              "metric": 'rmse',
              }

    k=1
    splits=2
    avg_score=0


    kf = KFold(n_splits=splits, shuffle=True, random_state=200)
    print('\nStarting KFold iterations...')
    for train_index,test_index in kf.split(X):

        
        df_X=X[train_index,:]
        df_y=y[train_index]
        val_X=X[test_index,:]
        val_y=y[test_index]

        
        dtrain = lgb.Dataset(df_X, label=df_y)
        dvalid = lgb.Dataset(val_X, label=val_y)
        model=lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid],
                        verbose_eval=2000, early_stopping_rounds=500)

        preds_x=pd.Series(model.predict(val_X))
        preds_x=[x if x>=0 else 0 for x in preds_x]
        acc=rmse(val_y,preds_x)
        print('Iteration:',k,'  rmsle:',acc)
        #SAVING MODEL
        Pkl_Filename = "Pickle_Model_"+str(k)+"_combi_"+str(i)+".pkl"  

        with open(Pkl_Filename, 'wb') as file:
            pickle.dump(model, file)
        print('MODEL SAVED...')
        if k==1:
            score=acc
            preds=pd.Series(preds_x)
            acct=pd.Series(val_y)
        
        else:
            preds=preds.append(pd.Series(preds_x))
            acct=acct.append(pd.Series(val_y))
            if score<acc:
                score=acc
                
        avg_score=avg_score+acc        
        k=k+1
    print('\n Best score:',score,' Avg Score:',avg_score/splits)
#     preds=preds/splits
    return(acct,preds)


# In[ ]:


get_ipython().run_cell_magic('time', '', "#TRAINING MODELS\nfor i in range(0,df.shape[0]):\n    #SPLITTING TRAINING DATA BY METER TYPE\n    print('TRAINING MODEL FOR METER TYPE: ',df.iloc[i,0],' AND PRIMARY USE:',df.iloc[i,1])  \n    X=df_train_BMW[(df_train_BMW.meter==df.iloc[i,0]) & (df_train_BMW.primary_use==df.iloc[i,1])].drop(['building_id','timestamp','meter','meter_reading','site_id','primary_use'],axis=1).values\n    y=np.log1p(df_train_BMW['meter_reading'][(df_train_BMW.meter==df.iloc[i,0]) & (df_train_BMW.primary_use==df.iloc[i,1])].values)\n    \n    #FITTING MODEL\n    val_y,preds_x=ASHRAE_predict_lgb(X,y,i)\n#     print(val_y.shape,preds_x.shape)\n    if i==0:\n        preds=pd.Series(preds_x)\n        acct=pd.Series(val_y)\n    else:\n        preds=preds.append(preds_x)\n        acct=acct.append(val_y)\n           \n    del X,y\n    gc.collect()\n    \n    \n# print(acct.shape,preds.shape)\nprint('OVER ALL ACCURACY:',rmse(acct,preds))    \n    \n    ")


# In[ ]:


del df_train_BMW
gc.collect()


# In[ ]:


41697600/20


# # PREDICTING METER READING

# In[ ]:


# #READING SAMPLE SUBMISSION FILE
# submission=pd.read_csv(path1+'sample_submission.csv')


# In[ ]:


# %%time
# #READING TEST DATA IN CHUNKS
# c_size=2084880
# k=1
# subf=pd.DataFrame()
# for df_test in pd.read_csv(path1+'test.csv',chunksize=c_size):
#     print(df_test.shape)
#     print('Predicting chunk:',k,' of 20')
    
#     df_test['timestamp']=pd.to_datetime(df_test['timestamp'])
    
#     #JOINING WITH BUILDING METADATA
#     df_test_BM=pd.merge(df_test,df_building_metadata,how='left',on='building_id')
    
#     #GARBAGE COLLECTION
#     del df_test
#     gc.collect()
    
#     #REPLACING NULL VALUES
#     df_test_BM['floor_count'].replace(np.nan,round(df_building_metadata['floor_count'].mean(),0),inplace=True)
#     df_test_BM['year_built'].replace(np.nan,df_building_metadata['year_built'].mode()[0],inplace=True)
    
    
#     # JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TEST SET ON site_id AND timestamp
#     cols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
#             'sea_level_pressure','wind_direction','wind_speed']
#     df_test_BMW=pd.merge(df_test_BM,df_weather_test[cols],how='left',on=['site_id','timestamp'])
    
#     #GARBAGE COLLECTION
#     del df_test_BM
#     gc.collect()
    
#     #EXTRACTING INFORMATION FROM timestamp FEATURE
#     df_test_BMW['month']=df_test_BMW.timestamp.dt.month
#     df_test_BMW['day']=df_test_BMW.timestamp.dt.day
#     df_test_BMW['hour']=df_test_BMW.timestamp.dt.hour
#     df_test_BMW['week_day']=df_test_BMW.timestamp.apply(lambda x:x.weekday())
#     df_test_BMW['week']=df_test_BMW.timestamp.apply(lambda x:x.isocalendar()[1])
    
#     print('Data Preparation Done')
#     for i in range(0,df.shape[0]):
#         sub=pd.DataFrame()
        
#         #SPLITTING TRAINING DATA BY METER TYPE
#         print('PREDICTING FOR METER TYPE: ',df.iloc[i,0],' AND PRIMARY USE:',df.iloc[i,1])  
#         X=df_test_BMW[(df_test_BMW.meter==df.iloc[i,0]) & (df_test_BMW.primary_use==df.iloc[i,1])].drop(['building_id','timestamp','meter','row_id','site_id','primary_use'],axis=1).values
#         sub['row_id']=df_test_BMW['row_id'][(df_test_BMW.meter==df.iloc[i,0]) & (df_test_BMW.primary_use==df.iloc[i,1])].values
        
        
        
#         gc.collect()
        
#         if X.shape[0]!=0:
#             Pkl_Filename1 = "Pickle_Model_"+str(1)+"_combi_"+str(i)+".pkl"  
#             Pkl_Filename2 = "Pickle_Model_"+str(2)+"_combi_"+str(i)+".pkl" 
        
        
#             with open(path+Pkl_Filename1, 'rb') as file:
#                 model1 = pickle.load(file)
        
#             with open(path+Pkl_Filename2, 'rb') as file:
#                 model2 = pickle.load(file)
        
#             sub['meter_reading1']=pd.Series(model1.predict(X))
# #             sub['meter_reading1']=[x if x>=0 else 0 for x in sub['meter_reading1']]
        
#             sub['meter_reading1']=sub['meter_reading1']+pd.Series(model1.predict(X))
#             sub['meter_reading1']=round(sub['meter_reading1'],4)/2
#             sub['meter_reading1']=np.expm1(sub['meter_reading1'])
        
#             subf=pd.concat([subf,sub],axis=0,ignore_index=True)
#             print('Shape of sub predicted chunk:',sub.shape)
#         else:
#             print('No Rows found:',X.shape)
    
#     print('Shape of final predicted chunk(2084880,2):',subf.shape)
#     df_test_BMW
#     gc.collect()
#     k=k+1


# In[ ]:


# subf['meter_reading1']=[x if x>=0 else 0 for x in subf['meter_reading1']]
# print('Shape of final predicted set (41697600,2):',subf.shape)
# subf.to_csv('sub_initial.csv',index=False)
# subf


# In[ ]:


# #CREATING SUBMISSION FILE
# submission_f=pd.merge(submission,subf,how='left',on='row_id')
# submission_f.drop('meter_reading',axis=1,inplace=True)
# submission_f.columns=['row_id','meter_reading']
# submission_f.to_csv('submission.csv', index=False)
# submission_f


# In[ ]:


# %%time
# #JOINING TRAIN SET AND BUILDING METADATA
# df_test_BM=pd.merge(df_test,df_building_metadata,how='left',on='building_id')
# print('AFTER JOINING OF BUILDING METADATA WITH TRAIN SET')
# feature_summary(df_test_BM)


# In[ ]:


# #VALUES REPLACING NULL VALUES
# print('Mean for floor_count is:',round(df_test_BM['floor_count'].mean(),0))
# print('Mode for year_built is:',df_test_BM['year_built'].mode()[0])

# #REPLACING NULL VALUES
# df_test_BM['floor_count'].replace(np.nan,round(df_test_BM['floor_count'].mean(),0),inplace=True)
# df_test_BM['year_built'].replace(np.nan,df_test_BM['year_built'].mode()[0],inplace=True)

# #FEATURE SUMMARY AFTER REPLACING NULL VALUES FOR FEATURES floor_count AND year_built
# print('Feature summary after replacing NULL values')
# feature_summary(df_test_BM)


# In[ ]:


# %%time
# # JOINING JOINED(TRAIN,BUILDING METADATA) WITH WEATHER TRAIN SET ON site_id AND timestamp
# cols=['site_id','timestamp','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',
#             'sea_level_pressure','wind_direction','wind_speed']
# df_test_BMW=pd.merge(df_test_BM,df_weather_test[cols],how='left',on=['site_id','timestamp'])
# feature_summary(df_test_BMW)


# In[ ]:


# #CLEARING DATAFRAMES
# del df_test_BM
# gc.collect()


# In[ ]:


# 41697600/20


# In[ ]:


# %%time
# #EXTRACTING INFORMATION FROM timestamp FEATURE
# df_test_BMW['month']=df_test_BMW.timestamp.apply(lambda x:x.month)
# df_test_BMW['day']=df_test_BMW.timestamp.apply(lambda x:x.day)
# df_test_BMW['hour']=df_test_BMW.timestamp.apply(lambda x:x.hour)
# df_test_BMW['week_day']=df_test_BMW.timestamp.apply(lambda x:x.weekday())
# df_test_BMW['week']=df_test_BMW.timestamp.apply(lambda x:x.isocalendar()[1])

# #GARBAGE COLLECTION
# gc.collect()


# In[ ]:


#JOINING TRAIN SET WITH BUILDING METADATA ON site_id
# df_train_building=df_train.join(df_build_meta,)


# In[ ]:


# df_test=pd.read_csv(path1+'test.csv',nrows=100000)


# In[ ]:


# df_test.head()


# In[ ]:


# df_submission=pd.read_csv(path1+'sample_submission.csv',usecols=['row_id'])


# In[ ]:


# feature_summary(df_submission)


# In[ ]:


# df_weather_test=pd.read_csv(path1+'weather_test.csv')
# feature_summary(df_weather_test)

