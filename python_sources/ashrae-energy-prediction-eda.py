#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# Check the page: https://www.kaggle.com/c/ashrae-energy-prediction/overview

# # Dependency

# In[ ]:


#matplotlab inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# # Utility functions

# In[ ]:


# Data loading functions
def loadTrainData():
    building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
    train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
    weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")  
    return building_metadata,train,weather_train

def loadTestData():
    test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
    weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
    return test,weather_test

def summary(dfs):
    for df in dfs:
        print(df[1]+':\n')
        print(df[0].info(null_counts=True))
        print('\n\n')
        
def convertType(df): 
    for col in df.columns:
        colType = str(df[col].dtypes)
        if colType[:3]=='int':
            if df[col].max()<=np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif df[col].max()<=np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif df[col].max()<=np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif colType[:5]=='float':
            if df[col].max()<=np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif df[col].max()<=np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df



# Visualize Trend (Moving Average) and Seasonality (Serial Correlation)
def plotTrendSeasonality(axes,df,building_id=0,meter=0,span=20,plotRaw=True):
    if plotRaw:
        s = df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading']
        axes[0].plot(s,label='Raw')

    # Exponentially-weighted moving average (EWMA)
    rolling = s.ewm(span=span)
    axes[0].plot(rolling.mean(),label='EWMA')

    # Autocorrelation
    if not s.empty:
        axes[1].acorr(s-rolling.mean(),usevlines=True, normed=True, maxlags=500)
        
        
           
# Detect the constant meter reading for select meter and building
def getSpanIdx(df,building_id,meter,maxLength=24):
    tempData = df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading']
    x= tempData.values
    idx = np.where(x[:-1]!=x[1:])[0]
    idx = np.concatenate((idx,[len(x)-1]))
    sp = np.concatenate(([idx[0]+1],np.diff(idx)))
    if any(sp>maxLength):
        return tempData.index[np.concatenate([range(x-y+1,x+1) for (x,y) in zip(idx[sp>maxLength],sp[sp>maxLength])])]
    else:
        return None

def plotConstantSpan(ax,df,building_id=0,meter=0,maxLength=24,plotRaw=True):
    if plotRaw:
        ax.plot(df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading'],label='Raw')
    idx = getSpanIdx(df=df,building_id=building_id,meter=meter,maxLength=maxLength)
    if idx is not None:
        ax.scatter(x=df_train.loc[idx].index,y=df_train.loc[idx]['meter_reading'],c='r',label='Constant Span')
        
        
        
# Visualize random building on select/random meter and site
def plotRandBuild(df_train,df_building_metadata,site_id=None, meter=None, maxLength=24):
    if site_id is None:
        rng = np.random.default_rng()
        site_id = rng.integers(0,df_building_metadata.site_id.max()+1)
    if meter is None:
        rng = np.random.default_rng()
        meter = rng.integers(0,4)
    building_id = df_building_metadata[df_building_metadata['site_id']==site_id]['building_id'].sample(n=1).values[0]
    
    fig,axes = plt.subplots(1,2,figsize=[30,5])
    axes[0].set_title('Site_id = {}, Building_id = {}, Meter = {}'.format(site_id,building_id,meter))
    axes[1].set_title('Autocorrelation of residual')
    axes[1].set_xlabel('Lags')
    plotTrendSeasonality(axes,df=df_train,building_id=building_id,meter=meter,plotRaw=True)
    plotConstantSpan(ax=axes[0],df=df_train,building_id=building_id,meter=meter,plotRaw=False)
    axes[0].legend()


# # Import training data

# In[ ]:


df_building_metadata, df_train, df_weather_train = loadTrainData()
summary([(df_building_metadata,'building_metadata'), (df_train,'train'), (df_weather_train,'weather_train')])


# Reduce size for numeric columns.

# In[ ]:


df_building_metadata = convertType(df_building_metadata)
df_train = convertType(df_train) 
df_weather_train = convertType(df_weather_train)
summary([(df_building_metadata,'building_metadata'), (df_train,'train'), (df_weather_train,'weather_train')])


# Merge data

# In[ ]:


df_train_merge = df_train.merge(df_building_metadata, how='left',on='building_id').merge(df_weather_train,how='left',on=['site_id','timestamp'])
summary([(df_train_merge,'building_metadata')])


# # EDA
# 
# We summarize the comments at the beginning of each section and then show the stats and visualizations.

# ## Building data
# 
# * Missing values in year_built and floor_count
# * Inf in year_built
# 
# 
# * No obvious linear dependency, except a bit between floor_count and square_feet
# 
# 
# * The top primary_use is from Education, as well as Office, Entertainment/public assembly, Lodging/residential, and Public services
# * Public services are mainly on site 3 and 8
# * Site 7 and 11 are only for Education

# In[ ]:


print(df_building_metadata.describe())
sns.heatmap(df_building_metadata.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)
sns.pairplot(df_building_metadata)
sns.catplot(y='primary_use',kind='count',data=df_building_metadata.sort_values('primary_use'),color='c')
sns.catplot(y='primary_use',kind='count',data=df_building_metadata.sort_values('primary_use'),color='c',col='site_id',height=5,aspect=.3)
sns.catplot(x='site_id',y='square_feet',data=df_building_metadata,kind='bar')


# ## Weather training data
# 
# Note: timeseries visualizations per site are commented out as the figure takes too much memory
# 
# * Missing values in all columns, except for site_id. cloud_coverage and precip_depth_1_hr contain a lot of missing values
# * Most of precip_depth_1_hr are 0
# * Timeseries data on site 15 are about 300 less than on other sites
# 
# 
# * aire_temp and dew_temp are highly correlated
# * cloud_coverage is of int type
# 
# 
# * Abnormal temp on site 13
# * Missing cloud coverage on site 7,11
# * Abnormal cloud coverage on site 1,5,6,9,10,14
# * Missing precip data on site 1,5,12 (or no precip at these locations)
# * Missing pressure data on site 5
# * Abnormal pressure behavior on site 1,12
# * A horizontal margin of wind speed appares on most sites
# * The pattern of wind speed on site 12 is different from others
# 
# 

# In[ ]:


print(df_weather_train.describe())
print('\n')
print('Timeseries data count per site: \n')
print(df_weather_train.site_id.value_counts(sort=False))
sns.heatmap(df_weather_train.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)
sns.pairplot(df_weather_train)
sns.catplot(x='site_id',y='cloud_coverage',data=df_weather_train,kind='box')
g = sns.FacetGrid(df_weather_train,subplot_kws=dict(projection='polar'),despine=False,height=10,sharex=False, sharey=False)
g.map(sns.countplot,'wind_direction',alpha=.5)
g = sns.FacetGrid(df_weather_train,col='site_id',subplot_kws=dict(projection='polar'),despine=False,col_wrap=4)
g.map(sns.countplot,'wind_direction',alpha=.5)
# sns.relplot(x='timestamp',y='air_temperature',data=df_weather_train,col='site_id')
# sns.relplot(x='timestamp',y='precip_depth_1_hr',data=df_weather_train,col='site_id')
# sns.relplot(x='timestamp',y='sea_level_pressure',data=df_weather_train,col='site_id')
# sns.relplot(x='timestamp',y='wind_speed',data=df_weather_train,col='site_id')


# ## Meter training data
# 
# Given each meter type (0:electricity, 1:chilledwater, 2:steam, 3:hotwater), we will explore the data at different levels as follows:
# * meter at all locations
# * meter on each site
# * meter in each building
# 
# Note that we may not drill down all the analysis to the building level given that many buildings. Instead, we will only explore the meters in certain buildings to identify outliers and missing/ill-condition data. For example, the timeseries visualization per meter.
# 
# Key results:
# * Meter 0 has the most observations, while the others have much less
# * No apparent correlation between overall meter_reading and other features
# * Meter 0 has low mean, max, and range
# * Meter 2 has the highest mean, deviation, max, and range
# * Meter 0 has some linear dependency to square_feet and floor_count
# * Meter 1 has small linear dependency to square_feet and year_build
# * Religious worship has extremely low consumption in meter 0. 
# * Religious and Warehouse/storage have no reading for meter 1-3. Maybe such equipment/meter are not installed for some reason.
# * With respect to year_built, patterns from meters 0-2 somehow have similar behavior, especially after 1950. This may be driven by the facts on Site 3,8,15.
# * Site 7 has extremely high mean reading in meter 0 from Education
# * Site 13 has extremely high mean reading in meter 2 from Education
# * Site 0 has extremely high mean reading in meter 1 from Education, Lodging/residential and Other
# * Site 6 has extremely high mean reading in meter 1 from Entertainment/public_assembly
# * Site 15 has extremely high mean reading in meter 3 from Education
# * Site 7 has high mean reading in meter 3 from Education
# * Site 10 has high mean reading in meter 3 from Entertainment/public_assembly
# * The overall daily mean meter_reading for meter 1 grows significantly in Q3, which is mainly driven by site 6
# * The overall daily mean meter_reading for meter 2 is extremely high in H1 and has a spike in Q4, which is mainly driven by site 13
# * Meter 0 on site 0 seems not working until mid of Q2
# * Meter 0 on sites 8,9, and 14 may contain outliers for certain days
# * Meter 0,2, and 3 on site 15 have consecutive missing values during the same time period in Q1
# * Meter 1 on site 13 may contain outliers in Q3 and Q4
# * Meter 3 on sites 1 may contain outliers for certain days
# * Meters from some buildings have conseutive reading for more than a day which is abnormal
# 
# 
# 

# In[ ]:


# Overall for all meters
print(df_train_merge.describe())
print('\n')
print('Timeseries data count per meter: \n')
print(df_train_merge.meter.value_counts(sort=False))
sns.heatmap(df_train_merge.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)

# Overall for each meter
sns.catplot(x='meter',y='meter_reading',data=df_train_merge,kind='box',showfliers=False)

plt.figure(figsize=(50,10))
for i in range(4):
    plt.subplot(1,4,i+1)
    sns.heatmap(df_train_merge[df_train_merge.meter==i].corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)
    
sns.catplot(x='meter_reading',y='primary_use',data=df_train_merge,col='meter',kind='bar',sharex=False)
sns.relplot(x='year_built',y='meter_reading',data=df_train_merge,hue='meter',kind='line',ci=None,aspect=5)
sns.catplot(y='meter_reading',x='site_id',data=df_train_merge,col='meter',kind='bar',sharey=False)


# In[ ]:


# Each meter per site
sns.catplot(x='meter_reading',y='primary_use',data=df_train_merge,col='meter',row='site_id',kind='bar',sharex=False)
sns.relplot(x='year_built',y='meter_reading',data=df_train_merge,hue='meter',kind='line',ci=None,aspect=5,row='site_id',facet_kws=dict(sharey=False))


# In[ ]:


# Timeseries for each meter
df_train_merge['timestamp'] = pd.to_datetime(df_train_merge['timestamp'])

plt.figure(figsize=[30,5])
plt.subplot(131)
sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.month,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Month')
plt.subplot(132)
sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.week,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Week')
plt.subplot(133)
sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Day of Year')

sns.relplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter']).mean().reset_index(),
            ci=None,row='meter',kind='line',facet_kws=dict(sharey=False),aspect=4)
sns.relplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter','site_id']).mean().reset_index(),
            ci=None,col='meter',row='site_id',kind='line',facet_kws=dict(sharey=False),aspect=4)

sns.relplot(x='timestamp',y='meter_reading',data=df_train_merge.set_index(['site_id','meter']).loc[(9,0)],kind='line',ci=None,aspect=4)


# In[ ]:


# Trend, seasonality and constant span for random meters
plotRandBuild(df_train,df_building_metadata)
plotRandBuild(df_train,df_building_metadata,meter=0)
plotRandBuild(df_train,df_building_metadata,site_id=6,meter=1)
plotRandBuild(df_train,df_building_metadata,site_id=13,meter=2)
plotRandBuild(df_train,df_building_metadata,site_id=15)

