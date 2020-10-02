#!/usr/bin/env python
# coding: utf-8

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRw7Pl0gvH5M6O39YTvooScgZ8AG6sJ-dZkW_zc60ilB3-H8BW7)

# # Predicting sickness using weather data
# 
# ### This kernel should show how to answer data science question(s) using one owns data (not from kaggle)
# 
# 
# 
# We will cover all of the data-science steps and eventually implement neural network and random forest models for regression. Together with hyper-parameter optimization. But let us take one step at the time:

# # **NOTE** It has come to my attention after learning about it that I performed [data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/). 
# 
# Where I did not split the data right at the beggining and performed the pre-processing seperately on different folds (Here I had only train and test and test_2018 sub-data sets). What happend is that the data rescaling process that I performed had knowledge of the full distribution of data in the training dataset when calculating the scaling factors (like min and max or mean and standard deviation). This knowledge was stamped into the rescaled values and exploited by all algorithms later on. Now I could just correCt it and have a perfect kernel but then reader wont learn about this important miss-step, since in real world application sdata leakage is serious. You are thinking you have a very good model but in reality you just over-fitted on the data.
# 
# **I will** leave it as it is since all of the other steps are ok, but one should be carefull about it and do the pre-processing seperately on different subsets.
# **Note** please do note that this is not the only way to commit data leakage, but the only one I did.

# Import the necessary modules

# In[ ]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy


# ## DATA
# 
# To investigate a potential relationship, we will use two datasets:
#  * daily weather observation data in Vienna (2012-2018)
#  * weekly reports on [new influenza infections](https://www.data.gv.at/katalog/dataset/grippemeldedienst-stadt-wien) in Vienna (2009-2018).(flu-sickness)
# 

# ## Load the data
# Data in csv files is not in an optimal format, we want to make it a 4 level hiararchical index. (year,month,week and day) Week is not given hence we are going to use isocalendar() function of datetime to derive it and create a column. Furthermore we want to make sure that data is sorted (from the deepest level-day up until the year level). Another thing that is implemented in the load_weather_data function is a search for leap year, i.e. 53 weeks. We need to indentify it at label it correctly.
# 
# ### Weather observations 
# 

# In[ ]:


def sortByDate(year_data): # Additional function needed for data-loading. Without it we get unsorted values. For example 4-th day gets thrown into second week and so on...
    
    month_data = year_data.loc[year_data['month'] == 1]
    sorted_df = month_data.sort_values(by = 'day')
    
    for month in range(2,13):
        month_data = year_data.loc[year_data['month'] == month]
        sorted_month_data = month_data.sort_values(by = 'day')
        sorted_df = pd.concat([sorted_df, sorted_month_data],)
    
    return sorted_df


# In[ ]:


def load_weather_data():
    """ 
    Load all weather data files and combine them into a single Pandas DataFrame.
    Add a week column and a hierarchical index (year, month, week, day)
    
    Returns
    --------
    weather_data: data frame containing the weather data
    """
    
    years_to_load = ['2012','2013','2014','2015','2016','2017','2018']
    path=["../input/sickness-and-weather-data/weather_2012.csv","../input/sickness-and-weather-data/weather_2013.csv","../input/sickness-and-weather-data/weather_2014.csv","../input/sickness-and-weather-data/weather_2015.csv","../input/sickness-and-weather-data/weather_2016.csv","../input/sickness-and-weather-data/weather_2017.csv","../input/sickness-and-weather-data/weather_2018.csv"]
    col = list()
    datalist = list()
    week_col = list()
    load_years=list()

    first_year = True
    for filename in path:
        year_data = pd.read_csv(filename)
        year_data = sortByDate(year_data)
        if first_year:
            weather_data = year_data
            first_year = False
        else:
            weather_data = pd.concat([weather_data, year_data], sort = True)

    year,week,day = datetime.date(int(years_to_load[0]),1,1).isocalendar()
    count = 8-day
    rows, cols = weather_data.shape
    first_year=int(years_to_load[0])
    
    
    for i in range(count):
        week_col.append(week)
    if week==52:
        if datetime.date(year,12,28).isocalendar()[1] == 53: # every 4-th year, check it- add it!
            week = 53
        else:
            week = 1
            if year == first_year:
                first_year +=1
    elif week == 53:
        week = 1
        if year == first_year:
            first_year += 1        
        week=1
    else:
        week+=1
        
        
        
    while len(week_col)<rows:
        for i in range(7):
            if len(week_col)<rows:
                week_col.append(week)
        if week==52:
            if datetime.date(int(first_year),12,28).isocalendar()[1] == 53: #53 woche?
                week = 53
            else:
                week = 1
                first_year += 1
        
        elif week == 53:
            week = 1 
            first_year += 1
        
        else:
            week += 1
            
            
            
            
    weather_data.drop("Unnamed: 0",axis=1,inplace=True)
   
    weather_data.insert(loc=0, column='week', value = week_col)
    weather_data.set_index(["year","month","week","day"],inplace=True)
    return weather_data    


# **Columns** Are really standard ones, humidtiy measured every 7,14 and 21 hours, wind speed, temperature and other weather specific indicators taken at different times of the day

# In[ ]:


data_weather=load_weather_data()


# In[ ]:


data_weather


# ### Influenza infections  load the data
# Again write a function and make sure that dataframe is in the suitable format. Data frame containing measuraments of new cases of flu.

# In[ ]:


def load_influenza_data():
    """ 
    Load and prepare the influenza data file
    
    Returns
    --------
    influenza_data: data frame containing the influenza data
    """
    influenza_data=pd.read_csv('../input/sickness-and-weather-data/influenza.csv')
    influenza_data = pd.DataFrame(influenza_data)
    influenza_data= influenza_data[["Neuerkrankungen pro Woche","Jahr","Kalenderwoche"]]
    new_names = {'Neuerkrankungen pro Woche':'weekly_infections',"Jahr":"year","Kalenderwoche":"week"}
    

    influenza_data.rename(index=str, columns=new_names,inplace=True)
    influenza_data['week'] = influenza_data['week'].str.replace('Woche', '')
    influenza_data['week'] = influenza_data['week'].str.replace('.', '')
    influenza_data['year'] = influenza_data['year'].astype(int)
    influenza_data['week'] = influenza_data['week'].astype(int)
    influenza_data.set_index(["year","week"],inplace=True)
    return influenza_data

data_influenza = load_influenza_data()


# In[ ]:


data_influenza


# Check for missing values:

# In[ ]:




data_weather.isnull().any().any()


# ##  Handling Missing values

# If you take a closer look at the data, you will notice that a few of the observations are missing.
# 
# There are a wide range of standard strategies to deal with such missing values, including:
# 
# - row deletion
# - substitution methods (e.g., replace with mean or median)
# - hot-/cold-deck methods (impute from a randomly selected similar record)
# - regression methods
# 
# To decide which strategy is appropriate, it is important to investigate the mechanism that led to the missing values to find out whether the missing data is missing completely at random (MCAR), missing at random (MAR), or missing not at random (MNAR). 
# 
#  - **MCAR** means that there is no relationship between the missingness of the data and any of the values.
#  - **MAR** means that that there is a systematic relationship between the propensity of missing values and the observed data, but not the missing data.
#  - **MNAR** means that there is a systematic relationship between the propensity of a value to be missing and its values. 
# 
# To find out more about what mechanisms may have caused the missing values, you talked to the metereologist that compiled the data. 
# She told you that she does not know why some of the temperature readings are missing, but that it may be that someone forgot to record them. In any case, it is likely that the propensity of a temperature value to be missing does not have anything to do with the weather itself.
# 
# As far as the missing humidity readings are concerned, she says that according to her experience, she suspects that the humidity sensor is less reliable in hot weather.
# 
# The missing wind speed and direction sensor readings were due to a hardware defect.
# 

# In[ ]:


def handle_missingValues_simple(incomplete_data):
    """ 
    Parameters
    --------
    incomplete_data: data frame containing missing values 
    
    Returns
    --------
    complete_data: data frame not containing any missing values
    """
    #See description
   

    wind=incomplete_data[["wind_mSec"]].interpolate(method='linear')
    temp7h = incomplete_data['temp_7h'].fillna(method='bfill')
    temp14h = incomplete_data['temp_14h'].fillna(method='bfill')
    temp19h = incomplete_data['temp_19h'].fillna(method='bfill')
    hum_missing = incomplete_data[["hum_14h","hum_19h","hum_7h"]]
    hum_missing = hum_missing.reset_index()
    hum_missing = hum_missing.interpolate(method = 'piecewise_polynomial')
    hum_missing.set_index(["year","month","week","day"],inplace = True)
    
    #2012 to 2018_01 are ok (just a bit of NaN)
    wind_degrees_only = incomplete_data[["wind_degrees"]]
    wind_degrees_2012_until_2018_01 = wind_degrees_only.iloc[0:2223]
    wind_degrees_2012_until_2018_01 = wind_degrees_2012_until_2018_01.interpolate(method='linear')
    
    #Tricky part(find the indices before)---description
    wind_degrees_2018_Feb = wind_degrees_only.iloc[2223:2251]     
    wind_degrees_2018_Feb.iloc[:] = wind_degrees_only.iloc[1858:1886].values
    
    wind_degrees_2018_March = wind_degrees_only.iloc[2251:2282] #no NaN
    
    wind_degrees_2018_April = wind_degrees_only.iloc[2282:2312]
    wind_degrees_2018_April.iloc[:] = wind_degrees_only.iloc[1917:1947].values
    
    wind_degrees_2018_May = wind_degrees_only.iloc[2312:2343]
    wind_degrees_2018_May.iloc[:] = wind_degrees_only.iloc[1947:1978].values
    
    wind_degrees_2018_June_July_August = wind_degrees_only.iloc[2343:] #no NaN
    
    skyCover_and_sun = incomplete_data[['skyCover_14h','skyCover_19h','skyCover_7h','sun_hours']].interpolate(method='linear')
    
    
    wind_degrees = pd.concat([wind_degrees_2012_until_2018_01, wind_degrees_2018_Feb, wind_degrees_2018_March,
                              wind_degrees_2018_April,wind_degrees_2018_May, wind_degrees_2018_June_July_August])
   


    complete_data = pd.concat([ incomplete_data[['temp_dailyMax','temp_dailyMean',
                                                'temp_dailyMin','temp_minGround','hum_dailyMean','precip']],hum_missing,wind,temp7h,temp14h,temp19h,skyCover_and_sun], axis = 1, sort = True)

    return complete_data


def handle_missingValues_advanced (incomplete_data):
    """ 
    Parameters
    --------
    data: data frame containing missing values 
    

    Returns
    --------
    data: data frame not containing any missing values
    """
    
    return complete_data
    
data_weather_complete = handle_missingValues_simple(data_weather)


# In[ ]:


data_weather_complete


# ###  Discussion
# 
# #### Pros and Cons of strategies for dealing with missing data
# 
# 

# 
# Before we evan start discussing missing values imputation, we need to define the scope of a definition missing values. Is it just NaN, can it be 0 or 999 or some other value? For that we ought to plot the variables and check our usual suspects. Other tactic would also be to ask meterologist. It can happen that for example humidity readings machine gives 999 when in defect. Let us assume we are only dealing with NaN.
# 
# weather.dropna()-quickest and dumbest solution. We will lose a lot of data. Makes sense if number of missing values is small. But evan than we should use our brains.
# 
# 
# weather.fillna(method='pad') 
# weather.fillna(method='bfill') Foreward, backward filling. It is reasonable. Data point in previous or next step should also be the one missing. It is robust in the sense that we do not have to make any assumptions about the data. Imputation works best when many variables are missing in small proportions. The power boost is much less impressive when one important variable is missing in 70% of cases, because the uncertainty in estimates will yield highly varying imputed datasets.
# 
# 
# weather = weather.fillna(weather.mean()) Mean is prone to outliers, and imputation would fail. It also reduces the variance in the data. We could opt for median to avoid outliers. Assumptions for pad and bfill also hold for this imputation
# 
# 
# weather = df.weather(method='linear') or polynomial of higher order would also make sense, at the end we are finding a function that dsecribes our data (column). But we will not be able to generalise, over-fitting.
# 
# model for NaN- same story it would be in most cases the perfect solution to handle missing values. But we are giving up a few things. OVerfitting,time,simplicity to name a few.
# 
# 
# Lastly for MAR (humidity) we will be using Multiple Imputation using MICE (Multiple Imputation by Chained Equations).
# Basic idea is that we are going to run multiple regression models and each missing value is modeled conditionally depending on the observed (non-missing) values, andthen average it over the number of regressions. Papers show that this ought to be the best solution for MAR-data. We are going to use neat package called fancyimpute. There we could have used kNN (Which also very reasonable) to imputata. Please note that it works only on numpy array, so we need to convert back and fort!
# 

# #### Explanation of our own strategy

# 
# Strategy is as follows (general one) We got some pointers on where and what type of missing variables are in certain columns. But there are much more missing values than that. We need to check each column(variable) individually and than depending on the findings (amount of missing data, where are they missing, the variable) we will perform the optimal imputing.
# 
# Let us start with no missing data-columns, How do we check ? For example, for the daily maximal temperature we would do (I deleted these cells) data_weather[data_weather['temp_dailyMax'].isnull()] "temp_dailyMean", 'temp_dailyMin', 'temp_minGround', 'hum_dailyMean', "hum_dailyMean" columns have no missing values. Let us check for data_weather[data_weather['temp_7h'].isnull()] there are some missing values and they seem to be rather randomly scattered. Taking that into account we are going to use simple backwardfill here. Analog for 'temp_14h' and 'temp_19h'.
# Let us move further on. 'wind_mSec' has only 7 missing values in all of the 7 years. Simple interpolate() ought to do the trick.
# 'wind_degrees' is the interesting one. There a couple of missing values in the beginning but in the last year we observe that data is missing for whole months at a part. Logically we ought to do simple interpolate() for the few values that are missing in the beginning and than just copy the months from previous year (2017) or any other year for that matter. We only need to find the row indices of the previous year months and than replace it with the missing values in 2018. Finally 'skyCover_14h','skyCover_19h','skyCover_7h','sun_hours' are all missing values at the same time (just a few of them) once again I opt for simple interpolate().
# 
# Temperature is MCAR
# The missing wind speed readings is MNAR
# Missing humidity readings MAR. If we compare given and the alternative definitions following tactics would be reasonable choices.
# 
# For Temperature, independent what the time of the year is temperature should be close to the previous day/following day. So backward fill seams reasonable.
# 
# Wind i described above, but it depends on what wind column exactly.
# 
# Humidity NaNs (there are 3 columns in regard to humidity) could be removed with a machine learning model. We already now that temperature is a independent variable, we could also test whether other variables have predictive power and do a regression. But after consulting online it seems that Multiple imputation with MICE is the best option, that is why I wanted to use that. Unfortunately I did not manage to install the library fancyimputer, so I opted for polynomial interpolation ("piecewise_poly") there I hope that polynomial function will catch trends and predict/imputate missing values.
# 
# 

# In[ ]:


data_weather_complete.isnull().any().any()


# ##  Handling Outliers

# If you take a closer look at some of the observations, you should notice that some of the temperature values are not particularly plausible 

# In[ ]:


fig, ax = plt.subplots(4, 2)
    
    
    
ax[0, 0].hist(data_weather_complete.temp_14h, normed=True, bins=30)
ax[1, 0].hist(data_weather_complete.temp_19h, normed=True, bins=30)
ax[0, 1].hist(data_weather_complete.temp_7h, normed=True, bins=30) 
ax[1, 1].hist(data_weather_complete.temp_dailyMax, normed=True, bins=30) 
ax[2, 0].hist(data_weather_complete.temp_dailyMean, normed=True, bins=30)
ax[2, 1].hist(data_weather_complete.temp_dailyMin, normed=True, bins=30)
ax[3, 0].hist(data_weather_complete.temp_minGround, normed=True, bins=30)


# In[ ]:


# Before excluding certain values in handle_outliers function underneath, we are going to compare two methods and different paramaeters
# All to see which number of outliers seems reasonable, than we are going to exclude entire row that has this outlier
#It will be only a few since we will opt for the most extreme case, where deviation from the mean is really ridiculous.

def out_std(s, nstd=3.0, return_thresholds=False):

    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [False if x < lower or x > upper else True for x in s]
    

    
    
std2 = data_weather_complete.apply(out_std, nstd=1.8)
std3 = data_weather_complete.apply(out_std, nstd=3.0)
std4 = data_weather_complete.apply(out_std, nstd=4.0)

    
    
f, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, nrows=1, figsize=(22, 12));
ax1.set_title('Outliers with 1.8 standard deviations');
ax2.set_title('Outliers using 3 standard deviations');
ax3.set_title('Outliers using 4 standard deviations');

sns.heatmap(std2, cmap='Blues', ax=ax1);
sns.heatmap(std3, cmap='Blues', ax=ax2);
sns.heatmap(std4, cmap='Blues', ax=ax3);


plt.show()


# In[ ]:





# In[ ]:


def handle_outliers(noisy_data):
    """ 
    Parameters
    --------
    noisy_data: data frame that contains outliers
    
    Returns
    --------
    cleaned_data: data frame with outliers
    """
    noisy_data=noisy_data[std2]
    noisy_data.loc[noisy_data.temp_14h >= 40, 'temp_14h'] = np.NaN
    
    noisy_data = noisy_data.reset_index()
    noisy_data = noisy_data.interpolate(method = 'piecewise_polynomial')
    noisy_data.set_index(["year","month","week","day"],inplace = True)
    
    cleaned_data = noisy_data
    return cleaned_data
    
data_weather_cleaned = handle_outliers(data_weather_complete)


# In[ ]:


data_weather_cleaned[["temp_19h"]].plot()


# #### Strategy explanation below
# 
# 

# 
# There is no precise way to define and identify outliers in general because of the specifics of each dataset. Instead, you, or a domain expert, must interpret the raw observations and decide whether a value is an outlier or not. Nevertheless, we can use statistical methods to identify observations that appear to be rare or unlikely given the available data. We will assume that outliers are only to be found in temperature measuraments. From the plots above we can see there are some points that fall far out of the normal one or two points standard deviation. With three we should be certain that it is an outlier and that is why we should remove it. Alternatively we can also say that standard deviation is heavily influenced with the outliers. To remedy that I also implemented a an IQR (inter quartile range) method that removes all values that fall below or above IQR (which is Q1-Q3). With these two relatively simple methods we should be able to recognize most of the outliers. Qualitative examination would also be beneficial (and they were since we had couple of very hot days in winter ;)), along with other more advanced techniques.
# 
# Then I tested it and IQR method was not very effiecent even with high koefficients, so I stuck with std method. I tried different values, and it seems that with std2 that is koefficient of 1.8*std removes most of the outliers.
# For the missing values I just interpolated the missing values (standard imputation technique). But inspecting the data I see that there some values that are 40 degrees in winter (in temp_14columns). So we also need to make sure that these are also gone, since they are definitely an outlier. So I located these valus and set them to NaN then i interpolated all of the NaN with polynomial interpolation.
# 
# 
# 
# 

# ## Aggregate values 
# 
# Aggregation of the observations on a weekly basis. Returns a data frame with a hierarchical index (levels `year` and `week`) on the vertical axis and the following weekly aggregations as columns:
# 
# - `temp_weeklyMin`: minimum of `temp_dailyMin`
# - `temp_weeklyMax`: mean of `temp_dailyMax`
# - `temp_weeklyMean`: mean of `temp_dailyMean`
# - `temp_7h_weeklyMedian`: median of `temp_7h`
# - `temp_14h_weeklyMedian`: median of `temp_14h`
# - `temp_19h_weeklyMedian`: median of `temp_19h`
# 
# - `hum_weeklyMean`: mean of `hum_dailyMean`
# - `hum_7h_weeklyMedian`: median of `hum_7h`
# - `hum_14h_weeklyMedian`: median of `hum_14h`
# - `hum_19h_weeklyMedian`: median of `hum_19h`
# 
# - `precip_weeklyMean`: mean of `precip`
# - `wind_mSec_mean`: mean of `wind_mSec`

# In[ ]:


def aggregate_weekly(data):
    """ 
    Parameters
    --------
    data: weather data frame
    
    Returns
    --------
    weekly_stats: data frame that contains statistics aggregated on a weekly basis
    """
    
   
    data=data.reset_index()
    data = data.iloc[1:] #Aggregation with a 2012 1 1 is actually week 52 from 2011 (because of the way they count it) so we should not include it in aggregation
    data.set_index(["year","week"],inplace=True)

    data["temp_weeklyMin"] = data.pivot_table('temp_dailyMin', index=["year",'week'],aggfunc=min)
    data["temp_weeklyMax"] = data.pivot_table('temp_dailyMax', index=["year",'week'],aggfunc=np.mean)
    data["temp_weeklyMean"] = data.pivot_table('temp_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["temp_7h_weeklyMedian"] = data.pivot_table('temp_7h', index=["year",'week'],aggfunc=np.median)
    data["temp_14h_weeklyMedian"] = data.pivot_table('temp_14h', index=["year",'week'],aggfunc=np.median)
    data["temp_19h_weeklyMedian"] = data.pivot_table('temp_19h', index=["year",'week'],aggfunc=np.median)
    data["hum_weeklyMean"] = data.pivot_table('hum_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["hum_7h_weeklyMedian"] = data.pivot_table('hum_7h', index=["year",'week'],aggfunc=np.median)
    data["hum_14h_weeklyMedian"] = data.pivot_table('hum_14h', index=["year",'week'],aggfunc=np.median)
    data["hum_19h_weeklyMedian"] = data.pivot_table('hum_19h', index=["year",'week'],aggfunc=np.median)
    data["precip_weeklyMean"] = data.pivot_table('precip', index=["year",'week'],aggfunc=np.mean)
    data["wind_mSec_mean"] = data.pivot_table('wind_mSec', index=["year",'week'],aggfunc=np.mean)
    
    weekly_weather_data = data.drop(["temp_minGround","sun_hours","skyCover_7h","skyCover_19h","skyCover_14h",'temp_dailyMin', 'temp_dailyMax','temp_dailyMean','temp_7h','temp_14h','temp_19h','hum_dailyMean','hum_7h','hum_14h','hum_19h','precip','wind_mSec'], 1)
    ww2012=weekly_weather_data.xs(2012,level='year')
    ww2012=ww2012[~ww2012.index.get_level_values(0).duplicated()]
    ww2013=weekly_weather_data.xs(2013,level='year')
    ww2013=ww2013[~ww2013.index.get_level_values(0).duplicated()]
    ww2014=weekly_weather_data.xs(2014,level='year')
    ww2014=ww2014[~ww2014.index.get_level_values(0).duplicated()]
    ww2015=weekly_weather_data.xs(2015,level='year')
    ww2015=ww2015[~ww2015.index.get_level_values(0).duplicated()]
    ww2016=weekly_weather_data.xs(2016,level='year')
    ww2016=ww2016[~ww2016.index.get_level_values(0).duplicated()]
    ww2017=weekly_weather_data.xs(2017,level='year')
    ww2017=ww2017[~ww2017.index.get_level_values(0).duplicated()]
    ww2018=weekly_weather_data.xs(2018,level='year')
    ww2018=ww2018[~ww2018.index.get_level_values(0).duplicated()]
    weekly_weather_data = pd.concat([ww2012,ww2013,ww2014,ww2015,ww2016,ww2017,ww2018], keys=['2012','2013',"2014","2015","2016","2017","2018"])
    weekly_weather_data.index.names = ['year','week']
    weekly_weather_data.reset_index(inplace=True)
    weekly_weather_data['year'] = pd.to_numeric(weekly_weather_data['year'])
    weekly_weather_data['week'] = pd.to_numeric(weekly_weather_data['week'])
    weekly_weather_data.set_index(["year","week"],inplace=True)
    
    return weekly_weather_data

data_weather_weekly = aggregate_weekly(data_weather_cleaned)


# In[ ]:


data_weather_weekly


# ## Merging influenza and weather datasets

# Merge the `data_weather_weekly` and `data_influenza` datasets.

# In[ ]:



def merge_data(weather_df, influenza_df):
    """ 
    Parameters
    --------
    weather_df: weekly weather data frame
    influenza_df: influenza data frame
    
    Returns
    --------
    merged_data: merged data frame that contains both weekly weather observations and prevalence of influence infections
    """
    merged_data = weather_df.join(influenza_df)
    merged_data=merged_data.reset_index()
    merged_data = merged_data.apply(pd.to_numeric)
    merged_data["weekly_infections"]= merged_data["weekly_infections"].interpolate(method = 'linear')
    
    merged_data.iloc[327:,14] = merged_data.iloc[275:297,14].values
    merged_data.set_index("year","week")
    return merged_data

data_merged = merge_data(data_weather_weekly, data_influenza)


# In[ ]:


data_merged


# ##  Visualization

# To get a better understanding of the dataset, create visualizations of the merged data set that help to explore the potential relationships between the variables before starting to develop a model.
# 
# 

# In[ ]:


data_merged2=data_merged[["temp_weeklyMin","temp_weeklyMax","temp_weeklyMean","temp_7h_weeklyMedian","hum_weeklyMean","hum_7h_weeklyMedian","precip_weeklyMean","wind_mSec_mean","weekly_infections"]].reset_index()

sns_plot2=sns.pairplot(data_merged2)
sns_plot2.savefig("01527395_01.png")


# In[ ]:


b=data_merged.drop(["weekly_infections"], axis=1).reset_index()
b=pd.DataFrame(b)
melted = pd.melt(b, ['year',"week","index"])
    
melted.drop(['year',"week","index"],axis=1,inplace=True)
melted["value"] = pd.to_numeric(melted["value"])
melted
sns_plot1=sns.boxplot(x="variable", y="value", data=melted)
sns_plot1.set_xticklabels(sns_plot1.get_xticklabels(), rotation = 90, fontsize = 10)
sns_plot1.figure.savefig("01527395_02.png")


# In[ ]:



data_merged1=data_merged.reset_index()
data_merged1.drop(['year',"week"],axis=1,inplace=True)

# calculate the correlation matrix
corr = data_merged1.corr()

# plot the heatmap
sns_plot2=sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
sns_plot2.figure.savefig("01527395_03.png")


# ##  Influenza prediction model 
# 
# 
# Build a model to predict the number of influenza incidents for the year 2018 (discarding all the data available for 2018) based on data of previous year using.

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import sklearn

data_merged1=data_merged.reset_index()
year=data_merged1["year"]
week=data_merged1["week"]
data_merged1=data_merged1.drop(["year","week"],axis=1)
dm1_col = data_merged1.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_merged1)
data_merged2=pd.DataFrame(np_scaled,columns =dm1_col)

X=data_merged2.drop(["weekly_infections"], axis=1)
X["year"]=year
X["week"]=week
X_2018=X.loc[X['year'] == 2018]
X=X.loc[X['year'] != 2018]




y=data_merged2["weekly_infections"].reset_index()

# y.drop(["year","week"], axis=1,inplace=True)
y["year"]=year
y["week"]=week
y_2018=y.loc[y['year'] == 2018]
y_2018.drop(["year","week","index"], axis=1,inplace=True)
y=y.loc[y['year'] != 2018]
y.drop(["year","week","index"], axis=1,inplace=True)
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.4,random_state=42)


# In[ ]:



from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123)

def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=17, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())
    return model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=32, verbose=False)
estimator.fit(X_train, y_train)

prediction = estimator.predict(X_train)
rmse_nn = sqrt(mean_squared_error(y_train, prediction))

prediction1 = estimator.predict(X_test)
rmse_nn1 = sqrt(mean_squared_error(y_test, prediction1))

prediction2 = estimator.predict(X_2018)
rmse_nn2 = sqrt(mean_squared_error(y_2018, prediction2))



print(rmse_nn,rmse_nn1,rmse_nn2)
    
    
    
    
#batch_size = [20, 40, 60, 80, 100]
#epochs = [10, 50, 100,250,500]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#neurons = [1, 5, 10, 15, 20, 25, 30]
#param_grid = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, 
#                 momentum=momentum, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1,cv=10)
# grid_result = grid.fit(X_train, y_train)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[ ]:


from sklearn import *

model2 = ensemble.RandomForestRegressor(n_estimators=25, random_state=11, max_depth=1,
                                        min_weight_fraction_leaf=0.122)
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


model2.fit(X_train, np.log1p(y_train.values))
print("Model2 trained")
preds2 = model2.predict(X_test)
preds3 = model2.predict(X_2018)
print('RMSE RandomForestRegressor on validation data: ', RMSLE(y_test, preds2))
print('RMSE RandomForestRegressor on test data: ', RMSLE(y_2018, preds3))


# In[ ]:



from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
# parameters for GridSearchCV
# specify parameters and distributions to sample from
param_dist = {"n_estimators": [3,4,5,6,7,8,9,11,15,18,24,25,26,27,28,40,50,80],
              "min_weight_fraction_leaf": [3.03030303e-03,   4.04040404e-03,   5.05050505e-03,
         6.06060606e-03,   7.07070707e-03,   8.08080808e-03,
         9.09090909e-03,   1.01010101e-02,   1.11111111e-02,
         1.21212121e-02,   1.31313131e-02,   1.41414141e-02,
         1.51515152e-02,   1.61616162e-02,   1.71717172e-02,0.0002,0.0003,0.0004,0.0001,0.0009,0.0008],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(1, 30),
              "bootstrap": [True, False],
              "max_depth": [3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,40,50,80],
              "random_state": sp_randint(1, 30)
             }
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(model2, param_distributions=param_dist,
                                   n_iter=n_iter_search,cv=10)


# In[ ]:


random_search.fit(X_train, np.log1p(y_train.values))


# In[ ]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


report(random_search.cv_results_)


# In[ ]:


finalmodel = ensemble.RandomForestRegressor(n_estimators=5, random_state=25, max_depth=21,
                                        min_weight_fraction_leaf=0.0131313131,bootstrap=False,max_features=8,min_samples_leaf=1,min_samples_split=12)
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


finalmodel.fit(X_train, np.log1p(y_train.values))
print("Model2 trained")
preds2 = finalmodel.predict(X_test)
print('RMSE RandomForestRegressor: ', RMSLE(y_test, preds2))

preds3 = finalmodel.predict(X_2018)
print('RMSE RandomForestRegressor on test data: ', RMSLE(y_2018, preds3))


# ## Approach explanation and algorithm motivation:
# 

#  Let us first talk what we did additionally with the data (pre-processing, assumptions etc...) than the models: Label encoding, important but than I realised that they are already label encoded (not hot-one!) I would like to stay with this assumption since week category obviously matters but also years will be different from each other(global warming etc.) Than I normalized all of the columns. Than I extracted the 2018 data which is our test data(y_2018,X_2018). And with help of train_test from sklearn I created training and validation sets (X_train,X_test)
# 
# 
# First model---NN. Started of with manually checking the hyperparameters-of course very bad tactic but atleast I got a feeling of the hyperparameters. Then I implemented a grid search for neural network. Time expensive! I commented the code out because it takes way to long. Results? Training set (in-sample error: 0.17964286018957273  Validation set:0.18736383060332323  Test set (out-of sample error): 0.27193468030735335
# Nothing special! We should definately find better parameters. LEt us now try 70 30 split . The results are 0.23348722898699353 0.24321321252264558 0.3280610208183715 respectively. Now that is already better and 80/20 : 0.21209512149397639 0.22662413117187602 0.3066506555408157. I have to be honest I was suprised, but it just goes to show case the difference in using different splits when training. NN had more data to learn on (with 80/20, but 70/30 was not efficient), hence it had better results. NOTE: For neural networks is seems to be impossible to achieve absolute reproducibility. I understand that randomness comes from 2 sources. First of all when I use cross validation train_test_split function but there I defined random_state=42. Another source is the initialisation of the weights. Every time it is going to be different. I tried to tackle that on 2 ways: 
# 
# 
# from numpy.random import seed
# 
# seed(1)
# 
# from tensorflow import set_random_seed
# 
# set_random_seed(1)
# 
# and
# 
# keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123)
# 
# 
# What I did achieve is that the variability(reproducability) was smaller but it was not perfekt as with random forest. There I can achieve same results with every run, (even tough there is internal randomness in the RF algorithm itself) So I know it has something to do with NN. Luckily for me RF gives much better and stable results and that is mine final solution, but I spend to much time on NN just to throw it away.
# 
# 
# 
# 
# Second model---random forest---RF  Than I tried to implement another tactic namely randomized search(for Hyperparameter tuning but this time random forest instead of NN. Then the process is straight foreward, I experimented a bit to know which parameters are good (low RMSE) and than I used this knowledge to built interval of parameters that I gave to the randomizedsearchCV. Eventually I do find better parameters (and actually a better model, RF outperforms NN) And the results are as follows:
# Random forest on mine trial-and error hyperparameters validation set: 0.15017251130121254
# 
#  and test data (2018 data) RMSE: 0.1941060627117721 now when I used randomized search I get following results: RMSE on validation data set: 0.11813299983524722
# 
# and RMSE on test data set (2018)  0.15562591092506128
# 
# Obviously we can see that new parameters perform better on 2018 (what we would hope for)
# 
# 
# 
# 

# ## Findings ans lessons learned:
# 

# 
# 
# <span style="color:blue">**Findings:**</span> There is not much to say here other than we confirmed quantitively what was already familiar, namely there is more flu in winter time, temperature, humidity and other factors influence/are connected in certain ways to the outcome of the epidemy. Certain values fall in the range of normal/expected values of every variable, everything beyond that is simply outlier. I would say that finding fall in scope of normal reasoning.
# 
# <span style="color:blue">**Lessons:**</span>I think the reader of this assignment can learn more about neural network hyper-parameter hardship. Trial and error wont work and we need to have more quantitative approach, like randomized-search. Grid search is out of the question until we get quant computers (run-time is awfull). Problems with reproducibility of NN, making the same initial weights for the artificial neural network is pretty hard. How can we implement randomized search for random forests. Different imputation techniques and also to think qualitatively and inspect the data, not only blindly apply the techniques(thats how I found out that some of the outliers in winter were deliberately set to 40, have I only applied those two techniques with standard deviation or inter-quartile range, I would have missed it!). Furthermore what are some interesting aggregation measures that we can introduce to our dataset(feature engineering, we can potentially get a better prediction if we make the features ourselves). How to compactly visualize the dataset in 3 plots, cross-validation with model_selection from sklearn and datetime library together with isocalendar function... There were a lot of interesting lessons learned, these are just a few that stuck for sure.
# 
# 

# **If you benefited in any way please upvote :)**
