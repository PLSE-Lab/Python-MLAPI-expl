#!/usr/bin/env python
# coding: utf-8

# ## This is Part 1 of a project where i attempt to classify race finish status of drivers at each race of a Formula 1 season.
# ### In this notebook, i will create the dataset containing selected features and perform EDA (mainly with the help of Altair package), while in [Part 2](https://www.kaggle.com/coolcat/f1-binary-classification-of-race-finish-status), I will perform the classification.
# 

# #### Import packages and setup

# In[ ]:


from IPython.display import HTML
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from itertools import groupby
import pickle
import os
import sklearn 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer

pd.options.mode.chained_assignment = None 

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', font_scale=1.5)


# In[ ]:


import json  # need it for json.dumps
import altair as alt
from altair.vega import v3
alt.renderers.enable('notebook')

##-----------------------------------------------------------
# This whole section 
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "This code block sets up embedded rendering in HTML output and provides the function `render(chart, id='vega-chart')` for use below. <br/>",
    "Credits to kaggle use @notslush for this plug. <br/>"
    "Check out and vote his/her awesome notebook: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey/notebook"
)))


# ## Import datasets
# 
# #### Where can I find publicly available Formula 1 data? What are the potential features that could help  **identify whether a driver will finish a race or not**? These are the questions I had when i first started out on this crazy project to use machine learning algorithms to predict race finishes.
# #### Initial brainstorming based on my contextual knoweledge of the sport threw up these potential areas to look into:
# - Weather
# - Safety Car appearances
# - Qualifying position
# - Pitstops
# - Tyres
# - Overtaking
#     
# #### 1. I found a [kaggle dataset](https://www.kaggle.com/cjgdev/formula-1-race-data-19502017) that contains information of qualifying position, pit stop timings and  race results. This same data can also be extracted with an API call from the [Ergast Developer API](The Ergast Developer API). 
# 
# #### 2. As for weather and safety car statistics, I pieced together the information myself for each race based on information published on [this blog](http://www.f1strategyreport.com/). This blog is also an excellent resource to gain insight to race strategy!
# 
# #### 3. Pirelli has a[ section on its site](https://racingspot.pirelli.com/global/en-ww/infographics) dedicated to its infographics which details each race's allocated tyre types, number of tyre sets per type each driver chooses to bring along for each race, as well as each driver's stint length (ie. number of laps spent on a tyre type). The infographics are all images, so unfortunately there is manual work involved in recording the information.
# 
# #### 4. Race overtaking information can be gained by have a paid account with [www.cliptheapex.com.](http://www.cliptheapex.com) Because the overtaking information is only available to paid users, I will not share it publicly on kaggle. 
# 
# #### Below, I import all the relevant datasets.

# #### 1) Dataframe of Race finish statuses and position numbers

# In[ ]:


df_results = pd.read_csv('../input/formula-1-race-data-19502017/results.csv')
df_results.head()


# #### 2) Dataframe containing qualifying information of each driver at each race

# In[ ]:


df_qualifying = pd.read_csv('../input/formula-1-race-data-19502017/qualifying.csv')
df_qualifying.head()


# #### 3) Dataframe containing pitstop information at each race

# In[ ]:


df_pitStops = pd.read_csv('../input/formula-1-race-data-19502017/pitStops.csv')
df_pitStops.head()


# #### 4) Dataframe containing weather labels and SC appearance labels (Overtaking figures and SC Laps can be ignored)

# In[ ]:


weather = pd.read_csv('../input/formula-1-race-finish-status/Weather_SafetyCar.csv')
weather.head()


# #### 5) Dataframe containing number an type of tyre sets selected by each driver at each race. 

# In[ ]:


selected_sets = pd.read_csv('../input/formula-1-race-finish-status/Selected_Tyre_Sets.csv')
selected_sets.head()


# #### Import dataframes of drivers and race identification information

# In[ ]:


df_drivers = pd.read_csv('../input/formula-1-race-data-19502017/drivers.csv', encoding ='ISO-8859-1')
df_races = pd.read_csv('../input/formula-1-race-data-19502017/races.csv',  encoding ='ISO-8859-1')

# Cleaning:
# Some drivers have the same surnames, resolve this by replacing driverRef of the non-current driver with the full name
df_results.replace("max_verstappen", "verstappen", inplace=True)
df_results.replace("jolyon_palmer", "palmer", inplace=True)
df_results.replace("kevin_magnussen", "magnussen", inplace=True)
df_results.replace("brandon_hartley", "hartley", inplace=True)

df_drivers.replace("max_verstappen", "verstappen", inplace=True)
df_drivers.replace("jolyon_palmer", "palmer", inplace=True)
df_drivers.replace("kevin_magnussen", "magnussen", inplace=True)
df_drivers.replace("brandon_hartley", "hartley", inplace=True)

df_drivers.loc[75, "driverRef"] = 'jan_magnussen'
df_drivers.loc[49, "driverRef"] = 'jos_verstappen'
df_drivers.loc[155, "driverRef"] = 'jonathan_palmer'
df_drivers.loc[155, "driverRef"] = 'jonathan_palmer'
df_drivers.loc[813, "driverRef"] = 'di resta'

df_races.loc[942, "name"] = "Azerbaijan Grand Prix"


# In[ ]:


df_drivers.head()


# In[ ]:


df_races.head()


# #### DATASET TEMPLATE/FRAME

# In[ ]:


template = selected_sets[['year', 'name', 'driverRef']]


#  ## Pre-process data
#  
# ** Now that we have imported some datasets, let's check which columns can be utilized as features. (The dataframe name is in brackets after the column name)**
# - Ordinal variables:
#     *     position (df_results): Race finish position
#     *     position (df_qualifying):  Qualifying position
# - Continuous variables:
#     *    milliseconds (df_pitStops): Time to complete a pit stop
#     *    Medium, Soft, Super Soft, Ultra Soft, Hard (selected_sets): Number of tyre sets of each type which each driver has confirmed pre-race to be bring along to a race weekend
#     *    SC Laps (weather): Total number of laps led by a Safety Car in a race
# - Categorical variables:
#     *    Weather (weather):  Dry, Varied, Wet
#     *    statusId (df_results): Refer to this [glossary](http://ergast.com/api/f1/status) provided by Ergast API for what each ID represents.
#     
# ** Data Transformation to be done:**
# 1.  All categorical variables  will be dummy coded. A dummy variable represents one level of a categorical variable. 
#     - Weather: Straightforward way of dummy coding. For example, presence of Dry weather is represented by 1 and absence is represented by 0.
#     - statusId: I choose to  re-categorize the IDs (to narrow down the number of categories) before dummy coding: 
#         - 1: Finished
#         - 3: Accident 
#         - 4: Collision (Group ID 3 and 4 together)
#         - 11: +1 Lap (It actually indicates that a driver has a finished the race, albeit one lap slower than the race leader)
#         - 12, 13, 14: +2 Lap, +3 Lap, +4 Lap
#         - Other statuses will be categorized as "Technical Failure"
# 
# **Can we move on to creating a master dataset of these features? Not so fast.... Ground truth values for some features do not exist pre-race. In fact, only qualifying positions and selected tyre sets are known pre-race. How do we know before a race starts that there will be 5 Safety Car laps?  Or that a driver will complete a pit stop at a certain timing? By simply using these values as they are in our training and test sets, we are committing a serious  offence of data leakage.**
#   
#  Let's detour for a bit and **create sub-models**  to derive new feature values:
# 
# **Approach 1:  aggregate historical information (scope: past 3 seasons) to estimate a probabilty of an event occurring. This 'probability' will then be used as a new feature.**
#  - statusId: For eg. if a driver was involved in a race collision for only 1 out of the past 3 Australian Grand Prixs (AUS GP), then 'Accident Probability'  for this driver at the  2018 AUS GP will be 33%. This is repeated for each status category.
#  - weather: For eg. if there was rain for none of the past 3 Australian Grand Prixs, then 'Wet Weather Probability'  for this race in 2018 will be 0%.
#  - SC Laps: Instead of predicting the number of SC laps, I simplify the SC prediction model to just a boolean category of SC or no SC occurring.  For eg. if a SC appeared in all past 3 AUS GP,, then 'SC Probability'  for this race in 2018 will be 100%.
#  - milliseconds: For each race per driver for the past 3 seasons, sum up the pitstop timings and divide by 3.
#  - position (race finish position): Convert the ordinal variables to categories first. The categories are: Podium, 4th to 10th position,  More than 10th position, Did Not Finish. Next, calculate probabilities of occurrence of each category.
#  - *Pros*: This approach helps to overcome the fact that we have incomplete data during data modeling phase by estimating data instead. It is quick to code.
#  - *Cons:*  Potential Information loss because this simple aggregation does not utilize 'year' as an identifier.
#   
# **Approach 2: categorize historical information.**
#  - For eg. if a driver was involved in a race collision in the 2014 AUS GP only,  suffered a technical failure in 2015 AUS GP but completed the 2016 AUS GP, then I will assign a value of 1. 
#  - For each combination of race scenarios, a different category value will be assigned.  
#  - *Pros*: Prevents information loss as 'year' which race event occurred is taken
#  into consideration in the categorization.
#  - *Cons:*, because of the large number of possible combinations, it becomes tedious to assign categorical values and also results in large number of dummy variables. Downstream, there could be many iterations of feature selection required.
# 
# **The below code block is  a class containing functions that help to pre-process data for feature engineeering using Approach 1.**

# In[ ]:


class DataPreprocess():
    
    """
    This class contains functions that help to pre-process data for feature engineeering.
    """
    def __init__(self, results, year_range, calc_method):
        self.results = results
        self.year_range = year_range
        self.calc_method = calc_method

    def remove_outliers(self, df, groupby_fields, agg_method, col, threshold):

        g = df.groupby(groupby_fields)[col].agg(agg_method).reset_index()
        return g[g[col] < g[col].quantile([0, threshold])[threshold]]

    def calc_stats_wrapper(self, function, df, col, groupby_field, agg_method):

        g_all = pd.DataFrame()

        if (self.calc_method == "rolling_value"):

            ranges = [range(self.year_range[idx]-3, self.year_range[idx]) for idx,value in enumerate(self.year_range)]

            for r in ranges:
                g = function(df, r, col, groupby_field, agg_method)
                g['year'] = r[-1] + 1
                g_all = pd.concat([g_all, g])
                
                results = self.results[self.results['year'] == r[-1]+1]
                drivers = list(results.driverRef.unique())
                if groupby_field[0]=='driverRef':
                    g_all = g_all[g_all['driverRef'].isin(drivers)]

            return g_all

        elif (self.calc_method == 'one_year'):

            for r in self.year_range:
                try:
                    g = function(df, [r], col, groupby_field, agg_method)
                    g['year'] = r
                    g_all = pd.concat([g_all, g])
                except:
                    pass
            return g_all

        raise ValueError("Only rolling_value and one_year are available options for calc_method")

        
    def calc_proportion(self, df, yr_range, col, groupby_field, agg_method):
        """
        A multi-purpose function to find proportion of an element amongst a group.
        """  
        
        df = df[df['year'].isin([yr_range[-1]])] 
        g = df.groupby(groupby_field)[col].agg([agg_method]).reset_index()
        
        # Because we are finding the proportion amongst the drivers participating in a season, filter the drivers accordingly.
        results = self.results[self.results['year'] == yr_range[-1]+1]
        drivers = list(results.driverRef.unique())
        df = df[df['driverRef'].isin(drivers)]
        
        if len(groupby_field) > 1:

            df_overall = df.groupby(groupby_field[1:])[col].agg([agg_method]).reset_index()
            df_overall.rename(columns={agg_method: agg_method+' (overall)'}, inplace=True)
            df_new = pd.merge(g, df_overall, on=groupby_field[1:], how='left')

            df_new['proportion'] = (df_new[agg_method] / df_new[agg_method+' (overall)'])
            df_new.drop([agg_method, agg_method +' (overall)'], axis=1, inplace=True)

            return df_new

        elif len(groupby_field) == 1:

            total = float(df[col].agg([agg_method])[agg_method])

            for i, row in g.iterrows():
                g.loc[i, 'proportion'] = float(g.loc[i, agg_method]) / total
            g.drop([agg_method], axis=1, inplace=True)

            return g
        
        
    def calc_avg(self, df, yr_range, col, groupby_field, agg_method):
        """
        Functions to calculate average count of an element within a group over a specified range of years.
        """    
        df = df[df['year'].isin(yr_range)] 
        g = df.groupby(groupby_field)[col].agg([agg_method]).reset_index()
        return g
        
        
    def calc_rate(self, df, yr_range, col, groupby_fields, agg_method):
        """
        Function to calculate percentage/rate of an element occurring over a specified range of years
        """     
        df = df[df['year'].isin(yr_range)] 

        g = pd.DataFrame(df.groupby(groupby_fields)[col].value_counts())
        g.rename(columns={col:agg_method}, inplace=True)
        g = g.reset_index()

        g_overall = pd.DataFrame(df.groupby(groupby_fields)[col].agg(agg_method).rename("total")).reset_index()

        g = pd.merge(g, g_overall, on=groupby_fields, how='left')
        g['percentage'] = (g[agg_method] / g['total']).apply(lambda x: round(x,2))

        gPT = pd.pivot_table(g, index=groupby_fields, columns=[col], values='percentage').reset_index()
        gPT.fillna(0, inplace=True)

        return gPT


# ## Create features
# #### The below code block is a class containing functions enabling feature engineering. As explained above, to derive feature values, we are creating sub-models.

# In[ ]:


class CreateFeatures():
    
    def __init__(self, year_range, calc_method):
        self.calc_method = calc_method
        self.year_range = year_range
        
    def calc_indiv_stats(self, df_qualifying, df_results, weather, df_pitStops, df_races, df_drivers):
        
        # Feature: Qualifying position
        qual = self.preprocess_results(df_qualifying, df_races, df_drivers)
        qual = qual[['year', 'name', 'driverRef', 'position']]
        
        df_results_new = self.preprocess_results(df_results, df_races, df_drivers)
        results = self.categorize_finish_pos_status(df_results_new)
        
        # Initialze class to calculate statistics
        PP = DataPreprocess(results, self.year_range, self.calc_method)
        
        # Feature: Race finish status
        status = PP.calc_stats_wrapper(PP.calc_rate, results, 'statusId', ['driverRef', 'name'], 'count')
        
        # Feature: DNF reason category
        pos = PP.calc_stats_wrapper(PP.calc_rate, results, 'position', ['driverRef', 'name'], 'count')
    
        weather = self.SC_binary_label(weather)
    
        # Feature: Safety car
        sc = PP.calc_stats_wrapper(PP.calc_rate, weather, 'SC', ['name'], 'count')

        # Feature: Wet weather rate
        ww = PP.calc_stats_wrapper(PP.calc_rate, weather, 'weather', ['name'], 'count')

        pitStops = self.preprocess_pitstops(df_pitStops, qual)
        pS_notouliers = PP.remove_outliers(pitStops, ['driverRef', 'name', 'year'], 'sum', "milliseconds", 0.95)
        
        # Feature: Average pitStop timing per driver for the past 3 season
        pS_avg = PP.calc_stats_wrapper(PP.calc_avg, pS_notouliers, 'milliseconds', ['driverRef', 'name'], 'mean')
        pS_avg = pS_avg.rename(columns={'mean': 'pitStop timing (avg)'})
        
        # Feature: Proportion of pitStop timings amongst drivers for the past year.
        pS_d = PP.calc_stats_wrapper(PP.calc_proportion, pS_notouliers, 'milliseconds', ['driverRef'], 'sum')
        pS_d = pS_d.rename(columns={'proportion': 'pitStop timing prop(driver)'})

        # Target Variable: StatusId
        target_var = self.extract_target_variable(results)

        return results, weather, pitStops, qual, status, pos, sc, ww, pS_avg, pS_d, target_var
   

    def preprocess_results(self, data, df_races, df_drivers):
        # Merge reference names to IDs
        results = pd.merge(data, df_drivers[['driverId', 'driverRef']], on=['driverId'], how='left')
        results = pd.merge(results, df_races[['raceId', 'year', 'name']], on=['raceId'], how='left')
        
        return results
    
    def preprocess_qualifying_pos(self, data, df_races, df_drivers):

        qual_pos = data[['raceId', 'driverId', 'position']]
        qual_pos = self.preprocess_results(qual_pos, df_races, df_drivers)
        qual_pos = qual_pos[qual_pos['year'].isin(self.year_range)]
        qual_pos.drop(['raceId', 'driverId'], axis=1, inplace=True)
        
        return qual_pos
    
    def categorize_finish_pos_status(self, data):
    
        # Feature: Finish position
        results = data.copy()
        results['position'] = results['position'].replace(range(1,4) ,"Podium")
        results['position'] = results['position'].replace(range(5,10) , "Pos 4 to 10")
        results['position'] = results['position'].replace(np.nan , "Did not finish")
        mask = ~results['position'].isin(['Podium',"Pos 4 to 10", "Did not finish"])
        results['position'] = results['position'].mask(mask, "Pos > 10")

        # Feature: Reason category for not finishing race
        results['statusId'] = results['statusId'].replace([1,11,12,13,14] ,"Finished")
        results['statusId'] = results['statusId'].replace([3,4] , "Accident / Collision")
        mask = ~results['statusId'].isin(['Finished',"Accident / Collision"])
        results['statusId'] = results['statusId'].mask(mask, "Technical Failure")

        return results  
    
    def SC_binary_label(self, data):
        
        data['SC Laps'].fillna(0, inplace=True)
        data['SC'] = np.where(data['SC Laps'] > 0, "SC", "No SC")
        
        return data
            
    def preprocess_pitstops(self, data, qual):
    
        pitStops = self.preprocess_results(data, df_races, df_drivers)
        g = pd.merge(qual[['year', 'name', 'driverRef']], pitStops, on=['year', 'name', 'driverRef'], how='left').fillna(0)
        g = g.sort_values('stop', ascending=False).groupby(['year', 'name', 'driverRef']).first().reset_index()  

        return g
    
    def extract_target_variable(self, data):
        
        status = data[['year', 'name', 'driverRef', 'statusId']]
        status.replace('Finished', 1, inplace=True)
        status.replace('Accident / Collision', 0, inplace=True)
        status.replace('Technical Failure', 0, inplace=True)

        return status


# In[ ]:


cf = CreateFeatures([2015, 2016, 2017], 'rolling_value')
results, weather, pitStops, qual, status, pos, sc, ww, pS_avg, pS_d, target_var = cf.calc_indiv_stats(df_qualifying, df_results, weather, df_pitStops, df_races, df_drivers)


# ## Create dataset
# #### The below code block contains functions to create a master dataset containing the feature columns (allows flexibility in choosing which features to include in dataset)

# In[ ]:


class CreateDataset():
    def __init__(self, add_qual_pos=True, add_status=True, add_finish_pos=True, add_safety_car=True, add_weather=True, add_pitStop=True, add_tyre_sets=True):

        self.add_qual_pos = add_qual_pos
        self.add_status = add_status
        self.add_finish_pos = add_finish_pos
        self.add_safety_car = add_safety_car
        self.add_weather = add_weather
        self.add_pitStop = add_pitStop
        self.add_tyre_sets = add_tyre_sets

    def merge_all_stats(self, template, qual, status, pos, sc, ww, pS_avg, pS_d, target_var, tyre_sets):

        # Template to merge all feature to
        df = template.copy()
        
        # Merge dataframe containing target variable
        df = pd.merge(df, target_var, on=['year', 'name', 'driverRef'], how='left')
        
        # Feature: Qualifying position
        if self.add_qual_pos==True:   
            df = pd.merge(df, qual, on=['year', 'name', 'driverRef'], how='left')
        
        # Feature: Finishing position category
        if self.add_finish_pos==True:   
            pos = pos.drop(['Pos > 10'], axis=1)
            df = pd.merge(df, pos, on=['year','name', 'driverRef'], how='left')

        # Feature: DNF reason category
        if self.add_status==True:
            status = status.drop(['Technical Failure', 'Finished'], axis=1)
            df = pd.merge(df, status, on=['year','name', 'driverRef'], how='left')
          
        # Feature: Safety Car
        if self.add_safety_car==True:
            sc = sc.drop(['No SC'], axis=1)
            df = pd.merge(df, sc, on=['year','name'], how='left')
           
        # Feature: Wet weather rate
        if self.add_weather==True:
            ww = ww.drop(['Varied'], axis=1)
            df = pd.merge(df, ww, on=['year','name'], how='left') 
            
        # Feature: Pitstop Timings
        if self.add_pitStop==True:
            df = pd.merge(df, pS_avg, on=['year', 'name', 'driverRef'], how='left')  
            df = pd.merge(df, pS_d, on=['year', 'driverRef'], how='left')   
            
        # Feature: Selected Tyre Sets as ordinal categorical vaues
        if self.add_tyre_sets == True:
            df = pd.merge(df, selected_sets, on=['year', 'name', 'driverRef'], how='left') 

        return df
    
    def handling_missing_values(self, df):
        
        # Handle null values for specific columns in a specific way
        df = self.miscellaneous_cleaning(df)
        # Rest of null values belong to new drivers(rookies) and new tracks because they do not have any historical information 
        # Inpute null values with the column's median values
        imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        df_new = pd.DataFrame(imputer.fit_transform(df.drop(['year', 'name', 'driverRef', 'statusId'], axis=1)))
        df_new = pd.concat([df[['year', 'name', 'driverRef', 'statusId']], df_new], axis=1)
        df_new.columns = df.columns 
        
        return df_new

    def miscellaneous_cleaning(self, df):
        
        # Null values belong to drivers who did not set a qual time or participate in qualifying
        if self.add_qual_pos==True: 
            df['position'].fillna(22, inplace=True)

        # Null values belong to drivers who did not set a pitStop time during a race
        if self.add_pitStop==True: 
            df['pitStop timing (avg)'].fillna(0, inplace=True)
            
        return df


# In[ ]:


cd = CreateDataset()
dataset = cd.merge_all_stats(template, qual, status, pos, sc, ww, pS_avg, pS_d, target_var, selected_sets)
dataset.isnull().sum()


# In[ ]:


dataset_new = cd.handling_missing_values(dataset)
dataset_new.isnull().sum()


# In[ ]:


# There are 4 rows with null values for statusId. 
# They belong to hartley, a driver who joined toro rosso only in the lasst 4 races. Seems that the Eargast API results were not updated to reflect the changes.
dataset_new[dataset_new['statusId'].isnull()]


# In[ ]:


# Cleaning: Check the official results and inpute the statusId accordingly.
dataset_new.loc[16, 'statusId'] = 0
dataset_new.loc[32, 'statusId'] = 0
dataset_new.loc[52, 'statusId'] = 1
dataset_new.loc[372, 'statusId'] = 1


# In[ ]:


dataset_new.head()


# In[ ]:


dataset_new.to_csv('dataset.csv', index = False)


# ### Detailed explanation of each  column in dataset
# * ** statusId**: Binary value. 1 -> Driver finished a race. 0-> Driver did not finish a race
# * ** position:** Qualifying position of driver (Depends on the number of drivers participating per season. If a driver did not set a qualifying time, then null value will be imputed with 23.)
# * ** Podium: **Probability of driver finishing race in podium position. (Percentage figure derived by aggregating the occurrences of driver finishing in podium position at the same race over the past 3 seasons.)
# * ** Pos 4 to 10:** Probability of driver finishing in top 4 to 10 position.  (Percentage figure derived by aggregating the occurrences of driver finishing in top 4 to 10 position at the same race over the past 3 seasons.)
# * ** Accident / Collision: **Probability of driver being involved in an accident in a particular race (Percentage figure derived by aggregating the occurrences of driver being involved in an accident at the same race over the past 3 seasons.)
# * ** SC:** Probability of a Safety Car appearing at least once in a race. (Percentage figure derived by aggregating the occurrences of SC appearing >= 1 at the same race over the past 3 seasons.)
# * ** Dry:** Probability of having a race in dry weather conditions.  (Percentage figure derived by aggregating the dry weather occurrences at the same race over the past 3 seasons.)
# * ** Wet: **Probability of having a race in wet weather conditions.  (Percentage figure derived by aggregating the wet weather occurrences at the same race over the past 3 seasons.)
# * **pitStop timing prop(driver):** Average Pit Stop Timing (time to complete a pitstop) for each driver in the past season as a proportion of the total pitstop timings of that season
# * **pitStop timing (avg)**: Average Pit Stop Timing for each driver at each race aggregated over the past 3 seasons.
# * ** Medium: **Number of Medium tyres that a driver brought to a race weekend for the free pracice, qualifying and race sessions.. (Note: This is not the number of tyres which a driver used in the race, nor is it the number of tyres left for the race) 
# 

#  # EDA

# #### Hover over the dashboard to analyse information from individual races/drivers. (To remove the filter, move the cursor to the right edge of the plot, then gradually slide it to the right)
# -  As mentioned above, the logic i put in place to derive new feature values is by aggregating  historical information (scope: past 3 seasons).
# - Hovering over "Australian Grand Prix"' in the races dashboard below, we can see that the "Dry Weather Probability" value for the 2017 Australian Grand Prix is 100% because the track was dry (ie. no rain/wet weather) from 2014 to 2016. From the dashboard, we can also see that the track is prone to having a Safety Car appearance. The "SC Probability" value for 2017 AUS GP is 100% because there has been at least one SC appearance for 2014, 2015 and 2016's races. However, the sub-model's prediction is wrong, as in fact, there was no SC for 2017 AUS GP
# - Similarly, hovering over 'hamilton' in the drivers dashboard below, we can see that the "Accident Probability" value for Hamilton for the 2015 Belgian Grand Prix is 67% because he failed to finish the race in 2012 and 2014. Similarly,  Hamilton has a 0% "Accident Probability" for the 2017 Austrian Grand Prix because he has successfully finished all the races from 2014 to 2016.
#  

# In[ ]:


hover = alt.selection_single(on='mouseover', nearest=True, fields=['name'], empty='all')

base_ww = alt.Chart(ww).properties(
    width=700,
    height=100,
    title="Relationship between probability of Dry weather occurring and number of DNFs"
).add_selection(hover)

points_ww = base_ww.mark_rect().encode(
    x='name',
    y='year:O',
    color=alt.condition(hover, 'Dry:O', alt.value('lightgray'))
).interactive()

bar_ww = alt.Chart(weather).mark_bar().encode(
    x='year:O',
    y='weather',
    color=alt.Color('count()', legend=None)
).transform_filter(
    hover
)

base_sc = alt.Chart(sc).properties(
    width=700,
    height=100,
    title="Probability of Safety Car occurring at a race"
).add_selection(hover)

points_sc = base_sc.mark_circle().encode(
    x='name',
    y='year:O',
    size=alt.Size('SC:O'),
    color=alt.condition(hover, alt.value('black'), alt.value('grey')),
).interactive()

bar_sc = alt.Chart(weather).mark_bar().encode(
    x='year:O',
    y='SC:O',
    color=alt.Color('count()', legend=None)
).transform_filter(
    hover
)

bar_sclaps = alt.Chart(weather).mark_bar(color='black').encode(
    x='year:O',
    y='SC Laps:Q',
).transform_filter(
    hover
)

text = alt.Chart(dataset_new).mark_text(baseline='middle').encode(
    x='name',
    y='year:O',
    text='count()',
    color=alt.condition(alt.datum['SC'] == 1.0, alt.value('white'), alt.value('black'))
).transform_filter(
    alt.datum.statusId == 0
)

render(points_ww + text & bar_ww & points_sc & bar_sc & bar_sclaps)


# In[ ]:


results1317 = results[results['year'] > 2011]

hover = alt.selection_single(on='mouseover', nearest=True, fields=['driverRef'])

base_pos = alt.Chart(pos).properties(
    width=700,
    height=100,
    title="Relationship between probability of driver not finishing a race and actual DNF occurrence"
).add_selection(hover)

points_pos = base_pos.mark_rect().encode(
    x='driverRef',
    y='year:O',
    color=alt.condition(hover, 'Did not finish:O', alt.value('lightgray')),
).interactive()


line_pos = alt.Chart(pos).mark_point().encode(
    x='name',
    y='year:O',
    color=alt.Color('Did not finish:O'),
).transform_filter(
    hover
).properties(
    width=700,
    height=100,
    title="Probability of driver being involved in an accident/collision"
)

bar_pos = alt.Chart(results1317).mark_bar().encode(
    x='year:O',
    y='position',
    color=alt.Color('count()', legend=None)
).transform_filter(
    hover
)

bar_pos1 = alt.Chart(results1317).mark_bar().encode(
    x='name',
    y='year:O',
    color=alt.Color('position', scale=alt.Scale(domain=['Did not finish', 'Pos > 10', 'Pos 4 to 10', 'Podium'], range=['crimson', 'sandybrown', 'lightgreen', 'seagreen']))
).transform_filter(
    hover
)

text = alt.Chart(dataset_new).mark_text(baseline='middle').encode(
    x='driverRef',
    y='year:O',
    text='count()',
    color=alt.condition(alt.datum['SC'] == 1, alt.value('white'), alt.value('black'))
).transform_filter(
    alt.datum.statusId == 0
)

render(points_pos + text &  bar_pos & line_pos & bar_pos1)


# In[ ]:


index_list = ['year', 'name', 'driverRef']
target_var_list = ['statusId']

# List of drivers participating in the respective seasons
drivers16 = results[results['year'] == 2016].driverRef.unique()
drivers17 = results[results['year'] == 2017].driverRef.unique()

# Find the differences in drivers particiapting in 2016 and 2017
drivers_toremove = list(set(drivers16) - set(drivers17)) + list(set(drivers17) - set(drivers16))
drivers_toremove.extend(['hartley', 'vandoorne', 'button'])

# Data transformation: Qualifying position
def scoring(x):
    return x * x.name

qual_crosstab = pd.crosstab(dataset_new['driverRef'], dataset_new['position'])
qual_crosstab = qual_crosstab.apply(lambda x: scoring(x))
qual_crosstab['total'] = qual_crosstab.sum(axis=1)
qual_crosstab.sort_values('total', ascending=True, inplace=True)

# List of best qualifiers
qual_crosstab = qual_crosstab.reset_index()
qual_crosstab = qual_crosstab[~qual_crosstab['driverRef'].isin(drivers_toremove)]
best_qualifiers = list(qual_crosstab.driverRef)

def dnf_acc_plot():
    
    flatui = ["#ffb347", "#659CCA"]
    sns.set_palette(flatui)

    g = sns.factorplot(x="driverRef",
                       y="Did not finish",
                       order=best_qualifiers, # sort x-axis by best qualifying drivers
                       hue="year", 
                       data=dataset_new,
                       kind="violin", 
                       split=True, 
                       scale="count", 
                       inner="stick",
                       cut=0, 
                       bw=.5,
                       size=7, aspect=3)
    
    for ax in g.axes.flat: 
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 

    plt.suptitle('Distribution of probability of drivers not finishing races', y=1.05, fontsize=20)
    plt.title("x-axis is sorted by 2017's best to worst qualifying drivers from left to right", fontsize=16)
    
dnf_acc_plot()


# In[ ]:


pitStops16 = pitStops[(pitStops['year'] == 2016) & ~(pitStops['driverRef'].isin(drivers_toremove))]
pitStops17 = pitStops[(pitStops['year'] == 2017) & ~(pitStops['driverRef'].isin(drivers_toremove))]

heatmap_16 = alt.Chart(pitStops16).mark_rect().encode(
    x='name',
    y='driverRef',
    color=alt.Color('mean(milliseconds):Q', scale=alt.Scale(zero=False, domain=[0, 40000], scheme='redyellowblue'))
).properties(
    width=600,
    height=500
)

heatmap_17 = alt.Chart(pitStops17).mark_rect().encode(
    x='name',
    y='driverRef',
    color=alt.Color('mean(milliseconds):Q', title="Pit Stop Timing (in milliseconds)", scale=alt.Scale(zero=False, domain=[0, 40000], scheme='redyellowblue'))
).properties(
    width=600,
    height=500
)

text_16 = alt.Chart(pitStops16).mark_text(baseline='middle').encode(
    x='name',
    y='driverRef',
    text='stop',
    color=alt.condition((alt.datum['milliseconds'] ==0) and (alt.datum['milliseconds'] >35000), alt.value('white'), alt.value('black'))
)

text_17 = alt.Chart(pitStops17).mark_text(baseline='middle').encode(
    x='name',
    y='driverRef',
    text='stop',
    color=alt.condition((alt.datum['milliseconds'] ==0) and (alt.datum['milliseconds'] >35000), alt.value('white'), alt.value('black'))
)

pitStop_plot = alt.vconcat(
                    heatmap_16 + text_16, 
                    heatmap_17 + text_17,
                    title="Average Pit Stop Timings & Count of Pit Stops per race: 2016-2017"
                )

render(pitStop_plot)


# ## How well do the predictors relate to the target variable?

# In[ ]:


circ = alt.Chart(dataset_new).mark_point().encode(
    x='name',
    y='driverRef',
    size=alt.Size('position', title="Qualifying Position", scale=alt.Scale(range=[0, 500])),
    color=alt.Color('year:N', scale=alt.Scale(range=["#ffb347", "#659CCA"])),
).properties(
    width=600,
    height=600
)

fill_16 = circ.mark_point().encode(
    fill=alt.FillValue("#ffb347"),
    shape=alt.Shape('statusId', legend=alt.Legend(title="DNF"))
).transform_filter(
    (alt.datum.statusId == 0) & (alt.datum.year == 2016)
)

fill_17 = circ.mark_point().encode(
    fill=alt.FillValue("#659CCA"),
    shape=alt.Shape('statusId', legend=alt.Legend(title="DNF"))
).transform_filter(
    (alt.datum.statusId == 0) & (alt.datum.year == 2017)
)

bar_H = alt.Chart(dataset_new).mark_line(opacity=0.6).encode(
    x='name',
    y=alt.Y('count()', title='Number of drivers who Did Not Finish a race'),
    color=alt.Color('year:N', scale=alt.Scale(range=["#ffb347", "#659CCA"])),
).transform_filter(
    (alt.datum.statusId == 0) 
).properties(
    width=600
)

bar_V = alt.Chart(dataset_new).mark_line(opacity=0.6).encode(
    x=alt.X('count()', title='Number of races where driver Did Not Finish'),
    y='driverRef',
    color=alt.Color('year:N', scale=alt.Scale(range=["#ffb347", "#659CCA"])),
).transform_filter(
    (alt.datum.statusId == 0)
).properties(
    height=600
)

VC = alt.vconcat(
    circ+fill_16+fill_17,
    bar_H,
    title="(Filled circles indicate that driver DNF race)"
)

qual_plot = alt.hconcat(
    VC,
    bar_V,
    title="Driver's Qualifying Position vs Race Finish Status: 2016-2017"
)

render(qual_plot)


# In[ ]:


def boxplot(df, index_list, target_var_list):
    
    df_new = df.drop(index_list+target_var_list, axis=1)

    nrows = len(df_new.columns)//3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,20))
    
    my_pal = {0.0: "#FF0000", 1.0: "#88e188"}
    
    i=0
    for row in range(nrows):
        for col in range(ncols):
            g = sns.boxplot(y=df_new.iloc[:,i], x=df[target_var_list[0]], palette=my_pal, ax=axes[row][col])
            i=i+1
    plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
    
boxplot(dataset_new, index_list, target_var_list)


# In[ ]:


def barplots(df):
    
    df1 = df.drop(index_list, axis=1)
    
    def sephist(df, col):
        yes = df[df['statusId'] == 1][col]
        no = df[df['statusId'] == 0][col]
        return yes, no

    plt.figure(figsize=(20,30))
    for num in range(len(df1.columns)-1):
        plt.subplot(len(df1.columns)//2, 2, num+1)
        plt.hist((sephist(df1, df1.iloc[:,num].name)[0], sephist(df1, df1.iloc[:,num].name)[1]), bins=25, alpha=0.5, label=['FIN', 'DNF'], color=["#88e188", "#FF0000"])
        plt.legend(loc='upper right')
        plt.title(df1.iloc[:,num].name)
    plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
    
barplots(dataset_new)


# In[ ]:


scatterplot = alt.Chart(dataset_new).mark_circle(opacity=0.7).encode(
        alt.X(alt.repeat("column"), type='quantitative', scale=alt.Scale(domain=[-0.1, 1.1])),
        alt.Y(alt.repeat("row"), type='quantitative', scale=alt.Scale(domain=[-0.1, 1.1])),
        alt.Color('statusId:N', scale=alt.Scale(range=["#FF0000", "#88e188"])),
        alt.Size('count()', scale=alt.Scale(range=[0, 3000]))
    ).properties(
        width=250,
        height=250,
    ).repeat(
        row=['Did not finish', 'Accident / Collision', 'SC', 'Wet'],
        column=['Wet', 'SC', 'Accident / Collision', 'Did not finish'],
        title="Relationship between probabilites of various race scenarios occuring and driver's actual race finish status: 2016-2017"
    ).interactive()

render(scatterplot)


# In[ ]:


from pandas.plotting import parallel_coordinates

def parallel_coordinates_plot(df):
    
    df_new = df.drop(index_list+target_var_list, axis=1)
    
    SS = StandardScaler()
    Xs_train = pd.DataFrame(SS.fit_transform(df_new))
    Xs_train.columns = df_new.columns
    Xs_train_new = pd.concat([Xs_train.reset_index(drop=True), df[target_var_list].reset_index(drop=True)], axis=1)
    
    plt.figure(figsize=(15,10))
    parallel_coordinates(Xs_train_new, "statusId", color=["#FF0000", "#88e188"])
    plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Features values', fontsize=15)
    plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
    plt.xticks(rotation=90)
    plt.show()
    
parallel_coordinates_plot(dataset_new)


# In[ ]:


from pandas.plotting import radviz

def classification_radviz(df):
    
    df_new = df.drop(index_list+target_var_list, axis=1)
    
    SS = StandardScaler()
    Xs_train = pd.DataFrame(SS.fit_transform(df_new))
    Xs_train.columns = df_new.columns
    Xs_train_new = pd.concat([Xs_train.reset_index(drop=True), df[target_var_list].reset_index(drop=True)], axis=1)
    
    plt.figure(figsize=(10,10))
    radviz(Xs_train_new, 'statusId', color=["#FF0000", "#88e188"])
    plt.show()
    
classification_radviz(dataset_new)


# ### From the graphs above, there is no clear linear relationship between features and target variable, meaning that none of the features except for qualifying position give a good indication whether or not a driver will finish a race or not.  I expect that classification of race finish status will not be an easy task. Since we have non-linearly separable data, perhaps trees-based classifiers and neural networks will perform best with this dataset. Head over to [Part 2 ](https://www.kaggle.com/coolcat/f1-binary-classification-of-race-finish-status) to find out.

# In[ ]:





# In[ ]:





# In[ ]:




