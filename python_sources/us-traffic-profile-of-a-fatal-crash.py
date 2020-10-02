#!/usr/bin/env python
# coding: utf-8

# # Data Exploration at the National Level
# For the first step of this analysis, we decided to explore the data by ranking the states in function of the fatalities per thousands inhabitant. We crossed reference the reported fatalities from the US Traffric Fatality Records with the state population from the [2016 census data](https://www.census.gov/data/tables/2017/demo/popest/state-total.html).

# In[ ]:


import bq_helper as bqh #Import BigQueryHelper
import pandas as pd #Import Pandas
import matplotlib.pyplot as plt #Import Matplotlib
import numpy as np #Import numpy

us_traffic_fat = bqh.BigQueryHelper(active_project = 'bigquery-public-data',
                                   dataset_name = 'nhtsa_traffic_fatalities')

query = """
        SELECT
            state_name,
            CASE
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 0 THEN '12AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 1 THEN '1AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 2 THEN '2AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 3 THEN '3AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 4 THEN '4AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 5 THEN '5AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 6 THEN '6AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 7 THEN '7AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 8 THEN '8AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 9 THEN '9AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 10 THEN '10AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 11 THEN '11AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 12 THEN '12PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 13 THEN '1PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 14 THEN '2PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 15 THEN '3PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 16 THEN '4PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 17 THEN '5PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 18 THEN '6PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 19 THEN '7PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 20 THEN '8PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 21 THEN '9PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 22 THEN '10PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 23 THEN '11PM'
                END Hour_Of_Day,
            CASE
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 1 THEN 'Sunday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 2 THEN 'Monday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 3 THEN 'Tuesday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 4 THEN 'Wednesday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 5 THEN 'Thursday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 6 THEN 'Friday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 7 THEN 'Saturday'
            END Day_Of_Week,
            CASE
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 1 THEN 'January'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 2 THEN 'February'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 3 THEN 'March'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 4 THEN 'April'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 5 THEN 'May'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 6 THEN 'June'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 7 THEN 'July'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 8 THEN 'August'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 9 THEN 'September'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 10 THEN 'October'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 11 THEN 'November'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 12 THEN 'December'
            END Month,
            functional_system_name AS Trafficway_Type,
            type_of_intersection AS Intersection,
            light_condition_name AS Light_Condition,
            atmospheric_conditions_1_name AS Atmospheric_conditions,
            COUNT(DISTINCT consecutive_number) AS num_fatalities
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY 1,2,3,4,5,6,7,8
        ORDER BY state_name
        """

#Put query in a DF, join it with 2016_state_population file and add calculated field
df = us_traffic_fat.query_to_pandas(query)
df_state_population = pd.read_csv('../input/2016_states_population.csv')


# ## Fatalities Per States
# Some Explations

# In[ ]:


df_fat_per_states = df.loc[:,['state_name','num_fatalities']]
df_fat_per_states = df_fat_per_states.groupby(['state_name'])['num_fatalities'].sum().reset_index()

df_join = pd.merge(df_state_population, df_fat_per_states, how='inner', on='state_name')
df_join['crashes_per_thousand'] = df_join['num_fatalities'] / df_join['2016_population'] * 1000 #Calculate fatal carsh per 1000 inhabitants
df_join = df_join.sort_values(by=['crashes_per_thousand'], ascending=False) #Sort data by 'crashes_per_thousands'

#Define figure size, add label to x axis and plot graph

state_name_list = pd.Series.tolist(df_join['state_name'])
y_pos = np.arange(len(state_name_list))
plt.figure(figsize=(14,5))
plt.title('Fatalities Per Thousand Inhabitants Per State')
plt.ylabel('Fatalities Per Thousand')
plt.xlabel('States')
plt.xticks(y_pos, state_name_list, rotation='vertical')
plt.bar(y_pos, df_join['crashes_per_thousand'], align='center', alpha=0.5)
plt.show()


# ## Fatalities Per Hour Of the Day 
# Some explaination

# In[ ]:


total_fat = df['num_fatalities'].sum()
df_fat_per_hour = df.loc[:,['Hour_Of_Day','num_fatalities']]
df_fat_per_hour = df_fat_per_hour.groupby(['Hour_Of_Day'])['num_fatalities'].sum().reset_index()
df_fat_per_hour['fat_per_h_perc'] = df_fat_per_hour['num_fatalities'] / total_fat * 100

df_fat_per_hour = df_fat_per_hour.sort_values(by=['fat_per_h_perc'],ascending=False)

hod_list = pd.Series.tolist(df_fat_per_hour['Hour_Of_Day'])
hod_list_x_label = pd.Series.tolist(df_fat_per_hour['Hour_Of_Day'])
y_pos = np.arange(len(hod_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, hod_list_x_label)
plt.xlabel('Hour Of Day')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Hour (National)')
plt.bar(y_pos, df_fat_per_hour['fat_per_h_perc'],align='center', alpha=0.5)
plt.show()


# ## Fatalities Per Day Of Week
# Some Explainations

# In[ ]:


total_fat = df['num_fatalities'].sum()
df_fat_per_dow = df.loc[:,['Day_Of_Week','num_fatalities']]
df_fat_per_dow = df_fat_per_dow.groupby(['Day_Of_Week'])['num_fatalities'].sum().reset_index()
df_fat_per_dow['fat_per_dow_perc'] = df_fat_per_dow['num_fatalities'] / total_fat * 100

df_fat_per_dow = df_fat_per_dow.sort_values(by=['fat_per_dow_perc'], ascending=False)

dow_list = pd.Series.tolist(df_fat_per_dow['Day_Of_Week'])
dow_list_x_label = pd.Series.tolist(df_fat_per_dow['Day_Of_Week'])
y_pos = np.arange(len(dow_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, dow_list_x_label)
plt.xlabel('Day Of Week')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Day Of Week (National)')
plt.bar(y_pos, df_fat_per_dow['fat_per_dow_perc'],align='center', alpha=0.5)
plt.show()


# ## Fatalities Per Month
# Some Explainatio

# In[ ]:


total_fat = df['num_fatalities'].sum()
df_fat_per_moy = df.loc[:,['Month','num_fatalities']]
df_fat_per_moy = df_fat_per_moy.groupby(['Month'])['num_fatalities'].sum().reset_index()
df_fat_per_moy['fat_per_moy_perc'] = df_fat_per_moy['num_fatalities'] / total_fat * 100

df_fat_per_moy = df_fat_per_moy.sort_values(by=['fat_per_moy_perc'], ascending=False)

moy_list = pd.Series.tolist(df_fat_per_moy['Month'])
moy_list_x_label = pd.Series.tolist(df_fat_per_moy['Month'])
y_pos = np.arange(len(moy_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, moy_list_x_label)
plt.xlabel('Month Of Year')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Month (National)')
plt.bar(y_pos, df_fat_per_moy['fat_per_moy_perc'],align='center', alpha=0.5)
plt.show()


# # Fatalities Per Atmospheric Condition
# Some explaination

# In[ ]:


total_fat = df['num_fatalities'].sum()
df_fat_atcon = df.loc[:,['Atmospheric_conditions','num_fatalities']]
df_fat_atcon = df_fat_atcon.groupby(['Atmospheric_conditions'])['num_fatalities'].sum().reset_index()
df_fat_atcon['fat_per_atcon_perc'] = df_fat_atcon['num_fatalities'] / total_fat * 100

df_fat_atcon = df_fat_atcon.sort_values(by=['fat_per_atcon_perc'], ascending=False)

atcon_list = pd.Series.tolist(df_fat_atcon['Atmospheric_conditions'])
atcon_list_x_label = pd.Series.tolist(df_fat_atcon['Atmospheric_conditions'])
size = pd.Series.tolist(df_fat_atcon['fat_per_atcon_perc'])
y_pos = np.arange(len(atcon_list))
plt.figure(figsize=(8,8))
plt.xticks(y_pos, atcon_list_x_label, rotation='vertical')
plt.title('Fatalities Per Atmospheric Condition')
plt.pie(df_fat_atcon['fat_per_atcon_perc'])
plt.legend(['%s, %1.1f %%' % (l, s) for l, s in zip(atcon_list_x_label,size)], loc='best')
plt.show()


# ## Atmospheric Condition Exploration
# The results shown in the pie chart above, may seem counter intuitive. Indeed, we wold expect most of the fatal crashes to be happening under difficult weather conditions, though, 71.5% of the fatal accidents happened under clear conditions.
# 
# One route we would like to explore is the weight of states, in the total distribution, with abnormaly high average sun light per year - this may explain the results above

# In[3]:


import bq_helper as bqh
import pandas as pd

us_traffic_fat_per_state = bqh.BigQueryHelper(active_project = 'bigquery-public-data',
                                            dataset_name = 'nhtsa_traffic_fatalities')

query = """
        SELECT
            state_name,
            COUNT(DISTINCT consecutive_number) AS fatalities
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY
            1
        ORDER BY
            fatalities DESC
        """

df_clear_days_2016 = pd.read_csv('../input/2016_clear_days.csv')
print(df_clear_days_2016)
state_fatalities = us_traffic_fat_per_state.query_to_pandas(query)


# In[ ]:




