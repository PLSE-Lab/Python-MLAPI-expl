#!/usr/bin/env python
# coding: utf-8

# 
# With this exploration I intend to give answers on most interesting questions for traffic crashes. I hope the results are ultimately impactful and representative. What kind of questions? The analysis would start from a few general yet exciting areas and then aspire to very specific like "which car type is the safest". By the end I would present most prominent findings alltogether.
# 
# I've started this project a few days ago, and got literally carried away by the ideas and approaches to implement, so it turned out to be quite big. All beause I wanted _"to get to the bottom of it"_.
# 
# For most I recommend jump straight to the [Findings](#Findings) section: it presents all the interesting results.
# 
# Here is the table of contents:
# - <a href='#us_fatality_rates'>US Traffic Fatality Rates since 1966</a>
# - [Female Drivers](#Female-Drivers)
# - [US Map Projection](#US-Map-Projection)
# - [Reasons behind Fatal Crashes](#Reasons-behind-Fatal-Crashes)
#     - [Gathering data](#Gathering-data)
#     - [Prepare the Data](#Prepare-the-Data)
#     - [Train the models](#Train-the-models)
# - [Sanity check](#Sanity-check)
# * - [Findings](#Findings) - Jump to results!
# - [Conclusion](#Conclusion)
# 
# Follow with me, I'm sure to learn quite a number of interesting things from the dataset!

# ### Imports

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from matplotlib import gridspec
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
import numpy as np

# import package with helper functions 
import bq_helper
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# <a id='us_fatality_rates'></a>
# ### US Traffic Fatality Rates since 1966
# Alas, kaggle datasets have stats only for 2015 and 2016 year. But this question is way too interesting, and it'd be great to see the wider picture, so let's address original source: [nhtsa website for the data](https://cdan.nhtsa.gov/STSI.htm).
# 
# _Beforehand, I have an assumption that, obviously, the rate would grow greatly since 1966, because of rapid population and drivers increase._

# In[ ]:


us_fatality_trend_df = pd.read_excel('../input/US_traffic_indidents_trend_1966-2016.xlsx', skip_footer=(77), header=([6,7]))


# In[ ]:


def fatality_trend_since_60x():
    fig = plt.figure(figsize=(12,6))
        # Add subplots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    us_fatality_trend_df['Fatality Rate per 100,000','Licensed Drivers'].plot(ax=ax1, title='Traffic Fatalities per 100k drivers\nper year');
    plt.ylabel('Fatalities')
    us_fatality_trend_df['Fatalities','Unnamed: 0_level_1'].plot(ax=ax2, title='Fatalities total\nper year');
    fig.tight_layout()


# Wow! This is absolutely impressive picture! Traffic Fatality Rate in US decreased dramatically, not only per every 100k drivers, but an absolute number as well (it is all with ever growing population and traffic)!
# 
# Here is a guess: maybe it is just accidents became less deadlier? Let's see:

# In[ ]:


us_accidents_trend_df = pd.read_excel('../input/US_traffic_indidents_trend_1966-2016.xlsx', skiprows=86, skipfooter=18, header=[1,2,3])


# In[ ]:


fig = plt.figure(figsize=(12,6))
# Add subplots
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
us_accidents_trend_df['Injury Rate per','100,000 Licensed'].plot(ax=ax1, title='Traffic Injuries per 100k drivers\nper year');
plt.ylabel('Injuries')
us_accidents_trend_df['Injured','Unnamed: 0_level_1'].plot(ax=ax2, title='Injuries total\nper year');
plt.ylabel('Injuries')
fig.tight_layout()


# That's a sound answer: injury trend decreased greatly since last 40 years, as well as fatality rate on the road. Driving became much safer.

# #### Female Drivers
# How do women perform behind the steering wheel?

# In[ ]:


us_accidents_sex_df = pd.read_excel('../input/us_traffic_fatalities_per_sex.xlsx', skiprows=6, skipfooter=116, header=[1,2,3,4])
us_accidents_sex_df.head(3)


# In[ ]:


us_accidents_sex_2016_df = us_accidents_sex_df.loc[2016,[('Male (> 15 Years Old)','Involvement Rate','per 100K Licensed','Drivers'),('Female (> 15 Years Old)','Involvement Rate','per 100K Licensed','Drivers')]]
us_accidents_sex_2016_df


# In[ ]:


def barplot_sex():
    # Define Variables
    male_fem_in_fatalities = us_accidents_sex_2016_df
    ind = np.arange(2)
    width = 0.7

    # Plot graph
    fig, ax = plt.subplots(figsize=(10,6))
    bars_sex_drivers = ax.bar(ind, male_fem_in_fatalities, width, linewidth = 0.3, edgecolor='black', alpha = 0.8, label='Driver Sex involvement\nper 100k drivers');

    ax.set_title('Male vs Female Drivers fatal traffic crashes\nper 100k drivers', alpha=0.8, fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(['Male Drivers', 'Female Drivers'])
    ax.set_ylim(0,40)
    ax.set_ylabel('per 100k drivers', alpha = 0.7)
    ax.axhline(11.8, lw=0.6, linestyle='--')


# In[ ]:


barplot_sex()


# ### US Map Projection
# Traffic Fatalities 2016 - choropleth map projection per state. Two steps to build such projections:
# 1. Query accident_2016 table to count accidents by state
# 2. Plot results on a map!

# In[ ]:


QUERY = """SELECT
  state_number,
  state_name,
  COUNT(consecutive_number) AS accidents,
  SUM(number_of_fatalities) AS fatalities,
  SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
GROUP BY
  state_number, state_name
ORDER BY
  state_number """

# Estimate data size to be scanned with the query:
accidents.estimate_query_size(QUERY.format(2016))


# In[ ]:


accidents_state_df = accidents.query_to_pandas(QUERY.format(2016))
accidents_state_df.head(3)


# Using [plotly for map projection](https://plot.ly/python/choropleth-maps/).
# 
# Plotly uses state code abbreviations for choropleth map, therefore we have to bind each statename to it's respective code and then pass it to plotly.

# In[ ]:


# US codes dictionary
us_state_abbrev = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Dist of Columbia': 'DC'}
# Adding codes column to the dataframe:
accidents_state_df['state_code'] = accidents_state_df['state_name'].map(us_state_abbrev)
accidents_state_df.head(3)


# In[ ]:


choropleth_map_title = 'Traffic Fatalities by State - 2016<br>(hover for breakdown)'
def plot_choropleth_df(locations_series, data_series, text_series):
    scl_blue_brwn = [[0.0, 'rgb(171,217,233)'], [0.16666666666666666, 'rgb(224,243,248)'],
                     [0.3333333333333333, 'rgb(254,224,144)'], [0.5, 'rgb(253,174,97)'],
                     [0.6666666666666666, 'rgb(244,109,67)'], [0.8333333333333334, 'rgb(215,48,39)'],
     [1.0, 'rgb(165,0,38)']]


    data = [ dict(
            type='choropleth',
            colorscale = scl_blue_brwn,
            autocolorscale = False,
            locations = locations_series,
            z = data_series,
            locationmode = 'USA-states',
            text = text_series,
            marker = dict(
                line = dict(
                    color = 'rgb(255, 255, 255)',
                    width = 2)
                ),
            colorbar = dict(
                title = "Accidents")
            ) ]

    layout = dict(
            title = choropleth_map_title,
            geo = dict(
                 scope = 'usa',
                 projection = dict(type = 'albers usa'),
                 countrycolor = 'rgb(255, 255, 255)',
                 showlakes = True,
                 lakecolor = 'rgb(255, 255, 255)')
             )

    fig = dict(data=data, layout=layout)
    iplot(fig)


# In[ ]:


locations_series = accidents_state_df['state_code']
data_series = accidents_state_df['fatalities']
text_series = accidents_state_df['state_name'] + '<br>' + 'Accidents: ' + accidents_state_df['accidents'].astype(str) + '<br>' + 'Fatalities: '+ accidents_state_df['fatalities'].astype(str)

plot_choropleth_df(locations_series, data_series, text_series)


# California, Texas and Florida are blushing in red, having by far highest absolute number of fatalities for 2016.
# 
# Again, to have a correct picture we should consider number of drivers in each state. Hence, I'm addressing [external nhtsa data source](https://cdan.nhtsa.gov/STSI.htm) for number of licensed drivers per state.

# In[ ]:


drivers_per_state_2016_df = pd.read_excel('../input/Drivers_and_fatalities_per_state_2016.xlsx', skip_footer=(12), header=([6,7]), index=False)

# Here is mumbo-jumbo with indecies, because I want to map some values to accidents_state_df dataframe,
# while this dataframe has 'Dist of Columbia' spelled differently.
drivers_per_state_2016_df.reset_index(inplace=True, drop=False)
drivers_per_state_2016_df['state_code'] = drivers_per_state_2016_df['index'].map(us_state_abbrev)
drivers_per_state_2016_df.set_index('state_code', inplace=True)
drivers_per_state_2016_df = drivers_per_state_2016_df.iloc[:51]

# Now onto mapping - i.e. concatenating values: num of licensed drivers and fatalities per 100k drivers.
accidents_state_df['licensed_drivers_k'] = accidents_state_df['state_code'].map(drivers_per_state_2016_df['Licensed Drivers','(Thousands)'])
accidents_state_df['fatalities_per_100k_drivers'] = accidents_state_df['state_code'].map(drivers_per_state_2016_df['Fatalities per 100,000','Drivers'])

accidents_state_df.head(3)


# In[ ]:


locations_series = accidents_state_df['state_code']
data_series_per_drivers = accidents_state_df['fatalities_per_100k_drivers']
text_series_per_drivers = accidents_state_df['state_name'] + '<br>' +     'Drivers (thousands): ' + accidents_state_df['licensed_drivers_k'].astype(str) + '<br>' +     'Fatalities per 100k drivers: ' + accidents_state_df['fatalities_per_100k_drivers'].astype(str)


choropleth_map_title = 'Traffic Fatalities per 100k Drivers by State - 2016<br>(hover for breakdown)'

plot_choropleth_df(locations_series, data_series_per_drivers, text_series_per_drivers)


# Now that is realistic picture!
# 
# Turns out Missisipi, Kentucky, Oklahoma, Wyoming and New Mexico have **most fatal traffic**.
# 
# While Washington, New York, Massachusetts, Rhode Island and a tiny District of Columbia are among **most traffic-safest places in US**.

# Ok, we now know something about US traffic. Here comes the next most interesting question: WHY? Can we only using this dataset find out the reasons, why is Missisipi so dangerous, while Washington or New York seem to be a traffic heaven? If we CAN, that actually might have a real impact (no pun intended... with word "impact"), and potentially save lives.
# 
# Just scanning through all the 40 tables with numerous column data, we can see there are dozens if not hundred various factors to contribute to Fatality Rate. Here are some of them: car manufacturer, type of brakes, type of road intersection, drunk driver, race of people involved in a crash, weather conditions etc. - and each of those categories are broken down in quite a number of subcategories.
# 
# So... which factors are the most important in traffic fatality rate? A good approach to solve this would be to gather all the data together grouped by state, and calculate the correlation - build a correlation matrix for all the given features! That way we would instantly see most salient ones.
# 
# Let's do it then.

# ## Reasons behind Fatal Crashes
# Is it a wet road, a white drunkard behind the wheel, maybe it is someone drives a honda civic or... is it just type of brakes that influence it most... or maybe it is something entirely else? I spend some time thinking the way to solve it. It soon becomes obvious that doing it manually can be a waste of effort! There are hundreds of variables across 40 tables in the traffic dataset we have, so... it is a task for a machine! Yes, Machine Learning tools should shine here. The idea is to feed all the data into the math model and let it come back with the answers: which columns have largest "weight"! Wow, that's actually exciting, can't wait to see what we get!
# 
# Dive in!

# Here are the steps to go throgh (1 and 2nd steps would take most of the effort):
# 1. Gather the data
#     1. Define all features (columns) across all tables
#     2. Build a query/queries and gather all data into 1 dataframe.
# 2. Prepare the data
#     1. [One-hot-encode data](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f).
#     2. Group all the data by state, counting presence of each categorical feature per state (we do that, because we have traffic fatalities per state as our main measure).
# 3. Train the model
# 4. Get the features that most/least contribute to the fatality rate by getting model weights!
# 1. 5. Prove those features manually and visually by comparing data for states with top/least fatality rates plotting the graphs.

# ### Gathering data
# 
# Now getting acquinted with the tables. Here is a complete set of columns I find worthwhile:
# 
# General info on the accident, accident_{year} table: route_signingname,  type_ofintersection,  workzone, light_conditionname, atmospheric_conditions_1name, hour, week, month, number_of_drunkdrivers
# 
# See by state number of persons not in a vehicle involved, nmcrash_{year} table: number_of_persons_not_in_motor_vehicles_in_transportmvitof
# 
# For events involving non-motorists, find out contribution of non-vehicle person actions to the event, distract_{year} table: non_motorist_contributing_circumstances_name
# 
# See the distraction inputs recorded prior to the event, distract{year} table: driver_distracted_byname of distract{year} 
# 
# By driver impairment, using drimpair_{year} table: condition_impairment_at_time_of_crash_driver, condition_impairment_at_time_of_crash_drivername
# 
# See contribution of possible defects and maintenance conditions of the vehicles, using factor_{year} table: contributing_circumstances_motor_vehicle_name
# 
# maneuver_{year} table
# 
# By what was on the road, that driver tried to avoid prior the event, using maneuver_{year} table: driver_maneuvered_to_avoid_name
# 
# 
# For events involving non-motorists, find out contribution of non-vehicle person actions to the event, using nmcrash_{year} table: non_motorist_contributing_circumstances_name
# 
# Any physical impairments of non motorists, using nmimpair_{year} table: condition_impairment_at_time_of_crash_non_motorist_name
# 
# Parked and working vehicles involved, using parkwork_{year} table: unit_type column
# Possibly explore models of the vehicles, if the previous fraction large enough.
# 
# pbtype_{year} table
# 
# By a pedestrian/byciclist type, using pbtype_{year} table.
# person_{year} table
# 
# Slice by age, sex, race involvement into the event, using person_{year} table: age, sex, race, alcohol, drugs
# 
# See fraction of misuse of the restraint system: indication_of_misuse_of_restraint_system_helmet
# 
# safetyeq_{year} table
# 
# For non-motorists involved into the event, slice by the safetyequipment used, via safetyeq_{year} table: non_motorist_safety_equipment_use
# 
# Info on the vehicle involved: vehicle_make_name, body_type_name, vehicle_model_year, speeding_related, travel_speed, previous_other_moving_violation_convictions, trafficway_description, roadway_surface_type, driver_drinking (coincides with 21)
# vindecode_{year} table
# 
# by veicle type, using vindecode_{year} table: vehicle_type + Slices by the type of systems used in the vehicles.
# vision_{year} table
# 
# Slice by vision impediments: drivers_vision_obscured_by

# Now, query the data and collect to nice dataframes:

# In[ ]:


dataframe_list = []


# In[ ]:


q_accidets = '''
SELECT
  state_number,
  hour_of_crash,
  day_of_crash,
  month_of_crash,
  day_of_week,
  functional_system_name,
  land_use_name,
  route_signing_name,
  type_of_intersection,
  work_zone,
  light_condition_name,
  atmospheric_conditions_name,
  school_bus_related,
  number_of_drunk_drivers  
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
ORDER BY
  state_number
'''
accidents_general_df = accidents.query_to_pandas(q_accidets.format(2016))
accidents_general_df.head(3)


# Categorizing drunk drivers to just drunk, if there is at least one

# In[ ]:


def categorize_drunks(value):
    if int(value) >=1:
        return 'one_or_more'

accidents_general_df['number_of_drunk_drivers'] = accidents_general_df['number_of_drunk_drivers'].apply(categorize_drunks)
dataframe_list.append(accidents_general_df)
accidents_general_df.head(3)


# In[ ]:


q_driver_disraction = '''
SELECT
  state_number,
  driver_distracted_by_name 
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.distract_{0}`
ORDER BY
  state_number
'''
driver_distraction_df = accidents.query_to_pandas(q_driver_disraction.format(2016))
dataframe_list.append(driver_distraction_df)
driver_distraction_df.head(3)


# In[ ]:


q_driver_impairement = '''
SELECT
  state_number,
  condition_impairment_at_time_of_crash_driver_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.drimpair_{0}`
ORDER BY
  state_number
'''
driver_impair_df = accidents.query_to_pandas(q_driver_impairement.format(2016))
dataframe_list.append(driver_impair_df)
driver_impair_df.head(3)


# In[ ]:


q_possible_factor = '''
SELECT
  state_number,
  contributing_circumstances_motor_vehicle_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.factor_{0}`
ORDER BY
  state_number
'''
factor_df = accidents.query_to_pandas(q_possible_factor.format(2016))
dataframe_list.append(factor_df)
factor_df.head(3)


# In[ ]:


q_nonmotorist_contribution = '''
SELECT
  state_number,
  non_motorist_contributing_circumstances_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.nmcrash_{0}`
ORDER BY
  state_number
'''
nonmotorist_contribution_df = accidents.query_to_pandas(q_nonmotorist_contribution.format(2016))
dataframe_list.append(nonmotorist_contribution_df)
nonmotorist_contribution_df.head(3)


# In[ ]:


q_nonmotorist_impair = '''
SELECT
  state_number,
  condition_impairment_at_time_of_crash_non_motorist_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.nmimpair_{0}`
ORDER BY
  state_number
'''
nonmotorist_impair_df = accidents.query_to_pandas(q_nonmotorist_impair.format(2016))
dataframe_list.append(nonmotorist_impair_df)
nonmotorist_impair_df.head(3)


# In[ ]:


q_nonmotorist_prior = '''
SELECT
  state_number,
  non_motorist_action_circumstances_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.nmprior_{0}`
ORDER BY
  state_number
'''
nonmotorist_prior_df = accidents.query_to_pandas(q_nonmotorist_prior.format(2016))
dataframe_list.append(nonmotorist_prior_df)
nonmotorist_prior_df.head(3)


# In[ ]:


q_parkwork = '''
SELECT
  state_number,
  body_type AS parkwork_body_type_name,
  vehicle_model_year AS parkwork_vehicle_model_year
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.parkwork_{0}`
ORDER BY
  state_number
'''
q_parkwork_df = accidents.query_to_pandas(q_parkwork.format(2016))
q_parkwork_df.head(3)


# We can transform vehicle_model_year to categories, something like: to1990, 1991to2000, 2001to2010, after2010

# In[ ]:


def categorize(value):
    if 1900<int(value) <= 1990:
        return 'to1990'
    elif int(value) <= 2000:
        return '1991to2000'
    elif int(value) <= 2010:
        return '2001to2010'
    elif 2018 >int(value) > 2010:
        return '2011newer'

q_parkwork_df['parkwork_vehicle_model_year'] = q_parkwork_df['parkwork_vehicle_model_year'].apply(categorize)
dataframe_list.append(q_parkwork_df)
q_parkwork_df.head(3)


# In[ ]:


q_pedes_byci_type = '''
SELECT
  state_number,
  crash_group_pedestrian_name,
  crash_group_bicycle_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.pbtype_{0}`
ORDER BY
  state_number
'''
pedes_byci_type_df = accidents.query_to_pandas(q_pedes_byci_type.format(2016))
dataframe_list.append(pedes_byci_type_df)
pedes_byci_type_df.head(3)


# In[ ]:


q_person_info = '''
SELECT
  state_number,
  person_type_name,
  age,
  injury_severity_name,
  restraint_system_helmet_use,
  air_bag_deployed,
  police_reported_alcohol_involvement,
  police_reported_drug_involvement,
  hispanic_origin_name,
  race_name
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.person_{0}`
ORDER BY
  state_number
'''
person_info_df = accidents.query_to_pandas(q_person_info.format(2016))
person_info_df.head(3)


# Let's categorize by age groups.

# In[ ]:


def categorize_age(value):
    if 1<int(value) <= 8:
        return '1to10'
    elif 8<int(value) <= 16:
        return '8to16'
    elif 16<int(value) <= 25:
        return '16to25'
    elif 25 < int(value) <= 34:
        return '25to34'
    elif 34 < int(value) <= 55:
        return '34to55'
    elif 55 < int(value) <= 70:
        return '55to70'
    elif 70 < int(value) <= 95:
        return '70to95'
    elif 95 < int(value) <= 150:
        return 'above95'
    
person_info_df['age'] = person_info_df['age'].apply(categorize_age)
dataframe_list.append(person_info_df)
person_info_df.head(3)


# In[ ]:


q_nonmotorist_safetyeq = '''
SELECT
  state_number,
  non_motorist_safety_equipment_use
  
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.safetyeq_{0}`
ORDER BY
  state_number
'''
nonmotorist_safetyeq_df = accidents.query_to_pandas(q_nonmotorist_safetyeq.format(2016))
dataframe_list.append(nonmotorist_safetyeq_df)
nonmotorist_safetyeq_df.head(3)


# In[ ]:


q_vehicle = '''
SELECT
  state_number,
  vehicle_make_name,
  body_type_name,
  vehicle_model_year,
  travel_speed,
  compliance_with_license_restrictions,
  previous_recorded_suspensions_and_revocations,
  previous_speeding_convictions,
  speeding_related,
  trafficway_description,
  speed_limit,
  roadway_alignment,
  roadway_grade,
  roadway_surface_type,
  roadway_surface_condition
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}`
ORDER BY
  state_number
'''
vehicle_df = accidents.query_to_pandas(q_vehicle.format(2016))
vehicle_df.head(3)


# We can use some categorization of model_year and travel_speed.

# In[ ]:


def categorize_year(value):
    if 1900<int(value) <= 1990:
        return 'to1990'
    elif int(value) <= 2000:
        return '1991to2000'
    elif int(value) <= 2010:
        return '2001to2010'
    elif 2018 >int(value) > 2010:
        return '2011newer'
    
def categorize_speed(value):
    if 5<int(value) <= 20:
        return '5to20'
    elif 20<int(value) <= 40:
        return '20to40'
    elif 40<int(value) <= 60:
        return '40to60'
    elif 60 < int(value) <= 80:
        return '60to80'
    elif 80 < int(value) <= 100:
        return '80to100'
    elif 100 < int(value) <= 200:
        return '100and_more'

vehicle_df['vehicle_model_year'] = vehicle_df['vehicle_model_year'].apply(categorize_year)
vehicle_df['travel_speed'] = vehicle_df['travel_speed'].apply(categorize_speed)
dataframe_list.append(vehicle_df)
vehicle_df.head(3)


# In[ ]:


q_violation = '''
SELECT
  state_number,
  violations_charged_name
  
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.violatn_{0}`
ORDER BY
  state_number
'''
violation_df = accidents.query_to_pandas(q_violation.format(2016))
dataframe_list.append(violation_df)
violation_df.head(3)


# In[ ]:


q_vision = '''
SELECT
  state_number,
  drivers_vision_obscured_by_name
  
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vision_{0}`
ORDER BY
  state_number
'''
vision_df = accidents.query_to_pandas(q_vision.format(2016))
dataframe_list.append(vision_df)
vision_df.head(3)


# In[ ]:


print('dataframes collected ',len(dataframe_list))


# That was quite a data scavenge! All tables were thoroughly scanned, all the data collected into the dataframes.
# 
# Now, comes hardest part: Preprocess it into a nice form, suitable for a machine!

# ### Prepare the Data
# So, I've crafted the functions to do all the job.
# 
# The functions take dataframes as inputs, and build one long list of features, and one hot encoded array with values actually grouped and averaged by state.

# Function `hot_encode_df_features(dataframe)` simply takes 1 single dataframe and hot-encodes everything, also ensuring to construct a list of featurenames diligently. A small twist there: it checks if the feature(column) is binary, then it runs `LabelBinarizer` on it, rather than `OneHotEncoder`.

# In[ ]:


def hot_encode_df_features(dataframe):
    colnames = dataframe.columns.tolist()

    le = LabelEncoder()
    lb = LabelBinarizer()
    enc = OneHotEncoder()

    feature_list = []
    feature_array = np.zeros((len(dataframe),1))
    for colname in colnames:
        _ = le.fit(dataframe[colname].astype(str));

        if len(le.classes_)==2: # Checks if it is a binary feature, like "yes" "no", and encodes it as one binary feature
            dataframe[colname] = le.fit_transform(dataframe[colname].astype(str))
            feature_list = feature_list+([colname+'__'+str(le.classes_[1])])
            sparse_array = lb.fit_transform(dataframe[colname].astype(str))
        else: # Else label encodes it each with a separate feature
            dataframe[colname] = le.transform(dataframe[colname].astype(str));
            feature_list = feature_list+[colname+'__'+str(feature_name) for feature_name in list(le.classes_)]
            sparse_array = enc.fit_transform(dataframe[[colname]].astype(str)).toarray()

        # Stack the array to a feature_array
        feature_array = np.hstack((feature_array,sparse_array))
    del dataframe
    feature_array = feature_array[:,1:]
    
    return feature_list, feature_array


# Function `group_onehot_mean_by_column_matrix(df, colname_to_groupby, groupnames_list)` takes a dataframe, a column name to group by, and also a list of groups to go through. It calls previous function to get hot-encoded list, then calculates mean across the groups.

# In[ ]:


def group_onehot_mean_by_column_matrix(df, colname_to_groupby, groupnames_list):
    dataframe = df.copy()
    groupname_series = dataframe[colname_to_groupby]
    dataframe.drop(colname_to_groupby, axis=1, inplace=True)
    feature_list, hot_enc_array = hot_encode_df_features(dataframe)
    
    # Initialize dataset array, with rows=groupnames_list, and columns of feature list
    grouped_mean_array = np.zeros((len(groupnames_list),len(feature_list)))

    for i, groupname in enumerate(groupnames_list):
        idx = groupname_series.index[groupname_series==groupname]
        if idx.any():
            hot_mean_per_group_arr = np.mean(hot_enc_array[idx], axis=0)
            grouped_mean_array[i] = hot_mean_per_group_arr

    return feature_list, grouped_mean_array


# Function `get_combined_features_and_dataset(list_of_dataframes,colname_to_groupby,groupnames_list)` runs previously defined functions for a list of dataframes, and returns a combo feature list and dataset.

# In[ ]:


def get_combined_features_and_dataset(list_of_dataframes,colname_to_groupby,groupnames_list):
    combined_feature_list = []
    combined_grouped_hot_mean_array = np.zeros((len(groupnames_list),1))

    for dataframe in list_of_dataframes:
        feature_list, grouped_hot_mean_array = group_onehot_mean_by_column_matrix(dataframe, colname_to_groupby, groupnames_list)
        combined_feature_list = combined_feature_list+feature_list
        combined_grouped_hot_mean_array = np.hstack((combined_grouped_hot_mean_array,grouped_hot_mean_array))
        
    combined_grouped_hot_mean_array = combined_grouped_hot_mean_array[:,1:]

    return combined_feature_list,combined_grouped_hot_mean_array


# #### Run data preprocessing!

# In[ ]:


get_ipython().run_cell_magic('time', '', "print('dataframes ', len(dataframe_list))\ncolname_to_groupby = 'state_number'\ngroupnames_list = sorted(set(dataframe_list[0][colname_to_groupby]))\n\nfeature_list_combo, hot_mean_gr_arr_combo = get_combined_features_and_dataset(dataframe_list,colname_to_groupby,groupnames_list)")


# In[ ]:


print('Number of features: ',len(feature_list_combo))
print('Dataset shape: ',hot_mean_gr_arr_combo.shape)


# ### Train the models
# We have our data ready and preprocessed: it is 803 features! The data is grouped in 51 samples - each sample is a separate state, and we know fatality value each state. That's all is needed for a good modelling.
# 
# There is a number of good estimators for a regression task such as we have. So, the next step would be to choose the best one, fully tuned up.
# 
# I'll define hundreds of models and use sklearn GridSearchCV for training and estimation. Let's see how it goes!

# In[ ]:


# Define models (estimators and their params):
list_of_models = [
    {
        'name':'LinearRegression',
        'estimator':LinearRegression(),
        'hyperparameters':
        {
        }
    },
    {
        'name':'ElasticNet',
        'estimator':ElasticNet(),
        'hyperparameters':
        {
            "alpha": [0.1,0.3,0.5,0.8,1,1.5,3,5,10,15,20],
            "l1_ratio": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        }
    },
    {
        'name':'Ridge',
        'estimator':Ridge(),
        'hyperparameters':
        {
            'solver':['saga', 'lsqr'],
            "alpha": [0.1,0.3,0.5,0.8,1,1.5,2,3],
            'max_iter':[50,200]
        }
    },
    {
        'name':'SVR',
        'estimator':svm.SVR(),
        'hyperparameters':
        {
            "C": [0.1,0.3,0.5,0.8,1,3,5,10],
            'max_iter':[50,200,400]
        }
    }
]


# In[ ]:


# Define a function that performs Grid Search
def select_model(X, y, list_of_models):
    for model in list_of_models:
#             cv = ShuffleSplit(n_splits=50, test_size=0.05)
            print('\nSearching best params for {}...'.format(model['name']))
            estimator = model['estimator']
#             if model['name'] in ['MLPRegressor','GradientBoostingRegressor']:
#                 cv_num=30 # to run heavy models less times
            grid = GridSearchCV(estimator, param_grid=model['hyperparameters'], cv=30, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X, y)

            model['best_hyperparameters'] = grid.best_params_
            best_score = grid.best_score_
            model['best_score'] = best_score
            model['best_estimator'] = grid.best_estimator_
            print('best mse score for this model',best_score)
    #         print('best params for this model',best_params_)

    list_of_models_sorted_best = sorted(list_of_models, key=lambda x: x['best_score'], reverse=True)
    best_model = list_of_models_sorted_best[0]['name']
    best_score_achieved = list_of_models_sorted_best[0]['best_score']
    best_params = list_of_models_sorted_best[0]['best_hyperparameters']
    print('Best Model: {}, score: {}'.format(best_model,best_score_achieved))
    print('\nBest parameters: {}'.format(best_params))


    return list_of_models_sorted_best


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = hot_mean_gr_arr_combo\ny_train = accidents_state_df['fatalities_per_100k_drivers']\nbest_models = select_model(X_train, y_train, list_of_models)")


# The best model is determined, now on to getting feature weights.

# In[ ]:


model_ridge = Ridge(alpha=0.1, max_iter=200, solver='saga')
model_ridge.fit(X_train, y_train)
feature_weights_ridge = model_ridge.coef_


# Ok, great, we got best model. It can now show us what the leading fatality factors!

# In[ ]:


def show_important_features(feature_list, feature_weights, num_to_show=10):
    sorted_coef_index = feature_weights.argsort()
    largest_coeff_features_idx = sorted_coef_index[::-1][:num_to_show]
    print('Features of highest contribution to fatal accidents:')
    for idx in largest_coeff_features_idx:
        print(feature_list[idx], feature_weights[idx])

    smallest_coeff_features_idx = sorted_coef_index[:num_to_show]
    print('\nFeatures of least contribution to fatal accidents - that contribute to Safety:')
    for idx in smallest_coeff_features_idx:
        print(feature_list[idx], feature_weights[idx])


# In[ ]:


show_important_features(feature_list_combo, feature_weights_ridge, num_to_show=20)


# Wow that's interesting!
# 
# The ridge model claims that **most dangerous are**:
#     - Most fatal traffic accidents happen in places with speed limit 45 and 70, and yet they are not speed related! Seems speedlimit signs are there for a reason.
#     - The most dangerous roads are two-way and divided by a painted line!
#     - The most dangerous are non-intersections and it is usually Highway
#     - Most accidents happen during clear athmospheric conditions!
#     - And hey, it clearly shows that Jaywalking (pedestrian crossing road in improper place) contributes to Fatalities greatly.
#     - A large contribution is absence of AirBag.
#     - somehow Pickups are most dangerous of all types of cars.
#     
# factors that do not matter a lot, or we can call them **safest factors**:
#     - It is Urban area, local and municipal roads specifically are most safe
#     - Roads with speed limit 30 and 50
#     - Four-way intersections are quite safe
#     - Sedan/Hardtops cars are safe
#     - Drunk drivers do not matter for traffic fatalities! So, the model claims that of all reasons 'drunk drivers' is the least one. And that makes sense, you can imagine, that there is a fraction of all accidents, where a drunkard was spotted, but probably a much larger fraction of drunkards are driving the roads that have not crashed! That's what the model says.
#     - It is safe, when it's dark and streets are lighted

# 
# Now I would like to double check with another model - **ensemble of forests**.
# 
# Those who familiar with that type of model, know, that  those trees grow quite randomly, most of them too large and crooked. Therefore I will grow lots of trees, like a lot! And pick only the best and nice looking (with high predicting power). Then model weights are collected, and averaged.
# 
# The function below implements custom cross-fold validation, that  extracts feature weights only from the best models:

# In[ ]:


def get_gbr_feature_weights(X, y, folds):
    print('Growing forests...')
    feature_importances_arr = np.zeros((len(feature_list_combo)))
    y_pred_gbr = np.zeros((len(X_train)))
    mse = 0
    good_model_count = 0
    for iter in range(folds):
        X_tr, X_dev, y_tr, y_dev = train_test_split(X, y, test_size=0.05)
        model_gbr = GradientBoostingRegressor(n_estimators=2000,max_depth=10, learning_rate=0.05)
        model_gbr.fit(X_tr,y_tr)
        y_pred_gbr = model_gbr.predict(X_dev)
        this_mse = mean_squared_error(y_dev,y_pred_gbr)
        if this_mse < 20:
            good_model_count+=1
            mse += this_mse
            feature_importances_arr += model_gbr.feature_importances_
#             if iter==1:
#                 print(feature_importances_arr)

    print('Num of models with good mse: ',good_model_count)
    mse /= good_model_count
    print('Model mse: ',mse)
    feature_importances_arr /= good_model_count
    
    return feature_importances_arr


# Overall that would be a million of trees to grow! And would take about 10 minutes.

# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = hot_mean_gr_arr_combo\ny_train = accidents_state_df['fatalities_per_100k_drivers']\nfolds = 500\n\nfeature_importance_gbr = get_gbr_feature_weights(X_train, y_train, folds)")


# Forest looks promising, the model is noticebly better! Let's listen to those tree whispers:

# In[ ]:


show_important_features(feature_list_combo, feature_importance_gbr, 15)


# The list of most dangerous factors includes:
#     - Pickup car
#     - Freightliner trucks (a manufacturer name)
#     - emotional condition of a pedestrian
#     - roads with speedlimit of 25 and 70
#     - persons of age 70 to 90 involved in the accident!
#     - dark and not lighted conditions outside
#     - seems that among all persons, passengers suffer most
#     - somehow people of American Indian race (Includes Aleuts and Eskimos) are among most dangerous people on the road.
#     
# The safest options:
#     - cars by Daihatsu manufacturer - those cute japanese cars appear to be the safest.
#     - low speed and neighborhood electric vehicles are among most innocent factors
#     - leg-impaired drivers contribute least to the accidents
#     - walking wrong side of the road for a pedestrian does not matter much as well.

# ### Sanity check
# So far we have some sound claims from two models. It makes sense to check some of them manually and visualize it.
# 
# It would be vivid to compare Missisipi and NewYork city on Alcoholic drivers and, say, Pickup cars!
# 
# According to claims, we should expect no correlation of drunkards to the levels of fatality, in fact a safe NewYork would have even more drunkards. As for Pickup cars, we can presume that Missisipi is flooded with them, i.e. high correlation to the traffic fatality rate.

# In[ ]:


q_accidets_drunk = '''
SELECT
  state_number,
  count(number_of_drunk_drivers) AS drunk_drivers_count
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`
WHERE number_of_drunk_drivers > 0
GROUP BY state_number
ORDER BY
  state_number
'''
accidets_drunk_df = accidents.query_to_pandas(q_accidets_drunk.format(2016))
accidets_drunk_df.head(3)


# In[ ]:


q_vehicle_pickup = '''
SELECT
  state_number,
  COUNT(body_type) AS pickups_count
FROM
  `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_{0}`
WHERE body_type IN (30,31)
GROUP BY state_number
ORDER BY
  state_number
'''
vehicle_pickup_df = accidents.query_to_pandas(q_vehicle_pickup.format(2016))
vehicle_pickup_df.head(3)


# Stack it with the initial dataframe and get count per 100 fatalities per state

# In[ ]:


accidents_state_df['drunk_drivers_count'] = accidets_drunk_df['drunk_drivers_count'] / accidents_state_df['fatalities']*100
accidents_state_df['pickups_count'] = vehicle_pickup_df['pickups_count'] / accidents_state_df['fatalities']*100
ny_missi_accidents_df = accidents_state_df[accidents_state_df['state_name'].isin(['New York','Mississippi'])]
ny_missi_accidents_df


# In[ ]:


def barplot_drunkards():
    # Define Variables
    fatalities_per_100k = ny_missi_accidents_df['fatalities_per_100k_drivers']
    drunk_drivers = ny_missi_accidents_df['drunk_drivers_count']
    ind = np.arange(2)
    width = 0.7

    # Plot graph
    gs = gridspec.GridSpec(1, 3)
    fig= plt.figure(figsize=(10,6));
    ax1 = fig.add_subplot(gs[:,:-1])
    ax2 = fig.add_subplot(gs[:,-1:])
    bars_drunk_drivers = ax1.bar(ind, drunk_drivers, width, linewidth = 0.3, color=sns.xkcd_rgb["pale red"], edgecolor='black', alpha = 0.8, label='Test');
    bars_tot_fatal = ax2.bar(ind, fatalities_per_100k, width, linewidth = 0.3, edgecolor='black', alpha = 0.8, label='Fatality Rate');

    ax1.set_title('Drunks per 100 accidents\nMissisipi and NY', alpha=0.7, fontsize=14)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Missisipi drunks', 'New York drunks'])
    ax1.set_ylim(0,35)
    ax1.set_ylabel('Drunkards per 100 fatalities', alpha = 0.7)
    ax1.axhline(15, lw=0.6, linestyle='--')

    ax2.set_title('Total Fatalities\nper 100k drivers', alpha=0.7, fontsize=14)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['Missisipi', 'New York']);


# In[ ]:


def barplot_pickups():
    # Define Variables
    fatalities_per_100k = ny_missi_accidents_df['fatalities_per_100k_drivers']
    drunk_drivers = ny_missi_accidents_df['pickups_count']
    ind = np.arange(2)
    width = 0.7

    # Plot graph
    gs = gridspec.GridSpec(1, 3)
    fig= plt.figure(figsize=(10,8));
    ax1 = fig.add_subplot(gs[:,:-1])
    ax2 = fig.add_subplot(gs[:,-1:])
    bars_drunk_drivers = ax1.bar(ind, drunk_drivers, width, linewidth = 0.3, color=sns.xkcd_rgb["pale red"], edgecolor='black', alpha = 0.8, label='Test');
    bars_tot_fatal = ax2.bar(ind, fatalities_per_100k, width, linewidth = 0.3, edgecolor='black', alpha = 0.8, label='Fatality Rate');

    ax1.set_title('Pickups per 100 accidents\nMissisipi and NY', alpha=0.7, fontsize=14)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Missisipi Pickup Cars', 'New York Pickup Cars'])
    ax1.set_ylim(0,35)
    ax1.set_ylabel('Pickups per 100 fatalities', alpha = 0.7)
    ax1.axhline(12.1, lw=0.6, linestyle='--')

    ax2.set_title('Total Fatalities\nper 100k drivers', alpha=0.7, fontsize=14)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(['Missisipi', 'New York']);


# In[ ]:


barplot_drunkards()


# It is evident: alcohol is not the reason for states have higher Traffic Fatality Rate.

# In[ ]:


barplot_pickups()


# Perfect correlation!
# 
# Are Pickup cars is the reason why some states have more Fatalities? - The statistics says a sound YES. It is one of the reasons, why some states have higher fatality rates. Look: every 3-d fatality involves a Pickup car in Missisipi.
# 
# Yes, sanity tells that probably there are some other underlying reasons, not the Pickup itself. But rather that Rural people tend to buy Pickups, and we could see that Rural areas are most dangerous.

# ## Findings
# ------
# All the findigns I've found during the exploration are presented here.

# ### Traffic Fatality trend since 60x

# In[ ]:


fatality_trend_since_60x()


# Big success: huge growth of cars and drivers, yet fatality rate decreased greatly!

# ### Female Drivers
# Are women 'ok' to drive?

# In[ ]:


barplot_sex()


# Women are much better drivers in terms of safety!

# ### Fatality Rate by state
# Map out fatality levels.

# In[ ]:


plot_choropleth_df(locations_series, data_series_per_drivers, text_series_per_drivers)


# Quite a vivid picture.
# 
# Most dangerous States:
# - Missisipi, Kentucky, Oklahoma, Wyoming and New Mexico have most fatal traffic.
# 
# Safest States:
#  - Washington, New York, Massachusetts, Rhode Island and a tiny District of Columbia are among most traffic-safest places in US.

# ### Why some States are more fatal than others?
# What are the reasons that Missisipi has fare more fatalities than Washington or New York? The answer could give insights on a bigger picture, provide understanding how efforts could be focused, and hopefully (ultimately) save lives.

# A combined dataset of 800 columns, representing almost all tables, was fed into regression models.  Machine learning tools allowed to determine factors that contribute most to the Fatality Rates by State. As well as factors that are "safest", and have no contribution to the deaths in each State.
# 
# Here is output example of only largest/smallest features:

# In[ ]:


show_important_features(feature_list_combo, feature_weights_ridge, num_to_show=5)


# Turns out that most dangerous roads are
# - HighWays; also two-way roads divided only by a painted line; mostly non-intesection roads. 
# - accidents happen at clear atmospheric conditions
# - "Jaywalkers" cause quite a lot of crashes
# - Pickup cars contribute a lot to the fatalities
# - Most crashes are not speed-related.
# - car AirBags is a big deal, abscense of them cause certain death during crash.
# - Freightliner trucks (a manufacturer name) cause some trouble; 
# - dark and not lighted conditions outside
# - seems that among all persons, passengers suffer most
# - somehow people of American Indian race (Includes Aleuts and Eskimos) are among most dangerous people on the road.
#     
# The safest factors:
# - Urban areas, local and municipal roads;
# - four-way intersections are rather safe;
# - cars of Sedan/Hardtops type.
# - Daihatsu cars - cute type by japanese manufacturer.
# - low speed and neighborhood electric vehicles
# - it is safe when dark and streets are lighted.
# - leg-impaired drivers contribute least to the accidents
# - leg-impaired drivers contribute least to the accidents
# - (!)Interestingly, drunk drivers were determined as a safe option by the models! To be precise: drunk drivers do not cause some State to be more dangerous than the other. I.e. Missisipi is super traffic-fatal is not because of drunk drivers.

# What actions could be taken:
# - Looks like goverment might focus effort on two-way roads adding a wider intersection, focus on rural-area roads, AirBags regulations, pay closer attention on Pickup cars, Freightliner trucks.

# #### Do drunk drivers make any difference?
# So, Machine Learning approach defined drunk drivers make no difference for State Fatality Rate. Is it really so?

# In[ ]:


barplot_drunkards()


# As you can see, Missisipi has far more car fatalities, than NY, yet NY has higher drunkards concentration.
# Even though alcohol does not explain Fatality variations across State, still it is a large cause of car crash itself.
# 
# About 20% of all Fatal car Crashesh involve drunk driver.

# ## Conclusion
# Thanks for your attention! The exploration turned out to be quite big. I learned a lot along the way! Hope you too as well. [Findings](Findings) section presents all the interesting results. However, the code is all there, implementing BigQuery, pandas, statistics and Machine Learning techniques, and hope you can gain insights from it.
# 
# Actually this kaggle dataset is very specific, as it includes stats for fatal crashes only, so it is almost impossible to get more precise picture, find true causes, and make inferences. Yet there are numerous angles to explore!
# 
# With all that, I'm happy to complete this project (took me a few days) and would be glad to discuss your thoughts, inputs or questions. Feel free to chime in!
