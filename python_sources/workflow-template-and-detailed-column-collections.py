#!/usr/bin/env python
# coding: utf-8

# Original notebook with better formatting can be downloaded [here](https://github.com/frankherfert/Kaggle/blob/master/Sberbank_Workflow_Column-Collections.ipynb)
# 
# On your local jupyter notebook server, this will get a structure like this:
# 
# ![title](https://www.dropbox.com/s/vnj8mx1d3vovbhc/ToC.png?dl=1)

# # Workflow template
# This notebook is a template for a more structured workflow.
# 
# The sberbank dataset contains a lot of variables (291 in the regular train- and test-files, 100 in the macro-file) and working through all of those can become rather tedious even on a 1080+ screen. Newly created columns are added to the end of the dataframe and comparing new and old columns can be made easier with topic-related column-collections.

# ## Column collections
# 
# Under the feature-section of this notebook you can find all variables that belong together, grouped into arrays.
# 
# Most of the default variables are positioned next to each other in the source-dataframe but some are not sorted in an ideal way (e.g. the square-meters columns are not next to each other, kitch_sq is a few columns to the right, km-distance-columns are all over the place).
# 
# With the column-collections you can use **`train_df.loc[:3,sub_area_columns]`** where **`sub_area_columns`** is the array containing all relevant columns regarding areas. 
# 
# After you add new features, just add them to the array and display all topic-related columns with the same line.
# 

# 

# 

# #### Collapsible Headers
# To make the whole workflow easier this notebook is seperated by header and sub-headers for each topic. With the help of the "collapsible headers extension", working with them becomes much easier.
# 
# This extension allows you to collapse all headers in a hierarchical way.
# ![title](https://www.dropbox.com/s/0zzmi5ttk5ic39p/collapsible_headers.png?dl=1)

# 

# 

# 

# 

# 

# 

# #### Imports

# In[ ]:


import numpy as np
np.set_printoptions(linewidth=140) # numpy displays 75 lines by default which is not optimal on a larger screen.
import pandas as pd
pd.set_option('display.max_columns', 300) # increase number of columns and rows to print out
pd.set_option('display.max_rows', 300)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime

# the following 3 lines let your notebook take up more space on the screen / removes the unused width on the side
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:99% !important; }</style>"))
display(HTML("<style>table {float:left}</style>")) # makes the changelog table nicer


# In[ ]:


from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics


# 

# In[ ]:


def show_dtypes(df):
    for dtype in df.dtypes.unique():
        print(str(dtype).ljust(14), ":",list(df.dtypes[df.dtypes == dtype].index),"\n")

# returns all columns of a DataFrame with type float64, int64, uint8
def get_numeric_columns(df):
    return list(df.dtypes[(df.dtypes == "float64") | (df.dtypes == "int64") | (df.dtypes == "uint8")].index)


# creates dummy columns with prefix, merge with source dataframe
def create_dummy_columns(df, columns):
    for column in columns:
        df[column]    = df[column].apply(lambda x: str(x)) #convert to str just in case
        new_columns = [column + "_" + i for i in full[column].unique()] #only use the columns that appear in the test set and add prefix like in get_dummies
        df = pd.concat((df,    pd.get_dummies(df[column],    prefix = column)[new_columns]), axis = 1)
    return df


def del_columns(df, columns):
    for column in columns:
        if column in df.columns:
            del df[column]
            print("Deleted: ", column)
        else:
            print("Not in DataFrame: ",column)
    return df


# # Loading Data

# 

# In[ ]:


macro_df = pd.read_csv("../input/macro.csv", parse_dates=["timestamp"])
macro_df.shape


# 

# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=["timestamp"])
train_df["train_test"] = "train"

train_df = pd.merge(train_df, macro_df, how="left", on="timestamp")

print("Shape: ", train_df.shape,"\n")
train_df.info(memory_usage="deep")


# 

# In[ ]:


test_df = pd.read_csv("../input/test.csv", parse_dates=["timestamp"])
test_df["train_test"] = "test"

test_df = pd.merge(test_df, macro_df, how="left", on="timestamp")

print("Shape: ",test_df.shape, "\n")
test_df.info(memory_usage="deep")


# 

# In[ ]:


train_df["price_doc"].describe().apply(lambda x: '%.f' % x)


# In[ ]:


train_df.price_doc.sort_values(ascending=True).head(5)


# In[ ]:


train_df.price_doc.sort_values(ascending=False).head(5)


# In[ ]:


ulimit = np.percentile(train_df.price_doc.values, 99.9)
llimit = np.percentile(train_df.price_doc.values, .1)
train_df.loc[train_df['price_doc']>ulimit, ["price_doc"]] = ulimit
train_df.loc[train_df['price_doc']<llimit, ["price_doc"]] = llimit


# In[ ]:


train_df["price_doc"].describe().apply(lambda x: '%.f' % x)


# In[ ]:


train_y = np.array(train_df["price_doc"])
print(train_y[0:10])

# Since our metric is "RMSLE", let us use log of the target variable for model building rather than using the actual target variable.
train_y = np.log1p(train_y)
print(train_y[0:10])


# # Exploratory Data Analysis

# In[ ]:


# put in your code here


# # Features

# In[ ]:


# show_dtypes(train_df)


# 

# In[ ]:


# price_doc: sale price (this is the target variable)
# id: transaction id
# timestamp: date of transaction
# full_sq: total area in square meters, including loggias, balconies and other non-residential areas
# life_sq: living area in square meters, excluding loggias, balconies and other non-residential areas
# floor: for apartments, floor of the building
# max_floor: number of floors in the building
# material: wall material
# build_year: year built
# num_room: number of living rooms
# kitch_sq: kitchen area
# state: apartment condition
# product_type: owner-occupier purchase or investment
# sub_area: name of the district

# The dataset also includes a collection of features about each property's surrounding neighbourhood, and some features that are constant across each sub area (known as a Raion). Most of the feature names are self explanatory, with the following notes.

# full_all: subarea population
# male_f, female_f: subarea population by gender
# young_*: population younger than working age
# work_*: working-age population
# ekder_*: retirement-age population
# n_m_{all|male|female}: population between n and m years old
# build_count_*: buildings in the subarea by construction type or year
# x_count_500: the number of x within 500m of the property
# x_part_500: the share of x within 500m of the property
# _sqm_: square meters
# cafe_count_d_price_p: number of cafes within d meters of the property that have an average bill under p RUB
# trc_: shopping malls
# prom_: industrial zones
# green_: green zones
# metro_: subway
# _avto_: distances by car
# mkad_: Moscow Circle Auto Road
# ttk_: Third Transport Ring
# sadovoe_: Garden Ring
# bulvar_ring_: Boulevard Ring
# kremlin_: City center
# zd_vokzaly_: Train station
# oil_chemistry_: Dirty industry
# ts_: Power plant


# 

# In[ ]:


train_df["null_count"] = train_df.isnull().sum(axis=1)
test_df["null_count"] = test_df.isnull().sum(axis=1)


# In[ ]:


# these make it easier to see if a property has an unusually high or low price if you are not familiar with Rubles
train_df["price_doc_euro"] = np.round(train_df["price_doc"]*0.016,0)
train_df["price_doc_dollars"] = np.round(train_df["price_doc"]*0.02,0)


# 

# 

# In[ ]:


time_columns = ["timestamp"]
train_df.loc[:3,time_columns]


# In[ ]:


train_df["time_year"] = train_df["timestamp"].dt.year
test_df["time_year"] = test_df["timestamp"].dt.year

train_df["time_month_of_year"] = train_df["timestamp"].dt.month
test_df["time_month_of_year"] = test_df["timestamp"].dt.month

train_df["time_yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month-200000
test_df["time_yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month-200000

train_df["time_week_of_year"] = train_df["timestamp"].dt.weekofyear
test_df["time_week_of_year"] = test_df["timestamp"].dt.weekofyear

train_df["time_yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear-200000
test_df["time_yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear-200000

train_df["time_day_of_year"] = train_df["timestamp"].dt.dayofyear
test_df["time_day_of_year"] = test_df["timestamp"].dt.dayofyear

train_df["time_day_of_month"] = train_df["timestamp"].dt.day
test_df["time_day_of_month"] = test_df["timestamp"].dt.day

train_df["time_day_of_week"] = train_df["timestamp"].dt.weekday
test_df["time_day_of_week"] = test_df["timestamp"].dt.weekday


# In[ ]:


time_columns = ["timestamp", "time_year", "time_month_of_year", "time_yearmonth", "time_week_of_year", "time_yearweek",
                "time_day_of_year", "time_day_of_month", "time_day_of_week"]
train_df.loc[:3,time_columns]


# 

# In[ ]:


sqm_floors_columns = ['full_sq', 'life_sq', 'kitch_sq', 'floor', 'max_floor', 'num_room']


# In[ ]:


train_df.loc[10005:10008,sqm_floors_columns]


# 

# In[ ]:


material_columns = ["material"]
train_df.loc[:3,material_columns]


# 

# In[ ]:


build_year_columns = ["build_year"]
train_df.loc[:3,build_year_columns]


# 

# In[ ]:


state_columns = ["state"]
train_df.loc[:3,state_columns]


# 

# In[ ]:


product_type_columns = ["product_type"]
train_df.loc[:3,product_type_columns]


# 

# ### sub_area : indust_part

# In[ ]:


sub_area_columns = ['sub_area', 'area_m', 'raion_popul', 'green_zone_part', 'indust_part']
train_df.loc[:3,sub_area_columns]


# 

# In[ ]:


education_columns = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion', 'children_school', 'school_quota', 'school_education_centers_raion', 
                     'school_education_centers_top_20_raion', 'university_top_20_raion', 'additional_education_raion',]
train_df.loc[:3,education_columns]


# 

# In[ ]:


healthcare_columns = ['hospital_beds_raion', 'healthcare_centers_raion']
train_df.loc[:3,healthcare_columns]


# 

# In[ ]:


culture_columns = ['sport_objects_raion', 'culture_objects_top_25', 'culture_objects_top_25_raion', 'shopping_centers_raion', 'office_raion']
train_df.loc[:3,culture_columns]


# 

# In[ ]:


industry_columns = ['thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 
                    'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion']

train_df.loc[:3,industry_columns]


# 

# 

# In[ ]:


ppl_count_all = ['full_all', 'male_f', 'female_f', 'young_all', 'young_male', 'young_female', 'work_all', 'work_male', 'work_female',
                 'ekder_all', 'ekder_male', 'ekder_female']
train_df.loc[:3,ppl_count_all]


# 

# In[ ]:


ppl_count_0_6 = ['0_6_all', '0_6_male', '0_6_female']
train_df.loc[:2,ppl_count_0_6]


# 

# In[ ]:


ppl_count_7_14 = ['7_14_all', '7_14_male', '7_14_female']
train_df.loc[:2, ppl_count_7_14]


# 

# In[ ]:


ppl_count_0_17 = ['0_17_all', '0_17_male', '0_17_female']
train_df.loc[:2, ppl_count_0_17]


# 

# In[ ]:


ppl_count_16_29 = ['16_29_all', '16_29_male', '16_29_female']
train_df.loc[:2, ppl_count_16_29]


# 

# In[ ]:


ppl_count_0_13 = ['0_13_all', '0_13_male', '0_13_female']
train_df.loc[:2, ppl_count_0_13]


# 

# 

# In[ ]:


build_count_mat_columns = ['raion_build_count_with_material_info', 'build_count_block', 'build_count_wood', 'build_count_frame', 
                           'build_count_brick', 'build_count_monolith', 'build_count_panel', 'build_count_foam', 'build_count_slag', 'build_count_mix']
train_df.loc[:2, build_count_mat_columns]


# 

# In[ ]:


build_count_date_columns = ['raion_build_count_with_builddate_info', 'build_count_before_1920', 'build_count_1921-1945', 
                           'build_count_1946-1970', 'build_count_1971-1995', 'build_count_after_1995']
train_df.loc[:2, build_count_date_columns]


# 

# 

# In[ ]:


columns = np.array(train_df.columns)
[item for item in columns if "1line" in item]


# In[ ]:


one_line_columns = ['water_1line', 'big_road1_1line', 'railroad_1line']
train_df.loc[:2,one_line_columns]


# 

# In[ ]:


loc_km_dist_columns = ['kindergarten_km', 'school_km', 'park_km', 'green_zone_km', 'industrial_km', 'water_treatment_km', 'cemetery_km', 
                       'incineration_km', 'water_km', 'oil_chemistry_km', 'nuclear_reactor_km', 'radiation_km', 
                       'power_transmission_line_km', 'thermal_power_plant_km', 'ts_km', 'big_market_km', 'market_shop_km', 'fitness_km',
                       'swim_pool_km', 'ice_rink_km', 'stadium_km', 'basketball_km', 'hospice_morgue_km', 'detention_facility_km', 
                       'public_healthcare_km', 'university_km', 'workplaces_km', 'shopping_centers_km', 'office_km', 'additional_education_km',
                       'preschool_km', 'big_church_km', 'church_synagogue_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km']
train_df.loc[:2,loc_km_dist_columns]


# 

# In[ ]:


loc_metro_columns = ['ID_metro', 'metro_min_avto', 'metro_km_avto', 'metro_min_walk', 'metro_km_walk']
train_df.loc[:2,loc_metro_columns]


# 

# In[ ]:


loc_railroad_pubtrans_columns = ['ID_railroad_station_walk', 'railroad_station_walk_km', 'railroad_station_walk_min',
                                 'ID_railroad_station_avto', 'railroad_station_avto_km', 'railroad_station_avto_min',
                                 'public_transport_station_km', 'public_transport_station_min_walk', 
                                 'railroad_km', 'zd_vokzaly_avto_km', 'ID_railroad_terminal', 
                                 'ID_bus_terminal', 'bus_terminal_avto_km']
train_df.loc[:2,loc_railroad_pubtrans_columns]


# 

# 

# In[ ]:


road_columns = ['mkad_km', 'ttk_km', 'sadovoe_km', 'bulvar_ring_km', 'kremlin_km',
                'ID_big_road1', 'big_road1_km',
                'ID_big_road2', 'big_road2_km']
train_df.loc[:2,road_columns]


# 

# In[ ]:


ecology_columns = ['ecology']
train_df.loc[:3, ecology_columns]


# 

# 

# In[ ]:


count_500_columns = ['green_part_500', 'prom_part_500', 'office_count_500', 'office_sqm_500', 'trc_count_500', 'trc_sqm_500', 
                     'cafe_count_500', 'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500', 
                     'cafe_count_500_na_price', 'cafe_count_500_price_500', 'cafe_count_500_price_1000', 'cafe_count_500_price_1500',
                     'cafe_count_500_price_2500', 'cafe_count_500_price_4000', 'cafe_count_500_price_high',
                     'big_church_count_500', 'church_count_500', 'mosque_count_500',
                     'leisure_count_500', 'sport_count_500', 'market_count_500']
train_df.loc[:3,count_500_columns]


# 

# In[ ]:


count_1000_columns = ['green_part_1000', 'prom_part_1000', 'office_count_1000', 'office_sqm_1000', 'trc_count_1000',
                     'trc_sqm_1000', 'cafe_count_1000', 'cafe_sum_1000_min_price_avg', 'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000',
                     'cafe_count_1000_na_price', 'cafe_count_1000_price_500', 'cafe_count_1000_price_1000', 'cafe_count_1000_price_1500',
                     'cafe_count_1000_price_2500', 'cafe_count_1000_price_4000', 'cafe_count_1000_price_high', 
                     'big_church_count_1000', 'church_count_1000', 'mosque_count_1000', 
                     'leisure_count_1000', 'sport_count_1000', 'market_count_1000']
train_df.loc[:3,count_1000_columns]


# 

# In[ ]:


count_1500_columns = ['green_part_1500', 'prom_part_1500', 'office_count_1500', 'office_sqm_1500', 'trc_count_1500', 'trc_sqm_1500', 
                      'cafe_count_1500', 'cafe_sum_1500_min_price_avg', 'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500', 'cafe_count_1500_na_price',
                      'cafe_count_1500_price_500', 'cafe_count_1500_price_1000', 'cafe_count_1500_price_1500', 'cafe_count_1500_price_2500',
                      'cafe_count_1500_price_4000', 'cafe_count_1500_price_high', 
                      'big_church_count_1500', 'church_count_1500', 'mosque_count_1500',
                      'leisure_count_1500', 'sport_count_1500', 'market_count_1500',]
train_df.loc[:3,count_1500_columns]


# 

# In[ ]:


count_2000_columns = ['green_part_2000', 'prom_part_2000', 'office_count_2000', 'office_sqm_2000', 'trc_count_2000', 'trc_sqm_2000', 
                      'cafe_count_2000', 'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000', 'cafe_count_2000_na_price',
                      'cafe_count_2000_price_500', 'cafe_count_2000_price_1000', 'cafe_count_2000_price_1500', 'cafe_count_2000_price_2500',
                      'cafe_count_2000_price_4000', 'cafe_count_2000_price_high',
                      'big_church_count_2000', 'church_count_2000', 'mosque_count_2000',
                      'leisure_count_2000', 'sport_count_2000', 'market_count_2000']
train_df.loc[:3,count_2000_columns]


# 

# In[ ]:


count_3000_columns = ['green_part_3000', 'prom_part_3000', 'office_count_3000', 'office_sqm_3000', 'trc_count_3000', 'trc_sqm_3000', 
                      'cafe_count_3000', 'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'cafe_count_3000_na_price',
                      'cafe_count_3000_price_500', 'cafe_count_3000_price_1000', 'cafe_count_3000_price_1500', 'cafe_count_3000_price_2500',
                      'cafe_count_3000_price_4000', 'cafe_count_3000_price_high',
                      'big_church_count_3000', 'church_count_3000', 'mosque_count_3000',
                      'leisure_count_3000', 'sport_count_3000', 'market_count_3000']
train_df.loc[:3,count_3000_columns]


# 

# In[ ]:


count_5000_columns = ['green_part_5000', 'prom_part_5000', 'office_count_5000', 'office_sqm_5000', 'trc_count_5000', 'trc_sqm_5000', 
                      'cafe_count_5000', 'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000', 'cafe_count_5000_na_price',
                      'cafe_count_5000_price_500', 'cafe_count_5000_price_1000', 'cafe_count_5000_price_1500', 'cafe_count_5000_price_2500',
                      'cafe_count_5000_price_4000', 'cafe_count_5000_price_high',
                      'big_church_count_5000', 'church_count_5000', 'mosque_count_5000',
                      'leisure_count_5000', 'sport_count_5000', 'market_count_5000']
train_df.loc[:3,count_5000_columns]


# 

# 

# In[ ]:


macro_oil_columns = ['oil_urals', 'brent']
train_df.loc[:2, macro_oil_columns]


# 

# In[ ]:


macro_gpd_columns = ['gdp_quart', 'gdp_quart_growth', 'gdp_deflator', 'gdp_annual', 'gdp_annual_growth', 'grp', 'grp_growth', 'income_per_cap',
       'real_dispos_income_per_cap_growth', 'salary', 'salary_growth']
train_df.loc[:2, macro_gpd_columns]


# 

# In[ ]:


macro_price_indexes_columns = ['cpi', 'ppi', 'rts', 'micex', 'micex_rgbi_tr', 'micex_cbi_tr', 'fixed_basket']
train_df.loc[:2, macro_price_indexes_columns]


# 

# In[ ]:


macro_price_indexes_columns = ['balance_trade', 'balance_trade_growth', 'net_capital_export', 'retail_trade_turnover',
                               'retail_trade_turnover_per_cap', 'retail_trade_turnover_growth']
train_df.loc[:2, macro_price_indexes_columns]


# 

# In[ ]:


macro_exchange_rates_columns = ['usdrub', 'eurrub']
train_df.loc[:2, macro_exchange_rates_columns]


# 

# In[ ]:


macro_build_contract_columns = ['average_provision_of_build_contract', 'average_provision_of_build_contract_moscow']
train_df.loc[:2, macro_exchange_rates_columns]


# 

# In[ ]:


macro_deposits_morgages_columns = ['deposits_value', 'deposits_growth', 'deposits_rate', 'mortgage_value', 'mortgage_growth', 'mortgage_rate']
train_df.loc[:2, macro_deposits_morgages_columns]


# 

# In[ ]:


macro_demographics_columns = ['labor_force', 'unemployment', 'employment', 'marriages_per_1000_cap', 'divorce_rate', 'pop_natural_increase', 
                              'pop_migration', 'pop_total_inc', 'childbirth', 'mortality', 'average_life_exp', 'infant_mortarity_per_1000_cap',
                              'perinatal_mort_per_1000_cap', 'incidence_population']
train_df.loc[:2, macro_demographics_columns]


# 

# In[ ]:


macro_invest_enterprises_columns = ['invest_fixed_capital_per_cap', 'invest_fixed_assets', 'invest_fixed_assets_phys', 'profitable_enterpr_share', 'unprofitable_enterpr_share', 
                                    'share_own_revenues', 'overdue_wages_per_cap', 'fin_res_per_cap']
train_df.loc[:2, macro_invest_enterprises_columns]


# 

# In[ ]:


macro_housing_columns = ['housing_fund_sqm', 'lodging_sqm_per_cap', 'water_pipes_share', 'baths_share', 'sewerage_share', 'gas_share',
                         'hot_water_share', 'electric_stove_share', 'heating_share', 'old_house_share', 
                         'rent_price_4+room_bus', 'rent_price_3room_bus', 'rent_price_2room_bus', 'rent_price_1room_bus', 
                         'rent_price_3room_eco',  'rent_price_2room_eco', 'rent_price_1room_eco',
                         'apartment_build', 'apartment_fund_sqm']
train_df.loc[:2, macro_housing_columns]


# 

# In[ ]:


macro_education_columns = ['load_of_teachers_preschool_per_teacher', 'child_on_acc_pre_school', 'load_of_teachers_school_per_teacher',
                           'students_state_oneshift', 'modern_education_share', 'old_education_build_share']
train_df.loc[:2, macro_education_columns]


# 

# In[ ]:


macro_healthcare_columns = ['provision_doctors', 'provision_nurse', 'load_on_doctors', 'power_clinics', 
                            'hospital_beds_available_per_cap', 'hospital_bed_occupancy_per_year']
train_df.loc[:2, macro_healthcare_columns]


# 

# In[ ]:


macro_retail_columns = ['provision_retail_space_sqm', 'provision_retail_space_modern_sqm']
train_df.loc[:2, macro_retail_columns]


# 

# In[ ]:


macro_food_culture_columns = ['turnover_catering_per_cap', 'theaters_viewers_per_1000_cap', 'seats_theather_rfmin_per_100000_cap', 'museum_visitis_per_100_cap',
                              'bandwidth_sports', 'population_reg_sports_share', 'students_reg_sports_share']
train_df.loc[:2, macro_food_culture_columns]


# # Modeling

# 

# In[ ]:


full = del_columns(train_df, ["id", "price_doc", "price_doc_euro", "price_doc_dollars"])
full = del_columns(test_df, ["id"])
print("\n", full.shape)


# In[ ]:


numeric_columns = get_numeric_columns(train_df)
print(train_df.shape, "to numeric only:", train_df[numeric_columns].shape)


# 

# In[ ]:


val_time = 1407
dev_indices = np.where(train_df["time_yearmonth"]<val_time)
val_indices = np.where(train_df["time_yearmonth"]>=val_time)
dev_X = train_df[numeric_columns].ix[dev_indices]
val_X = train_df[numeric_columns].ix[val_indices]
dev_y = train_y[dev_indices]
val_y = train_y[val_indices]
print(dev_X.shape, val_X.shape)


# 

# In[ ]:





# 

# In[ ]:


#import xgboost as xgb


# 

# In[ ]:





# ## Neural Network
# 

# In[ ]:





# 

# In[ ]:





# # Ensembles

# 

# In[ ]:





# ### Stacking

# In[ ]:





# In[ ]:




