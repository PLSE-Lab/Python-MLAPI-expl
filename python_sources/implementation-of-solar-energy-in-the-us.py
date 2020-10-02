#!/usr/bin/env python
# coding: utf-8

# # **Visualizing trends in the Implementation of Solar Energy in the US**
# In this python notebook, I seek to illustrate and analyze the implementation of solar energy in the United States. The source of data is Stanford University's [Deep Solar Project](http://web.stanford.edu/group/deepsolar/home), a deep learning framework that analyzed satellite images to detect solar panels throughout the country and recorded relevant environmental and socioeconomic factors for every location. I will visualize data by plotting these factors against each other as well as against solar panel data. Finally, I will provide analysis of the the plots and identify trends and correlations in the deployment of solar energy. 

# # **I. Importing Libraries **

# In[ ]:


import os              # accessing directory structure
import numpy as np     # linear algebra
import pandas as pd    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
# libraries for maps
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

# %matplotlib inline
# pyplt.style.use('seaborn-whitegrid')
# sns.set_context("poster")
print("Libraries imported.")


# # ** II. Reading and Preparing Data for Visualization **

# Read CSV file into a pandas data frame <br>
# Clean data 

# In[ ]:


# create data frames from the original file that do not produce errors
dt1 = pd.read_csv('../input/deepsolar_tract.csv', nrows=6077)
dt2 = pd.read_csv('../input/deepsolar_tract.csv', skiprows =6119, names=["Unnamed: 0", "tile_count", "solar_system_count", "total_panel_area", "fips", "average_household_income", "county", "education_bachelor", "education_college", "education_doctoral", "education_high_school_graduate", "education_less_than_high_school", "education_master", "education_population", "education_professional_school", "employed", "gini_index", "heating_fuel_coal_coke", "heating_fuel_electricity", "heating_fuel_fuel_oil_kerosene", "heating_fuel_gas", "heating_fuel_housing_unit_count", "heating_fuel_none", "heating_fuel_other", "heating_fuel_solar", "land_area", "per_capita_income", "population", "population_density", "poverty_family_below_poverty_level", "poverty_family_count", "race_asian", "race_black_africa", "race_indian_alaska", "race_islander", "race_other", "race_two_more", "race_white", "state", "total_area", "unemployed", "water_area", "education_less_than_high_school_rate", "education_high_school_graduate_rate", "education_college_rate", "education_bachelor_rate", "education_master_rate", "education_professional_school_rate",	"education_doctoral_rate", "race_white_rate", "race_black_africa_rate", "race_indian_alaska_rate", "race_asian_rate", "race_islander_rate", "race_other_rate", "race_two_more_rate", "employ_rate", "poverty_family_below_poverty_level_rate", "heating_fuel_gas_rate", "heating_fuel_electricity_rate", "heating_fuel_fuel_oil_kerosene_rate", "heating_fuel_coal_coke_rate", "heating_fuel_solar_rate", "heating_fuel_other_rate", "heating_fuel_none_rate", "solar_panel_area_divided_by_area", "solar_panel_area_per_capita", "tile_count_residential", "tile_count_nonresidential", "solar_system_count_residential", "solar_system_count_nonresidential", "total_panel_area_residential", "total_panel_area_nonresidential", "median_household_income", "electricity_price_residential", "electricity_price_commercial", "electricity_price_industrial", "electricity_price_transportation", "electricity_price_overall", "electricity_consume_residential", "electricity_consume_commercial", "electricity_consume_industrial", "electricity_consume_total", "household_count", "average_household_size", "housing_unit_count", "housing_unit_occupied_count", "housing_unit_median_value", "housing_unit_median_gross_rent", "lat", "lon", "elevation", "heating_design_temperature", "cooling_design_temperature", "earth_temperature_amplitude", "frost_days", "air_temperature", "relative_humidity", "daily_solar_radiation", "atmospheric_pressure", "wind_speed", "earth_temperature", "heating_degree_days", "cooling_degree_days", "age_18_24_rate", "age_25_34_rate", "age_more_than_85_rate", "age_75_84_rate", "age_35_44_rate", "age_45_54_rate", "age_65_74_rate", "age_55_64_rate", "age_10_14_rate", "age_15_17_rate", "age_5_9_rate", "household_type_family_rate", "dropout_16_19_inschool_rate", "occupation_construction_rate", "occupation_public_rate", "occupation_information_rate", "occupation_finance_rate", "occupation_education_rate", "occupation_administrative_rate", "occupation_manufacturing_rate", "occupation_wholesale_rate", "occupation_retail_rate", "occupation_transportation_rate", "occupation_arts_rate", "occupation_agriculture_rate", "occupancy_vacant_rate", "occupancy_owner_rate", "mortgage_with_rate", "transportation_home_rate", "transportation_car_alone_rate", "transportation_walk_rate", "transportation_carpool_rate", "transportation_motorcycle_rate", "transportation_bicycle_rate", "transportation_public_rate", "travel_time_less_than_10_rate", "travel_time_10_19_rate", "travel_time_20_29_rate", "travel_time_30_39_rate", "travel_time_40_59_rate", "travel_time_60_89_rate", "health_insurance_public_rate", "health_insurance_none_rate", "age_median", "travel_time_average", "voting_2016_dem_percentage", "voting_2016_gop_percentage", "voting_2016_dem_win", "voting_2012_dem_percentage", "voting_2012_gop_percentage", "voting_2012_dem_win", "number_of_years_of_education", "diversity", "number_of_solar_system_per_household", "incentive_count_residential", "incentive_count_nonresidential", "incentive_residential_state_level", "incentive_nonresidential_state_level", "net_metering", "feedin_tariff", "cooperate_tax", "property_tax", "sales_tax", "rebate", "avg_electricity_retail_rate"])

# append data frames into a dataframe solardata
solardata = dt1.append(dt2)


# Replace all NA values in each data column with the mean of the column <br>
# View a summary of the Data

# In[ ]:


# replace NA values with mean 
solardata.fillna(solardata.mean())


# In[ ]:


# return total number of rows and columns 
print("The dataset has " + str(solardata.shape[0]) + " rows and " + str(solardata.shape[1]) + " columns.")

#return the first 5 lines of the dataset
solardata.head()


# # **III. Data Visualization **

# > # Histograms

# Illustrating the frequency of counts of solar panels through histograms made with matplotlib

# In[ ]:


print("Histogram of Solar Panel Counts")
fig, ax = plt.subplots(figsize=(20, 7))
tile_count = solardata['tile_count']
plt.hist(tile_count, 35, range=[0, 400], facecolor='goldenrod', align='mid')
plt.show()


# From this histogram of solar panel counts, we see most prominently that most locations analyzed by DeepSolar have no solar panels. In order to see a more stasticially rich portion of this data, I plotted a second histogram of solar panel counts for areas with at least 20 solar panels and less than 100. 

# In[ ]:


print("Histogram of Solar Panel Counts - A Closer Look")
tile_count = solardata['tile_count']
fig, ax = plt.subplots(figsize=(20, 7))
plt.hist(tile_count, 30, range=[20, 200], facecolor='goldenrod', align='mid')
plt.show()


# In this second histogram of solar panel counts, we are now able to see the trend more clearly among data in the middle of the first graph. While most residences and non-residential areas have solar panel counts of less than 50, we can now identifiy the frequencies in counts greater or less than that with more clarity,  and the outlook for future solar panel demployment looks a bit more positive. 

# In[ ]:


print("Histogram of Solar Panel Area per Capita")
area_capita = solardata['solar_panel_area_per_capita']
fig, ax = plt.subplots(figsize=(20, 5))
plt.hist(area_capita, 40, range=[0, 1], facecolor='rosybrown', align='mid')
plt.show()


# To ensure that the perception of the data is not skewed by population size or density in an area, I created another pair of histograms that show the frequency of solar panel area per capita, ranging from 0 to 1. As expected from the first histogram, most of the data points are 0, and to get a clearer view of the middle of this data, I will plot another zoomed-in histogram.

# In[ ]:


print("Histogram of Solar Panel Area per Capita - A Closer Look")
area_capita = solardata['solar_panel_area_per_capita']
fig, ax = plt.subplots(figsize=(20, 8))
plt.hist(area_capita, 35, range=[0.1, 0.6], facecolor='rosybrown', align='mid')
plt.show()


# > # Bar Plots

# The second method of visualization was a bar plot of using NumPy and matplotlib.

# In[ ]:


print("Analysis of Bar Plot")
print()

# defining variables to display and for plotting
total_tile = int(solardata['tile_count'].sum())
print("Total Solar Panels:", total_tile)
total_system = int(solardata['solar_system_count'].sum())
print("Total Solar Panel Systems:", total_system)

avg_tile_count = total_tile/72495
avg_tile_per_system = total_tile/ total_system
print("Average number of solar panels:", round(avg_tile_count))
print("Average number of panels per solar panel system:", avg_tile_per_system)

print()
total_rtile = int(solardata['tile_count_residential'].sum())
print("Total Solar Panels for Residential Purposes:", total_rtile)
total_nrtile = int(solardata['tile_count_nonresidential'].sum())
print("Total Solar Panels for Non-Residential Purposes*:", total_nrtile)
print()
total_rsystem = int(solardata['solar_system_count_residential'].sum())
print("Total Solar Systems for Residential Purposes:", total_rsystem)
total_nrsystem = int(solardata['solar_system_count_nonresidential'].sum())
print("Total Solar Panels for Non-Residential Purposes*:", total_nrsystem)
print()

# begin plotting 
# set width of bar
barWidth = 0.2
 
# set height of bar based on data 
barstotal = [total_tile, total_system]
barsres = [total_rtile, total_rsystem]
barsnonres = [total_nrtile, total_nrsystem]
 
# Set position of bar on x axis
r1 = np.arange(len(barstotal))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots(figsize=(15, 8))
# Make plot for each category of bars 
plt.bar(r1, barstotal, color='goldenrod', width=barWidth, edgecolor='white', label='total')
plt.bar(r2, barsres, color='cadetblue', width=barWidth, edgecolor='white', label='residential')
plt.bar(r3, barsnonres, color='darkseagreen', width=barWidth, edgecolor='white', label='nonresidential')
 
# Add xticks on the middle of the group bars
plt.xlabel('Residential and Non-Residential Use of Solar Energy', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(barstotal))], ["Solar Panels", "Solar Systems"])
 
# Create legend & display plot
plt.legend()
plt.show()


# In this barplot, I sought to compare the amount of solar panels used in residential areas and in non-residential areas. <br>
# The first set of bars represents the total amount of solar panels in each category and the second set represents the total amount of solar systems. There is an average of approximately 30 solar panels in a given loation in the contiguous United States, and an average of about 1.5 solar panels per solar panel system. <br>
# The categories of this bar plot are, in order,  total (yellow), residential (blue), and nonresidential (green).  If stacked on top of each other, the residentual and nonresidential bars would be equivalent to the area of the total. For context,  use of solar panels for non-residential purposes include commercial, industrial and transportation uses of solar energy. 
# <br>
# <br>
# ***Data Analysis:*** <br>
# Overall, there are more solar panels deployed residentially than used in the commerical, industrial, and transportation sectors combined. Although there are feweer numbers of solar panel systems used for nonresidential purposes than residential, there are many more solar panels in each nonresidential system. The implications of this are that more homes in the United States have solar panels than corporations and businesses do, but if these entities have implemented a solar panel system, they usually include a great deal more panels than the average household. 
# 

# > # Correlation Heat Maps & Linear Regression Plots

# The next two methods of visualization used were correlation heat maps, created with matplotlib and seaborn, and linear regression plots using seaborn to further illusutrate patterns and trends in data. 

# In[ ]:


#define columns to find correlations  
energyenv_quant_data = ["total_panel_area", "total_panel_area_residential", "total_panel_area_nonresidential","electricity_price_overall", "avg_electricity_retail_rate", "electricity_consume_total", "incentive_count_residential", "incentive_count_nonresidential", "heating_fuel_gas_rate", "heating_fuel_solar_rate","daily_solar_radiation",  "air_temperature",]

fig, axe = plt.subplots(figsize=(14,14))
sns.set_context("poster")
# RdYlBu_r
sns.set(font_scale=1.3)
corrmap = sns.color_palette("BrBG", 100)
sns.heatmap(solardata[energyenv_quant_data].corr(),annot=True, fmt='.2f',linewidths=1,cmap = corrmap)


# This first heat map finds correlation factors between solar energy implementation and environmnetal factors such as temperature and solar radiation, as well as economic factors such as the average retail price of electricity and the economic incentives given to residents of areas that attempt to encourage implementation of solar energy.
# <br>
# Factors with the strongest positive correaltion have squares that range from light to dark blue-green, while factors with little to no correlation are light yellow,  and factors with a strong negative correlation range from a deeper yellow to yellow-brown. 
# <br>
# <br>
# ***Data Analysis:*** <br>
# There was a strong positive correlation between daily solar radiation wth the incentive count (both residential (0.67) and nonresidential (0.61)). This is likely due to the fact that local governments and companies have a greater tendency to offer incentives and people and businesses are more likely to take up this offer in areas that recieve more solar radiation in order to get a better deal for the money invested in installing a solar panel system. 
# <br>
# There was also a slighlty less strong correlation between electricity price and average retail rate and the incentive count. The factor among these environmental and economic variables that was the most trongly coorelated with solar energy was residential incentive count, the linear regression plot of which I will graph below. 
# <br> <br>
# Something important to keep in mind that this data is slightly skewed by the very high number of 0 values in this data set for variables lke the data set. Zero values were kept while performing this data visualition because they are an accurate representation of the reality of how many areas do not have solar energy, however it overshadows some trends in the portions of the solar data where there are more non-zero data points.

# In[ ]:


fig, axe = plt.subplots(figsize=(8, 8))
sns.regplot(x=solardata["incentive_count_residential"], y=solardata["total_panel_area_residential"], fit_reg=True)


# Above is a scatter plot and linear regression plot of residential incentive count and total panel area. I wanted to illustrate the precise behavior of data given by the correlation value of 0.39 between these two factors in the previous heat map. The aforementioned data skewing caused by the substantial amount of zero-value data points (which overlap heavily here) is very evident in this plot, but nonetheless, we can still see a positive correlation - people are more increasingly ikely to install solar panels on their homes when they are given more financial incentives to do so. 

# In[ ]:


socioecon_quant_data = ["total_panel_area", "total_panel_area_residential", "median_household_income","number_of_years_of_education", "age_median", "poverty_family_count", "per_capita_income", "population_density", "employ_rate", "gini_index", "diversity", "voting_2016_dem_percentage", 'voting_2016_gop_percentage']

fig, axe = plt.subplots(figsize=(15, 15))
sns.set_context("poster")

sns.set(font_scale=1.3)
colmap2 = sns.color_palette("coolwarm", 100)
sns.heatmap(solardata[socioecon_quant_data].corr(),annot=True, fmt='.2f',linewidths=1,cmap = colmap2)


# This second heat map seeks to identify correlations among socioeconomic factors among the US population and the implentation of solar energy. Some of these socioeconomic factors include household income, employment rate, years of education, GINI Index ( a measure of income inequality, with a high value indicating high levels of inequality), and voting behaviors in the 2016 election. In addition to external factors like a population's environment and recieval of economic incentives (as explored in the previous heat map), internal factors like these can also influence the way that members of a population view and act on the implementation of solar energy.  
# <br>
# ***Data Analysis:***
# The factors with the strongest positive correlation are more bright red, while factors with little to no correlation are colored white, and factors with light to dark blue are more negatively correlated. Some patterns that stood out were that median household income was very positvely correlated to number of years of education (0.73) and employment rate (0.47), and negatively corrrelated with the GINI index (-0.25). 
# <br>
# The factor most heavily correlated with solar energy was median household income, which produced a correlation factor of 0.24. I illustrate the precise behavior of this correaltion in the liear regression plot below. 

# In[ ]:


print("Linear Regression Plot of Median Household Income and Total Residential Panel Area")
fig, axe = plt.subplots(figsize=(8, 8))
sns.regplot(x=solardata["median_household_income"], y=solardata["total_panel_area_residential"], fit_reg=True)


# Above is a linear regression plot of median household income and total residential panel area. This data is almost normally distributed, if it were not for the outliers. The median of the data is between \$50,000 and $100,000. This data is also heavily skewed by zero-value points for solar panel area, and despite the fact that household income does not continually increase, we can still see that there is a positive correlation between the median household income and the use of solar energy. 

# # ** IV. Conclusion **

# > # Summary of Data Analysis 

# Members of households with a middle to high range in income are more likely to invest in solar energy if they have more disposable income and can afford it. To help  advance the deployment of solar energy, some local ans statewide entities and governments offer financial incentives to households - and corporations - such that those who previously could not afford installing solar panels are now able to recieve a financial reward. In addition to daily solar radiation and the retail price of electricity, median household income and economic incentives remain the top predictors that a paticular area will have implemented solar energy. 
# <br>
# 

# > # Practical Implications of Data 

# DeepSolar, and the correlations uncovered from anlyzing its database, has a great potential to allow researchers, electricty and energy companies, as well as local, state, and federal policy makers recognize the trends in current solar energy. With this knowledge, they will be better able to underrstand what environmental, social, and economic factors influence a popultion's use of solar energy. Hopefully, in the future, we will reduce the inequity of solar energy distribtion and be more equiped to advance, support, manage, and maintain the implentation of solar energy to power the United States. 