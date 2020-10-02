#!/usr/bin/env python
# coding: utf-8

# # Loading in the Necessary Modules and Taking a Look at the Dataset Folder

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

PATH_TO_CSV_FILE = os.path.join('..', 'input', 'daily-power-generation-in-india-20172020')
for dirname, _, filenames in os.walk(PATH_TO_CSV_FILE):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Configuring the setting for the Plots

# In[ ]:


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# ---

# # Our First Look at the Data

# In[ ]:


CSV_FILE_LIST = ['State_Region_corrected.csv', 'file.csv']

state_wise_area_share_df = pd.read_csv(os.path.join(PATH_TO_CSV_FILE, CSV_FILE_LIST[0]))
power_generation_df = pd.read_csv(os.path.join(PATH_TO_CSV_FILE, CSV_FILE_LIST[1]))

state_wise_area_share_df.head(10)


# In[ ]:


power_generation_df.head(10)


# # Let's take a look at the shape of the Dataset

# In[ ]:


POWER_GENERATION_SHAPE = power_generation_df.shape
STATE_WISE_AREA_SHARE_SHAPE = state_wise_area_share_df.shape

print("State Wise Power Share Dataset:")
print(f"Number of Rows: {STATE_WISE_AREA_SHARE_SHAPE[0]}")
print(f"Number of Columns: {STATE_WISE_AREA_SHARE_SHAPE[1]}")
print()

print("Power Generation Dataset:")
print(f"Number of Rows: {POWER_GENERATION_SHAPE[0]}")
print(f"Number of Columns: {POWER_GENERATION_SHAPE[1]}")


# # Questions:
# 1. What is the comparision between the Actual and Estimated Power Generated?
#     - Per `Region`
#     - Per type of power generation technique
#     - Per Year
# 2. What is the distribution of electricity (in terms of MU/10km2) in a particular region?
#     - Per power generation technique
#     - All power generation techniques taken together for that region
# 3. Which power generation technique dominates which region of India?
# 4. What is the: **percentage of power generation from each region** ***Vs*** **percentage of area occupied by that region**
#     - Create a stacked bar chart containing the state-wise breakup of every region's area
# 5. What is the region-wise break up of the area occupied by the corresponding state/union territory?

# ---

# # Reviewing and Cleaning the Dataset

# ### Firstly, what are the regions of India according to the dataset? Is it just *North*, *South*, *East* and *West* or is it more detailed?

# In[ ]:


state_wise_area_share_df['Region'].unique()


# In[ ]:


power_generation_df['Region'].unique()


# ### What are the State / Union territory (UT) of India, according to the dataset?
# #### This data will also give us an idea of how accurate/up-to-date this dataset is

# In[ ]:


state_wise_area_share_df['State / Union territory (UT)'].unique()


# ## Observations:
# - The set of regions in the two DataFrames are not exactly the same.
# - Other than the *Northern*, *Western*, *Southern* and *Eastern* values (similar to the regular *North*, *South*, *East* and *West*), we also have a common value called *Northeastern*, which refers to the famously known *'Seven Sisterly States'* otherwise called the *'Seven Sisters'*, which are the Eight North-East states of India:
#     1. Arunachal Pradesh
#     2. Assam
#     3. Manipur
#     4. Meghalaya
#     5. Mizoram
#     6. Nagaland
#     7. Sikkim
#     8. Tripura
# 
# 
# - It's great to see that the dataset shows Telangana **and** Andhra Pradesh and not one unified states which shows the data is quite up to date. [More information here (Wikipedia)](https://en.wikipedia.org/wiki/Andhra_Pradesh#Post-independence)
# 
# - Similarly, the mention of Ladakh and Jammu and Kashmir in the dataset also shows that the dataset is quite up to date, with the recent changes in the country. [More information here (Wikipedia)](https://en.wikipedia.org/wiki/Jammu_and_Kashmir_(union_territory))
# 
# ---
# 
# ### Here is an image of India with the regions and state boundaries highlighted:
# 
# ![Map of India with regions and state boundaries highlighted](https://i.imgur.com/U1unVf4.png)
# 
# **Note:** This image was taken from [National Power Portal Dashboard](https://npp.gov.in/dashBoard/cp-map-dashboard)
# 
# ---
# 
# ### How do we handle the value of *Central*?
# The best way to handle rows with the value *Central* in the `Region` column is by replacing those values with either the value: *Northern*, *Western*, *Southern*, *Eastern* or *NorthEastern* based on the state present in that row. 
# 
# Now, the data in the `power_generation_df` DataFrame which only has either the values: *Northern*, *Western*, *Southern*, *Eastern* or *NorthEastern* in its `Region` column, was got from the [National Power Portal Dashboard](https://npp.gov.in/dashBoard/cp-map-dashboard), so let's use the dashboard's classification of *Northern*, *Western*, *Southern*, *Eastern* and *NorthEastern* states, to classify the states in the `state_wise_area_share_df` DataFrame which contains the extra *central* value in the `Region` column.
# 
# Let's do this now...

# In[ ]:


# States in the North Region
NORTH_REGION_STATES = [
    'Ladakh',
    'Himachal Pradesh',
    'Uttarakhand',
    'Punjab',
    'Haryana',
    'Jammu and Kashmir',
    'Rajasthan',
    'Delhi',
    'Chandigarh',
    'Uttar Pradesh',
]

# States in the North Eastern Region
NORTH_EASTERN_REGION_STATES = [  
    'Arunachal Pradesh',
    'Assam',
    'Meghalaya',
    'Manipur',
    'Mizoram',
    'Nagaland',
    'Tripura',
]


# States in the South Region
SOUTH_REGION_STATES = [
    'Tamil Nadu',
    'Telangana',
    'Kerala',
    'Karnataka',
    'Andhra Pradesh',
    'Puducherry',
]

# States in the West Region
WEST_REGION_STATES = [
    'Maharashtra',
    'Gujarat',
    'Madhya Pradesh',
    'Dadra and Nagar Haveli and Daman and Diu',
    'Chhattisgarh',
    'Goa',
]

# States in the East Region
EAST_REGION_STATES = [
    'Odisha',
    'Bihar',
    'West Bengal',
    'Jharkhand',
    'Sikkim',
]


# In[ ]:


REGION_NAMES = {'N': 'Northern', 'W': 'Western', 'S': 'Southern', 'E': 'Eastern', 'NE': 'NorthEastern'}

state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(NORTH_REGION_STATES), 'Region'] = REGION_NAMES['N']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(SOUTH_REGION_STATES), 'Region'] = REGION_NAMES['S']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(EAST_REGION_STATES), 'Region'] = REGION_NAMES['E']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(WEST_REGION_STATES), 'Region'] = REGION_NAMES['W']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(NORTH_EASTERN_REGION_STATES), 'Region'] = REGION_NAMES['NE']

state_wise_area_share_df['Region'].unique()


# In[ ]:


state_wise_area_share_df.head(10)


# ---

# In[ ]:


COLUMN_HEADERS = power_generation_df.columns

COLUMN_HEADERS


# In[ ]:


power_generation_df.dtypes


# In[ ]:


for column_name in COLUMN_HEADERS:
    if column_name in ['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)']:
        power_generation_df[column_name] = power_generation_df[column_name].str.replace(',', '')
        power_generation_df[column_name] = pd.to_numeric(power_generation_df[column_name])


power_generation_df.head(10)


# In[ ]:


power_generation_df[['Thermal Generation Actual (in MU)', 'Nuclear Generation Actual (in MU)', 'Hydro Generation Actual (in MU)']].sum(axis=1)


# In[ ]:


# This also gives the same result as the previous cell
# power_generation_df.loc[:, ['Thermal Generation Actual (in MU)', 'Nuclear Generation Actual (in MU)', 'Hydro Generation Actual (in MU)']].sum(axis=1)


# In[ ]:


power_generation_df['Total Power Generation Actual (in MU)'] = power_generation_df[
                                                                                    [
                                                                                        'Thermal Generation Actual (in MU)',
                                                                                        'Nuclear Generation Actual (in MU)',
                                                                                        'Hydro Generation Actual (in MU)',
                                                                                    ]
                                                                                   ].sum(axis=1)


# In[ ]:


power_generation_df['Total Power Generation Estimated (in MU)'] = power_generation_df[
                                                                                        [
                                                                                            'Thermal Generation Estimated (in MU)',
                                                                                            'Nuclear Generation Estimated (in MU)',
                                                                                            'Hydro Generation Estimated (in MU)',
                                                                                        ]
                                                                                      ].sum(axis=1)


# In[ ]:


power_generation_df.head(10)


# ### Let's check if the values in the `Total Power Generation Actual (in MU)` column are correct

# In[ ]:


CHECKS = [
    math.isclose(624.23 + 30.36 + 273.27, 927.86),
    math.isclose(1106.89 + 25.17 + 72.00, 1204.06),
    math.isclose(576.66 + 62.73 + 111.57, 750.96),
]


if all(CHECKS):
    print('Looking Good!')
else:
    print('Something went wrong!!')


# ### Let's check if the values in the `Total Power Generation Estimated (in MU)` column are correct

# In[ ]:


CHECKS = [
    math.isclose(484.21 + 35.57 + 320.81, 840.59),
    math.isclose(1024.33 + 3.81 + 21.53, 1049.67),
    math.isclose(578.55 + 49.80 + 64.78, 693.13),
]


if all(CHECKS):
    print('Looking Good!')
else:
    print('Something went wrong!!')


# In[ ]:


power_generation_df['Date'] = pd.to_datetime(power_generation_df['Date'], format='%Y-%m-%d')

power_generation_df.head(10)


# In[ ]:


power_generation_df.dtypes


# In[ ]:


power_generation_df.isnull()


# In[ ]:


# power_generation_df.isnull().values


# In[ ]:


# type(power_generation_df.isnull().values)  # Converts the DataFrame into a numpy array


# In[ ]:


# power_generation_df.isnull().values.ravel()  # Flattens the 2D array into a 1D array


# In[ ]:


NO_OF_NAN_VALUES = np.count_nonzero(power_generation_df.isnull().values.ravel())
print(f"The number of NaN values in the dataset is: {NO_OF_NAN_VALUES}")


# In[ ]:


NO_OF_ROWS_WITH_NAN_VALUES = power_generation_df.shape[0] - power_generation_df.dropna().shape[0]
print(f"The number of rows with NaN values are: {NO_OF_ROWS_WITH_NAN_VALUES}")


# In[ ]:


AVG_NO_OF_NANs_PER_ROW = np.count_nonzero(power_generation_df.isnull().values.ravel()) / (power_generation_df.shape[0] - power_generation_df.dropna().shape[0])
print(f"So there are about {AVG_NO_OF_NANs_PER_ROW} NaN values per row")


# ## What are the missing values indicating??
# 
# So before we conclude that the `NaN` values in our dataset are from missing data and blame the dataset and their creators, we need to take a closer look as to what data is missing and why it could possibly be missing.

# In[ ]:


power_generation_df.isnull().any()


# So it looks like values seem to be missing only in two columns `Nuclear Generation Actual (in MU)` and `Nuclear Generation Estimated (in MU)`, so I feel that the possible reasons for this could be:
# 1. Data is missing
#     1. This is always our first instinct, but this can only be concluded if all the other hypothesis fail
# 2. Maybe the Nuclear plants were down for maintenance or some reason and hence did not produce any electrity on those days to record
# 3. Maybe there are no Nuclear plants in some regions meaning that there are no values to record there
# 
# Now, the most easiest hypothesis to try out is, whether or not there are any nuclear plants in some region(s)

# In[ ]:


power_generation_df[['Region', 'Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)']].groupby('Region').sum()


# In[ ]:


power_generation_df.columns


# In[ ]:


pd.DatetimeIndex(power_generation_df["Date"]).year.unique()


# In[ ]:


power_generation_df[['Region', 'Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)']].groupby('Region').sum()


# We notice something interesting here, which is that:
# There is *no nuclear power being generated* in the **Eastern** and **NorthEastern** regions, for over **4 years**.
# 
# Now, its highly unlikely that for 4 years the nuclear plants there have not produced any electricity due to maintenance or other such reasons. 
# So it is plausible that there are no nuclear power plants or at least no active nuclear power plants in the **Eastern** and **NorthEastern** regions of India.

# ### We can confirm this hypothesis by looking at the map of power generation plants in India
# #### Here is an image of India with the regions and state boundaries highlighted:
# 
# ![Map of India with regions and state boundaries highlighted](https://i.imgur.com/U1unVf4.png)
# 
# **Note:** This image was taken from [National Power Portal Dashboard](https://npp.gov.in/dashBoard/cp-map-dashboard)
# 
# 
# ### Conclusion:
# We observe that there are no nuclear power plants in the **Eastern** and **NorthEastern** regions of India, confirming our initial hypothesis.
# 
# ### What do we do with these columns?
# The right thing to do would be to fill the `NaN` values with 0 as that would show that no nuclear power is being generated in the **Eastern** and **NorthEatern** regions over the course of the 4 years of data we have.
# 
# We cannot remove the rows outright as that would remove the data regarding the other power generation techniques in the **Eastern** and **NorthEatern** regions of India.

# In[ ]:


power_generation_df.fillna(0.0, inplace=True)


# In[ ]:


power_generation_df


# In[ ]:


power_generation_df.describe()


# The above statistic is not correct with regards to the `Nuclear Generation` columns, as the values for mean, std etc, include the values from the **Eastern** and **NorthEastern** regions, which do not have any nuclear power plants (atleast no active ones). So the values from those regions are 0, and hence they would skew the stats for these columns.
# 
# We can correct this by producing these stats for every region separately...

# In[ ]:


power_generation_df.groupby('Region').describe().stack()


# ---

# # So let's start answering the questions, we had set out to solve...

# # Question 1
# ## a. What is the comparision between the Actual and Estimated Power Generated per *Region*?

# In[ ]:


power_generation_df.groupby('Region').sum().reset_index()


# In[ ]:


power_generation_df.groupby('Region').sum()["Total Power Generation Actual (in MU)"]
# power_generation_df.groupby('Region').sum()["Total Power Generation Estimated (in MU)"]


# In[ ]:


power_generation_df.groupby('Region').sum().index.values


# In[ ]:


power_generation_df.groupby('Region').sum()[["Total Power Generation Actual (in MU)", "Total Power Generation Estimated (in MU)"]]


# In[ ]:


TEMP = power_generation_df.groupby('Region').sum()[["Total Power Generation Actual (in MU)", "Total Power Generation Estimated (in MU)"]]

ax = plt.figure(figsize=(4,3), dpi=150)
plt.title("Actual Vs Estimated\nPower Generation per Region", y=1.08, fontsize=12)
TEMP.plot.bar(rot=75, ax=plt.gca(), width=.75)
plt.legend(fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5), loc='center left')
plt.ylabel('Power Generated in Mega Units (MU)')

plt.show()


# ## b. What is the comparision between the Actual and Estimated Power Generated per *type of power generation technique*?

# In[ ]:





# ## c. What is the comparision between the Actual and Estimated Power Generated per *Year*?

# In[ ]:





# 

# In[ ]:




