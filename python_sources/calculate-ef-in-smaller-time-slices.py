#!/usr/bin/env python
# coding: utf-8

# It is required to calculate emissions factor on a time slice with a small average emission factor in the past.
# 
# > Bonus points for smaller time slices of the average historical emissions factors, such as one per month for the 12-month period.
# 
# https://www.kaggle.com/c/ds4g-environmental-insights-explorer/overview/evaluation
# 
# I"ll try to calculate monthly averaged simplified emissions factor.

# In[ ]:


from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import tifffile as tiff 


# # List up given Sentinel 5P OFFL NO2

# In[ ]:


s5p_no2_timeseries = glob.glob('../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/*')


# In[ ]:


data_monthly_divided = {}


# In[ ]:


for data in s5p_no2_timeseries:
     
    data_date =  datetime.strptime(data[:77], '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_%Y%m')
    data_date = data_date.strftime("%Y/%m")
    
    if not data_date in data_monthly_divided.keys():
        data_monthly_divided[data_date] = []
        
    data_monthly_divided[data_date].append(data)


# In[ ]:


for key in sorted(data_monthly_divided.keys()):
    print("number of data in" , key, "is", len(data_monthly_divided[key]))

There seems  more data than days in each month.
# # Calculate Quanity of Electricity Generated

# In[ ]:


#From https://www.kaggle.com/paultimothymooney/overview-of-the-eie-analytics-challenge

def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):
    for i in range(0, len(dataframe)):
        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]
    return dataframe


# In[ ]:


power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)
power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)
power_plants['latitude'] = power_plants['latitude'].astype(float)
a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8
power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 


# In[ ]:


power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()
power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']][29:30]
quantity_of_electricity_generated = power_plants_df['estimated_generation_gwh'][29:30].values
print('Quanity of Electricity Generated: ', quantity_of_electricity_generated)


# # Calculate simplified emissions factor

# In[ ]:


month = []
emissions = []

for key in sorted(data_monthly_divided.keys()):
    average_emissions = []
    datas = data_monthly_divided[key]
    for data in datas:
        
        #You can use your model here!!!
        average_emission = np.nanmean(tiff.imread(data))
        
        average_emissions.append(average_emission)
    
    month.append(key)
    emissions.append(np.nanmean(average_emissions))           


# In[ ]:


results = pd.DataFrame(columns=['month', 'emission','emisson factor'])
results = pd.DataFrame({'emission':emissions,
                       'emission factor':emissions/quantity_of_electricity_generated},
                    index=month)


# In[ ]:


results


# In[ ]:


results["emission factor"].plot()
plt.title('Monthly Mean Simplified Emissions Factor in Puerto Rico')

