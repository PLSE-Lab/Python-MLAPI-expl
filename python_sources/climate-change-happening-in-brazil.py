#!/usr/bin/env python
# coding: utf-8

# ![](https://www.voicesofyouth.org/sites/default/files/images/2019-05/fcc2a43a-4de3-4054-bbe1-1911f3fea529.jpeg)

# **Climate change is one of the defining issues of our time. Over 30 UNESCO programmes in the sciences, education, culture and communication contribute to creating knowledge, educating and communicating about climate change, and to understanding the ethical implications for present and future generations.** (source: http://www.unesco.org/new/en/brasilia/natural-sciences/environment/climate-change/)

# Brazil mainly depends on the renewable energy sources when compared on other world's largest energy consumers. Brazil also ran a successful campaign to reduce deforestation by about 80 percent.(source: https://foreignpolicy.com/2019/01/04/brazil-was-a-global-leader-on-climate-change-now-its-a-threat/)
# 
# But, now Brazil is facing the climate change which can be seen in the dataset provided.
# 
# We will start the process by :
# 1. loading the necessary libraries
# 2. Insights of the dataset
# 3. Wrangling the dataset
# 4. Analysis

# ### 1. Loading the necessary libraries:
#     * pandas -> to handle the datasets
#     * seaborn and matplotlib -> to perform data visualizations

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### 2. Insights of the dataset:
#    First, we will load a single dataset, analyse the structure of the dataset and check whether there is a presence of missing dataset. We are performing this step because all the dataset have the same format. For this step, we will load belem data(station_belem)

# In[ ]:


belem_data = pd.read_csv('../input/temperature-timeseries-for-some-brazilian-cities/station_belem.csv')


# Let's explore the columns present in the dataset.

# In[ ]:


belem_data.columns


# The dataset has 18 columns and we will go through each column and their purpose.
# 1. YEAR - it represents the year of the data
# 2. JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC - it represents the temperature of the respective month
# 3. D-J-F - it represents the mean of the temperature of December, January and February
# 4. M-A-M - it represents the mean of the temperature of March, April and May
# 5. J-J-A - it represents the mean of the temperature of June, July and August
# 6. S-O-N - it represents the mean of the temperature of September, October and November
# 7. metANN - it represents the annual temperature mean

# ### 3. Wrangling the dataset:
# When I perform wrangling process in any dataset, I start by studying the columns of a dataset, which we did in the above step. The next step is that printing the first few records in order to check whether it has any missing data or any outliers. So, let's print few records.

# In[ ]:


belem_data.head()


# **whoa!!** we have found an abnormal data in the record. It seems that the missing data are represents by 999.90 
# #### (** WE NEED TO TAKE NECESSARY ACTION ABOUT THE CLIMATE CHANGE ELSE THIS MISSING DATA WILL BECOME THE ACTUAL DATA **)

# To handle the missing data, let's see what options we have:
# 1. we will remove the data from the dataset -> this will work if the entire row is missing (like row number 5 in the dataset above), but most of the rows have few missing data (i.e., only 4 to 5 missing data). so, this won't be a better choice.
# 
# 2. Replacing the missing data with the mean - Here, we will take the mean of the two rows above. By this step, we can replace the missing data without losing any real data

# Below, we created a function where it will replace the missing data with the mean value of the respective columns. This function is created because we can reuse the same function for multiple datasets.

# In[ ]:


def process_data(data, state):
    # Loading the dataset
    state_data = pd.read_csv('../input/temperature-timeseries-for-some-brazilian-cities/{}.csv'.format(data))
    
    # Looping through the columns of the dataset
    for i in range(state_data.shape[0]):
        for col in state_data.columns:
            # checks whether it's missing data or not
            if state_data.iloc[i][col] == 999.90:
                # calculating the mean value and replacing it
                mean_val = state_data.iloc[i-1][col]+state_data.iloc[i-2][col]+state_data.iloc[i-3][col]
                state_data.at[i, col] = mean_val/3
    
    # calculating the mean value (metANN)
    state_data['mean'] = state_data.apply(lambda x: x[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']].mean(), axis=1)
    state_data['state'] = state
    return state_data


# In[ ]:


belem_data = process_data('station_belem', 'Belem')


# In[ ]:


belem_data.head()


# Here, we can see that all the missing values has been replaced by the mean value of the prior two rows of the respective columns. Also, if you notice, the mean value (metANN) of 4th row is approximately equal to the calculated mean value with the missing data replaced.

# ### 3. Analysis:

# Let's plot the mean temperature of the year for Belem data and see how it's varying over the years.

# In[ ]:


fig, ax = plt.subplots(1,1)
fig.set_size_inches(20, 10)
sns.lineplot(x="mean", y="YEAR", data=belem_data)


# The information from the above graph is **VERY BAD** The temperature of Belem is rapidly increasing from 26 to 29 (approx.) over the year 1960 to 2020.
# 
# Let's load few more datasets and compare it. We will be using the function defined above which performs cleaning of data.

# In[ ]:


curitiba_data = process_data('station_curitiba', 'Curitiba')
fortaleza_data = process_data('station_fortaleza', 'Fortaleza')
goiania_data = process_data('station_goiania', 'Goiania')
macapa_data = process_data('station_macapa', 'Macapa')
manaus_data = process_data('station_manaus', 'Manaus')


# In[ ]:


weather_data = {
    'belem_data': belem_data,
    'curitiba_data': curitiba_data,
    'fortaleza_data': fortaleza_data, 
    'goiania_data': goiania_data, 
    'macapa_data': macapa_data,
    'manaus_data': manaus_data
}


# In[ ]:


fig, ax = plt.subplots(6,4)

fig.set_size_inches(20, 20)

for index, key in enumerate(weather_data.keys()):
    ax[index, 0].plot("YEAR", "D-J-F", data=weather_data[key])
    ax[index, 1].plot("YEAR", "M-A-M", data=weather_data[key])
    ax[index, 2].plot("YEAR", "J-J-A", data=weather_data[key])
    ax[index, 3].plot("YEAR", "S-O-N", data=weather_data[key])


# Overall, we can see that all the state in Brazil has an increase in their temperature over the time method. Also, from the graph above, **Curitiba** has facing unusual climate change over the year.

# Let's transform the dataset of belem and use Heatmap which is a suitable chart for the temperature dataset.

# In[ ]:


month_data = belem_data.drop(['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN', 'mean', 'state'], axis=1)
melted_data = month_data.melt(id_vars=["YEAR"], 
        var_name="Month", 
        value_name="Value")


# In[ ]:


fig, ax = plt.subplots(1,1)
fig.set_size_inches(25, 8)
ax = sns.heatmap(melted_data.pivot("Month","YEAR", "Value"),cmap="YlOrRd")


# So, we can visualize the heat of Belem over the year by the above chart. 
# 
# ### This is not good for Brazil. Actually, in most of the countries, it's worse. Let's make necessary steps in order to prevent the climate change.

# ##### Note: Feel free to comment your ideas or any suggestion in order to make the kernal more informative. Thank you for your support :)

# In[ ]:




