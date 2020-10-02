#!/usr/bin/env python
# coding: utf-8

# # Climate Change and CO2 levels in atmosphere
# In this notebook I am going to explore the changes in global mean temperatures, as well as the rise of CO2 concentrations in atmosphere.

# In[ ]:


#Library and data importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
data_greece = data_country[data_country["Country"] == "Greece"].copy()
data_greece["dt"] = pd.to_datetime(data_greece["dt"])

data_global = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")
data_global["dt"] = pd.to_datetime(data_global["dt"])
co2_ppm = pd.read_csv("../input/carbon-dioxide/archive.csv")


# In[ ]:


annual_mean_global = data_global.groupby(data_global["dt"].dt.year).mean()
reference_temperature_global = annual_mean_global.loc[1951:1980].mean()["LandAndOceanAverageTemperature"]
annual_mean_global["Anomaly"] = annual_mean_global["LandAndOceanAverageTemperature"] - reference_temperature_global

annual_mean_greece = data_greece.groupby(data_greece["dt"].dt.year).mean()
reference_temperature_greece = annual_mean_greece.loc[1951:1980].mean()["AverageTemperature"]
annual_mean_greece["Anomaly"] = annual_mean_greece["AverageTemperature"] - reference_temperature_greece


# I calculated the mean temperature of the 1951 - 1980 period to establish the global base mean temperature. This is standard practice in climate science.  The deviation from this temperature is added in the Anomaly column. I also created a new dataframe for my home country (Greece), and repeated this process.

# In[ ]:


plt.figure()
plt.style.use("fivethirtyeight")
annual_mean_global.loc[1960:2015]["Anomaly"].plot(figsize = (10,5), grid=True, legend=True)
plt.title("Annual anomaly from base mean temperature (Global)")
plt.xlabel('')
plt.ylabel('Temperature Anomaly')
plt.show()


# As we can see, the global mean temperature has grown steadily the past decades, leading to a temperature anomaly of about 0.75 celsius in 2015. As expected, this result is consistent with the scientific consensus on climate change.

# In[ ]:


plt.figure()
plt.style.use("fivethirtyeight")
annual_mean_greece.loc[1960:2012]["Anomaly"].plot(figsize = (10,5), grid=True, legend=True)
plt.title("Annual anomaly from base mean temperature (Greece)")
plt.xlabel('')
plt.ylabel('Temperature Anomaly')
plt.show()


# The temperature has also steadily increased in my home country Greece.

# In[ ]:


plt.figure()
plt.style.use("fivethirtyeight")
annual_co2_ppm = co2_ppm.groupby(co2_ppm["Year"]).mean()
annual_co2_ppm.loc[1960:2015]["Carbon Dioxide (ppm)"].plot(figsize = (10,5), grid=True, legend=True)
plt.title("Global annual CO2 levels in atmosphere")
plt.ylabel("CO2 parts per million")
plt.show()


# The CO2 levels in atmosphere have steadily risen in the 1950-2010 period, indicating a linear relation between greenhouse gases and global temperature.

# In[ ]:


annual_co2_temp = pd.merge(annual_mean_global.loc[1960:2015], annual_co2_ppm.loc[1960:2015], left_index=True, right_index=True)
annual_co2_temp = annual_co2_temp[["LandAndOceanAverageTemperature", "Anomaly", "Carbon Dioxide (ppm)"]].copy()
annual_co2_temp.corr()


# The correlation coefficient of CO2 and temperature anomaly is 0.92 , confirming the linear relation between the two variables.

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x="Anomaly",y="Carbon Dioxide (ppm)", data=annual_co2_temp)


# This scatter plot visualizes the linear relation between CO2 levels and temperature anomaly.

# In[ ]:




