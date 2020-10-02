#!/usr/bin/env python
# coding: utf-8

# # USA Evictions Analysis

# In[ ]:


#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 15,11
sns.set(style="darkgrid")


# In[ ]:


#Import data
us_df = pd.read_csv("/kaggle/input/us.csv")     #This data was pulled from the evictionlab. You can find more at https://evictionlab.org/


# In[ ]:


#Data cleanup
us_df.columns = [name.replace("-","_") for name in us_df.columns]   #Replace "-" with "_" in column titles. Easier to use .query()
us_df['name'] = list(map(str, us_df['name']))   #Change type to string for all elements of name column.
us_df = us_df[us_df['name'].str.match('^[0-9]')==False] #Remove any row where name column starts with a digit. We don't want tracts.
states = us_df.filter(["name","parent_location"]).query("parent_location == 'USA'")["name"].unique()    #Create a list of all states in the data.
counties = us_df["name"][us_df["name"].str.lower().str.contains("county")]  #Create a list of all rows with the word county in the name.
cities = us_df["name"][us_df["name"].str.lower().str.contains("city")]  #Create a list of all rows with the word city in the name.
us_df["tuple"] = list(zip(us_df["name"],us_df["parent_location"]))  #Add a column with the name and parent_location as a tuple for easier indexing.


# # Data Exploration

# First let's see how many rows and columns we have

# In[ ]:


pd.DataFrame(us_df.shape, columns=["us_df"], index=["Rows","Columns"])


# Next, let's get a sense of what sort of data we have with the columns and data types, and determine how complete the data are with percent missing.

# In[ ]:


pd.DataFrame({"Data Type": us_df.dtypes, "Percent Null": us_df.isnull().sum() / len(us_df) * 100})


# There seems to be a non-trivial amount of missing data, especially from the critical eviction related columns. Let's see if we can get a better idea of which locations are most effected by missing data.

# In[ ]:


missing_by_location = us_df.groupby(["name","parent_location"]).apply(lambda x: x['eviction_rate'].isna().sum()/17) #Each location should have 17 data points (one for each year from 2000-2016).


# In[ ]:


missing_by_location.sort_values(ascending=False)


# Since we will be working off of averages, we will want to be sure that we are only looking at items with a sufficient amount of data. For that reason, we will drop all rows where location is associated with more than 0.2 percent of eviction_rate data missing.
# 
# Before we do that, let's provide an analysis on parts of the data which won't be as impacted by the missing components.

# In[ ]:


sns.distplot(us_df["eviction_rate"][us_df["eviction_rate"]<=100],bins=100)


# In[ ]:


sns.distplot(us_df["poverty_rate"])


# In[ ]:


sns.distplot(us_df["pct_renter_occupied"])


# In[ ]:


sns.distplot(us_df["median_household_income"])


# Now that we have some visualizations of our data, we will go ahead and proceed with dropping rows with an unsuitable amount of data.

# In[ ]:


to_drop = list(missing_by_location[missing_by_location >0.01].index)
us_df = us_df[us_df["tuple"].isin(to_drop) == False]


# Now let's check out our columns and nulls again.

# In[ ]:


pd.DataFrame({"Data Type": us_df.dtypes, "Percent Null": us_df.isnull().sum() / len(us_df) * 100})


# In[ ]:


us_df.shape


# Clearly we have disposed of a good deal of the data, but our missing data is now manageable.
# 
# Now, let's look at the states and larger cities with the highest average eviction rate.

# In[ ]:


highest_avg_evicrate_states_df = us_df.query("parent_location=='USA'").groupby("name").agg("mean").sort_values(by="eviction_rate",ascending=False).head(5)
highest_avg_evicrate_states_df.filter(["eviction_rate"]).rename(columns={"eviction_rate":"avg_eviction_rate"})


# In[ ]:


highest_avg_evicrate_cities_df = us_df.query("parent_location in @states and population >= 100000 and name not in @counties and name not in @cities").groupby(["name","parent_location"]).agg("mean").sort_values(by="eviction_rate",ascending=False).head(5) #Show major cities (pop >= 100k) with highest average eviction rates.
highest_avg_evicrate_cities_df.filter(["eviction_rate"]).rename(columns={"eviction_rate":"avg_eviction_rate"})


# Let's go ahead and plot out the eviction rate time-series for the top 10 states and cities.

# In[ ]:


top_names = list(highest_avg_evicrate_states_df.index)
ts_data = us_df.query("name in @top_names and parent_location == 'USA'").filter(["year","name","eviction_rate"])
sns.lineplot(x="year",y="eviction_rate",data=ts_data,hue="name").set_title("Top 5 States by Year and Average Eviction Rate")
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)


# In[ ]:


top_names = list(highest_avg_evicrate_cities_df.index)
ts_data = us_df[us_df["tuple"].isin(top_names)]
sns.lineplot(x="year",y="eviction_rate",data=ts_data,hue="name").set_title("Top 5 Cities by Average Eviction Rate")
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)


# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(us_df.corr(),annot=True)


# We include some noteworthy observations from the heatmap below. We should add that as population, and renter_occupied_households go up, evictions seem to increase&mdash;&mdash;as we might expect. However, there is one noteworthy observation to make:
# - eviction_rate has a salient positive correlation with pct_af_american.
# - The percentage of white citizens has a negative correlation with poverty rate whereas the percentage of african american, hispanic, and american indian citizens all have a salient positive correlation with poverty rate.
# - The percentage of white citizens has a negative correlation with pct_renter_occupied whereas the percentage of african american and hispanic citizens all have a salient positive correlation with pct_renter_occupied.
# - pct_renter_occupied has a very strong correlation with population. This might suggest, as we would expect, that more densely populated areas such as cities see higher rates of renting.
