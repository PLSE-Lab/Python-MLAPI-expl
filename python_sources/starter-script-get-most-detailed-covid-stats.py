#!/usr/bin/env python
# coding: utf-8

# # Starter script to simply get most recent COVID stats by detailed locaiton
# 
# 
# ## Motivaiton
# This is a very simple pyhton script that can be imported/forked as a strating point for anlysis country based statistics for COVID. There are many APIs and various ways to access the data but this allows for very easy integration into any existing workflow.
# 
# ## Data
# The data that feeds this method comes from the [BING API ](https://bing.com/covid). It is updated on an hourly basis and pulls from many resources to have the most up to date stats of any references available.
# 
# ### COVID Stats fields: 
# 1. "totalConfirmed"
# 2. "totalDeaths"
# 3. "totalRecovered"
# 4. "totalRecoveredDelta"
# 5. "totalDeathsDelta"
# 6. "totalConfirmedDelta"
# 7. "lethality_rate"
# 8. "recovery_rate"
# 
# Deltas are updated on a 24 hour rolling window from Zulu/GMT.
# 
# #### If there are more fields anyone thinks are useful let me know in a comment!
# 
# 
# ## Geography
# The data is broken down by Country --> State/Province --> County/Municipality. All of the entries have their Lat/Long available for mapping or other reference. If lower level data is not avaiable the preceding, less specific, information is filled in with a prefix of * to indicate it is from other column. (i.e Country->State->State* if the County is missing)
# 
# ## Script

# In[ ]:


import pandas as pd
from pandas.io.json import json_normalize

def get_covid_data(country_list=None):
        """
        Takes a list of countries and returns most recent COVID data from Bing API:
        https://bing.com/covid/data
        
        If country list is blank it returns all countries data
        """
        
        #Get all COVID data by reading the json from the bing COVID api
        all_covid_data = pd.DataFrame(list(pd.read_json("https://bing.com/covid/data").areas))

        #Get data based on county ids; Can comment out to use all countires
        if country_list:
            selected_covid_data = all_covid_data[all_covid_data["id"].isin(country_list)]
        else:
            selected_covid_data = all_covid_data.copy()

        #Normalize json to pull out lower level geo features(i.e state-> county)
        #If there are no lower levels then return the series
        def json_norm_levels(series, record_path):
            normed_df = json_normalize(list(series), record_path)
            #If nothing to normalize return DF
            if normed_df.empty:
                return pd.DataFrame(series)
            else:
                return normed_df
        
        country_cases = pd.concat(list(selected_covid_data.areas.apply(json_norm_levels, record_path="areas"))
                                  ,sort=False)
        
        #Seperate id levels to get geolocaiton fedelity
        def list_rev(list_to_rev):
            list_to_rev.reverse()
            return list_to_rev
        
        breakdown = country_cases.id.str.split("_").apply(list_rev).apply(pd.Series)
        
        # If the countries returned do not have county or municipal data make new col
        # Fill with state/providence data
        if len(breakdown.columns) == 2:
            breakdown["county_municipal"] = None
            
        #Rename columns and fill in missing
        breakdown.columns = ["country", "state_province", "county_municipal"]
        
        #Fill missing county data with state and state data with country with indicator
        breakdown["state_province"].fillna(breakdown["country"] + "*", inplace=True)
        breakdown["county_municipal"].fillna(breakdown["state_province"] + "*", inplace=True)
        
        #Fill in missing stats data with 0
        covid_stats_cols = ["totalConfirmed","totalDeaths","totalRecovered",
                            "totalRecoveredDelta","totalDeathsDelta","totalConfirmedDelta"]
        country_cases[covid_stats_cols] = country_cases[covid_stats_cols].astype("float")
        country_cases[covid_stats_cols] = country_cases[covid_stats_cols].fillna(0)
        
        #Add additional stats
        #Lethality rate
        country_cases["lethality_rate"] = country_cases["totalDeaths"]/country_cases["totalConfirmed"]
        
        #Recovery rate
        country_cases["recovery_rate"] = country_cases["totalRecovered"]/country_cases["totalConfirmed"]
        
        #Return combined df
        return pd.concat([country_cases, breakdown], axis=1)

#Leave blank to get all countires
all_counrty_cases = get_covid_data()
all_counrty_cases.sort_values(by="recovery_rate")


# In[ ]:


#Can also specify certain countires with a list
specific_country_cases = get_covid_data(["spain", "germany", "russia"])
specific_country_cases.sample(10)


# # Sample Statistics

# ## Country Summary

# In[ ]:


covid_stats_cols = ["totalConfirmed","totalDeaths","totalRecovered","totalRecoveredDelta",
                    "totalDeathsDelta","totalConfirmedDelta", "lethality_rate", "recovery_rate"]
country_sums = all_counrty_cases.groupby("country")[covid_stats_cols].sum()
country_sums.sample(10)


# In[ ]:


country_sums.describe()


# ## State/Province Summary

# In[ ]:


state_sums = all_counrty_cases.groupby("state_province")[covid_stats_cols].sum()
state_sums.sample(10)


# In[ ]:


state_sums.describe()


# ## County/Municipality Summary

# In[ ]:


county_sums = all_counrty_cases.groupby("county_municipal")[covid_stats_cols].sum()
county_sums.sample(10)


# In[ ]:


#Removing the counties/municipalities that were filled by states and therefore end with *
county_sums_na_removed = county_sums[~county_sums.index.str.endswith("*")]
county_sums_na_removed.sample(10)


# In[ ]:


county_sums_na_removed.describe()

