#!/usr/bin/env python
# coding: utf-8

# # National Footprint accounts

# ## Overview of terms and our data

# ### Some basic concepts of ecological sustainabilty

# We can look at the world that we live in and its resources in terms of *supply* and *demand*.<br>
# 
# On the demand side: <br>
# **Ecoloigical Footprint (EF)** - Measures the quantity of nature it takes to support people or an economy.
# 
# On the supply side: <br>
# **Biocpacity (BC)** - Measures the capacity of a given biologically productive area to generate an on-going supply of renewable resources and to absorb its spillover wastes. <br>
# 
# Like in economics, when the demand (EF) exceeds the supply (BC) we are in a state of **Ecological Deficit**.<br>
# When the supply (BC) exceeds the demand (EF) we are in a state of **Ecological Reserve**.
# 
# **Global Hectare (GHA)** - A unit of land normalized by biological productivity across landtype. It starts with the total biological production and waste assimilation in the world, including crops, forests (both wood production and CO2 absorption), grazing and fishing. The total of these kinds of production, weighted by the richness of the land they use, is divided by the number of hectares used. <br>
# GHA can be devided per person in its corresponding geographical area (city, country, continent, world). In our data it appears as BC/EF per capita.
# 
# EF and BC are calculated for: <br>
# - **Crop land** - GHA of crop land available or demanded.
# - **Grazing land** - GHA of grazing land (used for meat, dairy, leather, etc.) Includes global hectares used for grazing, but not crop land used to produce feed for animals.
# - **Forest land** - GHA of forest land available (for sequestration and timber, pulp, or timber products) or demanded (for timber, pulp, or timber products).
# - **Fishing grounds** - GHA of marine and inland fishing grounds (used for fish & fish products).
# - **Built up land** - GHA of built-up land (land cover of human infrastructure).
# - **Carbon** - GHA of average forest required to sequester carbon emissions (for EF only. For BC it's calculated within the forest land section).

# ### Our data

# Our data source (available in through this link: https://www.kaggle.com/footprintnetwork/national-footprint-accounts-2018) includes records taken from **Global Footprint Network**, which is an international nonprofit organization founded in 2003, that collects and analyzes data to track the EF and BC for each country in the world and making this data availbale for all.
# 
# Let's load the data and see what we can detect:

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


df_io=pd.read_csv('../input/NFA 2018.csv')
df_io.head(10)


# Info of the data:

# In[ ]:


df_io.info()


# Column names:

# In[ ]:


df_io.columns


# We notice that the column name 'Percapita GDP (2010 USD)' can cause issues later, because of the spaces, so we rename it

# In[ ]:


df_io.rename(index=str, columns={'Percapita GDP (2010 USD)': 'Percapita_GDP_2010_USD'}, inplace=True)
df_io.columns


# So for each country we have also information regarding ISO alpha-3 code, UN_region, UN_subregion, Percapita GDP and population.<br>
# Let's see the types of **record** we have:

# In[ ]:


df_io['record'].unique()


# And count them:

# In[ ]:


df_io['record'].nunique()


# Why 10 types of records? <br>
# For each country and year included in the data, we can find mesurments of EF and BC in GHA and in per-capita units (4 out of 10 types of "record"), across all types of land/resource mentioned in the previous section and their sum total.<br>
# We note that EF represented in 4 categories.<br>
# The EF Consumption accounts for area required by consumption, while the EF of Production accounts only for the area required for production in the country only. They are related by the following equation:<br>
# **EF Consumption = EF Production + EF Imports - EF Exports** <br>
# EF Production, EF Imports and EF Exports also appear in both GHA and in per-capita units (remaining 6 out of 10)

# The first year of records:

# In[ ]:


df_io['year'].min()


# The last year of records:

# In[ ]:


df_io['year'].max()


# So the data ranges from 1961 to 2014.
# 
# Let's count Nan records for each type of record.

# In[ ]:


def num_nan(df):
    return df.shape[0]-df.count()

land_types=['crop_land', 'grazing_land', 'forest_land', 'fishing_ground','built_up_land', 'carbon', 'total']

(df_io[['record']+land_types]
    .groupby('record')
    .agg(num_nan)
)


# Looks like total column has no NAN valies, but we may want to remember the countries with missing records for the other columns

# In[ ]:


record_nan_countries = (df_io[['country']+['record']+land_types]
                            .set_index  ('country')
                            .isna       ()
                            .sum        (axis=1)
                            .loc        [lambda x: x>0]
                            .index
                            .unique     ()
                        )
record_nan_countries


# We do the same for country, GDP and population:

# In[ ]:


(df_io[['country','Percapita_GDP_2010_USD','population']]
            .groupby     ('country')
            .agg         (num_nan)
            .sort_values (['Percapita_GDP_2010_USD','population'], ascending=[0, 0])  
            .query       ('Percapita_GDP_2010_USD>0')
)


# again, we will keep the names of the countries with NAN in GDP, just in case we need it

# In[ ]:


GDP_nan_countries= (df_io[['country','Percapita_GDP_2010_USD','population']]
                        .groupby     ('country')
                        .agg         (num_nan)
                        .sort_values (['Percapita_GDP_2010_USD','population'], ascending=[0, 0])  
                        .query       ('Percapita_GDP_2010_USD>0')
                        .index
                    )
GDP_nan_countries


# #### Challenges:
# Not all countries are represented for all the years, therefore it's difficult to compare records for all countries and all the years. <br>
# First let's look at the counrties we have:

# In[ ]:


df_io['country'].unique()


# In[ ]:


df_io['country'].nunique()


# One thing we notice that "World" is listed as a country, so for global calculations we can use a dataframe that includes only  records for "World"

# In[ ]:


world = df_io[df_io['country']=='World']
world.head()


# In[ ]:


world.info()


# To see what exactly our data includes we want to know for each country the first year of records, the last year of records, and compare this range to the unique number of years of data per country, in order to see that the data we have is for all the years in range. For this we created a function to calculate the year range to help us in the aggrigation process.

# In[ ]:


def year_range(s): #calulates the year range for the countries
    return s.max()-s.min()+1


# In[ ]:


df_all_countries= df_io[df_io['country']!='World']
data_year_range=(df_all_countries
                        .groupby           ('country')['year']
                        .agg               ([np.min,np.max, year_range,'nunique'])
                        .sort_values       (['year_range'], ascending = False)
                 )

data_year_range


# In[ ]:


data_year_range.query("year_range!=nunique")
#if no rows are returned, the year ranges and the unique numbers of years match for all countries, thus the data is consecutive


# We count the year_range values

# In[ ]:


(data_year_range['year_range']
                         .value_counts()
                         .sort_index(ascending=False)
)


# Looks like 127 countries have 54 records. For those contries we look at the min/max years:

# In[ ]:


(data_year_range
             .query("year_range==54")['amin']
             .unique()
)             


# In[ ]:


(data_year_range
             .query("year_range==54")['amax']
             .unique()
)   


# So for all 127 countries that year range is 54, the data is consecutive, from 1961 to 2014.
# So we will slice them out to analyze the rest of the data

# In[ ]:


data_year_range.query("year_range!=54")


# We notice right away that New Zealend has 53 records and the one year its missing is for 2014. It's an important country in its region ao we would like not drop it from our data just because its missing 2014.<br>
# 
# Since 2014 for NZ doesn't even exist, we need to create it the entries. <br>
# The first approach was to copy the entries from NZ-2013, replace the year with 2014 and relevant vlaues with NaNs, concatinating to the NZ data and performing interpulation on group by records.
# Since 2014 is the last year in the series, this approach resulted in 2014 getting the exact same values as 2013, so the better approcah just to duplicate 2013 entries, change the year to 2014 and concatinate it to our original data.<br>

# **code for interpulation - not used**

# In[ ]:


#slicing out NZ data from out df_io
df_NZ_new= (df_all_countries
                     .loc[df_all_countries['country']=='New Zealand'] 
                     .loc[df_all_countries['year']==2013]
            )     
df_NZ_new['year']=2014
df_NZ_new[land_types+['Percapita_GDP_2010_USD','population']]=np.nan
df_NZ_new


# In[ ]:


#concatinate with existing data
df_NZ_exist= df_all_countries.loc[df_all_countries['country']=='New Zealand']

df_NZ=pd.concat([df_NZ_exist,df_NZ_new])

df_NZ


# Now we interpulate the data for each record type

# In[ ]:


inter_df=df_NZ[df_NZ['country']=='A']
inter_df
df_NZ[df_NZ['record']=='EFImportsPerCap']


# In[ ]:


inter_df=df_NZ[df_NZ['country']=='A']

for r in df_NZ['record'].unique():
    inter=df_NZ[df_NZ['record']==r].interpolate()
    inter_df=pd.concat([inter_df,inter[inter['year']==2014]])
    
inter_df


# Then we concatinate the new values for 2014 into our working df

# In[ ]:


df_all_countries = pd.concat([df_all_countries,inter_df])
df_all_countries
df_all_countries[df_all_countries['country']=='New Zealand']


# What we see is that when the NaN values are at the end of our data series, the interpulation function takes the last known value and duplicates it through the NaNs.<br>
# In hindsite, it would have been easier (and faster, but less educational ;-) ) just to slice out the data for NZ in 2013, replace the year to 2014 and concatinate it with our working DF. (see code below)

# In[ ]:


##slicing out NZ data from out df_io, then concatinating to df_io with year 2014
#df_NZ_new= (df_all_countries
#                     .loc[df_all_countries['country']=='New Zealand'] 
#                     .loc[df_all_countries['year']==2013]
#            )     
#df_NZ_new['year']=2014
#df_all_countries = pd.concat([df_all_countries,inter_df])
#df_all_countries
#df_all_countries[df_all_countries['country']=='New Zealand']


# we run year range again

# In[ ]:


data_year_range1=(df_all_countries
                                .groupby           ('country')['year']
                                .agg               ([np.min,np.max, year_range,'nunique'])
                                .sort_values       (['year_range'], ascending = False)
                 )
data_year_range1.loc['New Zealand']


# We decided to look at the data in a 40 year range, between 1975 and 2014. <br>
# We create a list of all the countries we want to drop, according to the data_year_range DF we created earlier

# In[ ]:


drop_countries=sorted(list(data_year_range1[(data_year_range1.amin>1973) | (data_year_range1.amax!=2014)].index))
drop_countries


# In[ ]:


df_all_clean= df_all_countries.drop(df_all_countries[df_all_countries.country.isin(drop_countries)].index)
#since some the remaining countries have data for years before 1975, we also drop those years from our working DF
df_all_clean.drop(df_all_clean[df_all_clean['year']<1975].index,inplace = True)    
df_all_clean.head()


# ## Analisys

# ### Regions

# We look at how our "Clean World" is devided into regions and sub-regions

# In[ ]:


df_all_clean.info()


# In[ ]:


(df_all_clean
             .groupby('UN_region')['country']
             .nunique()
             .sort_values(ascending=False)
)


# In[ ]:


(df_all_clean
             .groupby('UN_region')['country']
             .nunique()
             .plot
             .pie()
)


# In[ ]:


(df_all_clean
             .groupby('UN_subregion')['country']
             .nunique()
             .sort_values(ascending=False)
)


# We want to calculate the BC and EF for each region. <br> 
# For this we will use 2 pivot tables to sum the BC and EF total GHA and population for each region and year.

# In[ ]:


pt = (pd
        .pivot_table(df_all_clean,values = 'total',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')[['BiocapTotGHA','EFConsTotGHA']]
        .reset_index()
        .set_index('UN_region')
     )
pt2=(pd.pivot_table(df_all_clean,values = 'population',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')[['BiocapTotGHA']]
        .rename(index=str, columns={'BiocapTotGHA': 'population'})
        .reset_index()
        .drop(['year'],axis=1)
        .set_index('UN_region')
     )  
    


# Then we concatenate the 2 tables and create 2 new columns, calculateing BC and EF per capita of the region

# In[ ]:


result_pt = pd.concat([pt, pt2], axis=1, join='inner')
result_pt['BiocapPerCap_region']=result_pt['BiocapTotGHA']/result_pt['population']
result_pt['EFConsPerCap_region']=result_pt['EFConsTotGHA']/result_pt['population']
result_pt2 = (result_pt[['year','BiocapPerCap_region','EFConsPerCap_region']]
                .reset_index()
                .set_index(['UN_region','year'])
            )
result_pt2


# Now we will plot the 2 data series per region, across years, but first, in order to determine the limits of the y axis we look for the max value in the table

# In[ ]:


result_pt2.max()


# for plotting, we go over 2 indeces:
# - i = counter for UN_regions by index num, setting the ax location in the figure. 
# - k = goes over UN_regions by value, for the axes titles

# In[ ]:


N = 3
fig = plt.figure(figsize=(15, 15))
for i,k in enumerate(df_all_clean.UN_region.unique()):   
    ax_num = fig.add_subplot(N, N, i+1)
    ax_num.set_title (k)
    ax_num.set_ylim ((0,20))
    result_pt2.loc[k].plot(ax=ax_num)
    
    
fig.tight_layout()
plt.show()


# ## World

# Let's see how are doing as a whole

# In[ ]:


world.head()


# In[ ]:


pt_w_CAP = pd.pivot_table(world,values = 'total',index=['year'],columns=['record'],aggfunc='sum')[['BiocapPerCap','EFConsPerCap']]
(pt_w_CAP.plot()
)


# Sadly, we see that the ability of the world's biological resources to renew themselves have decreased dramatically in the last 50 years, while the EF of the world's population has not changed in the last 40 years, in a way that tries to sustain the planet for future genrations.
# 
# We will look specifically at the EF GHA, by land types, to try to determine which segments affects the general EF calculations.<br>
# We will use an area plot for that.

# In[ ]:


pt_w_EF_by_land=(world[world['record']=='EFConsTotGHA'][['year']+land_types[:-1]]
                    .set_index(['year'])
                )

pt_w_EF_by_land.plot.area(figsize=(12, 12))


# There's an increase in all land types along the years, but we can see that carbon emmisions has the mose significant growth, therefor we should all walk from one place to another, instead of driving ;-)
