#!/usr/bin/env python
# coding: utf-8

# # What are the risk correlated to COVID-19?  
# **US-specific population risk**  
# By: Myrna M Figueroa Lopez
# > Final: Apr 7 2020

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions, stats


# **What is the situation worldwide? Where does the US stand?**

# In[ ]:


#Dataset from the World Health Organization
World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/full_data(14).csv")

plt.figure(figsize=(21,8)) # Figure size
plt.title('Cases across the world as of April 6, 2020') # Title
World.groupby("location")['total_cases'].max().plot(kind='bar', color='teal')


# In[ ]:


World.corr().style.background_gradient(cmap='magma')


# The longest line represents the world's total, not a specific country.   
# If the reported data is 100% correct and properly reported, the **US** has a significantly high total of cases.   
# 

# In[ ]:


df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

#I droped FIPS column. 
##not relevant for this analysis.
USA=df.drop(['fips','county'], axis = 1) 
USA


# In[ ]:


plt.figure(figsize=(19,17))
plt.title('Cases by state') # Title
sns.lineplot(x="date", y="cases", hue="state",data=USA, palette="Paired")
plt.xticks(USA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


##For ease of visualization
NY=USA.loc[USA['state']== 'New York']
LA=USA.loc[USA['state']== 'Louisiana']
WA=USA.loc[USA['state']== 'Washington']
IL=USA.loc[USA['state']== 'Illinois']
Mich=USA.loc[USA['state']== 'Michigan']
PUR=USA.loc[USA['state']== 'Puerto Rico']


# In[ ]:


# Concatenate dataframes 
States=pd.concat([NY,LA,WA,IL,PUR,Mich]) 

States=States.sort_values(by=['date'], ascending=True)
States


# In[ ]:


plt.figure(figsize=(15,9))
plt.title('COVID-19 cases comparison of WA, IL, NY, LA, PR, and Michigan') # Title
sns.lineplot(x="date", y="cases", hue="state",data=States)
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()


# In[ ]:


USAg=USA.groupby(['date']).max()
USAg


# In[ ]:


USAg=USAg.sort_values(by=['cases'], ascending=True)
USAg


# **df on VULNERABILITIES in the US**

# In[ ]:


Vuln = pd.read_csv("../input/uncover/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv")


# In[ ]:


Vuln= Vuln[['state', 'e_uninsur', 'epl_pov','epl_unemp','epl_age65','epl_age17','epl_disabl']]


# In[ ]:


# converting and overwriting values in column 
Vuln["state"]=Vuln["state"].str.lower()
Vuln["state"]=Vuln["state"].str.title()


# In[ ]:


Vuln.head()


# In[ ]:


Vuln.describe()


# In[ ]:


Vuln.corr().style.background_gradient(cmap='viridis')


# df on **illness prevalence**

# In[ ]:


census = pd.read_csv("../input/uncover/us_cdc/us_cdc/500-cities-census-tract-level-data-gis-friendly-format-2019-release.csv")


# In[ ]:


census=census[['stateabbr','placename', 'geolocation', 'bphigh_crudeprev',
               'stroke_crudeprev', 'obesity_crudeprev', 'diabetes_crudeprev','arthritis_crudeprev',
               'cancer_crudeprev', 'casthma_crudeprev', 'copd_crudeprev', 'csmoking_crudeprev', 
               'highchol_crudeprev', 'kidney_crudeprev']]
census


# In[ ]:


#COPD prevalence
plt.figure(figsize=(19,7)) # Figure size
census.groupby("stateabbr")['copd_crudeprev'].max().plot(kind='bar', color='olive')


# In[ ]:


census=census.replace(to_replace =("ND","OK", "UT", 'AK', 'SD','AL','AR'),
                 value =("North Dakota", "Oklahoma", 'Utah', "Alaska", "South Dakota", "Alabama", "Arkansas"))


# In[ ]:


census=census.replace(to_replace =("NC","OR", "NV", 'AZ', 'SC','CA','CO'),
                 value =("North Carolina", "Oregon", 'Nevada', "Arizona", "South Carolina", "California", "Colorado"))


# In[ ]:


census=census.replace(to_replace =("MN","WY", "WV", 'WI', 'WA','VT','VA'),
                 value =("Minnessota", "Wyoming", 'West Virginia', "Wisconsin", "Washington", "Vermont", "Virginia"))


# In[ ]:


census=census.replace(to_replace =("FL","NE", "MT", 'HI', 'LA','NM','GA','KS'),
                 value =("Florida", "Nebraska", 'Montana', "Hawaii", "Louisiana", "New Mexico", "Georgia", "Kansas"))


# In[ ]:


census=census.replace(to_replace =("NY","NJ", "OH", 'RI', 'PA','TX','ID','KY'),
                 value =("New York", "New Jersey", 'Ohio', "Rhode Island", "Pennsylvania", "Texas", "Idaho", "Kentucky"))


# In[ ]:


census=census.replace(to_replace =("CT","DC", "DE", 'IA', 'IL','IN','MD','MA'),
                 value =("Connecticut", "District of Columbia", 'Delaware', "Iowa", "Illinios", "Indiana", "Maryland", "Massachussetts"))


# In[ ]:


census=census.replace(to_replace =("ME","MI", "MO", 'MS', 'TN'),
                 value =("Maine", "Michigan", 'Missouri', "Mississippi", "Tennessee"))


# In[ ]:


#arthritis prevalence
plt.figure(figsize=(19,7)) # Figure size
census.groupby("stateabbr")['arthritis_crudeprev'].max().plot(kind='bar', color='peru')


# In[ ]:


census=census.drop(['placename', 'geolocation'], axis = 1) 
census = census.rename(columns={'stateabbr': 'state'})


# In[ ]:


census = census.rename(columns={'bphigh_crudeprev': 'high bp prev', 'stroke_crudeprev': 'stroke prev'})


# In[ ]:


census=census.rename(columns={'diabetes_crudeprev': 'diabetes prev', 'cancer_crudeprev': 'cancer prev', 'arthritis_crudeprev': 'arthritis prev'})


# In[ ]:


census=census.rename(columns={'casthma_crudeprev': 'asthma prev', 'copd_crudeprev': 'copd prev', 'csmoking_crudeprev': 'smoking prev'})


# In[ ]:


census=census.rename(columns={'highchol_crudeprev': 'highChol prev', 'kidney_crudeprev': 'kidney prev'})
census


# In[ ]:


census.describe()


# In[ ]:


census.corr().style.background_gradient(cmap='cividis')


# **df on chronic illnesses in the US**

# In[ ]:


chronic = pd.read_csv("../input/uncover/us_cdc/us_cdc/u-s-chronic-disease-indicators-cdi.csv")


# In[ ]:


# iterating the columns 
for col in chronic.columns: 
    print(col)


# In[ ]:


chronic=chronic[['locationdesc','topic','question','datavalue']]
#replace NaNs with zeros in the df
chronic=chronic.fillna(0)


# In[ ]:


chronic = chronic.rename(columns={'locationdesc': 'state','datavalue': 'rate of illness','topic': 'chronic illness','question': 'specific illness'})


# In[ ]:


chronic.head(3)


# In[ ]:


plt.figure(figsize=(22,6)) # Figure size
plt.title('US chronic illnesses') # Title
sns.countplot(chronic['chronic illness'])
plt.xticks(rotation=45)


# In[ ]:


chronic.describe()


# In[ ]:


chronic.corr().style.background_gradient(cmap='cool')


# **df of illness ranking**

# In[ ]:


rank = pd.read_csv("../input/uncover/county_health_rankings/county_health_rankings/us-county-health-rankings-2020.csv")


# In[ ]:


rank=rank[['state','num_deaths', 'percent_female','percent_excessive_drinking', 
           'num_uninsured','percent_vaccinated','percent_black','percent_american_indian_alaska_native',
           'percent_asian', 'percent_native_hawaiian_other_pacific_islander', 'percent_hispanic', 
           'percent_non_hispanic_white']]
rank.head()


# In[ ]:


plt.figure(figsize=(16,8)) # Figure size
plt.title('States pre-COVID19 morbidity ranks') # Title
rank.groupby("state")['num_deaths'].max().plot(kind='bar', color='darkred')


# In[ ]:


rank.describe()


# In[ ]:


rank.corr().style.background_gradient(cmap='inferno')


# **df on COVID-19 Statistics**

# In[ ]:


stats = pd.read_csv("../input/uncover/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv")
#replace NaNs with zeros in the df
stats=stats.fillna(0)


# In[ ]:


# iterating the columns 
for col in stats.columns: 
    print(col)


# In[ ]:


stats.drop(['hash', 'fips', 'datechecked'], axis=1, inplace=True)
stats.head()


# In[ ]:


plt.figure(figsize=(14,8)) # Figure size
plt.title('total tests') # Title
stats.groupby("state")['totaltestresults'].max().plot(kind='bar', color='steelblue')


# In[ ]:


stats=stats[['date', 'state','positive','negative','hospitalized', 'death']]
stats.head()


# In[ ]:


stats=stats.replace(to_replace ="WA",
                 value ="Washington")


# In[ ]:


stats=stats.replace(to_replace ="SC", 
                 value ="South Carolina")


# In[ ]:


stats=stats.replace(to_replace =("NJ","FL", 'AL', "TX", "OR"),
                 value =("New Jersey", "Florida", "Alabama", "Texas", "Oregon"))


# In[ ]:


stats=stats.replace(to_replace =("AR","AZ", "NY", "CA", "AK"),
                 value =("Arkansas", "Arizona", 'New York', "California", "Alaska"))


# In[ ]:


stats=stats.replace(to_replace =("MT","WI", "NC", 'OH',"RI", "VA"),
                 value =("Montana", "Wisconsin", 'North Carolina','Ohio', "Rhode Island", 'Virginia'))


# In[ ]:


stats=stats.replace(to_replace =("TN","GA", "IL", 'NH', "MA"),
                 value =("Tennessee", "Georgia", 'Illinios', "New Hampshire", "Massachussetts"))


# In[ ]:


stats=stats.replace(to_replace =("CO","CT", "DC", 'DE', "GU"),
                 value =("Colorado", "Connecticut", 'District of Columbia', "Delaware", "Guam"))


# In[ ]:


stats=stats.replace(to_replace =("HI","IA", "ID", 'IN', "KS", 'KY'),
                 value =("Hawaii", "Iowa", 'Idaho', "Indiana", "Kansas", "Kentucky"))


# In[ ]:


stats=stats.replace(to_replace =("LA","MD", "MN", 'MI', "MO", 'MS'),
                 value =("Louisiana", "Maryland", 'Minnessota', "Michigan", "Missouri", "Missippippi"))


# In[ ]:


stats=stats.replace(to_replace =("ME","NV", "WV", 'NM', 'PA', "VT"),
                 value =("Maine", "Nevada", 'West Virginia', "New Mexico", "Pennsylvania", "Vermont"))


# In[ ]:


stats=stats.replace(to_replace =("ND","OK", "UT", 'PR', 'SD'),
                 value =("North Dakota", "Oklahoma", 'Utah', "Puerto Rico", "South Dakota"))


# In[ ]:


stats=stats.replace(to_replace =("VI","WY", "NE"),
                 value =("Virgin Islands", "Wyoming", "Nebraska"))


# In[ ]:


stats.head(3)


# In[ ]:


stats.describe()


# In[ ]:


stats.corr().style.background_gradient(cmap='plasma')


# # What could be the risks across the US?   
# In combining some of the dataframes provided by ROCHE, I visualize below some factors along with #COVID19 data.   
# Further statistical analysis would be needed to reach scientific conclusion in this data.   
# However, the presentation here could help in identifying future research angles relating to risk factors and   
# COVID-19.   

# In[ ]:


# Merging the dataframes                       
a=pd.merge(USA, stats, how ='inner', on =('state', "date"))
a


# In[ ]:


dfs1=pd.concat([a,rank,chronic], sort=True) 
dfs1.head()


# In[ ]:


# Merging the dataframes                       
b=pd.concat([dfs1, Vuln], sort=False) 


# In[ ]:


# Merging the dataframes                       
c=pd.concat([b, census], sort=False) 


# In[ ]:


#replace NaNs with zeros in the df
c=c.fillna(0)
c.head()


# In[ ]:


# iterating the columns to list their names
for col in c.columns: 
    print(col)


# In[ ]:


# Grouped df by date and state and extract a number of stats from each group
d=c.groupby(
   ['date', 'state'], as_index = False
).agg(
    {
         'hospitalized':max,    # max values 
         'cases':max,
         'deaths': max,
         'num_uninsured':max, 
         'percent_vaccinated': max, 
         'num_uninsured': max,
         'percent_american_indian_alaska_native':max,        
         'percent_asian':max,
         'percent_black':max,        
        'percent_excessive_drinking':max,
        'percent_female':max,
        'percent_hispanic':max,
        'percent_native_hawaiian_other_pacific_islander':max,
        'percent_non_hispanic_white':max,
        'epl_pov':max,
        'epl_unemp': max,
        'epl_age65':max,
        'epl_age17':max,
        'epl_disabl':max,
        'high bp prev':max,
        'stroke prev':max,
        'obesity_crudeprev':max,
        'diabetes prev':max,
        'arthritis prev':max,
        'cancer prev':max,
        'asthma prev':max,
        'copd prev':max,
        'smoking prev':max,
        'highChol prev':max,
        'kidney prev':max
         
    }
)
d


# In[ ]:


sub1=d[d.date==0]
sub2=d[d.date!=0]


# In[ ]:


sub2=sub2[['state', 'cases', 'deaths', 'hospitalized']]
sub2.head()


# In[ ]:


# Merging the dataframes                       
risks=pd.merge(sub1, sub2, how ='inner', on ='state')
risks=risks.drop(['date'], axis = 1) 


# In[ ]:


sum_column = risks["hospitalized_x"] + risks["hospitalized_y"]
risks["hospitalized"] = sum_column


# In[ ]:


risks=risks.drop(['hospitalized_x','hospitalized_y'], axis = 1) 


# In[ ]:


sum_column2 = risks["cases_x"] + risks["cases_y"]
risks["cases"] = sum_column2
sum_column3 = risks["deaths_x"] + risks["deaths_y"]
risks["deaths"] = sum_column3


# In[ ]:


risks=risks.drop(['cases_x','cases_y', 'deaths_x','deaths_y'], axis = 1) 
risks


# In[ ]:


# Grouped df by date and state and extract a number of stats from each group
r=risks.groupby(
   ['state'], as_index = False).agg(    
    {
         'hospitalized':max,    # max values 
         'cases':max,
         'deaths': max,
         'num_uninsured':max, 
         'percent_vaccinated': max, 
         'num_uninsured': max,
         'percent_american_indian_alaska_native':max,        
         'percent_asian':max,
         'percent_black':max,        
        'percent_excessive_drinking':max,
        'percent_female':max,
        'percent_hispanic':max,
        'percent_native_hawaiian_other_pacific_islander':max,
        'percent_non_hispanic_white':max,
        'epl_pov':max,
        'epl_unemp': max,
        'epl_age65':max,
        'epl_age17':max,
        'epl_disabl':max,
        'high bp prev':max,
        'stroke prev':max,
        'obesity_crudeprev':max,
        'diabetes prev':max,
        'arthritis prev':max,
        'cancer prev':max,
        'asthma prev':max,
        'copd prev':max,
        'smoking prev':max,
        'highChol prev':max,
        'kidney prev':max
         
    }
)

r


# In[ ]:


r.describe()


# In[ ]:


r.corr().style.background_gradient(cmap='cubehelix')


# While not verified, there could be correlation among risk factors presented above.   
# For example, there seems to be correlation between deaths, cases, and hospitalization.      
# However, a statistically-sound correlation does not mean causation.   
