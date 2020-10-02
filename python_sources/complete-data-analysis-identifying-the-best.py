#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# This is a Simple Dataset from different facebook marketing campaigns.
# 
# We can identify ideal campaign metrics and parameters for efficient results
# 
# STEPS INVOLVED :
# - sectional data analysis for key features 
# - Identifying the best parameters of each feature for efficient ads and High ROI
# - Final suggestions for the ideal campaign

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Data description

# In[ ]:


df = pd.read_csv('/kaggle/input/facebook-ad-campaign/data.csv')


# In[ ]:


df.shape


# - There are 15 columns and 1143 rows

# In[ ]:


df.info()


# FEATURE DESCRIPTION :
# - ad_id is the id of specific ad set | It is an numerical feature
# - Reporting_start and reporting_end are the start and end dates of the each ad
# - Campaign_id is the id assigned by the ad running company
# - fb_campaign_id is the id assigned by facebook for every ad set
# - age and gender talk about the demographics | It is a categorical feature
# - Interest1, Interest2, Interest3 are the user interests and likes of facebook users who were taregted for the ad
# - Impressiosn are the number of times the ad was shown to the users | 
# - Clicks is the number of time users clicked on the ad 
# - spent is the amount of money spent on each campaign 
# - Totalconversions is the number of users who have clicked the ad and have made a purchase or installed an app
# - approved_conversions tells how many became actual active users 

# In[ ]:


print(list(df.columns))


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# - We have 382 null values in total conversion and approved conversion
# - since total_conversion and approved_conversion are the key features for our data analysis and predictive analysis, we can remove the rows with null values in these two features. 

# In[ ]:


df= df.dropna()


# In[ ]:


df.shape


# - Since ad_id and fb_campaign_id are similar type of data and either one of them is enough to perform our data analysis we can drop one of them.
# - We can drop reporting_start and reporting_end also

# In[ ]:


df = df.drop(['reporting_start','reporting_end','fb_campaign_id'], axis=1)


# In[ ]:


df.head()


# # Preliminary data analysis

# In[ ]:


df.gender.value_counts()


# In[ ]:


df.age.value_counts()


# In[ ]:


df.interest1.value_counts()


# In[ ]:


df.interest2.value_counts()


# In[ ]:


df.interest3.value_counts()


# In[ ]:


sns.countplot(df.age)


# In[ ]:


sns.countplot(df.gender)


# In[ ]:


sns.distplot(df.clicks)


# In[ ]:


sns.distplot(df.spent)


# In[ ]:


sns.countplot(df.approved_conversion)


# # Data Analysis

# In[ ]:


# cost analysis
print('Campaign wise clicks')
print((df.groupby(['campaign_id'])).clicks.sum())
print('-------------------------')

print('Campaign wise amount spent')
print((df.groupby(['campaign_id'])).spent.sum())
print('--------------------------')


print('Campaign wise total conversions')
print((df.groupby(['campaign_id'])).total_conversion.sum())
print('---------------------------')

print('Campaign wise ad count')
print((df.groupby(['campaign_id'])).ad_id.count())
print('===========================')


# In[ ]:


campaign_1178_clicks = 9577
campaign_1178_cost = 16577.159998
campaign_1178_conv = 1050
campaign_1178_adcount = 243
campaign_1178_cpc = (campaign_1178_cost/campaign_1178_clicks)
campaign_1178_cpco = (campaign_1178_cost/campaign_1178_conv)
campaign_1178_cpad = (campaign_1178_cost/campaign_1178_adcount)

print('The cost per click of campaign_1178 is '+ str(campaign_1178_cpc))
print('The cost per conversion of campaign_1178 is '+ str(campaign_1178_cpco))
print('The cost per ad in campaign_1178 is '+ str(campaign_1178_cpad))
print('---------------------------------------------------------------')


campaign_936_clicks = 1984
campaign_936_cost = 2893.369999
campaign_936_conv = 537
campaign_936_adcount = 464
campaign_936_cpc = (campaign_936_cost/campaign_936_clicks)
campaign_936_cpco = (campaign_936_cost/campaign_936_conv)
campaign_936_cpad = (campaign_936_cost/campaign_936_adcount)

print('The cost per click of campaign_936 is '+ str(campaign_936_cpc))
print('The cost per conversion of campaign_936 is '+ str(campaign_936_cpco))
print('The cost per ad in campaign_936 is '+ str(campaign_936_cpad))
print('---------------------------------------------------------------')

campaign_916_clicks = 113
campaign_916_cost = 149.710001
campaign_916_conv = 58
campaign_916_adcount = 54
campaign_916_cpc = (campaign_916_cost/campaign_916_clicks)
campaign_916_cpco = (campaign_916_cost/campaign_916_conv)
campaign_916_cpad = (campaign_916_cost/campaign_916_adcount)

print('The cost per click of campaign_916 is '+ str(campaign_916_cpc))
print('The cost per conversion of campaign_916 is '+ str(campaign_916_cpco))
print('The cost per ad in campaign_916 is '+ str(campaign_916_cpad))
print('---------------------------------------------------------------')


# - From the above analysis it is very clear that campaign 916 is the most efficient and profitable campaign. 
# - The cpc and cost per conversion is very low compared other campaigns.
# - If we scale up the campaign 916 to the budget of campaign 1178 we might drive 5 times more results of campaign 1178

# Now we can create two dataframes, one for campaign 916 and another one for campaign 1178 and we can analyze them further

# In[ ]:


dfn = df.query('campaign_id =="916"')
dfn.head()


# In[ ]:


dfm = df.query('campaign_id =="1178"')
dfm.head()


# In[ ]:


# gender analysis


# In[ ]:


print('Gender based analysis')
print((df.groupby(['gender'])).total_conversion.sum())
print((df.groupby(['gender'])).ad_id.count())
print((dfn.groupby(['gender'])).total_conversion.sum())
print((dfn.groupby(['gender'])).ad_id.count())
print((dfm.groupby(['gender'])).total_conversion.sum())
print((dfm.groupby(['gender'])).ad_id.count())


# - Though we can get more conversion by targeting Males alone, the cost per conversion is high (inferred from campaign 1178 analysis) and the number of ads.
# - If there was to be more ads in campaign 916 the conversions would have been much higher
# - From the above analysis we can conclude that we need to target both male and female. 

# In[ ]:


#age analysis


# In[ ]:


print((df.groupby(['age'])).total_conversion.sum())
print((df.groupby(['age'])).ad_id.count())
print((dfn.groupby(['age'])).total_conversion.sum())
print((dfn.groupby(['age'])).ad_id.count())
print((dfm.groupby(['age'])).total_conversion.sum())
print((dfm.groupby(['age'])).ad_id.count())


# - Though we can get more conversion by targeting 30-34 and 35-39 alone, the cost per conversion is high (inferred from campaign 1178 analysis).
# - The number of ads are also different. If there was to be more ads in campaign 916 the conversions would have been much higher
# - From the above analysis we can conclude that we need to target all four age ranges 

# In[ ]:


# Interests analysis


# In[ ]:


(dfn.groupby(['interest1'])).total_conversion.sum()


# In[ ]:


(dfn.groupby(['interest2'])).total_conversion.sum()


# In[ ]:


(dfn.groupby(['interest3'])).total_conversion.sum()


# In[ ]:


(dfm.groupby(['interest1'])).total_conversion.sum()


# In[ ]:


(dfm.groupby(['interest2'])).total_conversion.sum()


# In[ ]:


(dfm.groupby(['interest3'])).total_conversion.sum()


# # Final Report  

# from the above analysis,
# - The ideal campaign for the most efficient results can be created with the following metrics :
#     - Gender : M | F (Both)
#     - Age : 30 - 49 (Including all four age bands)
#     - Interest1 : 16
#     - Interest2 : 19
#     - Interest3 : 20 
#     
# - We can try out different campaigns by slightly adjusting the interests. The list of possible interests are :
#     - Interest1 : 10 , 15 , 29
#     - Interest2 : 20
#     - Interest3 : 17 , 31 , 33
#  
# we can try differnt combinations of interests using the above lists and drive more efficient results 

# In[ ]:




