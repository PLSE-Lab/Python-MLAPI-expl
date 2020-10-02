#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


london=bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='london_crime')


# In[3]:


london.list_tables()


# In[4]:


london.head('crime_by_lsoa')


# In[5]:


borough_count="""select distinct borough from `bigquery-public-data.london_crime.crime_by_lsoa`;"""
boroughs=london.query_to_pandas_safe(borough_count)
boroughs


# In[6]:


lsoa_per_borough="""select borough,count(distinct lsoa_code) from `bigquery-public-data.london_crime.crime_by_lsoa` 
group by borough order by count(distinct lsoa_code) desc"""
count_per_borough=london.query_to_pandas_safe(lsoa_per_borough)
count_per_borough


# In[7]:


value_per_borough="""select borough,sum(value) from `bigquery-public-data.london_crime.crime_by_lsoa` 
group by borough order by sum(value) desc"""
val_per_borough=london.query_to_pandas_safe(value_per_borough)
val_per_borough


# 1. The above query was written to assess the severity of the crime per borough and with this we will be able to analyze which borough contributes to the overall crime in london.
# Next up it will be appropriate to check the categories that contribute to the crime in the city of london.

# In[8]:


box="""select sum(case when major_category='Burglary' then value else 0 end) as Burglary,
sum(case when major_category='Criminal Damage' then value else 0 end) as Criminal_damage,
sum(case when major_category='Drugs' then value else 0 end)as Drugs,
sum(case when major_category='Other Notifiable Offences' then value else 0 end) as Offences,
sum(case when major_category='Robbery' then value else 0 end) as Robbery,
sum(case when major_category='Theft and Handling' then value else 0 end) as Theft,
sum(case when major_category='Sexual Offences' then value else 0 end) as Sexual_offences,
sum(case when major_category='Fraud or Forgery' then value else 0 end) as fraud 
from `bigquery-public-data.london_crime.crime_by_lsoa` group by borough"""
box1=london.query_to_pandas_safe(box)
box1.columns
box1.boxplot(column=['Burglary', 'Criminal_damage', 'Drugs', 'Offences', 'Robbery', 'Theft', 'Sexual_offences', 'fraud'],rot=90,grid=False,)


# In[9]:


pivot="""select borough,sum(case when year=2008 then value else 0 end)as Yr_2008,
sum(case when year=2009 then value else 0 end)as Yr_2009,
sum(case when year=2010 then value else 0 end)as Yr_2010,
sum(case when year=2011 then value else 0 end)as Yr_2011,
sum(case when year=2012 then value else 0 end)as Yr_2012,
sum(case when year=2013 then value else 0 end)as Yr_2013,
sum(case when year=2014 then value else 0 end)as Yr_2014,
sum(case when year=2015 then value else 0 end)as Yr_2015,
sum(case when year=2016 then value else 0 end)as Yr_2016
from `bigquery-public-data.london_crime.crime_by_lsoa`
group by borough order by borough"""
pv=london.query_to_pandas_safe(pivot)
pv


# Westminster seem to have highest crime value and City of London has the minimum crime, but the crime is increasing since year 2008 to 2016. It seems intuitive to check which crime is happening in London which is increasing and can be curbed.

# In[10]:


pivot1="""select major_category,sum(case when year=2008 then value else 0 end)as Yr_2008,
sum(case when year=2009 then value else 0 end)as Yr_2009,
sum(case when year=2010 then value else 0 end)as Yr_2010,
sum(case when year=2011 then value else 0 end)as Yr_2011,
sum(case when year=2012 then value else 0 end)as Yr_2012,
sum(case when year=2013 then value else 0 end)as Yr_2013,
sum(case when year=2014 then value else 0 end)as Yr_2014,
sum(case when year=2015 then value else 0 end)as Yr_2015,
sum(case when year=2016 then value else 0 end)as Yr_2016
from `bigquery-public-data.london_crime.crime_by_lsoa`
where borough='City of London'
group by major_category order by major_category """
pv1=london.query_to_pandas_safe(pivot1)
pv1


# Violence against the person seems to be increasing in City of London year-wise. Theft and Handling consistently is on a high and contributes to the maximum crime in this borough.

# In[61]:


rq="""select * from (select bc,mj_c,dense_rank() over(partition by bc order by val desc) as rk from (select borough as bc,major_category as mj_c,sum(value) as val
from `bigquery-public-data.london_crime.crime_by_lsoa` group by borough,major_category)) where rk=1"""
rq1=london.query_to_pandas_safe(rq)
rq1


# Here I am able to tell which crime is predominant and what severity does each crime take in respective boroughs.

# In[62]:


import seaborn as sns; sns.set()
tim="""select borough,month,sum(value) as val,row_number() over(partition by borough order by sum(value) desc)  
from `bigquery-public-data.london_crime.crime_by_lsoa` 
where borough in 
(select bor_name from (select borough as bor_name,dense_rank() over(order by sum(value)) as rk 
from `bigquery-public-data.london_crime.crime_by_lsoa` group by borough) where rk<6) and major_category='Theft and Handling'
group by borough,month"""
tim1=london.query_to_pandas_safe(tim)
tim1
ax = sns.lineplot(x="month", y="val", hue="borough",data=tim1)


# 
