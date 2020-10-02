#!/usr/bin/env python
# coding: utf-8

# # Melb Airbnb Data Exploratory Analysis and Cleansing
# 
# This kernel shows some simple steps to do data cleansing with a quick peek of the data, using the Melbourne Airbnb Open Data. 
# 
# A cleansed dataset will be provided after this EDA for a more sophisticated analysis and dashboard.
# 
# # Table of contents
# 
# * [0. Loading libraries and data](#0.-Loading-libraries-and-data)
# * [1. Columns with null data](#1.-Columns-with-null-data)
# * [2. Check duplicated columns](#2.-Check-duplicated-columns)
# * [3. Check if column names make sense](#3.-Check--if-column-names-make-sense)
# * [4. Data quality](#4.-Data-quality)
#   * [4.1. Column *market*](#4.1.-Column-market)
#   * [4.2. Columns *weekly_price* and *monthly_price*](#4.2.-Columns-weekly_price-and-monthly_price)
#   * [4.3. Column *license*](#4.3.-Column-license)
#   * [4.4. Column *is_business_travel_ready*](#4.4.-Column-is_business_travel_ready)
# * [5. The dataset after cleasning](#5.-The-dataset-after-cleasning)
# 
# # 0. Loading libraries and data

# Click the ***output*** to see what files in the dataset. I will use the ***listings_dec18*** csv file.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Dataset shape:

# In[ ]:


lis = pd.read_csv('../input/listings_dec18.csv',low_memory=False)  # Using the listings_dec18.csv file to cleanse
print(lis.shape)


# Click the ***output*** to see what columns in our data.

# In[ ]:


lis.info()


# # 1. Columns with null data
# 
# Looking into the data preview tab on the dataset page is a good start to see which part of the data needs cleansing.
# 
# For example, columns ***thumbnail_url***, ***medium_url***, ***jurisdiction_names***, etc. have 100% null value. So let's drop these columns.
# 
# We can also see how many valid values in each column.
# 
# For example, columns ***square_feet*** only has 214 non-null valus. So let's drop it.

# # 2. Check duplicated columns
# 
#  Columns ***host_listings_count*** and ***host_total_listings_count***  are almost the same, as there are only 3 different records, which are NaN.
# 

# In[ ]:


print(sum(lis.host_listings_count !=lis.host_total_listings_count),"records in column host_listings_count are different from host_total_listings_count:")
print(list(lis[lis.host_listings_count!=lis.host_total_listings_count].host_total_listings_count))
#print(sum(lis.calculated_host_listings_count !=lis.host_total_listings_count),"records in column calculated_host_listings_count are different from host_total_listings_count.")


# While column ***calculated_host_listings_count*** are gotten by calculating how many times a host appears in the whole dataset. See the code below:

# In[ ]:


calculated_counts=lis.groupby(['host_id']).size().reset_index(name='calculated_num_listings')
calculated_counts.head()


# So let's use the ***calculated_host_listings_count*** and get rid of the other two.

# # 3. Check if column names make sense
# 
# After checked with the info on the City of Melbourne website, I found that the data in column *neighbourhood* is actually city names, and the data in column ***city*** is actually suburbs/inner suburbs, so let's change the column names ***neighbourhood*** to ***city***, and ***city*** to ***suburb***.

# # 4. Data quality
# 
# Now let's have a closer look on the data. Specifically, let's look at the columns ***market***, ***weekly_price*** and ***monthly_price***, ***license***, and ***is_business_travel_ready***.

# # 4.1. Column *market*
# 
# From the dataset preview, we see that there are 1% showing as 'Other' other than Melb and Other(International).
# We can see that from the unique values below,  'Phillip Island and Mornington Peninsula' and some others should be marked as "Melbourne".

# In[ ]:


print('Unique values in the market column:',"\n",lis.market.unique())


# Next, let's see the 'Guangzhou' one:

# In[ ]:


lis[lis.market=='Guangzhou']


# After looking into the listing ad and the host profile, it's clearly that this listing is in the Melb market.
# 
# In fact, all the listings scaped are all in the Melbourne market. So this ***market*** column doesn't have much meanings. Let's drop it.

# # 4.2. Columns *weekly_price* and *monthly_price*

# We see that 89% of ***weekly_price*** and 92% of ***monthly_price*** are NaN. This is because most hosts only set nightly prices and don't bother to set longer term prices. So we still keep these columns.

# # 4.3. Column *license*
# 
# We see that there are only a few hosts have licenses:

# In[ ]:


print('License:','\n',lis[(lis.license.notnull())&(lis.license != "GST")].license)


# Let's see who has the License 35753401805:

# In[ ]:


print('Host name of the three properties with license 35753401805:',list(lis[lis.license=='35753401805'].host_name))


# So the three listings with the same license are from the same host, which makes sense.
# And looks like only 4 hosts have license in Melb. (Host Lyn has two licenses.) That's great.  Although not sure what is this license for.

# In[ ]:


lis[(lis.license.notnull())&(lis.license != "GST")][['host_name','host_id','license']].drop_duplicates().sort_values(['host_name'])


# # 4.4. Column *is_business_travel_ready*
# 
# 
# 

# Column ***is_business_travel_ready*** shows all false, which is not the case as at least my listings are business travel ready (yes, I am an Airbnb host!). So drop this column.
# 

# 

# # 5. The dataset after cleasning
# 
# So after this preliminary cleansing, a cleansed data file is created without the following columns:
# * ***thumbnail_url***,
# * ***medium_url***,  
# * ***xl_picture_url***,
# * ***host_acceptance_rate***,
# * ***host_listings_count***, 
# * ***host_total_listings_count***, 
# * ***experiences_offered***, 
# * ***neighbourhood_group_cleansed***, 
# * ***market***,
# * ***is_business_travel_ready***,
# * ***jurisdiction_names***,
# 
# also changing the column names ***neighbourhood_cleansed*** to ***city***, and the original ***city*** column to ***suburb***, and spelling like 'neighbour' to 'neighbor'.

# 
