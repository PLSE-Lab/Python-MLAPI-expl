#!/usr/bin/env python
# coding: utf-8

# ### Title: Data Science for Good: Kiva Crowdfunding
# ### Date: Mar. 2nd 2018
# ### Background:
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. The mission is to connect people through lending to alleviate proverty. Kiva is in 83 countries, with about 2.7 Million borrowers. Kiva has funded around 1.11 Billion USD worth of loans. It also has around 450 volunteers worldwide.
# 
# ### Objective:
# For the locations in which Kiva has active loans, the objective is to pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# 
# 
# ### Approach:
# 0. Import libraries
# 1. Load & explore data
# 2. Make hypothesis
# 3. placeholder
# 4. placeholder
# 
# ### Conclusions:

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set()
sns.set(rc={'figure.figsize':(15,15)})
sns.set_palette(sns.cubehelix_palette(10, start=1.7, reverse=True))
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from ggplot import *
import os


# In[ ]:


os.listdir('../input')


# In[5]:


loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
theme_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')


# In[ ]:


loans.head()


# In[ ]:


loans.shape, loans.columns.values


# The first data set `loans` contains 671205 loan data for each loan transaction. The same borrower may borrow multiple times for different loan_ids.

# In[ ]:


loans.describe()


# summary of numerical columns shows that
# 1. id ranges from 653047 ~ 1340339
# 2. funded_amount/loan_amount ranges from 0/25 to 1000000
# 3. term in months range from 1 once a month to 158 times per month
# 4. lender_cn ranges from 0 to 2986 pls
# 
# __Note:__ for the first glance, I thought 'partner' means borrower, which is the identity we intend to research on their welfare level. To verify, I group 'partner ID' with 'country', and it turns out some partners are associated with more than one country. Later I realize that 'partner' means __field partner__, instead of borrower. So hold on further analysis.

# In[ ]:


loans.describe(include=['O'])


# Summary of categorical data:
#     1. the top activity is for Farming
#     2. out of 671k, there are only 15 unique sectors. The top one is Agriculture
#     3. 'use' has very high unique ratio as it is text-heavy -> suitable for text-mining
#     4. There are 87 countries but 86 country_codes. there must be a few messy data. 
#     5. There are 67 unique currencies, less than #countries -> shared currencies among some countries
#     6. *time should be continuous, instead of categorical data
#     7. 'tags', similar to use -> suitable for text-mining
#     8. borrower_genders -> not intuitive to have 11k unique genders -> dirty data or multiple borrowers can be reason
#     9. repayment_interval has only 4 unique values -> looks good and may be good for value_counts

# In[ ]:


locations.head()


# In[ ]:


locations.shape, locations.columns.values


# In[ ]:


locations.describe()


# This location dataset is important because it contains MPI (mulitdimensional poverty index) rating for specific regions. We will need to link our borrowers' data to this one to predict on MPI for our borrowers

# In[ ]:


locations.describe(include=['O'])


# Summary of `locations` dataset:
#     1. the dataset divides the world into 6 world_regions, 102 countirs(ISO/Country Code), and 984 individual locations (apparently some are missing). 
#     2. MPI in this dataset ranges from 0 to 0.744 for 984 non-null data

# In[ ]:


theme_ids.head()


# In[ ]:


theme_ids.shape


# In[ ]:


theme_ids.describe()


# In[ ]:


mergetb = pd.merge(theme_ids[theme_ids.id< 700000], loans[loans.id <700000], how='outer', left_on='id', right_on='id')


# In[ ]:


mergetb.shape


# In[ ]:


mergetb.loc[mergetb['Partner ID'] != mergetb['partner_id']]


# `theme_ids` seems not be a good dataset. It does not contain much new and substantial info, and the data size is not consistent with `loans`, which is the master dataset. -> ignore `theme_ids` for now

# In[ ]:


theme_region.head()


# In[ ]:


theme_region.shape


# In[ ]:


theme_region['Loan Theme ID'].nunique()


# In[ ]:


theme_region['Loan Theme Type'].nunique()


# In[ ]:


loans.info()


# In[ ]:


loans.head()


# In[ ]:


loans['borrower_genders'].value_counts()[0:10]


# 'borrower_genders' does not come with the form that we are expecting clarity and simplicity. So here I am making a function to simplify borrowers into ['Majority Female', 'Majority Male', 'Equal M and F']

# In[ ]:


loans[loans.borrower_genders.isnull()].borrower_genders


# In[ ]:


def process_gender(x):
    if type(x) is float and np.isnan(x):
        return 'nan'
    genders = x.split(",")
    Male_count = [sum(g.strip()=='male' for g in genders)]
    Female_count = [sum(g.strip() == 'female' for g in genders)]
    if Female_count > Male_count:
        return "Majority Female"
    elif Female_count < Male_count:
        return "Majority Male"
    else:
        return "Equal M & F"


# In[ ]:


loans['borrower_genders'] = loans['borrower_genders'].apply(process_gender)


# In[ ]:


loans['borrower_genders'].tail()


# In[ ]:


loans.columns


# In[ ]:


sns.distplot(loans['funded_amount'])


# In[ ]:


sns.regplot(x='loan_amount', y='funded_amount', data=loans)


# In[ ]:


ggplot(loans, aes(x='loan_amount', y='funded_amount'))+geom_point()


# The only continuous numeric data in the `loans` dataset are `funded_amount` and `loan_amount`. To get a feel of their relationship, the scatterplot above shows that the majority of funding requests are approved with the funded amount (__fall onto the diagonal line__), but the rest are on the __buttomright of the plane__, which means for these requests, the funded amount is less than loan amount. But interestingly, __the highest amount request of 100k gets approved__ for the full amount. __loan_amount = 50k or <=10k seem to have lower approval rate__.

# ### Top activity counts: Farming, General Store, Personal Housing Expenses, Food Production/Sales, Agriculture

# In[ ]:


sns.set(rc={'figure.figsize':(15,60)})
sns.countplot(y='activity', data=loans, order=loans['activity'].value_counts().index)


# ### Top activity average funded amount: 
# 1. Landscaping/Gardening: 3640, 2: Renewable Energy Products: 3074, 3. Technology: 2406

# In[ ]:


loans_gb_activity = loans.groupby(by='activity')
activity_rank = pd.DataFrame(loans_gb_activity.funded_amount.mean().sort_values(ascending=False).reset_index())


# In[ ]:


activity_rank.columns = ['activity', 'avg_funded_amount']
activity_rank[0:10]


# In[ ]:


sns.set(rc={'figure.figsize':(15, 60)})
sns.boxplot(x='funded_amount', y='activity', data=loans, order=activity_rank['activity'])


# For better visualization, we filter out __only loans with funded_amount <= 20k.__

# In[ ]:


sns.set(rc={'figure.figsize':(15, 60)})
sns.boxplot(x='funded_amount', y='activity', data=loans[loans.funded_amount<=20000], order=activity_rank['activity'])


# ### Top sector counts: Agriculture, Food, Retail, Services, Personal Use

# In[ ]:


sns.set(rc={'figure.figsize':(8, 8)})
sns.countplot(y='sector', data=loans, order=loans['sector'].value_counts().index)


# ### Top sectors with highest average funded amount: 
# 1. Wholesale:1449, 2. Entertainment: 1232, 3. Clothing: 1063, 4. Construction: 1008, 5. Health: 994

# In[ ]:


loans_gb_sector = loans.groupby(by='sector')
sector_rank = pd.DataFrame(loans_gb_sector.funded_amount.mean().sort_values(ascending=False).reset_index())


# In[ ]:


sector_rank.columns=['sector', 'avg_funded_amount']


# In[ ]:


sector_rank


# In[ ]:


sns.set(rc={'figure.figsize': (16, 16)})
sns.boxplot(x='funded_amount', y='sector', data=loans, order=sector_rank['sector'])


# Similarly, to better observe boxes in the boxplot, we filter out __loans with funded_amount<2000__.

# In[ ]:


sns.boxplot(x='funded_amount', y='sector', data=loans[loans.funded_amount<20000], order=sector_rank['sector'])


# Wondering, what exactly differences between `activity` and `sector`? It seems that there are 160+ unique activity and 15 unique sectors. Intuition tells us that activity can be a subclass of sector.
# So below, I am going to measure the relationship, and further plot quantitative charts as per sector, activity

# In[ ]:


loans_gb_sector_activity = loans.groupby(by=['sector', 'activity'])
loans_gb_sector_activity['funded_amount'].mean()


# In[ ]:


loans_gb_sector['activity'].unique(), loans_gb_sector['activity'].unique().apply(len)


# In[ ]:


sns.set(rc={'figure.figsize':(16, 80)})
fig, axs = plt.subplots(nrows=loans['sector'].nunique())
for i in range(loans['sector'].nunique()):
    sns.boxplot(x='activity', y='funded_amount', data=loans[loans.sector==loans['sector'].unique()[i]], ax=axs[i]).set_title('Funded amount distribution per activity when sector = '+loans['sector'].unique()[i])


# Each of the 15 boxplots is a good visualization showing what are the breakdown activities as per each sector. Although some sectors have more than 15 activities, the plot becomes displeasant. We can still get the idea and able to dig in further when needed. 

# ### Top country counts: Philippines, Kenya, El Salvador, Cambodia, Pakistan

# In[ ]:


sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(y='country', data=loans, order=loans['country'].value_counts().index)


# ### Countries with highest funded_amount:

# In[ ]:


loans_gb_country = loans.groupby(by='country')
avg_funded_amount_per_country = pd.DataFrame(loans_gb_country.funded_amount.mean().sort_values(ascending=False).reset_index())
avg_funded_amount_per_country.columns=['country', 'avg_funded_amount']


# In[ ]:


sns.boxplot(x='funded_amount', y='country', data=loans, order=avg_funded_amount_per_country['country'])


# In[ ]:


loans['country'].nunique(), loans['region'].nunique()


# In[ ]:


loans_gb_country.region.unique(), loans_gb_country.region.unique().apply(len).sum()


# From the above analysis, we can see that for 87 countries that Kiva has operations, there are in total 12821 regions. It won't be efficient to analyze each region nor country. Instead, it might be more effective to example all these locations togethor on a map. I am looking to incorporate geocode to create map plots.
# 
# I took the dataset that Mitchell Reynolds published in https://www.kaggle.com/mithrillion/kiva-challenge-coordinates to match on 'country' and 'region' to get geolocations.

# In[8]:


#geo_locations = pd.read_csv('input/kiva_locations.csv')
geo_locations  = pd.read_csv('../input/kiva-challenge-coordinates/kiva_locations.csv', sep='\t', error_bad_lines=False)


# In[9]:


geo_locations.head()


# In[ ]:


geo_locations.info()


# In[ ]:


geo_locations.groupby(['country']).region.unique().apply(len).sum()


# In[ ]:


loans = loans.merge(geo_locations, how='left', on=['country', 'region'])


# In[ ]:


locations.info()


# In[ ]:


MPI_loc = pd.merge(locations, geo_locations, how='inner', on=['country', 'region'])


# In[ ]:


MPI_loc.info()


# In[ ]:


MPI_loc = MPI_loc[MPI_loc.MPI.notnull()]
MPI_loc = MPI_loc.drop(['lat_x', 'lon'], axis=1)


# In[ ]:


MPI_loc['lat'] = MPI_loc['lat_y']
MPI_loc = MPI_loc.drop(['lat_y'], axis=1)


# In[ ]:


sns.set(rc={'figure.figsize':(8,8)})
sns.distplot(MPI_loc['MPI'])


# In[ ]:


import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


scl = [[1, "rgb(255, 0, 0)"], [0.6, "rgb(0, 255,0)"]]
#[ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
#    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [dict (
    type='scattergeo',
    lon = MPI_loc['lng'] , lat = MPI_loc['lat'] ,
    #marker=['red', 'blue'],
    text = MPI_loc['country'] +', '+MPI_loc['region'],
    mode = 'markers',
    marker = dict(
        size = 8,
        opacity = 0.5,
        reversescale = True,
        autocolorscale = False,
        #symbol = 'square',
        colorscale=scl,
        cmin = 0,
        color = MPI_loc['MPI'],
        cmax = MPI_loc['MPI'].max(),
        colorbar = dict(
            title="MPI colorscale")))]

layout = dict(
    title = 'MPI',
    geo = dict(
        scope='world',
        showland=True,
        landcolor = "rgb(225, 225, 225)",
        subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5)
    )

fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)


# Above map plot shows 110 locations where we have MPI data, and the color indicates MPI level. The reder the poorer the location is.

# In[ ]:


data = [dict (
    type='scattergeo',
    lon = geo_locations['lng'] , lat = geo_locations['lat'] ,
    #marker=['red', 'blue'],
    text = geo_locations['country'] +', '+geo_locations['region'],
    mode = 'markers',
    marker = dict(
        size = 8,
        opacity = 0.5,
        reversescale = True,
        autocolorscale = False,
        #symbol = 'square',
        #colorscale=scl,
        #cmin = 0,
        color = 'green'
        #cmax = MPI_loc['MPI'].max(),
        #colorbar = dict(
        #    title="MPI colorscale")
        ))]
layout = dict(
    title = 'Kiva borrower locations',
    geo = dict(
        scope='world',
        showland=True,
        landcolor = "rgb(225, 225, 225)"
        )
    )


# In[ ]:


fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)


# Much more than 110, the locations where Kiva's borrowers in are more than 12k. We certainly need to collect more data to represent these locations' welfare level.
