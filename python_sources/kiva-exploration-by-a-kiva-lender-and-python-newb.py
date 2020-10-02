#!/usr/bin/env python
# coding: utf-8

# In[320]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates

sns.set_style('darkgrid') # looks cool, man
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ![](https://i.imgflip.com/13cqor.jpg)
# <center>**Warning:**  Newb alert, dirty python incoming</center>
# 
# * [1. Introduction and My Personal Experience](#intro)
#   * [1.1 Approach (Data and Visual)](#approach)
#   * [1.2 Unexplored Themes](#unexplored)
# * [2. Data Prep](#dataprep)
#   * [2.1 Create Base Superset](#data_createset)
#   * [2.2 Create New Fields](#data_createnew)
#   * [2.3 Update Fields](#data_update)
#   * [2.4 Bad Seeming Data](#data_bad)
#   * [2.5 Completed Data](#data_complete)
# * [3. Distributions and Contributions](#dist)
#   * [3.1 Distribution of Loan Amount](#dist_loan)
#   * [3.2 Distribution of Funded Amount](#dist_fund)
#   * [3.3 Average Kiva Member Contribution](#avg_cont)
# * [4. Top Sectors and Activities](#tsa)
#   * [4.1 Top Sectors](#ts)
#   * [4.2 Absolute Top 30 Overall Activities, by Sector](#act30)
#   * [4.3 Funding Speed by World Region by Sector](#speed_wr)
#   * [4.4 Funding Speed by Group Type by Sector](#speed_gt)
# * [5. Loan Count by Gender and Group](#group)
# * [6. What's #Trending Kiva?](#trend)
# * [7. Loan Theme Types](#themes)
# * [8. Exploring Currency](#curr)
#   * [8.1 Currency Usage](#curr_usg)
#   * [8.2 Mean Loan by Currency for Top 20 Currencies](#curr_avg)
#   * [8.3 What's going on in Lebanon?](#curr_leb)
#   * [8.4 Lebanese Field Partners](#leb_fld)
# * [9. Bullet Loans for Agriculture](#bullet_agg)
#   * [9.1 El Salvador Investigation by Activity](#sal1)
#   * [9.2 El Salvador Loan Count Over Time](#sal2)
#   * [9.3 El Salvador Animal Loans](#sal3)
# * [10. Is the Philippines Really the Country with the Most Kiva Activity?](#phil)
# 

# <a id=intro>
# # 1. Introduction and My Personal Experience 
# 
# Kiva is a non-profit micro-funding/loan capitalization website.  Users from around the world come to lend money to those in need around the world.  Often these loans are for small entrepeneurs.  Kiva lenders receive no return on capital and are subject to loan defaults and currency exchange losses.  Loans are distributed by Kiva field partners to borrowers to improve their lives and help facilitiate their growth out of poverty.  Kiva field partner lenders do charge a local market interest rate.  Kiva funders insure capital is available and loans have funding to be made.  The default rate for these loans is very low in comparison [to other default rates](https://www.federalreserve.gov/releases/chargeoff/delallsa.htm).  If you have never used it I [invite you to give it a try!](https://www.kiva.org/invitedby/mikedev10)  17.8% of my loans were to the Philippines, I worked there a few weeks in 2003 and have worked from there in January 2016/17/18 ditching the Chicago winter.  Kiva users will find the Philippines and women come up often in the data, likely even more often in my own as I made a conscious effort to lend to both.  For a long time I only lent to women.  I also like groups.  I lend heavily more towards business type use vs. personal investment or personal use.
# 
# ![](https://www.doyouevendata.com/wp-content/uploads/2018/03/kiva.jpg)

# <a id=approach></a>
# ## 1.1 Approach (Data and Visual) 
# 
# I'm willing to bet my approach is not the best in regards to best practices in working with data, probably in part due to memory usage.  However I'm looking to make my life easier here and the dataset all fits, so we're going to roll with it.  I've attempted to tie kiva provided data together as best I can, along with additional MPI data to get a richer view of the areas and how they experience poverty, to create one large set to work with.  Seaborn produces a lot of pretty color plots, but I will *only be leveraging color when it has meaning* so as to avoid confusion.
# <a id=unexplored></a>
# ## 1.2 Unexplored Themes 
# potentially available to explore with additional data:
# 1. relation to education attainment (general)
# 2. relation to education attainment in areas of the world where girls have less rights or the cultural expectation to stay at home while their husband works, regardless of their education attainment
# 3. relation to educational attainment in areas of the world boys are more subject to violence ([Girls in the Middle East do better than boys in school by a greater margin than almost anywhere else in the world: a case study in motivation, mixed messages, and the condition of boys everywhere.](https://www.theatlantic.com/education/archive/2017/09/boys-are-not-defective/540204/))
# 4. relation to loan reporting and and interest with regards to religion (Buddhism - it doesn't seem expected to have some of the kiva reporting strings as part of lending; Islam - interest is not charged in Sharia complaint finance, although perhaps it kind of is, [it's a bit confusing](https://en.wikipedia.org/wiki/Riba).)
# 5. relation with weaker or stronger property rights (perhaps some metrics could be leveraged from Cato's [Human Freedom Index](https://www.cato.org/human-freedom-index))

# <a id=dataprep></a>
# # 2. Data Prep 
# <a id=data_createset></a>
# ## 2.1 Data Prep - Create Base Superset 
# 
# Let's take a look at the data in the sets we've got to work with.
# **MPI Poverty Metrics** (2 external, 1 kiva)

# In[321]:


df_mpi_ntl = pd.read_csv("../input/mpi/MPI_national.csv")
df_mpi_ntl.shape


# In[322]:


df_mpi_ntl[(df_mpi_ntl['ISO'] == 'AFG') | (df_mpi_ntl['ISO'] == 'ARM')]


# In[323]:


df_mpi_subntl = pd.read_csv("../input/mpi/MPI_subnational.csv")
df_mpi_subntl.shape


# In[324]:


df_mpi_subntl.head()


# In[325]:


df_kv_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
df_kv_mpi.shape


# In[326]:


df_kv_mpi.head()


# Why does kiva have so many more records, from the same root datasource?  Well, the file has junk in it.

# In[327]:


df_kv_mpi.tail()


# Let's combine this into a set of useful superset geographic poverty data.  Note the kiva provided MPI is the same as MPI Regional in the richer MPI data.  We'll take the geo stuff from there though.

# In[328]:


df_mpi = pd.merge(df_mpi_ntl[['ISO', 'Country', 'MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
                         'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural']], 
              df_mpi_subntl[['ISO country code', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional',
                            'Headcount Ratio Regional', 'Intensity of deprivation Regional']], how='left', left_on='ISO', right_on='ISO country code')
df_mpi.drop('ISO country code', axis=1, inplace=True)
df_mpi = df_mpi.merge(df_kv_mpi[['ISO', 'LocationName', 'region', 'geo', 'lat', 'lon']], left_on=['ISO', 'Sub-national region'], right_on=['ISO', 'region'])
df_mpi.drop('Sub-national region', axis=1, inplace=True)

#cols = df_mpi.columns.tolist()
#reorder it a bit more to my liking
cols = ['ISO', 'Country', 'MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban', 'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural', 
        'region', 'World region', 'LocationName', 'MPI National', 'MPI Regional', 'Headcount Ratio Regional', 'Intensity of deprivation Regional', 'geo', 'lat', 'lon']
df_mpi = df_mpi[cols]
df_mpi.shape 


# In[329]:


df_mpi[df_mpi['ISO'] == 'AFG'].head()


# In[330]:


#df_mpi['LocationName'].value_counts().head()
df_mpi.shape


# 

# Great, we didn't lose everything and should have a nice set of MPI data now.  Let's check out the loan data and make a superset to play with for visualization.

# In[331]:


df_kv_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_kv_loans.shape


# In[332]:


df_kv_loans.head()


# In[333]:


df_kv_theme = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_kv_theme.shape


# In[334]:


df_kv_theme.head()


# In[335]:


df_kv_theme_rgn = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_kv_theme_rgn.shape


# In[336]:


df_kv_theme_rgn.head()


# Some of this data...  does not line up well.  :/  We only have 6 MPI regions for Pakistan...  but 127 regions in our themes, and 146 regions in our loans.  What's going on here?

# In[337]:


len(df_mpi[df_mpi['ISO'] == 'PAK']['region'].unique())


# In[338]:


len(df_kv_theme_rgn[df_kv_theme_rgn['country'] == 'Pakistan']['region'].unique())


# In[339]:


len(df_kv_loans[df_kv_loans['country'] == 'Pakistan']['region'].unique())


# In[340]:


print("loan themes by region has " + str(len(df_kv_theme_rgn['region'].unique())) + " distinct values and " 
      + str(len(df_kv_theme_rgn['region'].str.lower().unique())) + " distinct lowered values.")


# In[341]:


print("kiva loans has " + str(len(df_kv_loans['region'].str.lower().unique())) + " distinct values and " 
       + str(len(df_kv_loans['region'].str.lower().str.lower().unique())) + " distinct lowered values.")


# In[342]:


print("mpi regions has " + str(len(df_mpi['region'].unique())) + " values.")


# In[343]:


# Youtube
HTML('<h3>How do we get all these different values to join??</h3><iframe width="560" height="315" src="https://www.youtube.com/embed/tpD00Q4N6Jk?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# It seems like the best best is to join from kiva loans to kiva themes, then kiva themes to theme regions; leveraging on country and region.  Then using mpi_region (not fully populated and per dataset notes, I assume is set by some kind of geo proximity) join it to the mpi data.

# In[344]:


#left join required, some data missing loan themes 671205 - 671199 = 6 missing
df_all_kiva = pd.merge(df_kv_loans, df_kv_theme, how='left', on='id')
df_all_kiva = df_all_kiva.merge(df_kv_theme_rgn, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'])
#df_all_kiva = df_all_kiva.merge(df_kv_mpi, how='left', on=['country', 'region'])
#df_all_kiva.head()
df_all_kiva = df_all_kiva.merge(df_mpi, how='left', left_on=['ISO', 'mpi_region'], right_on=['ISO', 'LocationName'])
#try cleaning this up a bit
df_all_kiva.drop('country_code', axis=1, inplace=True)
df_all_kiva.drop('Loan Theme Type_y', axis=1, inplace=True)
df_all_kiva.drop('geocode_old', axis=1, inplace=True)
df_all_kiva.drop('geo_y', axis=1, inplace=True)
df_all_kiva.drop('sector_y', axis=1, inplace=True)
df_all_kiva.drop(['LocationName_y', 'Country'], axis=1, inplace=True)
df_all_kiva = df_all_kiva.rename(index=str, columns={'region_x': 'region_kiva', 'region_y': 'region_mpi', 'Loan Theme Type_x': 'Loan Theme Type',
                                      'LocationName_x': 'LocationName_kiva', 'geocode': 'geocode_kiva', 'mpi_region': 'LocationName_kiva_mpi',
                                      'mpi_geo': 'geo_mpi', 'lat_y': 'lat_mpi', 'lon_y': 'lon_mpi',
                                       'geo_x': 'geo_kiva', 'lat_x': 'lat_kiva', 'lon_x': 'lon_kiva',
                                       'sector_x': 'sector', 'region_x': 'region_kiva', 'region_y': 'region_mpi',
                                       'partner_id': 'partner_id_loan', 'Partner ID': 'partner_id_loan_theme'
                                      })
#useful but dupey weird on this table; we can in theory aggregate to these anyway for our loans
df_all_kiva.drop(['number', 'amount'], axis=1, inplace=True)
df_all_kiva.head()


# Also grabbing some population data from another dataset.

# In[345]:


df_world_pop = pd.read_csv('../input/world-population/WorldPopulation.csv')
df_world_pop[['Country', '2016']].head()


# <a id=data_createnew></a>
# ## 2.2 Data Prep - Create New Fields
# 
# I also wanted to make a distinction about groups.  As a lender, I felt these were lower risk loans, as I believed the community would help eachother both to execute what the loan was for successfully, as well as if trouble, even help eachother in paying it back.  I broke the data up into somewhat arbitrary group sizes.  I wasn't sure how to deal with the NaN values either, so with some googling this is what I came up with to assign my group categories and mark the NaNs as well.  It is probably not the most elegant nor efficient python ever written.  However, it gets the job done.  

# In[346]:


def group_type(genders):

    try:
        float(genders)
        return np.nan

    except ValueError:

        grp = ''

        male_cnt = genders.split(', ').count('male')
        female_cnt = genders.split(', ').count('female')

        if(male_cnt + female_cnt == 0):
            return 'unknown'
        elif(male_cnt + female_cnt == 1):
            if(male_cnt == 1):
                return 'individual male'
            else:
                return 'individual female'
        elif(male_cnt == 1 & female_cnt == 1):
            return 'male + female pair'
        elif(male_cnt == 2 & female_cnt == 0):
            return 'male pair'
        elif(male_cnt == 0 & female_cnt == 2):
            return 'female pair'
        else:
            if(male_cnt == 0):
                grp = 'all female '
            elif(female_cnt == 0):
                grp = 'all male '
            else:
                grp = 'mixed gender '

        if(male_cnt + female_cnt > 5):
            grp = grp + 'large group (>5)'
        else:
            grp = grp + 'small group (3 to 5)'

        return grp


# In[347]:


df_all_kiva['group_type'] = df_all_kiva['borrower_genders'].apply(group_type)
df_all_kiva[['group_type']].head()


# Let's add the hashtags with their own columns as well.  This part takes a while to chooch through, kinda the most expensive single step even though it's not particularly insightful...

# In[348]:


def tag_hashtag(t, hashtag):
    
    try:
        float(t)
        return np.nan

    except ValueError:

        if(hashtag in t):
            return 1
        else:
            return 0

s = df_all_kiva['tags']
unq_tags = pd.unique(s.str.split(pat=', ', expand=True).stack())
unq_tags = [s for s in unq_tags if '#' in s]

for tag in unq_tags:
    df_all_kiva[tag] = df_all_kiva['tags'].apply(tag_hashtag, args=(tag,))
    
df_all_kiva[~df_all_kiva['tags'].isnull()][['#Parent', '#Woman Owned Biz', '#Elderly', '#Animals', '#Repeat Borrower', 'tags']].head()


# In exploring the data below, I found a loan for 100k.  Was this a real loan?  I figured it would be news if it was, so I google it; indeed it was a real loan.  [The kiva link is here](https://www.kiva.org/lend/1398161).  In fact, this also brings up the interesting point that kiva provided us with the real loan ids, so we can indeed go check out the actual loan page for any of these loans, at the URL: https://www.kiva.org/lend/ID- very cool!  Why not make that a field too...  Try going to the URLs to check them out!

# In[349]:


df_all_kiva[df_all_kiva['loan_amount'] == df_all_kiva['loan_amount'].max()]


# In[350]:


df_all_kiva['loan_URL'] = df_all_kiva['id'].apply(lambda x: 'https://www.kiva.org/lend/' + str(x))
df_all_kiva['loan_URL'].head()


# <a id=data_update></a>
# ## 2.3 Data Prep - Update Fields
# 
# World Region isn't set now in many a place where joins failed, although it's easy enough to update, as it is simply based on country.  Since India doesn't have MPI regions, it's not in the data anywhere, but I'm going to set the World Region to South Asia for them as well (same as Pakistan).  So let's do that with some ugleh python I wrote.  After that we'll get the dataframe to use proper datetimes for our timestamps as well.

# In[351]:


df_all_kiva[['country', 'World region']].head(5)


# In[352]:


# we'll do this in multiple lines to make it more readable
assoc_df = df_all_kiva[['country', 'World region']].merge(df_mpi_subntl[['Country', 'World region']].drop_duplicates(), how='left', left_on=['country'], right_on=['Country'])

df_all_kiva['World region_y'] = assoc_df.iloc[:,3].values
df_all_kiva['World region'] = df_all_kiva['World region'].fillna(df_all_kiva['World region_y'])
df_all_kiva.drop('World region_y', axis=1, inplace=True)
#df_all_kiva[df_all_kiva['country'] == 'India']['World region'] = 'South Asia'
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'India', 'South Asia', df_all_kiva['World region'])
df_all_kiva[['country', 'World region']].head(5)


# Setting some dates with the code below...

# In[353]:


df_all_kiva['date'] = pd.to_datetime(df_all_kiva['date'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['posted_time'] = pd.to_datetime(df_all_kiva['posted_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['funded_time'] = pd.to_datetime(df_all_kiva['funded_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['disbursed_time'] = pd.to_datetime(df_all_kiva['disbursed_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva[['date', 'posted_time', 'funded_time', 'disbursed_time']].head()


# Setting MPI National since I have some NaN still and can get the value... also making a display string as well...

# In[354]:


df_all_kiva = df_all_kiva.merge(df_all_kiva[['country', 'MPI National']].drop_duplicates().dropna(axis=0, how='any'), on='country', how='left')
df_all_kiva = df_all_kiva.rename(index=str, columns={'MPI National_y': 'MPI National'})
df_all_kiva.drop('MPI National_x', axis=1, inplace=True)

df_all_kiva['MPI National str'] = df_all_kiva['MPI National'].astype(float).round(3).astype(str).fillna('?')
df_all_kiva['country_mpi'] = df_all_kiva['country'] + ' - ' + df_all_kiva['MPI National str']
df_all_kiva.drop('MPI National str', axis=1, inplace=True)


# <a id=data_bad></a>
# ## 2.4 Data Prep - Bad Seeming Data
# 
# I say "seeming" in case someone points out that I've made some flaws in my logic... else I think this is bad data.
# 
# I also found some things in the data I couldn't quite resolve.  This included different partner ID values.  The first partner id and country in this example is Mexico, partner 294 - Kubo.financiero in Mexico -  from kiva_loans.csv.  However partner 199 for this loan from loan_themes_by_region.csv is for 199 - CrediCampo in El Salvador.  [Visiting the loan URL](https://www.kiva.org/lend/1340274), we find that the latter information is indeed associated with the loan.  Some of the 'use' column descriptions on these are weird, but going out to the URL they appear to all be real loans.  There are 54 records like this.  It's small enough to be considered a "don't care" I suppose in any case, among our 671,205 total; I've simply left them in untouched with no attempt to repair them for now.

# In[355]:


#df_all_kiva[(df_all_kiva['partner_id_loan'] != df_all_kiva['partner_id_loan_theme']) & (~df_all_kiva['partner_id_loan_theme'].isnull())][['id', 'country', 'partner_id_loan', 'partner_id_loan_theme', 'loan_URL', 'region_kiva', 'region_mpi', 'use']]
#54 total
df_all_kiva[df_all_kiva['id'] == 1340274][['id', 'country', 'partner_id_loan', 'partner_id_loan_theme', 'loan_URL', 'region_kiva', 'region_mpi']]


# <a id=data_complete></a>
# ## 2.5 Data Prep - Completed Data
# 
# A reordering to my liking, along with the characteristics of the data that we will move forward with.  Time to go fishing!

# In[356]:


cols = ['id', 'loan_amount', 'funded_amount', 'activity', 'sector', 'use', 'currency', 'lender_count', 'repayment_interval', 'term_in_months', 'date', 
 'posted_time', 'funded_time', 'disbursed_time',  'borrower_genders', 'group_type', 'Loan Theme ID', 'Loan Theme Type', 'forkiva',  'partner_id_loan', 
 'partner_id_loan_theme', 'Field Partner Name', 'rural_pct', 'World region', 'country', 'ISO', 'country_mpi', 'MPI National', 'MPI Urban', 'Headcount Ratio Urban', 
 'Intensity of Deprivation Urban', 'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural', 'LocationName_kiva', 'LocationName_kiva_mpi',  
 'names',  'region_kiva', 'region_mpi', 'MPI Regional', 'Headcount Ratio Regional', 'Intensity of deprivation Regional', 'geocode_kiva', 'geo_kiva', 'lat_kiva', 
 'lon_kiva', 'geo_mpi', 'lat_mpi', 'lon_mpi', 'loan_URL', 'tags', '#Elderly', '#Woman Owned Biz', '#Repeat Borrower', '#Parent', '#Vegan', '#Eco-friendly',
 '#Sustainable Ag', '#Schooling', '#First Loan', '#Low-profit FP', '#Post-disbursed', '#Health and Sanitation', '#Fabrics', '#Supporting Family', '#Single Parent',
 '#Biz Durable Asset', '#Interesting Photo', '#Single', '#Widowed', '#Inspiring Story', '#Animals', '#Refugee', '#Job Creator', '#Hidden Gem', '#Unique',
 '#Tourism', '#Orphan', '#Trees', '#Female Education', '#Technology', '#Repair Renew Replace']
#df_all_kiva.info()
df_all_kiva = df_all_kiva[cols]
df_all_kiva.info()


# I feel like this is as good as this available set is going to get.  I tried to lop off everything that appeared repeated and unusable.  The Kiva data seems to be more accurate in regards to location, with them attempting to choose the best representation of MPI, I attempted to tie in the richer MPI data, and make more use out of the original atomic Kiva data by running some functions against it to create some new columns.

# <a id=dist></a>
# # 3 Distributions and Contributions
# <a id=dist_loan></a>
# ## 3.1 Distribution of Loan Amount
# 
# Let's take a look at how requested loan amounts are distributed.
# 

# In[357]:


sns.set_palette

sns.distplot(df_all_kiva['loan_amount'])
plt.show()


# Oh my!  Tis quite a large graph along the x axis.  I double checked to make sure the loan amount in the data description was indeed in USD, and not local currency.  Let's see how we can get this chart to be a little more useful.

# In[358]:


for x in range(0,10):
    print('99.' + str(x) + 'th percentile loan_amount is: ' + str(df_all_kiva['loan_amount'].quantile(0.99 + x/1000)))


# Let's stick with 99th percentile for plotting this data.

# In[359]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(.99) ]['loan_amount'])
plt.show()


# <a id=dist_fund></a>
# ## 3.2 Distribution of Funded Amount
# 
# What percentage of loans are actually funded?  What's the funded distribution look like?

# In[360]:


df_all_kiva[df_all_kiva['loan_amount'] == df_all_kiva['funded_amount']]['id'].count() / df_all_kiva['id'].count()*100


# In[361]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(.99) ]['funded_amount'])
plt.show()


# Let's do the same for the count of lenders.  Generally the approach is to lend 25 on each loan and mitigate risk over many different loans, and this chart to look very similar as a result.  At least... that's what I used to do on prosper.com, and that's what I expected here...

# In[362]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['lender_count'] < df_all_kiva['lender_count'].quantile(.99) ]['lender_count'])
plt.show()


# <a id=avg_cont></a>
# ## 3.3 Average Kiva Member Contribution
# 
# The above graph is surprisingly dissimilar to me...  Let's take a look at the average amount lent.

# In[363]:


df_all_kiva['funded_amount'].sum() / df_all_kiva['lender_count'].sum()


# Wow - I didn't expect that at all.  It makes sense that funding the really big loans would have large values though, and there's a fair amount of those...  surely we close in on 25 per kiva user contribution as we move down to smaller loans, right?  A bit of my hack python later and...

# In[364]:


lst1 = range(100,0,-10)
lst2 = list()

for x in range(0, 10):
    #print('at ' + str(round((1 - x/10)*100, 0)) + 'th percentile loan amount, average lender lent: ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['loan_amount'].sum() / df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['lender_count'].sum(), 2)) + ' with average loan ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['loan_amount'].mean(), 2)) + ' and average number of lenders ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['lender_count'].mean(), 2)) )
    lst2.append(round(df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(1 - x/10) ]['funded_amount'].sum() / df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(1 - x/10) ]['lender_count'].sum(), 2))
    
dfavg = pd.DataFrame(
    {'percentile': lst1,
     'average_per_lender': lst2
    })

plt.figure(figsize=(10,5))
ax = sns.barplot(x='percentile', y='average_per_lender', data=dfavg, color='c')
ax.set_title('Average Lender Contribution by Percentile', fontsize=15)
plt.show()


# Of my *own* 997 loans, it appears I've put 50 in to 3 of them, and 25 into the rest.  I was expecting this to be pretty common, and the average contribution to be very close to 25 throughout.  However it appears that people put in $50+ much more often than I do myself!

# <a id=tsa></a>
# # 4 Top Sectors and Activities
# 
# <a id=ts></a>
# ## 4.1 Top Sectors

# In[365]:


plt.figure(figsize=(15,8))
plotSeries = df_all_kiva['sector'].value_counts()
ax = sns.barplot(plotSeries.values, plotSeries.index, color='pink')
ax.set_title('Top Sectors', fontsize=15)
plt.show()


# <a id=act30></a>
# ## 4.2 Absolute Top 30 Overall Activities, by Sector
# 
# Here we also see Agriculture, Retail, and Food strongly represented in the top activities for loans.

# In[366]:


#plt.figure(figsize=(15,8))
fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)
#plotSeries = df_all_kiva['activity'].value_counts().head(20)
df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
df_plot = df_plot.to_frame()
df_plot.columns = ['count']
df_plot.reset_index(level=1, inplace=True)
df_plot.reset_index(level=0, inplace=True)
df_plot = df_plot.sort_values('count', ascending=False).head(30)
sectors = df_plot['sector'].unique()
palette = itertools.cycle(sns.color_palette('hls', len(sectors)))

for s in sectors:
    df_plot['graphcount'] = np.where(df_plot['sector'] == s, df_plot['count'], 0)
    sns.barplot(x='graphcount', y='activity', data=df_plot,
            label=s, color=next(palette))
    
ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Sector'
leg.set_title(new_title)
ax.set_title('Top Overall Activities', fontsize=15)
plt.show()


# <a id=speed_wr></a>
# ## 4.3 Funding Speed by World Region by Sector
# 
# I tried to calculate a funding speed here, how quickly loans were funded by world region and sector.  The output was too skewed so I ended up graphing the log version of it.  I coupled it with an absolute count of loans as well.
# 1. Arab States - Health funds extremely fast.  However we can also see this was not very many loans.
# 2. We see a lot of similar rates, although Arab States overall seem to fund quickly.  Perhaps this is attributable to a boost from religious lenders?
# 3. The absolute chart shows us pretty similar information to what we already knew from above about hot sectors.  It does allow us to see it's a few regions really driving this.

# In[367]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'sector']]
#df_funding.head()

df_funding['days_to_fund'] = (df_funding['funded_time'] - df_funding['posted_time'])
df_funding['days_to_fund'] = df_funding['days_to_fund'].apply(lambda x: x.total_seconds()/60/60/24)
#df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
df_funding = df_funding.groupby(['World region', 'sector']).sum()
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)
df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
#df_funding.head()
#df_funding.groupby('')
#df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
#df_funding['num_days'] = (datetime.strptime(df_funding['funded_time'].split('+')[0], date_format) - datetime.strptime(df_funding['posted_time'].split('+')[0], date_format)).total_seconds()/60/60/24
#df_funding.groupby(['country']['sector'])
df_heat = df_funding[['World region', 'sector', 'funding_speed']]
f, ax = plt.subplots(figsize=(18, 8))
df_heat['funding_speed'] = np.log10(df_heat['funding_speed'])
#df_heat.pivot('World region', 'sector', 'funding_speed').info()
sns.heatmap(df_heat.pivot('World region', 'sector', 'funding_speed'), annot=True, linewidths=.5, ax=ax)
plt.show()


# In[368]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'sector']]
df_funding = df_funding[['World region', 'sector', 'funded_amount']].groupby(['World region', 'sector']).agg('count')
#df_funding
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)

df_heat = df_funding[['World region', 'sector', 'funded_amount']]
df_heat = df_heat.rename(index=str, columns={'funded_amount': 'count'})
f, ax = plt.subplots(figsize=(18, 8))
#df_heat.pivot('World region', 'sector', 'count').info()
sns.heatmap(df_heat.pivot('World region', 'sector', 'count'), annot=True, fmt='d', linewidths=.5, ax=ax)
plt.show()


# <a id=speed_gt></a>
# ## 4.4 Funding Speed by Group Type by Sector
# 
# This is the same idea, although now looking at borrower gender demographics.  We can see individuals in Sub-Saharan Africa take the longest to fund - although we can also see that is because they are in competition with a large amount of people.  Perhaps organizing or joining a group could help if obtaining funding is a problem.  Some of these loan counts are very low and arguably I should be excluding these groups, perhaps a revisit in the future, this was in part to test my learning python abilities, unfortunately nothing too big to draw from here.
# 

# In[369]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'group_type']]
#df_funding.head()

df_funding['days_to_fund'] = (df_funding['funded_time'] - df_funding['posted_time'])
df_funding['days_to_fund'] = df_funding['days_to_fund'].apply(lambda x: x.total_seconds()/60/60/24)
#df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
df_funding = df_funding.groupby(['World region', 'group_type']).sum()
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)
df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
#df_funding.head()
#df_funding.groupby('')
#df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
#df_funding['num_days'] = (datetime.strptime(df_funding['funded_time'].split('+')[0], date_format) - datetime.strptime(df_funding['posted_time'].split('+')[0], date_format)).total_seconds()/60/60/24
#df_funding.groupby(['country']['sector'])
df_heat = df_funding[['World region', 'group_type', 'funding_speed']]
f, ax = plt.subplots(figsize=(18, 8))
df_heat['funding_speed'] = np.log10(df_heat['funding_speed'])
#df_heat.pivot('World region', 'sector', 'funding_speed').info()
sns.heatmap(df_heat.pivot('World region', 'group_type', 'funding_speed'), annot=True, linewidths=.5, ax=ax)
plt.show()


# In[370]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'group_type', 'country', 'sector']]
df_funding = df_funding[['World region', 'group_type', 'funded_amount']].groupby(['World region', 'group_type']).agg('count')
#df_funding
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)

df_heat = df_funding[['World region', 'group_type', 'funded_amount']]
df_heat = df_heat.rename(index=str, columns={'funded_amount': 'count'})
f, ax = plt.subplots(figsize=(18, 8))
df_heat['count'].fillna(0)
df_heat = df_heat.pivot('World region', 'group_type', 'count')

sns.heatmap(df_heat, annot=True, fmt='g', linewidths=.5, ax=ax)
plt.show()


# <a id=group></a>
# # 5 Loan Count by Gender and Group
# 
# Women totally dominate Kiva, followed by individual men and pairs of men.  After that we're back in full force with women's groups both small and large, as well as mixed groups.  I have done a fair amount of group loans as I am of the mind they may both be lower risk and may help with the borrower achieving local success with the power of the group to help them through any stumbling points.

# In[371]:


df_stacked = df_all_kiva[['group_type', 'id']].groupby(['group_type']).agg('count')

df_stacked.reset_index(level=0, inplace=True)
df_stacked = df_stacked.rename(index=str, columns={'id': 'count'})

df_stacked = df_stacked.sort_values('count', ascending=False)
groups = df_stacked['group_type'].unique()

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)

for gt in groups:
    df_stacked['graphcount'] = np.where(df_stacked['group_type'] == gt, df_stacked['count'], 0)

    if ((gt == 'individual female') | ('all female' in gt)):
        if(gt == 'individual female'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='women', color='#f36cee')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#f36cee')
    elif ((gt == 'male + female pair') | ('mixed' in gt)):
        if(gt == 'male + female pair'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked,
                label='mixed', color='#8f0e87')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#8f0e87')
    else:
        if(gt == 'individual male'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='men', color='#08b1e7')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#08b1e7')
    
ax.set_xlabel('count of loans')
ax.legend(ncol=1, loc='best', frameon=True)

leg = ax.get_legend()
new_title = 'Gender'
leg.set_title(new_title)
ax.set_title('Group Size and Gender Mix', fontsize=15)
plt.show()


# <a id=trend></a>
# # 6 What's #Trending Kiva?
# 
# Let's take a look at the #most #popular #hashtags. 

# In[372]:


df_sum_tags = pd.DataFrame()
for tag in unq_tags:
        s = df_all_kiva[tag].sum()
        df_sum_tags = df_sum_tags.append(pd.DataFrame(s, index=[tag], columns=['count']))
        
df_sum_tags

#plt.figure(figsize=(15,9))
fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)
df_sum_tags.sort_values('count', inplace=True, ascending=False)
#df_sum_tags.sort_index(inplace=True)
sns.barplot(y=df_sum_tags.index, x=df_sum_tags['count'], color='#c44e52')
ax.set_xlabel('#count #of #loans')
ax.set_ylabel('#hashtag')
ax.set_title('#All #of #the #Hashtags', fontsize=15)
plt.show()


# <a id=themes></a>
# # 7 Loan Theme Types
# 
# Let's take a look at loan theme type.  Rather than count our specific subset of loans, it seems it may be more useful for us to simply use the number column from the loan_themes_by_region data, which contains the total number of loans the partner has made for that theme.  **NOTE:** The highest is General, which has both forkiva set as yes and no - however it isn't particularly interesting and skews the chart.  Thus in the code I have chosen to omit it.  We can see that some of the kiva categories are doing pretty well.

# In[373]:


df_themes = df_kv_theme_rgn.groupby(['Loan Theme Type', 'forkiva'])['number'].sum()
df_themes = df_themes.to_frame()

df_themes.reset_index(level=1, inplace=True)
df_themes.reset_index(level=0, inplace=True)
df_themes = df_themes.pivot(index='Loan Theme Type', columns='forkiva', values='number')

df_themes['No'] = df_themes['No'].fillna(0)
df_themes['Yes'] = df_themes['Yes'].fillna(0)

df_themes['total'] = df_themes['No'].fillna(0) + df_themes['Yes'].fillna(0)
df_themes = df_themes.sort_values(by='total', ascending=False).head(40)
df_themes.reset_index(level=0, inplace=True)

s_force_order = df_themes[df_themes['Loan Theme Type'] != 'General'].sort_values('total', ascending=False)['Loan Theme Type'].head(40)

# Initialize the matplotlib figure
fig, ax = plt.subplots(figsize=(15, 10))

sns.barplot(x='total', y='Loan Theme Type', data=df_themes[df_themes['Loan Theme Type'] != 'General'],
            label='No', color='#8ed3f4', order=s_force_order)

sns.barplot(x='Yes', y='Loan Theme Type', data=df_themes[df_themes['Loan Theme Type'] != 'General'],
            label='Yes', color='#0abda0', order=s_force_order)

ax.legend(ncol=2, loc='best', frameon=True)
ax.set(ylabel='Loan Theme Type',
       xlabel='number of loans')

leg = ax.get_legend()
new_title = 'for kiva?'
leg.set_title(new_title)
ax.set_title('Top Loan Theme Types (Excluding General) by forkiva', fontsize=15)
plt.show()


# <a id=curr></a>
# # 8 Exploring Currency
# 
# <a id=curr_usg></a>
# ## 8.1 Currency Usage
# 
# Let's take a look at countries used in multiple countries.  I've included a loan count for currencies that are used in 2 or more countries.  However sometimes this only meant a single loan, and the graph was big.  I also didn't like the black marks at the end of the charts when stack visually, so I only ended up plotting those countries which have had loans in multiple currencies where their were more than 20 loans in a currency.  ILS and JOD (Israel and Jordan) thus only show one bar, although in Palestine < 20 loans in each currency was lent (4 and 8, respecitively).  Developing countries sometimes end up with monetary systems pegged to more stable countries or straight out use the money of those countries as a result of lack of trust or poor management of their state currency.  Some notes:
# 1. The *Central African CFA Franc* **XAF** is pegged to the Euro at 1 Euro = 655.957 XAF. It is the currency for six independent states in central Africa: Cameroon, Central African Republic, Chad, Republic of the Congo, Equatorial Guinea and Gabon.
# 2. The *West African CFA Franc* **XOF** is pegged the same way.  It is the currency for Benin, Burkina Faso, Guinea-Bissau, Ivory Coast, Mali, Niger, Senegal, and Togo.
# 3. The following countries outside the US *only* use **USD**: Ecuador, East Timor, El Salvador, Marshall Islands, Micronesia, Palau, Turks and Caicos, British Virgin Islands, and Zimbabwe.  Cambodia has the Cambodien Riel (KHR) however foreign cards disberse USD in ATMs, and 90% of the country uses US Dollars, with the local currency generally used for change or anything worth less than a dollar.  We can see the vast amount of Palestine's loan counts are here as well.
# 4. Note Congo and The Democratic Republic of the Congo are different countries.

# In[374]:


min_num_loans = 21
df_currencies = df_all_kiva[['country', 'currency']].drop_duplicates().groupby('currency').count()
df_currencies.reset_index(level=0, inplace=True)
s_currencies = df_currencies[df_currencies['country'] > 1]['currency']
df_currencies = df_all_kiva[df_all_kiva['currency'].isin(s_currencies)].groupby(['country', 'currency'])['id'].count()
df_currencies = pd.Series.to_frame(df_currencies)
df_currencies.reset_index(level=1, inplace=True)
df_currencies.reset_index(level=0, inplace=True)
df_currencies.sort_values(['currency', 'id'], inplace=True)
df_currencies = df_currencies.rename(index=str, columns={'id': 'count'})

s_force_order = df_currencies[df_currencies['count'] >= min_num_loans].sort_values(['currency', 'count'], ascending=False).drop_duplicates()['country']

fig, ax = plt.subplots(1, 1, figsize=(15, 12), sharex=True)

currencies = df_currencies['currency'].unique()
palette = itertools.cycle(sns.color_palette('hls', len(currencies)))

df_piv = df_currencies[df_currencies['count'] >= min_num_loans].pivot(index='country', columns='currency', values='count')
df_piv.reset_index(level=0, inplace=True)

for c in currencies:
    sns.barplot(x=c, y='country', data=df_piv,
            label=c, color=next(palette), order=s_force_order)
    
ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Currency'
leg.set_title(new_title)
ax.set_title('Currencies Used Across Multiple Countries', fontsize=15)
plt.show()


# <a id=curr_avg></a>
# ## 8.2 Mean Loan by Currency for Top 20 Currencies
# 
# This is a bit of a tricky one...  I'm showing the average loan, and I'm showing the percentage of it by currency within the stacked bar.  Ie. if a country is only in USD the loan is all the single USD color.  If a country has three 100 USD loans and one 200 PHP loan, it would show a total bar length of 500/4 = $125.  60% of the bar would be in USD color and 40% of the bar in PHP color.  To try and keep the graph slightly less busy than plotting all of the data, only the amount disbursed in the top 20 currencies by amount are shown.  Countries are only shown when they have at least 250 loans in the data.
# 
# Ultimately, this took a lot of my hack python effort; and although the final data set is small, it seems the more currencies asked for the much more computationally expensive plotting the graph is.  Did it tell us anything interesting?  Only one thing, that the vast majority of countries disburse loans in a single currency, even if it's not their own.
# 
# The initial version of this graph was stacked incorrectly and only showed Lebanon, thus the research into it next below.  With the charts now properly stacking, we can see a few other countries with a fair amount of currency use that is not their own.  It is likely this is similarly related to fluctuations in the purchasing power and stability of their own currencies as well.

# In[400]:


num_top_currencies = 20
num_min_country_loans = 250

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
disc_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', 
               '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080', '#FFFFFF', '#000000']
sns.set_palette(disc_colors)


df_currencies = df_all_kiva[['country', 'currency', 'loan_amount', 'id']]
df_currencies = df_currencies.groupby(['country', 'currency']).agg({'loan_amount':'sum', 'id':'count'})
df_currencies = df_currencies.rename(index=str, columns={'id': 'count'})
#df_currencies.shape  #137
df_currencies.reset_index(level=1, inplace=True)
df_currencies.reset_index(level=0, inplace=True)

df_currencies_tot = df_currencies.groupby('country').sum()
df_currencies_tot.reset_index(level=0, inplace=True)
df_currencies_tot = df_currencies_tot.rename(index=str, columns={'loan_amount': 'sum_loan_amount'})
df_currencies_tot = df_currencies_tot.rename(index=str, columns={'count': 'sum_count'})



df_currencies = df_currencies.merge(df_currencies_tot, on='country')
#avg_loan_cur = average loan times ratio of disbursed currency of total currency
df_currencies['avg_loan_cur'] = (df_currencies['sum_loan_amount'] / df_currencies['sum_count']) * (df_currencies['loan_amount'] / df_currencies['sum_loan_amount'])
df_currencies = df_currencies[df_currencies['sum_count'] >= num_min_country_loans ]

# get top x many used currencies
df_limit_cur = df_currencies.groupby('currency')['sum_loan_amount'].sum().to_frame().sort_values('sum_loan_amount', ascending=False)
df_limit_cur.reset_index(level=0, inplace=True)
s_limit_cur = df_limit_cur['currency'].head(num_top_currencies)

df_currencies = df_currencies.pivot(index='country', columns='currency', values='avg_loan_cur')
#currencies = df_currencies.columns.tolist()
#currencies = ['USD', 'PHP', 'XAF']

df_currencies = df_currencies[s_limit_cur]
df_currencies.reset_index(level=0, inplace=True)
df_currencies.dropna(axis=0, how='all')

df_currencies['total_cur'] = 0
for c in s_limit_cur:
    df_currencies[c] = df_currencies[c].fillna(0)
    df_currencies['total_cur'] = df_currencies['total_cur'] + df_currencies[c]

fig, ax = plt.subplots(1, 1, figsize=(19, 12), sharex=True)

palette = itertools.cycle(sns.color_palette(palette=disc_colors, n_colors=22))

for c in s_limit_cur:
    sns.barplot(x='total_cur', y='country', data=df_currencies,
            label=c, color=next(palette))
    df_currencies['total_cur'] = df_currencies['total_cur'] - df_currencies[c]

ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('mean loan')
leg = ax.get_legend()
new_title = 'Currency'
leg.set_title(new_title)
ax.set_title('Mean Loan - By Disbursed Currency Percentage', fontsize=15)
plt.show()


# <a id=curr_leb></a>
# ## 8.3 What's going on in Lebanon?
# 
# I plotted some monthly data and also found a graph of inflation for the Lebanese Pound (LBP) - they were actually experiencing deflation between the end of 2014 and mid-year 2016.  This means the currency was actually gaining purchasing power.  This seems to account for the decline in USD and rise in LBP.  It is curious as to who is making the distinction, ie. were borrowers asking for LBP or were field partners offering/pushing it?  Inflation is more of the borrower's friend than deflation - the borrower would likely not reap any benefits of increased purchasing power as they are likely borrowing to buy something right away.  Inflation, however, means the borrower is paying back their loan with "cheaper" currency in that the currency now has less purchasing power.  In theory the interest rate the money is lent at accounts for both a small profit for the field partner as well as a factor to hedge against inflation.

# In[376]:


df_lebanon = df_all_kiva[df_all_kiva['country'] == 'Lebanon'][['currency', 'loan_amount', 'disbursed_time']]
df_lebanon['disbursed_time_month'] = df_lebanon['disbursed_time'] + pd.offsets.MonthBegin(-1)

df_lebanon = df_lebanon.groupby(['currency', 'disbursed_time_month']).sum()
df_lebanon.reset_index(level=1, inplace=True)
df_lebanon.reset_index(level=0, inplace=True)



fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_lebanon[df_lebanon['currency'] == 'LBP']['disbursed_time_month'], df_lebanon[df_lebanon['currency'] == 'LBP']['loan_amount'])
plt.plot(df_lebanon[df_lebanon['currency'] == 'USD']['disbursed_time_month'], df_lebanon[df_lebanon['currency'] == 'USD']['loan_amount'])
plt.legend(['LBP', 'USD'], loc='upper left')
ax.set_title('Loan Distribution by Currency in Lebanon', fontsize=15)
plt.show()


# <img align=center src=https://www.doyouevendata.com/wp-content/uploads/2018/03/lebanon.jpg>

# <a id=leb_fld></a>
# ## 8.4 Lebanese Field Partners
# 
# Digging deeper into the data, it looks like Lebanon has only two field partners.  Al Majmoua does most of their lending in USD, and was very little until the deflation period started.  It did increase but is tracking back towards very low numbers again.  Ibdaa Microfinance only deals in LBP and appears to be on quite the roller coaster in regards to lending to Kiva borrowers.

# In[377]:


df_lebanon = df_all_kiva[df_all_kiva['country'] == 'Lebanon'][['id', 'currency', 'loan_amount', 'disbursed_time', 'Field Partner Name', 'sector', 'activity']]
print ('USD Partner loan count:')
print(df_lebanon[df_lebanon['currency'] == 'USD']['Field Partner Name'].value_counts().head(15))
print ('LBP Partner loan count:')
print(df_lebanon[df_lebanon['currency'] == 'LBP']['Field Partner Name'].value_counts().head(15))


# In[378]:


df_lebanon = df_all_kiva[(df_all_kiva['country'] == 'Lebanon') & (df_all_kiva['currency'] == 'LBP')][['Field Partner Name', 'loan_amount', 'disbursed_time']]
df_lebanon['disbursed_time_month'] = df_lebanon['disbursed_time'] + pd.offsets.MonthBegin(-1)

df_lebanon = df_lebanon.groupby(['Field Partner Name', 'disbursed_time_month']).sum()
df_lebanon.reset_index(level=1, inplace=True)
df_lebanon.reset_index(level=0, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_lebanon[df_lebanon['Field Partner Name'] == 'Al Majmoua']['disbursed_time_month'], df_lebanon[df_lebanon['Field Partner Name'] == 'Al Majmoua']['loan_amount'])
plt.plot(df_lebanon[df_lebanon['Field Partner Name'] == 'Ibdaa Microfinance']['disbursed_time_month'], df_lebanon[df_lebanon['Field Partner Name'] == 'Ibdaa Microfinance']['loan_amount'])
plt.legend(['Al Majmoua', 'Ibdaa Microfinance'], loc='upper left')
ax.set_title('LBP Only Loans by Field Partner in Lebanon', fontsize=15)
plt.show()


# <a id=bullet_agg></a>
# # 9 Bullet Loans for Agriculture
# 
# Per [Kiva Labs - Financing Agriculture](https://www.kiva.org/about/impact/labs/financingagriculture) there is a drive towards extending bullet type loans in the Agriculture sector as a solution proposed to the uncertainty of farming life, whether raising crops or rearing animals.  These loans allow for the majority of pay back to made in in a lump sum at the end of the loan life - timing well with the farmer actually selling their crop they have tended to for a farming cycle, or an animal they have raised for years.  This graph contains all the countries where at least 100 loans total have been made.  It is ordered in descending order of MPI National, which may not be the best proxy for food security but it's what we've got to roll with.
# 
# We do have some takeaways here;
# 1. As a general rule, anything in red on this chart is open for improvement!
# 2. Countries near the top of the chart may have some of the strongest impact in reducing poverty as they are generally more impoverished.
# 3. Countries with many loans in red have a large opportunity to make an impact as well - loans are already being made through field partners, just not of the bullet type.
# 4. Mali and Nigeria were specifically mentioned in the Kiva article, and here we can see the vast majority of their loans are bullet loans - but there's also some other countries and field partners doing a great job with this too!
# 5. We don't have default data - perhaps this is lower for Agriculture bullet loans and Kiva could use this as a point to sell to field partners currently not offering them?
# 6. Some countries have no bullet loans - perhaps Kiva could concentrate limited corporate resources there, whereas in countries where field partners are already leveraging bullet loans, encourage the field partners to spread the good news to other field partners?
# 7. The list might be better sorted in order of food insecurity if some national level data is available.
# 
# 

# In[396]:


def isbullet(ri):

    if 'bullet' in ri:
        return 'bullet'
    else:
        return 'not bullet'
#df_bullet = df_all_kiva[df_all_kiva['sector'] == 'Agriculture'][['country', 'repayment_interval', 'MPI National']]
df_bullet = df_all_kiva[df_all_kiva['sector'] == 'Agriculture'][['country_mpi', 'repayment_interval', 'MPI National']]
df_bullet['loan_type'] = df_bullet['repayment_interval'].apply(isbullet)

#df_bullet = df_bullet.groupby(['country', 'loan_type', 'MPI National']).count()
df_bullet = df_bullet.groupby(['country_mpi', 'loan_type', 'MPI National']).count()

df_bullet = df_bullet.rename(index=str, columns={'repayment_interval': 'count'})

df_bullet.reset_index(level=2, inplace=True)
df_bullet.reset_index(level=1, inplace=True)
df_bullet.reset_index(level=0, inplace=True)

num_min_country_loans = 10

df_bullet = df_bullet[df_bullet['count'] >= num_min_country_loans ]

#df_bullet['MPI National str'] = df_bullet['MPI National'].astype(float).round(3).astype(str).fillna('?')
#df_bullet['country_mpi'] = df_bullet['country'] + ' - ' + df_bullet['MPI National str']

s_force_order = df_bullet[['MPI National', 'country_mpi']].sort_values('MPI National', ascending=False).drop_duplicates()['country_mpi']

df_piv = df_bullet.pivot(index='country_mpi', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

bts = ['bullet', 'not bullet']

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='country_mpi', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='country_mpi', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='upper right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('Count of Agricultural Loans by Loan Type, Ordered by National MPI Descending', fontsize=15)
plt.show()


# In[380]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>It appears Kiva has significant open opportunity to spreading the beneficial bullet loans for Agriculture borrowers, and can prioritize them by potential market size changes and poverty dimensions.')


# <a id=sal1></a>
# ## 9.1 El Salvador Investigation by Activity
# Let's take a look at El Salvador.  There's only a few field partners, and the majority of loans really comes from one, so that didn't appear to make a meaningful difference.

# In[381]:


df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval']]
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['activity', 'loan_type']).count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})
df_piv = df_els.pivot(index='activity', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

s_force_order = df_piv[['activity', 'total']].sort_values('total', ascending=False).drop_duplicates()['activity']

fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='activity', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='activity', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='center right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('El Salvador Agriculture Loans by Activity', fontsize=15)
plt.show()


# <a id=sal2></a>
# ## 9.2 El Salvador Loan Count Over Time

# In[382]:



df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval', 'disbursed_time']]
df_els['disbursed_time_month'] = df_els['disbursed_time'] + pd.offsets.MonthBegin(-1)
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['loan_type', 'disbursed_time_month'])[['activity']].count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'activity': 'count'})

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_els[df_els['loan_type'] == 'bullet']['disbursed_time_month'], df_els[df_els['loan_type'] == 'bullet']['count'], color='#67c5cb')
plt.plot(df_els[df_els['loan_type'] == 'not bullet']['disbursed_time_month'], df_els[df_els['loan_type'] == 'not bullet']['count'], color='#cb6d67')

plt.legend(['bullet', 'not bullet'], loc='upper left', frameon=True)
ax.set_title('El Salvador Loan Count', fontsize=15)
plt.show()


# <a id=sal3></a>
# ## 9.3 El Salvador Animal Loans
# What if we just look at the activities Livestock, Cattle, Poultry, Pigs?

# In[383]:


animals = ['Livestock', 'Cattle', 'Poultry', 'Pigs']
df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture') 
                    & (df_all_kiva['activity'].isin(animals))][['activity', 'repayment_interval', 'disbursed_time']]
df_els['disbursed_time_month'] = df_els['disbursed_time'] + pd.offsets.MonthBegin(-1)
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['loan_type', 'disbursed_time_month', 'activity'])[['repayment_interval']].count()
df_els.reset_index(level=2, inplace=True)
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})

linestyles = ['-', '--', '-.', ':']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

i = 0
for a in animals:
    if a in ['Livestock', 'Cattle']:
        ax1.plot(df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['count'], color='#67c5cb', linestyle=linestyles[i], label=a + ' ' + 'bullet', linewidth=3)
        ax1.plot(df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['count'], color='#cb6d67', linestyle=linestyles[i], label=a + ' ' + 'not bullet', linewidth=3)
    else:
        ax2.plot(df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['count'], color='#67c5cb', linestyle=linestyles[i], label=a + ' ' + 'bullet', linewidth=3)
        ax2.plot(df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['count'], color='#cb6d67', linestyle=linestyles[i], label=a + ' ' + 'not bullet', linewidth=3)
    
    i = i + 1

ax1.legend(loc='upper left', frameon=True)
ax2.legend(loc='upper left', frameon=True)
ax1.set_title('El Salvador Loan Count by Animal Activity', fontsize=15)
plt.show()


# Interestingly enough it appears Livestock has been on a big non-seasonal decline, meanwhile cattle has gone up - perhaps Kiva borrowers are switching stocks noting some kind of overall trend?  Whereas livestock can produce commodities (meat, milk, eggs, fur, etc.), cattle is generally primarily raised for meat, and we'll note that a separate (ungraphed) Activity value exists for the category Dairy.  Countries generally consume more beef as they get wealthier, although I didn't notice a major upswing on GDP or anything from a quick google.  Per https://www.export.gov/article?id=El-Salvador-Agricultural-Sector we have this quote *Dairy production is increasing due to government incentives and sanitary regulations that provide protection against contraband cheese from Nicaragua and Honduras.*  Perhaps the bump we are seeing is a result of this; although not fully categorized properly by the field partner?

# In[384]:


df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture') 
                    & (df_all_kiva['activity'] == 'Cattle')][['loan_URL', 'disbursed_time']].sort_values('disbursed_time', ascending=False)['loan_URL'].head()


# In[385]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>I manually sampled the 5 most recent Cattle loans in El Salvador above - 3 appeared to be for beef, 1 was a loan for auto repair to bring milk to market, and 1 was for dairy production.  Kiva should ensure field partners are coding things properly so that they are able to assess impacts and make decisions with cleaner data.')


# Unfortunately we don't see much of a rising trend vs. history for Pigs or Poultry.  Maybe these aren't very suitable for these type of loans, perhaps the duration is quick anyway?  Let's see what is done not so far away in Colombia, are bullet loans popular for these specific activities there?

# In[386]:


df_els = df_all_kiva[(df_all_kiva['country'] == 'Colombia') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval']]
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['activity', 'loan_type']).count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})
df_piv = df_els.pivot(index='activity', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

s_force_order = df_piv[['activity', 'total']].sort_values('total', ascending=False).drop_duplicates()['activity']
fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='activity', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='activity', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='center right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('Colombia Agriculture Loans by Activity', fontsize=15)
plt.show()


# In[387]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Colombia is offering a lot of bullet loans for Pigs and Poultry!  Kiva should seek to encourage their field partners to expand their bullet loan offerings to these activities in El Salvador.')


# <a id=phil></a>
# ## 10. Is the Philippines Really the Country with the Most Kiva Activity?
# These are the top 15 countries by loan count.

# In[388]:


plt.figure(figsize=(15,8))
plotSeries = df_all_kiva['country'].value_counts().head(15)
ax = sns.barplot(plotSeries.values, plotSeries.index, color='c')
ax.set_title('Top 15 Countries by Loan Count', fontsize=15)
ax.set_xlabel('count of loans')
plt.show()


# However, are they the countries with the most lending activity going on?  The Philippines is pretty big; Ecuador rather small in comparison.  What if we make a per capita adjustment?  Keeping life simple I'm just going to use 2016 population data.  We are going to do a per capita adjustment.

# In[389]:


min_loans_per_million = 100
# make countries by sector set to display
df_countries = df_all_kiva[['id', 'country', 'country_mpi', 'sector', 'MPI National']].groupby(['country', 'country_mpi', 'sector', 'MPI National'])['id'].count().to_frame()
df_countries.rename(index=str, columns={'id': 'count'}, inplace=True)
df_countries.reset_index(level=3, inplace=True)
df_countries.reset_index(level=2, inplace=True)
df_countries.reset_index(level=1, inplace=True)
df_countries.reset_index(level=0, inplace=True)
sectors = df_countries['sector'].unique()
#df_countries['MPI National str'] = df_countries['MPI National'].astype(float).round(3).astype(str).fillna('?')
#df_countries['country_mpi'] = df_countries['country'] + ' - ' + df_countries['MPI National str']
# adjust by population
df_countries = df_countries.merge(df_world_pop[['Country', '2016']], left_on=['country'], right_on=['Country'])
df_countries.drop('Country', axis=1, inplace=True)
df_countries['loans_per_mil'] = df_countries['count'] / df_countries['2016'] * 1000000
# get total loans per population
df_total_per_mil = df_all_kiva[['id', 'country']].groupby(['country'])['id'].count().to_frame()
df_total_per_mil.rename(index=str, columns={'id': 'count'}, inplace=True)
df_total_per_mil.reset_index(level=0, inplace=True)
df_total_per_mil = df_total_per_mil.merge(df_world_pop[['Country', '2016']], left_on=['country'], right_on=['Country'])
df_total_per_mil.drop('Country', axis=1, inplace=True)
df_total_per_mil['total_loans_per_mil'] = df_total_per_mil['count'] / df_total_per_mil['2016'] * 1000000
#restrict output to at least s many loans per million
#df_countries[df_countries['country'].isin(df_total_per_mil[df_total_per_mil['loans_per_mil'] >= min_loans_per_million]['country'])]
df_countries = df_countries.merge(df_total_per_mil[df_total_per_mil['total_loans_per_mil'] >= min_loans_per_million][['country', 'total_loans_per_mil']], on='country')
s_force_order = df_countries[['total_loans_per_mil', 'country_mpi']].sort_values('total_loans_per_mil', ascending=False).drop_duplicates()['country_mpi']

df_piv = df_countries.pivot(index='country_mpi', columns='sector', values='loans_per_mil')
df_piv.reset_index(level=0, inplace=True)
for s in sectors:
    df_piv[s] = df_piv[s].fillna(0)
    
#don't know how to keep this so i'm just going to make it again
df_piv['total_loans_per_mil'] = 0
for s in sectors:
    df_piv['total_loans_per_mil'] = df_piv['total_loans_per_mil'] + df_piv[s]
    
disc_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', 
               '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080', '#FFFFFF', '#000000']
sns.set_palette(disc_colors)
palette = itertools.cycle(sns.color_palette(palette=disc_colors, n_colors=22))

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

for s in sectors:    
    sns.barplot(x='total_loans_per_mil', y='country_mpi', data=df_piv,
            label=s, color=next(palette), order=s_force_order)
    #print('sector: ' + s + 'total_loans_per_mil: ' + str(df_piv[df_piv['country_mpi'] == 'Philippines - 0.052']['total_loans_per_mil']))
    df_piv['total_loans_per_mil'] = df_piv['total_loans_per_mil'] - df_piv[s]
    
ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Sector'
leg.set_title(new_title)
ax.set_title('Loans Per Million People', fontsize=15)
plt.show()


# On a per capita basis, the Philippines is actually 7th!  El Salvador on the other hand has a ton of usage - well over double the 2nd most popular country for Kiva loans!!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# More to come - just wanted to get what I've got so far published out!
# 
# Look it's me in the additional data snapshot!

# In[395]:


lender_df = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
lender_df['lender_URL'] = lender_df['permanent_name'].apply(lambda x: 'https://www.kiva.org/lender/' + str(x))
print(lender_df[lender_df['permanent_name'] == 'mikedev10']['lender_URL'])


# In[ ]:





# In[ ]:




