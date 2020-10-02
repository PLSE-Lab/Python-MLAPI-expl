#!/usr/bin/env python
# coding: utf-8

# # Greetings!
# 
# Hi, my name is Arthur Paulino and I'm a brazilian data scientist :)
# 
# I want to thank [Kiva](https://www.kiva.org/) for helping those who need. I acknowledge the huge amount of work that must have been put on such task.
# 
# Another big **thank you** for giving us the chance to be part of this. I am truly happy and I promise that I'll do my best in order to help.
# 
# # My style
# 
# I like to practice Data Science as if I am talking to the data. I make questions to myself and then I find a way to answer them by looking at the data. That's how my mind works.
# 
# Also, I'm more of a highly intuitive/abstract thinker so I hope I'll be able to make myself clear enough.
# 
# # Introduction
# 
# Kiva's initiative is based on Trust and Wisdom. Simply put, the workflow goes like this:
# 
# 1. People who need money get in touch with Kiva and borrow some;
# 2. The loan is posted on Kiva's online platform;
# 3. Lenders from all around the world become aware of the loan and decide whether they will help or not.
# 
# In my opinion, the genius of Kiva lies on lending money even before the loan is posted and funded and this is why I mentioned Trust and Wisdom. What a piece of beauty!
# 
# ## Challenge
# 
# So, why us, data scientists from Kaggle?! The [overview tab](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding) puts it very clearly:
# 
# > A good solution would connect the features of each loan or product to one of several poverty mapping datasets, which indicate the average level of welfare in a region on as granular a level as possible.
# 
# Thus, Kiva fellowes want to know more precisely where to focus on in order to improve their capabilities to help those who need our help **the most**.
# 
# # Strategy
# 
# My strategy is pretty straightforward. I want to group data by countries and train a model by using a national social indicator as a target. Then I want to use that model to predict the social indicator of rows grouped by regions, which is the most granular geographic level on these datasets. So my line of thought is the following:
# 
# 1. [Gather data and engineer features](#Gathering-data-and-engineering-features)
# 2. [Normalize national social indicators so that 0 means **poor** and 1 means **rich**](#Normalizing-national-social-indicators)
# 3. [Deal with missing data](#Dealing-with-missing-data)
# 4. [Create a feature that's a simple linear combination of the national social indicators then drop them](#Creating-a-feature-thats-a-linear-combination-of-the-social-indicators-then-dropping-them)
#     * I'll call it `bonstato`, which means *welfare* in [Esperanto](https://en.wikipedia.org/wiki/Esperanto)
# 5. [Perform data visualizations](#Performing-data-visualizations)
# 6. [Prepare train and test data](#Preparing-train-and-test-data)
# 7. [Train a model that maps loan features into `bonstato_national`](#Training-the-model)
# 8. [Predict `bonstato_regional`](#Predicting-bonstato_regional)
# 9. [Group up regions](#Grouping-up-regions)
# 10. [Conclusion](#Conclusion)

# # Gathering data and engineering features
# 
# This is going to be a long journey. Bear with me :)

# ### Basic setup

# In[ ]:


# importing usual libs
import pandas as pd
import numpy as np

# importing only specific functions from these libs to save up memory
from datetime import date as dt_date
from datetime import timedelta as dt_timedelta
from time import time as t_time
from gc import collect as gc_collect


# ## Importing raw loans data

# In[ ]:


raw_start = t_time()
start = t_time()
kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv', parse_dates=['posted_time', 'disbursed_time', 'funded_time']).drop(columns=['country_code', 'date', 'tags', 'use'])
print('Number of loans: {}'.format(kiva_loans.shape[0]))
print('Time taken: {:.3f}s'.format(t_time()-start))


# We've got exactly 671205 loans.
# 
# From now on we're going to perform several joins. As a safety measure, we should check if these joins aren't duplicating/removing loans.

# ### Defining auxiliary functions

# In[ ]:


def timer(given_start=None):
    if given_start is None:
        print('Time taken: {:.3f}s'.format(t_time()-start))
    else:
        print('Time taken: {:.3f}s'.format(t_time()-given_start))

def raw_timer():
    print('Total time taken: {:.3f}m'.format((t_time()-raw_start)/60))

def report(columns=None):
    timer()
    n_loans = int(kiva_loans['posted_time'].describe()['count'])
    if n_loans==671205:
        print('Loans are intact (671205)', end='\n' if columns is None else '\n\n')
    elif n_loans>671205:
        print('{} loans have been duplicated\n'.format(n_loans-671205), end='\n' if columns is None else '\n\n')
    else:
        print('{} loans are missing.\n'.format(671205-n_loans))
    if not columns is None:
        kiva_loans[['country']+columns].info()

kiva_country_names = {
    'Democratic Republic of the Congo': 'The Democratic Republic of the Congo',
    'Congo, Democratic Republic of the': 'The Democratic Republic of the Congo',
    'Republic of the Congo': 'Congo',
    'Congo, Republic of': 'Congo',
    'Myanmar': 'Myanmar (Burma)',
    'Ivory Coast': "Cote D'Ivoire",
    "Cote d'Ivoire": "Cote D'Ivoire",
    'VietNam': 'Vietnam',
    'Laos': "Lao People's Democratic Republic",
    'Bolivia, Plurinational State of': 'Bolivia',
    'Palestinian Territories': 'Palestine',
    'Somaliland region': 'Somalia',
    'Syrian Arab Republic': 'Syria',
    'Tanzania, United Republic of': 'Tanzania',
    'Viet Nam': 'Vietnam',
    'Palestine, State ofa': 'Palestine',
    'Macedonia, The former Yugoslav Republic of': 'Macedonia',
    'Moldova, Republic of': 'Moldova'
}

def get_standardized_country_name(obj):
    if type(obj) is float:
        return np.nan
    if obj in kiva_country_names:
        return kiva_country_names[obj]
    return obj

def get_standardized_region_name(obj):
    if type(obj) is float:
        return np.nan
    obj = obj.lower().replace('-', ' ').replace('.', ' ').replace(',', ' ').replace('_', ' ')
    words = obj.split(' ')
    for removed_word in ['', 'province', 'village', 'ville', 'district', 'town', 'city', 'region', 'reagion', 'community',
                         'commune', 'comunidad', 'odisha', 'aldea', 'kampong', 'kompong', 'ciudad']:
        while removed_word in words:
            words.remove(removed_word)
    return ' '.join(words)

def standardize_dataset(df):
    df['country'] = df['country'].apply(get_standardized_country_name)
    if 'region' in df.columns:
        df['region'] = df['region'].apply(get_standardized_region_name)

def rows_with_countries(df):
    return df[df['country'].notna()]

def rows_with_countries_and_regions(df):
    return df[(df['country'].notna()) & (df['region'].notna())]

def rows_with_coordinates(df):
    return df[(df['longitude'].notna()) & (df['latitude'].notna())]

def rows_without_coordinates(df):
    return df[(df['longitude'].isna()) | (df['latitude'].isna())]

def rows_with_countries_regions_and_coordinates(df):
    return df[(df['country'].notna()) & (df['region'].notna()) & (df['longitude'].notna()) & (df['latitude'].notna())]

def rows_with_loans(df):
    return df[df['posted_time'].notna()]


# ### Standardizing *kiva_loans*

# In[ ]:


start = t_time()
standardize_dataset(kiva_loans)
timer()


# ### Deriving features related to borrowers' genders
# 
# Related question:
# * Is the borrowers' group composition affected by the poverty level?
# 
# Features:
# * `borrower_count`: the number of borrowers
# * `borrower_female_count`: the percentage of female borrowers among the borrowers
# * `borrower_male_pct`: the percentage of male borrowers among the borrowers

# In[ ]:


start = t_time()
kiva_loans['borrower_count'] = kiva_loans['borrower_genders'].apply(lambda x: np.nan if type(x) is float else len(x.split(',')))
kiva_loans['borrower_female_pct'] = kiva_loans['borrower_genders'].apply(lambda x: np.nan if type(x) is float else x.count('female'))/kiva_loans['borrower_count']
kiva_loans['borrower_male_pct'] = 1.0 - kiva_loans['borrower_female_pct']
kiva_loans.drop(columns=['borrower_genders'], inplace=True)
gc_collect()
report(['borrower_count', 'borrower_female_pct', 'borrower_male_pct'])


# ### Deriving features related to the loan process durations
# 
# Related question:
# * Is Kiva's funding efficiency affectd by the poverty level?
# 
# Features:
# * `posting_delay_in_days`: time between the moment that the money was borrowed and the moment that the loan was posted on kiva's platform
# * `funding_delay_in_days`: time between the moment that the loan was posted on kiva's platform and the moment that the loan was completely funded
# * `total_delay_in_days`: time between the moment that the money was borrowed and the moment that the loan was completely funded

# In[ ]:


start = t_time()
kiva_loans['posting_delay_in_days'] = ((kiva_loans['posted_time'] - kiva_loans['disbursed_time'])/np.timedelta64(1, 's'))/86400
kiva_loans['posting_delay_in_days'] = kiva_loans['posting_delay_in_days'].apply(lambda x: max(0, x))
kiva_loans['funding_delay_in_days'] = ((kiva_loans['funded_time'] - kiva_loans['posted_time'])/np.timedelta64(1, 's'))/86400
kiva_loans['total_delay_in_days'] = kiva_loans['posting_delay_in_days'] + kiva_loans['funding_delay_in_days']
report(['posting_delay_in_days', 'funding_delay_in_days', 'total_delay_in_days'])


# ### Deriving features related to the money flow
# 
# Related question:
# * Is the money flow affected by the poverty level? Maybe it reflects the urgency of the loan...
# 
# Features:
# * `funded_amount_per_lender`: the average amount of money lended by each lender
# * `funded_amount_per_day`: the average amount of money funded each day
# * `loan_amount_per_borrower`: the average amount of money borrowed by each borrower
# * `loan_amount_per_month`: the average amount of money used by the borrower in each month

# In[ ]:


start = t_time()
kiva_loans['funded_amount_per_lender'] = kiva_loans['funded_amount']/kiva_loans['lender_count']
kiva_loans['funded_amount_per_day'] = kiva_loans['funded_amount']/kiva_loans['funding_delay_in_days']
kiva_loans['loan_amount_per_borrower'] = kiva_loans['loan_amount']/kiva_loans['borrower_count']
kiva_loans['loan_amount_per_month'] = kiva_loans['loan_amount']/kiva_loans['term_in_months']
report(['funded_amount_per_lender', 'funded_amount_per_day', 'loan_amount_per_borrower', 'loan_amount_per_month'])


# ### Deriving features related to missing loan funds
# 
# Related question:
# * Is the success of the loan funding process affected by poverty levels?
# 
# Feature:
# * `missing_funds`: the amount of money that wasn't funded
# * `missing_funds_pct`: the percentage of money that wasn't funded

# In[ ]:


start = t_time()
kiva_loans['missing_funds'] = kiva_loans['loan_amount'] - kiva_loans['funded_amount']
kiva_loans['missing_funds_pct'] = kiva_loans['missing_funds']/kiva_loans['loan_amount']
report(['missing_funds', 'missing_funds_pct'])


# ## Gathering several national social indicators
# 
# The following join will add some countries without loans to our dataset. We need them so we can compute `bonstato` across the world properly.

# In[ ]:


start = t_time()
country_stats = pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv').drop(columns=['country_name', 'country_code', 'country_code3']).rename(columns={'kiva_country_name':'country', 'region':'continent_region'})
standardize_dataset(country_stats)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(country_stats), how='outer', on='country')
del country_stats
gc_collect()
report(['continent', 'continent_region', 'population', 'population_below_poverty_line', 'hdi', 'life_expectancy', 'expected_years_of_schooling', 'mean_years_of_schooling', 'gni'])


# ### Deriving features *gni_per_capta* and *mean_years_of_schooling_pct*

# In[ ]:


start = t_time()
kiva_loans['gni_per_capta'] = kiva_loans['gni']/kiva_loans['population']
kiva_loans.drop(columns=['population', 'gni'], inplace=True)
kiva_loans['mean_years_of_schooling_pct'] = kiva_loans['mean_years_of_schooling']/kiva_loans['expected_years_of_schooling']
gc_collect()
report(['gni_per_capta', 'mean_years_of_schooling_pct'])


# ### Gathering Happiness Scores

# In[ ]:


delta = (kiva_loans['funded_time'].max() - kiva_loans['disbursed_time'].min()).total_seconds()/2
print('Date occurrences center: ~{}'.format(kiva_loans['disbursed_time'].min() + dt_timedelta(seconds=delta)))


# Let's use the dataset from 2015 because:
# 
# 1. It has more rows (I checked columns metrics tabs from [here](https://www.kaggle.com/unsdsn/world-happiness/data))
# 2. Date occurrences are somewhat centered at 2015.

# In[ ]:


start = t_time()
happiness_scores = pd.read_csv('../input/world-happiness/2015.csv', usecols=['Country', 'Happiness Score']).rename(columns={'Country':'country', 'Happiness Score':'happiness'})
standardize_dataset(happiness_scores)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(happiness_scores)[(happiness_scores['country']!='Congo (Brazzaville)') & (happiness_scores['country']!='Congo (Kinshasa)')], how='outer', on='country')
kiva_loans.loc[(kiva_loans['country']=='Congo') & (kiva_loans['region']=='Brazzaville'), 'happiness'] = happiness_scores.groupby('country').mean().at['Congo (Brazzaville)', 'happiness']
kiva_loans.loc[(kiva_loans['country']=='Congo') & (kiva_loans['region']=='Kinshasa'), 'happiness'] = happiness_scores.groupby('country').mean().at['Congo (Kinshasa)', 'happiness']
del happiness_scores
gc_collect()
report(['happiness'])


# ### Gathering Global Peace Index
# 
# Depending on the year that the loan was posted, we will use a different GPI.
# 
# Loans posted in 2017 will have GPI's from 2016 because the dataset doesn't have GPI's from 2017.

# In[ ]:


start = t_time()
gpi = pd.read_csv('../input/gpi2008-2016/gpi_2008-2016.csv')
gpi['score_2013'].fillna(gpi['score_2016'], inplace=True)
gpi['score_2014'].fillna(gpi['score_2016'], inplace=True)
gpi['score_2015'].fillna(gpi['score_2016'], inplace=True)

standardize_dataset(gpi)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(gpi), how='outer', on='country')

kiva_loans['gpi'] = np.nan
for i in range(kiva_loans.shape[0]):
    year = min(2016,kiva_loans.at[i, 'posted_time'].year)
    kiva_loans.at[i, 'gpi'] = kiva_loans.at[i, 'score_'+str(year)]

kiva_loans.drop(columns=[column for column in kiva_loans if column.count('score_')>0], inplace=True)
del gpi
gc_collect()
report(['gpi'])


# ## Importing *loan_theme_ids.csv* and gathering *Loan Theme Type* feature

# In[ ]:


start = t_time()
loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv').drop(columns=['Loan Theme ID', 'Partner ID'])
kiva_loans = pd.merge(left=kiva_loans, right=loan_theme_ids, how='left', on='id')
del loan_theme_ids
gc_collect()
report(['Loan Theme Type'])


# ## Importing *loan_coords.csv* and gathering *longitude* and *latitude*

# In[ ]:


start = t_time()
loan_coords = pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv').rename(columns={'loan_id':'id'})
kiva_loans = pd.merge(left=kiva_loans, right=loan_coords, how='left', on='id')
del loan_coords
gc_collect()
report(['longitude', 'latitude'])


# ## Importing *loan_themes_by_region.csv*

# In[ ]:


start = t_time()
loan_themes_by_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv').rename(columns={'Partner ID':'partner_id'})
standardize_dataset(loan_themes_by_region)
timer()


# ### Gathering feature *Field Partner Name*

# In[ ]:


start = t_time()
kiva_loans = pd.merge(left=kiva_loans, right=loan_themes_by_region[['partner_id', 'Field Partner Name']].drop_duplicates(subset=['partner_id']), how='left', on='partner_id').drop(columns=['partner_id'])
report(['Field Partner Name', 'longitude', 'latitude'])


# ### Gathering features *rural_pct*, *longitude* (more) and *latitude* (more) by merging on *country* and *region*

# In[ ]:


start = t_time()
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries_and_regions(loan_themes_by_region)[['country', 'region', 'rural_pct', 'lon', 'lat']].drop_duplicates(subset=['country', 'region']), how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['lon'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['lat'], inplace=True)
kiva_loans.drop(columns=['lon', 'lat'], inplace=True)
gc_collect()
report(['rural_pct', 'longitude', 'latitude', 'region'])


# ### Gathering features *region* (more) and *rural_pct* (more) by merging on *longitude* and *latitude*

# In[ ]:


start = t_time()
loan_themes_by_region.rename(columns={'lon':'longitude', 'lat':'latitude', 'region':'region_new', 'rural_pct':'rural_pct_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_coordinates(loan_themes_by_region)[['longitude', 'latitude', 'region_new', 'rural_pct_new']].drop_duplicates(subset=['longitude', 'latitude']), how='left', on=['longitude', 'latitude'])
kiva_loans['region'].fillna(kiva_loans['region_new'], inplace=True)
kiva_loans['rural_pct'].fillna(kiva_loans['rural_pct_new'], inplace=True)
kiva_loans.drop(columns=['region_new', 'rural_pct_new'], inplace=True)
gc_collect()
report(['region', 'rural_pct'])


# ### Deriving and gathering feature *forkiva_pct* by merging on *country* and *region*

# In[ ]:


start = t_time()
loan_themes_by_region['forkiva'].replace(['Yes', 'No'], [1, 0], inplace=True)
loan_themes_by_region.rename(columns={'region_new':'region', 'forkiva':'forkiva_pct'}, inplace=True)
loan_themes_by_region_avg = loan_themes_by_region.groupby(['country', 'region']).mean().reset_index()
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries_and_regions(loan_themes_by_region_avg)[['country', 'region', 'forkiva_pct']], how='left', on=['country', 'region'])
gc_collect()
report(['forkiva_pct'])


# ### Gathering feature *forkiva_pct* (more) by merging on *latitude* and *longitude*

# In[ ]:


start = t_time()
loan_themes_by_region.rename(columns={'forkiva_pct':'forkiva_pct_new'}, inplace=True)
loan_themes_by_region_avg = loan_themes_by_region.groupby(['latitude', 'longitude']).mean().reset_index()
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_coordinates(loan_themes_by_region_avg)[['latitude', 'longitude', 'forkiva_pct_new']], how='left', on=['longitude', 'latitude'])
kiva_loans['forkiva_pct'].fillna(kiva_loans['forkiva_pct_new'], inplace=True)
kiva_loans.drop(columns=['forkiva_pct_new'], inplace=True)
del loan_themes_by_region_avg, loan_themes_by_region
gc_collect()
report(['forkiva_pct'])


# ## Importing *kiva_mpi_region_locations.csv*

# In[ ]:


start = t_time()
kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv', usecols=['country', 'region', 'MPI', 'lon', 'lat'])
kiva_mpi_region_locations = kiva_mpi_region_locations[kiva_mpi_region_locations['MPI'].notna()]
standardize_dataset(kiva_mpi_region_locations)
report(['longitude', 'latitude'])


# ### Gathering features *MPI*, *longitude* (more) and *latitude* (more) by merging on *country* and *region*

# In[ ]:


start = t_time()
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries_and_regions(kiva_mpi_region_locations), how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['lon'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['lat'], inplace=True)
kiva_loans.drop(columns=['lon', 'lat'], inplace=True)
report(['MPI', 'longitude', 'latitude'])


# ### Gathering feature *MPI* (more) by merging on *longitude* and *latitude*

# In[ ]:


start = t_time()
kiva_mpi_region_locations.rename(columns={'lon':'longitude', 'lat':'latitude', 'MPI':'MPI_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_coordinates(kiva_mpi_region_locations).drop(columns=['country', 'region']).drop_duplicates(subset=['longitude', 'latitude']), how='left', on=['longitude', 'latitude'])
kiva_loans['MPI'].fillna(kiva_loans['MPI_new'], inplace=True)
kiva_loans.drop(columns=['MPI_new'], inplace=True)
del kiva_mpi_region_locations
gc_collect()
report(['MPI'])


# Unfortunately, only ~74.2k rows were properly filled with region specific MPIs

# ## Importing *MPI_subnational.csv* to get more MPIs by mergin on *country*
# 
# This file contains national MPIs, so we're going to use them in order to fill some MPIs that weren't filled previously. In the end, the MPI feature will be used as a national social indicator.

# In[ ]:


start = t_time()
MPI_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv', usecols=['Country', 'MPI National']).drop_duplicates(subset=['Country']).rename(columns={'Country':'country'})
standardize_dataset(MPI_subnational)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(MPI_subnational), how='left', on='country')
kiva_loans['MPI'].fillna(kiva_loans['MPI National'], inplace=True)
kiva_loans.drop(columns=['MPI National'], inplace=True)
del MPI_subnational
gc_collect()
report(['MPI'])


# Now we have ~583.5k rows properly filled, which is much better.
# 
# Notice that the region specific MPIs won't distort their countries' MPIs too much when we group by countries, since there are just a few of them. But they will help a lot when we're to compute our region specific welfare feature, `bonstato_regional`.

# ## Importing *loan_lenders.csv* and gathering *lenders* (to be dropped)

# In[ ]:


start = t_time()
loans_lenders = pd.read_csv('../input/additional-kiva-snapshot/loans_lenders.csv').rename(columns={'loan_id':'id'})
loans_lenders['lenders'] = loans_lenders['lenders'].apply(lambda x: x.replace(' ', ''))
kiva_loans = pd.merge(left=kiva_loans, right=loans_lenders, how='left', on='id').drop(columns=['id'])
del loans_lenders
gc_collect()
report(['lenders'])


# ## Importing *lenders.csv* so we know for how long the lenders have been acting on Kiva
# 
# This step creates a dictionary for faster search.

# In[ ]:


start = t_time()
lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv', usecols=['permanent_name', 'member_since'])
lenders_dict = {}
for i in range(lenders.shape[0]):
    lenders_dict[lenders.at[i, 'permanent_name']] = lenders.at[i, 'member_since']
del lenders
gc_collect()
timer()


# ### Deriving feature related to lenders experience on Kiva
# 
# Related question:
# * Is the average lender's experience related to the poverty level?
# 
# Feature:
# * `lenders_experience_in_days_avg`: the average experience of the lenders **when the loan was posted on Kiva's platform**.

# In[ ]:


start = t_time()

def get_lender_member_since(permanent_name):
    if permanent_name in lenders_dict:
        return lenders_dict[permanent_name]
    return np.nan

def avg_member_since(obj):
    if type(obj) is float:
        return np.nan
    member_sinces = []
    permanent_names = obj.split(',')
    for permanent_name in permanent_names:
        member_sinces.append(get_lender_member_since(permanent_name))
    return np.array(member_sinces).mean()

kiva_loans['lenders_members_since_avg'] = kiva_loans['lenders'].apply(avg_member_since)
kiva_loans['lenders_experience_in_days_avg'] = ((kiva_loans['posted_time'] - dt_date(1970,1,1)) / np.timedelta64(1, 's') - kiva_loans['lenders_members_since_avg'])/86400
kiva_loans.drop(columns=['lenders_members_since_avg', 'lenders'], inplace=True)

del lenders_dict
gc_collect()

report(['lenders_experience_in_days_avg'])


# # Normalizing national social indicators
# 
# Some features will be flagged with "**(decr)**", meaning that, originally, the higher they are, the poorer the countries are. These features will be inverted in the normalization process.
# 
# National social indicators:
# 
# * population_below_poverty_line **(decr)**
# * hdi
# * life_expectancy
# * expected_years_of_schooling
# * mean_years_of_schooling
# * gni_per_capta
# * mean_years_of_schooling_pct
# * MPI **(decr)**
# * happiness
# * gpi **(decr)**

# In[ ]:


start = t_time()

def normalize_feature(df, feature, decr=False):
    if decr:
        df[feature] = -1.0 * df[feature]
    df[feature] = df[feature]-df[feature].min()
    df[feature] = df[feature]/(df[feature].max()-df[feature].min())

indicators = ['population_below_poverty_line', 'MPI', 'hdi', 'life_expectancy', 'expected_years_of_schooling', 'mean_years_of_schooling', 'gni_per_capta', 'mean_years_of_schooling_pct', 'happiness', 'gpi']

for indicator in indicators:
    if indicator in ['population_below_poverty_line', 'MPI', 'gpi']:
        normalize_feature(kiva_loans, indicator, decr=True)
    else:
        normalize_feature(kiva_loans, indicator)

timer()


# # Dealing with missing data

# ## Filling empty continents

# In[ ]:


print(set(kiva_loans[kiva_loans['continent'].isna()]['country']))


# * According to [this link](http://www.answers.com/Q/What_continent_is_Guam_in), we can say that Guam is located at Oceania;
# 
# * According to [this link](http://www.answers.com/Q/What_continent_is_Guam_in), we can say that Vanuatu is located at Oceania;
# 
# * North Cyprus is from Asia;
# 
# * Taiwan is from Asia;
# 
# * Saint Vincent and the Grenadines and the Virgin Islands are from Americas.

# In[ ]:


start = t_time()
kiva_loans.loc[kiva_loans['country']=='Guam', 'continent'] = 'Oceania'
kiva_loans.loc[kiva_loans['country']=='Vanuatu', 'continent'] = 'Oceania'
kiva_loans.loc[kiva_loans['country']=='North Cyprus', 'continent'] = 'Asia'
kiva_loans.loc[kiva_loans['country']=='Taiwan', 'continent'] = 'Asia'
kiva_loans.loc[kiva_loans['country']=='Saint Vincent and the Grenadines', 'continent'] = 'Americas'
kiva_loans.loc[kiva_loans['country']=='Virgin Islands', 'continent'] = 'Americas'
report(['continent', 'longitude', 'latitude'])


# ## Inferring *longitude* and *latitude* by region names similarities
# 
# 
# Some regions are spelled very similarly, even though some of them may be labeled with coordinates and some of them may not. In these cases, we can infer the coordinates of some regions by copying the coordinates of the most similar region (by name), above a certain similarity threshold of course. For the sake of efficiency, let's require that the region names must at least start with the same letter.
# 
# I'm going to use a similarity threshhold of 0.75.

# In[ ]:


from nltk import edit_distance

SIMILARITY_THRESHHOLD = 0.75

start = t_time()

all_regions = rows_with_countries_and_regions(kiva_loans[['country', 'region', 'longitude', 'latitude']]).groupby(['country', 'region']).head().drop_duplicates(subset=['country', 'region']).reset_index(drop=True)

regions_with_coordinates = rows_with_coordinates(all_regions).rename(columns={'region':'region_with_coordinates'})
regions_with_coordinates['region_with_coordinates[0]'] = regions_with_coordinates['region_with_coordinates'].apply(lambda x: x[0])
regions_without_coordinates = rows_without_coordinates(all_regions).rename(columns={'region':'region_without_coordinates'}).drop(columns=['longitude', 'latitude'])
regions_without_coordinates['region_without_coordinates[0]'] = regions_without_coordinates['region_without_coordinates'].apply(lambda x: x[0])

cartesian = pd.merge(left=regions_without_coordinates, right=regions_with_coordinates, how='inner', on='country')
cartesian = cartesian[cartesian['region_without_coordinates[0]']==cartesian['region_with_coordinates[0]']].drop(columns=['region_without_coordinates[0]', 'region_with_coordinates[0]']).reset_index(drop=True)
cartesian['region_names_similarity'] = np.nan
for i in range(cartesian.shape[0]):
    region1 = cartesian.at[i, 'region_with_coordinates']
    region2 = cartesian.at[i, 'region_without_coordinates']
    cartesian.at[i, 'region_names_similarity'] = 1 - edit_distance(region1, region2)/(len(region1+region2))
cartesian.sort_values(by='region_names_similarity', ascending=False, inplace=True)
cartesian = cartesian[cartesian['region_names_similarity']>SIMILARITY_THRESHHOLD].drop_duplicates(subset=['region_without_coordinates'])
cartesian = cartesian.drop(columns=['region_with_coordinates', 'region_names_similarity']).rename(columns={'region_without_coordinates':'region', 'longitude':'longitude_new', 'latitude':'latitude_new'})
kiva_loans = pd.merge(left=kiva_loans, right=cartesian, how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['longitude_new'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['latitude_new'], inplace=True)
kiva_loans.drop(columns=['longitude_new', 'latitude_new'], inplace=True)

del cartesian, all_regions, regions_with_coordinates, regions_without_coordinates
gc_collect()

report(['longitude', 'latitude', 'region'])


# ## Grouping regions by *longitude* and *latitude*
# 
# Some regions are spelled differently but they refer to the very same place. Let's group them up and explicit their various names separated by `/`.
# 
# Regions without names will be called `'Unknown-<ID>'` when grouped.
# 
# Regions that can't group up will be left untouched.

# In[ ]:


start = t_time()

class Counter:
    def __init__(self):
        self.counter = 0
    def tick(self):
        self.counter += 1
        return str(self.counter-1)

unknown_id_counter = Counter()

def join_names(obj):
    unknown_name = 'Unknown-'+unknown_id_counter.tick()
    if type(obj) is float:
        return unknown_name
    obj = sorted([element if not type(element) is float else unknown_name for element in set(obj)])
    return '/'.join(obj)

new_region_names = rows_with_coordinates(kiva_loans).groupby(['longitude', 'latitude'])['region'].apply(join_names).reset_index()
kiva_loans.rename(columns={'region':'region_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_coordinates(new_region_names), how='left', on=['longitude', 'latitude'])
kiva_loans['region'].fillna(kiva_loans['region_new'], inplace=True)
kiva_loans.drop(columns=['region_new'], inplace=True)

del new_region_names
gc_collect()

report(['region'])


# ## Inferring missing data by grouping loans by *country/region*, *country*, *continent_region* and *continent*
# 
# Now we fill missing data (except for longitudes and latitudes) with mean values, first grouping by `country` and `region`, then grouping by `country`, then grouping by `continent_region`, then grouping by `continent`. If there are still missing values, we fill them with their global average values.
# 
# Missing `longitude` and `latitude` values will be filled when grouping only by `region`.
# 
# Then we will no longer need `continent_region`.

# In[ ]:


start = t_time()

def replace_numeric_features_with_groupby_mean(groupby):
    group = kiva_loans[groupby + numeric_features].groupby(groupby).mean().reset_index().rename(columns=numeric_features_map)
    merge = pd.merge(left=kiva_loans, right=group, how='left', on=groupby)
    for feature in numeric_features:
        kiva_loans[feature].fillna(merge[numeric_features_map[feature]], inplace=True)
    del merge, group
    gc_collect()

numeric_features = [column for column in kiva_loans.drop(columns=['longitude', 'latitude']).columns if kiva_loans[column].dtype==np.int64 or kiva_loans[column].dtype==np.float64]
numeric_features_map = {}
for feature in numeric_features:
    numeric_features_map[feature] = feature+'_new'
for groupby in [['country', 'region'], ['country'], ['continent_region'], ['continent']]:
    replace_numeric_features_with_groupby_mean(groupby)
for feature in numeric_features:
    mean = kiva_loans[feature].mean()
    kiva_loans[feature].fillna(mean, inplace=True)

numeric_features = ['longitude', 'latitude']
numeric_features_map = {}
for feature in numeric_features:
    numeric_features_map[feature] = feature+'_new'
replace_numeric_features_with_groupby_mean(['country', 'region'])

kiva_loans.drop(columns=['continent_region'], inplace=True)

del numeric_features_map, numeric_features
gc_collect()

report([column for column in kiva_loans.columns if kiva_loans[column].dtype==np.int64 or kiva_loans[column].dtype==np.float64])


# # Creating a feature thats a linear combination of the social indicators then dropping them
# 
# We're using many social indicators for two reasons:
# 
# 1. We're trying to take into consideration many facets of the population groups;
# 2. Our data has some missing values, thus using more columns decreases the effects of missing data in specific columns.
# 
# The weights came out of my mind. I find *peace* and *happiness* really important for any nation, but again... this is my personal belief!

# In[ ]:


start = t_time()

kiva_loans['bonstato_national'] = (2*kiva_loans['population_below_poverty_line'] + 
                                   kiva_loans['MPI'] +
                                   3*kiva_loans['hdi'] +
                                   2*kiva_loans['life_expectancy'] +
                                   kiva_loans['expected_years_of_schooling'] +
                                   kiva_loans['mean_years_of_schooling'] +
                                   kiva_loans['gni_per_capta'] +
                                   kiva_loans['mean_years_of_schooling_pct'] +
                                   4*kiva_loans['happiness'] +
                                   4*kiva_loans['gpi']
                                  )/20

kiva_loans.drop(columns=indicators, inplace=True)
gc_collect()

report(['bonstato_national'])


# # Performing data visualizations
# Let's see some stuff!

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')


# ## Visualizing time series

# In[ ]:


def plot_time_series(features, kind, series=['day', 'month', 'year', 'date']):
    
    start = t_time()
    
    if kind=='mean':
        aggreg=np.mean
    elif kind=='total':
        aggreg=np.sum
    
    vis_orig = kiva_loans[features+['posted_time']].copy()
    
    if 'missing_funds_pct' in features:
        vis_orig['missing_funds_pct'] = 1500*vis_orig['missing_funds_pct']

    if 'day' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.day
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('day of month')
        plt.ylabel(kind)

    if 'month' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.month
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('month')
        plt.ylabel(kind)

    if 'year' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.year
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('year')
        plt.ylabel(kind)

    if 'date' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.date
        vis.groupby('posted_time').apply(aggreg).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('date')
        plt.ylabel(kind)
    
    del vis_orig, vis
    gc_collect()
    timer(start)


# ### How does Kiva's money flow look like?

# In[ ]:


plot_time_series(['loan_amount', 'funded_amount'], 'total', ['day', 'month', 'year'])


# ### How do loans' and funds' mean values vary through time?

# In[ ]:


plot_time_series(['loan_amount', 'funded_amount'], 'mean', ['day', 'month', 'year'])


# ### The difference between *loan_amount* and *funded_amount* reflects Kiva's need for money. Can we see some pattern through time?

# In[ ]:


plot_time_series(['missing_funds', 'missing_funds_pct'], 'mean')


# I multiplied `missing_funds_pct` by 1500 to make these graphics look like each other. And they certainly do!
# 
# These features are strongly correlated, but the difference between their magnitudes can vary drastically, depending on the total loan amount in each contry and region.
# 
# Anyway, June seems to be a though month for Kiva...

# ### How does the lenders' disposal to lend money vary through time?

# In[ ]:


plot_time_series(['funded_amount_per_lender'], 'mean')


# ### *funded_amount_per_day* reflects Kiva's efficiency on gathering funds for loans. Let's see how it behaves through time.

# In[ ]:


plot_time_series(['funded_amount_per_day'], 'mean', ['month', 'year', 'date'])


# Hmm, what would those peaks be? Marketing campaigns? I don't know...

# ### Delays reflect the time interval for which Kiva's debt hasn't been completely paid, thus increasing delays can turn into a difficult situation.

# In[ ]:


plot_time_series(['posting_delay_in_days', 'funding_delay_in_days', 'total_delay_in_days'], 'mean', ['year', 'date'])


# Very interesting! Even though `funded_amount_per_lender` and `funded_amount_per_day` seem to be increasing and `funding_delay_in_days` seems to be decreasing, the average of `missing_funds` seems to be increasing. What's happening?
# 
# Maybe, just maybe, the increases in `funded_amount_per_lender` and `funded_amount_per_day` are reflecting inflation rates and lenders are simply not catchung up. Thus,  `missing_funds` rates go up and up.
# 
# The decrease in `funding_delay_in_days` may be caused by the decrease in the mean loan amount, as shown [here](#How-does-the-average-loan/fund-amount-vary-through-time?).

# ### Is the average experience of lenders on Kiva increasing? In other words, do Kiva's lenders stay on Kiva or do they leave after lending some money?

# In[ ]:


plot_time_series(['lenders_experience_in_days_avg'], 'mean', ['year', 'date'])


# ### Is there a pattern in when people group up in order to ask Kiva for loans?

# In[ ]:


plot_time_series(['borrower_count'], 'mean', ['year'])


# In[ ]:


plot_time_series(['borrower_male_pct'], 'mean', ['year'])


# ## Visualizing categorical features
# 
# Notice that I am taking `currency` and `Field Partner Name`  into consideration because some of their values appear more than once in some contries **and** some countries have more than one of some of their values. We want to be **more granular** than `country` and that's why we should not use features like `country`, `continent_region` or `continent`.

# In[ ]:


start = t_time()

for feature in ['currency', 'Field Partner Name']:
    for country in set(kiva_loans['country']):
        if country==np.nan:
            continue
        feature_values = list(set(kiva_loans[kiva_loans['country']==country][feature]))
        if len(feature_values) > 1 and np.nan not in feature_values:
            print('Example of country with more than 1 {}: {}'.format(feature, country))
            print('{} values: {}\n'.format(feature, '; '.join(feature_values)))
            break
    for feature_value in set(kiva_loans[feature]):
        if feature_value==np.nan:
            continue
        countries = list(set(kiva_loans[kiva_loans[feature]==feature_value]['country']))
        if len(countries) > 1 and not np.nan in countries and feature_value!='USD':
            print('Example of {} present in more than 1 country: {}'.format(feature, feature_value))
            print('Countries: {}\n'.format('; '.join(countries)))
            break

timer()


# In[ ]:


categorical_features = ['activity', 'sector', 'currency', 'Loan Theme Type', 'Field Partner Name', 'repayment_interval']


# In[ ]:


base_parameters = ['loan_amount', 'funded_amount', 'missing_funds']

def plot_categorical_feature(feature):
    
    start = t_time()
    
    vis = kiva_loans[base_parameters+[feature]].copy().groupby(feature).sum()
    vis['money_involved'] = vis['loan_amount'] + vis['funded_amount']
    vis['missing_funds_pct'] = vis['missing_funds']/vis['loan_amount']
    vis.drop(columns=base_parameters, inplace=True)
    
    n_categories = len(vis.index)
    
    for parameter in ['money_involved', 'missing_funds_pct']:
        if n_categories > 40:
            vis[[parameter]].sort_values(by=[parameter], ascending=False).head(20).plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1])
            plt.title('top 20 {}: {}'.format(feature, parameter))
            plt.xlabel('')
        else:
            vis[[parameter]].sort_values(by=[parameter], ascending=False).plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1])
            plt.title('{}: {}'.format(feature, parameter))
            plt.xlabel('')
        plt.show()
        plt.close('all')
    
    del vis
    gc_collect()
    timer(start)


# In[ ]:


for feature in categorical_features:
    plot_categorical_feature(feature)


# It's quite interesting how, in general, feature values that resemble more delicate social conditions (e.g. agriculture, farming, food, underserved, monthly/irregular repayment intervals etc) have more money involved (borrowed+funded) and feature values that resemble more stable social conditions (e.g. technology, entertainment, transportation, uber drivers, bullet repayment interval etc) have higher rates of missing funds percentage. It makes total sense and is a sign that our categorical features are pretty expressive.

# ## Visualizing relationships between loans' numerical features and *bonstato_national*
# Let's group the data by countries and check some correlations.

# In[ ]:


start = t_time()

kiva_loans_by_countries = rows_with_loans(kiva_loans).drop(columns=['latitude', 'longitude']).groupby('country').mean().reset_index()
corrs = kiva_loans_by_countries.corr()[['bonstato_national']].drop('bonstato_national')

pos_corrs = corrs[corrs['bonstato_national']>0][['bonstato_national']].sort_values(by='bonstato_national', ascending=False)
sns.heatmap(pos_corrs, ax=plt.subplots(figsize=(7,5))[1])
plt.title('Positive correlations between loans\' numerical features and bonstato_national')

neg_corrs = corrs[corrs['bonstato_national']<0][['bonstato_national']].sort_values(by='bonstato_national', ascending=True)
sns.heatmap(neg_corrs, ax=plt.subplots(figsize=(7,5))[1])
plt.title('Negative correlations between loans\' numerical features and bonstato_national')

del corrs, pos_corrs, neg_corrs
gc_collect()
timer()


# Amazing! These images are pretty revealing.
# 
# Borrowers from richer countries borrow money for longer periods of time. Their loans are funded less often and the percentages of men among the borrowers is higher than in poor countries.
# 
# Borrowers from poorer countries group up in higher numbers in order to ask Kiva for loans and their groups of borrowers have higher percentages of women than in richer countries. These borrowers also borrow money with higher repayment parcels.
# 
# Poorer countries are better supported by lenders be *i*) loans are funded by more lenders and *ii*) funded amounts reach higher rates.
# 
# Curious fact: even though loans from poorer countries take slightly longer to be posted on Kiva's online platform, their delays between disbursing times and total funding times are still smaller than the delays of loans made in richer countries.

# ## Visualizing distributions of loans' categorical features among rich/poor countries

# In[ ]:


def plot_categorical_feature_with_bonstato_threshold(feature, bonstato_threshold):
    
    start = t_time()
    
    n_loans_rich = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']>bonstato_threshold)].shape[0]
    n_loans_poor = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']<=bonstato_threshold)].shape[0]
    
    vis_rich = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']>bonstato_threshold)].groupby(feature)[feature].count().sort_values(ascending=False)/n_loans_rich
    vis_poor = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']<=bonstato_threshold)].groupby(feature)[feature].count().sort_values(ascending=False)/n_loans_poor
    
    top_rich = False
    if len(vis_rich.index) > 20:
        top_rich = True
        vis_rich = vis_rich.head(20)
    
    top_poor = False
    if len(vis_poor.index) > 20:
        top_poor = True
        vis_poor = vis_poor.head(20)
    
    vis_rich.plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1], color='blue')
    plt.xlabel('')
    title = feature+' on rich countries'
    if top_rich:
        title = 'top 20 ' + title
    plt.title(title)
    
    vis_poor.plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1], color='red')
    plt.xlabel('')
    title = feature+' on poor countries'
    if top_poor:
        title = 'top 20 ' + title
    plt.title(title)
    
    plt.show()
    plt.close('all')
    
    del vis_rich, vis_poor
    gc_collect()
    timer(start)


# Let's find the [Pareto](https://en.wikipedia.org/wiki/Pareto_principle) threshold on `bonstato_national`. First, let's see how many countries are helped by Kiva.

# In[ ]:


kiva_loans_by_countries.shape[0]


# Okay, since 20% of 87 = 17, we sort countries by `bonstato_national` and find the mean value of `bonstato_national` between the 17th (`index=16`) and the 18th (`index=17`) country.

# In[ ]:


start = t_time()
bonstato_threshold = kiva_loans_by_countries.sort_values(by='bonstato_national', ascending=False).reset_index().loc[16:17, 'bonstato_national'].mean()
del kiva_loans_by_countries
gc_collect()
print('Bonstato threshold: {}'.format(bonstato_threshold))
report()


# The following cell generates several output files containing the distributions of loans' categorical features among rich and poor countries.

# In[ ]:


for feature in categorical_features:
    plot_categorical_feature_with_bonstato_threshold(feature, bonstato_threshold)


# # Preparing train and test data

# ## Ignoring countries without the presence of Kiva

# In[ ]:


start = t_time()
kiva_loans = rows_with_loans(kiva_loans)
report([])


# We're back to the original height: each row represents a loan

# ## Fixing some coordinates
# 
# After plotting some regions, I noticed that a few of them are misplaced. It's worth it fixing them "manually".

# In[ ]:


start = t_time()

def replace_regions(continent, lat_less_than=np.inf, lat_greater_than=-np.inf, lon_less_than=np.inf, lon_greater_than=-np.inf):
    kiva_subset = kiva_loans[(kiva_loans['continent']==continent)
                             & (kiva_loans['longitude']<=lon_less_than)
                             & (kiva_loans['longitude']>=lon_greater_than)
                             & (kiva_loans['latitude']<=lat_less_than)
                             & (kiva_loans['latitude']>=lat_greater_than)]
    
    countries = list(set(kiva_subset['country']))
    for country in countries:
        longitude_median = kiva_loans[kiva_loans['country']==country]['longitude'].median()
        latitude_median = kiva_loans[kiva_loans['country']==country]['latitude'].median()
        kiva_loans.at[kiva_subset[kiva_subset['country']==country].index, 'longitude'] = longitude_median
        kiva_loans.at[kiva_subset[kiva_subset['country']==country].index, 'latitude'] = latitude_median
        
kiva_loans.loc[(kiva_loans['country']=='Kenya') & (kiva_loans['region']=='mogadishu, somalia'), 'country'] = 'Somalia'
kiva_loans.loc[(kiva_loans['country']=='Mexico') & (kiva_loans['region']=='tierra amarilla'), 'country'] = 'United States'
kiva_loans.loc[(kiva_loans['country']=='Peru') & (kiva_loans['region']=='santa fe'), 'country'] = 'United States'

replace_regions('Americas', lon_greater_than=-40)
replace_regions('Oceania', lat_greater_than=30)
replace_regions('Asia', lon_less_than=15)
replace_regions('Asia', lon_less_than=31, lat_less_than=31)
replace_regions('Asia', lon_less_than=40, lat_less_than=10)
replace_regions('Africa', lon_greater_than=50)
replace_regions('Africa', lon_less_than=-18)
replace_regions('Africa', lat_greater_than=20)

timer()


# ## One-Hot-Encoding categorical features

# In[ ]:


start = t_time()
kiva_loans = pd.get_dummies(data=kiva_loans, dummy_na=True, drop_first=True, columns=categorical_features)
report()


# ## Creating *kiva_countries* and *kiva_regions*
# 
# `kiva_countries` is our trainning data and `kiva_regions` is our test data.

# In[ ]:


start = t_time()
kiva_countries = kiva_loans.drop(columns=['latitude', 'longitude']).groupby(['country']).mean().reset_index()
kiva_regions = rows_with_countries_and_regions(kiva_loans).groupby(['country', 'region']).mean().reset_index()
del kiva_loans

print('Train data size: {}'.format(kiva_countries.shape))
print('Test data size: {}'.format(kiva_regions.shape))

gc_collect()
timer()


# # Training the model
# 
# Since our training data is ridiculously smaller than our test data, we better abide by a conservative way of dealing with uncertainties.
# 
# 1. We shall ensemble many models;
# 2. For model scoring, we shall use Cross-Validation score with the [Leave-One-Out](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#Leave-one-out_cross-validation) strategy;
# 3. Cross-Validation uses the **negative mean squared error**, so we will invert it and multiply it by -1. Then we will be able to use it as a weight for the ensembling process.
#     * Actually, we could just use the original negative mean squared error, but positive numbers look better for our understanding.

# In[ ]:


from sklearn.model_selection import cross_val_score

start = t_time()
X = kiva_countries.drop(columns=['country', 'bonstato_national'])
y = kiva_countries['bonstato_national']

model_scores = pd.DataFrame({'name':[], 'model':[], 'score':[], 'training(s)':[]})

def score_model(model):
    return -1/np.array(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=87)).mean()

def add_model(model):
    start=t_time()
    score = score_model(model)
    model.fit(X,y)
    name = str(model).split('(')[0]
    training = t_time()-start
    print('{}\n\tscore: {:.3f}\n\ttraining(s): {:.3f}\n'.format(name, score, training))
    model_score = pd.DataFrame({'name':[name], 'score':[score], 'model':[model], 'training(s)':[training]})
    return model_scores.append(model_score, ignore_index=True)


# ## Ensemble models

# In[ ]:


from sklearn.ensemble import *

start = t_time()

models = [
    RandomForestRegressor(n_estimators=50, random_state=42),
    AdaBoostRegressor(random_state=42),
    BaggingRegressor(n_estimators=30, random_state=42),
    ExtraTreesRegressor(n_estimators=30, random_state=42),
    GradientBoostingRegressor(random_state=42)
]

for model in models:
    model_scores = add_model(model)

timer()


# ## Decision Trees

# In[ ]:


from sklearn.tree import *

start = t_time()

models = [
    DecisionTreeRegressor(random_state=42),
    ExtraTreeRegressor(random_state=42)
]

for model in models:
    model_scores = add_model(model)

timer()


# ## Linear models

# In[ ]:


from sklearn.linear_model import *

start = t_time()

models = [
    Lasso(normalize=True, random_state=42),
    LinearRegression(),
    PassiveAggressiveRegressor(max_iter=1500, random_state=42),
    Ridge(random_state=42),
    TheilSenRegressor(random_state=42)
]

for model in models:
    model_scores = add_model(model)

timer()


# ## Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

start = t_time()

model_scores = add_model(KNeighborsRegressor())

timer()


# ## Kernel Ridge

# In[ ]:


from sklearn.kernel_ridge import KernelRidge

start = t_time()

model_scores = add_model(KernelRidge())

timer()


# ## Gaussian Process

# In[ ]:


from sklearn.gaussian_process import GaussianProcessRegressor

start = t_time()

model_scores = add_model(GaussianProcessRegressor(random_state=42))

timer()


# ## Resulting Models

# In[ ]:


del kiva_countries
gc_collect()
model_scores.drop(columns=['model']).sort_values(by='score', ascending=False).reset_index(drop=True)


# # Predicting *bonstato_regional*
# 
# Slow now for we're still dealing with uncertainties.
# 
# We've got a set of models and they make different predictions. The resulting prediction is the weighted average of each model's output, where the weights are the models' scores. Pretty standard.
# 
# The new idea is the `certainty`, which is computed as the inverse of the squared root standard deviation of the models' outputs. `certainty` is then normalized so that it varies from 0 to 1.
# 
# The resulting `bonstato_regional` is computed as a linear combination of the resulting predictions and `bonstato_national`, balanced by `certainty`.
# 
# `bonstato_regional = certainty*prediction + (1 - certainty)*bonstato_national`
# 
# This way `bonstato_regional` gets smoothed by `bonstato_national`, which we know for sure.

# In[ ]:


start = t_time()

def get_predictions(X):
    predictions = pd.DataFrame()
    for i in range(model_scores.shape[0]):
        name = model_scores.at[i, 'name']
        score = model_scores.at[i, 'score']
        model = model_scores.at[i, 'model']
        predictions[name] = score * model.predict(X)
    transp = predictions.transpose()
    result = []
    certainty = []
    for column in transp.columns:
        result.append(transp[column].sum())
        certainty.append(1/(transp[column].std())**0.5)
    predictions.drop(columns=predictions.columns, inplace=True)
    predictions['result'] = np.array(result).transpose()/model_scores['score'].sum()
    predictions['certainty'] = np.array(certainty).transpose()
    normalize_feature(predictions, 'certainty')
    return predictions

predictions = get_predictions(kiva_regions.drop(columns=['country', 'region', 'longitude', 'latitude', 'bonstato_national']))
bonstato = predictions['certainty']*predictions['result'] + (1-predictions['certainty'])*kiva_regions['bonstato_national']

kiva_regions['bonstato_regional'] = bonstato

output = kiva_regions[['country', 'region', 'longitude', 'latitude', 'bonstato_regional']].sort_values(by='bonstato_regional').reset_index(drop=True)

gc_collect()

timer()


# ### Poorest regions

# In[ ]:


output.head(10)


# ### Richest regions

# In[ ]:


output.tail(10)


# The following cell generates the output files `bonstato_by_countries_and_regions.csv` and `bonstato_by_countries_and_regions_sorted_by_countries.csv`.

# In[ ]:


start = t_time()
output.to_csv('bonstato_by_countries_and_regions.csv', index=False)
output.sort_values(by=['country', 'bonstato_regional']).to_csv('bonstato_by_countries_and_regions_sorted_by_countries.csv', index=False)
del output
timer()


# # Grouping up regions
# 
# Some regions are too close from each other for us to notice them properly. Let's cluster them with [k-Means](https://en.wikipedia.org/wiki/K-means_clustering) so we can plot them. 400 clusters should do it.
# 
# I sorted the data so that the poorer clusters come out on top of the richer ones.
# 
# The following cell creates the output files `bonstato_by_clusters.csv` and `bonstato_by_clusters_sorted_by_countries.csv`.

# In[ ]:


import plotly.offline as py
from sklearn.cluster import KMeans

py.init_notebook_mode()

start = t_time()

kiva_regions = rows_with_coordinates(kiva_regions)

kiva_regions['cluster'] = KMeans(400, random_state=42).fit_predict(kiva_regions[['longitude', 'latitude']])
kiva_regions_clusters_groupby = kiva_regions[['country', 'region', 'longitude', 'latitude', 'bonstato_regional', 'cluster']].groupby('cluster')
kiva_regions_clusters = kiva_regions_clusters_groupby.mean().reset_index(drop=True)
kiva_regions_clusters['countries'] = kiva_regions_clusters_groupby['country'].apply(join_names)
kiva_regions_clusters['regions'] = kiva_regions_clusters_groupby['region'].apply(join_names)
kiva_regions_clusters = kiva_regions_clusters[['countries', 'regions', 'longitude', 'latitude', 'bonstato_regional']].sort_values(by='bonstato_regional', ascending=False).reset_index(drop=True)

def truncate(obj):
    if len(obj) > 56:
        obj = obj[:53]+'...'
    return obj

data = [ dict(
    type = 'scattergeo',
    lon = kiva_regions_clusters['longitude'],
    lat = kiva_regions_clusters['latitude'],
    text = 'Countries: '+kiva_regions_clusters['countries'].apply(truncate)+'<br>Regions: '+kiva_regions_clusters['regions'].apply(truncate)+'<br>Bonstato: '+kiva_regions_clusters['bonstato_regional'].apply(str),
    marker = dict(
        size = 20,
        line = dict(
            width=0
        ),
        reversescale = True,
        cmin = kiva_regions_clusters['bonstato_regional'].min(),
        color = kiva_regions_clusters['bonstato_regional'],
        cmax = kiva_regions_clusters['bonstato_regional'].max(),
        opacity = 1,
        colorbar=dict(
            title="Bonstato"
        )
    )
)]

layout = dict(
    title = 'Bonstato by clusters',
    geo = dict(
        showcountries = True
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)

kiva_regions_clusters = kiva_regions_clusters.sort_values(by='bonstato_regional').reset_index(drop=True)
kiva_regions_clusters.to_csv('bonstato_by_clusters.csv', index=False)
kiva_regions_clusters.sort_values(by=['countries', 'bonstato_regional']).to_csv('bonstato_by_clusters_sorted_by_countries.csv', index=False)

del kiva_regions_clusters, kiva_regions, kiva_regions_clusters_groupby
gc_collect()

timer()
raw_timer()


# # Conclusion
# 
# Coding this kernel has been an **intense** experience and I am very happy that everything went well.
# 
# I didn't think that I'd need to resort to so many different areas of knowledge in order to achieve what I wanted. This kernel contains:
# 
# * Neat dataframe manipulations
# * Intuitive feature engineering
# * Natural language processing
# * Missing data inferences
# * Exploratory analysis and visualizations
# * Machine learning
# * Clusterization
# 
# The kernel was also built in an efficient way (it runs pretty quickly, considering its length).
# 
# My goal has been achieved and I am pleased by the results. The answer is YES, Kiva's loans **can** characterize [region specific poverty levels](https://www.kaggle.com/arthurpaulino/data-science-with-compassion/output).
# 
# I hope you have enjoyed the trip as much as I have.
# 
# With Love,<br>
# Arthur Paulino
