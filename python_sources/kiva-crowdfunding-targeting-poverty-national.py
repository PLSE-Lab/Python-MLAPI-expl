#!/usr/bin/env python
# coding: utf-8

# # Kiva Crowdfunding - Targeting Poverty at a National Level
# ***
# Kiva is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. 
# 
# This notebook series is my contribution to the Data Science for Good: Kiva Crowdfunding challenge. 
# The objective is to help Kiva to better understand their borrowers and build more localized models to estimate the poverty levels in the regions where Kiva has active loans.
# 
# Kiva Crowdfunding notebook series:
#   - [Part I - Understanding Poverty]
#   - [Part II - Targeting Poverty at a National Level]
#   - [Part III - Targeting Poverty at a Subnational Level]
#   - [Part IV - Adding a Financial Dimension to the MPI]
#   - [Part V - Investigating Nightlight as a Poverty Indicator]
# 
# [Part I - Understanding Poverty]: https://www.kaggle.com/taniaj/kiva-crowdfunding-understanding-poverty
# [Part II - Targeting Poverty at a National Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-national
# [Part III - Targeting Poverty at a Subnational Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-sub-nat
# [Part IV - Adding a Financial Dimension to the MPI]: https://www.kaggle.com/taniaj/kiva-crowdfunding-adding-a-financial-dimension
# [Part V - Investigating Nightlight as a Poverty Indicator]: https://www.kaggle.com/taniaj/kiva-crowdfunding-investigating-nightlight
# 
# The series in broken down into five notebooks. The first notebook is an exploratory analysis of the data to get a feeling for what we are working with. The second notebook examines external datasets and looks at how MPI and other indicators can be used to get a better understanding of poverty levels of Kiva borrowers at a national level. The third notebook examines external data at a subnational level to see how Kiva can get MPI scores based on location at a more granular level than is currently available. The fourth notebook attepts to build a better poverty index at a subnational level by adding a financial dimension. The fifth notebook examines nightlight data as a poverty indicator.
# 
# This is the second notebook of the series. The aim here is to identify other useful ways in which poverty scores can be assigned to target Kiva borrowers. During this second stage of development, a number of external datasets were identified, which could potentially provide useful features for new indices for determining poverty. However, the majority of the datasets identified provided statistics only at a national level. This is probably not accurate enough to be helpful to Kiva as an index so the work is presented simply as a record of the author's progress and in the hope that learnings can be gained when building indices at a more detailed level.
# 
# [Part I - Understanding Poverty]: https://www.kaggle.com/taniaj/kiva-crowdfunding-understanding-poverty
# [Part II - Targeting Poverty at a National Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-national
# [Part III - Targeting Poverty at a Subnational Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-sub-nat
# [Part IV - Adding a Financial Dimension to the MPI]: https://www.kaggle.com/taniaj/kiva-crowdfunding-adding-a-financial-dimension
# 
# ### Contents
#    1. [Kiva's Current Poverty Targeting System](#kiva_current_system)
#    2. [Human Development Index](#hdi)
#    3. [Financial Inclusion](#financial_inclusion)
#        * [Steps to calculating the Findex ](#findex_calc_steps)
#        * [Findex Score compared to MPI Score](#findex_vs_mpi)
#    4. [Telecommunications Access](#telecommunications_access)
#    5. [Gender inequality](#gender_inequality)
#    6. [Conclusion](#conclusion)
#    7. [References](#refere)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from scipy.stats.mstats import gmean
from scipy.stats.stats import pearsonr

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

get_ipython().system('cp ../input/images/cell_subscription_levels.png .')


# In[ ]:


# Original Kiva datasets
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
kiva_mpi_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

# Additional Kiva datasets
mpi_national_df = pd.read_csv("../input/mpi/MPI_national.csv")
# The subnational Kiva data has been enhanced with lat/long data
#mpi_subnational_df = pd.read_csv("../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv")

# Human Development Reports
hdi_df = pd.read_csv("../input/human-development/human_development.csv")

# UNDP gender inequality data
gender_development_df = pd.read_csv("../input/gender-development-inequality/gender_development_index.csv")
gender_inequality_df = pd.read_csv("../input/gender-development-inequality/gender_inequality_index.csv")

# World Bank population data
world_pop_df = pd.read_csv("../input/world-population/WorldPopulation.csv")

# World Bank Findex data
findex_df = pd.read_csv("../input/findex-world-bank/FINDEXData.csv")

# World Bank cellular subscription data
cellular_subscription_df = pd.read_csv("../input/world-telecommunications-data/Mobile cellular subscriptions.csv")


# ## 1. Kiva's Current Poverty Targeting System <a class="anchor" id="kiva_current_system"/>
# ***
# Kiva currently assigns scores to field partners and loan themes based on their location using the Multi-dimensional Poverty Index (MPI) and the Global Findex dataset for financial inclusion. (Ref: https://www.kaggle.com/annalie/kivampi)
# 
# Nation-level MPI Scores are broken into rural and urban scores and an average of these scores, weighted by rural_pct, is assigned to each Kiva field partner. Where field partners serve multiple countries (there are a few cases), a volume-weighted average is taken.
# 
# Sub-National MPI Scores are used for targeting within countries by assigning a given loan or loan theme to a given MPI region and aggregating the scores as a volume-weighted average.
# 
# Financial Inclusion Scores have also already been explored to a limited extent by Kiva to measure financial inclusion per country, using the Global Findex database published by the World Bank. 
# 
# The code below calculates the MPI at national level according to Kiva's weighting method (for use later in this notebook).

# In[ ]:


# Join datasets to get rural_pct
mpi_national_df.rename(columns={'Country': 'country'}, inplace=True)
loan_themes_country_df = loan_themes_by_region_df.drop_duplicates(subset=['country'])
mpi_national_df = mpi_national_df.merge(loan_themes_country_df[['country', 'rural_pct']], on=['country'], how='left')

# There are 52 nulls in rural_pct so lets fill these with the median value
mpi_national_df['rural_pct'].fillna(mpi_national_df['rural_pct'].median(), inplace=True)

# Calculate national mpi according to Kiva's method
mpi_national_df['MPI'] = mpi_national_df['MPI Rural']*mpi_national_df['rural_pct']/100 + mpi_national_df['MPI Urban']*(100-mpi_national_df['rural_pct'])/100


# In[ ]:


mpi_national_df.sample()


# ## 2. Human Development Index <a class="anchor" id="hdi"/>
# ***
# The Human Development Index (HDI) is a summary measure of achievements in key dimensions of human development: a long and healthy life, access to knowledge, and a decent standard of living. It was first released in 1990 and has become widely accepted and used for policy formation. The MPI is actually a logical extension of this index so it is interesting to see how closely the two correlate.

# In[ ]:


colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= hdi_df.Country,
        locationmode='country names',
        z=hdi_df['Human Development Index (HDI)'],
        text=hdi_df.Country,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Findex'),
)]
layout = dict(
            title = 'Human Development Index',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# In[ ]:


# Join
mpi_hdi_df = mpi_national_df.merge(hdi_df[['Country', 'Human Development Index (HDI)']], left_on=['country'], right_on=['Country'])


# In[ ]:


# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_hdi_df.loc[:, 'Human Development Index (HDI)'], mpi_hdi_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='Human Development Index (HDI)', data=mpi_hdi_df)


# There is, as expected, a relatively high correlation between the HDI and MPI.

# ## 3. Financial Inclusion <a class="anchor" id="financial_inclusion"/>
# ***
# Kiva has already done some work with Financial Inclusion (https://www.kaggle.com/annalie/kivampi) however there is still room to explore other possibilities for a Financial Inclusion Index. A simple possibility is to combine existing key financial inclusion indicators.
# 
# The World Bank lists the following as key indicators for financial inclusion based on the data they have collected:
# 
#     - percentage of people (age 15+) who have an account
#     - percentage of people (age 15+) who have formal savings
#     - percentage of people (age 15+) who have access to formal borrowing

# In[ ]:


# Keep relevant indicators only
findex_key_ind_df = findex_df.loc[(findex_df['Indicator Name'] == 'Account (% age 15+) [ts]') 
                                  | (findex_df['Indicator Name'] == 'Borrowed from a financial institution (% age 15+) [ts]')
                                  | (findex_df['Indicator Name'] == 'Saved at a financial institution (% age 15+) [ts]')]
# Keep relevant Countries only (those for which we have MPI)
# Note: there are less countries available in findex than in Kiva loans.
#findex_key_ind_df['Country Name'].unique()

findex_key_ind_df = findex_key_ind_df[findex_key_ind_df['Country Name'].isin(kiva_mpi_locations_df['country'].unique())]

findex_key_ind_df.sample(5)


# In[ ]:


# Pivot
findex_pivot_df = findex_key_ind_df.pivot(index='Country Name', columns='Indicator Name', values='MRV').reset_index().rename_axis(None, axis=1)
findex_pivot_df.columns = ['country_name', 'account', 'formal_savings', 'formal_borrowing']

findex_pivot_df.sample(5)


# #### Steps to calculating the Findex  <a class="anchor" id="findex_calc_steps"/>
# 
# We will use the method used to calcualte the HDI as a guideline. 
# 
# Step 1: Create dimension indices. 
# 
#     In our case, the three component indicators are all expressed as percentages so lets keep it simple and go with a minimum of 0 and maximum of 100 for each indicator.
#     
# Step 2. Aggregating the dimensional indices
# 
#     The geometric mean of the dimensional indices is calculated to produce the Findex.

# In[ ]:


findex_pivot_df['findex'] = gmean(findex_pivot_df.iloc[:,1:4],axis=1)
findex_pivot_df.head()


# In[ ]:


colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= findex_pivot_df.country_name,
        locationmode='country names',
        z=findex_pivot_df['findex'],
        text=findex_pivot_df.country_name,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Findex'),
)]
layout = dict(
            title = 'Findex (Kiva Countries)',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# #### Findex Score compared to MPI Score  <a class="anchor" id="findex_vs_mpi"/>
# 
# Lets merge the findex score with kiva_mpi_locations and have a look how the two scores are related.

# In[ ]:


# Join
mpi_findex_national_df = mpi_national_df.merge(findex_pivot_df[['country_name', 'findex']], left_on=['country'], right_on=['country_name'])
mpi_findex_national_df.drop('country_name', axis=1, inplace=True)

mpi_findex_national_df.sample()


# In[ ]:


# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_findex_national_df.loc[:, 'findex'], mpi_findex_national_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='findex', data=mpi_findex_national_df)


# In[ ]:


plt.subplot(121).set_title("MPI distribuion")
sns.distplot(mpi_findex_national_df.MPI)

plt.subplot(122).set_title("Findex distribuion")
sns.distplot(mpi_findex_national_df.findex)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# - Countries with high poverty levels often score low on the financial inclusion index but there is no clear correlation between the two metrics, indicating that the Findex could be a useful measure that is unrelated to poverty. This is examined in more detail in Part IV of this notebook series (https://www.kaggle.com/taniaj/kiva-crowdfunding-adding-a-financial-dimension).
# 
# - Note when examining the histograms, that the higher MPI scores show greater deprivation, while lower findex scores show greater deprivation. 
# 
# Unfortunately there are also quite a few countries that have been dropped following the merge, due to missing Findex values. (78 out of the original 102 countries in the mpi_national_df list have been assigned a Findex score.)

# ## 4. Telecommunications Access <a class="anchor" id="telecommunications_access"/>
# ***
# At first telecommunications access may not seem particularly important or relevant when trying to estimate poverty or financial inclusion, however, especially notably in certain regions in Africa, the proliferation of mobile telephone services has opened a new way to extend financial services to people who don't have a regular bank account.
# Indeed, telecommunications is included in the MPI with relatively small weighting and in quite a vague way as part of the definition of part of one of the three poverty indicators.
# 
# <div class="alert alert-block alert-info">
# <p/>
# Standard of living, Assets: not having at least one asset related to access to information (radio, television or telephone) or having at least one asset related to information but not having at least one asset related to mobility (bike, motorbike, car, truck, animal cart or motorboat) or at least one asset related to livelihood (refrigerator, arable land or livestock).
# </div>
# 
# Perhaps telecommunications access can be used as an idicator or as part of a larger index to better understand financial inclusion.

# In[ ]:


# Most countries have data for 2016 but not all. Create new MRV column for most recent values. There are still a few nulls after this.
cellular_subscription_df['MRV'] = cellular_subscription_df['2016'].fillna(cellular_subscription_df['2015'])

# Keep only relevant columns
cellular_subscription_df = cellular_subscription_df[['Country Name', 'Country Code','MRV']]

cellular_subscription_df.sample(5)


# In[ ]:


colorscale = [[0.0, 'rgb(230, 240, 255)'], [0.2, 'rgb(179, 209, 255)'], [0.4, 'rgb(102, 163, 255)'],              [0.6, 'rgb(26, 117, 255)'], [0.8, 'rgb(0, 71, 179)'], [1.0, 'rgb(0, 31, 77)']]
data = [dict(
        type='choropleth',
        locations= cellular_subscription_df['Country Name'],
        locationmode='country names',
        z=cellular_subscription_df['MRV'],
        text=cellular_subscription_df['Country Name'],
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='Cellular Subscription'),
)]
layout = dict(
            title = 'Cellular Subscription Levels',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# In[ ]:


# Join
mpi_cell_national_df = mpi_national_df.merge(cellular_subscription_df[['Country Name', 'MRV']], left_on=['country'], right_on=['Country Name'])
mpi_cell_national_df.drop('Country Name', axis=1, inplace=True)


# In[ ]:


# Compare scores 
print("Correlation, p-value: ", pearsonr(mpi_cell_national_df.loc[:, 'MRV'], mpi_cell_national_df.loc[:, 'MPI']))
sns.regplot(x='MPI', y='MRV', data=mpi_cell_national_df)


# In[ ]:


plt.subplot(121).set_title("MPI distribuion")
sns.distplot(mpi_cell_national_df.MPI)

plt.subplot(122).set_title("Telcomm Access distribuion")
sns.distplot(mpi_cell_national_df.MRV)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# - There are quite a few countries with more than 100 cellular subscriptions per 100 people!
# 
# - Countries with high poverty levels often have less cellular subscriptions per 100 people but there is no clear correlation between the two metrics, indicating that the telecommunication access could also be a useful indicator that is unrelated to poverty. 
# 
# - Note when examining the histograms, that the higher MPI scores show greater deprivation, while lower Telecom Access scores show greater deprivation.
# 
# Although it already may play a small part in the MPI (Standard of living, Assets, as discussed previously) having a mobile phone subscription is very different from, for example having access to a radio or television, which MPI does not differentiate. Access to a mobile phone and cellular subscription enables access to finance options like mobile money and cryptocurrencies, without requiring a formal bank account.
# 
# There are a few countries that have been dropped following the merge, due to missing telecoms access values. (88 out of the original 102 countries in the mpi_national_df list have this value poplated.)

# ## 5. Gender inequality <a class="anchor" id="gender_inequality"/>
# ***
# Gender inequality data and indices are available from the World Bank and may be used to further differentiate between poverty levels among males and females requesting Kiva loans.
# 
# Gender Development Index (GDI): measures gender inequalities in achievement in three basic dimensions of human development: 
#     - health, measured by life expectancy at birth 
#     - education, measured by expected years of schooling for children and mean years of schooling for adults ages 25 years and older 
#     - command over economic resources, measured by estimated earned income
# 
# Gender Inequality Index (GII): reflects gender-based disadvantage in three dimensions:
#     -  reproductive health, 
#     -  empowerment and 
#     - the labour market for as many countries as data of reasonable quality allow. 
# It shows the loss in potential human development due to inequality between female and male achievements in these dimensions. It ranges from 0, where women and men fare equally, to 1, where one gender fares as poorly as possible in all measured dimensions.
# 
# Lets further explore the indices.

# In[ ]:


gender_development_df[gender_development_df['2015'].isnull()]


# In[ ]:


gender_inequality_df[gender_inequality_df['2015'].isnull()]


# In[ ]:


# There are not many relevant missing values so lets just drop these for now. 
gender_development_df = gender_development_df.dropna(subset=['2015'])
gender_inequality_df = gender_inequality_df.dropna(subset=['2015'])
# Keep only relevant columns.
gender_development_df = gender_development_df[['Country', 'HDI Rank (2015)', '2015']]
gender_inequality_df = gender_inequality_df[['Country', 'HDI Rank (2015)', '2015']]


# In[ ]:


colorscale = [[0.0, 'rgb(255, 255, 255)'], [0.2, 'rgb(234, 250, 234)'], [0.4, 'rgb(173, 235, 173)'],              [0.6, 'rgb(91, 215, 91)'], [0.8, 'rgb(45, 185, 45)'], [1.0, 'rgb(31, 122, 31)']]
data = [dict(
        type='choropleth',
        locations= gender_development_df.Country,
        locationmode='country names',
        z=gender_development_df['2015'],
        text=gender_development_df.Country,
        colorscale = colorscale,
        colorbar=dict(autotick=False, tickprefix='', title='GDI'),
)]
layout = dict(
            title = 'Gender Development',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# Lets do some quick map plots to get an overview of the data.

# In[ ]:


data = [dict(
        type='choropleth',
        locations= gender_inequality_df.Country,
        locationmode='country names',
        z=gender_inequality_df['2015'],
        text=gender_inequality_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Gender Inequality',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# There are no surprises here.
# 
# - Gender development is the lowest in Africa and parts of South Asia.
# - Gender inequality still exists in all countries (white ones indicate missing data, not utopia).
# - Gender inequality is the highest in many African countries.
# - Relatively high gender inequality aso exists in parts of South Asia and South America.
# 
# We can conclude that gender inequality, especially in the regions were poverty is rampant, is significant. The MPI does not take into account gender in any way so we may well be able to build a better indicator of poverty by taking into account the gender of a borrower applying for a loan. 

# ## 6. Conclusion  <a class="anchor" id="conclusion"/>
# ***
# This notebook has examined a number of indicators on a national level, which are currently not included in the OPHI MPI. All of these - financial inclusion, telecommunication access and gender equality could be good additional dimensions for a new multidimensional poverty index. 
# 
# The HDI, as a "predecessor" of the MPI was also initially examined to get an idea of what sort of correlation values are observed between these two indices in comparison to correlations between the other indicators and the MPI. The results show that indeed, the other indicators examined have much lower correlation to the MPI than the HDI and could well add value if combined into a new index.

# ## 7. References <a class="anchor" id="references"/>
# ***
# * [Global Findex](https://globalfindex.worldbank.org/)
# * [International Telecommunications Union](https://www.itu.int/en/Pages/default.aspx)
# * [Kiva's Current Poverty Targeting System](https://www.kaggle.com/annalie/kivampi/code)
# * [Kiva Website](https://www.kiva.org/)
# * [Multidimensional Poverty Index](http://hdr.undp.org/en/content/multidimensional-poverty-index-mpi)
# * [UNDP Poverty Indices - Technical Notes](http://hdr.undp.org/sites/default/files/hdr2016_technical_notes.pdf)
# * [World Bank Databank](http://databank.worldbank.org/data/home.aspx)
