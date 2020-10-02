#!/usr/bin/env python
# coding: utf-8

# One of my favourite papers of all time is a 1985 paper entitle, "Perception of price when price information is costly: evidence from residential electricity demand".  In this paper, Shin uses some really elegant regression modelling to try to determine whether consumers make decision on electricity consumption, based on their marginal price (ei the price for their next unit under a progressive block pricing schedule) or based on their average price over their units consumed.  
#   
# For businesses and policy makers this can provide crucial insight into the efficacy of different pricing policies in changing consumer behaviour.  In water and electricity consumption this can be critical in ensuring equitible consumption and in ensuring the sustainability of the energy or water supplier.  For economist researching demand, modelling consumer behaviour is critical.  Typically, when economists think about consumer behaviour they are interested in how income, price and the price of substitutes affect demand.  To investigate these affects they look to the idea of elasticity- how do changes in price and income affect changes demand.  For Shin, the primary interest for whether changes in average price or changes in marginal price affected changes in demand.  To do this he used a regression model.  
# 
# To investigate this elasticity of price and income, economists typically apply a log transform to their endogenous variables (like quantity) and exogenous variables (like price and incomes) to approximate this idea of percent change. In Shin's paper, he defines the following model:
#   
# $ln(\textit{units of demand}) = \beta_1 ln(\textit{income}) + \beta_2 ln(\textit{units of demand at a previous timestep}) + \beta_3 ln(\textit{marginal price}) + \beta_4 ln(\frac{\textit{average price}}{\textit{marginal price}}) + ... +\beta_0$
# 
# Using this model, he not only analyze the elasticity of margin price and income but, using $\beta_4$, whether consumers make decisions based on averaeg price or marginal price.  Using this model, if $\beta_4$ is positive consumers make decisions based on average price and if $\beta_4$ is negative then their is evidence that consumers use merginal price.   
# The major challenge in using the AguaH dataset is mainly that I could not find a real block-pricing schedule for period over which the data was collected.  I managed to track down the current Block Pricing Schedule, with some help but was unable to find the historic schedule on Internet Archive. This will force us to asssume that these are Real Prices and that the structure of the schedules have not changed over time but instead moved in lock-step inline with wages and inflation. 
#   
# [1] http://aguadehermosillo.gob.mx/aguah/tarifas/ accessed 20 March 2020  
# [2] Shin, J.S., 1985. Perception of price when price information is costly: evidence from residential electricity demand. The review of economics and statistics, pp.591-598.

# In[ ]:


get_ipython().system(' pip install --upgrade pip')
get_ipython().system(' pip install hvplot')


# In[ ]:


# load imports and set extension
from bisect import bisect
from functools import reduce
from operator import add

import holoviews as hv
import hvplot.pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import probplot
from statsmodels.regression.linear_model import OLS


hv.extension('bokeh')


# In[ ]:


# Set seed
pd.np.random.seed(0)

FILTER = 0.5 # filter for stratified sampling groups with little coverage
FRAC = 0.1 # subsample the data

# Filter for domestic dwellings, "SOCIAL": 0 excluded as different schedule
INCOME = {
          "DOMESTICO BAJA": 0,
          "DOMESTICO MEDIO": 1,
          "DOMESTICO RESIDENCIAL": 2,
          }

# Spanish Month Mappings
ABBREVIATIONS = {'ENE': 0,
               'FEB': 1,
               'JUL': 2,
               'JUN': 3,
               'MAR': 4,
               'MAY': 5,
               'NOV': 6,
               'OCT': 7,
               'SEP': 8,
               'ABR': 9,
               'AGO': 10,
               'DIC': 11}


# Proxy block-tarrif structure
# http://aguadehermosillo.gob.mx/aguah/tarifas/ : accessed 6 March 2020
RATES = [0, 8.57, 11.09, 11.09, 11.09, 11.27, 17.17, 17.17, 17.17, 54.01, 54.01, 54.01, 54.96, 54.96, 59.09]   
BRACKETS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 56, 70, 75]
BASE_RATE = 86.11


# In[ ]:


# Tarrif calculator
base_rate = np.multiply(np.diff([0, *BRACKETS]), 
                        RATES[:-1]).cumsum()
def block_pricing(units):
    i = bisect(BRACKETS, units)
    if not i:
        return 0
    rate = RATES[i]
    bracket = BRACKETS[i-1]
    tarrif_in_bracket = units - bracket
    tarrif_in_bracket = tarrif_in_bracket * rate
    total_tarrif = base_rate[i-1] + tarrif_in_bracket
    return total_tarrif + BASE_RATE

def marginal_rate(units):
    i = bisect(BRACKETS, units)
    
    return RATES[i]

def block(units):
    i = bisect(BRACKETS, units)
    
    return i


# This dataset is massive, and among other preprocessing steps, was a step to subsample and filter the data to focus on domestic users.  While one challenge issues in this dataset is to optimally impute data, given the size of the data, I opted to drop rows with missing data and assumed that this was not correlated with featrues in the data.  Another assumption I made in modelling income, was to assume that income was correlated to residential areas marked as social housing, low-income housing, middle-income houding and high-income housing.  One of the interesting features I was keen to investigate in this study was how seasonability- or a winter effect- may change consumption habbits. 

# In[ ]:


data = pd.read_csv('/kaggle/input/water-consumption-in-a-median-size-city/AguaH.csv')

resample_after_dropna = (data
                         .groupby(['TU', 'USO2013'])
                         .apply(lambda df: 1- (df
                                            .isna()
                                            .any(1)
                                            .mean()))
                         .reset_index()
                         .rename(columns={0:'resampling'}))
        
        
data = (data
        .dropna()
        .merge(resample_after_dropna, on=['TU', 'USO2013'])
        .sample(frac=FRAC, weights='resampling'))

date_mapping = {date: i for i, date in enumerate(data.columns[data.columns.str.startswith('f.1_')])}
reverse_date_mapping = {i: date for date, i in date_mapping.items()}
reverse_year_mapping = {i: 2000 + int(d[-2:]) for d, i in date_mapping.items()}
reverse_month_mapping = {i: ABBREVIATIONS[date.split('_')[1]] for date, i in date_mapping.items()}


# In[ ]:


filter_groups = (data
 .where(lambda x: x.TU.isin(list(INCOME.keys())))
 .groupby(['TU', 'USO2013'])['M'].count())

filter_groups


# In[ ]:


filter_groups = (filter_groups
                 .reset_index()
                 .where(lambda df: df.M > df.M.quantile(FILTER))
                 .rename(columns={'M': 'filter'}))


# In[ ]:


tidy = (data
        .where(lambda x: x.TU.isin(list(INCOME.keys()))) # filter for domestic users
        .merge(filter_groups, on=['TU', 'USO2013'], how='left').dropna().drop(columns=['filter', 
                                                                                       'resampling'])
         .merge(filter_groups, on=['TU', 'USO2013'], how='left').dropna().drop(columns=['filter'])
#             .groupby(['TU', 'USO2013'])
#             .apply(lambda df: df.sample(frac=FRAC)) # stratified sample
#             .reset_index(drop=True) # stratified sampling of domestic user
            .dropna()
        .reset_index()
        .rename(columns=dict(**date_mapping,
                             **{'index':'user'}))
        .melt(id_vars=['user','USO2013','TU','DC','M','UL'],
              var_name='date',
              value_name='quantity') # melt measurement timeseries
        .assign(month = lambda df: df.date.replace(reverse_month_mapping), # get month
                year = lambda df: df.date.replace(reverse_year_mapping), # get year
                income = lambda df: df.TU.replace(INCOME), # get pseudo income
                block = lambda df: df.quantity.apply(block),
                tarrif = lambda df: df.quantity.apply(block_pricing), # get tarrif
                marginal =  lambda df: df.quantity.apply(marginal_rate)) # get marginal rate
        .assign(average = lambda df: df.tarrif / df.quantity) # compute average rate
        .dropna())


# In[ ]:


# get quantity at lag
tidy = tidy.merge((tidy
                   .loc[:,['date','user','quantity']]
                   .assign(date = lambda df: df.date + 1,
                           quantitylag = lambda df: df.quantity)
                   .drop(columns=['quantity'])), on=['date','user'], how='left').dropna()

tidy.head()


# As the data was in wide-format a major challenge in modelling, was to 'tidy' the dataset into a long format. Given more infromation on the 'M' column it may be interest for other investigate spatial correlation in the dataset and perhaps using the user-number mixed effects in the data.  

# In[ ]:


# Correlation matrix
(tidy
 .loc[:,['quantity','marginal','average', 'income', 'quantitylag', 'year']]
 .corr())


# Looking at the correlation matrix it is clear that many features appear highly correlated. This may be diffficult to control for given the data, and may given futher analysis provide motivation for a two-step regression procedure. 

# In[ ]:


# construct shin design matrix
d_matrix = (tidy
            .loc[:,['quantity','marginal','average','income','quantitylag', 'month', 'year', 'block']]
            .assign(marginal_over_average = lambda df: df.marginal / df.average)
            .assign(winter = lambda df: df.month.apply(lambda x: (np.cos(2 * np.pi * (x/11))))
                                                       .add(1) # shift the cosine up
                                                       .divide(2) # make between 0 and 1
                                                       .add(1e-6) # add jitter for log transform
                                                      )
            .assign(block = lambda df: df.block.apply(pd.np.exp))
            .assign(year = lambda df: df.year - df.year.min() + 1e-6)
            .drop(columns=['average', 'month'])
            .transform(np.log)
            .assign(bias = 1)
            .replace({np.inf: np.nan, -np.inf: np.nan})
            .dropna())

block_pricing_mixed_effects = pd.get_dummies(d_matrix['block'].astype(str) 
                                             + '_' 
                                             + d_matrix['year'].apply(np.exp).add(tidy.year.min() - 1e-6).astype(str), 
                                             prefix = 'block')

landuse_indicators = pd.get_dummies(tidy['USO2013'],
                                    prefix = 'landuse').drop(columns=['landuse_MX'])


d_matrix = (pd.concat([d_matrix.drop(columns=['block']),
                      block_pricing_mixed_effects], axis=1)
            .join(landuse_indicators))

d_matrix


# In[ ]:


d_matrix.iloc[:, :7].corr()


# The final design matrix, required doing our log transforms and constructing a 'winter' variable by taking the cosine over the months.  To controll for real inflation and changes in the prince within blocks over time a year variable and mixed affects were added to the model. This does not control for the fact that block width and position may have changed, but shoud control for some of the error in our model around the uncertainty of our pricing structure. 

# In[ ]:


# do regression analysis
X, Y = d_matrix.drop(columns=['quantity']), d_matrix.loc[:,['quantity']]
model = OLS(Y,X)
results = model.fit()

results.summary()


# Without going into to much depth, it appears our model is a good fit to the data.  Looking at the marginal_over_average	coefficient of -2.5675, there seems to be evidence for the fact that consumers make decisions based on average-price not marginal. Looking at the coefficient on winter, it seems that users may consume less water in summer, which may be for a variety of reasons.  Oddly, looking at income there seems to be a small negative substitution effect on income, that all else equal, higher income households may be more efficient in their water use or may be better able to avoid some water usage as their incomes increase. 

# In[ ]:


# plot errors distribution
residuals = results.predict(X).rename('errors').subtract(Y.quantity)

(((residuals
 .hvplot.kde(label='Residuals', xlabel='residuals'))) *
(pd.Series(np.random.normal(0,residuals.std(),size=(1000)))
 .hvplot.kde(label='Distribution of Centred Normal', xlabel='epsilon')))


# In[ ]:


## qq plot
theoretical_quantiles, sample_quantiles = probplot(residuals / residuals.std())[0]

(hv.Curve([[-2.5, -2.5], [2.5,2.5]]).opts(line_width=1) *
 pd.DataFrame({'theoretical_quantiles': theoretical_quantiles, 'sample_quantiles': sample_quantiles})
 .sample(frac=0.005)
 .hvplot.scatter(x='theoretical_quantiles', y='sample_quantiles', title='QQ Plot', 
                 xlabel = 'Theoretical Quantiles', ylabel = 'Sample Quantiles',
                 size=1))


# Looking at the quatiles of our residuals, there seems to be some poor fit at the extremes of our data. This may suggest some outliers in our data or perhaps omitted variables or polynomial terms. 

# In[ ]:


# plot heteroskedasticity
(reduce(add, [(X
               .loc[:,[col]]
               .assign(residuals = residuals)
               .dropna()
               .hvplot.scatter(y='residuals', x=col, datashade=True, width=350, height=250)) 
              for col in X.columns 
              if not (col.startswith('block') or 
                      col.startswith('bias') or 
                      col.startswith('landuse'))])
 .cols(2)
 .opts(title="Heteroskedasticity"))


# Another major concern, is the slight heteroskedasticity on our marginal_over_average, marginal and quantity_lag, this may create some doubt on the findings of our model and may provide evidence that the structure of our blocks are incorrectly specified. Looking at our mixed-effects estimates we see that the errors on black 12 in 2010 and 2012 are large suggesting that these blocks may have been shifted over this time and may be introducing error into our model. 

# I would love to get your ideas and feedback on this in kernel. This is for sure not a definitive model and their may be room to explore and investigate GLM's or IV's. 
