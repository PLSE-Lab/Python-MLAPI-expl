#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Predictors by State
# This notebook explores 23 predictors that could be useful for analyzing the COVID19 pandemic. The predictors are described in detail in the dataset's description.  
# 
# Whereas there has been much research into COVID19 as it affects nation states, this dataset focuses on individual states in the US, with the addition of the District of Columbia. The purpose of this notebook is to analyze the predictors themselves, not to create analytical models. This is due to the fact that COVID19 numbers (tests, infections, deaths) are updated constantly; thus, current models are preliminary and are trying to hit a moving target.

# In[ ]:


get_ipython().system('pip install pingouin')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv('../input/covid19-state-data/COVID19_state.csv')
df.head()


# ## Statistical Distributions
# Kernel Density Estimation and Histogram for numerical predictors across all 50 states and DC

# In[ ]:


NUM_ROWS = 6
NUM_COLS = 4

cols_of_interest = df[df.columns[1:-1]]
col_names = cols_of_interest.columns

f, axes = plt.subplots(6, 4, figsize=(9, 9), sharex=False)
c = 0
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        g = sns.distplot(cols_of_interest[col_names[c]], ax=axes[i, j], axlabel=False)
        g.tick_params(left=False, bottom=False)
        g.set(yticklabels=[])
        g.set_title(col_names[c], fontsize=12)
        c += 1
f.tight_layout()


# ### QQ Plots of Selected Factors
# The distribution of some factors looks to be normal while others are exponential. QQ plots can be used to confirm these assumptions and provide a different perspective on a factor's distribution.

# The following QQ plots fit an exponential distribution to the Infected column.

# In[ ]:


fig, ax = plt.subplots(ncols=2, figsize=(8,4))
pg.qqplot(df['Infected'], dist='expon', ax=ax[0], confidence=False)
ax[0].set_title('QQ Plot Infections in All States + DC')

df_no_ol = df[(df.State != 'New York') & (df.State != 'New Jersey')]

pg.qqplot(df_no_ol['Infected'], dist='expon', ax=ax[1])
ax[1].set_title('QQ Plot Infections with Outliers Removed')
[ax[i].set_xlabel('Exponential Distribution Quantiles') for i in range(0,len(ax))]
[ax[i].set_ylabel('Infected Sample Quantiles') for i in range(0,len(ax))]
fig.tight_layout()


# Note how in the first QQ plot there are 2 major outliers from the data, further inspection showed that these were from New York and New Jersey. The outliers were removed and the second plot fits the exponential distribution nicely.

# In[ ]:


ax = pg.qqplot(df_no_ol['Smoking Rate'], dist='norm')
ax.set_xlabel('Normal Distribution Quantiles')
ax.set_ylabel('Smoking Rate Sample Quantiles')
ax.set_title('QQ Plot Smoking Rate in All States + DC')
plt.show()


# The Smoking Rate column follows a normal distribution.

# ## Outliers in COVID19 Numbers
# It is vitally important to call out the imbalance of the COVID19 metrics (Infected, Tested and Deaths). These numbers are being constantly updated and the virus is still rather new in United States, comparatively speaking.

# In[ ]:


f, ax = plt.subplots(1, 3)
df['Deaths'].plot.box(grid=True, ax=ax[0])
df['Infected'].plot.box(grid=True, ax=ax[1])
df['Tested'].plot.box(grid=True, ax=ax[2])
f.tight_layout()


# As shown above, each of the COVID19 metrics has a good deal of outliers. This should be taken into context when creating and interpreting models. These boxplots should expand as more data comes is made available and results are standardized.

# ## Correlations and Feature Reduction
# Correlations of each numerical column including COVID19 tests, infections and deaths

# In[ ]:


def make_corr_map(df, title='Correlation Heat Map', size=(9,7)):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr)) # for upper triangle
    f, ax = plt.subplots(figsize=size)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)


# In[ ]:


make_corr_map(df)


# As shown in the heat map above, some predictors are highly correlated with others. For example, Population is highly positively correlated with Physicians, Hospitals, Med-Lar Airports and ICU beds.  

# ### Model-based Feature Reduction
# Just as a quick heuristic, a random forest regressor can be used to get an idea of feature importances.

# #### Random Forest Regressor

# In[ ]:


X = df[df.columns[4:-1]]
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor(max_depth=10).fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor (y=Deaths)')
plt.show()


# This result is definitely not expected! The next cell digs deeper into why Physicians may be an important feature.

# In[ ]:


plt.scatter(np.log(df['Deaths'], where=df['Deaths'] != 0), np.log(df['Physicians']))
plt.xlabel('Deaths')
plt.ylabel('Physicians')
plt.title('Log Log Plot of Deaths vs Physicians')
plt.show()


# The log log plot shows a clear trend. This shows that there exists a relationship y=X<sup>n</sup>.  
# 
# Looking back the heat map above, Physicians is also highly correlated with Hospitals and Med-Large Airports. According to US news articles about COVID19, the virus was transmitted from abroad and likely entered the US through airports. International airports with high throughput are also indicative of larger cities, where the virus spreads faster. Effectively, Physicians could be a proxy for other highly correlated variables relating to larger cities, where the virus is more pronounced. 
# 
# In this situation, the heuristic of using a random forest regressor to judge feature importances did not work out well. However, feature reduction still needs to occur. The next portion of the notebook demonstrates a reduction of features through qualitative reasoning.

# ## Grouping the Predictors
# Another way to reduce the number of features is to qualitatively group them into categories. The predictors in this dataset could mostly be grouped into **Socioeconomic** and **Public Health** categories, with some overlap between the two. Furthermore, predictors such as Med-Large Airports and Temperature do not fit into the above categories, but would nevertheless be interesting to see in analytical models.

# In[ ]:


socio = ['Population', 'Pop Density', 'Gini', 'Income', 'GDP', 'Unemployment', 'Sex Ratio',
         'Health Spending', 'Urban', 'Age 0-25', 'Age 26-54', 'Age 55+']
health = ['ICU Beds', 'Smoking Rate', 'Flu Deaths', 'Respiratory Deaths', 'Physicians', 'Hospitals', 
          'Health Spending', 'Pollution', 'Age 0-25', 'Age 26-54', 'Age 55+', 'Tested']

df_socio, df_health = df[socio], df[health]


# ### Correlations Revisited

# #### Socioeconomic Predictors

# In[ ]:


make_corr_map(df_socio, 'Socioeconomic Predictors Correlation', (6,6))


# It is clear from the above heat map that GDP is highly correlated with other factors such as Pop Density, and Gini is highly correlated with Sex Ratio. As a rule of thumb, highly correlated features should be removed.

# In[ ]:


X = df_socio.drop(['GDP', 'Gini', 'Income'], axis=1)
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor().fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor')
plt.show()


# Qualitatively, this result make sense, population should be a good predictor of deaths. The next cell takes a closer look at the relationship between Population and Deaths.

# In[ ]:


plt.scatter(np.log(df['Deaths'], where=df['Deaths'] != 0), np.log(df['Population']))
plt.xlabel('Deaths')
plt.ylabel('Population')
plt.title('Log Log Plot of Deaths vs Population')
plt.show()


# Note that if one were to plot these variables without the log scale, outliers would clearly be shown on the Deaths axis. In this case, New York has many more deaths than other states, causing it to be an outlier. However, as COVID19 numbers continue to be reported, these models should adjust accordingly.

# #### Public Health Predictors

# In[ ]:


make_corr_map(df_health, 'Public Health Predictors Correlation', (6,6))


# In[ ]:


X = df_health.drop(['Age 0-25', 'Physicians', 'Tested'], axis=1)
X_s = RobustScaler().fit_transform(X) # scale the data
rfr = RandomForestRegressor().fit(X_s, df['Deaths'])
plt.bar(X.columns, rfr.feature_importances_)
plt.xticks(fontsize=10, rotation=90)
plt.title('Feature Importances based on Random Forest Regressor')
plt.show()


# In this case, ICU Beds could be a proxy metric for large urban centers. The heat map at the top of the Correlations and Feature Reduction section shows that ICU Beds is highly correlated with Physicians, Hospitals and Med-Large Airports. Knowing that New York City is the hardest hit area, this result makes sense. However, note again that New York is an outlier in the Deaths column. As more COVID19 data comes in, these models will likely change.

# ## School Closures Predictor
# School Closures is the sole categorical predictor in the dataset. The following plot shows this data in (hopefully!) readable format.

# In[ ]:


df['School Closure Date'] = pd.to_datetime(df['School Closure Date'])
df['Date Encoding'] = df['School Closure Date'].dt.day

fig, ax = plt.subplots(figsize=(12,5))
g = sns.stripplot(x='State', y='Date Encoding', data=df, ax=ax, color='blue', size=8);
ax.set_title('School Closures in March 2020 due to COVID19')
ax.set_xticklabels(labels=df['State'], rotation=90)
ax.set_ylabel('School Closure Date (March)')
ax.set_xlabel('')
plt.grid()
plt.show()


# # Conclusion
# The purpose of this notebook was to analyze the features of the COVID19 State dataset. It is clear from the results, especially in feature selection, that more qualitative analysis is necessary to attribute any sort of causation to the predictors. That work will be left to the actual scientists and those with much more domain knowledge than myself!   
# <br>
# Furthermore, this notebook is not exhaustive; there are a seemingly infinite number of insights that could come from this data. Please feel free to experiment and provide constructive feedback! 
