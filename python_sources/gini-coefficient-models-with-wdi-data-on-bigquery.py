#!/usr/bin/env python
# coding: utf-8

# The Gini coefficient is a measure of income distribution inequality in a country. Values closer to 100% mean that there is more inequality, while values close to 0 mean that income in a country is more evenly spread. In this notebook we want to study the relationship between the Gini index (World Bank estimate) and different economic indicators in the WDI data set, such as:
# 
# * Inflation, consumer prices (annual %)
# * Tax revenue (% of GDP)
# * Time required to start a business (days)
# * Unemployment, total (% of total labor force)
# 
# The outcome is to understand if a change macroeconomic policies can impact the degree of income inequality in a country, as measured by the Gini coeffcient.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sqlite3


# ## Load and explore data

# Load WDI data from BigQuery

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="worldbank_wdi")


# In[ ]:


bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")
bq_assistant.list_tables()


# In[ ]:


query_str = """SELECT country_code, year, indicator_code, indicator_value from `patents-public-data.worldbank_wdi.wdi_2016` 
               WHERE year BETWEEN 1960 AND 2015 AND 
               indicator_code IN ('SL.UEM.TOTL.NE.ZS','FP.CPI.TOTL.ZG',
                                     'IC.REG.DURS','GC.TAX.TOTL.GD.ZS','SI.POV.GINI') 
                AND indicator_value<>0
            """


# Estimate query size

# In[ ]:


bq_assistant.estimate_query_size(query_str)


# In[ ]:


wdi_df = wdi.query_to_pandas_safe(query_str)


# Count how many values we have for each indicator (across all countries and years)

# In[ ]:


wdi_df.groupby('indicator_code').count()['indicator_value']


# See top rows and size of DataFrame

# In[ ]:


wdi_df.head()


# In[ ]:


wdi_df.shape


# Pivot dataframe so the indicators are on columns

# In[ ]:


wdi_df_piv = wdi_df.pivot_table(index=['country_code','year'], 
                                columns=['indicator_code'], 
                                values=['indicator_value'], fill_value=np.nan).reset_index()


# In[ ]:


wdi_df_piv.shape


# In[ ]:


wdi_df_piv.head()


# In[ ]:


wdi_df_piv.columns = ['country_code','year'] + list(wdi_df_piv.columns.droplevel())[2:]


# In[ ]:


wdi_df_piv.head()


# In[ ]:


wdi_df_piv.columns


# Rearrange and rename columns

# In[ ]:


wdi_df_mod = wdi_df_piv[['country_code','SI.POV.GINI','year',                                              
                        'FP.CPI.TOTL.ZG', 'GC.TAX.TOTL.GD.ZS',
                        'IC.REG.DURS',
                        'SL.UEM.TOTL.NE.ZS']]


# In[ ]:


wdi_df_mod.columns = ['CountryCode','Gini','Year', 
                      'Inflat', 'TaxRev', 'BusDay', 'Unempl']


# In[ ]:


wdi_df_mod.head()


# Describe data

# In[ ]:


wdi_df_mod.describe()


# Count missing data per year

# In[ ]:


wdi_df_mod.groupby(['Year']).count().T


# Correlation

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[ ]:


wdi_corr = wdi_df_mod.iloc[:,1:].corr()
mask = np.zeros(wdi_corr.shape, dtype=bool)
mask[np.tril_indices(len(mask))] = True
sns.heatmap(wdi_corr, annot = True, mask = mask);


# Pair plot

# In[ ]:


sns.pairplot(wdi_df_mod);


# Distribution of Gini

# In[ ]:


wdi_df_mod['Gini'].mean()


# In[ ]:


sns.distplot(wdi_df_mod['Gini'].dropna());


# In[ ]:


sns.distplot(np.log(wdi_df_mod['Gini'].dropna()));


# ## Models

# ### Simple OLS model

# In[ ]:


wdi_df_clean = wdi_df_mod.dropna()


# In[ ]:


from sklearn.preprocessing import RobustScaler
rob_sc = RobustScaler()


# Apply log transform on Gini and scale independent variables

# In[ ]:


Gini_log = np.log(wdi_df_clean['Gini'])
X_sc = rob_sc.fit_transform(wdi_df_clean.iloc[:,2:])


# In[ ]:


wdi_sc = pd.concat([wdi_df_clean.iloc[:, 0],
                    wdi_df_clean.iloc[:, 2], 
                    Gini_log,
                    pd.DataFrame(X_sc, 
                                index=wdi_df_clean.index,
                                columns = [x + '_sc' for x in wdi_df_clean.iloc[:,2:].columns])],
                    axis=1)


# In[ ]:


wdi_df_clean.head()


# In[ ]:


wdi_sc.head()


# In[ ]:


wdi_sc.shape


# Correlation and pair plot with transformed data

# In[ ]:


wdi_corr_sc = wdi_sc.iloc[:,2:].corr()
mask = np.zeros(wdi_corr_sc.shape, dtype=bool)
mask[np.tril_indices(len(mask))] = True
sns.heatmap(wdi_corr_sc, annot = True, mask = mask);


# In[ ]:


sns.pairplot(wdi_sc.iloc[:,2:]);


# In[ ]:


y = wdi_sc['Gini']


# In[ ]:


ols = smf.ols('Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc',
                     data=wdi_sc)
olsf = ols.fit()
# Print out the statistics
olsf.summary()


# In[ ]:


sns.distplot(olsf.resid);


# In[ ]:


sns.regplot(np.exp(olsf.fittedvalues),np.exp(y));


# In[ ]:


sns.regplot(olsf.fittedvalues, olsf.resid, color="g", lowess = True);


# The residual plot suggests heteroscedasticity.

# In[ ]:


fig, axs = plt.subplots(ncols=4, figsize=(30, 5))
sns.regplot(wdi_sc.iloc[:,4], olsf.resid, ax=axs[0], color="r", lowess = True);
sns.regplot(wdi_sc.iloc[:,5], olsf.resid, ax=axs[1], color="r", lowess = True);
sns.regplot(wdi_sc.iloc[:,6], olsf.resid, ax=axs[2], color="r", lowess = True);
sns.regplot(wdi_sc.iloc[:,7], olsf.resid, ax=axs[3], color="r", lowess = True);


# #### Heteroskedasticity test

# In[ ]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(olsf.resid, olsf.model.exog)
lzip(name, test)


# In[ ]:


rmse_ols = (np.sqrt(mean_squared_error(np.exp(y), np.exp(olsf.fittedvalues))))
performance =  pd.DataFrame([['Simple OLS', rmse_ols, test[1]]], columns=['model','rmse', 'het'])
performance


# The model is not great. Are we missing something? Is there country variation that we fail to capture?

# In[ ]:


wdi_res = wdi_sc
wdi_res['Residuals'] = olsf.resid


# In[ ]:


box = sns.boxplot(x="CountryCode", y="Residuals", data=wdi_res);
box.set_xticklabels(box.get_xticklabels(), rotation=90);


# Add country-region data from a SQLite database

# In[ ]:


conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')


# In[ ]:


country_df = pd.read_sql_query("SELECT CountryCode,Region,IncomeGroup FROM Country", conn)


# In[ ]:


country_df.groupby('IncomeGroup').count()


# In[ ]:


wdi_res.shape


# In[ ]:


wdi_region = wdi_res.merge(country_df, left_on='CountryCode', right_on='CountryCode')


# In[ ]:


wdi_region.shape


# In[ ]:


wdi_region.head()


# In[ ]:


box = sns.boxplot(x="IncomeGroup", y="Residuals", data=wdi_region,
                 order=['High income: OECD', 'High income: nonOECD', 'Upper middle income', 'Lower middle income', 'Low income']);
box.set_xticklabels(box.get_xticklabels(), rotation=90);


# In[ ]:


wdi_region.groupby(['IncomeGroup']).count()


# In[ ]:


wdi_region['IncomeGroup'] = np.where(wdi_region['IncomeGroup']=='Low income', 
                                'Lower middle income',
                                wdi_region['IncomeGroup'])


# In[ ]:


box = sns.boxplot(x="IncomeGroup", y="Residuals", data=wdi_region,
                 order=['High income: OECD', 'High income: nonOECD', 'Upper middle income', 'Lower middle income']);
box.set_xticklabels(box.get_xticklabels(), rotation=90);


# In[ ]:


wdi_region.groupby(['IncomeGroup']).count()


# We want to include income group in the model, but without hot encoding as we don't want to make income group a predictor.

# ### Multilevel model with varying intercept by Income Group

# In[ ]:


wdi_region.head()


# In[ ]:


mvi = smf.mixedlm("Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc", data=wdi_region,
                 groups="IncomeGroup")
mvif = mvi.fit()
print(mvif.summary())


# In[ ]:


mvif.random_effects


# In[ ]:


sns.regplot(mvif.fittedvalues,mvif.resid, color="g", lowess = True);


# In[ ]:


name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(mvif.resid, mvif.model.exog)
lzip(name, test)


# In[ ]:


rmse_mvi = np.sqrt(mean_squared_error(np.exp(y), np.exp(mvif.fittedvalues)))
performance.loc[1] =  ['Income Group, Varying Intercept', rmse_mvi,  test[1]]
performance


# ### Multilevel model with varying intercept and varying slopes by Income Group

# In[ ]:


def plot_df_scatter_columns(df, y_column, grouping, rel_col):
    for z in df[rel_col]:    
        sns.lmplot(x = z, y = y_column, data = df, hue = grouping) 

rel_col = ['Year_sc', 'Inflat_sc', 'TaxRev_sc',
       'BusDay_sc', 'Unempl_sc']

plot_df_scatter_columns(wdi_region, 'Gini', "IncomeGroup", rel_col)


# In[ ]:


mvis = smf.mixedlm("Gini ~  Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc", data=wdi_region,
                 groups="IncomeGroup",
                  #re_formula="~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc"
                  re_formula="~ BusDay_sc"
                )
mvisf = mvis.fit()
print(mvisf.summary())


# In[ ]:


mvisf.random_effects


# In[ ]:


sns.regplot(mvisf.fittedvalues, mvisf.resid, color="g", lowess = True);


# In[ ]:


name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(mvisf.resid, mvisf.model.exog)
lzip(name, test)


# In[ ]:


rmse_mvis = np.sqrt(mean_squared_error(np.exp(y), np.exp(mvisf.fittedvalues)))
performance.loc[2] =  ['Income Group, Varying Intercept and Slope', rmse_mvis, test[1]]
performance


# In conclusion, the varying intercept and slope model suggest that increasing tax levels, inflation and decreasing the time to register a business have the effect of reducing income distribtion inequality.

# ### Appendix 1 - Random forest model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor()


# In[ ]:


rf.fit(wdi_region.iloc[:,4:8], wdi_region['Gini'])


# In[ ]:


Gini_pred = rf.predict(wdi_region.iloc[:,4:8])


# In[ ]:


rmse_rf = np.sqrt(mean_squared_error(np.exp(y), np.exp(Gini_pred)))


# In[ ]:


wdi_region.iloc[:,4:8].columns


# In[ ]:


rf.feature_importances_


# In[ ]:


rmse_rf


# In[ ]:


sns.regplot(Gini_pred,y-Gini_pred, color="g", lowess = True);


# In[ ]:


performance.loc[3] =  ['Rf', rmse_rf, np.nan]
performance


# ### Appendix 2 - OLS with Income Group as dummy

# In[ ]:


wdi_region.head()


# In[ ]:


olsd = smf.ols('Gini ~ Inflat_sc + TaxRev_sc + BusDay_sc + Unempl_sc + C(IncomeGroup) ',
                     data=wdi_region)
olsdf = olsd.fit()
# Print out the statistics
olsdf.summary()


# In[ ]:


sns.regplot(olsdf.fittedvalues, olsdf.resid, color="g", lowess = True);


# In[ ]:


name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(olsdf.resid, olsdf.model.exog)
lzip(name, test)


# In[ ]:


rmse_ols_dummy = (np.sqrt(mean_squared_error(np.exp(y), np.exp(olsdf.fittedvalues))))
performance.loc[4] =  ['Income Group Dummy', rmse_ols_dummy, test[1]]
performance

