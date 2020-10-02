#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'> $\color{Red}{\text{Guidance on Vector Auto Regression for Beginner's}}$ </h1>
# <h2>The increasing spread of the coronavirus across countries has prompted many governments to introduce unprecedented measures to contain the epidemic. These are priority measures that are imposed by a sanitary situation, which leave little room for other options as health should remain the primary concern. These measures have led to many businesses being shut down temporarily,widespread restrictions on travel and mobility, financial market turmoil, an erosion of confidence and heighted uncertainty.</h2>
# 
# 
# <img src="https://www.balcanicaucaso.org/var/obc/storage/images/aree/croazia/croazia-conseguenze-economiche-da-covid-19-201468/1963446-1-ita-IT/Croazia-conseguenze-economiche-da-Covid-19.jpg" width="1200px">
# 

# 

# ## <h1 align='left'> $\color{green}{\text{Statsmodels}}$ </h1>
# <ul>
# <li>The library built on other packages like Numpy, Scipy. </li>
# <li>Library contains the function used for testing the statistical model and to build a model. </li>
# <li>Statistical testing tool like adfuller(AUGUMENTED DICKEY-FULLER), rmse and aic.</li>
# </ul>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic


# In[ ]:


filepath = '../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv'
df = pd.read_csv(filepath)

print(df.shape)  # (123, 8)
df.tail()


# # <h1 align='left'> $\color{Green}{\text{Feature Extraction}}$ </h1>

# ## <h1 align='left'> $\color{green}{\text{Comment}}$ </h1>
# Dropped the column containing 'NAN' values

# In[ ]:


df = df.dropna(axis=1)


# In[ ]:


df.tail()


# In[ ]:


df.shape


# # <h1 align='left'> $\color{green}{\text{Comment}}$ </h1>
# * Number of column reduced is 128. 
# * In order to reduced the column further, we are using correlation to get relevant features. Correlation target is set to 0.98. 

# In[ ]:


cor = df.corr()
#Correlation with output variable
cor_target = abs(cor["Price"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.98]
relevant_features


# In[ ]:


df = df[[ 'BritishVirginIslands_total_cases' , 'Cyprus_total_deaths',
        'Grenada_total_cases', 'Guyana_total_deaths'   , 'Niger_total_cases' ,
         'Singapore_total_deaths', 'SriLanka_total_deaths','Vatican_total_cases','Price'] ]


# In[ ]:


df.shape


# # <h1 align='left'> $\color{green}{\text{Grangercausality test}}$ </h1>
# 
# The grangercausality test used to find the determinant between two variable in the series. 
# It uses prior datasets to find the correlation. i.e., X is cause of Y or Y is cause of X.
# It uses Bottom up/top down approach to see if the variables are generated independently or not from each other.
# The test gives value as null hypothesis i.e, variation in y does not interrupted by x.
# Grangercausality test used to find the dependencies between the variable in particular instantaneous time. 
# 
# # <h1 align='left'> $\color{green}{\text{Null Hypothesis}}$ </h1>
# Used to test unit root between the time series.
# 

# In[ ]:



from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns)           


# In[ ]:


nobs = 30
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)


# In[ ]:


test =df_test[['Price']]


# # <h1 align='left'> $\color{green}{\text{ADF Test}}$ </h1>
# In order the predict the time series is Stationary or not. We use <B>Augumented Dickey-Fuller Test(ADF Test)</B>.
# 
# 
# What is **Stationary**?
#  Series is stationary when the **mean and variance are constant over a time**.
#  
#  
# What is **non-stationary**?
#  Series is non-stationary when the **mean and variance are dependent on a time**.
#  
#  
#  ![equation_2.png](attachment:equation_2.png)

# In[ ]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


# In[ ]:


# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


# 1st difference
df_differenced = df_train.diff().dropna()


# In[ ]:


# ADF Test on each column of 1st Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


# Second Differencing
df_differenced = df_differenced.diff().dropna()


# In[ ]:


# ADF Test on each column of 2nd Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


# third Differencing
df_differenced = df_differenced.diff().dropna()


# In[ ]:


# ADF Test on each column of 3rd Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# # <h1 align='left'> $\color{green}{\text{Comments}}$ </h1>
# *  First order difference was used to convert Non-stationary Series to Stationary series.
# *  And ADFnuller function is used to evaluate the Stationary.
# 

# # <h1 align='left'> $\color{green}{\text{Vector Autoregression}}$ </h1>
# Vector autoregression (VAR) is a stochastic process model used to capture the linear interdependencies among multiple time series. 
# VAR models generalize the univariate autoregressive model (AR model) by allowing for more than one evolving variable. 
# ![VAR.svg](attachment:VAR.svg)

# In[ ]:


model = VAR(df_differenced)
for i in [1,2,3,4,5]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# In[ ]:


x = model.select_order(maxlags=5)
x.summary()


# In[ ]:


model_fitted = model.fit()
model_fitted.summary()


# In[ ]:


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input


# In[ ]:


# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast


# In[ ]:


def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


# In[ ]:


df_results = invert_transformation(df_train, df_forecast, second_diff=True)        


# In[ ]:


predict = df_results['Price_forecast']


# In[ ]:


len(predict)


# In[ ]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt( mean_squared_error(test,predict))


# In[ ]:


import matplotlib.pyplot as plt
# zoom plot
plt.figure(figsize=(20,10))
plt.plot(test)
plt.plot(predict, color='green')
plt.title('Actual Vs Predicted')
plt.show()


# references:
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# https://towardsdatascience.com/prediction-task-with-multivariate-timeseries-and-var-model-47003f629f9
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
# https://en.wikipedia.org/wiki/Vector_autoregression#:~:text=Vector%20autoregression%20(VAR)%20is%20a,more%20than%20one%20evolving%20variable.
# 
