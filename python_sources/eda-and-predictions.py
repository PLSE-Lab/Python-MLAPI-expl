#!/usr/bin/env python
# coding: utf-8

# # Predicting Daily Forecast of Confirmed COVID-19 Cases and Fatalities
# ***
# ## Import Packages

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

get_ipython().run_line_magic('matplotlib', 'inline')

# We are required to do this in order to avoid "FutureWarning" issues.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# ## Read in the training data and look at the head

# In[ ]:


cov = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')


# In[ ]:


cov.head()


# In[ ]:


cov.info()


# ## Our training data has 8 columns and 170,040 rows (cases). Its arannged by country in alphabetical order with each country displaying all dates from January 22nd until March 21st.

# In[ ]:


cov.describe()


# In[ ]:


cov['Date']


# ## Create a modified dataframe that lists Date as the index and another column with the sum of the Confirmed cases from all countries on each date.

# In[ ]:


cov_confirmed = cov.groupby('Date')[['ConfirmedCases']].sum()


# In[ ]:


cov_confirmed.tail()


# ## Create a plot_series function to generate line plots with 4 basic arguments

# In[ ]:


# Code modified from code written by Matthew Garton.

def plot_series(cov_confirmed, cols=None, title='Title', xlab=None, ylab=None, steps=1):
    
    # Set figure size to be (18, 9).
    plt.figure(figsize=(18,9))
    
    # Iterate through each column name.
    for col in cols:
            
        # Generate a line plot of the column name.
        # You only have to specify Y, since our
        # index will be a datetime index.
        plt.plot(cov_confirmed[col])
        
    # Generate title and labels.
    plt.title(title, fontsize=26)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    
    # Enlarge tick marks.
    plt.yticks(fontsize=18)
    plt.xticks(cov_confirmed.index[0::steps], fontsize=18);


# ## Generate plot of Worldwide Confirmed Cases of COVID-19

# In[ ]:


# Generate a time plot.
plot_series(cov_confirmed, ['ConfirmedCases'], title = 'Worldwide Count of Confirmed COVID-19 Cases', steps = 14)


# ### Difference to generate a stationary model that can be forecasted

# In[ ]:


# first 5 values of the COVID-19 series.
cov_confirmed['ConfirmedCases'][0:5]


# ## First 5 values of the COVID-19 series differenced once.

# In[ ]:


cov_confirmed['ConfirmedCases'][:5].diff(1)


# ## First 5 values of the COVID-19 series differenced twice.

# In[ ]:


cov_confirmed['ConfirmedCases'][:5].diff(1).diff(1)


# ## First 5 values of the COVID-19 series differenced thrice.

# In[ ]:


cov_confirmed['ConfirmedCases'][:5].diff(1).diff(1).diff(1)


# ## Create first_diff_confirmed and second_diff_confirmed columns in cov_confirmed

# In[ ]:


cov_confirmed['first_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1)
cov_confirmed['second_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1).diff(1)
cov_confirmed['third_diff_confirmed'] = cov_confirmed['ConfirmedCases'].diff(1).diff(1).diff(1)
cov_confirmed.head()


# ## Plot first diff to see how much more stationary the data looks

# In[ ]:


# Examine confirmed cases, differenced once.
plot_series(cov_confirmed,
            ['first_diff_confirmed'],
            title = "Change in Confirmed Cases from Day to Day",
            steps=14)


# ## Still trending upward, lets look at the second diff

# In[ ]:


# Examine confirmed cases, differenced twice.
plot_series(cov_confirmed,
            ['second_diff_confirmed'],
            title = "Change in Confirmed Cases from Day to Day",
            steps=14)


# ## Looking better, but still a slight upward trend, lets try one more

# In[ ]:


# Examine confirmed cases, differenced thrice.
plot_series(cov_confirmed,
            ['third_diff_confirmed'],
            title = "Change in Confirmed Cases from Day to Day",
            steps=14)


# ## The Third Diff looks stationary, lets test them all with Augmented Dickey Fuller test

# In[ ]:


# Import Augmented Dickey-Fuller test.
from statsmodels.tsa.stattools import adfuller

# Run ADF test on original (non-differenced!) data.
adfuller(cov_confirmed['ConfirmedCases'])


# ## Time to interpret the results

# In[ ]:


# Code written by Joseph Nelson.

def interpret_dftest(dftest):
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    return dfoutput


# ## Starting with the non-differenced data

# In[ ]:


# Run ADF test on original (non-differenced!) data.

interpret_dftest(adfuller(cov_confirmed['ConfirmedCases']))


# ## with an alpha of 0.01, a p-value of nearly 1 is too high, lets look at the results of the first diff

# In[ ]:


# Run the ADF test on our once-differenced data.
interpret_dftest(adfuller(cov_confirmed['first_diff_confirmed'].dropna()))


# ## The p-value appears to be 1 for the first diff as well

# In[ ]:


# Run the ADF test on our twice-differenced data.
interpret_dftest(adfuller(cov_confirmed['second_diff_confirmed'].dropna()))


# ## Now, the second diff provides a p-value that is significantly lower than 0.01 - we'll go with this one. No need to even bother testing the third diff.
# 

# ## Fit an Arima Model using d=2

# ### Create train-test split

# In[ ]:


# Create train-test split.
y_train, y_test = train_test_split(cov_confirmed['second_diff_confirmed'], test_size=0.1, shuffle=False)


# ## Grid Search to determeine best p and q

# In[ ]:


# Starting AIC, p, and q.
best_aic = 99 * (10 ** 16)
best_p = 0
best_q = 0
# Use nested for loop to iterate over values of p and q.
for p in range(5):
    for q in range(5):
        # Insert try and except statements.
        try:
            # Fitting an ARIMA(p, 2, q) model.
            print(f'Attempting to fit ARIMA({p}, 2, {q}).')
            # Instantiate ARIMA model.
            arima = ARIMA(endog = y_train.dropna(), # endog = Y variable
                          order = (p, 2, q)) # values of p, d, q
            # Fit ARIMA model.
            model = arima.fit()
            # Print out AIC for ARIMA(p, 2, q) model.
            print(f'The AIC for ARIMA({p},2,{q}) is: {model.aic}')
            # Is my current model's AIC better than our best_aic?
            if model.aic < best_aic:
                # If so, let's overwrite best_aic, best_p, and best_q.
                best_aic = model.aic
                best_p = p
                best_q = q
        except:
            pass
print()
print()
print('MODEL FINISHED!')
print(f'Our model that minimizes AIC on the training data is the ARIMA({best_p},2,{best_q}).')
print(f'This model has an AIC of {best_aic}.')


# ## Instantiate Best Model and Plot the Data

# In[ ]:


# Instantiate best model.
model = ARIMA(endog = y_train.dropna(),  # Y variable
              order = (4, 2, 2))
# Fit ARIMA model.
arima = model.fit()
# Generate predictions based on test set.
preds = model.predict(params = arima.params,
                      start = y_test.index[0],
                      end = y_test.index[-1])
# Plot data.
plt.figure(figsize=(12,8))
# Plot training data.
plt.plot(y_train.index, pd.DataFrame(y_train).diff(), color = 'blue')
# Plot testing data.
plt.plot(y_test.index, pd.DataFrame(y_test).diff(), color = 'orange')
# Plot predicted test values.
plt.plot(y_test.index, preds, color = 'green')
plt.title(label = 'Twice-Differenced Confirmed Cases with ARIMA(0, 2, 1) Predictions', fontsize=16)
plt.show();


# ## Now to have a look at the fatalities using the same methods...... future work

# In[ ]:


cov_fatal = cov.groupby('Date')[['Fatalities']].sum()


# In[ ]:


cov_fatal.tail()


# In[ ]:


# Code modified from code written by Matthew Garton.

def plot_series(cov_fatal, cols=None, title='Title', xlab=None, ylab=None, steps=1):
    
    # Set figure size to be (18, 9).
    plt.figure(figsize=(18,9))
    
    # Iterate through each column name.
    for col in cols:
            
        # Generate a line plot of the column name.
        # You only have to specify Y, since our
        # index will be a datetime index.
        plt.plot(cov_fatal[col])
        
    # Generate title and labels.
    plt.title(title, fontsize=26)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    
    # Enlarge tick marks.
    plt.yticks(fontsize=18)
    plt.xticks(cov_confirmed.index[0::steps], fontsize=18);


# In[ ]:


# Generate a time plot.
plot_series(cov_fatal, ['Fatalities'], title = 'Worldwide Count of COVID-19 Fatalities', steps = 14)


# In[ ]:





# In[ ]:





# In[ ]:




