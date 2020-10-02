#!/usr/bin/env python
# coding: utf-8

# # Swedish central bank interest rate and inflation
# 
# Original link and decription at https://www.kaggle.com/cnygaard/sweden-interest-rate-inflation
# 
# Inspiration: Question: How does central bank interest rate effect inflation? What are the interest rate inflation rate delays? Verify ROC R^2 inflation/interest rate causation.

# In[ ]:


import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn import cross_validation, linear_model, metrics, ensemble
### R2 estimation
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.graphics.regressionplots import plot_leverage_resid2
###
get_ipython().run_line_magic('pylab', 'inline')

pd.options.mode.chained_assignment = None


# ### Load and clear the data

# In[ ]:


raw = pd.read_csv('../input/Interestrate and inflation Sweden 1908-2001.csv')
raw.head()


# In[ ]:


raw.info()


# In[ ]:


# let's drop all NA
data = raw.dropna()
# price level uses comma to separate thousands, let's fix that
data['Price level'] = data['Price level'].apply(lambda x: re.sub(',', '', x))

cols = ['Period', 'Central bank interest rate diskonto average', 'Price level']
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
data.info()


# In[ ]:


# plot period/interest rate
sns.pairplot(data)


# In[ ]:


data.corr(method='pearson')


# In[ ]:


data.corr(method='spearman')


# ## Trivia model

# In[ ]:


def time_series_cv(estimator, X, y, folds=5, metrics_f=metrics.mean_squared_error):
    '''
    Performs cross validation on ESTIMATOR and the data (X, y) using
    forward chaining.
    The score is the result of metrics_f(prediction, train_part_y)
    Here is the example of the data split to 6 folds
    for list [1 2 3 4 5 6]:
    TRAIN | TEST
    [1] | [2]
    [1 2] | [3]
    [1 2 3] | [4]
    [1 2 3 4] | [5]
    [1 2 3 4 5] | [6]
    '''
    assert X.shape[0] == y.shape[0], "Features and targets of different sizes {} != {}".format(    X.shape[0], y.shape[0]
    )

    results = []
    fold_size = int(np.ceil(X.shape[0] / folds))

    for i in range(1, folds):
        split = i*fold_size
        trainX, trainY = X[:split], y[:split]
        testX, testY = X[split:], y[split:]
        estimator.fit(trainX, trainY)
        predictions = estimator.predict(testX)
        results.append(metrics_f(predictions, testY))

    return np.array(results)


# In[ ]:


# formally, inflation <--> int.rate dependency is kind of a time series, so we have to use 
# "time-series" version of CV
regressor = linear_model.LinearRegression(fit_intercept=True)
report = time_series_cv(regressor, data[['Central bank interest rate diskonto average']],
                                                     data[['Inflation']], folds=5, metrics_f=metrics.mean_absolute_error)
print(report)
print('mean={}; std={}'.format(np.mean(report), np.std(report)))
data['Inflation'].plot(kind='hist', title='Inflation hist')


# In[ ]:


regressor.fit(data[['Central bank interest rate diskonto average']],
              data[['Inflation']])
print(regressor.coef_, regressor.intercept_)
plt.plot(data['Central bank interest rate diskonto average'],
         data['Inflation'], '*', label='original points')
w0, w1 = regressor.intercept_[0], regressor.coef_[0][0]
x = np.arange(0, 12, 0.2)
plt.plot(x, w0 + w1 * x, label='predicted curve')
plt.legend(loc=2)


# ## R-squared estimation

# In[ ]:


data.rename(columns={'Central bank interest rate diskonto average': 'Interest'}, inplace=True)
data.head()


# In[ ]:


formula = 'Inflation ~ Interest'
model = smf.ols(formula, data=data)
fitted = model.fit()

print(fitted.summary())


# In[ ]:


print('Breusch-Pagan test: p=%f' % sms.het_breushpagan(fitted.resid, fitted.model.exog)[1])


# The error seems to be homoscedastic. The significance calculated correctly, but the estimation may be biased

# In[ ]:




