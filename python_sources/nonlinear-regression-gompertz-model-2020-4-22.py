#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize


# ## Importing dataset

# In[ ]:


train_dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test_dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
submission = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')


# In[ ]:


submission.head()


# ## View information about the dataset.

# In[ ]:


train_dataset.info()


# In[ ]:


train_dataset.describe()


# In[ ]:


train_dataset.head()


# In[ ]:


test_dataset.head()


# In[ ]:


train_dataset.isna().sum()


# In[ ]:


test_dataset.isna().sum()


# ## Combine **'Province_State'** and **'Country_Region'**

# In[ ]:


train_dataset['Province_State'].fillna('', inplace = True)
test_dataset['Province_State'].fillna('', inplace = True)


# In[ ]:


train_dataset['Country_Region'] = train_dataset['Country_Region'] + ' ' + train_dataset['Province_State']
test_dataset['Country_Region'] = test_dataset['Country_Region'] + ' ' + test_dataset['Province_State']
del train_dataset['Province_State']
del test_dataset['Province_State']


# In[ ]:


train_dataset.head()


# In[ ]:


test_dataset.head()


# In[ ]:


# How many countries
train_dataset['Country_Region'].describe()


# In[ ]:


country_list = train_dataset['Country_Region'].unique()


# In[ ]:


train_date = train_dataset.Date.unique()
train_date


# In[ ]:


test_date = test_dataset.Date.unique()
test_date


# In[ ]:


train_days = np.arange(len(train_date))
train_days


# In[ ]:


train_days[train_date == '2020-04-02']


# In[ ]:


train_days[train_date == '2020-04-22']


# In[ ]:


test_days = np.arange(len(test_date)) + 71
test_days


# In[ ]:


train_end = train_days[train_date == '2020-04-22']
test_start = test_days[0]


# In[ ]:


train_end


# In[ ]:


test_start


# In[ ]:


Day = np.zeros(len(train_dataset))
for ii in range(len(train_date)):
    Day[train_dataset.Date == train_date[ii]] = train_days[ii]
train_dataset['Day'] = Day


# In[ ]:


train_dataset.head(5)


# In[ ]:


Day = np.zeros(len(test_dataset))
for ii in range(len(test_date)):
    Day[test_dataset.Date == test_date[ii]] = test_days[ii]
test_dataset['Day'] = Day


# In[ ]:


test_dataset.head(5)


# ## Top 10 confirmed cases countries (2020-04-22)

# In[ ]:


top_comfirmedcases = train_dataset[train_dataset.Date == '2020-04-22'].sort_values(by = 'ConfirmedCases', ascending = False)
top_comfirmedcases.head(10)


# In[ ]:


def country_plot(country):
    train = train_dataset[train_dataset['Country_Region'] == country]
    test = test_dataset[test_dataset['Country_Region'] == country]
    
    # X_train
    x_train = train.Day.values
    confirmed_train = train.ConfirmedCases.values
    fatalities_train = train.Fatalities.values
    
    # Plot figures
    # Confirmed cases
    plt.figure(figsize = (15, 3))
    plt.subplot(1, 2, 1)
    plt.xlabel('Days')
    plt.ylabel('Confirmed cases')
    plt.title(country)
    plt.plot(x_train, confirmed_train)
    plt.grid()

    # Fatalities
    plt.subplot(1, 2, 2)
    plt.xlabel('Days')
    plt.ylabel('Fatalities')
    plt.title(country)
    plt.plot(x_train, fatalities_train, color = 'orange')
    plt.grid()
    plt.show()


# In[ ]:


for country in top_comfirmedcases.Country_Region[0:9].values:
    country_plot(country)


# ## Gompertz model
# 
# $$f(t) = \theta_{1} e^{-\theta_{2} e^{-\theta_{3} t}}$$
# 
# where
# 
# - $\theta_{1}$ is an asymptote, since $\lim_{t \to \infty} f(t) = \theta_{1}$ 
# - $\theta_{2}$ sets the displacement along the x-axis (translates the graph to the left or right). Symmetry is when $\theta_{2} = \log(2)$.
# - $\theta_{3}$ sets the growth rate (y scaling)
# 
# Reference: [wiki](https://en.wikipedia.org/wiki/Gompertz_function)

# In[ ]:


def Gompertz(t, theta1, theta2, theta3):
    '''
    theta1: The asymptote.
    theta2: The displacement along the x-axis.
    theta3: The growth rate.
    '''
    f = theta1 * np.exp(-theta2 * np.exp(-theta3 * t))
    return f


# In[ ]:


x = np.linspace(start = -2, stop = 5, num = 50)
y1 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 1)
y2 = Gompertz(x, theta1 = 5, theta2 = 1.5, theta3 = 1)
y3 = Gompertz(x, theta1 = 5, theta2 = 2, theta3 = 1)

plt.figure(figsize = (12, 8))
plt.plot(x, y1, label = 'y1')
plt.plot(x, y2, label = 'y1')
plt.plot(x, y3, label = 'y1')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


x = np.linspace(start = -2, stop = 5, num = 50)
y1 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 0.1)
y2 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 0.5)
y3 = Gompertz(x, theta1 = 5, theta2 = 1, theta3 = 1)

plt.figure(figsize = (12, 8))
plt.plot(x, y1, label = 'y1')
plt.plot(x, y2, label = 'y1')
plt.plot(x, y3, label = 'y1')
plt.legend()
plt.grid()
plt.show()


# ## Example: 'Korea, South '

# In[ ]:


country = 'Korea, South '
train = train_dataset[train_dataset['Country_Region'] == country]
test = test_dataset[test_dataset['Country_Region'] == country]

# X_train
x_train = train.Day.values
confirmed_train = train.ConfirmedCases.values
fatalities_train = train.Fatalities.values

# X_test
x_test = test.Day.values
country_plot(country)


# ## **Least-Squared-Estimation**: scipy.optimize.curve_fit

# In[ ]:


popt_confirmed, pcov_confirmed = curve_fit(f = Gompertz, 
                                           xdata = x_train, 
                                           ydata = confirmed_train, 
                                           p0 = [3 * max(confirmed_train), 1, 1], 
                                           maxfev = 800)


# In[ ]:


popt_confirmed


# In[ ]:


pcov_confirmed


# In[ ]:


def curve_plot(x_train, y_train, x_test, est):
    plt.figure(figsize = (12, 5))
    plt.xlabel('Days')
    plt.ylabel('Cases')
    plt.title(country)
    plt.scatter(x_train, y_train, color = 'r')
    plt.plot(x_train, Gompertz(x_train, *est), label = 'Fitting curve (train)')
    plt.plot(x_test, Gompertz(x_test, *est), label = 'Fitting curve (test)')
    plt.axvline(x = test_start, color = 'r', linestyle = ':', label = 'test_start = %.f' % (test_start))
    plt.axvline(x = train_end, color = 'b', linestyle = ':', label = 'train_end = %.f' % (train_end))
    plt.legend()
    plt.show()


# In[ ]:


curve_plot(x_train = x_train, y_train = confirmed_train, x_test = x_test, est = popt_confirmed)


# In[ ]:


popt_fatalities, pcov_fatalities = curve_fit(f = Gompertz, 
                                             xdata = x_train, 
                                             ydata = fatalities_train, 
                                             p0 = [3 * max(fatalities_train), 1, 1], 
                                             maxfev = 800)


# In[ ]:


popt_fatalities


# In[ ]:


pcov_fatalities


# In[ ]:


curve_plot(x_train = x_train, y_train = fatalities_train, x_test = x_test, est = popt_fatalities)


# ## **Minimized loss function:** scipy.optimize.minimize
# 
# Consider the nonlinear regression model
# 
# $$y_{i} = f(t_{i};\theta) + \varepsilon_{i},\quad  i=1, 2, ..., n$$
# 
# The function is given by
# 
# $$f(t;\theta) = \theta_{1} e^{-\theta_{2} e^{-\theta_{3} t}}$$
# 
# , where $\theta_{1} > 0$, $\theta_{2} > 0$, $\theta_{3} > 0$.
# 
# The estimator $(\hat{\theta}_{1}, \hat{\theta}_{2}, \hat{\theta}_{3})$ is obtained by minimizing loss function
# 
# $$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} [y_{i} - f(t_{i};\theta)]^{2}$$

# In[ ]:


# Minimize the Loss function: MSE
def growth_curve(x, y):
    # Loss function
    def l_fun(params): 
        theta1 = np.exp(params[0])
        theta2 = np.exp(params[1])
        theta3 = np.exp(params[2])
        mse = np.mean((y - Gompertz(x, theta1, theta2, theta3)) ** 2)
        return mse

    p0 = [np.log(3 * max(y)), 0, 0]
    res = minimize(fun = l_fun, x0 = p0, method = 'L-BFGS-B')
    mse = res.fun

    # convergence_res
    convergence_res = {'MSE': mse,
                       'nfev': res.nfev, 
                       'nit': res.nit, 
                       'status': res.status}

    # Estimator
    est = np.exp(res.x)
    return est, convergence_res


# In[ ]:


# Confirmed cases
est_confirmed, convergence_res = growth_curve(x = x_train, y = confirmed_train)
convergence_res


# In[ ]:


curve_plot(x_train = x_train, y_train = confirmed_train, x_test = x_test, est = est_confirmed)


# In[ ]:


# Confirmed cases
est_fatalities, convergence_res = growth_curve(x = x_train, y = fatalities_train)
convergence_res


# In[ ]:


curve_plot(x_train = x_train, y_train = fatalities_train, x_test = x_test, est = est_fatalities)


# ## Submission

# In[ ]:


confirmed_pred = np.zeros(len(test_dataset))
fatalities_pred = np.zeros(len(test_dataset))


# In[ ]:


for country in country_list:
    train = train_dataset[train_dataset['Country_Region'] == country]
    test = test_dataset[test_dataset['Country_Region'] == country]
    
    # X_train
    x_train = train.Day.values
    confirmed_train = train.ConfirmedCases.values
    fatalities_train = train.Fatalities.values
    
    # X_test
    x_test = test.Day.values

    # Confirmed cases
    confirmed_est, confirmed_convergence = growth_curve(x = x_train, y = confirmed_train)
    
    # Fatalities    
    fatalities_est, fatalities_convergence = growth_curve(x = x_train, y = fatalities_train)
    
    # Predictions
    confirmed_pred[test_dataset.Country_Region == country] = Gompertz(x_test, *confirmed_est)
    fatalities_pred[test_dataset.Country_Region == country] = Gompertz(x_test, *fatalities_est)


# In[ ]:


submission['ConfirmedCases'] = confirmed_pred
submission['Fatalities'] = fatalities_pred


# In[ ]:


submission.to_csv('submission.csv', index = False)

