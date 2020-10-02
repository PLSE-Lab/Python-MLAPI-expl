#!/usr/bin/env python
# coding: utf-8

# **1&nbsp;&nbsp;[Introduction](#1)**  
# **2&nbsp;&nbsp;[Environment](#2)**  
# &nbsp;&nbsp;&nbsp;&nbsp;2.1&nbsp;&nbsp;[Libraries](#2.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;2.2&nbsp;&nbsp;[Data](#2.2)  
# **3&nbsp;&nbsp;[Optimisation](#3)**  
# &nbsp;&nbsp;&nbsp;&nbsp;3.1&nbsp;&nbsp;[Constants, parameters and variables](#3.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;3.2&nbsp;&nbsp;[Objective function](#3.2)  
# &nbsp;&nbsp;&nbsp;&nbsp;3.3&nbsp;&nbsp;[Constraints](#3.3)  
# &nbsp;&nbsp;&nbsp;&nbsp;3.4&nbsp;&nbsp;[Solution](#3.4)  
# **4&nbsp;&nbsp;[Evaluation](#4)**  
# &nbsp;&nbsp;&nbsp;&nbsp;4.1&nbsp;&nbsp;[Parameterised portfolio](#4.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;4.2&nbsp;&nbsp;[Optimal portfolio](#4.2)  
# **5&nbsp;&nbsp;[Conclusion](#5)**

# ## 1 Introduction<a id="1"></a>

# Given a set of $n$ assets:
# * $\mathbf{p} \in \mathbb{R}_+^n$ is the price vector where $p_i$ is the price of asset $i$.
# * $\mathbf{x} \in \mathbb{Z}^n$ is the portfolio allocation vector where $x_i$ is the number of shares in asset $i$ to buy. To prevent shorting: $\mathbf{x}\geq0$.
# * In one time period, $\mathbf{r} \in \mathbb{R}^n$ is the return vector where $r_i$ is the return on asset $i$. The return is the difference in price divided by the price at the beginning of the period: $r_i = \frac{p_{t+1}-p_t}{p_t}$. Portfolio return $R$ is given by: $R = \mathbf{r}^T\mathbf{x}$.
# * The return vector $\mathbf{r}$ can be modelled as a random variable with mean $\mathbb{E}[\mathbf{r}]=\mathbf{\mu}$ and covariance $\mathbb{E}[(\mathbf{r}-\mathbf{\mu})(\mathbf{r}-\mathbf{\mu})^T]=\Sigma$. It follows that the portfolio return is also a random variable with mean $\mathbb{E}[R]=\mathbf{\mu}^T\mathbf{x}$ and variance $\mathrm{Var}[R]=\mathbf{x}^T\Sigma\mathbf{x}$.
# * $k \in \mathbb{R}_+^n$ is the maximum amount in one asset.

# The optimal portfolio allocation is found by maximising the expected portfolio fractional return and minimising the portfolio variance. This becomes a mixed-integer quadratic programming problem:
# $$\mathrm{arg}\max_{\mathbf{x}} \mathbf{\mu}^T\mathbf{x}-\frac{1}{2} \mathbf{x}^T\Sigma\mathbf{x}$$
# $$\mathrm{subject\ to}$$
# $$\mathbf{x} \in \mathbb{Z}^n$$
# $$\mathbf{x}\geq0$$
# $$p_i\mathbf{x} \leq k$$

# An equivalent optimisation problem is to set an upper bound $\sigma^2$ on the portfolio variance and maximise the expected portfolio fractional return:
# $$\mathrm{arg}\max_{\mathbf{x}} \mathbf{\mu}^T\mathbf{x}$$
# $$\mathrm{subject\ to}$$
# $$\mathbf{x} \in \mathbb{Z}^n$$
# $$\mathbf{x}\geq0$$
# $$\mathbf{x}^T\Sigma\mathbf{x} \leq \sigma^2$$
# $$p_i\mathbf{x}\leq k$$

# ## 2 Environment<a id="2"></a>

# ### 2.1 Libraries<a id="2.1"></a>

# Load libraries into notebook

# In[ ]:


import os                               # Operating system
import math                             # Mathematics
import numpy as np                      # Arrays
import pandas as pd                     # Dataframes
import matplotlib.pyplot as plt         # Graphs
from matplotlib import cm               # Colours
import scipy                            # Scientific computing
import cvxpy as cp                      # Convex optimisation
from mpl_toolkits.mplot3d import Axes3D # 3D graphs


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2.2 Data<a id="2.2"></a>

# Get list of stock names

# In[ ]:


stocks = [stock.split('.')[0] for stock in sorted(os.listdir('/kaggle/input/australian-historical-stock-prices'))]


# Create a dataframe containing dates

# In[ ]:


dates = pd.date_range('2000-01-01', '2020-03-31') # Create date range from 01-01-2000 to 31-03-2020
data = pd.DataFrame({'Time': dates})              # Add dates to dataframe with column name


# Append the adjusted closing price of each stock to a dataframe keyed on date

# In[ ]:


for stock in stocks:                                                          # For each stock
    prices = pd.read_csv(                                                     # Read prices into dataframe
        '/kaggle/input/australian-historical-stock-prices/' + stock + '.csv', # Get filename
        usecols=['Date', 'Adj Close']                                         # Select date and adjusted closing price
    )
    prices['Date'] = pd.to_datetime(prices['Date'])                           # Typecast dates to datetimes
    prices.rename(                                                            # Rename columns
        columns={"Date": "Time", "Adj Close": stock},
        inplace=True
    )
    data = pd.merge(                                                          # Add stock to master dataframe
        data,                                                                 # Initially contains dates only
        prices,                                                               # Insert stock prices
        how='left',                                                           # Left outer join
        on=['Time'],                                                          # Key on the time column
        sort=False
    )


# Remove non-trading days

# In[ ]:


data = data[data['Time'].dt.weekday < 5] # Remove weekend dates
data = data.dropna(axis=0, how='all') # Remove empty rows


# ## 3 Optimisation<a id="3"></a>

# ### 3.1 Constants, parameters and variables<a id="3.1"></a>

# Get last price for each stock

# In[ ]:


p = data     .drop(['Time'], axis=1)     .tail(1)     .to_numpy()


# Calculate weekly returns from 1 January 2019 onwards

# In[ ]:


r = data[(data['Time'].dt.weekday == 4) & (data['Time'] >= '2019-01-01')]     .drop(['Time'], axis=1)     .pct_change(fill_method='ffill')


# Calculate expected return and covariance matrix

# In[ ]:


sigma = r.cov().to_numpy()
mu = r.mean().to_numpy()


# Get number of stocks

# In[ ]:


n = len(stocks)


# Set optimisation variable and parameters

# In[ ]:


x = cp.Variable(shape=n, integer=True)
threshold = cp.Parameter(nonneg=True) # maximum portfolio variance
k = cp.Parameter(nonneg=True) # maximum allocation into one stock


# Formulate portfolio mean and variance

# In[ ]:


mean = mu.T*x
variance = cp.quad_form(x, sigma)


# ### 3.2 Objective function<a id="3.2"></a>

# Define the objective function (maximise expected portfolio return)

# In[ ]:


objective = cp.Maximize(mean)


# ### 3.3 Constraints<a id="3.3"></a>

# Define optimisation constraints

# In[ ]:


constraints = [
    x >= 0,                                 # no shorting
    variance <= threshold                   # upper bound on portfolio variance
]
for pi in p:
    constraints = constraints + [pi*x <= k] # upper bound on single-stock allocation


# ### 3.4 Solution<a id="3.4"></a>

# Initialise the optimisation problem using objective function and constraints

# In[ ]:


problem = cp.Problem(objective, constraints)


# Solve optimisation problem for each parameter combination

# In[ ]:


z_values = []
k_values = np.arange(1000, 5000, 1000)
threshold_values = np.arange(1, 5.5, 0.5)
for threshold_value in threshold_values:
    for k_value in k_values:
        threshold.value = threshold_value
        k.value = k_value
        problem.solve()
        if problem.status != 'optimal': continue
        counts = x.value.round()
        investments = p*counts
        returns = mu@investments[0]
        z_values.append(returns)


# ## 4 Evaluation<a id="4"></a>

# ### 4.1 Parameterised return<a id="4.1"></a>

# Plot expected portfolio return as a function of portfolio variance and maximum single-asset allocation

# In[ ]:


if len(z_values) == 0:
    print('No optimal solutions')
else:
    Z = np.reshape(z_values, (len(k_values), len(threshold_values)))
    figure = plt.figure(figsize = (12,10))
    axes = figure.add_subplot(111, projection='3d')
    for i in range(len(k_values))[::-1]:
        c = cm.jet(i/float(len(k_values)))
        axes.bar(
            threshold_values,
            Z[i,:],
            zs=k_values[i],
            zdir='y',
            width=0.4
        )
    axes.set_xlabel('Portfolio variance')
    axes.set_ylabel('Maximum single-asset allocation')
    axes.set_zlabel('Portfolio return')
    axes.set_title('Expected portfolio return vs. portfolio variance and maximum single-asset allocation')
    plt.show()


# ### 4.2 Optimal portfolio<a id="4.2"></a>

# Calculate optimal portfolio using highest variance and maximum single-asset allocation

# In[ ]:


stocks_optimal = np.array(stocks)[np.where(counts > 0)]
counts_optimal = counts[counts>0]
prices_optimal = np.around(np.array(p), 2)[0][np.where(counts > 0)]
investments_optimal = np.around(investments, 2)[investments > 0]
capital_optimal = np.around(counts_optimal@prices_optimal, 2)
risk_optimal = np.around(counts.T@sigma@counts, 2)
return_optimal = np.around(52*returns/capital_optimal, 3)


# Print results

# In[ ]:


print('Stocks:\t\t', stocks_optimal)
print('Counts:\t\t', counts_optimal)
print('Prices:\t\t', prices_optimal)
print('Investments:\t', investments_optimal)
print('Capital:\t', capital_optimal)
print('Return:\t\t', return_optimal)
print('Risk:\t\t', risk_optimal)


# ## 5 Conclusion<a id="5"></a>

# CSL and FPH, both growth companies in the Australian biotechnology sector, dominate the portfolio making up more than 80% of the total investment. While the expected portfolio return may have been maximised for a given portfolio variance based on historical price data, a more risk-averse portfolio will require further diversification. The expected return is based on past prices which is not a reasonable indicator of future performance.
