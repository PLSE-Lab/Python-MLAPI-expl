#!/usr/bin/env python
# coding: utf-8

# # **1. Objective:**
# 
# ### *Comparing  portfolio construction methods using different approaches based on the S&P 500 stocks :*
# 
# ### The compared approaches are:
# 
#    #### 1- Equally Weighted Approach-**EW**, in which each stock gets an equal weight.
#    
#   #### 2- Efficient Frontier Approach == Maximum Sharpe Ration Approach- **MSR** in which the weights are calculated to maximize the sharpe ratio.
#    
#   #### 3- Global Minimum Variance Approach - **GMV** in which the weights are calculated to minimize the volatility of the portfolio.
#    
#    

# ### *In order to compare the returns of these approaches in an unseen period, we need to divide our data into two periods:*
# 
#   **1- From 2013-2015 - Train Period**
#  
#   **2- From 2015 to 2018 - Test Period**
#   
#   It's noteworthy to mention that I haven't used any machine learning solution in this part of the project and the "taining" and "test" words are used only to define the "seen" and "unseen" data sets.

# ## 2. Importing Libraries
# Firstly, let's import libraries and os:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 3. Reading Data Set
# Now let's read the csv file from kaggle data set; 
# 
# In order to easily divide our data set into two or more time periods, we can parse dates in the data set:

# In[ ]:


stocks_df = pd.read_csv("../input/sandp500/all_stocks_5yr.csv", index_col = 0, parse_dates= True)


# ## 4. EDA
# Let's run a simple EDA on the data set to take a glance at our data:

# In[ ]:


stocks_df.head()


# In[ ]:





# In[ ]:


stocks_df.tail()


# ### *Stock Names (Symbols)
# Let's look at the unique stock names in the data frame:

# In[ ]:


unique_stocks_names = stocks_df.Name.unique()
unique_stocks_names


# ### *Price Chart
# We can plot the price changes of the stocks using each day's close price. 
# Let's define a function to simplify future plotting tasks:

# In[ ]:


def select_and_plot_stock(stock):
  get_ipython().run_line_magic('matplotlib', 'inline')
  stock_df = stocks_df.loc[stocks_df.Name == stock, :]
  plt.figure(figsize = (10,10))
  plt.plot(stock_df.index, stock_df.close)
  plt.title(f"Stock Name : {stock}")
  plt.xlabel("Date")
  plt.ylabel("Close Price")
  plt.grid = True
  plt.show()


# Now let's plot two instances:

# In[ ]:


select_and_plot_stock("AAL")
select_and_plot_stock("AMZN")


# ### *Daily Returns
# Now let's build a data frame of the daily returns of each stock.
# 
# We can benefit from a loop to build the daily return of each stock:

# In[ ]:


stocks = unique_stocks_names
daily_return = pd.DataFrame() #create a DataFrame of daily returns
for n in range(len(stocks)):
        stock_df = stocks_df.loc[stocks_df.Name == stocks[n], :]
        stock_daily_return = stock_df["close"].pct_change() #usring pct_change function can give us the daily return
        daily_return[stocks[n]] = stock_daily_return
        n += 1
daily_return = daily_return.drop(daily_return.index[0],axis=0) #drop the first column which doesn't have data in daily return data frame
daily_return


# > Now let's define a function to be able to plot daily returns for any list of stocks:

# In[ ]:


def plot_daily_return(stock_names):
    for n in range(len(stock_names)):
        plt.figure(figsize = (20,10))
        plt.subplot(len(stock_names),1,n+1)
        plt.plot(daily_return.index, daily_return[stock_names[n]])
        plt.title(f"Stock Name : {stock_names[n]}")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.grid = True
        plt.show()
        plt.grid = True
        n+=1


# For instance, let's plot daily returns of three randomly selected stocks:

# In[ ]:


names = ["AMZN", "EBAY", "AAL"]
plot_daily_return(names)


# ### * Daily Returns Distribution
# We can also plot the daily return distribution for each stock using histogram plot.
# 
# Let's define a function for this application:

# In[ ]:


def dist_plot_return(stock_names):
    for n in range(len(stock_names)):
        plt.figure(figsize = (10,10))
        plt.subplot(len(stock_names),1,n+1)
        daily_return_stock = daily_return.loc[:, stock_names[n]]
        daily_return_stock.plot.hist(bins = 50)
        plt.title(f"Stock Name : {stock_names[n]}")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.grid = True
        plt.show()
        plt.grid = True
        n+=1


# For instance, let's plot the same three stocks:

# In[ ]:


names = ["AMZN", "EBAY", "AAL"]
dist_plot_return(names)


# ### * Annualized Return
# we can calculate the annualized return of a given stock returns:

# In[ ]:


def annualize_rets(r, periods_per_year):
  """
  Annulalizes a set of returns
  """

  compounded_grouwth = (1+r).prod()
  n_periods = r.shape[0]
  return compounded_grouwth**(periods_per_year/n_periods)-1


# In[ ]:



annualize_rets(daily_return["AMZN"], 252) #The operational days of the stocks market in a year is 252 days


# ### *Drawdown
# Also, we can see the drawdown of a given returns set.
# 
# Here, we assume that the starting equity is $1000 and will see what will be the fial equity after each day:

# In[ ]:


def drawdown(returns_series: pd.Series):
  """Takes a time series of asset returns. 
     returns a DataFrame with columns for the wealth index,
     the previous peaks, and
     the percentage drawdown
  """
  wealth_index = 1000*(1+returns_series).cumprod()
  previous_peaks = wealth_index.cummax()
  drawdowns = (wealth_index - previous_peaks)/previous_peaks
  return pd.DataFrame({"Wealth" : wealth_index, "Previous Peak" : previous_peaks, "Drawdown" : drawdowns})


# Let's build the "AMZN" stock drawdown DataFrame.

# In[ ]:


drawdown(daily_return["AMZN"])


# Now we can plot the peaks and wealth index of the stock:

# In[ ]:


plt.figure(figsize = (10, 10))
drawdown(daily_return["AMZN"])[["Wealth", "Previous Peak"]].plot()
plt.show()


# We also can see the plot of the drawdown itself:

# In[ ]:


plt.figure(figsize = (10, 5))
drawdown(daily_return["AMZN"])["Drawdown"].plot()


# ### *Skewness and Kurtosis
# There are some other parameters that are important for a portfolio analyst such as 
# the Skewness and Kurtosis of the return distribution graph:

# In[ ]:


def skewness (r):
  """
  Alternative to scipy.stats.skew()
  computes the skewness of the supplied series or dataframe
  returns a float or series
  """
  demeaned_r = r - r.mean()
  #use the population standard deviation, so set dof=0
  sigma_r = r.std(ddof=0)
  exp = (demeaned_r ** 3).mean()
  return exp/sigma_r ** 3


# In[ ]:


skewness_df = pd.DataFrame(skewness(daily_return))
skewness_df.columns = ["Skewness"]
skewness_df


# Let's see the skewness graphs for all of the stocks:

# In[ ]:


for n in range (0,506, 30):
    sample_skewness_df = skewness_df[n : n+20].sort_values(by = ["Skewness"])
    get_ipython().run_line_magic('matplotlib', 'inline')
    sample_skewness_df.plot.bar()
    plt.show()


# We can also calculate the kurtosis parameter for return distributions:

# In[ ]:


def kurtosis(r):
  demeaned_r = r - r.mean()
  sigma_r = r.std(ddof=0)
  exp = (demeaned_r**4).mean()
  return exp/sigma_r**4


# In[ ]:


kurtosis_df = pd.DataFrame(kurtosis(daily_return))
kurtosis_df.columns = ["Kurtosis"]
kurtosis_df


# In[ ]:


for n in range (0,506, 30):
    sample_kurtosis_df = kurtosis_df[n : n+20].sort_values(by = ["Kurtosis"])
    get_ipython().run_line_magic('matplotlib', 'inline')
    sample_kurtosis_df.plot.bar()
    plt.show()


# You can see that our returns are not normally distributed at all (Kurtosis=3 and Skewness =0 represents a Normal (Gaussian) distribution)

#  ## 5. Portfolio Construction
# For calculating weights for our desire portfolio and test them on the unseen data, we need to divide our data set into 2 periods.
# 
# I chose two periods as training and testing periods :
# 
#   1- 2013 to 2015 period as my training data
#   
#   2- 2016 to 2018 period as my testing data

# ### *Dividing data set into two periods:
# In order to build this data sets, we simply can benefit from the date parsing capability of pandas library we used while reading our csv file:

# In[ ]:


er = annualize_rets(daily_return[:"2015"],252) #annualized expected return in the 2013 to 2015 period
er


# ### *Covariance Matrix
# For the efficient frontier approach, we need the covariance matrix of the stocks. 
# 
# Fortunately, pandas has a method simply builds this matrix:

# In[ ]:


cov = daily_return.cov()
cov.head()


# ### *Calculating Portfolio Returns
# Now, let's define a function to calculate the annualized return of a given portfolio using the weights array of the stocks:

# In[ ]:


def portfolio_return(weights, returns):
  """
  weights --> Returns
  """
  return weights.T @ returns # transposing the weights matrix and multiply it by returns


# For each approach, we need an array representing the weights of the stocks in our portfolio.
# In the **EW approach**, we only need to allocate weights equally to each stock.
# In the MSR approach though, we need to calculate the optimized weights that give us a portfolio with the maximum sharpe ratio.
# Finally, in the GMV approach, we need to calculate the weights which give us a portfolio with the minimum volatility.
# 
# In order to find the optimum weights based on each approach, I used the minimizing optimizer method in numpy.

# To do this task, we need to define some functions:
# 
# 
# ### *Portfolio Volatility
# First of all, a function which calculates the volatility of the portfolio given the weights

# In[ ]:


def portfolio_vol(weights, covmat):
  """
  Weights --> Vol
  """
  return (weights.T @ covmat @ weights)**0.5


# ### *Optimizer Function
# Then, the minimizer function which gives the optimum weights for a given return;
# As we need the minumum volatility for the GMV approach, and the maximum Sharpe Ratio for the MSR approach, we can benefit from this minimizer method in the scipy library. (A trick here is that we can find the  **minimum of the "negative Sharpe Ratio"**  instead of the possible maximum Sharpe Ratio)

# In[ ]:


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
  """
  target_ret ==> W 
  """
  n = er.shape[0] #number of weights
  init_guess = np.repeat(1/n, n) #makes a tuple of n tuples for weights
  bounds = ((0.0, 1.0),)*n #defines the max and min of our possible weights
  return_is_target = {
      "type" : "eq",
      "args" : (er,),
      "fun" : lambda weights, er :
       target_return - portfolio_return(weights, er)
  } 
    
  weights_sum_to_one = {
      "type" : "eq",
      "fun" : lambda weights: np.sum(weights) - 1
      }
  results = minimize(
      portfolio_vol, init_guess,
                     args = (cov,), method = "SLSQP",
                     options = {"disp" : False},
                     constraints = (return_is_target, weights_sum_to_one),
                     bounds = bounds
                    )
  return results.x


# In[ ]:


def optimal_weights(n_points, er, cov):
  """
  Generates a list of weights to run the optimizer on, to minimize the volatility
  """
  target_rs = np.linspace(er.min(), er.max(), n_points)
  weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
  return weights


# ### *MSR Approach
# Let's define a function for the MSR Approach; As I mentioned before, we can find the minimum "negative Sharpe Ratio" so that we could find the maximum possible Sarpe Ratio and the allocated weights:

# In[ ]:


def msr(riskfree_rate, er, cov):
  """
  Returns the weights of the portfolio that gives you the maximum sharpe ratio given the riskfree rate and expected returns and a covariancce matrix
  """
  n = er.shape[0] 
  init_guess = np.repeat(1/n, n) 
  bounds = ((0.0, 1.0),)*n 
  weights_sum_to_one = {
      "type" : "eq",
      "fun" : lambda weights: np.sum(weights) - 1
  }
  def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
    """
    Returns the negative of the sharpe ratio, given weights
    """
    r = portfolio_return(weights, er) 
    vol = portfolio_vol(weights, cov) 
    return -(r - riskfree_rate)/vol

  results = minimize(neg_sharpe_ratio, init_guess,
                     args=(riskfree_rate, er, cov,), method="SLSQP",
                     options={"disp" : False},
                     constraints = (weights_sum_to_one),
                     bounds=bounds
                    )
  return results.x


# ### *Global Minimum Variance (GMV) Approach
# We can benefit from the same msr function we built for the msr approach. But in this case, our risk free rate is assumed to be zero and our expected return series doesn't have any role:

# In[ ]:


def gmv(cov):
  """
  Returns the weights of the Global Minimum Vol Portfolio given covariance matrix
  """
  n = cov.shape[0]
  return msr(0, np.repeat(1,n), cov)


# ### *Plotting Efficient Frontier Curve, MSR, GMV, and EW points
# Now, let's define a function to plot three spots on the returns-volatility graph using three different approaches (EF, GMV, EW.
# 
# I used the **midnight blue** color for the **GMV approach**, the **red** color for the **EW** approach, and the **green** color for the **MSR** approach

# In[ ]:


def plot_ef(n_points, er, cov, show_cml=False, show_ew=False, show_gmv=False, riskfree_rate=0, style=".-"):
  """
  Plots the efficient frontier curve, msr, gmv, and ew points.
  """
  weights = optimal_weights(n_points, er, cov)
  rets = [portfolio_return(w, er) for w in weights]
  vols = [portfolio_vol(w, cov) for w in weights]
  ef = pd.DataFrame({"Returns" : rets, "Volatility" : vols})
  ax = ef.plot.line(x="Volatility", y = "Returns", style = style)
  if show_gmv:
    w_gmv = gmv(cov)
    r_gmv = portfolio_return(w_gmv, er)
    vol_gmv = portfolio_vol(w_gmv, cov)
    #displat EW
    ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker = "o", markersize=12)
  if show_ew:
    n = er.shape[0]
    w_ew = np.repeat(1/n , n)
    r_ew = portfolio_return(w_ew, er)
    vol_ew = portfolio_vol(w_ew, cov)
    #displat EW
    ax.plot([vol_ew], [r_ew], color="red", marker = "o", markersize=10)  
  if show_cml:
    ax.set_xlim(left = 0)
    rf = 0.1
    w_msr = msr(riskfree_rate, er=er, cov=cov)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov)
    #Add CML
    cml_x = [0, vol_msr]
    cml_y = [riskfree_rate, r_msr]
    ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed", markersize = 12, linewidth =2)
  return ax


# In[ ]:


plot_ef(20, er, cov, show_cml=True, show_ew=True, show_gmv=True, riskfree_rate=0.03)


# ### *Calculating each portfolio returns using allocated weights:
# Now, let's see the weight array of each approach and calculate the annualized return in the training period for each one:

# ### ** *EW Approach: **

# In[ ]:


ew_weights = np.repeat(1/er.shape[0], er.shape[0])
ew_weights[:30] # We can see the firts 30 allocated weights


# In[ ]:


portfolio_return(ew_weights, er)


# **MSR Approach:**

# In[ ]:


msr_weights = msr(0.03, er, cov)
msr_weights[:30]


# In[ ]:


portfolio_return(msr_weights, er)


# **GMV Approach:**

# In[ ]:


gmv_weights = gmv(cov)
gmv_weights[:30]


# In[ ]:


portfolio_return(gmv_weights, er)


# ## 6. Testing portfolio weights on the test period:
# Using the calculated weights in each construction method, we can now calculate our actual returns for the unseen data (2016-2018 period):

# Let's define our annualized return data set for this period

# In[ ]:


rets = annualize_rets(daily_return["2016":],252)
rets


# Now let's see what would be our actual return if we used the obtained weights in each approach.
# 

# **EW Approach:**

# In[ ]:


portfolio_return(ew_weights, rets)


# **MSR Approach:**

# In[ ]:


portfolio_return(msr_weights, rets)


# **GMV Approach:**

# In[ ]:


portfolio_return(gmv_weights, rets)

