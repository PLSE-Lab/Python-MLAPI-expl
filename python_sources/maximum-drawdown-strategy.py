#!/usr/bin/env python
# coding: utf-8

# # Maximum Drawdown Volatility Measure
# 
# A notebook dedicated to understanding volatility measures on real-world data. 
# 
# **Maximum Drawdown:** A maximum drawdown (MDD) is the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. Maximum drawdown is an indicator of downside risk over a specified time period. It can be used both as a stand-alone measure or as an input into other metrics such as "Return over Maximum Drawdown" and the Calmar Ratio. Maximum Drawdown is expressed in percentage terms.
# 
# **Formula**
# 
# $MDD = \frac{Trough Value - Peak Value}{Peak Value}$
# 

# ## Import Necessary libraries

# In[ ]:


# Data Analysis
import numpy as np 
import pandas as pd 

# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns


# ## Read in the data 

# In[ ]:


monthly_data = pd.read_csv('../input/python/Portfolios_Formed_on_ME_monthly_EW.csv',
                           header=0, index_col=0, parse_dates=True, na_values=-99.99)
monthly_data.shape


# In[ ]:


# Let's look at the data
monthly_data.head()

## Data Preprocessing

Collect the data for Small Cap and Large Cap Companies named as Low10 and High 10
# In[ ]:


# Extract the data
returns = monthly_data[['Lo 10', 'Hi 10']]
returns.columns = ['SmallCap', 'LargeCap']
returns.head()


# In[ ]:


# Convert Returns to percentages 
returns = returns/100
returns


# Plot the returns to see the trend.

# In[ ]:


# Make a line plot of the returns
returns.plot.line()


# The index is actually messed up and we need to fix this by converting it to time series to get better visualisations. 

# In[ ]:


returns.index = pd.to_datetime(returns.index, format="%Y%m")
returns


# This looks a lot better. Now, we have monthly data for each small cap index and large cap index from the year 1926 to 2018. Pretty cool! However, the time stamp is given as first of the month however, these are the monthly averaged returns. So, we don't really need the date and we can just save the month-year representation. 

# In[ ]:


returns.index = returns.index.to_period('M')
returns


# In[ ]:


# Plot the data again 
returns.plot.line()


# Since the index is in time series format, we can do a lot of data manipulation using dates very easily. We can quickly extract the sections as well. 

# In[ ]:


# Get the data in the year 1960 
returns['1960']


# ## Compute Drawdowns 
# 
# Steps to compute the drawdowns. 
# 
# 1. **Compute the Wealth Index:** Value of a portfolio when it compounds over time over returns. 
# 2. **Compute the previous peak** 
# 3. **Compute drawdown** 
# 
# Let's start with LargeCap data.

# In[ ]:


# Compute the wealth index by starting with 1000 dollars
# The starting value won't matter with drawdowns

wealth_index = 1000*(1+returns['LargeCap']).cumprod()
wealth_index.head()


# In[ ]:


# Plot the wealth index over time 
wealth_index.plot.line()


# The cumulative effect is evident. 

# In[ ]:


# Compute the previous peaks 
previous_peaks = wealth_index.cummax()
previous_peaks.head()


# In[ ]:


# Plot the previous peaks
previous_peaks.plot.line()


# This is line is always moving upward. The points where we get a loss is where it flattens because we are only keep the cumulative max value. 

# In[ ]:


# Calculate the drawdown in percentage
drawdown = (wealth_index - previous_peaks)/previous_peaks


# In[ ]:


# Plot the drawdown 
drawdown.plot.line()


# This is very interesting. The crash of 1929 was much worse than 2000 or 2008 where people lost over 80% of their wealth. This was decimating. Let's look at some important values. 

# In[ ]:


drawdown.head()


# Drawdowns are calculating the negative returns and hence, they are between 0 and -1.

# In[ ]:


# Get the worst drawdown 
drawdown.min()


# In[ ]:


drawdown.idxmin()


# The worst drawdown was roughly 84% and it was on 1932-05.
# 
# As a time series, there are many operations that we can do on this data. 

# In[ ]:


# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {drawdown['1975':].min()} on {drawdown['1975':].idxmin()}")


# In[ ]:


# Get the worst drawdown in the 90s 
print(drawdown['1990':'1999'].min())
print(drawdown['1990':'1999'].idxmin())


# Finally, we need to make a combined plot of peaks wealth index to visualise drawdowns.

# In[ ]:


# Combine Plots 
wealth_index.plot.line()
previous_peaks.plot.line()


# In[ ]:


# Make a drawdown function
def compute_drawdown(return_series: pd.Series):
    '''
        ARGS: 
            Takes in a series of returns
            
        RETURNS:
            Wealth index
            Previous Peaks 
            Percent Drawdowns            
    '''
    
    # Calculate the wealth previous peaks and drawdowns
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    # Create a dataframe 
    drawdown_data = pd.DataFrame({'Wealth': wealth_index, 
                                  'Peaks': previous_peaks,
                                  'Drawdown': drawdowns})
    return drawdown_data


# In[ ]:


# Get the data for small cap stocks 
small_cap_drawdowns = compute_drawdown(returns['SmallCap'])
small_cap_drawdowns


# It's important to plot the wealth and peak together to understand where the biggest dropdowns occurred. 

# In[ ]:


# Lets plot the wealth and the peaks 
small_cap_drawdowns[['Wealth', 'Peaks']].plot.line()


# In[ ]:


small_cap_drawdowns['Drawdown'].plot.line()


# Small Caps generally show more losses and that goes in line with our knowledge of large and small cap stocks. 

# In[ ]:


small_cap_drawdowns['Drawdown'].min()


# In[ ]:


small_cap_drawdowns['Drawdown'].idxmin()


# The damage was also similar in 1932. Let's check for damages since 1975.

# In[ ]:


# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {small_cap_drawdowns['Drawdown']['1975':].min()} on {small_cap_drawdowns['Drawdown']['1975':].idxmin()}")


# In[ ]:


# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {drawdown['1975':].min()} on {drawdown['1975':].idxmin()}")


# So, the 2008 crash was clearly worse for small cap stocks. 

# In[ ]:




