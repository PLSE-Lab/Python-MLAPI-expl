#!/usr/bin/env python
# coding: utf-8

# As a value investor and typically a non-believer in technical analysis, I decided to play around with some techniques to predict Nifty (Indian stock market index) over the next year. **The biggest challenge in using Machine learning techniques has been that past is not a reliable predictor of future **(at least I believe so). Even if we cannot predict exact values, we can try to predict probabilities for different range. 
# 
# **Combination of old statistical technique (Monte carlo simulation) and state of the art timeseries algorithm (Facebook fbprophet)**. Let's use Monte Carlo simulation to predict Nifty range- probabilistic values over the next year and then use Facebook open source library fbprophet.

# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy import stats


# We will use last 1 year of Nifty values for our prediction mode. Easily available for public on NSE website. Many people might prefer to use longer term data- however based on my experience with changing macro trends, 1 year is a good period. However, folks who prefer to fork this notebook can try on longer periods also.

# In[ ]:


nifty=pd.read_csv("../input//Nifty_data.csv")
nifty.head()


# Use this dataset to calculate the mean and standard deviation of the index levels. Some people prefer to call standard deviation as volatility also especially in corporate finance.

# In[ ]:


nifty_returns=(nifty['Close']/nifty['Open'])-1

volatility= np.std(nifty_returns)
trading_days=len(nifty_returns)
mean=(nifty.loc[trading_days-1,'Close']/nifty.loc[0,'Open'])-1

print('Annual Average Nifty return',mean)
print('Annual volatility',volatility*np.sqrt(trading_days))
print('Number of trading days',trading_days)


# Let's create a normal distribution with mean of 12.8% and average volatility (or standard deviation) of 9.2%. The below code will just be a 1 random distribution.

# In[ ]:


daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

index_returns=[10980]  
                               
for x in daily_returns:
    index_returns.append(index_returns[-1]*x)

plt.plot(index_returns)
plt.show()


# Let's run the random distributions for 1000 times.

# In[ ]:


for i in range(1000):
    daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

    index_returns=[10980]  
    
    for x in daily_returns:
        index_returns.append(index_returns[-1]*x)

    plt.plot(index_returns)

plt.show()


# With current Nifty levels of 11,278- our random normal distributions give a wide range from downside of 9000 to an upside of 16,000 (ignoring the 17,000 outlier distribution). For derivative traders, this is important that possible downside is very low from crystal ball analysis (or monte carlo simulation).

# In[ ]:


index_result=[]

for i in range(1000):
    daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

    index_returns=[10980]  
    
    for x in daily_returns:
        index_returns.append(index_returns[-1]*x)
 
    index_result.append(index_returns[-1])

plt.hist(index_result)
plt.show()


# In[ ]:


print('Average expected value of Nifty:',np.mean(index_result))
print('10 percentile:',np.percentile(index_result,10))
print('90 percentile:',np.percentile(index_result,90))


# From Monte carlo simulation, Nifty is expected to be hovering around **12500 levels **over the next year with range between **11,000 to 14,000 levels** (from 10 to 90 percentile, ignoring the 10 percentile outliers).

# Now let us use Facebook fbprophet library to predict Nifty levels and see how it compares to the standard monte carlo simulation. Use the same Nifty data file as above- index levels for the past 1 year.

# In[ ]:


from fbprophet import Prophet


# In[ ]:


nifty.head()


# In[ ]:


nifty=nifty.iloc[:,0:2]
nifty.head()


# Since we need only timestamp and prediction value, let's use only date and open column for timeseries prediction.
# Prophet API expects date to be in the format of pandas- YYYY-MM-DD. Convert the datetime format in file to the required format. Also, Prophet expects columns to be named as ds and y.

# In[ ]:


nifty['Date']= pd.to_datetime(nifty['Date'])
nifty.rename(columns={'Date':'ds','Open':'y'},inplace=True)


# Create an object of Prophet class and use the fit and predict methods (similar to sklearn API format).

# In[ ]:


model=Prophet()
model.fit(nifty)

predict_df=model.make_future_dataframe(periods=252)
predict_df.tail()


# In[ ]:


forecast=model.predict(predict_df)
forecast.tail()


# In[ ]:


fig1=model.plot(forecast)


# In[ ]:


fig2=model.plot_components(forecast)


# Using **Prophet library, Nifty levels are expected to reach around 12,100 levels while Monte Carlo simulation suggested mean of 12,600 levels.** 
# 
# Prophet is extrapolating last 1 year data- which has not been great for market returns. Prophet is suggesting muted market returns because of last 1 year trend while monte carlo is simulating normal distributions with given mean and high volatility (9.2%). 
# 
# **Prophet is taking last 1 year performance too seriously.** 
# 
# However, none of the models are aware that in the next 1 year, Indian general elections are due and results of elections might well make all this analysis redundant. 
# 
