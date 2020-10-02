#!/usr/bin/env python
# coding: utf-8

# # WTI Crude Oil Prediction - Complete solution
# What is the issue ?
# WTI is the West Texas Intermediate is the price of Crude oil drilled from USA. WTI and Brent which is used by OPEC countries is the standard to measure Crude Oil prices world over. The situation is now complicated with reduced demand and increased supply. Previously Crude oil used to be primarily sourced from OPEC and USA produced oil only for its consumption. Now with the advent of Shale drilling Technique USA is able to produce more oil than Saudi Arabia the primary produced of Crude. This has triggered an Economic war where OPEC, USA and Russia want to out produce each other. Shale producers are not having deep pockets like OPEC which is sitting on huge cash dump. But the added complication is Saudi depends on only oil to run its economy and has no other sources of income prolonged low prices could affect their economy greatly compared to Russia and USA which have other sources of Income. On the demand side too India and China the top importers have temporarily shutdown due to Corona and their pull also affects the Price.
# 
# 
# 

# # Download results
# 
# We are using Time-series modeling to predict the WTI by determing seasonality and flutuation.
# 
# To get the solution click on the "Copy & Edit" button on right hand corner and click Run All. The solution will be present in the /Kaggle/Working directory which you can download. In case you have any new ideas do share with me I will try to incorporate

# # We are downloading the packages to use in our code.
# Pandas - Data frame manipulation
# Matplotlib - Draw graphs
# statsmodels.api - to run the model

# In[ ]:


import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt


# In[ ]:


#Download the Crude Oil Trends
data = pd.read_csv("../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-15.csv", parse_dates=["Date"])
#Convert Date column to date format
data["Date"] = pd.to_datetime(data["Date"])
# Setting Index for efficient data access
data.set_index("Date", inplace=True)

#Create a List with Dates from 1-Apr to 21- Aug. We are predicting from 1-Apr to check accuracy
Date1 = pd.date_range('2020-04-01', periods=146, freq='D')

#Create frame Test2 with future dates.  
columns = ['Date','Price']    
Test2 = pd.DataFrame(columns=columns)
Test2['Price'] = pd.to_numeric(Test2['Price'])
Test2["Date"] = pd.to_datetime(Date1)
Test2 = Test2.fillna(0)
#Remove Weekends as in source data and store data frame as Test1 results to be loaded in Test 1
Test1 = Test2[Test2["Date"].dt.weekday < 5]
Test1["Date"] = pd.to_datetime(Test1["Date"])
Test1['Price'] = pd.to_numeric(Test1['Price'])
#Test1.set_index("Date", inplace=True)
#Test2 = Test2.fillna(0)
#print(Test1)
print(Test1.head(n=20))
print(Test1.tail(n=20))


# In[ ]:


## data75 = data.rolling(window=75).mean().dropna()
train = data[:"2020-04-30"]
test = data["2020-05-01":]


# In[ ]:


#Visualize Data in a line Graph notice the Deep drop in price from Mar-2020 due to Corona.
train["2017-01-01":].plot()
test.plot()
plt.show()


# In[ ]:


#View the Seasonality and Trend.
decomposition = sm.tsa.seasonal_decompose(train, model='addititve', period=7)
fig = decomposition.plot()
plt.show()


# In[ ]:


# Using Auto-Regression approach
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
model = AutoReg(train, lags=30, trend="c").fit()
yhat = model.predict(train.shape[0], train.shape[0]+test.shape[0]-1  )
res=pd.DataFrame({"Date":test.index,"Pred":yhat, "Act":test["Price"].values})
res.set_index("Date", inplace=True)
res["Act"].plot(label="Act")
res["Pred"].plot(label="Pred")
plt.title("Actual vs. Predicted")
plt.legend()
plt.show()
print("RMSE",mean_squared_error(res["2020-05-01":]["Act"], res["2020-05-01":]["Pred"]))


# In[ ]:


# Using Auto-Regression approach
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
# fit model
model = AutoReg(train, lags=30).fit()
#print(model.summary())
# make prediction for April 2020 and compare with Real data the difference is less than 1$
yhat = model.predict(train.shape[0], train.shape[0]+  )
res=pd.DataFrame({"Date":test["2020-04-01":"2020-04-30"].index,"Pred":yhat, "Act":test["2020-04-01":"2020-04-30"]["Price"].values})
#print(res)

#making Prediction from July to August 2020
yhat = model.predict(train.shape[0], train.shape[0]+146 )
yhat = pd.to_numeric(yhat)
#print(yhat.reset_index(drop=True))
#STore prediction in Test1
Test1["Price"] = yhat.reset_index(drop=True)
#print(Test1[Test1['Date']>"2020-07-15"])
#Removing date below 16-Jul-2020
Test2 = Test1[Test1['Date']>"2020-07-15"]
#Print Result in CSV file
Test2.to_csv('/kaggle/working/Submission_future.csv', index=False)
Test3 = Test1[Test1['Date']>"2020-04-28"]
Test3 = Test3[Test3['Date']<"2020-08-22"]
print(Test3)
Test3.to_csv('/kaggle/working/Submission_old.csv', index=False)
  



# In[ ]:





# In[ ]:




