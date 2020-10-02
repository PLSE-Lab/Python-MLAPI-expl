#!/usr/bin/env python
# coding: utf-8

# # **Data preparation**

# In[ ]:


#import data file
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


# ### Data Preparation_Train Data

# In[ ]:


df_train.head()


# In[ ]:


print(f"Unique Countries: {(df_train.Country_Region.nunique())}")
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")
#we have 184 unique cuntries
#we have 115 days from 22/01/2020 to 15/05/2020


# In[ ]:


df_train.shape[0]
#we have 35,995 observations


# In[ ]:


print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")
#WE will use Panel regression when Unit: Unique_Region 
#Time variable: Days
#This is balanced panel


# In[ ]:


df_train.Country_Region.value_counts()
#shows that some countries have more than one region such as US, China and Canada


# In[ ]:


print(f"Number of rows without Country_Region : {df_train.Country_Region.isna().sum()}")
#no N/A for Country_region var


# In[ ]:


#creat new column/var = Unique_Region
df_train["Unique_Region"]=df_train.Country_Region


# In[ ]:


df_train 


# In[ ]:


df_train[df_train.Province_State.isna()==False]
#The countries that have more than one region 


# In[ ]:


#If country has more than one region >> change Unique_Region to be "Province_State.Country_region"
#If country has only one region or NaN >> Unique_Region is same as "Country_region"
df_train.Unique_Region[df_train.Province_State.isna()==False]=df_train.Province_State+" . "+df_train.Country_Region


# In[ ]:


df_train.sample(5)


# In[ ]:


#Drop 3columns >> Id, Province_State and Country_region
df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)


# In[ ]:


df_train
#35995rows * 4columns


# ### Data Preparation_Test Data

# In[ ]:


df_test.head()


# In[ ]:


print(f"Period :{df_test.Date.nunique()} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")
#this data set has 43days from 02/04/2020 to 14/05/2020


# In[ ]:


print(f"Total Regions : {df_test.shape[0]/43}")
#have 313 units as Training Data


# Total regions in test is same as train data

# In[ ]:


#creat new column/var = Unique_Region
df_test["Unique_Region"]=df_test.Country_Region
#If country has more than one region >> change Unique_Region to be "Province_State.Country_region"
#If country has only one region or NaN >> Unique_Region is same as "Country_region"
df_test.Unique_Region[df_test.Province_State.isna()==False]=df_test.Province_State+" . "+df_test.Country_Region


# In[ ]:


df_test.sample(5)


# In[ ]:


df_test.drop(labels=["Province_State","Country_Region"], axis=1, inplace=True)


# In[ ]:


df_test


# In[ ]:


len(df_test.Unique_Region.unique())
#we have 313 units


# ### Data Preparation_Submission data

# In[ ]:


df_sub
#13459rows * 3columns


# In[ ]:


train_dates=list(df_train.Date.unique())
test_dates=list(df_test.Date.unique())


# In[ ]:


# Dates in train only = 72days
only_train_dates=set(train_dates)-set(test_dates)
print("Only train dates : ",len(only_train_dates))
#dates in train and test = 43days
intersection_dates=set(test_dates)&set(train_dates)
print("Intersection dates : ",len(intersection_dates))
#dates in only test = 0days
only_test_dates=set(test_dates)-set(train_dates)
print("Only Test dates : ",len(only_test_dates))
#we will use 72days to predict 43days


# ## Data Preparation_Train data >> 71 days

# In[ ]:


#duplicate df_train2
df_train2 = pd.DataFrame()
df_train2['Date'] = df_train.Date
df_train2['ConfirmedCases'] = df_train.ConfirmedCases
df_train2['Fatalities'] = df_train.Fatalities
df_train2['Unique_Region'] = df_train.Unique_Region


# In[ ]:


#drop from 115days to 71days
df_train2.drop(df_train2[df_train2.Date>df_test.Date.min()]. index, inplace=True)
df_train2.drop(df_train2[df_train2.Date == df_test.Date.min()]. index, inplace=True)


# In[ ]:


df_train2.sample(5)
#22223rows * 5columns


# In[ ]:


print(f"Unique_Region : {df_train2.Unique_Region.nunique()} units")
print(f"Period : {df_train2.Date.nunique()} days")
print(f"From : {df_train2.Date.min()} To : {df_train2.Date.max()}")


# # Panel regression

# ## Predict Cases using Panel regression (Fixed effects)
# We decided to use Panel regression since our data is recorded both cross section and time series features and we use Fixed effect bc from Hausman test, P-value less than 0.05. So fixed effect is better than random effect
# 
# When cross section is Unique_region = 313 units and time variable is time = 70 days From : 2020-01-23 To : 2020-04-01 (drop day1 since we dont have lag term data) >> using file df_train2
# 
# After we get the panel regression equation we will use the equation to predict number of cases for 43 days From : 2020-04-02 To : 2020-05-14 >> using file df_train3
# * y = today's cases
# * x = yesterdays's cases
# 
# code turtorials;
# https://stackoverflow.com/questions/49067495/fixed-effects-model-using-python-linearmodels
# https://stackoverflow.com/questions/24195432/fixed-effect-in-pandas-or-statsmodels/44836199#44836199
# https://stackoverflow.com/questions/50863691/pandas-convert-date-object-to-int

# In[ ]:


#create y, x, fatalities, fatalities lag, Unique_Region and time varaibles
df_panel = pd.DataFrame()
df_panel['y'] = df_train2.ConfirmedCases
df_panel['x'] = df_panel.y.shift(1)
df_panel['Fatalities'] = df_train2.Fatalities
df_panel['Fata_lag'] = df_panel.Fatalities.shift(1)
df_panel['Unique_Region'] = df_train2.Unique_Region
df_panel['time'] = df_train2.Date


# In[ ]:


#Drop day1 of each region since we dont have value of x value
df_panel.drop(df_panel[df_panel.time=='2020-01-22']. index, inplace=True)


# In[ ]:


#convert datetime to interger
df_panel['time'] = pd.to_numeric(df_panel.time.str.replace('-',''))
print(df_panel['time'])


# In[ ]:


df_panel


# In[ ]:


#create data as panel data
df_panel = df_panel.set_index(['Unique_Region','time'])


# In[ ]:


df_panel


# In[ ]:


get_ipython().system('pip install  linearmodels')


# In[ ]:


#run panel regression with fixed effects
from linearmodels.panel import PanelOLS
mod = PanelOLS(df_panel.y, df_panel.x, entity_effects=True)
# mod = PanelOLS.from_formula('y ~ x + EntityEffects', df_panel)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)
#coeff of x is 1.069


# ## Predict Fatalities using Linear regression
# our equation is
# fatalities today = B1 * (fatalities yesterday) + B2 * (Cases' yesterday)
# 

# In[ ]:


mod = PanelOLS.from_formula('Fatalities ~ Fata_lag + x + EntityEffects', df_panel)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)
#we got coeff for fatalities' yesterday and cases' yesterday equal to 1.0771 and 0.0014  respectively


# # Predict using dynamic forecast

# ## Predict cases

# In[ ]:


#duplicate df_train3
df_train3 = pd.DataFrame()
df_train3['Date'] = df_train.Date
df_train3['ConfirmedCases'] = df_train.ConfirmedCases
df_train3['Fatalities'] = df_train.Fatalities
df_train3['Unique_Region'] = df_train.Unique_Region


# In[ ]:


#drop from 115days to 44days include 2020-04-01
df_train3.drop(df_train3[df_train3.Date<'2020-04-01']. index, inplace=True)
df_train3.drop(df_train3[df_train3.Date=='2020-05-15']. index, inplace=True)
#create lag var for predict data
df_train3['Fata_lag'] = df_train3.Fatalities.shift(1)
df_train3['x'] = df_train3.ConfirmedCases.shift(1)
#drop from 44days to 43days 
df_train3.drop(df_train3[df_train3.Date<'2020-04-02']. index, inplace=True)


# In[ ]:


df_train3


# In[ ]:


#predict cases (y_cases) for day1 of each cross sectional : 2020-04-02
#using previous actual case for day1
df_train3['y_cases'] = round(1.0690*(df_train3.x))


# In[ ]:


df_train3.head(50)


# In[ ]:


df_train3.tail(45)


# In[ ]:


#predict cases (y_cases) for other 42days of each cross sectional : 2020-04-03 until 2020-05-14
#dynamic forecast
#End python code after 5 seconds
import time
import threading

def listen():
    for i in range(1,df_train3.shape[0]):
        df_train3.loc[df_train3['Date'] != '2020-04-02', 'x'] = df_train3.y_cases.shift(1)
        df_train3['y_cases'] = round(1.0690*(df_train3.x))

t = threading.Thread(target=listen)
t.daemon = True
t.start()

time.sleep(5)


# In[ ]:


df_train3.head(50)


# In[ ]:


df_train3.tail(45)


# ## Predict Fatality 
# 

# In[ ]:


#predict fatality rate (predictrate) for day1 of each cross sectional : 2020-04-02
#using previous actual rate and previous actual case for day 1
df_train3['y_fata'] = round(1.0771*(df_train3.Fata_lag) + 0.0014*(df_train3.x))


# In[ ]:


df_train3


# In[ ]:


#predict fatality rate for other 42days of each cross sectional : 2020-04-03 until 2020-05-14
#dynamic forecast
#End python code after 5 seconds
import time
import threading

def listen():
    for i in range(1,df_train3.shape[0]):
        df_train3.loc[df_train3['Date'] != '2020-04-02', 'Fata_lag'] = df_train3.y_fata.shift(1)
        df_train3['y_fata'] = round(1.0771*(df_train3.Fata_lag) + 0.0014*(df_train3.x))

t = threading.Thread(target=listen)
t.daemon = True
t.start()

time.sleep(5)


# In[ ]:


df_train3


# In[ ]:


final_df=pd.DataFrame(columns=["y_cases","y_fata"])
final_df=pd.concat([final_df,df_train3], ignore_index=True)


# # Submission
# use y_cases and y_fata from df_train3

# In[ ]:


df_sub


# In[ ]:


#replace data
df_sub.ConfirmedCases=final_df.y_cases
df_sub.Fatalities=final_df.y_fata


# In[ ]:


df_sub


# In[ ]:


df_sub.to_csv("submission.csv", index=None)

