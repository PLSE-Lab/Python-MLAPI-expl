#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")


# In[ ]:


train_df.head()


# **Lets extract data about India out and plot it.**

# In[ ]:


india = train_df[train_df["Country_Region"]=="India"]


# In[ ]:


india.head()


# **It started slow but as you can see it has started to increase rapidly and its not a good sign.**

# In[ ]:


india.tail()


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(india["ConfirmedCases"])
plt.xlabel("Time")
plt.ylabel("Number of Confirmed Cases")


# **Both the plot looks pretty same. For a minute I thought that I have done some mistake but not. **

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(india["Fatalities"])
plt.xlabel("Time")
plt.ylabel("Number of Fatalities")


# **I was checking how many states/province are there in china**

# In[ ]:


china_states = train_df[train_df["Country_Region"]=="China"]["Province_State"].unique()


# In[ ]:


train_df["Province_State"].unique()


# **As there are many countries for which the province/state column is nan so i thought to put the country name if it is nan.**

# In[ ]:


def province(state,country):
    if state == "nan":
        return country
    return state


# **I filled all the nan value with a string nan this string could be anything.**

# In[ ]:


train_df = train_df.fillna("nan")


# In[ ]:


train_df["Province_State"] = train_df.apply(lambda x: province(x["Province_State"],x["Country_Region"]),axis=1)


# In[ ]:


train_df


# In[ ]:


china_states


# **Italy is struggling right now so i thought to so see whats going on.**

# In[ ]:


italy = train_df[train_df["Country_Region"]=="Italy"]


# **The condition has not improved a lot its still increasing and thats not good.**

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(italy["ConfirmedCases"])
plt.xlabel("Time")
plt.ylabel("Number of Confirmed Cases")


# **Again you see both the plot looks the same.**

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(italy["Fatalities"])
plt.xlabel("Time")
plt.ylabel("Number of Fatalities Cases")


# # **One Thing I have noticed that the curve of a countries confirmed cases and fatalities looks pretty same And it make sense if you think, if the confirmed cases fluctuates deaths would also increase or decrease. **

# In[ ]:


from datetime import datetime


# In[ ]:


train_df.info()


# **Converting the date column to datetime datatype so that we can extract day and month from it**

# In[ ]:


train_df["Date"] = pd.to_datetime(train_df["Date"])


# In[ ]:


train_df["month"] = train_df["Date"].dt.month


# In[ ]:


train_df.head()


# **Extracting the day of the month**

# In[ ]:


train_df['day'] = train_df['Date'].dt.day


# In[ ]:


train_df.tail()


# **No need fot the date column now, so dropping it**

# In[ ]:


train_df.drop('Date',axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


from sklearn import preprocessing


# # I thought of using the country and state column as a variable but the results were not good so I excluded them. So labelencoder is not required and but I kept it.(you can remove this as this will not effect the result).

# In[ ]:


def labelencoder(data):
    le = preprocessing.LabelEncoder()
    new_data = le.fit_transform(data)
    return new_data


# In[ ]:


train_df["Country_Region"] = labelencoder(train_df["Country_Region"].values)
train_df["Province_State"] = labelencoder(train_df["Province_State"].values)


# In[ ]:


train_df.head()


# # Here I did all the steps for the test data as I did with train.

# In[ ]:


test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")


# In[ ]:


test_df["Date"] = pd.to_datetime(test_df["Date"])


# In[ ]:


test_df['day'] = test_df['Date'].dt.day


# In[ ]:


test_df["month"] = test_df["Date"].dt.month


# In[ ]:


test_df.drop('Date',axis=1,inplace=True)
test_df.head()


# In[ ]:


test_df.fillna("nan",inplace=True)
test_df["Province_State"] = test_df.apply(lambda x: province(x["Province_State"],x["Country_Region"]),axis=1)


# In[ ]:


test_df.head()


# In[ ]:


test_df["Country_Region"] = labelencoder(test_df["Country_Region"].values)
test_df["Province_State"] = labelencoder(test_df["Province_State"].values)


# In[ ]:


test_df.head()


# In[ ]:


countries = train_df["Country_Region"].unique()


# # This is the important part.
# 
# # Using a linear regressor wont work well as you can see the curve is not really linear.
# # Using just Linear regression will just coincide with some of the data.
# # Here is a link to a good post about it https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386)

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


poly_reg_cc = PolynomialFeatures(degree = 4)
poly_reg_ft = PolynomialFeatures(degree = 4)

reg_cc = LinearRegression()
reg_ft = LinearRegression()

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = train_df[train_df["Country_Region"]==country]["Province_State"].unique()
    for state in states:
        train_df_filt = train_df[(train_df["Country_Region"]==country)&(train_df["Province_State"]==state)]
        y_train_cc = train_df_filt["ConfirmedCases"].values
        y_train_ft = train_df_filt["Fatalities"].values
        
        
        X_train = train_df_filt[["month","day"]]
    
        
        test_df_filt = test_df[(test_df["Country_Region"]==country)&(test_df["Province_State"]==state)]
        X_test = test_df_filt.drop('ForecastId',axis=1)
        X_test = X_test[["month","day"]]
        test_Id = test_df_filt["ForecastId"].values
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        scaler_cc = MinMaxScaler()
        scaler_ft = MinMaxScaler()
        
        y_train_cc=y_train_cc.reshape(-1,1)
        y_train_ft=y_train_ft.reshape(-1,1)
        
        y_train_cc=scaler_cc.fit_transform(y_train_cc)
        y_train_ft=scaler_ft.fit_transform(y_train_ft)
        
        y_train_cc = y_train_cc.flatten()
        y_train_ft = y_train_ft.flatten()
        
        X_train_poly = poly_reg_cc.fit_transform(X_train)
        reg_cc.fit(X_train_poly,y_train_cc)
        X_test_poly = poly_reg_cc.fit_transform(X_test)
        test_cc = reg_cc.predict(X_test_poly)
        
        test_cc = test_cc.reshape(-1,1)
        test_cc = scaler_cc.inverse_transform(test_cc)
        test_cc = test_cc.flatten()
        
        X_train_poly = poly_reg_ft.fit_transform(X_train)
        reg_ft.fit(X_train_poly,y_train_ft)
        X_test_poly = poly_reg_ft.fit_transform(X_test)
        test_ft = reg_ft.predict(X_test_poly)
        
        test_ft = test_ft.reshape(-1,1)
        
        test_ft = scaler_ft.inverse_transform(test_ft)
        test_ft = test_ft.flatten()
        
        df = pd.DataFrame({'ForecastId': test_Id, 'ConfirmedCases': test_cc, 'Fatalities': test_ft})
        
        df_out = pd.concat([df_out, df], axis=0)


# In[ ]:


df_out[:20]


# In[ ]:


df_out.head()


# In[ ]:


df_out["Fatalities"] = df_out["Fatalities"].apply(int)
df_out["ConfirmedCases"] = df_out["ConfirmedCases"].apply(int)
df_out[:10]


# In[ ]:


df_out["ForecastId"] = df_out["ForecastId"].astype('int32')
df_out["Fatalities"] = df_out["Fatalities"].astype('int32')
df_out["ConfirmedCases"] = df_out["ConfirmedCases"].astype('int32')
df_out.info()
df_out.to_csv("submission.csv",index=False)
sub = pd.read_csv("submission.csv")
sub[:20]


# # This is my first kernel explanation, I am not that good at explaining but the code is pretty clear in it self. I hope you are safe and fit.
# # Please upvote if you think it helped
# 

# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

train["Province_State"] = train["Province_State"].fillna('')
test["Province_State"] = test["Province_State"].fillna('')

train["Month"], train["Day"] = 0, 0
for i in range(len(train)):
    train["Month"][i] = (train["Date"][i]).split("-")[1]
    train["Day"][i] = (train["Date"][i]).split("-")[2]
    
test["Month"], test["Day"] = 0, 0
for i in range(len(test)):
    test["Month"][i] = (test["Date"][i]).split("-")[1]
    test["Day"][i] = (test["Date"][i]).split("-")[2]


# In[ ]:


for i in range(len(train)):
    if train["Province_State"][i] != '':
        train["Country_Region"][i] = str(train["Province_State"][i]) + " (" + str(train["Country_Region"][i]) + ")"
       
for i in range(len(test)):
    if test["Province_State"][i] != '':
        test["Country_Region"][i] = str(test["Province_State"][i]) + " (" + str(test["Country_Region"][i]) + ")"
        
train.drop(columns = "Province_State", inplace=True)
test.drop(columns = "Province_State", inplace=True)

train.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)
test.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)


# In[ ]:


train.tail()


# In[ ]:


i = 0
for value in train["Country/State"].unique():
    if i < len(train):
        j = 1
        while(train["Country/State"][i] == value):
            train["Day"][i] = j
            j += 1; i += 1
            if i == len(train):
                break

i = 0
for value in test["Country/State"].unique():
    if i < len(test):
        j = 72
        while(test["Country/State"][i] == value):
            test["Day"][i] = j
            j += 1; i += 1
            if i == len(test):
                break


# In[ ]:


train = train.drop(columns = ["Date"])
test = test.drop(columns = ["Date"])


# In[ ]:


countries = train["Country/State"].unique()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg_cc = PolynomialFeatures(degree = 4)
poly_reg_ft = PolynomialFeatures(degree = 4)

from sklearn.linear_model import LinearRegression
reg_cc = LinearRegression()
reg_ft = LinearRegression()

from sklearn.preprocessing import StandardScaler

sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for value in countries:
    train_temp = train.loc[train["Country/State"] == value]
    test_temp = test.loc[test["Country/State"] == value]
    train_temp_cc = train_temp["ConfirmedCases"].loc[train["Country/State"] == value].to_frame()
    train_temp_ft = train_temp["Fatalities"].loc[train["Country/State"] == value].to_frame()
    
    train_temp_X = train_temp.iloc[:, 4:6]
    test_temp_X = test_temp.iloc[:, 2:4]
    sc1 = StandardScaler()
    train_temp_X = sc1.fit_transform(train_temp_X)
    test_temp_X = sc1.transform(test_temp_X)
    
    sc_cc = StandardScaler()
    sc_ft = StandardScaler()
    train_temp_cc = sc_cc.fit_transform(train_temp_cc)
    train_temp_ft = sc_ft.fit_transform(train_temp_ft)
    
    X_poly = poly_reg_cc.fit_transform(train_temp_X)
    reg_cc.fit(X_poly, train_temp_cc)
    test_cc = sc_cc.inverse_transform(reg_cc.predict(poly_reg_cc.fit_transform(test_temp_X)))
    
    X_poly = poly_reg_ft.fit_transform(train_temp_X)
    reg_ft.fit(X_poly, train_temp_ft)
    test_ft = sc_ft.inverse_transform(reg_ft.predict(poly_reg_ft.fit_transform(test_temp_X)))
    
    a = int(train["Day"].loc[train["Country/State"] == "India"].max())
    b = int(a - test_temp["Day"].min())
    
    test_cc[0:b+1] = sc_cc.inverse_transform(train_temp_cc)[(a-b-1):(a)]
    test_ft[0:b+1] = sc_ft.inverse_transform(train_temp_ft)[(a-b-1):(a)]
    
    test_cc = test_cc.flatten()
    test_ft = test_ft.flatten()
    sub_temp = pd.DataFrame({'ForecastId': test_temp["ForecastId"].loc[test["Country/State"] == value],
                             'ConfirmedCases': test_cc, 'Fatalities': test_ft})
    sub = pd.concat([sub, sub_temp], axis = 0)


# In[ ]:


sub["ForecastId"] = sub["ForecastId"].astype('int32')


# In[ ]:


sub[:20]


# In[ ]:


sub.to_csv("submission.csv", index = False)


# In[ ]:




