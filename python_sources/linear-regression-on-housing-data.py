#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataframe =  pd.read_csv("../input/MELBOURNE_HOUSE_PRICES_LESS.csv")


# In[ ]:


dataframe.dtypes


# In[ ]:


dataframe.head()


# In[ ]:


dataframe['Type'].unique()


# In[ ]:


dataframe["Date"] = pd.to_datetime(dataframe["Date"],dayfirst=True)


# In[ ]:


len(dataframe["Date"].unique())/4


# In[ ]:


var = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").std()
count = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").count()
mean = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()


# In[ ]:


mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))


# In[ ]:


means = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").mean()
errors = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].sort_values("Date", ascending=False).groupby("Date").std()


# In[ ]:


dataframe.columns


# In[ ]:


means.columns


# In[ ]:


means.head()


# In[ ]:


errors.head()


# In[ ]:


means.drop(["Price",
            "Postcode",
           "Distance", "Propertycount"],axis=1).plot(yerr=errors)


# In[ ]:


dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()


# In[ ]:


pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
dataframe[(dataframe["Type"]=="h") & 
          (dataframe["Distance"]<14) &
          (dataframe["Distance"]>13.7) 
          #&(dataframe["Suburb"] =="Northcote")
         ].sort_values("Date", ascending=False).dropna().groupby(["Suburb","SellerG"]).mean()


# In[ ]:


sns.kdeplot(dataframe[(dataframe["Suburb"]=="Northcote")
         & (dataframe["Type"]=="u")
         & (dataframe["Rooms"] == 2)]["Price"])


# In[ ]:


plt.figure(figsize=(20,15))
my_axis = sns.kdeplot(dataframe["Price"][((dataframe["Type"]=="u") &
                                (dataframe["Distance"]>8) &
                                (dataframe["Distance"]<10) &
                                (dataframe["Rooms"] > 2)#&
                                #(dataframe["Price"] < 1000000)
                               )])
my_axis.axis(xmin=0, xmax=2000000)


# In[ ]:


sns.lmplot("Distance","Price",dataframe[(dataframe["Rooms"]<=4) & 
                                         (dataframe["Rooms"]> 2) & 
                                        (dataframe["Type"]=="h") &
                                        (dataframe["Price"]< 1000000)
                                       ].dropna(),hue="Rooms", size=10)


# In[ ]:


dataframe.columns


# In[ ]:


dataframe[(dataframe["Rooms"]>2) & (dataframe["Type"] == "h")].dropna().groupby("Distance").mean().plot()


# In[ ]:


sns.pairplot(dataframe.dropna())


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(dataframe[dataframe["Type"] == "h"].corr(), annot=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


dataframe_dr = dataframe.dropna().sort_values("Date")


# In[ ]:


from datetime import date


# In[ ]:


all_Data = []


# In[ ]:


###########
##Find out days since start
days_since_start = [(x - dataframe_dr["Date"].min()).days for x in dataframe_dr["Date"]]


# In[ ]:


dataframe_dr["Days"] = days_since_start


# In[ ]:


#suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Type", "Method"]])
suburb_dummies = pd.get_dummies(dataframe_dr[["Type", "Method"]])
#suburb_dummies = pd.get_dummies(dataframe_dr[[ "Type"]])
#suburb_dummies = pd.get_dummies(dataframe_dr[["Suburb", "Method"]])


# In[ ]:


all_Data = dataframe_dr.drop(["Address","Price","Date", "SellerG","Suburb","Type","Method","CouncilArea","Regionname"],axis=1).join(suburb_dummies)


# In[ ]:


X = all_Data


# In[ ]:


y = dataframe_dr["Price"]


# In[ ]:


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
X.columns


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)
ranked_suburbs


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test, predictions)
plt.ylim([200000,1000000])
plt.xlim([200000,1000000])


# In[ ]:


sns.distplot((y_test-predictions),bins=50)


# In[ ]:


from sklearn import metrics


# In[ ]:


print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




