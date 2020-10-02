#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
prices = pd.read_csv("../input/dwdm-petrol-prices/Petrol Prices.csv")


# In[ ]:


df = pd.DataFrame(prices)
df.head(7)


# In[ ]:


df.tail(4)


# In[ ]:


import matplotlib.pyplot as plt
data3 = df.copy()
data3["Date"]= data3["Date"].str.split(" ", expand = True)
data3['Date'].value_counts().plot(kind='bar' , title=" Records for each month",
           figsize=(12,8))
plt.ylabel("Fequency")
plt.xlabel("Months")


# In[ ]:


#y=[]
#data4.at[143,'Date']= 'Aug 18 2016'
#s = data4['Date']
#for i in range(229):
 #  l =time.mktime(datetime.datetime.strptime(s[i],"%b %d %Y").timetuple())
  #  y.append(l)
#for i in range(3):    
 #   y.append('NaN')
#print(y)


# In[ ]:


data4 = df.copy()

# new data frame with split value columns 
new = data4["Date"].str.split(" ",n = 3, expand = True) 
  
# making separate first name column from new data frame 
data4["Month"]= new[0] 
 
# making separate last name column from new data frame 
data4["Day"]= new[1] 
data4["Year"]= new[2]
#data4['TimeStamp'] = y

   
 
# df display 
data4


# In[ ]:


data4 = data4.fillna(0)
data4


# In[ ]:


data4['Day']= data4['Day'].apply(int)
data4['Year'] = data4['Year'].apply(int)

data4.dtypes


# In[ ]:


#use if lamba to change nan to 0 or change to int


# In[ ]:



import datetime
y=[]
length = len(data4)
date_text = data4['Date']
for i in range(length):
    if (date_text[i] == 0 or date_text[i] =='ug 18 2016'): 
        y.append(0)
    else: 
        l = datetime.datetime.strptime(date_text[i],"%b %d %Y")
        y.append(l)

data4['TimeStamp'] = y
data4


# In[ ]:


data4.replace(0, np.nan ,inplace=True)
data4['TimeStamp']=pd.to_datetime(data4['Date'], errors='coerce')
data4.dtypes


# In[ ]:


data2 = data4[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','TimeStamp']]
data2


# In[ ]:


data2.reindex(columns=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','TimeStamp'])
#lines = data2.plot.line()


# In[ ]:



data2.plot(kind="line", # or `us_gdp.plot.line(`
    x='TimeStamp',     
    y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'],
     
    title="Gas Prices per Period",
    figsize=(25,20)
)
#plt.title("From %d to %d" % (
 #   data2['TimeStamp'].min(),
  #data2['TimeStamp'].max()
#),size=8)
plt.suptitle("Gas Prices per Period",size=12)
plt.ylabel("Gas Prices")


# In[ ]:


data2['Propane'].pct_change(periods= 4,fill_method='ffill')


# In[ ]:


data2['Propane'].pct_change(periods= 4,fill_method='ffill').plot( title="percentage change for every 4 time periods",
           figsize=(25,8))
plt.ylabel("Fequency of change")
plt.xlabel("Number of Change")


# In[ ]:


kdata= data4[['Gasolene_87', 'Gasolene_90', 'Month','Day','Year','TimeStamp']]
kdata


# In[ ]:


cluster_data = kdata[['Gasolene_87', 'Gasolene_90']]


cluster_data = cluster_data.fillna( kdata.median() )
#Get rid of missing data
missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)


# In[ ]:


data_values = cluster_data.iloc[ :, :].values
data_values

from sklearn.cluster import KMeans

# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)
wcss = []
for i in range( 1, 15 ):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit_predict( data_values )
    wcss.append( kmeans.inertia_ )
    
plt.plot( wcss, 'ro-', label="WCSS")
plt.title("Computing WCSS for KMeans++")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data


# In[ ]:


cluster_data['cluster'].value_counts()


# In[ ]:


cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Gas Prices across groups')
plt.xlabel("Clusters")
plt.ylabel("Frequency")


# In[ ]:


import seaborn as sns
sns.pairplot( cluster_data, hue="cluster")


# In[ ]:


grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data


# In[ ]:


grouped_cluster_data.describe()


# In[ ]:



grouped_cluster_data.plot(subplots=True,)


# In[ ]:


#Average per year
kdata.groupby('Year')['Gasolene_87','Gasolene_90'].mean()


# In[ ]:





# 2. What can you say about each cluster?
#    for each cluster , Gasolene 90 is always higher than Gasolene 87

# 3. Can you justify your process (i.e. related to Task 12)?
# 
#  The kmeans was used to find if groups exist amoung the gas prices and predict what groups future prices belong too

# 4. Name ONE (1) time series forecasting method and explain how it works (be sure to include a citation)
# 
# The autoregression (AR) method models the next step in the sequence as a linear function of the observations at prior time steps.
# 
# The notation for the model involves specifying the order of the model p as a parameter to the AR function, e.g. AR(p). For example, AR(1) is a first-order autoregression model.
# 
# The method is suitable for univariate time series without trend and seasonal components.
# 
# **For example:**
# 
# **yhat = b0 + b1*X1**
# 
# Where yhat is the prediction, b0 and b1 are coefficients found by optimizing the model on training data, and X is an input value.
# 
# This technique can be used on time series where input variables are taken as observations at previous time steps, called lag variables.
# 
# For example, we can predict the value for the next time step (t+1) given the observations at the last two time steps (t-1 and t-2). As a regression model, this would look as follows:
# 
# 
# **X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)**
# 
# Because the regression model uses data from the same input variable at previous time steps, it is referred to as an autoregression (regression of self).
# 
# https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
# https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/

# In[ ]:




