#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot
import calendar
from sklearn.cluster import KMeans
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df1 =pd.read_csv('/kaggle/input/Petrol Prices.csv')
df1


# In[ ]:


df1.head(7)


# In[ ]:


df1.tail(4)


# In[ ]:


df1


# In[ ]:


#delete ug 18 data error
df1[df1["Date"]=="ug 18 2016"]

df1 = df1.drop(df1[df1.Date=="ug 18 2016"].index)


#convert to datetime

df1.insert(1,"date2", pd.to_datetime(df1["Date"]))

df1






# In[ ]:


x =df1['date2'].groupby([df1.date2.dt.month]).agg('count')




x.values


# In[ ]:


pltdf = pd.DataFrame({"Month": x.index, "Count": x.values})
pltdf


# In[ ]:


pltdf.plot(kind="bar",
    x='Month',
    y='Count',
    title="Total per Month",
    figsize=(12,8)
)
plt.ylabel("Count")
plt.xlabel("Month")


# In[ ]:


df1


# In[ ]:


day=df1["date2"].dt.day

df1.insert(3, "Day", day.values)



# In[ ]:


month=df1["date2"].dt.month

month.values


df1.insert(2, "Month", month.values)

df1


# In[ ]:




Year=df1["date2"].dt.year

Year.values


df1.insert(3, "Year", Year.values)


# In[ ]:


df1.replace({'Month' : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sept", 10 : "Oct", 11 : "Nov", 12 : "Dec",   }})


# In[ ]:


df1


# In[ ]:


#remaking index
dfIC=pd.DataFrame({"Gasolene_87": df1["Gasolene_87"] , "Gasolene_90": df1["Gasolene_90"], "Auto_Diesel": df1["Auto_Diesel"],"Kerosene": df1["Kerosene"], "Propane": df1["Propane"],"Butane": df1["Butane"],"HFO": df1["HFO"],"Asphalt": df1["Asphalt"],"ULSD": df1["ULSD"],"Ex_Refinery": df1["Ex_Refinery"], })
dfIC = dfIC.reset_index(drop=True)

dfIC


# In[ ]:


dfIC.plot(kind="line",
    title="Gas Prices over Period",
   
    figsize=(12,8)
)
plt.ylabel("Type")


# In[ ]:


HFO_pc= dfIC["HFO"].pct_change(periods=4)

HFO_pc


# In[ ]:


HFO_pc.plot(kind="line",
    title="HFO Percentage Change over Period[4]",
   
    figsize=(12,8)
)
plt.ylabel("Type")


# In[ ]:


df2=pd.DataFrame({"Day": df1["Day"], "Month": df1["Month"], "Year": df1["Year"], "Gasolene_87": df1["Gasolene_87"], "Gasolene_90": df1["Gasolene_90"] })

df2


# In[ ]:



#clustering


cluster_data = df2[['Gasolene_87','Gasolene_90']]
cluster_data.head()


# In[ ]:


#there was data missing

missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


cluster_data = cluster_data.fillna( df2.median() )

missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


# In[ ]:


data_values = cluster_data.iloc[ :, :].values
data_values


# In[ ]:


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


# Predicting the elbow at 3. Using 3 clusters

# In[ ]:


kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300) 
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data


# In[ ]:


cluster_data['cluster'].value_counts()


# In[ ]:


cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of gas')


# In[ ]:


kmeans = KMeans(n_clusters=3).fit(data_values)
centroids = kmeans.cluster_centers_
print(centroids)


#scatter plot nots showing  correct values


plt.scatter(data_values[:,0], data_values[:,1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)


# In[ ]:


sns.pairplot( cluster_data, hue="cluster")


# In[ ]:


grouped_cluster_data1 = cluster_data.groupby('cluster')
grouped_cluster_data1


# In[ ]:


grouped_cluster_data1.describe()


# In[ ]:


grouped_cluster_data2 = cluster_data.groupby(df2['Year'])
grouped_cluster_data2


# In[ ]:



grouped_cluster_data2.describe()


# In[ ]:


avg_price_year = df2.groupby('Year')['Gasolene_87','Gasolene_90'].agg(np.mean)
avg_price_year


# Autoregression
# 
# This time series forcasting method uses the observations of a prior time model to predict the next step in a sequence. 
# 
# In other words, it uses previous observations as an input into a regression analysis to predict the value of some factor at a subsequent time. 
# 
# 
# 
#                                             example
#                  an auto regresssion can be used to tell the value at time(+1) by using the results of the results                  at the last two time steps; time(-1) and time(-2)
#                  
#       regression formula
#                  
#   y= b0+ b1*z1 
#   
#  
#           predicted value of y at time(+1) using values of y at time(-1) and time(-2)
#   y(t+1)= b0 + b1* z(t-1) + b2 * z(t-2)
#   
#   
#   
