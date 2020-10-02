#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime,time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_path = '/kaggle/input/cacSurveys3May2016-withColumnNames.csv'
data = pd.read_csv(data_path)
data


# In[ ]:


data.dropna()


# In[ ]:


new_data = data.sample(n= 10000)
new_data


# In[ ]:


# Drop a row by condition


# In[ ]:


new = new_data.drop(columns=['Column 13'])


# In[ ]:


final = new.rename(columns={"Column 1":"ID"})
final


# In[ ]:


missing_data_results = new_data.isnull().sum()
print(missing_data_results)


# In[ ]:


data2 = new_data.fillna( data.median() )


# In[ ]:


data2


# In[ ]:


missing_data_results = data2.isnull().sum()
print(missing_data_results)


# In[ ]:


new = data2.drop(columns=['Column 13'])


# In[ ]:


final = new.rename(columns={"Column 1":"ID"})
final


# In[ ]:


date= pd.to_datetime(final['Date'],errors = 'coerce')
final['Date']=date 
#data1['Date'] = data1['Date'].astype('datetime64[ns]') 


# In[ ]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
final['Parish'] = label_encoder.fit_transform(final['Parish']) 
final['Town'] = label_encoder.fit_transform(final['Town']) 
final['Shop Type'] = label_encoder.fit_transform(final['Shop Type']) 
final['Good or Service Name'] =label_encoder.fit_transform(final['Good or Service Name'])


# In[ ]:


final


# In[ ]:


data = pd.DataFrame({
    'Parish':final['Parish'],
    'Prices':final['Price']
})
data


# In[ ]:


data.corr()


# In[ ]:


plt.scatter(final['Parish'],final['Price'])
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import linear_model
import statsmodels.api as sm


# In[ ]:


X_data = final[['Parish','Town']]
Y_data = final['Price'] 


# In[ ]:


#Splitting Data using a 70/30 split
X_train, X_test, y_train,y_test = train_test_split(X_data,Y_data, test_size = 0.30)


# In[ ]:


regres = linear_model.LinearRegression()
regres.fit(X_train,y_train)


# In[ ]:


regres.coef_


# In[ ]:


print("Regression Coefficients")
pd.DataFrame(regres.coef_,index= X_train.columns,columns=["Coefficient"])


# In[ ]:


regres.intercept_


# In[ ]:


test_predicted = regres.predict(X_test)
test_predicted


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))


# In[ ]:


# R squared
print('Variance score: %.2f' % r2_score(y_test, test_predicted))


# In[ ]:


cluster = final[['Good or Service Name','Parish']]
cluster.plot(kind='scatter',x='Good or Service Name',y='Parish')


# In[ ]:


final_values = cluster.iloc[:, :].values
final_values


# In[ ]:


from sklearn.cluster import KMeans
wcss =[]
for i in range (1, 15):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit_predict(final_values)
    wcss.append(kmeans.inertia_)
    plt.plot(wcss, 'ro-', label="WCSS")
    plt.title("Using Kmeans to Compare two Data")
    plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()
    


# In[ ]:


kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300)
cluster["cluster"] = kmeans.fit_predict(final_values)
cluster


# In[ ]:


from matplotlib.pyplot import plot
final.plot.scatter( x='Code', y='Price', title='Comparing Item codes to Prices ')
plot


# this shows that more items are sold at a lower cost and this can help to predict future sales

# In[ ]:


datas = final[['Good or Service Name','ID']]
df =pd.DataFrame(datas)
df


# In[ ]:


result = df.groupby('Good or Service Name').count()
result
x = result.rename(columns={"ID": "Output"})
x.head(87)


# In[ ]:


x.plot(kind="bar", figsize =(12,8))


# In[ ]:




