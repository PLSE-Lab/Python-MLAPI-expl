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


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime


# In[ ]:


from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
Xtrain = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
Xtest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")


# In[ ]:


Xtrain


# In[ ]:


Xtest


# In[ ]:


df


# In[ ]:


#let's check the number of NaN values
Xtrain.Province_State.isnull().sum()


# GETTING ALL THE NAMES OF COUNTRY PRESENT IN DATA SET

# In[ ]:


#breaking data in countries with lesser cases and with countries with higher cases
region_list = list(Xtrain.Country_Region.unique())
region_list


# # CATEGORIZING THE COUNTRIES ACCOREDING TO CASES

# In[ ]:



countries_500=[]
countries_500_cases = []
for i in region_list:
    x= Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if (max_cases <500.0) & (max_cases > 250.0):
        countries_500.append(i)
        countries_500_cases.append(max_cases)
#print(countries_500)
#print(countries_500_cases)
df500 = pd.DataFrame({'country':countries_500})
df500_cases = pd.DataFrame({'cases': countries_500_cases})  
df500 = pd.concat([df500,df500_cases],axis=1)
#new_index = df500.countrie_500_cases.sort_values(ascending= True).index.values#
#sortes_data = df500.reindex(new_index)   

plt.figure(2,figsize=(15,7))
sns.barplot(df500.country,df500.cases,palette='Blues')
x=plt.xticks(rotation = 80)
x=plt.ylim(150,550)
plt.title('Countries with less than 500 cases')


# In[ ]:


countries_1k=[]
countries_1k_cases = []
for i in region_list:
    x= Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if (max_cases <1000.0) & (max_cases >500.0):
        countries_1k.append(i)
        countries_1k_cases.append(max_cases)
    
df_1k = pd.DataFrame({'country':countries_1k})
df_1k_cases = pd.DataFrame({'cases': countries_1k_cases})  
df_1k = pd.concat([df_1k,df_1k_cases],axis=1)
new_index = df_1k.cases.sort_values(ascending=True).index.values
df_1k = df_1k.reindex(new_index)
plt.figure(2,figsize=(15,7))
sns.barplot(df_1k.country,df_1k.cases,palette='BuGn')
x=plt.xticks(rotation = 80)
x=plt.ylim(400,1050)
plt.title('Countries with less than 1k cases')


# In[ ]:


#countries with 5k corona cirus cases
countries_5k=[]
countries_5k_cases=[]
for i in region_list:
    x=Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if  (max_cases >1000.0) & (max_cases<5000.0):
        countries_5k.append(i)
        countries_5k_cases.append(max_cases)
df_5k = pd.DataFrame({'country':countries_5k})
df_5k_cases = pd.DataFrame({'cases': countries_5k_cases})  
df_5k = pd.concat([df_5k,df_5k_cases],axis=1)
new_index = df_5k.cases.sort_values(ascending=True).index.values
df_5k = df_5k.reindex(new_index)
plt.figure(2,figsize=(15,11))
sns.barplot(df_5k.country,df_5k.cases,palette='PRGn_r')
x=plt.xticks(rotation = 80)
#x=plt.ylim(700,5100)
plt.title(' Countries having 5k  cases')


# In[ ]:


countries_10k=[]
countries_10k_cases=[]
for i in region_list:
    x=Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if  (max_cases >5000.0) & (max_cases<10000.0):
        countries_10k.append(i)
        countries_10k_cases.append(max_cases)
df_10k = pd.DataFrame({'country':countries_10k})
df_10k_cases = pd.DataFrame({'cases': countries_10k_cases})  
df_10k = pd.concat([df_10k,df_10k_cases],axis=1)
new_index = df_10k.cases.sort_values(ascending=True).index.values
df_10k = df_10k.reindex(new_index)
plt.figure(2,figsize=(15,7))
sns.barplot(df_10k.country,df_10k.cases,palette='Purples')
x=plt.xticks(rotation = 80)
#x=plt.ylim(3050,10500)
plt.title("countriees upto 10k cases")


# In[ ]:


countries_35k=[]
countries_35k_cases=[]
for i in region_list:
    x=Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if  (max_cases >10000.0) & (max_cases<35000.0):
        countries_35k.append(i)
        countries_35k_cases.append(max_cases)
df_35k = pd.DataFrame({'country':countries_35k})
df_35k_cases = pd.DataFrame({'cases': countries_35k_cases})  
df_35k = pd.concat([df_35k,df_35k_cases],axis=1)
new_index = df_35k.cases.sort_values(ascending=True).index.values
df_35k = df_35k.reindex(new_index)
plt.figure(2,figsize=(15,7))
sns.barplot(df_35k.country,df_35k.cases,palette='PuBuGn')
x=plt.xticks(rotation = 80)
#x=plt.ylim(9000,25000)
plt.title('Countries upto 35k Cases ----In  Danger')


# In[ ]:


countries_k=[]
countries_k_cases=[]
for i in region_list:
    x=Xtrain.ConfirmedCases[Xtrain.Country_Region == i]
    max_cases= max(x)
    if  (max_cases >35000.0 ) & (max_cases < 150000.0):
        countries_k.append(i)
        countries_k_cases.append(max_cases)
df_k = pd.DataFrame({'country':countries_k})
df_k_cases = pd.DataFrame({'cases': countries_k_cases})  
df_k = pd.concat([df_k,df_k_cases],axis=1)
new_index = df_k.cases.sort_values(ascending=True).index.values
df_k = df_k.reindex(new_index)
plt.figure(2,figsize=(15,7))
sns.barplot(df_k.country,df_k.cases,palette='winter_r')
x=plt.xticks(rotation = 80)
#x=plt.ylim(20000,270000)
plt.title('Countries on 3rd phase')


# In[ ]:


f_count= []
for i in df_35k.country:
    x= Xtrain.Fatalities[Xtrain.Country_Region == i ]
    count = max(x)
    f_count.append(count)
df_35k.insert(2,"Fatilities",f_count,True)    


# In[ ]:


f_count= []
for i in df_k.country:
    x= Xtrain.Fatalities[Xtrain.Country_Region == i ]
    count = max(x)
    f_count.append(count)
df_k.insert(2,"Fatilities",f_count,True)    
df_k


# In[ ]:


plt.figure(1,figsize=(15,9))
sns.jointplot(x='Fatilities',y='cases',kind='kde',data=df_35k)


# In[ ]:


sns.jointplot(x='Fatilities',y='cases',kind='kde',data=df_k)


# In[ ]:


plt.figure(1,figsize=(12,7))
sns.pointplot(x='cases', y='Fatilities', data=df_35k,hue='country',join=False,palette='gist_rainbow')
ax=plt.xticks(rotation=90)
plt.title('Fatilities on Confirmed Cases')


# In[ ]:


plt.figure(1,figsize=(11,7))
sns.pointplot(x='cases', y='Fatilities', data=df_k,hue='country',join=True)
plt.title('Fatilities on Confirmed Cases')


# In[ ]:


danger =[]
for i in region_list:
    x= Xtrain[Xtrain.Country_Region == i]
    max_cases = x.ConfirmedCases.max()
    if  max_cases>150000:
        danger.append(x)
df_France= pd.DataFrame(danger[0]) 
df_Italy= pd.DataFrame(danger[2])
df_Spain =pd.DataFrame(danger[3])
df_US = pd.DataFrame(danger[4])
df_France.reset_index()
df_Italy.reset_index()
df_Spain.reset_index()
df_US.reset_index()


# In[ ]:


dates= [datetime.strptime(ts, "%Y-%m-%d") for ts in df_France.Date]
dates =[datetime.strftime(ts,"%Y-%m-%d") for ts in dates]
df_F= pd.DataFrame({"Date":dates})
df_F['year'],df_F['month'],df_F['day']=df_F['Date'].str.split('-').str


# In[ ]:


df_France=pd.merge(df_France,df_F)
dates= [datetime.strptime(ts, "%Y-%m-%d") for ts in df_Italy.Date]
dates =[datetime.strftime(ts,"%Y-%m-%d") for ts in dates]
df_F= pd.DataFrame({"Date":dates})
df_F['year'],df_F['month'],df_F['day']=df_F['Date'].str.split('-').str


# In[ ]:


df_Italy=pd.merge(df_F,df_Italy)
dates= [datetime.strptime(ts, "%Y-%m-%d") for ts in df_Spain.Date]
dates =[datetime.strftime(ts,"%Y-%m-%d") for ts in dates]
df_F= pd.DataFrame({"Date":dates})
df_F['year'],df_F['month'],df_F['day']=df_F['Date'].str.split('-').str
df_Spain=pd.merge(df_F,df_Spain)


# In[ ]:


dates= [datetime.strptime(ts, "%Y-%m-%d") for ts in df_US.Date]
dates =[datetime.strftime(ts,"%Y-%m-%d") for ts in dates]
df_F= pd.DataFrame({"Date":dates})
df_F['year'],df_F['month'],df_F['day']=df_F['Date'].str.split('-').str
df_US=pd.merge(df_US,df_F)


# In[ ]:


df_France['Fatalities'].plot(kind='kde',logy=True)
df_Spain['Fatalities'].plot(kind='kde',logy=True)
df_Italy['Fatalities'].plot(kind='kde',logy=True)
df_US['Fatalities'].plot(kind='kde',logy=True)
plt.title('Logarithmic rate of Fatalities of countries at high risk')


# In[ ]:


sns.distplot(a=df_Spain["Fatalities"],label=True,kde=True,color='blue',bins=3)
plt.title('Fatality rate in Spain')


# In[ ]:


sns.distplot(a=df_Italy["Fatalities"],label=True,kde=True,color='purple',bins=3)
plt.title('Fatality rate in Italy')


# In[ ]:


plt.figure(4,figsize=(19,10))
plt.subplot(2,2,1)
sns.regplot(x=df_US.Fatalities,y=df_US.ConfirmedCases,fit_reg=True,units=df_US.month)
plt.title('US')
plt.subplot(2,2,2)
sns.regplot(x=df_France.Fatalities,y=df_France.ConfirmedCases,fit_reg=True)
plt.title("France")
plt.subplot(2,2,3)
sns.regplot(x=df_Spain.Fatalities,y=df_Spain.ConfirmedCases,fit_reg=True)
plt.title("Spain")
plt.subplot(2,2,4)
sns.regplot(x=df_Italy.Fatalities,y=df_Italy.ConfirmedCases,fit_reg=True)
plt.title("Italy")
plt.subplots_adjust(hspace=.8,wspace=.8)
plt.show()


# In[ ]:


df_US


# In[ ]:


data_us=df_US.groupby(['Province_State']).std()


# In[ ]:


data_us.Fatalities.plot()
plt.figure(1,figsize=(19,10))
a=plt.xticks(rotation=80)


# In[ ]:


data=df_US.groupby(['Province_State','month','day']).max()
data=data.drop(['Id'],axis=1)
data.plot()

plt.figure(1,figsize=(19,10))
a=plt.xticks(rotation=80)


# In[ ]:


sns.lmplot(x='ConfirmedCases',y='Fatalities',data=data_us)


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train.Province_State.fillna('State',inplace=True)
train.head(5)


# In[ ]:



test.Province_State.fillna('State',inplace=True)
test


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
import os
import warnings


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model1= RandomForestRegressor(random_state=999)


# In[ ]:


train["Date"] = pd.to_datetime(train["Date"]).dt.strftime("%Y%m%d")


# In[ ]:


test["Date"] = pd.to_datetime(test["Date"]).dt.strftime("%Y%m%d")


# In[ ]:


series_col=['Country_Region','Province_State']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[ ]:


Xtrain=train.copy()
Xtest = test.copy()
for col in series_col:
    Xtrain[col]=encoder.fit_transform(train[col])
    Xtest[col]=encoder.transform(test[col])


# In[ ]:


Xtrain.Province_State.unique()


# In[ ]:


features=['Date','Province_State','Country_Region']
cases = ['ConfirmedCases']


# In[ ]:


model1.fit(Xtrain[features],Xtrain[cases])
pred1=model1.predict(Xtest[features])


# In[ ]:


confirmed_case=[]
for i in pred1:
    c=int(i)
    confirmed_case.append(c)


# In[ ]:


model1=RandomForestRegressor(random_state=999)


# In[ ]:



features=['Date','Province_State','Country_Region']
deaths=['Fatalities']

model1.fit(Xtrain[features],Xtrain[deaths])

prediction= model1.predict(Xtest[features])


# In[ ]:


deaths=[]
for i in prediction:
    d=int(i)
    deaths.append(d)


# In[ ]:


submission = pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':confirmed_case,'Fatalities':deaths})


# In[ ]:


submission


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




