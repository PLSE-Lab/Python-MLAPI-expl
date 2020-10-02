#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


training_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
test_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")


# In[ ]:


training_data.head()


# In[ ]:


training_data.info()


# In[ ]:


training_data.isnull().sum()


# In[ ]:


training_data["Country_Region"].unique()


# In[ ]:


test_data.info()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


training_data.drop(["Province_State","County","Id"],axis=1,inplace=True)
training_data.head()


# In[ ]:


test_data.drop(["Province_State","County","ForecastId"],axis=1,inplace=True)
test_data.head()


# In[ ]:


training_data.isnull().sum()


# In[ ]:


training_data.info()


# In[ ]:


training_data.describe()


# In[ ]:


training_data["Target"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


training_data.hist(figsize=(20,15))


# In[ ]:


sns.set(style="darkgrid")
sns.countplot(x="Target",data=training_data)


# In[ ]:


plt.pie(x=training_data.groupby(by=["Target"])["TargetValue"].sum(),labels=training_data["Target"].unique(),autopct='%1.1f%%')


# In[ ]:



plt.pie(x=training_data.groupby(by=["Country_Region"])["TargetValue"].sum(),labels=training_data["Country_Region"].unique(),autopct='%.0f%%',
           radius=3.0,wedgeprops = {'linewidth': 0.0,"edgecolor":"k"},pctdistance=0.8,labeldistance=1.5,textprops={"fontsize":20},shadow=True,
           startangle=-90,rotatelabels=True)
plt.show()


# In[ ]:


last_date=training_data.Date.max()
df=training_data[training_data["Date"]==last_date]
df


# In[ ]:





# In[ ]:


df=df.groupby(by=["Country_Region"],as_index=False)["TargetValue"].sum()
df


# In[ ]:


countries=df.nlargest(5,"TargetValue")
countries


# In[ ]:


cases=training_data.groupby(by=["Date","Country_Region"],as_index=False)["TargetValue"].sum()
cases


# In[ ]:


cases=cases.merge(countries,on="Country_Region")
cases


# In[ ]:


plt.figure(figsize=(15,10))
sns.set(style="darkgrid")
sns.lineplot(x="Date",y="TargetValue_x",hue="Country_Region",data=cases)


# In[ ]:


training_data.corr()


# In[ ]:


training_data.drop(["Target"],inplace=True,axis=1)
test_data.drop(["Target"],inplace=True,axis=1)
training_data


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


training_data["Country_Region"]=le.fit_transform(training_data["Country_Region"])


# In[ ]:


training_data.head()


# In[ ]:


test_data["Country_Region"]=le.fit_transform(test_data["Country_Region"])


# In[ ]:


test_data.head()


# In[ ]:


training_data.Date=training_data.Date.apply(lambda x:x.split("-"))


# In[ ]:


test_data.Date=test_data.Date.apply(lambda x:x.split("-"))


# In[ ]:


def month_day(dataset):
    month=[]
    day=[]
    for i in dataset.Date:
        month.append(int(i[1]))
        day.append(int(i[2]))
    dataset["month"]=month
    dataset["day"]=day
    dataset=dataset.drop(["Date"],axis=1)
    return dataset


# In[ ]:


training_data=month_day(training_data)
test_data=month_day(test_data)
training_data.head()


# In[ ]:


test_data.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[ ]:


y=training_data["TargetValue"].values


# In[ ]:


training_data.drop(["TargetValue"],axis=1,inplace=True)
training_data.head()


# In[ ]:


x=scaler.fit_transform(training_data)
x


# In[ ]:


from xgboost import XGBRegressor
xgb=XGBRegressor()


# In[ ]:


from sklearn.model_selection import cross_val_score
performance=cross_val_score(xgb,x,y,cv=10,scoring="neg_mean_absolute_error",n_jobs=-1)
mae=-performance


# In[ ]:


mae


# In[ ]:


mae.mean()


# In[ ]:


test_data=scaler.transform(test_data)
test_data


# In[ ]:


xgb.fit(x,y)
prediction_xgb=xgb.predict(test_data)
prediction_xgb=np.around(prediction_xgb)
prediction_xgb


# In[ ]:


xgb_1500=XGBRegressor(n_estimators=1500,max_depth=15)


# In[ ]:


xgb_1500.fit(x,y)


# In[ ]:


prediction=xgb_1500.predict(test_data)


# In[ ]:


prediction=np.around(prediction)
prediction


# In[ ]:


submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")
submission.head()


# In[ ]:


test_copy=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')


# In[ ]:


output = pd.DataFrame({'Id': test_copy.ForecastId  , 'TargetValue': prediction})
output.head()


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=0.8*a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=1.2*a['q0.95']


# In[ ]:





# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()


# In[ ]:


sub.info()

