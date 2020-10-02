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


#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

#Machine Learning
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
test.head()


# Understanding and Cleaning the Data

# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# Data Discription for String Columns

# In[ ]:


train.describe(include=[np.object])


# # Handeling Null Values

# In[ ]:


train.isna().any()


# In[ ]:


train.isna().sum()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Droping of Unwanted Data to make model more Predictive

# In[ ]:


for df in [train,test]:
    df.drop("County",axis=1,inplace=True)
    df.drop("Province_State",axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Data Visualization

# # Performing Correlation Matrix for Train Data

# In[ ]:


corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# 

# In[ ]:


fig = px.pie(train, values='TargetValue', names='Target', title='ConfirmedCases & Fatalities')
fig.show()


# In[ ]:


fig = px.pie(train, values='TargetValue', names='Country_Region', title='ConfirmedCases & Fatalities Percentile by Country')
fig.update_traces(textposition='inside')
fig.show()


# # Preparing data for Model

# Converting String Date into Integer for both Train and Test Datasets

# In[ ]:


train["Date"] = pd.to_datetime(train["Date"]).dt.strftime("%Y%m%d")


# In[ ]:


test["Date"] = pd.to_datetime(test["Date"]).dt.strftime("%Y%m%d")


# Appling Label Encoding for Categorial features  

# In[ ]:


from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
train['Country_Region']= le.fit_transform(train['Country_Region']) 
train['Target']= le.fit_transform(train['Target']) 
test['Country_Region']= le.fit_transform(test['Country_Region']) 
test['Target']= le.fit_transform(test['Target']) 


# In[ ]:


train.tail()


# In[ ]:


test.head()


# Slipting Data based on Predictors and Target values

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state = 0)


# Fitting the model RandomForestRegressor 

# In[ ]:


model = RandomForestRegressor(n_jobs=-1)
scores = []
model.set_params(n_estimators=200)
model.fit(X_train, y_train)
scores.append(model.score(X_test, y_test))
score = model.score(X_test, y_test)
print(round(score*100,2))


# In[ ]:


test.drop(['ForecastId'],axis=1,inplace=True)
test.index.name = 'Id'
test.head()


# # Prediction

# In[ ]:


y_pred = model.predict(X_test)
y_pred


# # Output

# In[ ]:


predictions = model.predict(test)

pred_list = [int(x) for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)


# # Preparing Submission File

# In[ ]:


q05 = output.groupby('Id')['TargetValue'].quantile(q=0.05).reset_index()
q50 = output.groupby('Id')['TargetValue'].quantile(q=0.5).reset_index()
q95 = output.groupby('Id')['TargetValue'].quantile(q=0.95).reset_index()

q05.columns=['Id','0.05']
q50.columns=['Id','0.5']
q95.columns=['Id','0.95']


# In[ ]:


concatDF = pd.concat([q05,q50['0.5'],q95['0.95']],1)
concatDF['Id'] = concatDF['Id'] + 1
concatDF.head(10)


# # Submission

# In[ ]:


sub1 = pd.melt(concatDF, id_vars=['Id'], value_vars=['0.05','0.5','0.95'])
sub1['ForecastId_Quantile']=sub1['Id'].astype(str)+'_'+sub1['variable']
sub1['TargetValue']=sub1['value']
sub1=sub1[['ForecastId_Quantile','TargetValue']]
sub1.reset_index(drop=True,inplace=True)
sub1.to_csv("submission1.csv",index=False)
sub1.head(10)


# In[ ]:


# part 2


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV, KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


train = train.drop(['County','Province_State','Country_Region','Target'],axis=1)
test = test.drop(['County','Province_State','Country_Region','Target'],axis=1)
train


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# In[ ]:


def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]
test_date_min = test['Date'].min()
test_date_max = test['Date'].max()
def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]
def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day
train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])
test['Date']=test['Date'].dt.strftime("%Y%m%d").astype(int)
train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)
from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
test = test.drop(['ForecastId'], axis=1)
model = XGBRegressor(n_estimators = 1000 , random_state = 0)
model.fit(X_train, y_train)

scores = []

scores.append(model.score(X_test, y_test))

y_pred2 = model.predict(X_test)
y_pred2

predictions = model.predict(test)

pred_list = [x for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']
a


# In[ ]:


a['Id'] =a['Id']+ 1
a


# In[ ]:


sub2=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub2['variable']=sub2['variable'].str.replace("q","", regex=False)
sub2['ForecastId_Quantile']=sub2['Id'].astype(str)+'_'+sub2['variable']
sub2['TargetValue']=sub2['value']
sub2=sub2[['ForecastId_Quantile','TargetValue']]
sub2.reset_index(drop=True,inplace=True)
sub2.to_csv("submission2.csv",index=False)
sub2.head()


# In[ ]:


## part 3


# Simple Covid19 Random Forest
# Acknowledgements
# https://www.kaggle.com/nischaydnk/covid19-week5-visuals-randomforestregressor

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
mms = MinMaxScaler()
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


ufeatures = pd.read_csv("../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv")


# In[ ]:


#Data Preprocessing

train=pd.merge(train,ufeatures,on=["Country_Region"],how="left")
test=pd.merge(test,ufeatures,on=["Country_Region"],how="left")
train.columns


# In[ ]:


train = train.drop(['County','Province_State','Country_Region',"Population","Country_Code"],axis=1)
test = test.drop(['County','Province_State','Country_Region',"Population","Country_Code"],axis=1)
train.Lockdown_Type = train.Lockdown_Date.astype("category")
train.Lockdown_Type = train.Lockdown_Type.cat.codes

test.Lockdown_Type = test.Lockdown_Date.astype("category")
test.Lockdown_Type = test.Lockdown_Type.cat.codes
train.Target = train.Target.astype("category")
train.Target = train.Target.cat.codes

test.Target = test.Target.astype("category")
test.Target = test.Target.cat.codes
train.columns


# In[ ]:


train.Date_FirstFatality.fillna("2222-02-02",inplace=True)
test.Date_FirstFatality.fillna("2222-02-02",inplace=True)

train.Date_FirstConfirmedCase.fillna("2222-02-02",inplace=True)
test.Date_FirstConfirmedCase.fillna("2222-02-02",inplace=True)
train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])

train['Date_FirstConfirmedCase']=pd.to_datetime(train['Date_FirstConfirmedCase'])
test['Date_FirstConfirmedCase']=pd.to_datetime(test['Date_FirstConfirmedCase'])

train['Date_FirstFatality']=pd.to_datetime(train['Date_FirstFatality'])
test['Date_FirstFatality']=pd.to_datetime(test['Date_FirstFatality'])
train.isnull().sum()


# In[ ]:


test['Date']=test['Date'].dt.strftime("%Y%m%d").astype(int)
train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)

test['Date_FirstConfirmedCase']=test['Date_FirstConfirmedCase'].dt.strftime("%Y%m%d").astype(int)
train['Date_FirstConfirmedCase']=train['Date_FirstConfirmedCase'].dt.strftime("%Y%m%d").astype(int)


test['Date_FirstFatality']=test['Date_FirstFatality'].dt.strftime("%Y%m%d").astype(int)
train['Date_FirstFatality']=train['Date_FirstFatality'].dt.strftime("%Y%m%d").astype(int)
train.Lockdown_Date.fillna("2222-02-02",inplace=True)
test.Lockdown_Date.fillna("2222-02-02",inplace=True)
train['Lockdown_Date']=pd.to_datetime(train['Lockdown_Date'])
test['Lockdown_Date']=pd.to_datetime(test['Lockdown_Date'])
test['Lockdown_Date']=test['Lockdown_Date'].dt.strftime("%Y%m%d").astype(int)
train['Lockdown_Date']=train['Lockdown_Date'].dt.strftime("%Y%m%d").astype(int)
test_date_min = test['Date'].min()
test_date_max = test['Date'].max()
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
train.columns


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id',"Lockdown_Date","Lockdown_Type","Target"], axis=1)
target = train["TargetValue"]

test.drop(['ForecastId',"Lockdown_Date","Lockdown_Type","Target"],axis=1,inplace=True)
test.index.name = 'Id'
Id=test.index
Id= test.index


# In[ ]:


#Random Forest

model = RandomForestRegressor(n_jobs=-1,random_state=141,n_estimators=100) # ?

model.fit(predictors, target)


# In[ ]:


def feature_imp(df,model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

feature_imp(predictors,model).plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)


# In[ ]:


predictions_rf = model.predict(test)


# In[ ]:


#Prediction


# In[ ]:



pred_list = [int(x) for x in predictions_rf]

output = pd.DataFrame({'Id': Id, 'TargetValue': pred_list})   

a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()

a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a['Id'] =a['Id']+ 1

sub3=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub3['variable']=sub3['variable'].str.replace("q","", regex=False)
sub3['ForecastId_Quantile']=sub3['Id'].astype(str)+'_'+sub3['variable']
sub3['TargetValue']=sub3['value']
sub3=sub3[['ForecastId_Quantile','TargetValue']]
sub3['TargetValue']=sub3['TargetValue'].astype(int)
sub3.reset_index(drop=True,inplace=True)
sub3.to_csv("submission3.csv",index=False)


# In[ ]:


#Quantile_sub(predictions_rf,"submission3")


# In[ ]:


# part 4


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
sample_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
test = test.rename(columns = {'ForecastId' : 'Id'})
train = train.drop(columns = ['County' , 'Province_State'])
test = test.drop(columns = ['County' , 'Province_State'])
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = train.iloc[:,1].values
train.iloc[:,1] = labelencoder.fit_transform(X.astype(str))

X = train.iloc[:,5].values
train.iloc[:,5] = labelencoder.fit_transform(X)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = test.iloc[:,1].values
test.iloc[:,1] = labelencoder.fit_transform(X)

X = test.iloc[:,5].values
test.iloc[:,5] = labelencoder.fit_transform(X)
train.Date = pd.to_datetime(train.Date).dt.strftime("%Y%m%d").astype(int)
test.Date = pd.to_datetime(test.Date).dt.strftime("%Y%m%d").astype(int)
test.head()


# In[ ]:


x = train.iloc[:,1:6]
y = train.iloc[:,6]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.001, random_state = 0 )
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline_dt = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', RandomForestRegressor())])
pipeline_dt.fit(x_train , y_train)
prediction = pipeline_dt.predict(x_test)
score = pipeline_dt.score(x_test,y_test)
print('Score: ' + str(score))


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(prediction,y_test)
print(val_mae)


# In[ ]:


X_test = test.iloc[:,1:6]
predictor = pipeline_dt.predict(X_test)
prediction_list = [x for x in predictor]
sub = pd.DataFrame({'Id': test.index , 'TargetValue': prediction_list})
sub['TargetValue'].value_counts()


# In[ ]:


p=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
r=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
p.columns = ['Id' , 'q0.05']
q.columns = ['Id' , 'q0.5']
r.columns = ['Id' , 'q0.95']
p = pd.concat([p,q['q0.5'] , r['q0.95']],1)
p['q0.05']=p['q0.05'].clip(0,10000)
p['q0.05']=p['q0.5'].clip(0,10000)
p['q0.05']=p['q0.95'].clip(0,10000)
p


# In[ ]:


p['Id'] =p['Id']+ 1
p


# In[ ]:


sub4=pd.melt(p, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub4['variable']=sub4['variable'].str.replace("q","", regex=False)


# In[ ]:


sub4['ForecastId_Quantile']=sub4['Id'].astype(str)+'_'+sub4['variable']
sub4['TargetValue']=sub4['value']
sub4=sub4[['ForecastId_Quantile','TargetValue']]
sub4.reset_index(drop=True,inplace=True)
sub4.to_csv("submission4.csv",index=False)
sub4


# In[ ]:


# part 5


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
training_data["Country_Region"]=le.fit_transform(training_data["Country_Region"])
training_data.head()


# In[ ]:


test_data["Country_Region"]=le.fit_transform(test_data["Country_Region"])
test_data.head()


# In[ ]:


training_data.Date=training_data.Date.apply(lambda x:x.split("-"))
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
y=training_data["TargetValue"].values
training_data.drop(["TargetValue"],axis=1,inplace=True)
training_data.head()


# In[ ]:


x=scaler.fit_transform(training_data)
x


# In[ ]:


from xgboost import XGBRegressor
xgb=XGBRegressor()
from sklearn.model_selection import cross_val_score
performance=cross_val_score(xgb,x,y,cv=10,scoring="neg_mean_absolute_error",n_jobs=-1)
mae=-performance
mae


# In[ ]:


mae.mean()
test_data=scaler.transform(test_data)
test_data


# In[ ]:


xgb.fit(x,y)
prediction_xgb=xgb.predict(test_data)
prediction_xgb=np.around(prediction_xgb)
prediction_xgb


# In[ ]:


xgb_1500=XGBRegressor(n_estimators=1500,max_depth=15)
xgb_1500.fit(x,y)


# In[ ]:


prediction=xgb_1500.predict(test_data)
prediction=np.around(prediction)
prediction


# In[ ]:


submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")
submission.head()


# In[ ]:


test_copy=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
output = pd.DataFrame({'Id': test_copy.ForecastId  , 'TargetValue': prediction})
output.head()


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']


# In[ ]:


sub5=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub5['variable']=sub5['variable'].str.replace("q","", regex=False)
sub5['ForecastId_Quantile']=sub5['Id'].astype(str)+'_'+sub5['variable']
sub5['TargetValue']=sub5['value']
sub5=sub5[['ForecastId_Quantile','TargetValue']]
sub5.reset_index(drop=True,inplace=True)
sub5.to_csv("submission5.csv",index=False)
sub5.head()


# In[ ]:


sub.info()


# In[ ]:


# end blend


# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=(0.3*sub1['TargetValue']+0.1*sub2['TargetValue']+0.1*sub3['TargetValue']+3.5*sub4['TargetValue']+1.0*sub5['TargetValue'])/5

sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()

