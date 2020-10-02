#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
sample_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Target'].value_counts()


# In[ ]:


test['Target'].value_counts()


# In[ ]:


test = test.rename(columns = {'ForecastId' : 'Id'})


# In[ ]:


print(test.shape)
print(sample_submission.shape)


# In[ ]:


sample_submission.head()


# In[ ]:


train.head(1)
train = train.drop(columns = ['County' , 'Province_State'])
test = test.drop(columns = ['County' , 'Province_State'])


# In[ ]:


train.isnull().sum()


# In[ ]:


test.head(1)


# In[ ]:


test.isnull().sum()


# In[ ]:


train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = train.iloc[:,1].values
train.iloc[:,1] = labelencoder.fit_transform(X.astype(str))

X = train.iloc[:,5].values
train.iloc[:,5] = labelencoder.fit_transform(X)





# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = test.iloc[:,1].values
test.iloc[:,1] = labelencoder.fit_transform(X)

X = test.iloc[:,5].values
test.iloc[:,5] = labelencoder.fit_transform(X)



# In[ ]:


train.Date = pd.to_datetime(train.Date).dt.strftime("%Y%m%d").astype(int)
test.Date = pd.to_datetime(test.Date).dt.strftime("%Y%m%d").astype(int)
test.head()


# In[ ]:


x = train.iloc[:,1:6]
y = train.iloc[:,6]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2, random_state = 0 )


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline_dt = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', RandomForestRegressor())])
pipeline_dt.fit(x_train , y_train)
prediction = pipeline_dt.predict(x_test)


# In[ ]:


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


# In[ ]:


prediction_list = [int(x) for x in predictor]


# In[ ]:


prediction_list


# In[ ]:


sub = pd.DataFrame({'Id': test.index , 'TargetValue': prediction_list})


# In[ ]:


sub['TargetValue'].value_counts()


# In[ ]:


p=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
r=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


p.columns = ['Id' , 'q0.05']
q.columns = ['Id' , 'q0.5']
r.columns = ['Id' , 'q0.95']


# In[ ]:


p = pd.concat([p,q['q0.5'] , r['q0.95']],1)


# In[ ]:


p['q0.05']=p['q0.05'].clip(0,10000)
p['q0.05']=p['q0.5'].clip(0,10000)
p['q0.05']=p['q0.95'].clip(0,10000)
p


# In[ ]:


p['Id'] =p['Id']+ 1
p


# In[ ]:


#Taken Help from Kernel at https://www.kaggle.com/nischaydnk/covid19-week5-visuals-randomforestregressor#Submission
sub=pd.melt(p, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub


# In[ ]:


sub


# In[ ]:




