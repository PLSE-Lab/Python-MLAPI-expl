#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ins_df = pd.read_csv("../input/insurance.csv")


# In[ ]:


ins_df


# In[ ]:


df = ins_df


# In[ ]:


df


# In[ ]:


df.isna().sum()


# In[ ]:


df.index


# In[ ]:


df.head(10)


# In[ ]:


df.tail(15)


# In[ ]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df.describe().T


# In[ ]:


ins_corr=df.corr()
ins_corr


# In[ ]:


ins_cov=df.cov()
ins_cov


# In[ ]:


sns.heatmap(ins_corr,vmin=-1,vmax=1,center=0,annot=True)


# In[ ]:


sns.pairplot(data=df,hue='children')


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['children'],size=df['bmi'])


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['children'],size=df['age'])


# In[ ]:


sns.pairplot(data=df,hue='region')


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['region'],size=df['bmi'])


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['region'],size=df['age'])


# In[ ]:


sns.pairplot(data=df,hue='smoker')


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['smoker'],size=df['bmi'])


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['smoker'],size=df['age'])


# In[ ]:


sns.pairplot(data=df,hue='sex')


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['age'], y=df['expenses'],hue=df['sex'],size=df['bmi'])


# In[ ]:


plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['bmi'], y=df['expenses'],hue=df['sex'],size=df['age'])


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


# In[ ]:


cat_col=['smoker','region','sex']
num_col=[i for i in df.columns if i not in cat_col]
num_col


# In[ ]:


# one-hot encoding
one_hot=pd.get_dummies(df[cat_col])
ins_procsd_df=pd.concat([df[num_col],one_hot],axis=1)
ins_procsd_df.head(10)


# In[ ]:


#label encoding
ins_procsd_df_label=df
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in cat_col:
    ins_procsd_df_label[i] = label_encoder.fit_transform(ins_procsd_df_label[i])
ins_procsd_df_label.head(10)


# In[ ]:


#using one hot encoding
x=ins_procsd_df.drop(columns='expenses')
y=df[['expenses']]


# In[ ]:


x


# In[ ]:


y


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1234)


# In[ ]:


model = LinearRegression()

model.fit(train_x,train_y)


# In[ ]:


# Print Model intercept and co-efficent
print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)


# In[ ]:


cdf = pd.DataFrame(data=model.coef_.T, index=x.columns, columns=["Coefficients"])
cdf


# In[ ]:


# Print various metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("Predicting the train data")
train_predict = model.predict(train_x)
print("Predicting the test data")
test_predict = model.predict(test_x)
print("MAE")
print("Train : ",mean_absolute_error(train_y,train_predict))
print("Test  : ",mean_absolute_error(test_y,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y,train_predict))
print("Test  : ",mean_squared_error(test_y,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y,train_predict))
print("Test  : ",r2_score(test_y,test_predict))
print("MAPE")
print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)


# In[ ]:


#Plot actual vs predicted value
plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted expenses",fontsize=25)
plt.xlabel("Actual expenses",fontsize=18)
plt.ylabel("Predicted expenses", fontsize=18)
plt.scatter(x=test_y,y=test_predict)


# In[ ]:


#using label encoding
x1=ins_procsd_df.drop(columns='expenses')
y1=ins_procsd_df_label[['expenses']]


# In[ ]:


x1


# In[ ]:


y1


# In[ ]:


# split data into train and test
train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y1,test_size=0.3,random_state=1234)


# In[ ]:


# Create Linear regression model with train and test data
model = LinearRegression()

model.fit(train_x1,train_y1)


# In[ ]:


# Print Model intercept and co-efficent
print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)


# In[ ]:


cdf1 = pd.DataFrame(data=model.coef_.T, index=x1.columns, columns=["Coefficients"])
cdf1


# In[ ]:


# Print various metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print("Predicting the train data")
train_predict = model.predict(train_x1)
print("Predicting the test data")
test_predict = model.predict(test_x1)
print("MAE")
print("Train : ",mean_absolute_error(train_y1,train_predict))
print("Test  : ",mean_absolute_error(test_y1,test_predict))
print("====================================")
print("MSE")
print("Train : ",mean_squared_error(train_y1,train_predict))
print("Test  : ",mean_squared_error(test_y1,test_predict))
print("====================================")
import numpy as np
print("RMSE")
print("Train : ",np.sqrt(mean_squared_error(train_y1,train_predict)))
print("Test  : ",np.sqrt(mean_squared_error(test_y1,test_predict)))
print("====================================")
print("R^2")
print("Train : ",r2_score(train_y1,train_predict))
print("Test  : ",r2_score(test_y1,test_predict))
print("MAPE")
print("Train : ",np.mean(np.abs((train_y1 - train_predict) / train_y1)) * 100)
print("Test  : ",np.mean(np.abs((test_y1 - test_predict) / test_y1)) * 100)


# In[ ]:


#Plot actual vs predicted value
plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted expenses",fontsize=25)
plt.xlabel("Actual expenses",fontsize=18)
plt.ylabel("Predicted expenses", fontsize=18)
plt.scatter(x=test_y1,y=test_predict)


# In[ ]:


print("MAPE")
print("Train : ",np.mean(np.abs((train_y1 - train_predict) / train_y1)) * 100)
print("Test  : ",np.mean(np.abs((test_y1 - test_predict) / test_y1)) * 100)


# In[ ]:




