#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn as sk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") #ignores all warnings


# In[ ]:


data = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
data=data[["MinTemp","MaxTemp","Rainfall","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"]]

data["RainToday"].replace("Yes",1,inplace=True)
data["RainToday"].replace("No",0,inplace=True)

data["RainTomorrow"].replace("Yes",1,inplace=True)
data["RainTomorrow"].replace("No",0,inplace=True)

plt.matshow(data.corr(),cmap='gray')
plt.xticks(range(data.shape[1]),data.columns,rotation=90)
plt.yticks(range(data.shape[1]),data.columns)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()


# In[ ]:


model = GaussianNB()


# In[ ]:


data.dropna(inplace=True)
properties = data[["MinTemp","MaxTemp","Rainfall","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]]
label = data[["RainTomorrow"]]


# In[ ]:


#properties["RainToday"]=properties["RainToday"].apply(lambda x:0 if x == "No" else 1)
#label["RainTomorrow"]=label["RainTomorrow"].apply(lambda x:0 if x == "No" else 1)
#properties.replace([-np.inf,np.inf],np.nan)
# properties["RainToday"].replace("Yes",1,inplace=True)
# properties["RainToday"].replace("No",0,inplace=True)
# label["RainTomorrow"].replace("Yes",1,inplace=True)
# label["RainTomorrow"].replace("No",0,inplace=True)

class CustomModel():
    def __init__(self):
        pass
    
    def fit(self,x,y):
        self.x=x
        self.y=y
        
    def predict(x):
        pass


# In[ ]:


train_x,test_x,train_y,test_y=train_test_split(properties,label,test_size=0.01)

model.fit(train_x,train_y)
predict_y=model.predict(test_x)
print("MSE:"+str(sk.metrics.mean_squared_error(test_y,predict_y)))
print("Accuracy:"+str(sk.metrics.accuracy_score(test_y,predict_y)))


# In[ ]:


s=data.sample(10)
psample = s[["MinTemp","MaxTemp","Rainfall","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]]
lsample = s[["RainTomorrow"]]
#properties.replace([-np.inf,np.inf],np.nan)
psample["RainToday"]=psample["RainToday"].apply(lambda x:0 if x is "Yes" else 1)
lsample["RainTomorrow"]=lsample["RainTomorrow"].apply(lambda x:0 if x is "No" else 1)
#properties.replace([-np.inf,np.inf],np.nan)
for i in properties["RainToday"]:
    if i is 0:
        print(i)
    else:
        print(
print(model.predict(psample),"h",lsample)

