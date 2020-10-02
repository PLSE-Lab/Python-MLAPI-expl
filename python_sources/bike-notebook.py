#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


train=pd.read_csv("../input/bike-sharing-demand/train.csv")
test=pd.read_csv("../input/bike-sharing-demand/test.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# In[ ]:


full_data=train.append(test, sort=False)
print(full_data.shape)
full_data.head()


# In[ ]:


full_data["datetime"]=pd.to_datetime(full_data["datetime"])


# In[ ]:


full_data["year"]=full_data["datetime"].dt.year
full_data["month"]=full_data["datetime"].dt.month
full_data["day"]=full_data["datetime"].dt.day
full_data["hour"]=full_data["datetime"].dt.hour
full_data["weekday"]=full_data["datetime"].dt.weekday


# In[ ]:


full_data.head()


# In[ ]:


full_data2=full_data.set_index(["datetime"])


# In[ ]:


full_data2.head()


# In[ ]:


Season={1:"spring", 2:"summer", 3:"fall", 4:"winter"}


# In[ ]:


full_data.season=[Season[item] for item in full_data.season]
full_data.head(20)


# In[ ]:


print(full_data["season"].isna().sum())
print(full_data["weather"].isna().sum())


# In[ ]:


#1: Clear, Few clouds, Partly cloudy, Partly cloudy
#2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
    
spring_weather={1:"Clear", 2:"Mist+Cloudy", 3: "Light Snow", 4:"Heavy Rain + Ice Pallets"}
summer_weather={1:"Few clouds", 2:"Mist+Broken Clouds", 3:"Light Rain + Thunderstorm", 4:"Ice Pallets"}
fall_weather={1:"Partly cloudy", 2:"Mist+Few clouds", 3:"Scattered clouds", 4:"Thunderstorm+Mist"}
winter_weather={1:"Partly cloudy", 2:"Mist", 3:"Light Rain", 4:"Snow + Fog"}


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def weatherChange():
    for i in range(full_data.shape[0]):
        if full_data.iloc[i, 1]=="spring":
            full_data.iloc[i, 4]=spring_weather[full_data.iloc[i]["weather"]]
        elif full_data.iloc[i, 1]=="summer":
            full_data.iloc[i, 4]=summer_weather[full_data.iloc[i]["weather"]]
        elif full_data.iloc[i, 1]=="fall":
            full_data.iloc[i, 4]=fall_weather[full_data.iloc[i]["weather"]]
        elif full_data.iloc[i, 1]=="winter":
            full_data.iloc[i, 4]=winter_weather[full_data.iloc[i]["weather"]]
        else:
            pass
        
    full_data.set_index(["datetime"], inplace=True)
    full_data.head()
    full_data.to_csv("full_data_datetime.csv")

#weatherChange()


# In[ ]:


full_data_datetime=pd.read_csv("../input/full-data-datetime/full_data_datetime.csv")
full_data_datetime.head()


# In[ ]:


full_data_datetime.drop(["datetime"], axis=1,inplace=True)


# In[ ]:


full_data_datetime.head()


# In[ ]:


dum_df=pd.get_dummies(full_data_datetime, drop_first=True)
dum_df.head()


# In[ ]:


dum_df.shape


# In[ ]:


X=dum_df.drop(["registered", "count", "casual"], axis=1)
y=dum_df["registered"]+dum_df["casual"]+dum_df["count"]


# In[ ]:


y.shape


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


y.isna().sum()


# In[ ]:


X.shape


# In[ ]:


#X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=2019, test_size=0.3)


# In[ ]:


X_train=X.iloc[:10885,:]
X_test=X.iloc[10886:,:]
y_train=y.iloc[:10885]
y_test=y.iloc[10886:]


# In[ ]:


y_train.tail()


# In[ ]:


y_test.head()


# In[ ]:


y_train.shape


# In[ ]:


X_train.isna().sum()


# In[ ]:


y_train.isna().sum()


# In[ ]:



X_train["atemp"]=X_train["atemp"].astype(int)
X_train["windspeed"]=X_train["windspeed"].round()
X_train.head()


# In[ ]:


#X_train=X_train.astype(str)
#y_train=y_train.astype(str)


# In[ ]:


X_train.isna().sum()


# In[ ]:


y_train.isna().sum()


# In[ ]:


decTree = DecisionTreeRegressor(random_state=2019,max_depth=9, min_samples_leaf=5, min_samples_split=20)
decTree.fit(X_train, y_train)
y_pred = decTree.predict(X_test)


# In[ ]:


'''mlp = MLPClassifier(hidden_layer_sizes=(3,2,2),random_state=2018)
mlp.fit( X_train , y_train )
y_pred = mlp.predict(X_test)'''

#from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


y_pred


# In[ ]:


sampleSubmission=pd.read_csv("../input/bike-sharing-demand/sampleSubmission.csv")
sampleSubmission.head()


# In[ ]:


sampleSubmission.tail()


# In[ ]:


sampleSubmission.shape, y_pred.size


# In[ ]:


sampleSubmission["count"]=y_pred
sampleSubmission.head()


# In[ ]:


sampleSubmission.tail()


# In[ ]:


sampleSubmission.set_index(["datetime"], inplace=True)


# In[ ]:


sampleSubmission.to_csv("y_pred.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




