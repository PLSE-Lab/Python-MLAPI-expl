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


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv("../input/train.csv")
test =pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


for items in train.columns:
    if train[items].std() == 0:
        print(items)
        train.drop(items,axis = 1,inplace = True)


# In[ ]:


test.drop("Soil_Type7",axis = 1,inplace = True)
test.drop("Soil_Type15",axis = 1,inplace  =True)


# In[ ]:





# In[ ]:


train.head()


# In[ ]:


sns.countplot(train.Cover_Type)


# In[ ]:


plt.figure(figsize = (7,7))
sns.scatterplot("Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",hue = "Cover_Type",data = train)


# In[ ]:


for i in [1,2,3,4]:
    
    sns.barplot(x = "Cover_Type",y = "Wilderness_Area" + str(i),data  =train)
    plt.show()


# In[ ]:


for i in range(len(train.columns)):
    if train.columns[i] == "Wilderness_Area4":
        
        print(i)


# In[ ]:


for i in range(0,10):
    sns.violinplot(train.Cover_Type,train.drop("Id",axis = 1).iloc[:,i])
    plt.show()


# In[ ]:


train.head()


# In[ ]:


train["net_hyd_distance"]=np.sqrt(train["Vertical_Distance_To_Hydrology"]**2 + train["Horizontal_Distance_To_Hydrology"]**2)
test["net_hyd_distance"]=np.sqrt(test["Vertical_Distance_To_Hydrology"]**2 + test["Horizontal_Distance_To_Hydrology"]**2)


# In[ ]:


train["mean_distance_aminity"] = (train["Horizontal_Distance_To_Hydrology"] + train["Horizontal_Distance_To_Roadways"] + train["Horizontal_Distance_To_Fire_Points"])/3
test["mean_distance_aminity"] = (test["Horizontal_Distance_To_Hydrology"] + test["Horizontal_Distance_To_Roadways"] + test["Horizontal_Distance_To_Fire_Points"])/3


# In[ ]:


train["closeness1"] = train["Horizontal_Distance_To_Hydrology"] - train["Horizontal_Distance_To_Roadways"]
train["closeness2"] = train["Horizontal_Distance_To_Hydrology"] - train["Horizontal_Distance_To_Fire_Points"]
train["closeness3"] = train["Horizontal_Distance_To_Roadways"] - train["Horizontal_Distance_To_Fire_Points"]
test["closeness1"] = test["Horizontal_Distance_To_Hydrology"] - test["Horizontal_Distance_To_Roadways"]
test["closeness2"] = test["Horizontal_Distance_To_Hydrology"] - test["Horizontal_Distance_To_Fire_Points"]
test["closeness3"] = test["Horizontal_Distance_To_Roadways"] - test["Horizontal_Distance_To_Fire_Points"]


# In[ ]:


predictor = train.drop(["Id","Cover_Type"],axis = 1)
target = train["Cover_Type"]


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier()
rnd.fit(predictor,target)


# In[ ]:


importances = rnd.feature_importances_
indices = np.argsort(importances)[::-1]


# In[ ]:


plt.title("top 10 feature importances")
plt.bar(predictor.columns[indices[0:10]],importances[indices[0:10]])
plt.xticks(predictor.columns[indices[0:10]],rotation = 90)


# In[ ]:


#elevation plays a major role
importances


# In[ ]:


for i in range(0,10):
    sns.distplot(predictor.iloc[:,i])
    plt.show()


# In[ ]:


for item in ["Horizontal_Distance_To_Fire_Points","Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Vertical_Distance_To_Hydrology"]:
    
    sns.scatterplot("Elevation",item,hue= "Cover_Type",data = train)
    plt.show()


# In[ ]:


sns.scatterplot(train["Elevation"] -train["Vertical_Distance_To_Hydrology"],train["Vertical_Distance_To_Hydrology"],hue = train["Cover_Type"])


# In[ ]:



sns.scatterplot(train["Elevation"] - .2*train["Horizontal_Distance_To_Hydrology"],train["Horizontal_Distance_To_Hydrology"],hue = train["Cover_Type"])


# In[ ]:


plt.figure(figsize = (10,15))
sns.scatterplot(train["Elevation"] - .02*train["Horizontal_Distance_To_Roadways"],train["Horizontal_Distance_To_Roadways"],hue = train["Cover_Type"])


# In[ ]:


#feature engineering 
train["Elevation_roadways"] = train["Elevation"] - .02*train["Horizontal_Distance_To_Roadways"]
train["Elevation_vd"] = train["Elevation"] - train["Vertical_Distance_To_Hydrology"]
train["Elevation_hd"] = train["Elevation"] - .2*train["Horizontal_Distance_To_Hydrology"]
test["Elevation_roadways"] = test["Elevation"] - .02*test["Horizontal_Distance_To_Roadways"]
test["Elevation_vd"] = test["Elevation"] - test["Vertical_Distance_To_Hydrology"]
test["Elevation_hd"] = test["Elevation"] - .2*test["Horizontal_Distance_To_Hydrology"]


# In[ ]:





# In[ ]:


predictor = train.drop(["Id","Cover_Type"],axis = 1)
target = train.Cover_Type


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[ ]:


rnd1 = RandomForestClassifier(random_state = 123,n_estimators = 175)


# In[ ]:


print(cross_val_score(rnd1,predictor,target,cv = 3).mean())


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from  sklearn.ensemble import AdaBoostClassifier
ex = ExtraTreesClassifier(random_state  =123,n_estimators = 1000)
ex1 = AdaBoostClassifier(base_estimator = ex,n_estimators = 5)


# In[ ]:


cross_val_score(ex1,predictor,target,cv = 3).mean()


# In[ ]:


ex.fit(predictor,target)


# In[ ]:


prediction = ex.predict(test.drop("Id",axis = 1))


# In[ ]:


test["Cover_Type"] = prediction


# In[ ]:


my_submission = test[["Id","Cover_Type"]]


# In[ ]:


my_submission


# In[ ]:


sns.countplot(test["Cover_Type"])


# In[ ]:


test["Cover_Type"].value_counts(normalize = True)*100


# In[ ]:


prediction = cross_val_predict(ex1,predictor,target,cv = 3)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


sns.heatmap(confusion_matrix(target,prediction)/np.sum(confusion_matrix(target,prediction),axis = 1).reshape(-1,1),annot = True)


# In[ ]:


#increase 1,2 accuracy


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(target,prediction))


# In[ ]:


#1 and 2 are hard to seperate


# In[ ]:


train1 = train


# In[ ]:


predictor = train.drop(["Id","Cover_Type"],axis = 1)
target = train.Cover_Type


# In[ ]:


target = target.replace([1,2],0)


# In[ ]:


target = target.replace([3,4,5,6,7],1)


# In[ ]:


ex = ExtraTreesClassifier(random_state = 123,n_estimators = 1000)
ex1 = AdaBoostClassifier(base_estimator = ex,n_estimators = 10)


# In[ ]:


cross_val_score(ex,predictor,target,cv = 3).mean()


# In[ ]:


ex.fit(predictor,target)


# In[ ]:


prediction = ex.predict(test.drop(["Id","Cover_Type"],axis = 1))


# In[ ]:


test["Cover_Type"] = prediction


# In[ ]:





# In[ ]:


data = pd.concat([train[train["Cover_Type"] == 1],train[train["Cover_Type"] == 2]])


# In[ ]:


predictor1 = data.drop(["Id","Cover_Type"],axis = 1)
target = data.Cover_Type


# In[ ]:


predict = cross_val_predict(ex1,predictor1,target,cv = 3)


# In[ ]:


ex1.fit(predictor1,target)


# In[ ]:


sns.heatmap(confusion_matrix(target,predict)/np.sum(confusion_matrix(target,predict),axis = 1).reshape(-1,1),annot = True)


# In[ ]:


data_test = test[test["Cover_Type"] == 0 ]


# In[ ]:


prediction1 = ex1.predict(data_test.drop(["Id","Cover_Type"],axis = 1))


# In[ ]:


data_test["real"] = prediction1


# In[ ]:


data_test["real"]


# In[ ]:


data1 = train[train["Cover_Type"] != 1]


# In[ ]:


data1 = data1[data1["Cover_Type"] != 2]


# In[ ]:


data1.Cover_Type.value_counts()


# In[ ]:


predictor2 = data1.drop(["Id","Cover_Type"],axis = 1)
target2 = data1.Cover_Type


# In[ ]:


ex = ExtraTreesClassifier(random_state = 123,n_estimators = 500)


# In[ ]:


predict = cross_val_predict(ex1,predictor2,target2,cv = 3)


# In[ ]:


sns.heatmap(confusion_matrix(target2,predict)/np.sum(confusion_matrix(target2,predict),axis = 1).reshape(-1,1),annot = True)


# In[ ]:


ex1.fit(predictor2,target2)


# In[ ]:


test.drop("real",axis = 1,inplace = True)


# In[ ]:


dataset = test[test["Cover_Type"] == 1]


# In[ ]:


prediction2 = ex1.predict(dataset.drop(["Id","Cover_Type"],axis = 1))


# In[ ]:


dataset.drop("Cover_Type",axis = 1,inplace = True)


# In[ ]:


dataset["Cover_Type"] = prediction2


# In[ ]:


data_test.drop("Cover_Type",axis = 1,inplace = True)


# In[ ]:


data_test["Cover_Type"] = data_test["real"]


# In[ ]:


data_test.drop("real",axis = 1,inplace = True)


# In[ ]:


data_test.head()


# In[ ]:


test = pd.concat([dataset,data_test]).sort_index()


# In[ ]:


my_submission3 = test[["Id","Cover_Type"]]


# In[ ]:


my_submission3.to_csv("my_submisssion3",index = False)


# In[ ]:




