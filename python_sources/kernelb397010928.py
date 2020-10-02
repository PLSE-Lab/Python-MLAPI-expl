# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:12:29 2019

@author: Preetvijay
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
train = pd.read_csv("../input/Train.csv")
test = pd.read_csv("../input/Test.csv")
subm = pd.read_csv("../input/sample_submission.csv")
train_data = train[train.columns[2:14]]
test_data = test[test.columns[2:14]]
print(test["is_holiday"].value_counts())
date = train["date_time"]
date = list(date)
day = []
month = []
for i in date:
    dt = datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S')
    
    temp = str( dt.date()).split("-")
    day.append(temp[2])
    month.append(temp[1])
day = pd.DataFrame(day,columns = ["day"])
month = pd.DataFrame(month,columns = ["month"])
train_data = pd.concat([day,month,train_data],axis = 1)
date = test["date_time"]
date = list(date)
day = []
month = []
for i in date:
    dt = datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S')
    
    temp = str( dt.date()).split("-")
    day.append(temp[2])
    month.append(temp[1])
day = pd.DataFrame(day,columns = ["day"])
month = pd.DataFrame(month,columns = ["month"])
test_data = pd.concat([day,month,test_data],axis = 1)
#test_data = test_data.drop("weather_description", axis=1)
Y = train['traffic_volume']
train_data.isnull().sum()
counts = train_data["weather_type"].value_counts()
#train_data = train_data.drop("weather_description", axis=1)
#train_data = train_data[train_data['weather_type'].isin(counts[counts > 20].index)]
train_data["weather_type"].value_counts()
train_data["clouds_all"].value_counts()
print(train_data["clouds_all"].median())
# =============================================================================
# plt.figure(figsize=(300,300)
# =============================================================================
sns.boxplot(x=train_data["air_pollution_index"],y = Y)
plt.show()
from sklearn.preprocessing import LabelEncoder
train_len = len(train_data["weather_type"])
cat_data = pd.concat([train_data[["weather_type","weather_description"]],test_data[["weather_type","weather_description"]]],axis = 0)
lb  = LabelEncoder()
cat_data["weather_type"] = lb.fit_transform(cat_data["weather_type"])
cat_data["weather_description"] = lb.fit_transform(cat_data["weather_description"])

cat_data = pd.get_dummies(cat_data, columns=["weather_type","weather_description"], prefix = ["weather_type","weather_description"])


train_data = train_data.drop(["weather_type","weather_description"],axis = 1)

test_data = test_data.drop(["weather_type","weather_description"],axis = 1)

# =============================================================================
# from sklearn import preprocessing
# train_data = preprocessing.scale(train_data)
# test_data = preprocessing.scale(test_data)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=5)
# train_data = pca.fit_transform(train_data)
# train_data = pd.DataFrame(data = train_data)
# test_data = pca.fit_transform(test_data)
# test_data = pd.DataFrame(data = test_data)
# =============================================================================
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
train_data = pd.concat([train_data,cat_data[:train_len]],axis = 1)
#########################################test_data#############################################

test_cat = cat_data[train_len:]
test_cat.reset_index(drop=True, inplace=True)
test_data = pd.concat([test_data,test_cat],axis = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size=0.2)
############################################################################################
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=100)
reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
reg3 = LinearRegression()

regr = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])

# =============================================================================
# regr = RandomForestRegressor(max_depth=5, random_state=0,
#                           n_estimators=100)
# 
# =============================================================================
regr.fit(X_train,y_train)
predict = regr.predict(X_test)
count = 0
predictor = len(predict)
for i in range(len(predict)):
    print(list(y_test)[i],"--->",predict[i])
    if((list(y_test)[i] - predict[i])<20):
        count = count + 1
accuracy  = (count/predictor)*100
print(accuracy)
print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))
test_label = regr.predict(test_data).astype(int)
##############################################################################
Lable = pd.DataFrame(test_label,columns = ["traffic_volume"])
time = test["date_time"]
submission = pd.concat([time,Lable],axis = 1)
submission.to_csv("Predictor.csv",index=False)
#subm = pd.read_csv("B:\my lab\DataSets\Predictor.csv")

