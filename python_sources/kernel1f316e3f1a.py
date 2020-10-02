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


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


buff = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train = pd.DataFrame(buff)
buff = pd.read_csv("/kaggle/input/titanic/test.csv")
df_test =pd.DataFrame(buff)
combine = [df_train, df_test]


# In[ ]:


df_train.head(20)


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()
print("-" * 40)
df_test.info()


# * Age, Embarked are almost full features that should be completed.
# * Cabin is too incomplete - drop.
# * Fare also needs to be completed in the test set.
# 

# In[ ]:


df_train.corr()


# * **Pclass** is highly relevent -> ordering should be changed because of heirarchy in data is backwards.
# * **Fare** is decently relevent -> (come up with something later)
# * **Age** seems unrelated but intuitivly should be related -> need to do feature engineering
# * **Parch and SibSp** unrelevent alone but should be combined

# In[ ]:


sns.distplot(df_train[df_train["Survived"] == 1]["Age"], bins=50)
sns.distplot(df_train[df_train["Survived"] == 0]["Age"], bins=50)


# Largest difference here is the 0-10 age grouping

# In[ ]:


#creating the "young" feature
for data in combine:
    arr = []
    for i in range(len(data)):
        if data["Age"].iloc[i] < 15:
            arr.append(1)
        else:
            arr.append(0)
    data["Young"] = arr
df_train[df_train["Young"] == 1] #78 children within data


# assuming a child is 0-14 years of age (the statistical significance within the data)

# In[ ]:


df_train["Fare"].median()


# In[ ]:


for data in combine:
    for i in range(len(data)):
        if data["Fare"].isna()[i] == True:
            data["Fare"].iloc[i] = data["Fare"].median()
        if data["Embarked"].isna()[i] == True:
            data["Embarked"].iloc[i] = "S"  # S is the median value


# estimating data with few missing points

# In[ ]:


for data in combine:
    data.drop(columns=["Cabin"], inplace=True)


# In[ ]:


df_train


# In[ ]:


#switching the heirarchy of Pclass
for data in combine:
    data["Pclass"] = data["Pclass"].map({1:3, 3:1, 2:2})


# In[ ]:


#switching Sex to binary encoding
for data in combine:
    data["Sex"] = data["Sex"].map({"male": 0, "female" : 1})


# In[ ]:


df_train


# In[ ]:


#extracting title
for data in combine:
    data["Title"] = data["Name"].apply(lambda name: name.split(".")[0].split(" ")[-1])


# In[ ]:


for data in combine:
    data["Title"] = data["Title"].map({
"Mr" : 1,
"Miss" : 2,
"Mrs" : 3,
"Master" : 4,
"Dr" : 5,
"Rev" : 5,
"Mlle" : 5,
"Col" : 5,
"Major" : 5,
"Ms" : 5,
"Lady" : 5,
"Countess" : 5,
"Sir" : 5,
"Don"  : 5,
"Capt" : 5,
"Jonkheer" : 5,
"Mme" : 5})


# 5 means a rare title 

# In[ ]:


for data in combine:
    data["Family"] = data["Parch"] + data["SibSp"]


# In[ ]:


sns.distplot(df_train[df_train["Family"] == 0]["Survived"])
sns.distplot(df_train[df_train["Family"] != 0]["Survived"])


# people travelling alone are more prone to dying 

# In[ ]:


#create travelling alone feature
for data in combine:
    data["Alone"] = data["Family"].apply(lambda x : 1 if x < 1 else 0)


# In[ ]:


df_train


# In[ ]:


#people associated with royalty
royal_tickets = list(df_train[df_train["Title"] == 5]["Ticket"])
sns.distplot(df_train[df_train["Ticket"].isin(royal_tickets)]["Survived"])
sns.distplot(df_train[~df_train["Ticket"].isin(royal_tickets)]["Survived"])


# Seems much more likley for a person to survive if they are with a person of royalty

# In[ ]:


sns.distplot(df_train[df_train["Title"] == 5]["Survived"])


# In[ ]:


for data in combine:
    data["WithRoyalty"] = data["Ticket"].apply(lambda tick: 1 if tick in royal_tickets else 0)


# In[ ]:


df_train["Embarked"].value_counts()


# In[ ]:


for data in combine:
    data["Embarked"] = data["Embarked"].map({"Q" : 1, "C" : 2, "S" : 3})


# In[ ]:


df_train.corr()


# **drop**
# * WithRoyalty (experiment first)
# * Family
# * Parch
# * SibSp
# * Age
# 
# Still have pretty weak correlations for a couple features but still relevent enough to include in model :/

# In[ ]:


#creating model without using age -> might have to come back and fix
for data in combine:
    data.drop(columns=[ "Family", "Parch", "SibSp", "Age", "Name", "Ticket", "PassengerId"], inplace=True)


# In[ ]:


df_train


# In[ ]:


y = np.array(df_train["Survived"].values)
x = np.array(df_train.drop(columns="Survived").values)
np.reshape(x,(891,len(list(df_train.columns)) -1,1))
print("X shape:", x.shape)
num_classes = x.shape[1]


# In[ ]:


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x


# Normalize?

# In[ ]:


'''from sklearn.decomposition import PCA
pca = PCA(0.92)
x = pca.fit_transform(x)
x.shape
'''


# No real reason for pca on such a small dataset

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


model = Sequential()
model.add(Dense(num_classes * 2, activation="relu", input_shape = (x.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(num_classes ** 2, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes ** 3, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mse"])
history = model.fit(x,y, epochs=35, batch_size = 1,validation_split=0.05, shuffle=True )


# In[ ]:


plt.plot(history.history["mse"]) 
plt.plot(history.history["val_mse"])


# seems to be at the optimal fitting point

# In[ ]:


df_test["Title"].iloc[414] = 1


# In[ ]:


x_test = StandardScaler().fit_transform(np.array(df_test.values))
final = model.predict(x_test)


# In[ ]:


final = [int(np.round(x[0])) for x in final]


# stand alone neural net got ~78 % acc -> try ensemble

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=3, random_state=0).fit(x[:800],y[:800])
gbc.score(x[800:], y[800:])


# In[ ]:


final1 = gbc.predict(x_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




# Training classifiers
dtc = DecisionTreeClassifier(max_depth=3).fit(x[:800],y[:800])
knn = KNeighborsClassifier(n_neighbors=4).fit(x[:800],y[:800])
svc = SVC(kernel='rbf', probability=True).fit(x[:800],y[:800])
print(dtc.score(x[800:], y[800:]))
print(knn.score(x[800:], y[800:]))
print(svc.score(x[800:], y[800:]))

final2 = dtc.predict(x_test)
final3 = knn.predict(x_test)
final4 = svc.predict(x_test)


# In[ ]:


pred_data = [final, final1, final2, final3, final4]
pred = []
for i in range(len(final)):
    sum_d = 0
    for j in range(len(pred_data)):
        if j == 0: sum_d += 2 * pred_data[j][i]    #double weigh the initial model
        else: sum_d += pred_data[j][i]
    if sum_d > 3:
        pred.append(1)
    else:
        pred.append(0)

    


# In[ ]:


ans = pd.DataFrame(pred)
ans["Survived"] = pred
ans["PassengerId"] = list(range(892,892 + len(df_test) ))
ans = ans[["PassengerId", "Survived"]]
ans
'''
ans = pd.DataFrame()
ans["Survived"] = final
ans["PassengerId"] = list(range(892,892 + len(df_test) ))
ans = ans[["PassengerId", "Survived"]]
'''
ans


# In[ ]:


ans.to_csv("submission.csv", index=False)


# In[ ]:


len(ans[ans["Survived"] == 1])


# number of survivors

# In[ ]:




