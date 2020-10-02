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


# # Loading and Tidying Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


train = pd.read_csv("/kaggle/input/airline-passenger-satisfaction/train.csv")
test = pd.read_csv("/kaggle/input/airline-passenger-satisfaction/test.csv")
train.head()


# * We need to convert our target feature to 1's 0's 

# In[ ]:


train["satisfaction"].value_counts() # i checked if there is another class


# In[ ]:


train["satisfaction"] = [1 if each == "satisfied" else 0 for each in train["satisfaction"]]
test["satisfaction"] = [1 if each == "satisfied" else 0 for each in test["satisfaction"]]


# In[ ]:


train.info()


# * Unnamed: 0 and id column wont do any good for us, so lets just drop them
# * And we have no missing values. Thats nice. 
# * (Later Update: Actually we do have missing values, i did not noticed till i get to the modeling part.)

# In[ ]:


train.drop(["Unnamed: 0","id"],axis=1,inplace = True)
test.drop(["Unnamed: 0","id"],axis=1,inplace = True)


# In[ ]:


#i will rename some of the long named columns
train.rename(columns = {"Inflight wifi service": "wifi",
                        "Departure/Arrival time convenient": "timeconv",
                        "Ease of Online booking": "onlinebooking",
                        "Departure Delay in Minutes": "depdel",
                        "Arrival Delay in Minutes": "arrdel"}, inplace = True)

test.rename(columns = {"Inflight wifi service": "wifi",
                        "Departure/Arrival time convenient": "timeconv",
                        "Ease of Online booking": "onlinebooking",
                        "Departure Delay in Minutes": "depdel",
                        "Arrival Delay in Minutes": "arrdel"}, inplace = True)


# In[ ]:


train.describe()


# * most columns are categorical, we have 2 numerical columns, and both seem to have some outliers. especially the departure delay feature. 
# * we can just drop them because it will hurt our model but delay causes unsatisfactory (naturally) i'll decide to drop or not drop after eda.

# # Exploratory Data Analysis

# * first i want to look at delays, and after that i will look at other features.

# In[ ]:


g = sns.FacetGrid(data = train,col = "satisfaction", height = 6)
g.map(sns.distplot, "depdel")
plt.show()


# In[ ]:


sns.boxplot(train["depdel"])
plt.show()


# In[ ]:


g = sns.FacetGrid(data = train,col = "satisfaction", height = 6)
g.map(sns.distplot, "arrdel")
plt.show()


# In[ ]:


sns.boxplot(train["arrdel"])
plt.show()


# * as expected graphs are so skewed, lets sort values and see if peoples who delayed their flights are satisfied or not.

# In[ ]:


train.sort_values("depdel").groupby("Customer Type").tail(10)


# * seems like we dont have a bias in long delay times, so we can drop those outliers in order to help our modeling.

# In[ ]:


Q3 = np.quantile(train["depdel"],0.75)
Q1 = np.quantile(train["depdel"],0.25)

IQR = Q3 - Q1

step = IQR * 3

maxm = Q3 + step
minm = Q3 - step

train = train[train["depdel"] < maxm]


# * Now even if i visualize the delay times again it will look skewed, because most of our flights are took place with 0 delay (like half of all data). 

# In[ ]:


g = sns.FacetGrid(data = train,col = "satisfaction", height = 6)
g.map(sns.distplot, "Flight Distance")
plt.show()


# * It seems that if flight distance increases satisfaction is going up.

# In[ ]:


cat_cols = ["Gender","Customer Type","Type of Travel","Class","wifi","timeconv","onlinebooking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service","Baggage handling","Checkin service","Inflight service","Cleanliness"]


# In[ ]:


def ctgplt(df,variable,to):
    
    "Function for visualization of categorical variables."
    
    var = df[variable]
    values=var.value_counts()
    
    f, ax = plt.subplots(figsize = (8,8))
    sns.countplot(x = variable, hue = to, data = df)
    
    plt.show()
    
    print("{}:\n{}".format(variable,values))


# In[ ]:


for i in cat_cols:
    ctgplt(train, i, "satisfaction")


# * As i expected disyoal customers are not satisfied at all
# * People who are going to a personal travel are probably flying in eco class and thats why their satisfaction is so low. i will visualize this to confirm
# * Most people in business class are satisfied compared to other classes
# * Most people are complaint in wifi levels 1-2-3, but in 0 and 5 almost no complain at all. That is some weird statistics.
# * Time conv. satisfactions are almost equal to each other except 1 class.
# * Online booking have some negative impact on satisfaction
# * Gate location have same impact too. if airport is too big it is really exhaustive to get to the gate
# * Food and drink is as expected
# * Online boarding has some serious impact on low classes
# * Seat comfort has some negativity but also has some positive impact too
# * Peoples have some complaint on flight entertainment.
# * On board, leg room, inflight services and Cleanliness almost have the same distribution. like many features
# 

# In[ ]:


f, ax = plt.subplots(figsize = (12,8))
train["Type of Travel"].hist(by = train["Class"],xrot =30,ax=ax)
plt.show()


# * We can see that most of the travels in eco class are personal travels.
# * And dissatisfactory on the business travels are also coming from eco class. 

# In[ ]:


corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(16, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, ax=ax)
plt.show()


# * we can see that arrdel and depdel are highly correlated and they are not correlated with our target. so we can drop them.
# * Also gate location is not related with our target feature, i will drop that too.

# In[ ]:


train.drop(["arrdel","depdel","Gate location"],axis=1,inplace=True)
test.drop(["arrdel","depdel","Gate location"],axis=1,inplace=True)


# # Modeling

# In[ ]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier
import lightgbm as lgb


# In[ ]:


labelenc = ["Gender","Customer Type","Type of Travel","Class"]
scal = []
for each in train.columns:
    if train[each].dtype == "int64" or train[each].dtype == "float64":
        scal.append(each)

le = LabelEncoder()
scaler = MinMaxScaler()

# Label Encoder
for each in labelenc:
    train[each] = le.fit_transform(train[each])
    test[each] = le.transform(test[each])

# MinMax Scaler
train[train.columns] = scaler.fit_transform(train[train.columns])
test[test.columns] = scaler.transform(test[test.columns])


# In[ ]:


train.head()


# In[ ]:


X = train.drop("satisfaction",axis=1)
y = train["satisfaction"]

X_test = test.drop("satisfaction",axis=1)
y_test = test["satisfaction"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


xg = XGBClassifier(n_estimators = 300,max_depth = 3)

xg.fit(X_train, y_train)

print("train score: ", xg.score(X_train, y_train))
print("vald score: ", xg.score(X_val, y_val))

preds = xg.predict(X_test)

print("test score",accuracy_score(y_test,preds))
print("roc-auc",roc_auc_score(y_test,preds))


# In[ ]:


cf_matrix = confusion_matrix(y_test, preds)
sns.heatmap(cf_matrix,annot = True, fmt="g",cmap="Greens")
plt.show()


# In[ ]:


importances = pd.Series(data=xg.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize=(8,8))
importances_sorted.plot(kind='barh')
plt.title('Features Importances')
plt.show()


# In[ ]:


lg=LGBMClassifier(max_depth = 7, n_estimators = 250,n_jobs=-1)

lg.fit(X_train,y_train)

print("train score: ", lg.score(X_train, y_train))
print("vald score: ", lg.score(X_val, y_val))

preds = lg.predict(X_test)

print("test score",accuracy_score(y_test,preds))
print("roc-auc",roc_auc_score(y_test,preds))


# * We can see that lightgbm gave us a better result

# In[ ]:


cf_matrix = confusion_matrix(y_test, preds)
sns.heatmap(cf_matrix,annot = True, fmt="g",cmap="Greens")
plt.show()


# In[ ]:


importances = pd.Series(data=lg.feature_importances_,
                        index= X_train.columns)

importances_sorted = importances.sort_values()
plt.figure(figsize=(8,8))
importances_sorted.plot(kind='barh')
plt.title('Features Importances')
plt.show()


# * now i will select first 10 features and fit the model again lets see if our score will improve or not

# In[ ]:


X_train_selected = X_train[["Leg room service","Seat comfort","timeconv","Customer Type","Type of Travel","Baggage handling","Inflight service","wifi","Age","Flight Distance"]]
X_val_selected = X_val[["Leg room service","Seat comfort","timeconv","Customer Type","Type of Travel","Baggage handling","Inflight service","wifi","Age","Flight Distance"]]
X_test_selected = X_test[["Leg room service","Seat comfort","timeconv","Customer Type","Type of Travel","Baggage handling","Inflight service","wifi","Age","Flight Distance"]]

lg2 = LGBMClassifier(max_depth=5, n_estimators=150, n_jobs=-1)

lg2.fit(X_train_selected, y_train)

print("train score: ", lg2.score(X_train_selected, y_train))
print("vald score: ", lg2.score(X_val_selected, y_val))

preds = lg2.predict(X_test_selected)

print("test score",accuracy_score(y_test,preds))
print("roc-auc",roc_auc_score(y_test,preds))


# * As expected, it did not. But our runtime is now shorter, if we tune the hyperparameters we could get a better accuracy. But i will not do that now cause it takes time. Why dont you do give it a try?

# ### **Thank you for reading, I hope you like it. See you in another notebook.**
