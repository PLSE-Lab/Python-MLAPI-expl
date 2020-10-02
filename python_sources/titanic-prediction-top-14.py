#!/usr/bin/env python
# coding: utf-8

# # Beginner's Approach to Titanic Dataset Analysis. Upvote if you like it! 

# Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# Reading

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
submission = test[["PassengerId"]]


# Exploring

# In[ ]:


train


# In[ ]:


train = train.drop_duplicates()


# In[ ]:


print("______TRAIN_______")
print(train.info())
print("______TEST________")
print(test.info())


# Cleaning Data 
# * PassengerID may be dropped.
# * Pclass may be made string.
# * Title may be extracted from Name and then Name may be dropped.
# * Age may be rounded.
# * Cabin may be dropped.

# In[ ]:


def clean1(df):
    
    Title = []
    for name in df["Name"]:
        if "Mr." in name:
            Title.append("Mr")
        elif "Mrs." in name:
            Title.append("Mrs")
        elif "Miss." in name:
            Title.append("Miss")
        elif "Master." in name:
            Title.append("Master")
        elif "Rev." in name:
            Title.append("Rev")
        elif "Don." in name:
            Title.append("Don")
        elif "Dr." in name:
            Title.append("Dr")
        elif "Mme." in name:
            Title.append("Miss")
        elif "Ms." in name:
            Title.append("Mrs")
        elif "Major." in name:
            Title.append("Major")
        elif "Mrs" in name:
            Title.append("Mrs")
        elif "Mr" in name:
            Title.append("Mr")
        elif "Mlle." in name:
            Title.append("Miss")
        elif "Col." in name:
            Title.append("Col")
        elif "Capt." in name:
            Title.append("Capt")
        elif "Countess." in name:
            Title.append("Countess")
        elif "Jonkheer." in name:
            Title.append("Jonkheer")
        else:
            Title.append("Other")
            
    df["Title"] = Title
    
    df = df.astype({
        "Pclass" : str,
    })

    df["Age"] = round(df["Age"])
    
    df = df.drop(columns = ["PassengerId", "Name", "Cabin"])
    
    return df


# In[ ]:


train = clean1(train)
test = clean1(test)


# In[ ]:


print("______TRAIN_______")
print(train.isnull().sum())
print("______TEST________")
print(test.isnull().sum())


# Cleaning Data
# * Age, Embarked and Fair may be filled using some patterns

# In[ ]:


train[train["Embarked"].isnull()]
# pd.crosstab(train["Embarked"], train["Survived"])
# pd.crosstab(train["Embarked"], train["Pclass"])
# pd.crosstab(train["Embarked"], train["Title"])


# Embarked is likely "S" for both

# In[ ]:


ind = train[train["Embarked"].isnull()].index
train.loc[ind, "Embarked"] = "S"


# In[ ]:


test[test["Fare"].isnull()]


# In[ ]:


sns.catplot(x = "Fare", y = "Age", hue = "Pclass", data = train)


# In[ ]:


sns.catplot(x = "Fare", y = "Age", hue = "Sex", data = train)


# In[ ]:


sns.catplot(x = "Fare", y = "Age", hue = "Embarked", data = train)


# We may take Fare = average Fare of passengers with:
# 1. 45<Age<75
# 2. Sex = male
# 3. Pclass = 3
# 4. Embarked = S

# In[ ]:


fare = np.nanmean(train["Fare"][
    (train["Age"] > 45) & (train["Age"] < 75) & 
    (train["Sex"] == "male") & 
    (train["Pclass"] == "3") &
    (train["Embarked"] == "S")
])
ind = test[test["Fare"].isnull()].index
test.loc[ind, "Fare"] = fare


# In[ ]:


sns.catplot(x = "Embarked", y = "Age", hue = "Pclass", kind = "box", data = train)


# In[ ]:


sns.catplot(x = "Embarked", y = "Age", hue = "Sex", kind = "box", data = train)


# We may take Age = median Age of passengers with same:
# 1. Embarked
# 2. Pclass
# 3. Sex

# In[ ]:


def clean2(df, train):
    ind = df[df["Age"].isnull()].index
    for i in ind:
        emb = df.loc[i]["Embarked"]
        pc = df.loc[i]["Pclass"]
        sex = df.loc[i]["Sex"]
        
        age = np.nanmedian(train["Age"][(train["Embarked"] == emb) & (train["Pclass"] == pc) & (train["Sex"] == sex)])
        df.loc[i, "Age"] = age

    return(df)


# In[ ]:


train = clean2(train, train)
test = clean2(test, train)


# Clean Data
# * Sex may be one-hot encoded.
# * Embarked may be one-hot & frequency encoded.
# * Ticket may be frequency encoded.
# * Title may be frequency encoded.
# * Pclass may be one-hot & frequency encoded.

# In[ ]:


cols = ["Sex", "Embarked", "Pclass"]
    
train_temp = train[cols]
train_ind = train_temp.index
    
test_temp = test[cols]
test_ind = test_temp.index
    
enc = OneHotEncoder(drop = "first")
# To avoid multi-collinearlty (dummy variable trap) due to one-hot-encoding, use drop = "first"
enc.fit(train_temp)
        
train_hot = enc.transform(train_temp).toarray()
train_hot = pd.DataFrame(train_hot, columns = enc.get_feature_names(cols), index = train_ind)
train = train.drop(columns = ["Sex"])
train = pd.concat([train, train_hot], axis = 1)
    
test_hot = enc.transform(test_temp).toarray()
test_hot = pd.DataFrame(test_hot, columns = enc.get_feature_names(cols), index = test_ind)
test = test.drop(columns = ["Sex"])
test = pd.concat([test, test_hot], axis = 1)


# In[ ]:


def GetFreqMap(train_data):
    cols = ["Embarked", "Ticket", "Title", "Pclass"]
    MyMap = {}
    for col in cols:
        temp = {}
        temp = train_data[col].value_counts()/train_data.shape[0]
        temp = defaultdict(lambda : 0, temp)
        MyMap[col] = temp
    
    return MyMap 


# In[ ]:


def GetMeanMap(train_data):
    cols = ["Embarked", "Ticket", "Title", "Pclass"]
    MyMap = {}
    for col in cols:
        temp = {}
        categories = train_data[col].value_counts().keys()
        for cat in categories:
            n1 = train_data[col][(train_data[col] == cat) & (train_data["Survived"] == 1)].shape[0]
            n2 = train_data[col][(train_data[col] == cat)].shape[0]
            temp[cat] = n1/n2
        
        temp = defaultdict(lambda : 0, temp)
        MyMap[col] = temp
        
    return MyMap


# In[ ]:


def MapMe(df, FM, MM):
    temp_freq = pd.DataFrame()
    temp_mean = pd.DataFrame()
    
    cols = ["Embarked", "Ticket", "Title", "Pclass"]
    for col in cols:
        temp_freq[col + "_freq"] = df[col].map(FM[col])
        temp_mean[col + "_mean"] = df[col].map(MM[col])
        
    df = pd.concat([df, temp_freq, temp_mean], axis = 1)
    df = df.drop(columns = ["Embarked", "Ticket", "Title", "Pclass"])
    
    return df


# In[ ]:


FM = GetFreqMap(train)
MM = GetMeanMap(train)
train = MapMe(train, FM, MM)
test = MapMe(test, FM, MM)


# In[ ]:


train_data = train.loc[:, train.columns != "Survived"]
train_labels = train[["Survived"]]
test_data = test

# scaler = MinMaxScaler()
# scaler.fit(train_data)
# cols = train_data.columns

# train_ind = train_data.index
# train_data = scaler.transform(train_data)
# train_data = pd.DataFrame(train_data, columns = cols, index = train_ind)

# test_ind = test_data.index
# test_data = scaler.transform(test_data)
# test_data = pd.DataFrame(test_data, columns = cols, index = test_ind) 


# Clean Data
# * Check for collinearity

# In[ ]:


plt.figure(figsize = (15, 15))
sns.heatmap(train_data.corr(), cmap = "YlGnBu", annot = True)


# Removing Pclass_3 and Embarked_S

# In[ ]:


train_data = train_data.drop(columns = ["Pclass_3", "Embarked_S"])
test_data = test_data.drop(columns = ["Pclass_3", "Embarked_S"])


# Modeling

# In[ ]:


def ModelAccuracy(model, train_data, train_labels):
    train_data, test_data, train_labels, test_labels = train_test_split(
        train_data, 
        train_labels, 
        test_size = 0.2,
    )
    
    if model == "RandomForest":
        classifier = RandomForestClassifier(n_estimators = 100)
    elif model == "DecisionTree":
        classifier = DecisionTreeClassifier()
    elif model == "XGBoost":
        classifier = XGBClassifier()
    elif model == "Logistic":
        classifier = LogisticRegression(solver = "lbfgs") 
    elif model == "SVM":
        classifier = SVC(gamma = "scale")
    elif model == "GradientBoost":
        classifier = GradientBoostingClassifier()
    elif model == "AdaBoost":
        classifier = AdaBoostClassifier()
    elif model == "LDA":
        classifier = LinearDiscriminantAnalysis()
    elif model == "QDA":
        classifier = QuadraticDiscriminantAnalysis()
    elif model == "CatBoost":
        classifier = CatBoostClassifier()

    classifier.fit(train_data, train_labels.values.ravel())
    prediction = classifier.predict(test_data)

    accuracy = metrics.accuracy_score(test_labels, prediction)
    return accuracy


# In[ ]:


models = ["RandomForest", "DecisionTree", "XGBoost", "Logistic", "SVM", "GradientBoost", 
          "AdaBoost", "LDA", "QDA", "CatBoost"]

performance_table = []
for i in range(1):
    scores = []
    for model in models:
        accuracy = ModelAccuracy(model, train_data, train_labels)
        scores.append(accuracy)
    performance_table.append(scores)

performance_table = pd.DataFrame(performance_table, columns = models)


# In[ ]:


performance_table


# In[ ]:


plt.style.use('ggplot')
plt.style.use('seaborn-white')

palette = plt.get_cmap("Set1")

for i, column in enumerate(performance_table):
    plt.plot(performance_table[column], color = palette(i), label = column)

plt.legend(loc = 2, ncol = 2, bbox_to_anchor = (0,1.5))
plt.title("Line Plot")
plt.xlabel("Run")
plt.ylabel("Accuracy")


# In[ ]:


performance_table.describe()


# In[ ]:


df = performance_table.describe()
mean_list = list(df.loc["mean", :])

palette = plt.get_cmap("Set1")

for i, mean in enumerate(mean_list):
    plt.axhline(y = mean, color = palette(i), label = models[i], linewidth = 2)
    
plt.legend(loc = 2, ncol = 2, bbox_to_anchor = (0,1.5))
plt.xticks([])
plt.ylim(min(mean_list) - 0.01, max(mean_list) + 0.01)
plt.ylabel("Accuracy")    


# In[ ]:


classifier = CatBoostClassifier()
classifier.fit(train_data, train_labels.values.ravel())
prediction = classifier.predict(test_data)


# In[ ]:


submit = pd.DataFrame(list(zip(submission["PassengerId"], prediction)), columns = ["PassengerId", "Survived"])
submit["Survived"] = submit["Survived"].map({
    1.0 : 1,
    0.0 : 0,
})
submit.to_csv("submit.csv", index = False)


# In[ ]:




