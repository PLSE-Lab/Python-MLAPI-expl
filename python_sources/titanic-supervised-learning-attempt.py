#!/usr/bin/env python
# coding: utf-8

# # RMS Titanic
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1280px-RMS_Titanic_3.jpg)
# 
# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. RMS Titanic was the largest ship afloat at the time she entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, chief naval architect of the shipyard at the time, died in the disaster.
# 
# https://en.wikipedia.org/wiki/RMS_Titanic
# 
# From the WikiPedia page, we have the following table
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn import metrics #accuracy measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# # This is my attempt for the Titanic Competition

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")
sns.pairplot(train, height = 2.5 )
plt.show()


# In[ ]:


#correlation analysis
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


train.info()


# In[ ]:


test.head()


# In[ ]:


test["Survived"] = 3.0


# In[ ]:


test.head()


# In[ ]:


combined = pd.concat([test, train], axis="rows", sort=False)


# In[ ]:


combined.head()


# In[ ]:


combined["Survived"] = combined["Survived"].astype("int32")


# In[ ]:


combined.info()


# In[ ]:


combined.describe()


# In[ ]:


combined.describe(include=['O'])


# In[ ]:


combined.isna().sum()


# In[ ]:


combined["Age"].describe()


# In[ ]:


combined["Cabin"].unique()


# In[ ]:


combined["Embarked"].unique()


# In[ ]:


combined[combined["Embarked"].isna()]


# In[ ]:


combined['Embarked'].fillna('S',inplace=True)


# In[ ]:


combined.isna().sum()


# In[ ]:


#analysing the data
#Gender of all the travelling passangers
fig = plt.figure(figsize=(16,8)) 
labels = combined['Sex'].value_counts(sort = True).index
sizes = combined['Sex'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)
plt.title('Gender',size = 20)
plt.legend()
plt.show()


# In[ ]:


#analysing the data
#passangers survived
fig = plt.figure(figsize=(16,8)) 
labels = train['Survived'].value_counts(sort = True).index
sizes = train['Survived'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)
plt.title('Survived',size = 20)
plt.show()


# In[ ]:


#analysing the data
fig = plt.figure(figsize=(16,8)) 
label= ("C = Cherbourg", "Q = Queenstown", "S = Southampton")
combined.groupby('Embarked').mean().sort_values(by='Fare',ascending='False')['Fare'].plot(kind='barh',title='Average fair for embarkment point', label=label,fontsize=10)
plt.ylabel('Fare') 
plt.legend()
plt.tight_layout()


# 
# **As per WikiPedia**
# 
# * Total Passengers on Titanic were 2224. 
# * Number of passengers saved 710
# 
# # Observations based on Data
# 
# 
# **Passengers**
# 
# 
# * We have a data of 891 passengers 
# * Average age of Passengers was 29 years
# * Oldest person on the Titanic was 80 years old
# * The biggest family was of 8 members
# * Most of the members were male (577/891) - 64.8%
# 
# **Survived**
# * Only 38.4% of passengers survived
# 
# **Fare**
# * Maximum fare paid ~ 512
# * Average fare paid was ~ 32
# * Maximum Average fare was from Cherbourg
#   

# # Preparing data for model building
# 
# 1. Filling the NA values
# 2. New features

# In[ ]:


combined['Title'] = combined.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


combined['Title'].unique()


# In[ ]:


combined['Title'].value_counts()


# 
# 
# From the above we see that there are different categories of titles present now we will normalise them into broader category below.
# 
# The below is the reference from the https://medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9.
# 

# In[ ]:


normalized_title = {
            'Mr':"Mr",
            'Mrs': "Mrs",
            'Ms': "Mrs",
            'Mme':"Mrs",
            'Mlle':"Miss",
            'Miss':"Miss",
            'Master':"Master",
            'Dr':"Officer",
            'Rev':"Officer",
            'Col':"Officer",
            'Capt':"Officer",
            'Major':"Officer",
            'Lady':"Royalty",
            'Sir':"Royalty",
            'the Countess':"Royalty",
            'Dona':"Royalty",
            'Don':"Royalty",
            'Jonkheer':"Royalty"
            
}


# In[ ]:


combined.Title = combined.Title.map(normalized_title)


# In[ ]:


combined['Title'].value_counts()


# In[ ]:


combined.Cabin = combined.Cabin.fillna('U')


# In[ ]:


combined.Cabin.value_counts()


# In[ ]:


combined.Cabin = combined.Cabin.map(lambda x: x[0])


# In[ ]:


combined.Cabin.value_counts()


# In[ ]:


grouped = combined.groupby(['Sex','Title','Pclass'])


# In[ ]:


combined.head()


# In[ ]:


combined.info()


# In[ ]:


combined.isna().sum()


# In[ ]:


#Mr         757
#Miss       262
#Mrs        200
#Master      61
#Officer     23
#Royalty      6
#Name: Title, dtype: int64

def get_age_title(df,title):
    return df[df["Title"] == title]["Age"].median()


# In[ ]:


df1 = combined.copy()


# In[ ]:


df1["Age_test"] = df1["Title"].apply(lambda x: get_age_title(df1,x))


# In[ ]:


df1.isna().sum()


# In[ ]:


df1.head()


# In[ ]:


df1["Age"].fillna(df1["Age_test"],inplace=True)


# In[ ]:


df1.isna().sum()


# In[ ]:


df1.head()


# In[ ]:


df2 = df1.drop(["Age_test"],axis="columns")


# In[ ]:


df2.head()


# In[ ]:


df2['Fare'].fillna((df2['Fare'].mean()), inplace=True)


# In[ ]:


df2.isna().sum()


# # Data cleaning complete

# In[ ]:


#let us identify the family size
df2["familySize"] = df2['SibSp'] + df2['Parch'] + 1


# In[ ]:


df2.head()


# In[ ]:


def age_group(age):
    if age <= 5:
        return 1
    elif ((age > 5) & (age <=10)):
        return 2
    elif ((age > 10) & (age <=20)):
        return 3
    elif ((age > 20) & (age <= 45)):
        return 4
    elif ((age > 45) & (age <= 55)):
        return 5
    elif age > 55: 
        return 6


# In[ ]:


df21 = df2.copy()


# In[ ]:


df21["age_group"] = df21["Age"].apply(lambda x: age_group(x))


# In[ ]:


df21.head()


# In[ ]:


df2.select_dtypes('object').columns


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
df21[df21["Survived"] == 1].groupby("Sex")["Survived"].sum().plot(kind="bar", title="Survived w.r.t Gender")


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
df21[df21["Survived"] == 1].groupby("age_group")["Survived"].sum().plot(kind="bar", title="Survived w.r.t Gender")


# In[ ]:


df21.Sex = df21.Sex.map({"male":0,"female":1})
df21.head()


# In[ ]:


title_dummies = pd.get_dummies(df21.Title , prefix = "Title")


# In[ ]:


cabin_dummies = pd.get_dummies(df21.Cabin , prefix = "Cabin")
pclass_dummies = pd.get_dummies(df21.Pclass , prefix ="Pclass")
embarked_dummies = pd.get_dummies(df21.Embarked , prefix = "Embarked")


# In[ ]:


df3 = pd.concat([df21 , title_dummies,cabin_dummies,
                             pclass_dummies,embarked_dummies],axis = 1)


# In[ ]:


df21.head()


# In[ ]:


df4 = df3.drop(['Name','Ticket','Pclass','Embarked','Cabin', 'Title','Age'],axis="columns")


# In[ ]:


df4.head()


# In[ ]:


df4.info()


# # Conversion to non-object types complete

# In[ ]:


final_test = df4[df4["Survived"] == 3]


# In[ ]:


final_test.drop(['Survived'],axis="columns", inplace=True)


# In[ ]:


final_train = df4[df4["Survived"] != 3]


# In[ ]:


final_test.head()


# In[ ]:


final_train.head()


# We have the Test and Train data set ready now we can try various ML models
# 
# - final_train
# - final_test
# 

# # Train- Test split from the Training set

# In[ ]:


X = final_train.drop('Survived', axis=1) 
y = final_train.Survived


# In[ ]:


label = final_train.Survived
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=42)


# # LogisticRegression

# In[ ]:


lm_model = LogisticRegression()


# In[ ]:


lm_model.fit(X_train,y_train)


# In[ ]:


lm_model.predict(X_test)


# In[ ]:


print(lm_model.intercept_)


# In[ ]:


lm_model.coef_


# In[ ]:


test_predictions = lm_model.predict(X_test)


# In[ ]:


metrics.accuracy_score(test_predictions,y_test)


# In[ ]:


print(classification_report(y_test,test_predictions))


# In[ ]:


confusion_matrix(y_test,test_predictions)


# In[ ]:


predictions = lm_model.predict(final_test)


# In[ ]:


passengerid = final_test["PassengerId"]
output = pd.DataFrame({"PassengerId":passengerid , "Survived" : predictions})


# In[ ]:


output.to_csv("logistic_regression.csv",index = False)


# # Decision Tree Classifier

# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# In[ ]:


test_predictions = dtree.predict(X_test)


# In[ ]:


print(classification_report(y_test,test_predictions))


# In[ ]:


confusion_matrix(y_test,test_predictions)


# In[ ]:


metrics.accuracy_score(test_predictions,y_test)


# In[ ]:


d_predictions = dtree.predict(final_test)


# In[ ]:


passengerid = final_test["PassengerId"]
output = pd.DataFrame({"PassengerId":passengerid , "Survived" : d_predictions})


# In[ ]:


output.to_csv("dtreeClassifier.csv",index = False)


# # Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc_test_predict = rfc.predict(X_test)


# In[ ]:


metrics.accuracy_score(rfc_test_predict,y_test)


# In[ ]:


print(classification_report(y_test,rfc_test_predict))


# In[ ]:


confusion_matrix(y_test,rfc_test_predict)


# In[ ]:


rfc_predictions = rfc.predict(final_test)


# In[ ]:


passengerid = final_test["PassengerId"]
output = pd.DataFrame({"PassengerId":passengerid , "Survived" : rfc_predictions})


# In[ ]:


output.to_csv("rfClassifier.csv",index = False)


# # SVC

# In[ ]:


svc_model = SVC()


# In[ ]:


svc_model.fit(X_train,y_train)


# In[ ]:


svc_test_predict = svc_model.predict(X_test)


# In[ ]:


metrics.accuracy_score(svc_test_predict,y_test)


# In[ ]:


confusion_matrix(y_test,svc_test_predict)


# In[ ]:


print(classification_report(y_test,svc_test_predict))


# # KNN

# In[ ]:


model=KNeighborsClassifier() 
model.fit(X_train,y_train)
prediction5=model.predict(X_test)
metrics.accuracy_score(prediction5,y_test)


# In[ ]:




