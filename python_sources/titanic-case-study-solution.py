#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --user imblearn


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


test_data=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


test_data.head()


# In[ ]:


test_data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


# Delete the "Cabin" column from the dataframe
data = data.drop("Name", axis=1)
data.head()


# In[ ]:


#print(data.Survived.unique())
#print(data.Pclass.unique())
#print(data.Sex.unique())
#print(data.Age.unique())

#print(data.SibSp.unique())
#print(data.Parch.unique())
#print(data.Fare.unique())
print(data.Embarked.unique())


# In[ ]:


data[data["Embarked"]=="Q"]


# In[ ]:


data.shape[0]


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data[data["Sex"]=="female"].mean()


# In[ ]:


male_age_mean= 30.72
female_age_mean= 27.91


# In[ ]:


index=data[data['Embarked'].isnull()].index.tolist()


# In[ ]:


index


# In[ ]:


data=data.drop(index,axis=0)


# In[ ]:


print(data[data["Sex"]=="male"].boxplot(column="Age"))


# In[ ]:


data.corr(method="pearson")


# In[ ]:


data_temp=data.dropna()


# In[ ]:


from sklearn import linear_model


# In[ ]:


Lmodel=linear_model.LinearRegression()


# In[ ]:


Lmodel.fit(data_temp[["Pclass","SibSp","Parch","Fare"]],data_temp.Age)


# In[ ]:


Lmodel.coef_


# In[ ]:


Lmodel.intercept_


# In[ ]:


Lmodel.predict([[3,0,0,8.4583]])


# In[ ]:


data_na=data[data['Age'].isnull()]


# In[ ]:


data_na=data_na.drop(["Age","Sex","PassengerId","Survived","Cabin","Embarked","Ticket"],axis=1)


# In[ ]:


data_na.head()


# In[ ]:


data_na["Age"]=Lmodel.predict(data_na)


# In[ ]:


data_na.head()


# In[ ]:


data_na['Age'] = data_na['Age'].apply(np.ceil)


# In[ ]:


data_na


# In[ ]:


Age_na_index=data_na.index


# In[ ]:


data.loc[data_na.index,:].Age


# In[ ]:


#for i in Age_na_index:
data.loc[data_na.index,"Age"] = data_na.Age
#data.loc[data_na.index,:].Age
data_na.Age


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape[0]


# In[ ]:


index_m=data[(data["Age"]<0) & (data["Sex"]=="male")].index


# In[ ]:


index_f=data[(data["Age"]<0) & (data["Sex"]=="female")].index


# In[ ]:


data.loc[index_m,"Age"]=29.0


# In[ ]:


data.loc[index_f,"Age"]=22.0


# In[ ]:


data[data["Fare"]==66.6] 


# In[ ]:


data.isnull().sum()


# In[ ]:


data.Cabin.unique()


# In[ ]:


data=data.drop(["Cabin"],axis=1)


# In[ ]:


data.head()


# ### Performing encoding for the categorical variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


label_encoder = LabelEncoder()
label_encoder1 = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["Embarked"] = label_encoder1.fit_transform(data["Embarked"])


# In[ ]:


data.tail()


# ## Now we will apply the models on the data

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[ ]:


data.corr(method="pearson")


# In[ ]:


y=data["Survived"]


# In[ ]:


X=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]


# In[ ]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


# In[ ]:


y_train.shape


# In[ ]:


lr=LogisticRegression()
svm=SVC()
rf=RandomForestClassifier(n_estimators=100)


# In[ ]:


lr.fit(X_train_res,y_train_res)


# In[ ]:


svm.fit(X_train_res,y_train_res)


# In[ ]:


rf.fit(X_train_res,y_train_res)


# In[ ]:


print("Liner Regression score",lr.score(X_test,y_test))
print("SVM score",svm.score(X_test,y_test))
print("Random Forest",rf.score(X_test,y_test))


# In[ ]:


lr.predict(X_test)


# In[ ]:


svm.predict(X_test)


# In[ ]:


rf.predict(X_test)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


avg_LR=cross_val_score(LogisticRegression(),X,y)
avg_LR.mean()


# In[ ]:


avg_SVC=cross_val_score(SVC(),X,y)
avg_SVC.mean()


# In[ ]:


avg_RFC=cross_val_score(RandomForestClassifier(n_estimators=100),X,y)
avg_RFC.mean()


# ## Changing number of factors to increase accuracy

# In[ ]:


data.corr(method="pearson")


# In[ ]:


##Since there is minimum dependency on SibSp.Droping it from the dataset


# In[ ]:


#X=data[["Pclass","Sex","Age","Parch","Fare","Embarked"]]
X=data[["Pclass","Sex","Parch","Fare","Embarked"]]


# In[ ]:


y=data["Survived"]


# In[ ]:


avg_LR=cross_val_score(LogisticRegression(),X,y)
avg_LR.mean()


# In[ ]:


avg_SVC=cross_val_score(SVC(),X,y)
avg_SVC.mean()


# In[ ]:


avg_RFC=cross_val_score(RandomForestClassifier(n_estimators=50),X,y)
avg_RFC.mean()


# In[ ]:


rf=RandomForestClassifier(n_estimators=50)


# In[ ]:


rf.fit(X,y)


# In[ ]:


data.Embarked.unique()


# ## Importing test data for predicting

# In[ ]:


test=pd.read_csv("../input/titanic/test.csv")


# In[ ]:


test.head()


# In[ ]:


label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
test["Sex"] = label_encoder2.fit_transform(test["Sex"])
test["Embarked"] = label_encoder3.fit_transform(test["Embarked"])


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test[test["Fare"].isnull()==True]


# In[ ]:


test.loc[152,"Fare"]=8


# In[ ]:


#X_test=test[["Pclass","Sex","Age","Parch","Fare","Embarked"]]
X_test=test[["Pclass","Sex","Parch","Fare","Embarked"]]


# In[ ]:


Survived_temp=rf.predict(X_test)
Survived_temp


# In[ ]:


test["Survived"]=Survived_temp


# In[ ]:


test[["PassengerId","Survived"]].to_csv("prediction.csv",index=False,header=True)


# In[ ]:




