#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train_data= pd.read_csv('../input/titanic/train.csv' )
train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


## Drop columns - Passenger Id, Name ,Ticket.
train_data.drop(['PassengerId'], axis=1, inplace=True)
train_data.drop(['Name'], axis=1, inplace=True)
train_data.drop(['Ticket'], axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


## Train Data has missing values in columns - Age,Cabin,Embarked

# Handle missing data in Embarked Column
Mode = train_data['Embarked'].mode().values[0]
train_data['Embarked'].fillna(Mode ,inplace=True)
train_data.info()           


# In[ ]:


train_data.Embarked.value_counts()


# In[ ]:


## 0= not allocated cabin , 1= allocated cabin
train_data['Allocated_Cabin'] =np.where(train_data['Cabin'].isna(),0,1)
train_data.drop(columns=['Cabin'], axis=1, inplace = True)
train_data.head()


# In[ ]:


## Parch and SibSp Columns can be merged to form a new Feature

train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch']
train_data.head()


# In[ ]:


## After Calculating Family Size, now Parch and Sibsp can be dropped safely
train_data.drop(['SibSp'], axis=1,inplace=True)
train_data.drop(['Parch'], axis=1, inplace=True)
train_data.head()


# In[ ]:


features= train_data.iloc[:,1:8].values
label =train_data.iloc[:,[0]].values
#label


# In[ ]:


print(features.shape)
features


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


##Using Imputer to fill missing values in Age column. 
Ageimputer = Imputer(missing_values='NaN', strategy='mean', axis=0)


# In[ ]:


features[:,[2]] = Ageimputer.fit_transform(features[:,[2]])


# In[ ]:


pd.DataFrame(features).head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()
features[:,4] = encode.fit_transform(features[:,4])
features[:,1] = encode.fit_transform(features[:,1])


# In[ ]:


pd.DataFrame(features).head()


# In[ ]:


features.shape


# In[ ]:


encode.classes_


# In[ ]:


## One hot encoding for Embarked Column values

from sklearn.preprocessing import OneHotEncoder

hotencode= OneHotEncoder(categorical_features=[4])
features= hotencode.fit_transform(features).toarray()


# In[ ]:


hotencode.get_feature_names()


# In[ ]:


pd.DataFrame(features).head()


# In[ ]:


features.shape


# In[ ]:


##One hot encoding for Sex Column Values

hotencode2= OneHotEncoder(categorical_features=[1])
features= hotencode2.fit_transform(features).toarray()


# In[ ]:


hotencode2.get_feature_names()


# In[ ]:


features.shape


# In[ ]:


final_data = pd.DataFrame(features)
final_data.head()


# In[ ]:


train_data.corr()


# In[ ]:


pd.DataFrame(features).corr()


# In[ ]:


## Vizualize correlation Matrix

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10,10))   
sns.heatmap(train_data.corr(),annot=True,fmt=".0%")
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))         ##HeatMap of Features Data
sns.heatmap(pd.DataFrame(features).corr(),annot=True,fmt=".0%")
plt.show()


# In[ ]:


plot= sns.kdeplot(features[:,6][(label[:,0] == 0)], color="Red", shade = True)
plot =sns.kdeplot(features[:,6][(label[:,0] == 1)],color ="Blue" ,shade=True)

plot.set_xlabel("Age")
plot.set_ylabel("Frequency")
plot = plot.legend(["Not Survived","Survived"])


# In[ ]:


labelValue = train_data.Survived.value_counts()    ## No.of values Label- 0 & 1 does not match that means it is unbalanced dataset
labelValue


# In[ ]:


sns.countplot(train_data.Survived ,label='count')


# # Test Data

# In[ ]:


## Validation DataSet
val_data= pd.read_csv("../input/titanic/test.csv")
PassengerId = val_data['PassengerId']
val_data.head()
#PassengerId


# In[ ]:


val_data.info()


# In[ ]:


## Validation Data has missing values in columns - Age,Fare,Cabin

val_data.describe()


# In[ ]:


## Handle Missing Data - Age,Fare,Cabin

val_data['Age'].fillna(int(val_data.Age.mean()) , inplace = True)
val_data.info()


# In[ ]:


val_data['Fare'].fillna(int(val_data.Fare.median()) ,inplace=True)
val_data.info()


# In[ ]:


## 0= not allocated cabin , 1= allocated cabin
val_data['Allocated_Cabin'] =np.where(val_data['Cabin'].isna(),0,1)
val_data.drop(columns=['Cabin'], axis=1, inplace = True)
val_data.head()


# In[ ]:


val_data['Family_Size'] = val_data['SibSp']+val_data['Parch']
val_data.head()


# In[ ]:


val_data.drop(['PassengerId','Name','Ticket','SibSp','Parch'], axis=1, inplace=True)
val_data.head()


# In[ ]:


Val_Features= val_data.iloc[:,:].values
Val_Features


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encodeVal = LabelEncoder()
Val_Features[:,4] = encodeVal.fit_transform(Val_Features[:,4])
Val_Features[:,1] = encodeVal.fit_transform(Val_Features[:,1])


# In[ ]:


Val_Features.shape


# In[ ]:


encodeVal.classes_


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

Val_hotencode= OneHotEncoder(categorical_features=[4])
Val_Features= Val_hotencode.fit_transform(Val_Features).toarray()


# In[ ]:


Val_Features.shape


# In[ ]:


Val_hotencode= OneHotEncoder(categorical_features=[1])
Val_Features= Val_hotencode.fit_transform(Val_Features).toarray()


# In[ ]:


Val_Features.shape


# In[ ]:


Val_Features


# In[ ]:


## Feature Scaling Of Test DataSet
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
Scale_Val_Features = sc.fit_transform(Val_Features)


# In[ ]:


Scale_Val_Features


# In[ ]:


# Train Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(features,label,test_size=0.2,
                                              random_state=10)


# In[ ]:


## Feature Scaling
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# # Model

# In[ ]:


def models(X_train,y_train):
    ## Logistic Regression Model
        from sklearn.linear_model import LogisticRegression
        logis= LogisticRegression(C=50)
        logis.fit(X_train, y_train)
        train_score1 =logis.score(X_train,y_train)
        test_score1 =logis.score(X_test,y_test)

        ## Random Forest Model
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=5)
        rf.fit(X_train,y_train)
        train_score2 =rf.score(X_train,y_train)
        test_score2 =rf.score(X_test,y_test)

        ## KNN
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier(n_neighbors=11)
        knc.fit(X_train,y_train)
        train_score3 =knc.score(X_train,y_train)
        test_score3 =knc.score(X_test,y_test)

        ## SVC
        from sklearn.svm import SVC
        sv = SVC()
        sv.fit(X_train, y_train)
        train_score4 =sv.score(X_train,y_train)
        test_score4 =sv.score(X_test,y_test)

        ## XgBoost
        from xgboost import XGBClassifier
        boost= XGBClassifier(learning_rate=0.01)
        boost.fit(X_train,y_train)
        train_score5 =boost.score(X_train,y_train)
        test_score5 =boost.score(X_test,y_test)

        ## Print Accuracy
        print("Logistic train score: ", train_score1, "Test score : ",test_score1)
        print("Random Forest train score: ", train_score2, "Test score : ",test_score2)
        print("KNN train score: ", train_score3, "Test score : ",test_score3)
        print("SVC train score: ", train_score4, "Test score : ",test_score4)
        print("Xgboost train score: ", train_score5, "Test score : ",test_score5)
        
        return logis,rf,knc,sv,boost


# In[ ]:


model=models(X_train,y_train)


# In[ ]:


model


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


for i in range(len(model)):
    print("Model ", i)
    cm= confusion_matrix(y_test,model[i].predict(X_test))
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]
    print(cm)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


for i in range(len(model)):
    print("Model ", i)
    Report = classification_report(y_test,model[i].predict(X_test))
    print(Report)


# In[ ]:


pred=model[3].predict(Scale_Val_Features)  ##SVC so far best model
pred


# In[ ]:


Final_Result = pd.DataFrame({ 'PassengerId': PassengerId,
                               'Survived': pred})


# In[ ]:


Final_Result.to_csv(r'ResultSubmission.csv',index=False)


# In[ ]:




