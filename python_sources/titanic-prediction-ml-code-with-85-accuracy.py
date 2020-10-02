# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score,f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['dataset'] = 'train'
test['dataset'] = 'test'

train = train[train['Fare'] <= 200]


labels = np.array(train['Survived'])

labels_data = pd.concat((train['PassengerId'],train['Survived']),axis=1)

passenger_id = np.array(test['PassengerId'].copy()).reshape(-1,1)

train  = train.drop('Survived',axis=1)

data = pd.concat((train,test),axis=0)

x=[]

for item in data['Name']:
    if 'Mr.' in item:
        x.append('Mr')
    elif 'Mrs.' in item:
         x.append('Mrs')
    elif 'Miss' in item:
         x.append('Miss')
    elif 'Master' in item:
         x.append('Master')
    elif 'Rev' in item:
         x.append('Rev')
    else:
         x.append('Others')
         
data['Title'] = x

for item in data['Sex']:
    for changer in data['Title']:
        if item == 'female':
            changer.replace('Mr','Mrs')
            
data_mr = data[data['Title'] == 'Mr']
data_mrs = data[data['Title'] == 'Mrs']
data_master = data[data['Title'] == 'Master']
data_miss = data[data['Title'] == 'Miss']
data_rev = data[data['Title'] == 'Rev']
data_other = data[data['Title'] == 'Others']

mr_med = data_mr['Age'].median()
other_med = data_other['Age'].median()
mrs_med = data_mrs['Age'].median()
miss_med = data_miss['Age'].median()
mas_med = data_master['Age'].median()
rev_med = data_rev['Age'].median()

data_mr['Age'] = data_mr['Age'].fillna(mr_med)
data_mrs['Age'] = data_mrs['Age'].fillna(mrs_med)
data_miss['Age'] = data_miss['Age'].fillna(miss_med)
data_master['Age'] = data_master['Age'].fillna(mas_med)
data_rev['Age'] = data_rev['Age'].fillna(rev_med)
data_other['Age'] = data_other['Age'].fillna(other_med)

data_1 = pd.concat((data_mr,data_mrs,data_miss,data_master,data_rev,data_other),axis=0)

data_1 = data_1.sort_values('PassengerId')

data=data_1

#----------------------------------------------------------------
data['Fare'][data['Fare'] == 0] = np.nan

data['Embarked'] = data['Embarked'].fillna(method='ffill')

data_class_1 = data[data['Pclass'] == 1]
data_class_2 = data[data['Pclass'] == 2]
data_class_3 = data[data['Pclass'] == 3]

cls_1_med = data_class_1['Fare'].median()
cls_2_med = data_class_2['Fare'].median()
cls_3_med = data_class_3['Fare'].median()

data_class_1['Fare'] = data_class_1['Fare'].fillna(cls_1_med)
data_class_2['Fare'] = data_class_2['Fare'].fillna(cls_2_med)
data_class_3['Fare'] = data_class_3['Fare'].fillna(cls_3_med)


data_class_1['Cabin'] = data_class_1['Cabin'].fillna('First Class Cabins')
data_class_2['Cabin'] = data_class_2['Cabin'].fillna('Second Class Cabins')
data_class_3['Cabin'] = data_class_3['Cabin'].fillna('Third Class Cabins')


data_2 = pd.concat((data_class_1,data_class_2,data_class_3),axis=0)

data_2 = data_2.sort_values('PassengerId')

data=data_2

#------------------------------------------------------       

     
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Sex'] = data['Sex'].astype('category')

data['Embarked'] = data['Embarked'].map({'C':0,'Q':1,'S':2,})
data['Embarked'] = data['Embarked'].astype('category')

data['Title'] = data['Title'].map({'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rev':5,'Others':6})
data['Title'] = data['Title'].astype('category')

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['Cabin'] = data['Cabin'].astype('category')

data['Pclass'] = data['Pclass'].astype('category')




data = data.drop(['Name','PassengerId','Ticket','SibSp','Parch'],axis=1)
data = data.drop('Cabin',axis=1)


#-----------------------------------------------------------

train_data = data[data['dataset'] == 'train']
test_data = data[data['dataset'] == 'test']

#train_data

cat_data_1 = train_data[['Sex','Embarked','Title','Pclass']]

scale_data_1 = train_data.drop(['Sex','Embarked','Title','Pclass'],axis=1)

cat_data_1 = pd.get_dummies(cat_data_1)

cat_data_1 = cat_data_1.drop('Sex_1',axis=1)

identifier_1 = list(scale_data_1['dataset'])

scale_data_1 = scale_data_1.drop('dataset',axis=1)

scaler = StandardScaler()

scaler.fit(scale_data_1)

scale_data_1 = scaler.transform(scale_data_1)

#test_data

cat_data_2 = test_data[['Sex','Embarked','Title','Pclass']]

scale_data_2 = test_data.drop(['Sex','Embarked','Title','Pclass'],axis=1)

cat_data_2 = pd.get_dummies(cat_data_2)

cat_data_2 = cat_data_2.drop('Sex_1',axis=1)

identifier_2 = list(scale_data_2['dataset'])

scale_data_2 = scale_data_2.drop('dataset',axis=1)

scaler = StandardScaler()

scaler.fit(scale_data_2)

scale_data_2 = scaler.transform(scale_data_2)

#-----------------------------------------------

train_data_1 = np.concatenate((scale_data_1,cat_data_1),axis=1)
test_data_1 = np.concatenate((scale_data_2,cat_data_2),axis=1)

train_data_1 = pd.DataFrame(train_data_1)
test_data_1 = pd.DataFrame(test_data_1)



pca_data = pd.concat((train_data_1,test_data_1),axis=0)

pca_data = pca_data.reset_index()

pca = PCA(n_components=12)
pca_data = pca.fit_transform(pca_data)
explained_variance = pca.explained_variance_ratio_

train_final = pca_data[:871,]
test_final = pca_data[871:,]


train_final = pd.DataFrame(train_final)
test_final = pd.DataFrame(test_final)

train_final = train_final.values
test_final = test_final.values


X_train,X_test,y_train,y_test = train_test_split(train_final,
                                             labels,
                                             test_size=0.25,
                                             random_state=21323)


logreg = SVC(kernel='linear')

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

print(f1_score(y_test,y_pred))

y_pred_final = logreg.predict(test_final)

passenger_id = passenger_id.ravel()

submission = pd.DataFrame({'PassengerId':passenger_id,
                           'Survived':y_pred_final})

submission.to_csv('gender_submission.csv',index=False)