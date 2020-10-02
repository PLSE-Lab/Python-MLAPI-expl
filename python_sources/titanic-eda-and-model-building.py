#!/usr/bin/env python
# coding: utf-8

# <h4> Titanic Data - EDA and Model Building
DESCRIPTIVE ABSTRACT:

The titanic data describes the survival status of individual passengers on the
Titanic.

The titanic data does not contain information for the crew, but it does contain
actual and estimated ages for almost 80% of the passengers.

VARIABLE DESCRIPTIONS:

Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival - Survival (0 = No; 1 = Yes)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare (British pound)
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


# Import Data

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train['Source'] = 'Train'
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test['Source'] = 'Test'
df = pd.concat([df_train,df_test],0)
df.head()


# ---

# Number of rows and columns

# In[ ]:


print (df.shape)


# In[ ]:


df.info()


# <h5> Missing Ratio 

# In[ ]:


mr = (df.isna().sum()/len(df))*100
mr.sort_values(ascending=False)


# ---

# <h5> Cabin 

# Create a column 'Cabin_Status' : 0 where Cabin is NotAvailable and 1 otherwise

# In[ ]:


df['Cabin'].fillna('NotAvailable',inplace=True)


# In[ ]:


df["Cabin_Status"] = df['Cabin'].apply(lambda x : 0 if x=='NotAvailable' else 1)


# In[ ]:


df.head()


# In[ ]:


sns.barplot(df['Cabin_Status'] , df['Survived'])
plt.show()


# <h5> We can see that people without Cabin have a very low rate of Survival compared to people with Cabin

# In[ ]:


sns.barplot(df['Embarked'], df['Survived'] , hue=df['Sex'])
plt.show()


# <h5> The column 'Embarked' is not influencing the Survival rate as Sex and Pclass explain the same thing|

# Dropping Columns

# In[ ]:


df.drop(['Cabin','Ticket','Embarked'],1,inplace=True)


# In[ ]:


df.head()


# ---

# <h5>Married

# Married status is 1 if name has 'Mr. or Mrs' and 0 otherwise

# In[ ]:


df['Married']= np.NAN
df['Married'][df['Name'].str.contains('Mr.|Mrs.')] = 1
df['Married'][~(df['Name'].str.contains('Mr.|Mrs.'))] = 0
df['Married']=df['Married'].astype(int)


# Dropping Name 

# In[ ]:


df.drop('Name',1,inplace=True)


# ---

# <h5>Age

# In[ ]:


sns.distplot(df['Age'].dropna())


# In[ ]:


df['Age'].describe()


# In[ ]:


sns.barplot(df['Married'], df['Age'], hue=df['Pclass'] )
plt.show()


# <h5> The Married status and Pclass are important in determining the age of the person ,
#     we will use this to fill  the missing values in the Age column

# In[ ]:


def impute_age(cols):
    married = cols[0]
    pclass = cols[1]
    if married==0 and pclass==1:
        return 32
    elif married==0 and pclass==2:
        return 20
    elif married==0 and pclass==3:
        return 15
    elif married==1 and pclass==1:
        return 42
    elif married==1 and pclass==2:
        return 32
    elif married==1 and pclass==3:
        return 30


# In[ ]:


df['Age'][df['Age'].isna()] = df[['Married','Pclass']].apply(impute_age,axis=1)


# <h5>Senior Citizen or Not

# If age is greater than 60 classify them as Senior Citizen

# In[ ]:


df['Senior_Citizen'] = df['Age'].apply(lambda x : 1 if x>60 else 0)


# In[ ]:


sns.barplot(df['Senior_Citizen'] , df['Survived'])


# <h5> We can see Senior Citizens have lower survival rate then others 

# ----

# <h5> Updated Missing Ratio

# In[ ]:


mr = (df.isna().sum()/len(df))*100
mr.sort_values(ascending=False)


# ---

# In[ ]:


df.head()


# ---

# <h5>Family Members

# We can combine Parch and Sibsp to get the number of Family Members the person has 

# In[ ]:


df['Family_members'] = df['SibSp']  + df['Parch']


# ---

# Encoding Sex & PClass column

# In[ ]:


df['Sex'] = df['Sex'].map({'male':1 , 'female':0})


# In[ ]:


df.head()


# In[ ]:


df['Pclass'] = df['Pclass'].astype('object')


# In[ ]:


dummies = pd.get_dummies(df[['Pclass']],drop_first=True)
dummies.head(2)


# In[ ]:


df.drop(['Pclass'],1,inplace=True)


# In[ ]:


df = pd.concat([df,dummies],1)


# In[ ]:


df.head()


# ----

# <h5> Standardize

# In[ ]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
df[['Age','Fare']] = scale.fit_transform(df[['Age','Fare']])


# In[ ]:


df.head()


# ---

# <h5>Seperate Train and Test 

# In[ ]:


df_train = df[df['Source']=='Train']
df_test = df[df['Source']=='Test']


# In[ ]:


df_train.drop(['PassengerId' , 'Source'],1,inplace=True)
df_test.drop('Source',1,inplace=True)


# ---

# In[ ]:


df_train['Survived'] = df_train['Survived'].astype(int)


# In[ ]:


X = df_train.drop('Survived',1)
y = df_train['Survived']


# ---

# <h5>Train and Validation Split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train , X_val , y_train , y_val = train_test_split(X,y,test_size=0.30,random_state=123)


# ---

# <h5>PCA

# In[ ]:


from sklearn.decomposition import PCA
import numpy as np


# In[ ]:


pca = PCA()
pca = pca.fit(X_train)


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.show()


# ![](http://)We can see that 6 features can explain almost 98% of the variation 

# In[ ]:


pca_final = PCA(6)
X_train_pca = pca_final.fit_transform(X_train)
X_val_pca = pca_final.transform(X_val)


# ---

# <h5>Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression()
lr = lr.fit(X_train , y_train)


# In[ ]:


y_pred_lr = lr.predict(X_val)


# In[ ]:


from sklearn.metrics import classification_report , confusion_matrix

from sklearn.preprocessing import binarize


# In[ ]:


y_pred_prob_yes = lr.predict_proba(X_val)


# In[ ]:


y_pred_lr = binarize(y_pred_prob_yes , 5/10)[:,1]


# In[ ]:


confusion_matrix(y_val,y_pred_lr)


# In[ ]:


print(classification_report(y_val,y_pred_lr))


# <h5> Classification Accuracy for Logistic Regression : 0.82

# ---

# <h5> KNN

# In[ ]:


from collections import Counter


# In[ ]:


Counter(y_train)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(3)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_val)
confusion_matrix(y_val,y_pred_knn)


# In[ ]:


print(classification_report(y_val , y_pred_knn))


# <h5>Classification Accuracy for KNN : 0.81

# ---

# <h5>Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier()


# Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {
    
    'n_estimators':[10,20,30],
    'criterion': ['gini','entropy'],
    'max_depth': [10,15,20,25],
    'min_samples_split' : [5,10,15],
    'min_samples_leaf': [2,5,7],
    'random_state': [42,135,777],
    'class_weight': ['balanced' ,'balanced_subsample']
}


# In[ ]:


rf_tuned = GridSearchCV(estimator=rf , param_grid=param_grid , n_jobs=-1)


# In[ ]:


rf_tuned.fit(X_train , y_train)


# In[ ]:


rf_tuned.best_params_


# In[ ]:


rff = RandomForestClassifier(**rf_tuned.best_params_)
rff.fit(X_train , y_train)
y_pred_rff = rff.predict(X_val)
print(classification_report(y_val,y_pred_rff))


# In[ ]:


confusion_matrix(y_val , y_pred_rff)


# <h5> Classification Accuracy for Tuned Random Forest : 0.83

# ---

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_val)
confusion_matrix(y_val,y_pred_rf)


# In[ ]:


print(classification_report(y_val , y_pred_rf))


# <h5>Classification Accuracy for Random Forest : 0.82

# ---

# <h4>Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import binarize


# In[ ]:


ada_boost = GradientBoostingClassifier()
ada_boost.fit(X_train , y_train)


# In[ ]:


y_pred_ada = ada_boost.predict(X_val)
y_pred_ada_prob = ada_boost.predict_proba(X_val)


# In[ ]:


from sklearn.preprocessing import binarize
for i in range(5,9):
    cm2=0
    y_pred_prob_yes=ada_boost.predict_proba(X_val)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_val,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    


# In[ ]:


y_pred_ada_prob = binarize(y_pred_ada_prob , 7/10)[:,1]


# In[ ]:


confusion_matrix(y_val , y_pred_ada_prob)


# In[ ]:


print(classification_report(y_val,y_pred_ada_prob))


# <h5> Classification Accuracy for Gradient Boosting : 0.86

# ---

# The following code is for Kaggle submission

# ---

# Submission Predictions

# In[ ]:


df_test.head(2)


# In[ ]:


X_test = df_test.drop('Survived',1)


# In[ ]:


pass_id = X_test['PassengerId']
X_test.drop('PassengerId',1,inplace=True)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_test['Fare'][X_test['Fare'].isna()] = 5


# ---

# **********************

# In[ ]:


X_train.head(2)


# In[ ]:


X_test.head(2)


# In[ ]:


X_test[['Age','Fare']] = scale.transform(X_test[['Age','Fare']])


# In[ ]:


y_pred_final = ada_boost.predict_log_proba(X_test)
y_pred_final = binarize(y_pred_final , 7/10)[:,1]


# In[ ]:


y_pred_sub_df = pd.DataFrame(y_pred_final , columns=['Survived'])
y_pred_sub_df['Survived'] = y_pred_sub_df['Survived'].astype(int)
y_pred_sub_df.head(2)


# In[ ]:


pass_id_df = pd.DataFrame(pass_id , columns=['PassengerId'])
pass_id_df.head(2)


# In[ ]:


submission_df = pd.concat([pass_id_df , y_pred_sub_df],1)
submission_df.head()


# In[ ]:


submission_df.shape


# In[ ]:


submission_df.to_csv('Titanic_Submissions_GB.csv' , index=False)


# ---

# In[ ]:




