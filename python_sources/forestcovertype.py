#!/usr/bin/env python
# coding: utf-8

# ## Forest Cover Type Prediction

# ### Import necessary packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


# ### Read input files

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ### About the data

# In[ ]:


train_data.shape


# In[ ]:


train_data.info()


# In[ ]:


train_data.head()


# In[ ]:


train_data.dtypes


# In[ ]:


test_data.shape


# In[ ]:


test_data.info()


# In[ ]:


test_data.head()


# In[ ]:


test_data.dtypes


# ### Is there any constant features?

# In[ ]:


# check constant columns
remove_cols = []
for col in train_data.columns:
    if train_data[col].std() == 0: 
        remove_cols.append(col)
print remove_cols


# In[ ]:


remove_cols1 = []
for col in test_data.columns:
    if test_data[col].std() == 0: 
        remove_cols1.append(col)
print(remove_cols1)


# #### It shows that, train_data have some constant features, but test_data have not. So we shouldn't remove those features

# ### Checking Null Values 

# In[ ]:


train_data.isnull().any().values


# In[ ]:


test_data.isnull().any().values


# ### Wilderness_Area

# In[ ]:


'''
Create a new column 'Wilderness_Area' 

1 for Wilderness_Area1
2 for Wilderness_Area2
3 for Wilderness_Area3
4 for Wilderness_Area4

'''

for i in train_data.index:
    if train_data['Wilderness_Area1'][i] == 1:
        train_data.set_value(i,'Wilderness_Area',1)
    elif train_data['Wilderness_Area2'][i] == 1:
        train_data.set_value(i,'Wilderness_Area',2)
    elif train_data['Wilderness_Area3'][i] == 1:
        train_data.set_value(i,'Wilderness_Area',3)
    elif train_data['Wilderness_Area4'][i] == 1:
        train_data.set_value(i,'Wilderness_Area',4)


# In[ ]:


for i in test_data.index:
    if test_data['Wilderness_Area1'][i] == 1:
        test_data.set_value(i,'Wilderness_Area',1)
    elif test_data['Wilderness_Area2'][i] == 1:
        test_data.set_value(i,'Wilderness_Area',2)
    elif test_data['Wilderness_Area3'][i] == 1:
        test_data.set_value(i,'Wilderness_Area',3)
    elif test_data['Wilderness_Area4'][i] == 1:
        test_data.set_value(i,'Wilderness_Area',4)


# In[ ]:


#Wilderness_Area 
plt.figure(figsize=(10,6))
sns.countplot(x="Wilderness_Area", data=train_data)
plt.ylabel('Count', fontsize=10)
plt.xlabel('Wilderness_Area', fontsize=10)
plt.xticks(rotation='vertical')
plt.title("Frequency of Wilderness_Area", fontsize=14)
plt.show()


# In[ ]:


#Wilderness_Area for test_data
plt.figure(figsize=(10,6))
sns.countplot(x="Wilderness_Area", data=test_data)
plt.ylabel('Count', fontsize=10)
plt.xlabel('Wilderness_Area', fontsize=10)
plt.xticks(rotation='vertical')
plt.title("Frequency of Wilderness_Area", fontsize=14)
plt.show()


# ### Soil Type

# In[ ]:


all_columns = train_data.columns
soil_type = []
for each in all_columns:
    if each.startswith(("Soil_Type")):
        soil_type.append(each)
print(soil_type)


# In[ ]:


for i in train_data.index:
    for ind,each in enumerate(soil_type,1):
        if train_data[each][i] == 1:
            train_data.set_value(i,'Soil_Type',ind)


# In[ ]:


len(train_data.Soil_Type)


# In[ ]:


#Soil_Type 
plt.figure(figsize=(10,6))
sns.countplot(x="Soil_Type", data=train_data)
plt.ylabel('Count', fontsize=10)
plt.xlabel('Soil_Type', fontsize=10)
plt.xticks(rotation='vertical')
plt.title("Frequency of Soil_Type", fontsize=14)
plt.show()


# In[ ]:


for i in test_data.index:
    for ind,each in enumerate(soil_type,1):
        if test_data[each][i] == 1:
            test_data.set_value(i,'Soil_Type',ind)


# In[ ]:


Counter(test_data.Soil_Type)


# In[ ]:


#Soil_Type 
plt.figure(figsize=(10,6))
sns.countplot(x="Soil_Type", data=test_data)
plt.ylabel('Count', fontsize=10)
plt.xlabel('Soil_Type', fontsize=10)
plt.xticks(rotation='vertical')
plt.title("Frequency of Soil_Type", fontsize=14)
plt.show()


# In[ ]:


train_data.drop(soil_type,axis=1,inplace=True)


# In[ ]:


train_data.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis=1,inplace=True)


# In[ ]:


test_data.drop(soil_type,axis=1,inplace=True)


# In[ ]:


test_data.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis=1,inplace=True)


# In[ ]:


train_data_without_id = train_data.copy()
train_data_without_id.drop(['Id'],axis=1,inplace=True)


# In[ ]:


test_data_without_id = test_data.copy()
test_data_without_id.drop(['Id'],axis=1,inplace=True)


# In[ ]:


#finding Correlation
import seaborn as sns
corr = train_data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True)


# ##### There is no highly correlated columns

# ### Class imbalance checking

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x="Cover_Type", data=train_data)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Cover_Type', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Class Imbalance Checking", fontsize=15)
plt.show()


# ##### There is no class imbalance

# ### -----------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Train_Test_Split

# In[ ]:


X = train_data_without_id.drop(['Cover_Type'],axis=1)


# In[ ]:


X.columns


# In[ ]:


y = train_data_without_id['Cover_Type']


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ### Logistic Regression

# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
logestic = LogisticRegression()
logestic.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(logestic,X_train,y_train, cv=kfold, scoring=scoring)
acc_log = results.mean()
log_std = results.std()
acc_log


# ### KNN

# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(knn,X_train,y_train, cv=kfold, scoring=scoring)
acc_knn = results.mean()
knn_std = results.std()
acc_knn


# ### SVC

# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
svc = SVC(kernel = 'rbf')
svc.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(svc,X_train,y_train, cv=kfold, scoring=scoring)
acc_svc = results.mean()
svc_std = results.std()


# ### Decision Tree

# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
dTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dTree.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(dTree,X_train,y_train, cv=kfold, scoring=scoring)
acc_dt = results.mean()
dt_std = results.std()
acc_dt


# ### RandomForest

# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
randomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
randomForest.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(randomForest,X_train,y_train, cv=kfold, scoring=scoring)
acc_rf = results.mean()
rf_std = results.std()
acc_rf


# ### Naive_Bayes

# In[ ]:


nb = GaussianNB()
nb.fit(X_train,y_train)

scoring = 'accuracy'
results = cross_val_score(nb,X_train,y_train, cv=kfold, scoring=scoring)
acc_nb = results.mean()
nb_std = results.std()
acc_nb


# In[ ]:


result_df = pd.DataFrame({
                            'Model': ['LogisticRegression','SVC','KNN', 'Decision Tree','Random Forest','Naive_Bayes'],
                            'Score': [acc_log, acc_svc, acc_knn, acc_dt, acc_rf,acc_nb]
                         })
result_df.sort_values(by='Score', ascending=False)


# In[ ]:


ypred = randomForest.predict(X_test)


# In[ ]:


accuracy_score(y_test, ypred)


# In[ ]:


test_scaled = sc.transform(test_data_without_id)


# In[ ]:


y_pred_final = randomForest.predict(test_scaled)


# In[ ]:


Counter(y_pred_final)


# In[ ]:


submission_file = pd.DataFrame({"Id":test_data.Id,"Target":y_pred_final})


# In[ ]:


submission_file.to_csv("Forest_coverType_prediction_submission.csv",index=False)


# In[ ]:




