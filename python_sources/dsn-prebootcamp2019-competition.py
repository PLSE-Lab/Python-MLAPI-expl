#!/usr/bin/env python
# coding: utf-8

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


#importing all library needed
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score
from pandas import Series


# In[ ]:


#import Train data
dataset = pd.read_csv(r'C:\Users\DELL\Desktop\DSN_BootCamp2019\train.csv')
dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


#checking for null values
dataset.isnull().sum()


# In[ ]:


dataset.Qualification.value_counts()


# In[ ]:


#Getting mode for null column
mod=dataset.Qualification.mode()
mod=dataset.Qualification.mode().iloc[0]
print (mod)


# In[ ]:


#filling missing data
dataset['Qualification'].fillna('mod', inplace = True)
#dataset.isnull().sum()


# In[ ]:


#Getting age from year of birth column
import datetime
now = datetime.datetime.now() 
dataset['age'] = now.year - dataset['Year_of_birth']
#Getting year of experience from year of recruitment column
dataset['Year_of_Experince'] = now.year - dataset['Year_of_recruitment']
#dataset.head(3)


# In[ ]:


#categorise age into young and old
dataset['young_age'] = np.where((dataset['age']<40), 1,0)
dataset['old_age'] = np.where((dataset['age']>40), 1,0)
#dataset.head()


# In[ ]:


#categorise last performance score into low and high
#taset['medium_perf'] = np.where((dataset['Last_performance_score']>0.4)&(dataset['Last_performance_score']<0.6), 1,0)
dataset['low_perf'] = np.where((dataset['Last_performance_score']<0.51), 1,0)
dataset['high_perf'] = np.where((dataset['Last_performance_score']>0.5), 1,0)
#dataset.head()


# In[ ]:


# dataset.age.head()


# In[ ]:


#feature Engineering
#gender_dum = pd.get_dummies(dataset['Gender'])
#gender_dum.head()
dataset=pd.get_dummies(dataset,columns=['Gender','Foreign_schooled','Past_Disciplinary_Action','Previous_IntraDepartmental_Movement'])
#dataset=pd.concat([dataset, df_dummies], axis=1)
#dataset.drop(columns=['EmployeeNo','Gender_Female','Past_Disciplinary_Action_No', 'Foreign_schooled_No', 'Foreign_schooled','Past_Disciplinary_Action','Previous_IntraDepartmental_Movement','Gender','Qualification','Year_of_recruitment','Division', 'Year_of_birth', 'Channel_of_Recruitment', 'State_Of_Origin','Marital_Status','No_of_previous_employers','Previous_IntraDepartmental_Movement_No'],inplace=True)
dataset=dataset.drop(columns=['EmployeeNo','Qualification','Last_performance_score','Gender_Female','Past_Disciplinary_Action_No', 'Foreign_schooled_No','Division','Channel_of_Recruitment','Marital_Status','Previous_IntraDepartmental_Movement_No','State_Of_Origin','No_of_previous_employers','Year_of_recruitment','Year_of_birth'])
#dataset.drop(columns=['Year_of_birth'],inplace=True)
dataset.info()


# In[ ]:


#function definition for Evalution
def evaluate(y_test, pred):
    from sklearn.metrics import f1_score, classification_report, accuracy_score
    print('F1_SCORE: ', f1_score(y_test, pred))
    print('ACCURACY SCORE: ', accuracy_score(y_test, pred))
    
    print('Distinct counts in the test set: ')
    print(y_test.value_counts())
    
    print('Distinct counts in the prediction: ')
    predictions = pd.DataFrame(pred)
    print(predictions[0].value_counts())


# In[ ]:


# dataset.medium_perf.value_counts()
# #dataset.low_perf.value_counts()
# dataset.high_perf.value_counts()


# In[ ]:


#printing features correlation chart
plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#sns.heatmap(all_data.loc['train'].corr(), annot=True)
plt.show()


# In[ ]:


#label visualisation
pos = dataset[dataset["Promoted_or_Not"] == 1].shape[0]
neg = dataset[dataset["Promoted_or_Not"] == 0].shape[0]
print("Positive examples = {}".format(pos))
print("Negative examples = {}".format(neg))
print("Proportion of positive to negative examples = {:.2f}%".format((pos / neg) * 100))
sns.countplot(dataset["Promoted_or_Not"])
plt.xticks((0, 1), ["Not Promoted", "Promoted"])
plt.xlabel("Promoted_or_Not")
plt.ylabel("Count")
plt.title("Class counts");


# In[ ]:


#data spliting
y = dataset.Promoted_or_Not
x=dataset.drop(columns=['Promoted_or_Not'],axis =1)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y,random_state=27)
print (X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


#building my model for logistics regression
from sklearn.linear_model import LogisticRegression
 lm = LogisticRegression()
model = lm.fit(X_train, y_train)
#predictions = lm.predict(X_test)
#lm.fit(x, y)
#lm.predict(X_train)
#pred=lm.predict(X_test)

#
evaluate(y_test, predictions)


# In[ ]:


#building my model for XGBOOST
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate(y_test,y_pred)
#predictions = [round(value) for value in y_pred]


# In[ ]:


#building my model for Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10).fit(X_train,y_train)
rfc.predict(X_test)
pred = rfc.predict(X_test)
evaluate(y_test,pred)


# In[ ]:


#importing my Test data
test1 = pd.read_csv(r'C:\Users\DELL\Desktop\DSN_BootCamp2019\test.csv')
test1.head()
testy= test1.copy()


# In[ ]:


#checking for null values
test1.isnull().sum()


# In[ ]:


#Getting mode for null column
mod=test1.Qualification.mode()
mod=test1.Qualification.mode().iloc[0]
print (mod)


# In[ ]:


#filling the null values
test1['Qualification'].fillna('mod', inplace = True)


# In[ ]:


test1.drop(columns=['EmployeeNo'], inplace = True)
test1.info()


# In[ ]:


#feature engineering
import datetime
now = datetime.datetime.now() 
test1['age'] = now.year - test1['Year_of_birth']
test1['Year_of_Experince'] = now.year - test1['Year_of_recruitment']
#test1.head(3)


# In[ ]:


#featuring engineering
test1=pd.get_dummies(test1,columns=['Gender','Foreign_schooled','Past_Disciplinary_Action','Previous_IntraDepartmental_Movement'])
test1=test1.drop(columns=['Gender_Female','Past_Disciplinary_Action_No', 'Foreign_schooled_No','Division','Qualification','Channel_of_Recruitment','Marital_Status','Previous_IntraDepartmental_Movement_No','State_Of_Origin','No_of_previous_employers','Year_of_recruitment','Year_of_birth'])
#test1=test1.drop(columns=['Previous_IntraDepartmental_Movement_Yes','Past_Disciplinary_Action_Yes','Foreign_schooled_Yes','Training_score_average','Gender_Male','Trainings_Attended','Targets_met','Previous_Award','Qualification','Last_performance_score','Gender_Female','Past_Disciplinary_Action_No', 'Foreign_schooled_No','Division','Channel_of_Recruitment','Marital_Status','Previous_IntraDepartmental_Movement_No','State_Of_Origin','No_of_previous_employers','Year_of_recruitment','Year_of_birth'])

test1.info()


# In[ ]:


# testy['Promoted_or_Not']= predictions

# submission = testy[['EmployeeNo', 'Promoted_or_Not']]
# submission.to_csv('submit.csv', index = False)


# In[ ]:


#Making prediction on the test data
pred=rfc.predict(test1)
y_pred = model.predict(test1)
 dataset=dataset.drop(columns=['EmployeeNo','low_perf','high_perf','Previous_IntraDepartmental_Movement_Yes','Past_Disciplinary_Action_Yes','Foreign_schooled_Yes','Training_score_average','Gender_Male','Trainings_Attended','Targets_met','Previous_Award','Qualification','Last_performance_score','Gender_Female','Past_Disciplinary_Action_No', 'Foreign_schooled_No','Division','Channel_of_Recruitment','Marital_Status','Previous_IntraDepartmental_Movement_No','State_Of_Origin','No_of_previous_employers','Year_of_recruitment','Year_of_birth'])


# In[ ]:


#getting prediction result into CSV
testy['Promoted_or_Not']= pred

submission = testy[['EmployeeNo', 'Promoted_or_Not']]
submission.to_csv('submit1.csv', index = False)

