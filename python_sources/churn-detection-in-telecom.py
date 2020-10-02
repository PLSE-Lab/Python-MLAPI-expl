#!/usr/bin/env python
# coding: utf-8

# In[ ]:


a


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()


# In[ ]:


data.tail()


# In[ ]:


data.sample(5)


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()
data.describe()


# In[ ]:



data.isnull().sum()


# In[ ]:


# we will convert totalchargers into numeric type

data.TotalCharges= pd.to_numeric(data.TotalCharges,errors='coerce')
data.isnull().sum()


# In[ ]:


#we will drop these 11 rows.

data.dropna(inplace=True)
data.shape


# In[ ]:


# we will drop custome id as well, iloc work on numeric number
data1=data.iloc[:,1:]
data1.dropna(inplace=True)
data1.head(12)


# In[ ]:


# we will replace no phone service to NO ,No internet service to no ,
data1['Churn'].replace(to_replace='Yes',value=1 ,inplace=True)
data1['Churn'].replace(to_replace='No',value=0 ,inplace=True)
data1['MultipleLines'].replace(to_replace='No phone service',value='No',inplace=True)
data1['OnlineSecurity'].replace(to_replace='No internet service',value='No',inplace=True)
data1['OnlineBackup'].replace(to_replace='No internet service',value='No',inplace=True)
data1['DeviceProtection'].replace(to_replace='No internet service',value='No',inplace=True)
data1['TechSupport'].replace(to_replace='No internet service',value='No',inplace=True)
data1['StreamingTV'].replace(to_replace='No internet service',value='No',inplace=True)
data1['StreamingMovies'].replace(to_replace='No internet service',value='No',inplace=True)

data1.head(15)


# In[ ]:


#heat map

import seaborn as sns
corre=data1.corr()
sns.heatmap(corre,annot=True,cmap='viridis',linewidth=3)


# In[ ]:


# creating dummy variable

df_dummy=pd.get_dummies(data1)
df_dummy.head()


# In[ ]:


#Getting correlation of churn with other variable

import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
df_dummy.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[ ]:



data1.columns


# In[ ]:





# In[ ]:


#Visualization

import matplotlib.pyplot as plt

data1['gender'].value_counts().plot.bar(color='magenta')
plt.ylabel('Number ')
plt.xlabel('Gender')
plt.show()
data1['InternetService'].value_counts().plot.bar(color='red')
plt.ylabel('Number ')
plt.xlabel('Type of Internet Service')
plt.show()
data1['SeniorCitizen'].value_counts().plot.bar(color='green')
plt.ylabel('Number ')
plt.xlabel('Senior Citizen')
plt.show()
data1['Contract'].value_counts().plot.bar(color='orange')
plt.ylabel('Number ')
plt.xlabel('Type of contract')
plt.show()
data1['PaymentMethod'].value_counts().plot.bar(color='blue')
plt.ylabel('Number ')
plt.xlabel('Type of Payment method')
plt.show()



# In[ ]:


data1['Contract'].value_counts().plot.bar(color='orange')
plt.ylabel('Number ')
plt.xlabel('Type of contract')
plt.show()
data1['PaymentMethod'].value_counts().plot.bar(color='blue')
plt.ylabel('Number ')
plt.xlabel('Type of Payment method')
plt.show()


# In[ ]:


# some more visualization



x=pd.crosstab(data1['gender'],data1['OnlineSecurity'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()

x=pd.crosstab(data1['gender'],data1['PaymentMethod'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()

x=pd.crosstab(data1['SeniorCitizen'],data1['PaymentMethod'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()


# In[ ]:



x=pd.crosstab(data1['gender'],data1['Contract'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()

x=pd.crosstab(data1['gender'],data1['StreamingMovies'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()

x=pd.crosstab(data1['PaymentMethod'],data1['Contract'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()

x=pd.crosstab(data1['gender'],data1['OnlineBackup'])
x.plot(kind = 'bar', stacked = True, figsize = (4, 4))
plt.show()


# In[ ]:


data1['Churn'].value_counts().plot.bar(color='blue')
plt.ylabel('Number ')
plt.xlabel('churn or not')
plt.show()


# In[ ]:


y = (data1['Churn'].value_counts()*100.0 /len(data1)).plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )
plt.show()

y = (data1['gender'].value_counts()*100.0 /len(data1)).plot.pie(autopct='%.1f%%', labels = ['Male', 'Female'],figsize =(5,5), fontsize = 12 )
plt.show()


y = (data1['SeniorCitizen'].value_counts()*100.0 /len(data1)).plot.pie(autopct='%.1f%%', labels = ['0', '1'],figsize =(5,5), fontsize = 12 )
plt.show()


# In[ ]:


y = (data1['PaymentMethod'].value_counts()*100.0 /len(data1)).plot.pie(autopct='%.1f%%', labels = ['Electronic check', 'Mailed check','Bank transfer (automatic)','Credit card (automatic)'],          figsize =(6,6), fontsize = 10)
plt.show()

y=(data1['Contract'].value_counts()*100.0/len(data1)).plot.pie(autopct='%.1f%%', labels = ['Month-to-month', 'Two year','One year'],figsize=(6,6),fontsize=10)
plt.show()



# In[ ]:


#model fitting

y=df_dummy['Churn'].values
X=df_dummy.drop(columns=['Churn'])

# scaling all variable to range of 0 to 1 for better accuracy

from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

#******************************************************************************************

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33 ,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


model=LogisticRegression()
output=model.fit(X_train ,y_train)

Predict=model.predict(X_test)
print("Logistc Regression Accuracy")
print(metrics.accuracy_score(y_test,Predict)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Predict))

from sklearn.metrics import confusion_matrix

print("Classification report")
print(classification_report(y_test,Predict))
print(output)
print("Confusion matrix")
print(confusion_matrix(y_test,Predict))


# In[ ]:


# Decision tree

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
output=model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("model accuracy")
print(metrics.accuracy_score(y_test,prediction)*100)
print("Area under curve", )
print(roc_auc_score(y_test,prediction))
print("Classification report")
print(classification_report(y_test,prediction))


# In[ ]:


# Random forest

from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33 ,random_state=10)
rf =RandomForestClassifier(n_estimators=70,max_depth=2,random_state=10,criterion='gini')
output1=rf.fit(X_train,y_train)
Predict_out=model.predict(X_test)
print("Random forest Accuracy")
print(metrics.accuracy_score(y_test,Predict_out)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Predict_out))
print ("\n Classification report : \n",classification_report(y_test,Predict_out))


# In[ ]:


# support vector machine

from sklearn.svm import SVC
model_svm= SVC(kernel='linear')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)
model_svm=SVC(gamma='auto')
output1=model_svm.fit(X_train,y_train)
Prediction1=model_svm.predict(X_test)
print("Support vector machine Accuracy")
print(metrics.accuracy_score(y_test,Prediction1)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Prediction1))
print("Classification report")
print(classification_report(y_test,Prediction1))



# In[ ]:


# ADABOOST

from sklearn.ensemble import AdaBoostClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)
ABC =AdaBoostClassifier()
output=ABC.fit(X_train,y_train)
Prediction1=ABC.predict(X_test)
print(" Ada Boost Classifier Accuracy")
print(metrics.accuracy_score(y_test,Prediction1)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Prediction1))
print("Confusion matrix")
print(confusion_matrix(y_test,Predict))
print("Classification report")
print(classification_report(y_test,Prediction1))
print(output1)


# In[ ]:


# XGBoost


from xgboost import XGBClassifier
xgb=XGBClassifier()
output=xgb.fit(X_train,y_train)
prediction_xg=xgb.predict(X_test)
print("model accuracy")
print(metrics.accuracy_score(y_test,prediction_xg)*100)
print("Area under curve", )
print(roc_auc_score(y_test,prediction_xg))
print("Classification report")
print(classification_report(y_test,Prediction1))

