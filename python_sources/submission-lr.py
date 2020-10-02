#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Download all data libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Download all model Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from numpy import loadtxt
from xgboost import XGBClassifier


# In[ ]:


#Read Data
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')


# In[ ]:


sns.heatmap(df_train.isnull())


# In[ ]:


sns.heatmap(df_test.isnull())


# In[ ]:


df_train.info()


# In[ ]:


df_train.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


df_test.info()


# In[ ]:


test_index = df_test['Unnamed: 0']


# In[ ]:


df_test.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


pd.set_option('display.max_columns',100)


# In[ ]:


df_train.describe()


# In[ ]:


#sns.pairplot(df_train)


# In[ ]:


X=df_train.drop('Class',axis=1)
y=df_train['Class']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=162)


# In[ ]:


LR=LogisticRegression()


# In[ ]:


LR.fit(X_train, y_train)


# In[ ]:


print(classification_report(y_test, LR.predict(X_test)))


# In[ ]:


X.head()


# In[ ]:


df2=pd.get_dummies(df_train['V2'],drop_first=True)
df3=pd.get_dummies(df_train['V3'],drop_first=True)
df4=pd.get_dummies(df_train['V4'],drop_first=True)
df5=pd.get_dummies(df_train['V5'],drop_first=True)
df7=pd.get_dummies(df_train['V7'],drop_first=True)
df8=pd.get_dummies(df_train['V8'],drop_first=True)
df9=pd.get_dummies(df_train['V9'],drop_first=True)
df16=pd.get_dummies(df_train['V16'],drop_first=True)


# In[ ]:


df2.head()


# In[ ]:


df2.columns = ['V2_1', 'V2_2','V2_3','V2_4','V2_5','V2_6','V2_7','V2_8','V2_9','V2_10','V2_11']
df3.columns = ['V3_1', 'V3_2']
df4.columns = ['V4_1', 'V4_2','V4_3']
df5.columns = ['V5_1']
df7.columns = ['V7_1']
df8.columns = ['V8_1']
df9.columns = ['V9_1', 'V9_2']
df16.columns = ['V16_1', 'V16_2','V16_3']


# In[ ]:


df2.head()


# In[ ]:


df_train_simp=df_train.drop(['V2','V3','V4','V5','V7','V8','V9','V16'],axis=1)


# In[ ]:


df_train_simp.head()


# In[ ]:


df_train_simp=pd.concat([df_train_simp,df3, df4,df2,df5,df7,df8,df9,df16], axis=1)


# In[ ]:


df_train_simp.head()


# In[ ]:


#The simplified train data
#df_train_simp.to_csv('traindata.csv',index=False)


# In[ ]:


g2=pd.get_dummies(df_test['V2'],drop_first=True)
g3=pd.get_dummies(df_test['V3'],drop_first=True)
g4=pd.get_dummies(df_test['V4'],drop_first=True)
g5=pd.get_dummies(df_test['V5'],drop_first=True)
g7=pd.get_dummies(df_test['V7'],drop_first=True)
g8=pd.get_dummies(df_test['V8'],drop_first=True)
g9=pd.get_dummies(df_test['V9'],drop_first=True)
g16=pd.get_dummies(df_test['V16'],drop_first=True)


# In[ ]:


g2.columns = ['V2_1', 'V2_2','V2_3','V2_4','V2_5','V2_6','V2_7','V2_8','V2_9','V2_10','V2_11']
g3.columns = ['V3_1', 'V3_2']
g4.columns = ['V4_1', 'V4_2','V4_3']
g5.columns = ['V5_1']
g7.columns = ['V7_1']
g8.columns = ['V8_1']
g9.columns = ['V9_1', 'V9_2']
g16.columns = ['V16_1', 'V16_2','V16_3']


# In[ ]:


df_test_simp=df_test.drop(['V2','V3','V4','V5','V7','V8','V9','V16'],axis=1)


# In[ ]:


df_test_simp=pd.concat([df_test_simp,g3, g4,g2,g5,g7,g8,g9,g16], axis=1) 
#simplified test data
#The simplified train data
#df_train_simp.to_csv('traindata.csv',index=False)


# In[ ]:


df_test_simp.head()


# In[ ]:


#Simplified data set for train
A=df_train_simp.drop('Class',axis=1)
b=df_train_simp['Class']


# In[ ]:


df_train_simp.plot.scatter(x='V1', y='V6')


# In[ ]:


df_train_simp.plot.scatter(x='V12', y='V6')


# In[ ]:


error_rate = [] 

for k in range(1,170):
    
    X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.3, random_state=k)
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)
    pred_k = LogReg.predict(X_test)
    
    error_rate.append(np.mean(pred_k != y_test))


# In[ ]:


plt.figure(figsize=(50,15))
plt.style.use('ggplot')

plt.plot(range(1,170), error_rate, color='blue',
        linestyle='dashed', marker='o',
        markerfacecolor='red')

plt.title('Error Rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


#Normalizing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,30))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Scrapped thiis idea as accuracy was lesser


# In[ ]:


#Grid Search
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
confusion_matrix(y_test,y_pred_acc)


# In[ ]:


#print(LR.feature_importances_)
#feat_importance=pd.Series(J.feature_importances_,index=A.columns)
#feat_importance.nlargest(33).plot(kind='barh')
#plt.rcParams['figure.figsize']=[40,25]
#plt.show()


# In[ ]:


#This is done a these coloms have barely any importande
#Function was applied on RandomForestClassifier to come to this conclusion
E=df_train_simp.drop(['V2_8','V2_11','V16_1','V16_2','V16_3','V2_5','V2_2','Class'],axis=1)
f=df_train_simp['Class']


# In[ ]:


E.head()


# In[ ]:


f.head()


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(E,f,test_size=0.3,random_state=162)
LR2=LogisticRegression(C=25,penalty='l2')
LR2.fit(X_train,y_train)
LR2.score(X_test,y_test)


# In[ ]:


#Split the data to check accuracy
X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.3, random_state=162)


# In[ ]:


LR1=LogisticRegression(C=25,penalty='l2')


# In[ ]:


LR1.fit(X_train,y_train)


# In[ ]:


LR1.score(X_test,y_test)


# In[ ]:


#Simplified data set for fit
C=df_train_simp.drop('Class',axis=1)
d=df_train_simp['Class']


# In[ ]:


LR2.fit(C,d)


# In[ ]:


pred = LR2.predict_proba(df_test_simp)
pred.shape


# In[ ]:


pred


# In[ ]:


result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred[:,1])
result.head()


# In[ ]:


result.to_csv('output.csv', index=False)


# In[ ]:




