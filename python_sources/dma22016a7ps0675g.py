#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_train=pd.read_csv('../input/dma2data/train.csv')
data_train_og=data_train
data_train.head()


# In[ ]:


data_train.info()


# In[ ]:


columns=data_train.columns.values

categoric_columns=[]

for col in columns:
    if data_train[col].dtype.name=='object':
        categoric_columns.append(col)

for col in categoric_columns:
    print(col)
    print(data_train[col].value_counts())
    print('\n')


# In[ ]:


#Columns dropped because '?' is mode
to_drop=['Worker Class', 'Enrolled', 'MIC', 'MOC', 'MLU', 'Reason', 'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen', 'Fill']
data_train=data_train.drop(to_drop,1)
data_train.info()


# In[ ]:


data_train=data_train.replace(to_replace='?', value=np.nan)


columns=data_train.columns.values

categoric_columns=[]

for col in columns:
    if data_train[col].dtype.name=='object':
        categoric_columns.append(col)

for col in categoric_columns:
    print(col)
    print(data_train[col].unique())
    print(len(data_train[col].unique()))
    print('\n')


# In[ ]:


#Columns dropped due to large number of unique values
to_drop2=['ID', 'COB SELF', 'COB MOTHER', 'COB FATHER', 'Detailed'] 
data_train=data_train.drop(to_drop2,1)
data_train.info()


# In[ ]:


remaining_categoric=['Schooling', 'Married_Life', 'Cast', 'Hispanic', 'Sex', 'Full/Part', 'Tax Status', 'Summary', 'Citizen']
dataD = pd.get_dummies(data_train, columns=remaining_categoric)
dataD.info()


# In[ ]:


# Min Max Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(dataD)
dataN = pd.DataFrame(np_scaled)
dataN.columns=dataD.columns
dataN.info()


# In[ ]:


dataN.head()


# In[ ]:


dataN['Class'].value_counts()


# In[ ]:


Y=dataN['Class']
X=dataN.drop(['Class'],1)
X.info()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


from imblearn.over_sampling import SMOTE
smo=SMOTE(random_state=42)
X_train_res, Y_train_res = smo.fit_sample(X_train, Y_train.ravel())


# In[ ]:


#Logistic Regression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train_res,Y_train_res)
lg.score(X_val,Y_val)


# In[ ]:


Y_pred_LR = lg.predict(X_val)
print(confusion_matrix(Y_val, Y_pred_LR))


# In[ ]:


print(classification_report(Y_val, Y_pred_LR))


# In[ ]:


print(roc_auc_score(Y_val, Y_pred_LR))


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train_res,Y_train_res)
nb.score(X_val,Y_val)


# In[ ]:


Y_pred_NB = nb.predict(X_val)
print(confusion_matrix(Y_val, Y_pred_NB))


# In[ ]:


print(roc_auc_score(Y_val, Y_pred_NB))


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(1,15):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train_res,Y_train_res)
    acc_train = dTree.score(X_train_res,Y_train_res)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,Y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(max_depth = 6, min_samples_split=i, random_state = 42)
    dTree.fit(X_train_res,Y_train_res)
    acc_train = dTree.score(X_train_res,Y_train_res)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,Y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


dTree = DecisionTreeClassifier(max_depth=6, random_state = 42)
dTree.fit(X_train_res,Y_train_res)
dTree.score(X_val,Y_val)


# In[ ]:


Y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(Y_val, Y_pred_DT))


# In[ ]:


print(roc_auc_score(Y_val, Y_pred_DT))


# In[ ]:


data_test=pd.read_csv('../input/dma2data/test_1.csv')
data_test_og=data_test
data_test.head()


# In[ ]:


data_test.info()


# In[ ]:


data_test=data_test.replace(to_replace='?', value=np.nan)

data_test=data_test.drop(to_drop+to_drop2,1)

data_testD = pd.get_dummies(data_test, columns=remaining_categoric)


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_testD)
data_testN = pd.DataFrame(np_scaled)
data_testN.columns=data_testD.columns
data_testN.info()


# In[ ]:


X_test=data_testN
X_test


# In[ ]:


smo=SMOTE(random_state=42)
X_res, Y_res = smo.fit_sample(X, Y.ravel())

lg_test = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg_test.fit(X_res,Y_res)

Y_test=lg_test.predict(X_test)


# In[ ]:


Y_final=pd.DataFrame(Y_test).astype(int)
Y_final


# In[ ]:


test_ids=data_test_og['ID']
test_ids


# In[ ]:


final_results=pd.concat([test_ids,Y_final], axis=1).reindex()
final_results.columns=['ID', 'Class']
final_results


# In[ ]:


final_results.to_csv('2016A7PS0675G_A2_LR.csv', index=False)
final_results.info()


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(final_results)


# In[ ]:





# In[ ]:





# In[ ]:




