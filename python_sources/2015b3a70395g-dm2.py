#!/usr/bin/env python
# coding: utf-8

# # Datamining Assignment 2

# In[ ]:


import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[ ]:


data_orig = pd.read_csv("../input/train.csv")
data = data_orig


# ## Data Preprocessing

# In[ ]:


data = data.replace({'?':np.nan})


# In[ ]:


data.isnull().sum(axis = 0)


# In[ ]:


nan_col = ['Worker Class','Fill','Teen','PREV','Live','MOVE','REG','MSA','State','Area','Reason','MLU','MOC','MIC','Enrolled']


# In[ ]:


data = data.drop(nan_col, axis = 1)


# In[ ]:


pd.set_option('display.max_columns', 60)
data.head()


# In[ ]:


null_columns = data.columns[data.isna().any()]
null_columns


# In[ ]:


for c in null_columns:
    data[c] = data[c].fillna(data[c].mode()[0])


# In[ ]:


for col in data.columns:
    print(col , data[col].unique())
    print(" ")


# In[ ]:


categorical_col = ['COB SELF','COB MOTHER','COB FATHER','Detailed']
data = data.drop(categorical_col, axis = 1)
data.head()


# In[ ]:


new_data = pd.get_dummies(data, columns=['Married_Life','Cast', 'Hispanic', 'Sex','Full/Part', 'Tax Status',
                                        'Summary', 'Citizen','Schooling'])


# In[ ]:


X_train = new_data
X_train  = X_train.drop(['Class'], axis=1)
y_train = new_data['Class']


# ### Feature Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


#Using RandomForestClassifier for feature selection
plt.figure(figsize=(60,60))
model = AdaBoostClassifier(random_state=42)
model = model.fit(X_train,y_train)
features = X_train.columns
importances = model.feature_importances_
impfeatures_index = np.argsort(importances)
#print([features[i] for i in impfeatures_index])
sns.barplot(x = [importances[i] for i in impfeatures_index], y = [features[i] for i in impfeatures_index])
plt.xlabel('value', fontsize=32)
plt.ylabel('parameter', fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)
plt.show()


# In[ ]:


#Selecting top features based on their importance according to the above graph
impfeatures = features[impfeatures_index[-25:]]
X_train = X_train[[features for features in impfeatures]]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)


# #### oversampling

# In[ ]:


from imblearn.over_sampling import RandomOverSampler


# In[ ]:


ros = RandomOverSampler(random_state=0)
X_resampled1, y_resampled1 = ros.fit_resample(X_train, y_train)


# ## Applying Classification Algorithms

# ### Random Forest

# In[ ]:


score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42,class_weight='balanced')
    rf.fit(X_resampled1,y_resampled1)
    sc_train = rf.score(X_resampled1,y_resampled1)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[ ]:


rf = RandomForestClassifier(n_estimators=14, random_state = 42,class_weight='balanced')
rf.fit(X_resampled1,y_resampled1)
rf.score(X_val,y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


print(roc_auc_score(y_val,y_pred_RF))


# In[ ]:


print(classification_report(y_val, y_pred_RF))


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(min_samples_split=i, random_state = 42,class_weight='balanced')
    dTree.fit(X_resampled1,y_resampled1)
    acc_train = dTree.score(X_resampled1,y_resampled1)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
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


dTree = DecisionTreeClassifier(class_weight='balanced', random_state = 42)
dTree.fit(X_resampled1,y_resampled1)
dTree.score(X_val,y_val)


# In[ ]:


y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))


# In[ ]:


print(roc_auc_score(y_val,y_pred_DT))


# In[ ]:


print(classification_report(y_val, y_pred_DT))


# ### Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


score_train_AB = []
score_test_AB = []

for i in range(1,20,1):
    ab = AdaBoostClassifier(n_estimators=i, random_state = 42)
    ab.fit(X_resampled1,y_resampled1)
    sc_train = ab.score(X_resampled1,y_resampled1)
    score_train_AB.append(sc_train)
    sc_test = ab.score(X_val,y_val)
    score_test_AB.append(sc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,20,1),score_train_AB,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,20,1),score_test_AB,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[ ]:


ab = AdaBoostClassifier(n_estimators=500, random_state = 42)
ab.fit(X_resampled1,y_resampled1)
ab.score(X_val,y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_AB = ab.predict(X_val)
confusion_matrix(y_val, y_pred_AB)


# In[ ]:


print(roc_auc_score(y_val,y_pred_AB))


# In[ ]:


print(classification_report(y_val, y_pred_AB))


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB


# In[ ]:


nb = NB()
nb.fit(X_resampled1,y_resampled1)
nb.score(X_val,y_val)


# In[ ]:


y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


print(roc_auc_score(y_val,y_pred_AB))


# In[ ]:


print(classification_report(y_val, y_pred_NB))


# ## Processing Test Dataset

# In[ ]:


X_test1 = pd.read_csv('../input/test.csv')
X_test1 = X_test1.replace({'?':np.nan})


# In[ ]:


ID = X_test1['ID']


# In[ ]:


del_col = ['Worker Class','Fill','Teen','PREV','Live','MOVE','REG','MSA',
           'State','Area','Reason','MLU','MOC','MIC','Enrolled',
          'COB SELF','COB MOTHER','COB FATHER','Detailed'
          ]

X_test1 = X_test1.drop(del_col, axis = 1)


# In[ ]:


null_columns_test = X_test1.columns[X_test1.isna().any()]
null_columns_test


# In[ ]:


for c in null_columns_test:
    X_test1[c] = X_test1[c].fillna(X_test1[c].mode()[0])


# In[ ]:


X_test1 = pd.get_dummies(X_test1, columns=['Married_Life','Cast', 'Hispanic', 'Sex','Full/Part', 'Tax Status',
                                        'Summary', 'Citizen','Schooling'])


# In[ ]:


X_test1 = X_test1[[features for features in impfeatures]]


# In[ ]:


preds = ab.predict(X_test1)


# In[ ]:


df = pd.DataFrame(columns=['ID', 'Class'])
df['ID']=ID
df['Class']=preds
df.head()


# In[ ]:


#df.to_csv('2015b3a70395g.csv',index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title} </a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)


# In[ ]:




