#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split
np.random.seed(42)


# Pre Processing

# In[3]:


t1 = pd.read_csv("../input/opcode_frequency_benign.csv")
t2 = pd.read_csv("../input/opcode_frequency_malware.csv")
test = pd.read_csv("../input/Test_data.csv")


# In[4]:


test.drop(['Unnamed: 1809'],axis=1,inplace=True)


# In[5]:


t1['Class'] = np.zeros(shape=t1.shape[0])
t2['Class'] = np.ones(shape=t2.shape[0])


# In[6]:


t1.drop(['FileName'],axis=1,inplace=True)
t2.drop(['FileName'],axis=1,inplace=True)


# In[7]:


train = t1.append(t2,ignore_index=False)


# In[8]:


drop = []
for i in train.columns:
    if(i!='Class'):
        if(train[i].unique().tolist()==[0] and test[i].unique().tolist()==[0]):
            drop.append(i)


# In[9]:


train.drop(drop,axis=1,inplace=True)
test.drop(drop,axis =1,inplace=True)


# In[10]:


columns= list(train.columns)
columns.remove('Class')
ss = StandardScaler()
train[columns]=ss.fit_transform(train[columns])
test[columns]=ss.transform(test[columns])


# In[11]:


train = train.sample(frac=1).reset_index(drop=True)


# In[12]:


X = train
X = X.drop('Class',axis=1)
y = train['Class']


# Gradient Boosting

# In[13]:


model = GradientBoostingClassifier(n_estimators = 100,random_state=42)
model.fit(X,y)
features = X.columns
thresholds = model.feature_importances_

zipped = zip(thresholds,features)
zipped = sorted(zipped,reverse=True)
sorted_features = [features for thresholds,features in zipped]


# In[14]:


X = X[sorted_features[:100]]
y = y


# In[15]:


X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.30, random_state=42)


# Bayesian Classification

# In[16]:


nb = GaussianNB()
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(nb, X, y, cv=10, scoring=(scorer), return_train_score=True)
print(cv_results.keys())
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# Logistic Regression

# In[17]:


lg = LogisticRegression(C=1)
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(lg, X, y, cv=10, scoring=(scorer), return_train_score=True)
print (cv_results.keys())
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# KNN Classification

# In[18]:


train_roc = []
test_roc = []
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    roc_auc_train = roc_auc_score(y_train,knn.predict(X_train))
    train_roc.append(roc_auc_train)
    roc_auc_test = roc_auc_score(y_val,knn.predict(X_val))
    test_roc.append(roc_auc_test)


# In[19]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_roc,color='blue', linestyle='dashed', marker='o')
test_score,=plt.plot(range(1,15),test_roc,color='red',linestyle='dashed', marker='o')
plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])
plt.xlabel('K neighbors')
plt.ylabel('ROC_AUC_Score')


# In[20]:


knn = KNeighborsClassifier(n_neighbors=5)
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(knn, X, y, cv=10, scoring=(scorer), return_train_score=True)
print (cv_results.keys())
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# Decision Tree Classification

# In[21]:


train_roc = []
test_roc = []
for i in range(1,15):
    dt = DecisionTreeClassifier(max_depth=i,random_state=42)
    dt.fit(X_train,y_train)
    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))
    train_roc.append(roc_auc_train)
    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))
    test_roc.append(roc_auc_test)


# In[22]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_roc,color='blue', linestyle='dashed', marker='o')
test_score,=plt.plot(range(1,15),test_roc,color='red',linestyle='dashed', marker='o')
plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])
plt.xlabel('max_depth')
plt.ylabel('ROC_AUC_Score')


# In[23]:


train_roc = []
test_roc = []
for i in range(2,15):
    dt = DecisionTreeClassifier(max_depth=15,min_samples_split=i,random_state=42)
    dt.fit(X_train,y_train)
    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))
    train_roc.append(roc_auc_train)
    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))
    test_roc.append(roc_auc_test)


# In[24]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,15),train_roc,color='blue', linestyle='dashed', marker='o')
test_score,=plt.plot(range(2,15),test_roc,color='red',linestyle='dashed', marker='o')
plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])
plt.xlabel('max_samples_split')
plt.ylabel('ROC_AUC_Score')


# In[25]:


dt = DecisionTreeClassifier(max_depth=15,min_samples_split=5,random_state=42)
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(dt, X, y, cv=10, scoring=(scorer), return_train_score=True)
print (cv_results.keys())
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# AdaBoost

# In[26]:


adb = AdaBoostClassifier(n_estimators = 200,random_state=42)
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(adb, X, y, cv=10, scoring=(scorer), return_train_score=True)
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# Random Forest

# In[27]:


rf = RandomForestClassifier(n_estimators = 300,max_depth=25,min_samples_split=3,random_state=42)
scorer = make_scorer(roc_auc_score)
cv_results = cross_validate(rf, X, y, cv=10, scoring=(scorer), return_train_score=True)
print (cv_results.keys())
print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))
print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))


# In[28]:


rf.fit(X,y)
final_df = test.copy()
final_df['Class'] = rf.predict(test[sorted_features[:100]])
csv = final_df[['FileName','Class']]
csv['Class']=csv['Class'].astype(int)
csv.to_csv('Submission_10.csv',index=False)


# In[29]:


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
create_download_link(csv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




