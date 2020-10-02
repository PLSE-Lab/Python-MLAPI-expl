#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


x = np.zeros((3, 3))


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')


# In[ ]:


data.type.value_counts()


# In[ ]:


import seaborn as sns 


# In[ ]:


data.isFraud.value_counts().plot.bar()


# In[ ]:


plt.ylabel('Amount')
plt.xlabel('Type')
data.type.value_counts().plot.bar()


# ## Plotting graph of sin

# In[ ]:


x = np.linspace(-10, 10, 1000)
y = np.cos(x)


# In[ ]:


plt.plot(x, y)


# In[ ]:


data.isFraud.value_counts()


# In[ ]:


data[data.type == 'PAYMENT'].newbalanceOrig.head()


# In[ ]:


data[data.type == 'PAYMENT'].newbalanceOrig.mean()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data.head()


# ## EDA (Explaratory data analysis)

# In[ ]:


data_frauded = data[data.isFraud == 1]


# In[ ]:


data_fair = data[data.isFraud == 0]


# In[ ]:


data_frauded.newbalanceDest.median()


# In[ ]:


data_fair.newbalanceDest.median()


# In[ ]:


data.head()


# In[ ]:


(data_frauded.oldbalanceOrg - data_frauded.newbalanceOrig).mean()


# In[ ]:


(data_fair.oldbalanceOrg - data_fair.newbalanceOrig).mean()


# ## Plot class balance 

# In[ ]:


import seaborn as sns


# In[ ]:


import matplotlib.pyplot as plt


# ## Turn on custom styles 

# In[ ]:


plt.style.use('ggplot')


# ## Types Matter!

# In[ ]:


data.type.value_counts()


# In[ ]:


plt.ylabel('Amount')
plt.xlabel('Type')
data_fraud.type.value_counts().plot.bar()
data.type.value_counts().plot.bar()


# In[ ]:


data_fraud = data[data.isFraud == 1] 


# In[ ]:


# dustribution of types in fraud 
data_fraud.type.value_counts().plot.bar()


# In[ ]:


data_fair = data[data.isFraud == 0] 


# In[ ]:


plt.ylabel('type')
data_fair.type.value_counts().plot.pie()


# In[ ]:


data.isFraud.value_counts()


# In[ ]:


n_fraud = data.isFraud.value_counts()[1]
n_fair = data.isFraud.value_counts()[0]
n_fraud / (n_fraud + n_fair)


# In[ ]:


data.columns


# In[ ]:


data.oldbalanceOrg.min()


# In[ ]:


data.oldbalanceOrg.max()


# In[ ]:


data.oldbalanceOrg.mean()


# In[ ]:


data_frauded.oldbalanceOrg.isna().value_counts()


# In[ ]:


sns.distplot((data_fair.oldbalanceOrg), kde=False, bins=200)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,6))
sns.distplot((data_frauded.oldbalanceOrg), kde=False, bins=200, ax=ax[0])
sns.distplot((data_fair.oldbalanceOrg), kde=False, bins=200, ax=ax[1])
ax[0].set_title('Dist oldBalancOrig with Frauded person')
ax[1].set_title('Dist oldBalancOrig with Fair person')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,6))
sns.distplot((data_frauded.newbalanceOrig), kde=False, bins=200, ax=ax[0])
sns.distplot((data_fair.newbalanceOrig), kde=False, bins=200, ax=ax[1])
ax[0].set_title('Dist newBalancOrig with Frauded person')
ax[1].set_title('Dist newBalancOrig with Fair person')
plt.savefig('plot1')


# In[ ]:


sns.distplot((data_fair.newbalanceOrig[data_fair.newbalanceOrig < 1000000]), kde=False, bins=200, ax=ax[1])
plt.show()


# In[ ]:


data_fair.newbalanceOrig[data_fair.newbalanceOrig < 1000000][data_fair.newbalanceOrig > 100000].head()


# In[ ]:





# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,6))
sns.distplot((data_fraud.newbalanceOrig[data_fraud.newbalanceOrig < 1000000]), kde=False, bins=200, ax=ax[0])
sns.distplot((data_fair.newbalanceOrig[data_fair.newbalanceOrig < 700000][data_fair.newbalanceOrig > 10000]), kde=False, bins=200, ax=ax[1])
ax[0].set_title('Dist oldBalancOrig with Frauded person')
ax[1].set_title('Dist oldBalancOrig with Fair person')


# In[ ]:


data.head()


# In[ ]:


from  sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=10, ) #metric='manhattan')


# In[ ]:


y = data.isFraud.values


# In[ ]:


X = data.drop(labels=['type', 'nameOrig', 'nameDest', 'isFraud'], axis=1)


# In[ ]:


X.head()


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[:300000], y[:300000], test_size=0.3)
clf = KNeighborsClassifier(n_neighbors=10,) #metric='manhattan')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
roc_auc_score(y_test, y_pred[:, 1])


# ## Let's use types 

# In[ ]:


payment : 1
transfer: 2
payment !< transfer
1 < 2

onehot encoding 
payment : (1, 0, 0, 0, 0)
transfer : (0, 1, 0, 0, 0)


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.nameOrig.nunique()


# In[ ]:


data = pd.concat([pd.get_dummies(data.type, ), data], axis=1)


# In[ ]:


data.head()


# In[ ]:


X = data.drop(labels=['type', 'nameOrig', 'nameDest', 'isFraud'], axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[:300000], y[:300000],
                                                    test_size=0.3, 
                                                    random_state=42,)


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=19, metric='manhattan')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))


# In[ ]:


res = []
for i in range(1, 25, 1):
    clf = KNeighborsClassifier(n_neighbors=i, metric='manhattan')
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    score = roc_auc_score(y_test, y_pred[:, 1])
    res.append(score)
    print(score)


# In[ ]:


plt.xticks(range(1, 25, 2))
plt.xlabel('N neighborhoods')
plt.ylabel('auc roc score')
plt.legend()
plt.plot(res)


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=19, metric='manhattan')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))


# In[ ]:


from sklearn import metrics


# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ## Normalisation tricks

# In[ ]:


from sklearn.preprocessing import scale, StandardScaler


# In[ ]:


# yet another normalization
#X_norm = scale(X)
#
scl = StandardScaler()
X_norm = scl.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_norm[:300000], y[:300000],
                                                    test_size=0.3, 
                                                    random_state=42,)


# In[ ]:


clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ## Unsupervised learning (PCA)

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


pca = PCA(n_components=7)


# In[ ]:


X_pca = pca.fit_transform(X)


# In[ ]:


X_pca.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_pca[:300000], y[:300000],
                                                    test_size=0.3, 
                                                    random_state=42,)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


del X_pca


# ## Unsuperviesed learning (KMeans)

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_pca[:300000], y[:300000],
                                                    test_size=0.3, 
                                                    random_state=42,)


# In[ ]:


kmeans = KMeans(n_clusters=2)


# In[ ]:


kmeans.fit(X_train)


# In[ ]:


y_pred = kmeans.transform(X_test)
norm_c = 1 / y_pred.sum(axis=1)
y_pred[:, 1] *= norm_c
y_pred[:, 0] *= norm_c


# In[ ]:


print(roc_auc_score(y_test, y_pred[:, 0]))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 0])
auc = metrics.roc_auc_score(y_test, y_pred[:, 0])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


import xgboost


# In[ ]:


clf = xgboost.XGBClassifier(max_depth=3,
                            n_estimators=50,)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X.values[:300000], y[:300000],
                                                    test_size=0.3, 
                                                    random_state=42,)

clf = xgboost.XGBClassifier(max_depth=5, 
                            n_estimators=100, n_jobs=-1, )
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:, 1]))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


X_pca[:50, :2].shape


# In[ ]:


sizes = y[:5000]
sizes[sizes == 1] = 4
sizes[sizes == 0] = 1


# In[ ]:


plt.scatter(x= X_pca[:5000, 6], y=X_pca[:5000, 3], c=y[:5000], cmap='viridis', s=sizes)


# In[ ]:


plt.scatter(x= X_pca[:5000, 0][y[:5000] == 1], y=X_pca[:5000, 1][y[:5000] == 1], cmap='viridis')


# In[ ]:


#T-NSE


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


X_tnse = TSNE(n_components=2).fit_transform(X[:5000])


# In[ ]:


sizes = y[:5000]
sizes[sizes == 1] = 4
sizes[sizes == 0] = 1


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


12 -> 2


# In[ ]:


plt.scatter(x= X_tnse[:1000, 0], y=X_tnse[:1000, 1], c=y[:1000], cmap='viridis', s=sizes[:500]*20)


# In[ ]:


X[:2].values


# In[ ]:


X_tnse[:10]


# In[ ]:





# In[ ]:




