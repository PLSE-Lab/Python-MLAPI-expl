#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[34]:


import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


# In[145]:


LandT_Data =  pd.read_csv("../input/landtdataset/train.csv", header=0)
#LandT_Data_test = pd.read_csv("./input/landt-test/test_bqCt9Pv.csv", header=0)


# In[146]:


LandT_Data[LandT_Data.index == 71593]


# In[147]:


rows = [134575,71593]
LandT_Data.drop(LandT_Data.index[rows],inplace=True)


# In[148]:


LandT_Data['ltv'] = LandT_Data['ltv']**2


# In[149]:


LandT_Data = LandT_Data.drop(['UniqueID','DD_year','MobileNo_Avl_Flag'], axis=1)


# In[150]:


LandT_Data['PRI.CURRENT.BALANCE'] = np.log(20000+LandT_Data['PRI.CURRENT.BALANCE'])
LandT_Data['asset_cost'] = np.log(1+LandT_Data['asset_cost'])
LandT_Data['disbursed_amount'] = np.log(1+LandT_Data['disbursed_amount'])


# In[151]:


LandT_Data.dtypes


# In[152]:


LandT_Data.isna().sum()


# In[153]:


LandT_Data = LandT_Data.dropna(axis=0)


# In[154]:


cat_attr=['PERFORM_CNS.SCORE.DESCRIPTION','Employment.Type','Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag']


# In[155]:


num_attr = ['disbursed_amount','asset_cost','ltv','branch_id','supplier_id','manufacturer_id','Current_pincode_ID','dob_day','dob_month',
            'dob_year','DD_day','DD_month','State_ID','Employee_code_ID','PERFORM_CNS.SCORE','PRI.NO.OF.ACCTS',
            'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS',
            'SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT',
            'NEW.ACCTS.IN.LAST.SIX.MONTHS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','AVERAGE.years','Average_months','Credit_years','Credit_months',
            'NO.OF_INQUIRIES']


# In[156]:


y= LandT_Data["loan_default"]
X= LandT_Data.drop('loan_default',axis=1)


# In[157]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3 , random_state = 2, stratify=y)
X_train.shape


# In[158]:


X_train_imputed_dummies = pd.get_dummies(columns=cat_attr, data=X_train,prefix=cat_attr, prefix_sep="_", drop_first=True)
X_test_imputed_dummies = pd.get_dummies(columns=cat_attr, data=X_test,prefix=cat_attr, prefix_sep="_", drop_first=True)

X_train_imputed_dummies.shape


# In[159]:


X_train_imputed_dummies = X_train_imputed_dummies.drop('PERFORM_CNS.SCORE.DESCRIPTION_Not Scored: More than 50 active Accounts found',axis=1)


# In[160]:


scaler = StandardScaler()
scaler.fit(X_train[num_attr])

X_train_imputed_dummies[num_attr]=scaler.transform(X_train_imputed_dummies[num_attr])
X_test_imputed_dummies[num_attr]=scaler.transform(X_test_imputed_dummies[num_attr])


# In[161]:


X_test_imputed_dummies.columns


# In[167]:


clf = tree.DecisionTreeClassifier(max_depth=8)
clf = clf.fit(X_train_imputed_dummies,y_train)


# In[168]:


np.argsort(clf.feature_importances_)


# In[169]:


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
pd.DataFrame([X_train_imputed_dummies.columns[indices],np.sort(importances)[::-1]])


# In[170]:


train_pred = clf.predict(X_train_imputed_dummies)
print(accuracy_score(y_train,train_pred))

test_pred = clf.predict(X_test_imputed_dummies)
print(accuracy_score(y_test,test_pred))


# In[171]:


fpr, tpr, thresholds = roc_curve(y_train, train_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr,tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[30]:


import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve


# Create classifiers
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)



plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train_imputed_dummies, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test_imputed_dummies)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test_imputed_dummies)
        prob_pos =             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value =         calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()


# In[31]:


classifier = RandomForestClassifier(n_estimators=1000,max_depth=15)
clf = Pipeline(memory = './',steps=[('classifier', classifier)])

clf.fit(X=X_train_imputed_dummies, y=y_train)


# In[32]:


y_train_pred = clf.predict(X_train_imputed_dummies)
print(accuracy_score(y_train,y_train_pred))


y_pred = clf.predict(X_test_imputed_dummies)
print("Test Accuracy = ",accuracy_score(y_test,y_pred))


# In[33]:


fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr,tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




