#!/usr/bin/env python
# coding: utf-8

# # Importing required libararies

# In[ ]:


import numpy as np
import pandas as pd

from pandas import DataFrame,Series

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading the dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/bank-term-deposit-subscribers/bank-additional-full.csv', sep = ';')


# ## Understanding the dataset

# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# - No Missing / Null values in the dataset

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Attribute Information:
Attribute Information:
------------------------    

Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# In[ ]:


print('Number of unique values in each column:')
for col in df.columns[0:]:
    print(col,':')
    print('nunique =', df[col].nunique())
    print('unique =', df[col].unique())
    print()


# # Data Preprocessing
job               41188 non-null object
marital           41188 non-null object
education         41188 non-null object
default           41188 non-null object
housing           41188 non-null object
loan              41188 non-null object
contact           41188 non-null object
month             41188 non-null object
day_of_week       41188 non-null object
# In[ ]:


df['y'] = df['y'].replace({'yes':1,'no':0})


# In[ ]:


# Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


cols = df.select_dtypes(object).columns

for i in cols:
    df[i] = le.fit_transform(df[i])


# # Base Model building 

# In[ ]:


y = df['y']
x = df.drop(['y'], axis = 1)


# In[ ]:


y.head()


# In[ ]:


x.head()


# In[ ]:


# split into train and test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, random_state=1)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Visualize the Y variable for oversampling check - bi class

sns.countplot(df['y'])
plt.show()


# In[ ]:


term_dep_subs = len(df[df['y'] == 1])
no_term_dep_subs = len(df[df['y'] == 0])
total = term_dep_subs + no_term_dep_subs

term_dep_subs = (term_dep_subs / total) * 100
no_term_dep_subs = (no_term_dep_subs / total) * 100

print('term_dep_subs:',term_dep_subs)
print('no_term_dep_subs:',no_term_dep_subs)


# # Classification Evaluation Metrics

# In[ ]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,r2_score


# In[ ]:


def disp_confusion_matrix(model, x, y):
    ypred = model.predict(x)
    cm = confusion_matrix(y,ypred)
    ax = sns.heatmap(cm,annot=True,fmt='d')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.show()
    
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    print('Accuracy =',accuracy)
    print('Precision =',precision)
    print('Recall =',recall)
    print('F1 Score =',f1)


# In[ ]:


def disp_roc_curve(model, xtest, ytest):
    yprob = model.predict_proba(xtest)
    fpr,tpr,threshold = roc_curve(ytest,yprob[:,1])
    roc_auc = roc_auc_score(ytest,yprob[:,1])

    print('ROC AUC =', roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC Curve (area = %0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# In[ ]:


#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import NearMiss
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
# from collections import Counter
# from sklearn.model_selection import KFold, StratifiedKFold
import scipy.stats as st


# # Base Model

# In[ ]:


y = df['y']
x = df.drop(['y'], axis = 1)


# In[ ]:


import statsmodels.api as sm


# In[ ]:


X_sm = x
X_sm = sm.add_constant(X_sm)
lm = sm.Logit(y,X_sm).fit()
lm.summary()


# ## Logistic Regression

# In[ ]:


# Base Model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print('Training set score = {:.3f}'.format(logreg.score(x_train,y_train)))

print('Test set score = {:.3f}'.format(logreg.score(x_test,y_test)))
#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


disp_confusion_matrix(logreg, x_test, y_test)
disp_roc_curve(logreg, x_test, y_test)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

print('Training score =', dt.score(x_train, y_train))
print('Test score =', dt.score(x_test, y_test))
#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


disp_confusion_matrix(dt, x_test, y_test)
disp_roc_curve(dt, x_test, y_test)


# ## KNN

# In[ ]:


# knn 5 default

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('Training score =', knn.score(x_train, y_train))
print('Test score =', knn.score(x_test, y_test))
#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


disp_confusion_matrix(knn, x_test, y_test)
disp_roc_curve(knn, x_test, y_test)


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print('Training score =', nb.score(x_train, y_train))
print('Test score =', nb.score(x_test, y_test))
#print("R squared: {}".format(r2_score(y_true=y_test,y_pred=y_pred)))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


disp_confusion_matrix(knn, x_test, y_test)
disp_roc_curve(knn, x_test, y_test)


# # Feature Engineering:

# In[ ]:




