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


import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from matplotlib import style
#sta matplotlib to inline and displays graphs below the corresponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from sklearn.datasets import *
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/purchase_prediction.csv")
df.head()


# In[ ]:


#lets drop the 'user id' which is not useful for our prediction further
df.drop("User ID",axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


#Data Imputation
df.isnull().sum()


# In[ ]:


df.describe(include='all').T


# In[ ]:


df.info()


# we can see that there is no null or missing values

# In[ ]:


df.Purchased.value_counts()


# #######The values are imbalanced in Label, We need to do resampling Techniques
# 

# In[ ]:


#DATA WRANGLING
from sklearn import preprocessing  
df=df.apply(preprocessing.LabelEncoder().fit_transform)
df.head()


# In[ ]:


from statsmodels.stats.proportion import proportions_ztest


# In[ ]:


# for Gender and Attrition
pd.crosstab(df['Purchased'],df['Gender'])


# In[ ]:


count=np.array([77,66])
obs=np.array([204,196])


# In[ ]:


zstat,pvalue=proportions_ztest(count,obs)
print('z value: %0.3f, p value: %0.3f' %(zstat,pvalue))


# ###its accepting Null hypothesis which is the proportion of male = proportin of female

# In[ ]:


from scipy.stats import ttest_ind


# In[ ]:


grp=df.groupby('Purchased')
grp_0=grp.get_group(0)
grp_1=grp.get_group(1)


# In[ ]:


mean1=grp_1.Age.mean()
mean1


# In[ ]:


mean0=grp_0.Age.mean()
mean0


# In[ ]:


ttest_ind(grp_0['Age'],grp_1['Age'])


# ######based on the Pvalue which is very less and showing that very less probability of Null Hypothesis to become True

# #####as it is lesser than significant level(0.05), we have to accept the ALTERNATE HYPOTHESIS and reject Null Hypothesis

# In[ ]:


#lets find  the pvalue for EstimatedSalary
grp=df.groupby('Purchased')
grp_0=grp.get_group(0)
grp_1=grp.get_group(1)


# In[ ]:


mean1=grp_1.EstimatedSalary.mean()
mean1


# In[ ]:


mean0=grp_0.EstimatedSalary.mean()
mean0


# In[ ]:


ttest_ind(grp_0['EstimatedSalary'],grp_1['EstimatedSalary'])


# ####### As we can see the pvalue is lesser than significant level accept Alternate Hypothesis which are the mean of Estimated Salary 0purchased

# #####Lets drop the column Gender as it is accepting NULL Hypothesis to be true

# 

# In[ ]:


df.drop('Gender',1,inplace=True)


# In[ ]:


df.head()


# ##since there are in different units, we have to scale the data after splitting the X,y labels

# In[ ]:


y=df.Purchased
X=df.drop('Purchased',1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler
se=StandardScaler()
X_train=se.fit_transform(X_train)
X_test=se.transform(X_test)


# In[ ]:


print(X_train.shape,X_test.shape)


# In[ ]:


#apply SMOTE viz resampling technique
from  imblearn.over_sampling import SMOTE


# In[ ]:


sm=SMOTE(random_state=1,ratio=1.0)
X_train,y_train=sm.fit_sample(X_train,y_train)


# In[ ]:


## Apply Logistic Regression with balanced data by SMOTE
from sklearn.linear_model import LogisticRegression
smote=LogisticRegression()
smote.fit(X_train,y_train)
somote_pred=smote.predict(X_test)


# In[ ]:


#checking Accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import classification_report
from mlxtend.evaluate import confusion_matrix


# In[ ]:


accuracy_score(y_test,somote_pred)


# In[ ]:


print('classification:\n',classification_report(y_test,somote_pred))


# In[ ]:


f1_score(y_test,somote_pred)


# In[ ]:


print('recall score:',recall_score(y_test,somote_pred))


# In[ ]:


print('precision_score',precision_score(y_test,somote_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,somote_pred),annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# ## TP=41 FP=7 TN=58 FN=14

# #### APPly KNN MODEL
# 

# In[ ]:


df.head()


# In[ ]:


y=df.Purchased
X=df.drop('Purchased',1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler
se=StandardScaler()
X_train=se.fit_transform(X_train)
X_test=se.transform(X_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier()


# In[ ]:


param={'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}
gs=GridSearchCV(knn,param,cv=5,scoring='roc_auc')
gs.fit(X_train,y_train)


# In[ ]:


param={'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}
gs=GridSearchCV(knn,param,cv=5,scoring='roc_auc')
gs.fit(X_train,y_train)


# In[ ]:


gs.best_params_


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=23,weights='uniform')


# In[ ]:


knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


# In[ ]:


print('Accuracy Score:',accuracy_score(y_test,y_pred))


# In[ ]:


print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))


# In[ ]:


print('The classification report:',classification_report(y_test,y_pred))


# In[ ]:


print('recall score:',recall_score(y_test,y_pred))


# In[ ]:


print('precision_score',precision_score(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# #### TP=43 TN=62 FP=5 FN=10

# #### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[ ]:


parms= {'criterion':['entropy','gini']}
grids=GridSearchCV(dt,parms,cv=10,scoring='roc_auc')
grids.fit(X_train,y_train)


# In[ ]:


grids.best_params_


# In[ ]:


dt=DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(X_train,y_train)


# In[ ]:


y_pred=dt.predict(X_test)


# In[ ]:


print('Accuracy Score:',accuracy_score(y_test,y_pred))


# In[ ]:


print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))


# In[ ]:


print('The classification report:',classification_report(y_test,y_pred))


# In[ ]:


print('recall score:',recall_score(y_test,y_pred))


# In[ ]:


print('precision_score',precision_score(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# TP=41 FN=11 TN=61 FP=7

# #### Naive Bayes

# In[ ]:


#import a Library
from sklearn.naive_bayes import GaussianNB


# In[ ]:


gb=GaussianNB()


# In[ ]:


gb.fit(X_train,y_train)
y_pred=gb.predict(X_test)


# In[ ]:


print('Accuracy Score:',accuracy_score(y_test,y_pred))


# In[ ]:


print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))


# In[ ]:


print('The classification report:',classification_report(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# TP=39 FN=8 TN=64 FP=9

# #### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=50,random_state=0)


# In[ ]:


rf_var=[]
for val in np.arange(1,50):
    rf=RandomForestClassifier(criterion='entropy',n_estimators=val,random_state=0)
    kfold = KFold(shuffle=True,n_splits=5, random_state=0)
    cv_results = cross_val_score(rf, X, y, cv=kfold, scoring='roc_auc')
    rf_var.append(np.var(cv_results,ddof=1))
    print(val,np.var(cv_results,ddof=1))


# In[ ]:


x_axis=np.arange(1,50)
plt.plot(x_axis,rf_var)


# In[ ]:


rf=RandomForestClassifier(n_estimators=17,random_state=0)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_pred=rf.predict(X_test)


# In[ ]:


print('Accuracy Score:',accuracy_score(y_test,y_pred))


# In[ ]:


print('the confusion matrix:\n',confusion_matrix(y_test,y_pred))


# In[ ]:


print('The classification report:',classification_report(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# TP=45 FN=11 TN=61 FP=3

# #### SECOND Way Of APPROACH By Using Cross Validation

# In[ ]:


models=[]
models.append(('Logistic',smote))
models.append(('Naive',gb))
models.append(('knn',knn))
models.append(('DT',dt))
models.append(('RF',rf))


# In[ ]:


# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
for name,model in models:
    kfold = KFold(n_splits=5, random_state=0,shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.var(ddof=1))
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# #### From the above chart we can see that the KNN has good accuracy since our data is numerical and the accuracy will fall down in case of Decision Tree
# 

# In[ ]:




