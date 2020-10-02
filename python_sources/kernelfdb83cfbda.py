#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df=pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.duplicated().sum()


# There is no null values and duplicate rows

# In[ ]:


df.describe().T


# In[ ]:


df.Attrition.value_counts()


# In[ ]:


df.BusinessTravel.value_counts()


# In[ ]:


df.Department.value_counts()


# In[ ]:


df.EducationField.value_counts()


# In[ ]:


df.Gender.value_counts()


# In[ ]:


df.JobRole.value_counts()


# In[ ]:


df.MaritalStatus.value_counts()


# In[ ]:


df.Over18.value_counts()


# In[ ]:


df.OverTime.value_counts()


# cat_col = df.select_dtypes(exclude=np.number).columns
# num_col = df.select_dtypes(include=np.number).columns
# print(cat_col)
# print(num_col)
# plt.figure(figsize=(40,40))
# sns.heatmap(data=df[num_col],annot=True)

# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns

# In[ ]:


df['Attrition'].replace({'No':0,'Yes':1},inplace=True)


# In[ ]:


df.corr().Attrition


# In[ ]:


cat_col = df.select_dtypes(exclude=np.number).columns
num_col = df.select_dtypes(include=np.number).columns


# In[ ]:


encoded_cat_col = pd.get_dummies(df[cat_col])


# In[ ]:


emp_ready_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)


# In[ ]:


X = emp_ready_model.drop(columns='Attrition')
y = emp_ready_model['Attrition']
                         


# In[ ]:


from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


train_Pred = logreg.predict(X_train)


# In[ ]:


metrics.confusion_matrix(y_train,train_Pred)


# In[ ]:


metrics.accuracy_score(y_train,train_Pred)


# In[ ]:


test_Pred = logreg.predict(X_test)


# In[ ]:


metrics.confusion_matrix(y_test,test_Pred)


# In[ ]:


metrics.accuracy_score(y_test,test_Pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, test_Pred))


# In[ ]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# KNN algorithm

# In[ ]:


emp_ready_model.head()


# In[ ]:


X = emp_ready_model.drop(columns='Attrition').values
y = emp_ready_model['Attrition'].values
                  


# In[ ]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# In[ ]:


k = 5
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[ ]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[ ]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[ ]:


metrics.confusion_matrix(y_train, neigh.predict(X_train))


# In[ ]:


metrics.confusion_matrix(y_test, yhat)


# In[ ]:


Ks = 800
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[ ]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

