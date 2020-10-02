#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import seaborn as sns


# In[ ]:


df=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# DATA CLEANING

# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
df.isnull().sum()


# In[ ]:


df.info()


# AS the totalchaeges column is in object form converting it to float form, the errors part, gives NaN to non interger value

# In[ ]:


df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors='coerce')


# In[ ]:


df.info()


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
df.isnull().sum()


# In[ ]:


df.dropna(subset=["TotalCharges"], inplace=True)


# In[ ]:


df.drop('customerID',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# **EDA**

# In[ ]:


df_num=df.select_dtypes(exclude=["object_"])
df_num.head()


# In[ ]:


df_cat=df.select_dtypes(include=["object_"])
df_cat.head()


# In[ ]:


corr_df=df.corr()
corr_df


# This shows that direct correlation is between the tenure and total charges and it is logical. Also the monthly and total charge are correlated.
# 
# To check whether the totalcharges per tenure and the monthly charges are corelated more than this, then it would give a better understanding.

# In[ ]:


Charges_Per_Month=np.divide(df["TotalCharges"],df["tenure"])
df_num["Charges_Per_Month"]=Charges_Per_Month


# In[ ]:


corr_df=df_num.corr()
corr_df


# In[ ]:


df_num.head()


# In[ ]:


df_num.hist()


# In[ ]:


sns.countplot(x='Churn', hue='SeniorCitizen', data=df)


# In[ ]:


f=pd.melt(df,value_vars=sorted(df_cat))
g=sns.FacetGrid(f,col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation="vertical")
g=g.map(sns.countplot,'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()


# In[ ]:


#To mix no service with no

df["MultipleLines"].replace(to_replace="No phone service", value="No", inplace=True)

df["OnlineSecurity"].replace(to_replace="No internet service", value="No", inplace=True)

df["OnlineBackup"].replace(to_replace="No internet service", value="No", inplace=True)

df["DeviceProtection"].replace(to_replace="No internet service", value="No", inplace=True)


# In[ ]:


for a in df_cat.columns:
    sns.countplot(x='Churn', hue=a, data=df_cat)
    plt.show(block='False')  


# In[ ]:


f=pd.melt(df,value_vars=sorted(df_cat))
g=sns.FacetGrid(f,col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation="vertical")
g=g.map(sns.countplot,'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]
g.fig.tight_layout()
plt.show()


# In[ ]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

df['Partner']=label_encoder.fit_transform(df['Partner'])
df['MultipleLines']=label_encoder.fit_transform(df['MultipleLines'])
df['InternetService']=label_encoder.fit_transform(df['InternetService'])
df['OnlineSecurity']=label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup']=label_encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection']=label_encoder.fit_transform(df['DeviceProtection'])
df['Contract']=label_encoder.fit_transform(df['Contract'])
df['PaperlessBilling']=label_encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod']=label_encoder.fit_transform(df['PaymentMethod'])
df['Churn']=label_encoder.fit_transform(df['Churn'])
print(df.dtypes)


# In[ ]:


df.drop('PhoneService',axis=1,inplace=True)
df.drop('gender',axis=1,inplace=True)
df.drop('Dependents',axis=1,inplace=True)
df.drop('TechSupport',axis=1,inplace=True)
df.drop('StreamingTV',axis=1,inplace=True)
df.drop('StreamingMovies',axis=1,inplace=True)
print(df.dtypes)


# In[ ]:


df.head()


# In[ ]:


y=df["Churn"]
y.shape


# In[ ]:


#no=0, tes=1
y.head()


# In[ ]:


x=df.loc[:, df.columns!='Churn']
x.head()


# In[ ]:


from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:


sns.countplot(yTrain)


# In[ ]:


sns.countplot(yTest)


# In[ ]:


#Standardising the data

scaler=preprocessing.StandardScaler()
x=scaler.fit_transform(x)
x


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds=StratifiedKFold(n_splits=10, random_state=42)

#Gradient Boost Classifier

from sklearn.ensemble import GradientBoostingClassifier
gb_clf=GradientBoostingClassifier(random_state=41)

clone_clf=clone(gb_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for GBC", n_correct/len(y_pred))
    


# In[ ]:


from sklearn.svm import SVC
svc_clf=SVC(random_state=42)

clone_clf=clone(svc_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for SVM", n_correct/len(y_pred))
    


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc_clf=RandomForestClassifier(random_state=42)

clone_clf=clone(rfc_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for RandomForestClassifier", n_correct/len(y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier()

clone_clf=clone(knn_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for KNeighborsClassifierr", n_correct/len(y_pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc_clf=DecisionTreeClassifier(random_state=42)

clone_clf=clone(dtc_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for DecisionTreeClassifier", n_correct/len(y_pred))


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_clf=LogisticRegression(random_state=42)

clone_clf=clone(lr_clf)

clone_clf.fit(xTrain,yTrain)
y_pred=clone_clf.predict(xTest)
n_correct=sum(y_pred==yTest)
print("Result for LogisticRegression", n_correct/len(y_pred))


# In[ ]:


from sklearn.model_selection import cross_val_predict
y_pred=cross_val_predict(gb_clf,x,y,cv=10)

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)


# In[ ]:


from sklearn.metrics import precision_score, recall_score

print("Precision",precision_score(y,y_pred))
print("recall",recall_score(y,y_pred))


# In[ ]:


y_scores=cross_val_predict(gb_clf,x,y,cv=10, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds=precision_recall_curve(y,y_scores)


# In[ ]:


#Precision recall curve

def plot_curve(precisions, recalls, thresholds):
    plt.plot(thresholds,precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds,recalls[:-1], "b--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plot.ylim([0,1])

plot_curve(precisions, recalls, thresholds)
plt.show()



# In[ ]:


plt.plot(precisions, recalls)
plt.xlabel("Recall")
plt.ylabel("Precision")


# In[ ]:


y_new_scores=(y_scores>-0.5)
print("New Precision:", precision_score(y,y_new_scores))
print("New Recall:", recall_score(y,y_new_scores))


# In[ ]:


from sklearn.metrics import roc_curve
fpr,tpr, thresholds=roc_curve(y,y_scores)

def plot_roc(fpr,tpr, label=None):
    plt.plot(fpr,tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],"k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive rate")
    plt.ylabel("true Positive rate")

plot_roc(fpr,tpr, label=None)
plt.show()


# In[ ]:




