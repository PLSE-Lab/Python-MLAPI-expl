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


df = pd.read_csv("/kaggle/input/titanic/train.csv")
gen= pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test1_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df=test1_df.copy()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


test_df.head()


# In[ ]:


sns.heatmap(test_df.isnull(), cbar=False)
df.isnull().sum()


# In[ ]:


gen.head()


# DATA CLEANING

# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df["Cabin"]

Drop the name and passenger id, Cabin
# In[ ]:


df.drop('PassengerId',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True)
df.drop('Age', axis=1,inplace=True )
df.drop('Ticket', axis=1,inplace=True )
df.dropna(subset=['Embarked'],inplace=True )
df.head()


# In[ ]:





# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
df.isnull().sum()


# In[ ]:


df.nunique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_num=df.select_dtypes(exclude=["object_"])
df_num.head()


# In[ ]:


df_cat=df.select_dtypes(include=["object_"])
df_cat.head()


# In[ ]:


corr_df=df.corr()
corr_df


# In[ ]:


df_num.hist()


# In[ ]:


for a in df_num.columns:
    sns.countplot(x='Survived', hue=a, data=df_num)
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


sns.countplot(x='Survived', hue="Sex", data=df)
plt.show(block='False')  


# In[ ]:


sns.countplot(x='Survived', hue="Embarked", data=df)
plt.show(block='False')  


# sns.countplot(x='Survived', hue="Cabin", data=df)
# plt.show(block='False')  

# sns.countplot(x='Survived', hue="Ticket", data=df)
# plt.show(block='False')  

# In[ ]:


y=df["Survived"]
y.shape


# In[ ]:


df.head()


# In[ ]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

df['Sex']=label_encoder.fit_transform(df['Sex'])
df['Embarked']=label_encoder.fit_transform(df['Embarked'])
df.head()


# In[ ]:


x=df.loc[:, df.columns!='Survived']
x.head()


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

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for GBC", n_correct/len(y_pred))


# In[ ]:


from sklearn.svm import SVC
svc_clf=SVC(random_state=42)

clone_clf=clone(svc_clf)

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for SVM", n_correct/len(y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc_clf=RandomForestClassifier(random_state=42)

clone_clf=clone(rfc_clf)

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for RandomForestClassifier", n_correct/len(y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier()

clone_clf=clone(knn_clf)

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for KNeighborsClassifierr", n_correct/len(y_pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc_clf=DecisionTreeClassifier(random_state=42)

clone_clf=clone(dtc_clf)

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for DecisionTreeClassifier", n_correct/len(y_pred))


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_clf=LogisticRegression(random_state=42)

clone_clf=clone(lr_clf)

clone_clf.fit(x,y)
y_pred=clone_clf.predict(x)
n_correct=sum(y_pred==y)
print("Result for LogisticRegression", n_correct/len(y_pred))


# In[ ]:


from sklearn.model_selection import cross_val_predict
y_pred=cross_val_predict(rfc_clf,x,y,cv=10)

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


test_df.head()


# In[ ]:


test_df.drop('PassengerId',axis=1,inplace=True)
test_df.drop('Name',axis=1,inplace=True)
test_df.drop('Cabin',axis=1,inplace=True)
test_df.drop('Age', axis=1,inplace=True )
test_df.drop('Ticket', axis=1,inplace=True )
test_df.dropna(subset=['Embarked'],inplace=True )
test_df.head()


# In[ ]:





# In[ ]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

test_df['Sex']=label_encoder.fit_transform(test_df['Sex'])
test_df['Embarked']=label_encoder.fit_transform(test_df['Embarked'])
test_df.head()


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)
test_df.isnull().sum()


# In[ ]:


test_df.dropna(subset=["Fare"], inplace=True)


# In[ ]:


test1_df.dropna(subset=["Fare"], inplace=True)


# In[ ]:




from sklearn.ensemble import RandomForestClassifier
rfc_clf=RandomForestClassifier(random_state=42)

clone_clf=clone(rfc_clf)

clone_clf.fit(x,y)
predictions = clone_clf.predict(test_df)

output = pd.DataFrame({'PassengerId': test1_df.PassengerId, 'Survived': predictions})
output.to_csv('RANDOM_final.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:





# In[ ]:





# 

# In[ ]:




