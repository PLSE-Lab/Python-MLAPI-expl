#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import itertools
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# In[ ]:


print(df.shape)
print(df.columns)


# In[ ]:


df.info()


# In[ ]:


df['EmployeeCount'].unique()


# In[ ]:


df.drop(['EmployeeNumber', 'EmployeeCount'], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(df[df['Attrition'] == 'Yes']['Age'], label='Attrition')
sns.distplot(df[df['Attrition'] == 'No']['Age'], label='Non Attrition')
plt.xlabel('Age',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.title('Distribution of Age',fontsize=20);
plt.legend()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(df.loc[df['Attrition'] == 'Yes', 'MonthlyIncome'], label='Attrition')
sns.distplot(df.loc[df['Attrition'] == 'No', 'MonthlyIncome'], label='Non Attrition')
plt.xlabel('Monthly Income',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.title('Distribution of Monthly Income',fontsize=20);
plt.legend()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Gender', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Department', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='MaritalStatus', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='BusinessTravel', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='PerformanceRating', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='RelationshipSatisfaction', hue='Attrition', data=df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='YearsAtCompany', hue='Attrition', data=df)


# In[ ]:


fig = plt.figure(figsize=(12,6))
sns.boxplot(x="Gender",y="MonthlyIncome",data=df, hue="Attrition")
plt.title("Monthly Income - Gender", fontsize=20)
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Monthly Income", fontsize=15)


# In[ ]:


fig = plt.figure(figsize=(12,6))
sns.boxplot(x="MaritalStatus",y="MonthlyIncome",data=df, hue="Attrition")
plt.title("Monthly Income - Marital Status", fontsize=20)
plt.xlabel("Marital Status", fontsize=15)
plt.ylabel("Monthly Income", fontsize=15)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='PercentSalaryHike', hue='Attrition', data=df)


# In[ ]:


fig = plt.figure(figsize=(12,6))
sns.boxplot(x="Attrition",y="YearsSinceLastPromotion",data=df)
plt.title("Promotion effect on Attrition", fontsize=20)
plt.xlabel("Attrition", fontsize=15)
plt.ylabel("Years since Last Promotion", fontsize=15)


# In[ ]:


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df.head()


# In[ ]:


df = pd.get_dummies(df)
df.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Attrition', axis=1), df['Attrition'], 
                                                    test_size=0.15, random_state=0)


# In[ ]:


rf = RandomForestClassifier(n_estimators=30)
rf.fit(X_train, y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


def print_score(y_test, y_pred):
    print('Accuracy score {}'.format(accuracy_score(y_test, y_pred)))
    print('Recall score {}'.format(recall_score(y_test, y_pred)))
    print('Precision score {}'.format(precision_score(y_test, y_pred)))
    print('F1 score {}'.format(f1_score(y_test, y_pred)))


# In[ ]:


print_score(y_test, y_pred)


# In[ ]:


feat_importances = pd.Series(rf.feature_importances_, index=df.drop('Attrition', axis=1).columns)
print(feat_importances.sort_values(ascending=False))
feat_importances.sort_values(ascending=False).nlargest(10).plot(kind='barh')


# In[ ]:


def plot_confusion_matrix(cm, classes):
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm=cm, classes=["Non Attrition", "Attrition"])
print (classification_report(y_test, y_pred))


# In[ ]:


from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('\nAfter OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("\nAfter OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[ ]:


rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)


# In[ ]:


print_score(y_test, y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm=cm, classes=["Non Attrition", "Attrition"])
print (classification_report(y_test, y_pred))


# In[ ]:




