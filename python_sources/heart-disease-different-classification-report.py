#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[114]:


heart  = pd.read_csv('../input/heart.csv')
heart.shape


# In[115]:


rename = {'cp':'chest_pain'}
heart = heart.rename(rename,axis=1)


# In[116]:


sns.countplot(x='chest_pain', data= heart, hue='sex')


# In[117]:


print(heart['chest_pain'].value_counts())
print(heart['thal'].value_counts())


# In[118]:


no_diease = len(heart[heart.target==0])
yes_diease = len(heart[heart.target==1])

print('Total Percent of People who aren:t suffering from disease {:.2f}%'.format(no_diease/len(heart.target)*100))
print('Total Percent of People who are suffering from disease {:.2f}%'.format(yes_diease/len(heart.target)*100))


# In[119]:


sns.countplot(x='sex', data= heart)
plt.xlabel({0:'Female', 1:'Male'})


# In[120]:


female_patient = len(heart[heart['sex']==0])
male_patient = len(heart[heart['sex']==1])

print('Total Number of female Patients {:.2f}%'.format(female_patient/len(heart['sex'])*100))
print('Total Number of Male Patients {:.2f}%'.format(male_patient/len(heart['sex'])*100))


# In[121]:


plt.figure(figsize=(8,5))
sns.countplot(x='sex',data = heart, hue='target')
plt.xlabel(xlabel='Sex (0 - Female, 1 - Male)')
plt.ylabel('Frequency')
plt.legend(['No Disease', 'Yes Disease'])


# In[122]:


pd.crosstab(heart.sex, heart.target)


# In[123]:


sns.distplot(heart.age)
#sns.distplot(heart.target)


# In[124]:


sns.distplot(heart.thalach)


# In[125]:


sns.jointplot(x='age',y='thalach', data = heart, kind='hex')


# In[126]:


pd.crosstab(heart.sex, heart.target)


# In[127]:


pd.crosstab(heart.sex,heart.chest_pain)


# In[128]:


g= sns.catplot(x='chest_pain', data  = heart, hue='sex', col='target', kind='count', aspect=1, height=7)
plt.legend(['Female', 'Male'])
g.fig.suptitle('Chest_Pain vs Sex vs Target', fontsize=16)
(g.set_axis_labels("Types Of Chest Pain", "Frequency")
 .set_xticklabels(["typical angina", "atypical angina", "non-anginal pain","asymptomatic"])
 .set_titles("{col_name} {col_var}")
 .despine(left=True))


# In[129]:


chest_p = pd.get_dummies(heart['chest_pain'],prefix='chest_pain')
thals = pd.get_dummies(heart['thal'],prefix='thal')
slopss = pd.get_dummies(heart['slope'],prefix='slope')


# In[130]:


dataframes = [heart,chest_p,thals,slopss]
heart = pd.concat(dataframes, axis=1)


# In[131]:


remove = ['chest_pain', 'thal','slope']
heart = heart.drop(remove, axis=1)


# In[132]:


X = heart.drop('target', axis=1)
y= heart.target


# In[133]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[134]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
accu_lr = accuracy_score(y_pred,y_test)
print('Accuracy of Logistic Regression {:.2f}%'.format(accu_lr*100))


# In[135]:


from sklearn.svm import SVC
svc = SVC(random_state= 1)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
accu_svc = accuracy_score(y_pred_svc, y_test)
print('Accuracy of SVC {:.2f}%'.format(accu_svc*100))


# In[136]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)
accu_nb = accuracy_score(y_pred_nb, y_test)
print('Accuracy of Naive Bayes {:.2f}%'.format(accu_nb*100))


# In[137]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)
accu_dtc = accuracy_score(y_pred_dtc, y_test)
print('Accuracy of SVC {:.2f}%'.format(accu_dtc*100))


# In[138]:


from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,y_pred)
cm_svm = confusion_matrix(y_test,y_pred_svc)
cm_nb = confusion_matrix(y_test,y_pred_nb)
cm_dtc = confusion_matrix(y_test,y_pred_dtc)


# In[139]:


plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
sns.heatmap(cm_lr,annot=True)
plt.title('Logistic Regression')
plt.subplot(2,2,2)
sns.heatmap(cm_nb,annot=True)
plt.title('Naive Bayes')
plt.subplot(2,2,3)
sns.heatmap(cm_svm,annot=True)
plt.title('SVM')
plt.subplot(2,2,4)
sns.heatmap(cm_dtc,annot=True)
plt.title('Decision Tree')

