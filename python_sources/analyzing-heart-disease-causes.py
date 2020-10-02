#!/usr/bin/env python
# coding: utf-8
age: The person's age in years
sex: The person's sex (1 = male, 0 = female)
cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
chol: The person's cholesterol measurement in mg/dl
fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
thalach: The person's maximum heart rate achieved
exang: Exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
ca: The number of major vessels (0-3)
thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
target: Heart disease (0 = no, 1 = yes)
# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# In[ ]:


data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['cp'][data['cp'] == 1] = 'typical angina'
data['cp'][data['cp'] == 2] = 'atypical angina'
data['cp'][data['cp'] == 3] = 'non-anginal pain'
data['cp'][data['cp'] == 4] = 'asymptomatic'

data['fbs'][data['fbs'] == 0] = 'lower than 120mg/ml'
data['fbs'][data['fbs'] == 1] = 'greater than 120mg/ml'

data['restecg'][data['restecg'] == 0] = 'normal'
data['restecg'][data['restecg'] == 1] = 'ST-T wave abnormality'
data['restecg'][data['restecg'] == 2] = 'left ventricular hypertrophy'

data['exang'][data['exang'] == 0] = 'no'
data['exang'][data['exang'] == 1] = 'yes'

data['slope'][data['slope'] == 1] = 'upsloping'
data['slope'][data['slope'] == 2] = 'flat'
data['slope'][data['slope'] == 3] = 'downsloping'

data['thal'][data['thal'] == 1] = 'normal'
data['thal'][data['thal'] == 2] = 'fixed defect'
data['thal'][data['thal'] == 3] = 'reversable defect'


# In[ ]:


data['sex'] = data['sex'].astype('object')
data['cp'] = data['cp'].astype('object')
data['fbs'] = data['fbs'].astype('object')
data['restecg'] = data['restecg'].astype('object')
data['exang'] = data['exang'].astype('object')
data['slope'] = data['slope'].astype('object')
data['thal'] = data['thal'].astype('object')


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


sns.countplot('target', data=data)


# In[ ]:


sns.countplot('target', data=data, hue='sex', palette="Set1")


# In[ ]:


data[['target', 'sex']].groupby(['sex'], as_index=False).mean().sort_values(by='sex', ascending=False)


# In[ ]:


data['age'].hist()


# In[ ]:


sns.distplot(data['age'], color = 'red')


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot('age', hue='target', data=data)


# In[ ]:


sns.swarmplot('target', 'age', data=data)


# In[ ]:


sns.swarmplot('target', 'chol', data=data)


# In[ ]:


sns.countplot('target', hue='ca', data=data)


# In[ ]:


data.groupby(['target','ca']).size().unstack().plot(kind='bar', stacked=True, figsize=(10,8))
plt.show()


# In[ ]:


sns.countplot('target', hue='thal', data=data)


# In[ ]:


plt.figure( figsize=(20,8))
plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'red')


# In[ ]:


label = data['target']


# In[ ]:


label.unique()


# In[ ]:


label.value_counts()


# In[ ]:


data=data.drop(['target'], axis=1)


# In[ ]:


data.head()


# In[ ]:


label.shape


# In[ ]:


data = pd.get_dummies(data, drop_first=True)


# In[ ]:


data.head()


# In[ ]:


x = data


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, label, test_size = 0.2)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


mod1 = RandomForestClassifier()
mod1.fit(x_train, y_train)


# In[ ]:


mod2 = DecisionTreeClassifier()
mod2.fit(x_train, y_train)


# In[ ]:


pred_1 = mod1.predict(x_test)
pred_quant1 = mod1.predict_proba(x_test)[:, 1]
pred1 = mod1.predict(x_test)

pred_2 = mod2.predict(x_test)
pred_quant2 = mod2.predict_proba(x_test)[:, 1]
pred2 = mod2.predict(x_test)


# In[ ]:


score1_train=mod1.score(x_train, y_train)
print(f'Training Random Forest: {round(score1_train*100,2)}%')

score1_test=mod1.score(x_test,y_test)
print(f'Testing Random Forest: {round(score1_test*100,2)}%')


# In[ ]:


score2_train=mod2.score(x_train, y_train)
print(f'Training Decision Tree: {round(score2_train*100,2)}%')

score2_test=mod2.score(x_test,y_test)
print(f'Testing Decision Tree: {round(score2_test*100,2)}%')


# In[ ]:


from sklearn.metrics  import confusion_matrix


# In[ ]:


confusion_matrix(y_test, pred1)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, pred1), annot=True)


# In[ ]:


confusion_matrix(y_test, pred2)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, pred2), annot=True)


# In[ ]:


y_pred_quant1 = mod1.predict_proba(x_test)[:, 1]


# In[ ]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant1)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['figure.figsize'] = (15, 5)
plt.title('ROC curve for diabetes classifier', fontweight = 30)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()


# In[ ]:


import eli5 
from eli5.sklearn import PermutationImportance
perm1 = PermutationImportance(mod1, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm1, feature_names = x_test.columns.tolist())


# In[ ]:


perm2 = PermutationImportance(mod2, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm2, feature_names = x_test.columns.tolist())


# In[ ]:




