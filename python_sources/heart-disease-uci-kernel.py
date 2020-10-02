#!/usr/bin/env python
# coding: utf-8

# # An exploratory study of Heart Desease

# ## Introduction
# 
# The dataset is provided by [UCI Machine Learning Repository][1], the original database contains 76 attributes, but all published experiments refer to using a subset of 14 of them:
# 
# 1. Age: The person's age in years
# 2. Sex: The person's sex (1=male, 0=female)
# 3. Chest Pain type (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)*
# 4. Resting Blood Pressure (mm Hg on admission to the hospital)
# 5. Serum Cholestoral in mg/dl
# 6. Fasting Blood Sugar (> 120 mg/dl, 1 = true; 0 = false)
# 7. Resting Electrocardiographic results  (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)*
# 8. Maximum Heart Rate achieved 
# 9. Exercise Induced Angina (1 = yes; 0 = no)
# 10. Oldpeak = ST depression induced by exercise relative to rest 
# 11. The slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)*
# 12. Number of Major Vessels (0-3) colored by flourosopy 
# 13. Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)*
# 14. Heart disease (0 = no, 1 = yes)*
# 
# 
# *according to this [post][2] on the discussion of this dataset we have some problems. So here it goes the CORRECT description of the kaggle dataset:
# 
#     cp: chest pain type
#     -- Value 0: asymptomatic
#     -- Value 1: atypical angina
#     -- Value 2: non-anginal pain
#     -- Value 3: typical angina
# 
#     restecg: resting electrocardiographic results
#     -- Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria 
#     -- Value 1: normal
#     -- Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 
#     slope: the slope of the peak exercise ST segment
#     -- 0: downsloping
#     -- 1: flat
#     -- 2: upsloping
# 
#     thal: 
#     -- 1 = fixed defect
#     -- 2 = normal 
#     -- 3 = reversable defect
# 
#     target (maybe THE most important feature): 0 = disease, 1 = no disease
# 
#     A few more things to consider:
#     data #92, 158, 164, 163, 164 and 251 have ca=4 which is incorrect. In the original Cleveland dataset they are NaNs (so they should be removed)
#     data #48 and 281 have thal = 0, also incorrect. They are also NaNs in the original dataset.
# 
# 
# [1]:https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# [2]:https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877
# 
# 

# ## Steps of this Kernel
# 1. Check missing data
# 2. Fix the Dataset
# 3. Exploratory analysis

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/heart.csv')

data.head()


# In[ ]:


data.isnull().sum()


# No missing data, let's proceed to the next step.

# For correcting the database we should follow this:
# 
#     data #92, 158, 163, 164 and 251 have ca=4 which is incorrect. In the original Cleveland dataset they are NaNs (so they should be removed)
#     data #48 and 281 have thal = 0, also incorrect. They are also NaNs in the original dataset.

# In[ ]:


data.drop([48,92,158,163,164,251,281], inplace=True)


# Now that we fixed the dataset let's go to the next step.

# In[ ]:


data2 = data.copy()

data2.columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate Achieved',
       'Exercise Induced Angina', 'st Depression', 'st Slope', 'Number Major Vessels', 'Thalassemia', 'Heart Desease']

data2.replace({'Sex':{0:'Female',1:'Male'},
              'Chest Pain Type':{0:'Asymptomatic',1:'Atypical Angina',2:'Non-Anginal Pain',3:'Typical Angina'},
              'Fasting Blood Sugar':{1:'> 120 mg/dl',0:'< 120 mg/dl'},
              'Rest ECG':{0:'Left Ventricular Hypertrophy',1:'Normal',2:'ST-T wave abnormality'},
              'Exercise Induced Angina':{1:'Yes',0:'No'},
              'st Slope':{0:'Downsloping',1:'Flat',2:'Upsloping'},
              'Thalassemia':{1:'Fixed Defect',2:'Normal',3:'Reversable Defect'},
              'Heart Desease':{0:'Yes',1:'No'}
              },inplace=True)


# ## Analysing Age Influence on the other Variables

# In[ ]:


sns.set(rc={'figure.figsize':(8,6)},style='white',palette="Reds_d")


# First let's see the correaltion Age has with the other variables.

# In[ ]:


ax = sns.heatmap(data2.corr().round(2),annot=True)


# As we can see, age have some positive correaltion with Number of Major Vessels, Resting Blood Pressure, Cholesterol and st Depression. And a negative correlation with Max Heart Rate Achieved.

# In[ ]:


ax = sns.distplot(data2.Age)
pd.DataFrame(data2.Age.describe().round(2)).transpose()


# In[ ]:


ax = sns.kdeplot(data2[data2['Sex']=='Male']['Age'],shade=True)
ax = sns.kdeplot(data2[data2['Sex']=='Female']['Age'],shade=True)
ax = ax.set_xlabel('Age')
plt.legend(['Male','Female'])

data2.groupby('Sex').Age.describe().round(2)


# In[ ]:


ax = sns.boxplot(data=data2, x='Chest Pain Type', y='Age', width=0.5)
data2.groupby('Chest Pain Type').Age.describe().round(2)


# In[ ]:


ax = sns.lmplot(data=data2, x='Age', y='Resting Blood Pressure')
data2.groupby(pd.cut(data2["Age"],5))['Resting Blood Pressure'].describe()


# In[ ]:


ax = sns.lmplot(data=data2, x='Age', y='Cholesterol')
data2.groupby(pd.cut(data2["Age"],5))['Cholesterol'].describe()


# In[ ]:


ax = sns.boxplot(data=data2, x='Fasting Blood Sugar', y='Age', width=0.5)
data2.groupby('Fasting Blood Sugar').Age.describe().round(2)


# In[ ]:


ax = sns.boxplot(data=data2, x='Rest ECG', y='Age', width=0.5)
data2.groupby('Rest ECG').Age.describe().round(2)


# In[ ]:


ax = sns.lmplot(data=data2, x='Age', y='Max Heart Rate Achieved')
data2.groupby(pd.cut(data2["Age"],5))['Max Heart Rate Achieved'].describe()


# In[ ]:


ax = sns.boxplot(data=data2, x='Exercise Induced Angina', y='Age', width=0.5)
data2.groupby('Exercise Induced Angina').Age.describe().round(2)


# In[ ]:


ax = sns.lmplot(data=data2, x='Age', y='st Depression')
data2.groupby(pd.cut(data2["Age"],5))['st Depression'].describe()


# In[ ]:


ax = sns.boxplot(data=data2, x='st Slope', y='Age', width=0.5)
data2.groupby('st Slope').Age.describe().round(2)


# In[ ]:


ax = sns.boxplot(data=data2, x='Number Major Vessels', y='Age', width=0.5)
data2.groupby('Number Major Vessels').Age.describe().round(2)


# In[ ]:


ax = sns.boxplot(data=data2, x='Thalassemia', y='Age', width=0.5)
data2.groupby('Thalassemia').Age.describe().round(2)


# In[ ]:


ax = sns.boxplot(data=data2, x='Heart Desease', y='Age', width=0.5)
data2.groupby('Heart Desease').Age.describe().round(2)


# From the data obtained by the above graphs, we can say that Age has some weak impact on the other variables.

# ## Analysing the cause of Heart Desease

# In[ ]:


ax = sns.countplot(data=data2,x="Sex",hue="Heart Desease")


# In[ ]:


print(round(len(data2[(data2['Sex']=='Male') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Sex']=='Male')]),2))
print(round(len(data2[(data2['Sex']=='Female') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Sex']=='Female')]),2))


# Proportionally Man have more Heart Desease.

# In[ ]:


ax = sns.kdeplot(data2[data2['Heart Desease']=='Yes']['Age'],shade=True)
ax = sns.kdeplot(data2[data2['Heart Desease']=='No']['Age'],shade=True)
ax = ax.set_xlabel('Age')
ax = plt.legend(['Yes','No'])

data2.groupby('Heart Desease').Age.describe().round(2)


# We can see that heart deseases occur more on people with 60 years old

# In[ ]:


ax = sns.countplot(data=data2,x="Chest Pain Type",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Chest Pain Type']=='Typical Angina') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Typical Angina')]))
print(len(data2[(data2['Chest Pain Type']=='Non-Anginal Pain') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Non-Anginal Pain')]))
print(len(data2[(data2['Chest Pain Type']=='Atypical Angina') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Atypical Angina')]))
print(len(data2[(data2['Chest Pain Type']=='Asymptomatic') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Asymptomatic')]))


# Proportionally the Chest Pain: Asymptomatic have more Heart Deseases.

# In[ ]:


ax = sns.boxplot(data=data2,x='Heart Desease', y='Resting Blood Pressure')
data2.groupby('Heart Desease')['Resting Blood Pressure'].describe().round(2)


# Resting Blood Presure doesnt influence on HEart Desease.

# In[ ]:


ax = sns.boxplot(data=data2,x='Heart Desease', y='Cholesterol')
data2.groupby('Heart Desease')['Cholesterol'].describe().round(2)


# Cholesterol doesnt influence on heart desease.

# In[ ]:


ax = sns.countplot(data=data2,x="Fasting Blood Sugar",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Fasting Blood Sugar']=='> 120 mg/dl') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Fasting Blood Sugar']=='> 120 mg/dl')]))
print(len(data2[(data2['Fasting Blood Sugar']=='< 120 mg/dl') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Fasting Blood Sugar']=='< 120 mg/dl')]))


# No statistical difference.

# In[ ]:


ax = sns.countplot(data=data2,x="Rest ECG",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Rest ECG']=='Left Ventricular Hypertrophy') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='Left Ventricular Hypertrophy')]))
print(len(data2[(data2['Rest ECG']=='Normal') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='Normal')]))
print(len(data2[(data2['Rest ECG']=='ST-T wave abnormality') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='ST-T wave abnormality')]))


# Proportionally the Rest EEG: ST-T wave abnormality have more Heart Deseases.

# In[ ]:


ax = sns.boxplot(data=data2,x='Heart Desease', y='Max Heart Rate Achieved')
data2.groupby('Heart Desease')['Max Heart Rate Achieved'].describe().round(2)


# Max Heart Rate Achieved have lower values to Heart Desease.

# In[ ]:


ax = sns.countplot(data=data2,x="Exercise Induced Angina",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Exercise Induced Angina']=='No') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Exercise Induced Angina']=='No')]))
print(len(data2[(data2['Exercise Induced Angina']=='Yes') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Exercise Induced Angina']=='Yes')]))


# Proportionally Exercise Induced Angina have more Heart deseases.

# In[ ]:


ax = sns.boxplot(data=data2,x='Heart Desease', y='st Depression')
data2.groupby('Heart Desease')['st Depression'].describe().round(2)


# St Depression have higher values to Heart Desease

# In[ ]:


ax = sns.countplot(data=data2,x="st Slope",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['st Slope']=='Downsloping') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Downsloping')]))
print(len(data2[(data2['st Slope']=='Upsloping') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Upsloping')]))
print(len(data2[(data2['st Slope']=='Flat') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Flat')]))


# Proportionally st Slope of type Flat have more Heart Deseases.

# In[ ]:


ax = sns.countplot(data=data2,x="Number Major Vessels",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Number Major Vessels']==3) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==3)]))
print(len(data2[(data2['Number Major Vessels']==2) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==2)]))
print(len(data2[(data2['Number Major Vessels']==1) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==1)]))
print(len(data2[(data2['Number Major Vessels']==0) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==0)]))


# Proportionally 3 and 2 Major Vessels have more Heart Deseases.

# In[ ]:


ax = sns.countplot(data=data2,x="Thalassemia",hue="Heart Desease")


# In[ ]:


print(len(data2[(data2['Thalassemia']=='Fixed Defect') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Fixed Defect')]))
print(len(data2[(data2['Thalassemia']=='Normal') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Normal')]))
print(len(data2[(data2['Thalassemia']=='Reversable Defect') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Reversable Defect')]))


# Proportionally Thalassemia of Reversable effect have more Heart Desease.

# So, we can say the variables that influences in Heart Desease:
# > Sex
# 
# > Age
# 
# >Chest Pain
# 
# >Rest EEG
# 
# >Max HEart Rate Achieved
# 
# >Exercise Induced Angina
# 
# >St Depression
# 
# >st Slope
# 
# >Major Vessels
# 
# >Thalassemia

# In[ ]:


data2.drop(['Resting Blood Pressure','Cholesterol', 'Fasting Blood Sugar'],axis=1,inplace=True)

y = data2['Heart Desease']

data2.drop(['Heart Desease'],axis=1, inplace=True)

x = data2


# In[ ]:


y.replace({'Heart Desease':{'Yes':0,'No':1}},inplace=True)


# In[ ]:


x = pd.get_dummies(x)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))


clf.fit(X_train,y_train)
print("Test score: {:.2f}".format(accuracy_score(y_test,clf.predict(X_test))))
print("Cohen Kappa score: {:.2f}".format(cohen_kappa_score(y_test,clf.predict(X_test))))
ax = sns.heatmap(confusion_matrix(y_test,clf.predict(X_test)),annot=True)
ax= ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix')


# In[ ]:




