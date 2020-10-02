#!/usr/bin/env python
# coding: utf-8

# Attribute Information:
# 
# Only 14 attributes used: 
# 1. age : Patient age in years
# 2. sex : sex (1 = male; 0 = female)
# 3. cp : cp: chest pain type 
#                 -- Value 1: typical angina 
#                 -- Value 2: atypical angina 
#                 -- Value 3: non-anginal pain 
#                 -- Value 4: asymptomatic
# 4. trestbps : resting blood pressure (in mm Hg on admission to the hospital) 
# 5. chol : serum cholestoral in mg/dl 
# 6. fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg : resting electrocardiographic results 
#                 -- Value 0: normal 
#                 -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
#                 -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
# 8. thalach :  maximum heart rate achieved 
# 9. exang : exercise induced angina (1 = yes; 0 = no) 
# 10. oldpeak : ST depression induced by exercise relative to rest 
# 11. slope : the slope of the peak exercise ST segment 
#                -- Value 1: upsloping 
#                -- Value 2: flat 
#                -- Value 3: downsloping 
# 12. ca : number of major vessels (0-3) colored by flourosopy 
# 13. thal :  3 = normal; 6 = fixed defect; 7 = reversable defect 
# 14. num (the predicted attribute) : diagnosis of heart disease (angiographic disease status) 
#            -- Value 0: < 50% diameter narrowing 
#            -- Value 1: > 50% diameter narrowing 
# 
# 

# # Libraries Import

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# # **E.D.A. (Exploratory Data Analysis)**

# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,linewidths=2,)


# In[ ]:


sns.countplot(x='target',data=df,hue='sex')
plt.legend(['Female','Male'])
plt.xlabel('-ve Heart Diagnosis                   +ve Heart Diagnosis')


# In[ ]:


sns.pairplot(df,hue='target')


# In[ ]:


sns.distplot(df['age'],)


# In[ ]:


df['age'].describe()


# In[ ]:



df['Age_Category']= pd.cut(df['age'],bins=list(np.arange(25, 85, 5)))


# In[ ]:


plt.figure(figsize=(20,5))

plt.subplot(121)
df[df['target']==1].groupby('Age_Category')['age'].count().plot(kind='bar')
plt.title('Age Distribution of Patients with +ve Heart Diagonsis')

plt.subplot(122)
df[df['target']==0].groupby('Age_Category')['age'].count().plot(kind='bar')
plt.title('Age Distribution of Patients with -ve Heart Diagonsis')


# In[ ]:





# In[ ]:


df.nunique()


# In[ ]:


plt.figure(figsize=(6,5))
sns.countplot(x='cp',data=df,hue='target')
plt.xlabel('typical angina     atypical angina     non-anginal pain     asymptomatic')
plt.ylabel('Counts of Chest pain Type')


# In[ ]:


plt.figure(figsize=(7,5))
sns.countplot(x='fbs',data=df,hue='target')
plt.xlabel('fasting blood sugar < 120 mg/dl        fasting blood sugar > 120 mg/dl')
plt.ylabel('Count')


# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(x='exang',data=df,hue='target')
plt.xlabel('No                                       Yes')
plt.ylabel('Count')
plt.title('Exercise induced angina')


# In[ ]:


df = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal','Age_Category'])


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.drop(['age'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


y = df['target']
X = df.drop(['target'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
trees = DecisionTreeClassifier()


# In[ ]:


trees.fit(X_train,y_train)


# In[ ]:


y_pred = trees.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=750)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test) 


# In[ ]:


print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

