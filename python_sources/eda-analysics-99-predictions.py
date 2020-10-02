#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['gender'])
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Compare of Male and Female")


# In[ ]:


df['race/ethnicity'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Race vs Test Preparation Course")
sns.countplot(df['race/ethnicity'],hue=df['test preparation course'])
plt.xlabel("Group")
plt.ylabel("Count")


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['lunch'])
plt.xlabel("Lunch")
plt.ylabel("Count of Lunch")
plt.title("Comperation of Standard and Free Lunch")


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['parental level of education'])
plt.xlabel("Parent Education")
plt.ylabel("Count of Parent Education")
plt.title("Comperation of Parent Education")


# In[ ]:


plt.rcParams['figure.figsize'] = (18, 9)
plt.style.use('tableau-colorblind10')
sns.countplot(df['math score'])


# In[ ]:


sns.countplot(df['gender'],hue=df['race/ethnicity'])


# In[ ]:


sns.countplot(df['test preparation course'],hue=df['race/ethnicity'])


# In[ ]:


df['mathpass'] = np.where(df['math score']<40,'Fail','Pass')
df['mathpass'].value_counts().plot.pie()
plt.xlabel("State")
plt.ylabel("Count")
plt.title("Pass/Fail in Math")


# In[ ]:


sns.distplot(df['math score'])


# In[ ]:


df['readingpass'] = np.where(df['reading score']>40,'Pass','Fail')
df['readingpass'].value_counts().plot.pie()
plt.xlabel("State")
plt.ylabel("Count")
plt.title("Pass/Fail in Reading")


# In[ ]:


df['writingpass'] = np.where(df['writing score']>40,'Pass','Fail')
df['writingpass'].value_counts().plot.pie()
plt.xlabel("State")
plt.ylabel("Count")
plt.title("Pass/Fail in Writing")


# In[ ]:


df['totalScore'] = df['math score']+df['reading score']+df['writing score']
sns.distplot(df['totalScore'])


# In[ ]:


df['status'] = df.apply(lambda x : 'Fail' if x['mathpass'] == 'Fail' or 
                           x['readingpass'] == 'Fail' or x['writingpass'] == 'Fail'
                           else 'pass', axis = 1)
df['status'].value_counts().plot.pie()


# In[ ]:


from math import ceil
df['percentage'] = df['totalScore']/3

for i in range(0, 1000):
    df['percentage'][i] = ceil(df['percentage'][i])


# In[ ]:


df[['percentage','totalScore']]


# In[ ]:


sns.distplot(df['percentage'])


# In[ ]:


def grade(percentage, status):
  if status == 'Fail':
    return 'E'
  if(percentage >= 90):
    return 'O'
  if(percentage >= 80):
    return 'A'
  if(percentage >= 70):
    return 'B'
  if(percentage >= 60):
    return 'C'
  if(percentage >= 40):
    return 'D'
  else :
    return 'E'

df['grades'] = df.apply(lambda x: grade(x['percentage'], x['status']), axis = 1 )


# In[ ]:


df['grades'].value_counts().plot.pie()


# In[ ]:


pd.crosstab(df['parental level of education'],df['grades']).plot.bar(stacked=True)
plt.xlabel("Student Grades")
plt.ylabel("Counts")
plt.title("Parent Education VS Student Grade")


# In[ ]:


sns.countplot(df['grades'],hue=df['gender'])
plt.title("Grades VS Gender")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
ln = LabelEncoder()
object_columns = df.select_dtypes('object').columns
for i in object_columns:
    df[i] = ln.fit_transform(df[i])


# In[ ]:


X  = df.drop('percentage',axis=1)
y = df['percentage']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 45)


# In[ ]:


from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,r2_score


# In[ ]:


log = LinearRegression()
log.fit(x_train,y_train)
y_predict = log.predict(x_test)
r2_score(y_predict,y_test)


# In[ ]:


log = LogisticRegression()
log.fit(x_train,y_train)
y_predict = log.predict(x_test)
accuracy_score(y_predict,y_test)


# In[ ]:


log = DecisionTreeClassifier()
log.fit(x_train,y_train)
y_predict = log.predict(x_test)
accuracy_score(y_predict,y_test)

