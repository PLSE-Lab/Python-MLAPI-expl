#!/usr/bin/env python
# coding: utf-8

# <h1>Analysis of Titanic Disaster</h1>

# ![](https://img-s2.onedio.com/id-5bda2344d3904b96297c7d81/rev-0/raw/s-2e1fb20763765bc9ea221d742a17ee693be19558.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
df_gender=pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


print("================TRAIN INFO=========================")
df_train.info()


# In[ ]:


print("============TRAIN COLUMNS==================")
df_train.columns


# In[ ]:


df_train.columns=["passid","survived","pclass","name","gender","age","sibsp","parch","ticket","fare","cabin","embarked"]


# In[ ]:


df_train.columns


# In[ ]:


df_train.shape


# In[ ]:


print("=============TRAIN HEAD===================")
df_train.head()


# In[ ]:


print("=============TRAIN TAIL====================")
df_train.tail()


# In[ ]:


df_train.dtypes


# In[ ]:


df_train.corr()


# In[ ]:


print("================TEST INFO============================")
df_test.info()


# In[ ]:


print("================TEST COLUMNS====================")
df_test.columns


# In[ ]:


df_test.columns=["passid","pclass","name","gender","age","sibsp","parch","ticket","fare","cabin","embarked"]


# In[ ]:


df_test.columns


# In[ ]:


df_test.shape


# In[ ]:


print("=================TEST HEAD=================")
df_test.head()


# In[ ]:


print("===============TEST TAIL====================")
df_test.tail()


# In[ ]:


df_test.dtypes


# In[ ]:


print("================GENDER SUBMISSION INFO============================")
df_gender.info()


# In[ ]:


print("================GENDER SUBMISSION COLUMNS======================")
df_gender.columns


# In[ ]:


df_gender.columns=["passid","survived"]


# In[ ]:


df_gender.columns


# In[ ]:


df_gender.shape


# In[ ]:


print("===========GENDER SUBMISSION HEAD====================")
df_gender.head()


# In[ ]:


print("================GENDER SUBMISSION TAIL=======================")
df_gender.tail()


# In[ ]:


df_gender.dtypes


# In[ ]:


print("===================TRAIN NAN VALUES====================")
print("=========AGE NAN==============")
print(df_train["age"].value_counts(dropna=False)) #177 NaN 
print("=========CABIN NAN=====================")
print(df_train["cabin"].value_counts(dropna=False)) #687 NaN
print("========EMBARKED NAN====================")
print(df_train["embarked"].value_counts(dropna=False)) #2 NaN


# In[ ]:


print("======================TEST NaN VALUES=======================")
print("========================AGE NaN VALUES======================")
print(df_test["age"].value_counts(dropna=False))#86 NaN Values
print("===================FARE NaN VALUES==========================")
print(df_test["fare"].value_counts(dropna=False))#1 NaN Value
print("====================CABIN NaN VALUES========================")
print(df_test["cabin"].value_counts(dropna=False)) #327 NaN Value


# In[ ]:


df_train.describe()


# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df_train.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()


# In[ ]:


fig,axes=plt.subplots(nrows=2,ncols=1)
df_train.plot(kind='hist',y='age',bins=50,range=(0,100),normed=True,ax=axes[0])
df_train.plot(kind='hist',y='age',bins=50,range=(0,100),normed=True,ax=axes[1],cumulative=True)
plt.show()


# In[ ]:


print(df_train['gender'].value_counts(dropna=False))


# In[ ]:


sns.barplot(x='gender',y='age',data=df_train)
plt.show()


# In[ ]:


df_train['age']=df_train['age']
bins=[29,45,59,np.inf]
labels=["Young Adult","Middle-Aged Adults","Old Adults"]
df_train['age_group']=pd.cut(df_train['age'],bins,labels=labels)
sns.barplot(x='age_group',y='survived',data=df_train)
plt.show()


# In[ ]:


sns.barplot(x='gender',y='survived',data=df_train)
plt.show()


# In[ ]:


df_train.pclass.unique()


# In[ ]:


sns.barplot(x='pclass',y='age',data=df_train)
plt.show()


# In[ ]:


sns.barplot(x='survived',y='age',data=df_train)
plt.show()


# In[ ]:


df_train.embarked.unique()


# In[ ]:


sns.barplot(x='embarked',y='survived',data=df_train)
plt.show()


# In[ ]:


df_train.sibsp.unique()


# In[ ]:


sns.barplot(x='sibsp',y='age',data=df_train)
plt.show()


# In[ ]:


sns.barplot(x='gender',y='sibsp',data=df_train)
plt.show()


# In[ ]:


df_train.parch.unique()


# In[ ]:


sns.barplot(x='parch',y='age',data=df_train)
plt.show()


# In[ ]:


sns.barplot(x='parch',y='sibsp',data=df_train)
plt.show()


# In[ ]:


df_train.cabin.unique()


# In[ ]:


plt.figure(figsize=(15,30))
sns.barplot(x='pclass',y='cabin',data=df_train)
plt.show()


# #Second class -> **E77, D, F4, F2, E101, F33, D56**
# #Third class -> **G6,FG73, FE69, E10, FG63, E121,F38**
# 

# In[ ]:


sum(df_train.pclass==3)


# In[ ]:


sum(df_train.pclass==2)


# In[ ]:


sum(df_train.pclass==1)


# In[ ]:


sns.barplot(x='pclass',y='survived',data=df_train)
plt.show()


# ![](http://img07.deviantart.net/af19/i/2014/307/c/f/r_m_s__titanic_class_system_by_monroegerman-d787jna.png)

# In[ ]:


plt.figure(figsize=(15,30))
result = df_train.groupby(["cabin"])['survived'].aggregate(np.median).reset_index().sort_values('survived')
sns.barplot(x='survived', y="cabin", data=df_train, order=result['cabin'])
plt.title('cabin-survived')
plt.show()


# In[ ]:


df_train.head()


# In[ ]:


# Most common 15 Name or Surname of dying people
separate = df_train.name.str.split() 
a,b,c = zip(*separate)                    
name_list = a+b+c
name_count = Counter(name_list)         
most_common_names = name_count.most_common(15)  
x,y = zip(*most_common_names)
x,y = list(x),list(y)
    
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of dying people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of dying people')
plt.show()


# In[ ]:


name_list1 = list(df_train['name'])
pclass = []
survived =[]
age=[]
parch=[]
sibsp=[]



for i in name_list1:
    x = df_train[df_train['name']==i]
    pclass.append(sum(x.pclass)/len(x))
    survived.append(sum(x.survived) / len(x))
    age.append(sum(x.age) / len(x))
    parch.append(sum(x.parch) / len(x))
    sibsp.append(sum(x.sibsp) / len(x))
    
# visualization

f,ax = plt.subplots(figsize = (15,220))

sns.barplot(x=age,y=name_list1,color='c',alpha = 0.6,label='age')
sns.barplot(x=survived,y=name_list1,color='m',alpha = 0.7,label='survived')
sns.barplot(x=pclass,y=name_list1,color='g',alpha = 0.5,label='pclass' )
sns.barplot(x=parch,y=name_list1,color='y',alpha = 0.6,label='parch')
sns.barplot(x=sibsp,y=name_list1,color='r',alpha = 0.6,label='sibsp')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of pclass, survived, age, parch, sibsp', ylabel='name',
       title = "Percentage of Name's According to pclass, survived, age, parch, sibsp ")
plt.show()


# <h1> - Multiple Linear Regression</h1>
# 
# <b>survived,age,pclass<b>
#     
#    Pclass: Passenger's class (1st, 2nd, or 3rd)

# In[ ]:


#missing values pclass
sum(df_train.pclass.isna())


# In[ ]:


#missing values survived
sum(df_train.survived.isna())


# In[ ]:


#missing values parch
sum(df_train.age.isna())


# In[ ]:


df_train.columns


# In[ ]:


df_train.dtypes


# In[ ]:


df_train['age']=df_train['age'].fillna(-0.5)


# In[ ]:


x=df_train.iloc[:,[2,5]].values#pclass,parch


# In[ ]:


y=df_train.survived.values.reshape(-1,1)


# In[ ]:


multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)


# In[ ]:


print("b0:",multiple_linear_regression.intercept_)


# In[ ]:


print("b1,b2:",multiple_linear_regression.coef_)


# In[ ]:


df_train.age.max()


# In[ ]:


#pclass:1 and age 80 vs. plcass:2 and age 80
multiple_linear_regression.predict(np.array([[1,80],[2,80]]))


# In[ ]:


#pclass:1 and age 80 vs. plcass:3 and age 80
multiple_linear_regression.predict(np.array([[1,80],[3,80]]))


# In[ ]:


#pclass:2 and age 80 vs. plcass:3 and age 80
multiple_linear_regression.predict(np.array([[2,80],[3,80]]))


# In[ ]:


df_train.age.mean()


# In[ ]:


#pclass:1 and age 23 vs. plcass:2 and age 23
multiple_linear_regression.predict(np.array([[1,23],[2,23]]))


# In[ ]:


#pclass:1 and age 23 vs. plcass:3 and age 23
multiple_linear_regression.predict(np.array([[1,23],[3,23]]))


# In[ ]:


#pclass:2 and age 23 vs. plcass:3 and age 23
multiple_linear_regression.predict(np.array([[2,23],[3,23]]))


# In[ ]:


print('end')


# 
