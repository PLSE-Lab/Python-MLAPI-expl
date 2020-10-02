#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:#84E1FB">
# <br><br><br>
# <div align="center"><span style="font-size:36px;color:#003BA3"><b>Titanic: Machine Learning from Disaster</b></span></div>
# <br>
# <div align="center"><span style="font-size:30px;color:#003BA3"><b>Regression logistic</b></span></div>
# <br><br><br>
# </div>

# Firtly we can import the libraries and the two datasets (train.csv and test.csv)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# # Overview

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print(train.info())
print("="*50)
print(test.info())


# <p>We're not going to take into account the <code>Cabin</code> and <code>Ticket</code> variables.</p>
# <p>We're going to analysis the variables following:</p>
#     <ul>
#         <li><code>Pclass</code></li>
#         <li><code>Sex</code></li> 
#         <li><code>Embarked</code></li>
#         <li><code>Name</code> : We're going to extract the honorary title.</li>
#         <li><code>SibSp</code></li>
#         <li><code>Parch</code></li>
#         <li><code>Fare</code> : There are one only missing value in the test dataset. We can to replace it by the mean.</li>
#         <li><code>Age</code> : We're going to fill the missing values by values predicted by a linear regression.</li>
#     </ul>  

# # Data cleaning

# ## The qualitative variables 

# ### The <code>Pclass</code> variable

# In[ ]:


pd.crosstab(train.Pclass,train.Survived,margins=True)


# ### The <code>Sex</code> variable

# In[ ]:


pd.crosstab(train.Sex,train.Survived,margins=True)


# ### The <code>Embarked</code> variable

# In[ ]:


train.Embarked.fillna('C',inplace=True)


# In[ ]:


pd.crosstab(train.Embarked,train.Survived,margins=True)


# ### Chart

# We create a function who allows to doing diagrams who distinc the suvived and the not survived 

# In[ ]:


def graph_bar(category):
        
    train_survived = train[train["Survived"]==1]
    nb_train_survived=train_survived.groupby(category)['PassengerId'].nunique()

    train_not_survived = train[train["Survived"]==0]
    nb_train_not_survived=train_not_survived.groupby(category)['PassengerId'].nunique()
    
    Survived = list(nb_train_survived)
    Notsurvived = list(nb_train_not_survived)

    N = range(len(Survived))

    p1=plt.bar(N, Survived,color='#BCF0C6')
    p2=plt.bar(N, Notsurvived,bottom=Survived,color='#F98888')

    groups=(list(train[category].unique()))
    groups.sort()        
    plt.xticks(N, groups)
    plt.title("Survivors by the {} variable".format(category))
    plt.xlabel("{}".format(category))
    plt.ylabel("Number of persons")
    plt.legend((p1[0], p2[0]), ('Survived', 'Not Survived'))


# In[ ]:


plt.figure(figsize=(12,12))
plt.subplot(2, 2, 1)  
graph_bar("Embarked")
plt.subplot(2, 2, 2)
graph_bar("Sex")
plt.subplot(2, 1, 2)  
graph_bar("Pclass")
plt.show()


# ### The <code>Name</code> variable
# 

# We're going to extract the honorary title.

# In[ ]:


def name_sep(data):
    titles=[]
    for i in range(len(data)):
        title=data.iloc[i]
        title=title.split('.')[0].split(' ')[-1]
        titles.append(title)
    return titles


# In[ ]:


train['Title']=name_sep(train.Name)
test['Title']=name_sep(test.Name)


# In[ ]:


train.Title.value_counts()


# ### The changes

# We transform the variable <code>Sex</code> in dummies.

# In[ ]:


train=train.replace(['female','male'],[0,1])
test=test.replace(['female','male'],[0,1])


# In[ ]:


train=pd.concat((
    train,
    pd.get_dummies(train.Embarked,drop_first=False),
),axis=1)

test=pd.concat((
    test,
    pd.get_dummies(test.Embarked,drop_first=False),
),axis=1)


# We transform the variable <code>Pclass</code> in dummies.

# In[ ]:


train=pd.concat((
    train,
    pd.get_dummies(train.Pclass,drop_first=False),
),axis=1)

test=pd.concat((
    test,
    pd.get_dummies(test.Pclass,drop_first=False),
),axis=1)


# We group together the honorary titles that are few in number

# In[ ]:


for i in range(len(train)) :
    if train.loc[train.index[i], 'Title'] not in ["Mr","Miss","Mrs","Master"]:
        train.loc[train.index[i], 'Title']="Autre"
for i in range(len(test)) :
    if test.loc[test.index[i], 'Title'] not in ["Mr","Miss","Mrs","Master"]:
        test.loc[test.index[i], 'Title']="Autre"


# In[ ]:


pd.crosstab(train.Title,train.Survived,margins=True)


# In[ ]:


graph_bar('Title')


# We transform the variable Title in dummies.

# In[ ]:


train=pd.concat((
    train,
    pd.get_dummies(train.Title,drop_first=False),
),axis=1)

test=pd.concat((
    test,
    pd.get_dummies(test.Title,drop_first=False),
),axis=1)


# ## The quantitative variables 

# ### The <code>SibSp</code> variable

# In[ ]:


train.pivot_table(
    values='SibSp',
    index='Survived',
    columns='Pclass',
    aggfunc=np.mean)


# ### The <code>Parch</code> variable

# In[ ]:


train.pivot_table(
    values='Parch',
    index='Survived',
    columns='Pclass',
    aggfunc=np.mean)


# ### The <code>Fare</code> variable

# In[ ]:


train.pivot_table(
    values='Fare',
    index='Survived',
    columns='Pclass',
    aggfunc=np.mean)


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True )


# ### The <code>Age</code> variable

# In[ ]:


train.pivot_table(
    values='Age',
    index='Survived',
    columns='Pclass',
    aggfunc=np.mean)


# In[ ]:


for i in range(len(train)) :
    if train.loc[train.index[i], 'Age'] <10:
        train.loc[train.index[i], 'Age2']="[0-10["
    elif train.loc[train.index[i], 'Age'] <20:
        train.loc[train.index[i], 'Age2']="[10-20["
    elif train.loc[train.index[i], 'Age'] <30:
        train.loc[train.index[i], 'Age2']="[20-30["
    elif train.loc[train.index[i], 'Age'] <40:
        train.loc[train.index[i], 'Age2']="[30-40["
    elif train.loc[train.index[i], 'Age'] <50:
        train.loc[train.index[i], 'Age2']="[40-50["
    elif train.loc[train.index[i], 'Age'] <60:
        train.loc[train.index[i], 'Age2']="[50-60["
    else: 
        train.loc[train.index[i], 'Age2']="[60 et +"


# In[ ]:


graph_bar("Age2")


# We're going to fill the missing values by values predicted by a linear regression.

# For the train dataset : 

# In[ ]:


from sklearn.linear_model import LinearRegression
#We extract the persons where we know their age :
train_age=train[~train["Age"].isnull()]
#We split in two parts : the dependants variables and the independant variable.
train_x_age=train_age[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]
train_y_age=train_age["Age"]
#This is the datasets where we will apply the age prediction 
train1=train[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]

model = LinearRegression()
model.fit(train_x_age, train_y_age)
age_pred = model.predict(train1)

#We concat the orinal dataset with the age and the new colomn: the age predict  
df=pd.DataFrame(list(age_pred),columns=["Age3"])
train2=pd.concat((
    train1,
    df,
    train["Age"]
),axis=1)

# if we known the age then we let it, but if we don't known the age, we take the age predict. 
for i in range(len(train2)) :
    if train2.loc[train2.index[i], 'Age']>=0:
        train.loc[train2.index[i], 'Age']=train2.loc[train2.index[i], 'Age']
    else: 
        train2.loc[train2.index[i], 'Age']=train2.loc[train2.index[i], 'Age3']


# For the test datatest:

# In[ ]:


#We extract the persons where we know their age :
test_age=test[~test["Age"].isnull()]
#We split in two parts : the dependants variables and the independant variable.
test_x_age=test_age[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]
test_y_age=test_age["Age"]
#This is the datasets where we will apply the age prediction 
test1=test[["Sex","Parch","SibSp","Fare","C","Q","S","Autre","Master","Miss","Mr","Mrs"]]

model = LinearRegression()
model.fit(test_x_age, test_y_age)
age_pred = model.predict(test1)

#We concat the orinal dataset with the age and the new colomn: the age predict  
df=pd.DataFrame(list(age_pred),columns=["Age3"])
test2=pd.concat((
    test1,
    df,
    test["Age"]
),axis=1)

# if we known the age then we let it, but if we don't known the age, we take the age predict. 
for i in range(len(test2)) :
    if test2.loc[test2.index[i], 'Age']>=0:
        test2.loc[test2.index[i], 'Age']=test2.loc[test2.index[i], 'Age']
    else: 
        test2.loc[test2.index[i], 'Age']=test2.loc[test2.index[i], 'Age3']


# # Logistic regression

# Now, we are going to apply the logistic regression. 

# In[ ]:


train_X=train2[['Sex','Parch', "SibSp",'Fare', 'Q', 'S','Autre','Master','Miss','Mrs','Age']]
train_Y=train[['Survived']]
test_X=test2[['Sex','Parch',"SibSp",'Fare', 'Q', 'S','Autre','Master','Miss','Mrs','Age']]


# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(train_X, train_Y)


# # Precdiction 

# In[ ]:


y_proba = log_reg.predict_proba(test_X)
survived=[]
for i in y_proba:
    if i[0]<0.5:
        survived.append(1)
    else :
        survived.append(0)
        
survived=pd.DataFrame(survived,columns=["Survived"])

titanic=pd.concat((
    test['PassengerId'],
    survived
),axis=1)


# In[ ]:


titanic.head()


# In[ ]:


titanic.to_csv('titanic.csv',index=False)

