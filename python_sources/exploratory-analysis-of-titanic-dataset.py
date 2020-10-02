#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')

import warnings
warnings.simplefilter('ignore')


# In[3]:


train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# In[4]:


print('Shape of train data: {}' .format(train.shape))
print('Shape of test data: {}' .format(test.shape))
train_= train.drop(['PassengerId'], axis= 1)
test_= test.drop(['PassengerId'], axis= 1)


# #### Statical Description of train and test dataset

# In[5]:


print('Description of train data: ')
print(train_.describe())
print()
print('Description of test data: ')
print(test_.describe())


# In[6]:


train_.head()


# In[7]:


test_.head()


# #### Detail of the missing values in the train and test dataset

# In[8]:


null_value= pd.DataFrame({
    'Total_null_train': train_.isnull().sum(),
    'Percent_null_train': train_.isnull().sum()/train_.shape[0],
    'Total_null_test': test_.isnull().sum(),
    'Percent_null_test': test_.isnull().sum()/test_.shape[0]
})


# In[9]:


null_value


# ##### Detail of individual features present in train dataset

# In[10]:


for ind in null_value.index:
    print()
    print('The detail of feature: {}' .format(ind))
    print(train[ind].describe())


# #### Graphical representation

# In[11]:


plt.hist(train['Survived'], color= 'red')
plt.show()


# In[12]:


print('Analysis of the label feature: Survived on the basis of Sex')
sns.countplot('Survived', data= train_, hue= 'Sex')
plt.title('Survived on the basis of Sex')
plt.show()


# In[13]:


sns.countplot('Survived', data= train_, hue= 'Pclass')
plt.title('Survived on the basis of Paseenger Class')


# In[14]:


sns.countplot('Survived', data= train_, hue= 'Embarked')
plt.title('Survived on the basis of Emabarked')


# In[15]:


sns.countplot('Survived', data= train_, hue= 'SibSp')
plt.title('Survived on the basis of Sibling and spouse')
plt.legend(loc= 'best')


# In[16]:


plt.scatter(np.log(train['Fare']), train['Survived'])


# In[17]:


sns.lmplot('Fare', 'Survived', data= train_)


# In[18]:


plt.figure(figsize= (8,6))
sns.heatmap(train_.corr(), annot= True, cmap= 'viridis')


# In[19]:


null_value.index


# In[20]:


train['Age'].groupby(train['Pclass']).describe()


# In[21]:


sns.boxplot('Pclass', 'Age', data= train_, hue= 'Sex', )
plt.title('Boxplot Pclass vs Age')


# In[22]:


age_bin= pd.cut(train['Age'], bins=[1,10,20,30,40,50,60,70,80,90,100])


# In[23]:


age_bin.value_counts()


# In[24]:


age_bin.value_counts().plot(kind= 'pie')


# In[25]:


sns.boxplot('Pclass', 'Age', data= train)


# In[26]:


sns.boxplot('Embarked', 'Age', data= train)


# In[27]:


print('Average age of Pclass: {} and Sex: {}' .format(1, 'male'))
print(train[(train['Pclass']==1) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(1, 'female'))
print(train[(train['Pclass']==1) & (train['Sex']=='female')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(2, 'male'))
print(train[(train['Pclass']==2) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(2, 'female'))
print(train[(train['Pclass']==2) & (train['Sex']=='female')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(3, 'male'))
print(train[(train['Pclass']==3) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(3, 'female'))
print(train[(train['Pclass']==3) & (train['Sex']=='female')]['Age'].mean())


# In[28]:


train_['Age']= train_['Age'].fillna(train_['Age'].mode()[0])


# In[29]:


test_['Age']= test_['Age'].fillna(test_['Age'].mode()[0])


# In[30]:


train_['Embarked']= train_['Embarked'].fillna(train_['Embarked'].mode()[0])


# In[31]:


train_['Cabin']= train_['Cabin'].fillna('None')
test_['Cabin']= test_['Cabin'].fillna('None')


# In[32]:


test_['Fare']= test_['Fare'].fillna(test_['Fare'].mean())


# In[33]:


train_['Cabin']= [x[:1] for x in train_['Cabin']]
test_['Cabin']= [x[:1] for x in test_['Cabin']]


# In[34]:


Name= list(train_['Name'])
Name_= list(test_['Name'])


# In[35]:


salutation= []
for name in Name:
    last= name.split(',')[1]
    sal= last.split('.')[0]
    salutation.append(sal)

sal_test= []
for name_ in Name_:
    last_= name_.split(',')[1]
    sal_= last_.split('.')[0]
    sal_test.append(sal_)


# In[36]:


type(salutation)


# In[37]:


train_['Name']= [x for x in salutation]
test_['Name']= [x for x in sal_test]


# In[38]:


train_['Name'].value_counts()


# In[39]:


train_['Name']= [x.replace('Miss', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Master', 'Mr') for x in train_['Name']]
train_['Name']= [x.replace('Mlle', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Mme', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Dr', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Rev', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Col', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Major', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Don', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Capt', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Sir', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Jonkheer', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('the Countess', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Lady', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Ms', 'Other') for x in train_['Name']]


# In[40]:


test_['Name'].value_counts()


# In[41]:


test_['Name']= [x.replace('Miss', 'Mrs') for x in test_['Name']]
test_['Name']= [x.replace('Master', 'Mr') for x in test_['Name']]
test_['Name']= [x.replace('Rev', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Col', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Dona', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Ms', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Dr', 'Other') for x in test_['Name']]


# In[42]:


train_['Name'].value_counts()


# In[ ]:


test_['Name'].value_counts()


# In[43]:


train_.isnull().sum()


# In[44]:


test_.isnull().sum()

