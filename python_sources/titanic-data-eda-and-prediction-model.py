#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# import datasets
df_train = pd.read_csv("../input/train.csv",index_col="PassengerId")
df_test = pd.read_csv("../input/test.csv",index_col="PassengerId")


# In[ ]:


# view first five lines of training data
df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_test['Survived'] =- 999


# In[ ]:


total_df = pd.concat((df_train, df_test),axis=0,sort=True)


# In[ ]:


total_df.info()


# In[ ]:


total_df.iloc[5:11,0:7]


# In[ ]:


total_df.tail()


# In[ ]:


male_df = total_df[total_df['Sex'] == 'male']
female_df = total_df[total_df['Sex'] == 'female']


# In[ ]:


male_df.info()


# In[ ]:


female_df.info()


# In[ ]:


total_df.describe(include='all')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
total_df.Fare.plot(kind='box')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
total_df.Age.plot(kind='box')


# In[ ]:


total_df.Sex.value_counts()


# In[ ]:


total_df.Survived.value_counts()


# In[ ]:


total_df[total_df.Survived!= -999].Survived.value_counts()


# In[ ]:


total_df.Pclass.value_counts()


# In[ ]:


total_df.Pclass.value_counts().plot(kind='bar')


# In[ ]:


total_df.Age.plot(kind='hist',bins=20,color='g')


# In[ ]:


total_df.Age.plot(kind='kde',color='g')


# In[ ]:


total_df.Fare.plot(kind='hist',color='r',bins=20)


# In[ ]:


total_df.Fare.plot(kind='kde',color='r')


# In[ ]:


total_df.Age.skew()


# In[ ]:


total_df.Fare.skew()


# In[ ]:


total_df.plot.scatter(x='Age',y='Fare',alpha=0.4)


# In[ ]:


total_df.plot.scatter(x='Pclass',y='Fare',alpha=0.1)


# In[ ]:


total_df.groupby(['Pclass'])[['Age','Fare']].median()


# In[ ]:


total_df.groupby(['Pclass']).agg({"Age" : 'mean' ,'Fare' : 'mean'})


# In[ ]:


total_df.groupby(['Pclass','Embarked']).agg({"Age" : 'median' ,'Fare' : 'median'})


# In[ ]:


pd.crosstab(total_df.Sex,total_df.Pclass)


# In[ ]:


pd.crosstab(total_df.Sex,total_df.Pclass).plot(kind='bar')


# In[ ]:


total_df.pivot_table(index='Sex',columns='Pclass',values='Age',aggfunc='mean')


# In[ ]:


# Data Munging


# In[ ]:


total_df.info()


# In[ ]:


total_df[total_df.Embarked.isnull()]


# In[ ]:


total_df.Embarked.value_counts()


# In[ ]:


pd.crosstab(total_df[total_df.Survived != -999].Embarked,total_df[total_df.Survived != -999].Survived)


# In[ ]:


total_df.pivot_table(index='Pclass',columns='Embarked',values='Fare',aggfunc='median')


# In[ ]:


total_df.Embarked.fillna('C',inplace=True)


# In[ ]:


total_df.info()


# In[ ]:


total_df[total_df.Fare.isnull()]


# In[ ]:


total_df[(total_df.Pclass ==3) & (total_df.Embarked=='S')].Fare.median()


# In[ ]:


total_df[total_df.Age.isnull()].count()


# In[ ]:


print(total_df.Age.mean())
print(total_df.Age.median())
print(total_df.Age.mode())


# In[ ]:


total_df[total_df.Sex == 'male'].Age.median()


# In[ ]:


total_df[total_df.Sex == 'female'].Age.median()


# In[ ]:


total_df[total_df.Sex == 'male'].Age.median()


# In[ ]:


total_df[total_df.Age.notnull()].boxplot('Age','Pclass')


# In[ ]:


Median_Age = total_df.groupby('Pclass').Age.transform('median')


# In[ ]:


total_df.info()


# In[ ]:


name = total_df.loc[1,'Name'].split(',')


# In[ ]:


name[1].split('.')[0]


# In[ ]:


def getsalutation(name):
        sal = name.split(',')[1].split('.')[0].strip()
        return sal


# In[ ]:


total_df.Name.map(lambda x:getsalutation(x)).unique()


# In[ ]:


def getTitles(name):
    dicti={'mr':'Mr', 'mrs':'Mrs', 'miss':'Miss', 'master':'Master', 'don':'Sir', 'rev':'Sir', 'dr':'officer', 'mme':'Mrs', 
          'ms':'Mrs',
       'major':'officer', 'lady':'Lady', 'sir':'Sir', 'mlle':'Miss', 'col':'officer', 'capt':'officer', 'the countess':'Lady',
       'jonkheer':'Sir', 'dona':'Lady'}
    sal = (name.split(',')[1].split('.')[0].strip()).lower()
    return dicti[sal]


# In[ ]:


total_df['Title']=total_df.Name.map(lambda x:getTitles(x))


# In[ ]:


total_df[['Title','Name']].head()


# In[ ]:


total_df[total_df.Age.notnull()].boxplot('Age','Title')


# In[ ]:


Median_Age_title = total_df.groupby('Title').Age.transform('median')
Median_Age_title.unique()


# In[ ]:


total_df.Age.fillna(Median_Age_title, inplace=True)


# In[ ]:


total_df.info()


# In[ ]:


total_df['Age'].plot(kind='hist',bins=20)


# In[ ]:


total_df.Fare.plot(kind='box')


# In[ ]:


total_df[total_df.Fare == total_df.Fare.max()]


# In[ ]:


logfare = np.log(total_df.Fare)


# In[ ]:


pd.qcut(total_df.Fare,5)


# In[ ]:


total_df.isnull().sum()


# In[ ]:


total_df['Fare'].fillna(total_df['Fare'].mean(), inplace = True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


total_df.shape


# In[ ]:


total_df['Embarked'] = le.fit_transform(total_df['Embarked'])


# In[ ]:


total_df['FamilySize'] = total_df['SibSp'] + total_df['Parch']


# In[ ]:


def correlation_heatmap(df, method):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(method=method),
        cmap = colormap,
        square=True, 
        annot=True, 
        annot_kws={'fontsize':9 }
    )
    
    plt.title('Correlation Matrix', y=1.05, size=15)


# In[ ]:


correlation_heatmap(total_df, 'pearson')


# In[ ]:


# Drop low corrlations and high cardinality
to_drop = ['Ticket', 'Name', 'Title', 'Age','SibSp', 'Parch', 'FamilySize', 'Embarked']
#to_drop = ['Ticket', 'Name']
total_df = total_df.drop(to_drop, axis=1, inplace=False)


# In[ ]:


total_df.info()


# In[ ]:


total_df["CabinBool"] = (total_df["Cabin"].notnull().astype('int'))


# In[ ]:


total_df['Deck'] = total_df.Cabin.str.extract('([a-zA-Z]+)', expand=False)
total_df[['Cabin', 'Deck']].sample(10)
total_df['Deck'] = total_df['Deck'].fillna('Z')
total_df = total_df.drop(['Cabin'], axis=1)

# label
total_df['Deck'] = le.fit_transform(total_df['Deck'])


# In[ ]:


total_df.info()


# In[ ]:


total_df.head()


# In[ ]:


# Creade dummy variables for Sex and drop original, as well as an unnecessary column (male or female)
total_df = total_df.join(pd.get_dummies(total_df['Sex']))
total_df.drop(['Sex', 'male'], inplace=True, axis=1)


# In[ ]:


train = total_df[total_df['Survived'] != -999]
test = total_df[total_df['Survived'] == -999]


# In[ ]:


X_train, y_train = train.loc[:, train.columns != 'Survived'], train.loc[:, train.columns == 'Survived']
X_test = test.loc[:, train.columns != 'Survived']


# In[ ]:


X_test.index


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split training data into training and validation set
train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)

# Initialize model
model = RandomForestClassifier(n_estimators=100)
# Fit data
model.fit(train_X, train_y)
# Calc accuracy
acc = accuracy_score(model.predict(val_X), val_y)
print("Validation accuracy for Random Forest Model: {:.6f}".format(acc))


# In[ ]:


y_pred = pd.DataFrame(model.predict(X_test),index=X_test.index.copy(),columns=['Survived'])
pred= y_pred.to_csv('survived.csv', index=True)


# In[ ]:


importances = model.feature_importances_
sns.barplot(importances,train_X.columns)


# In[ ]:




