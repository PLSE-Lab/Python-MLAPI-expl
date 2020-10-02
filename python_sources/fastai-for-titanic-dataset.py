#!/usr/bin/env python
# coding: utf-8

# **Predict survival of passengers on Titanic Ship**

# Create new notebook from Notebooks tab on https://www.kaggle.com/c/titanic/notebooks to download the dataset.

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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# library for deep learning api interface
from fastai import *
from fastai.tabular import *

# library for visualization
import seaborn as sns
sns.set()


# **Load data using pandas**

# In[ ]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')


# **Quick peek into the data**

# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_test.info()


# In[ ]:


df_test.head()


# **Check for values missing in the data**

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# **Continuous variables scatter heatmap**

# In[ ]:


plt.subplots(figsize=(20,15))
sns.heatmap(df_train.corr(), cmap='coolwarm', annot=True)


# In[ ]:


sns.pairplot(data=df_train, hue='Survived', diag_kind='kde')
plt.show()


# **Categorial variables plot**

# In[ ]:


sns.catplot(x='Pclass', y='Embarked', hue='Survived', col='Sex', data=df_train, kind='violin')


# In[ ]:


sns.catplot(x='Parch', y='SibSp', hue='Survived', col='Sex', data=df_train, kind='bar')


# In[ ]:


sns.boxplot(data=df_train, x='Survived', y='Age')


# In[ ]:


df_train['Parch'].value_counts()


# In[ ]:


df_train['SibSp'].value_counts()


# **Create a new features**

# In[ ]:


# df_train['Surname'] = df_train.Name.str.split(',').apply(lambda x:x[0])
# df_train['Surname'].isnull().any()


# In[ ]:


df_train.Cabin


# In[ ]:


for df in (df_train, df_test):
    df['Deck'] = df.Cabin.str[0]
    df.loc[df.Cabin.isnull(), 'Deck'] = 'N'
    df['HasCabin'] = 1
    df.loc[df.Cabin.isnull(), 'HasCabin'] = 0
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df.FamilySize==1, 'IsAlone'] = 1
    df['IsChild'] = 0
    df.loc[df.Age<=12, 'IsChild'] = 1
    df['IsTeen'] = 0
    df.loc[(df.Age>12) & (df.Age<=19), 'IsTeen'] = 1
    df['IsAdult'] = 0
    df.loc[(df.Age>19) & (df.Age<=50), 'IsAdult'] = 1
    df['IsOld'] = 0
    df.loc[df.Age>50, 'IsOld'] = 1
    


# In[ ]:


df_train.head()


# Get sense of where most people boarded the ship

# In[ ]:


df_train['Embarked'].value_counts()


# In[ ]:


df_test['Embarked'].value_counts()


# **Use the highest occuring Deck and Embarked value to fill missing.
# Fo Age and Fare use mean.**

# In[ ]:


df_train['Age'].fillna(value=df_train['Age'].mean(), inplace=True)
df_train['Embarked'].fillna(value='S', inplace=True)
df_train.fillna(value={'Deck':'C'}, inplace=True)
df_train.isnull().sum()


# In[ ]:


df_test.fillna({'Fare': df_test['Fare'].mean()}, inplace=True)
df_test.fillna({'Age': df_test['Age'].mean()}, inplace=True)
df_test.fillna({'Embarked':'S'}, inplace=True)
df_test.fillna({'Deck':'C'}, inplace=True)
df_test.isnull().sum()


# In[ ]:


# df_train["Survived"][df_train["Survived"]==1] = "Survived"
# df_train["Survived"][df_train["Survived"]==0] = "Died"


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=df_train)


# In[ ]:


sns.barplot(x='Sex', y='Age', hue='Survived', data=df_train)


# In[ ]:


sns.barplot(x='Embarked', y='Pclass', hue='Survived', data=df_train)


# In[ ]:


sns.barplot(x='Sex', y='HasCabin', hue='Survived', data=df_train)


# In[ ]:


sns.violinplot(y='Pclass', x='IsAlone', hue='Survived', data=df_train)


# **Divide features into categoreis**

# In[ ]:


cat_names = ['Sex', 'Pclass', 'IsAlone', 'FamilySize', 'HasCabin', 'Deck', 'SibSp', 'Parch', 'Embarked', 'IsChild', 'IsTeen', 'IsAdult', 'IsOld']
cont_names = ['Age', 'Fare']
dep_var = 'Survived'
procs = [Categorify, Normalize]
# 


# In[ ]:


np.random.seed(2)


# In[ ]:


test_db = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[ ]:


train_db = (TabularList.from_df(df_train, cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_idx(list(range(200)))
            .label_from_df(cols=dep_var)
            .add_test(test_db, label=0)
            .databunch())


# In[ ]:


train_db.show_batch(10)


# In[ ]:


learn = tabular_learner(train_db, layers=[200], metrics=accuracy, emb_drop=0.001)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit(5, 1e-2)


# In[ ]:


learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.recorder.plot_lr()


# In[ ]:


# learn.save('Titanic')


# In[ ]:


predictions, _ = learn.get_preds(DatasetType.Test)
pred=np.argmax(predictions, 1)
pred.shape


# In[ ]:


pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


submission_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred})


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import HTML
import base64

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    return HTML(html)

create_download_link(submission_df)


# In[ ]:




