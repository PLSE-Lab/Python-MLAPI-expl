#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sklearn
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Import sklearn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# # EDA

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.head()


# ## 1. Sex influences survival rate (female>male)

# In[ ]:


# Use sex as one metric, because being female has significant higher survival rate
sns.catplot(x='Sex', y='Survived', data=df, kind='bar')


# In[ ]:


df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='male' else 1)


# ## 2. Age influences survival rate (lower age, higher survival rate)

# In[ ]:


df.loc[df['Age'].isnull(), 'Age'] = df['Age'].median()


# In[ ]:


# Define function that divide age into three group: <18 juniors; 18-56 Adults; >56 seniors
age_to_category = lambda x: 1 if x<18 else 2 if x<56 else 3


# In[ ]:


df['Age'] = df['Age'].apply(age_to_category)


# In[ ]:


# Juniors have higher survival rate than adult and seniors, and this could be included into the metrics
sns.catplot(x='Age', y='Survived', data=df, kind='bar')


# In[ ]:





# ## 3. Fare price influenced survival rate(higher fare price, higher survival rate)

# In[ ]:


# Fare price has been heavily right skewed, with mean>median
plt.figure(figsize=(20,10))
sns.distplot(df['Fare'], kde=False)
plt.show()


# In[ ]:


# Define three category of fares
fare_to_category = lambda x: 1 if x<20 else 2 if x<100 else 3


# In[ ]:


df['Fare'] = df['Fare'].apply(fare_to_category)


# In[ ]:


# Higher fare class has much higher survival rate than lower fare class
sns.catplot(x='Fare', y='Survived', data=df, kind='bar')


# ### 4. Embarked cargo (C>Q>S in terms of survival rate) 

# In[ ]:


sns.catplot(x='Embarked', y='Survived', data=df, kind='bar')


# In[ ]:


df['Embarked'].fillna(np.random.choice(df.dropna(subset=['Embarked'])['Embarked']), inplace=True)


# In[ ]:


df['Embarked'] = df['Embarked'].apply(lambda x: 0 if x=='S' else 1 if x=='C' else 2)


# In[ ]:


df[['Pclass', 'Age', 'SibSp','Parch', 'Fare']].corr()


# In[ ]:


# parent and children (Parch) and SibSp(siblings and spouse) are positive correlated, higher parch means higher sibsp
# Fare and pclass are negatively correlated. Upper class (1) means higher fare price
sns.heatmap(df[['Pclass', 'Age', 'SibSp','Parch', 'Fare']].corr())


# In[ ]:





# ## 5. Pclass influenced survival rate (lower pclass higher survival rate) 

# In[ ]:


sns.catplot(x='Pclass', y='Survived', data=df, kind='bar')


# In[ ]:





# ## 6. Family members to survival rate 

# In[ ]:


df['FamilyMember'] = df['Parch'] + df['SibSp']


# In[ ]:


# 3 Family members reached the highest survival rate
sns.catplot(x='FamilyMember', y='Survived', data=df, kind='bar')


# In[ ]:


df = df[['Pclass', 'Sex', 'Age', 'FamilyMember', 'Fare', 'Embarked', 'Survived']]
df.head()


# In[ ]:


# Seperate train and test data using stratified method
x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), 
                                                    df['Survived'], 
                                                    test_size=0.3, 
                                                    random_state=1,
                                                    stratify=df['Survived'])


# # Benchmark prediction using sex as the only feature

# In[ ]:


# Prediction accuracy using sex as the only feature
(x_test['Sex']==y_test).mean()


# In[ ]:


# Generate benchmark submission using sex as the only feature
df_test['Survived'] = df_test['Sex']
df_test[['PassengerId', 'Survived']].to_csv('submission_sex.csv', encoding='utf8', index=False)


# # Random Forest Prediction

# In[ ]:


# Use random forest as classifier, and optimize n_estimators
rf = RandomForestClassifier(random_state=1)
parameters = {'n_estimators':[10, 20, 50, 100]}


# In[ ]:


# Grid search for the best number of estimators
clf = GridSearchCV(rf, parameters, cv=5)


# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


clf.best_score_


# In[ ]:


clf.best_params_


# In[ ]:


(clf.predict(x_test)==y_test).mean()


# In[ ]:


confusion_matrix(y_test, clf.predict(x_test))


# In[ ]:


roc_auc_score(y_test, clf.predict(x_test))


# In[ ]:





# In[ ]:


# Processing test data for submission
df_test['FamilyMember'] = df_test['SibSp']+df_test['Parch']

df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test['Age'].median()

df_test['Age'] = df_test['Age'].apply(age_to_category)

df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

df_test['Fare'] = df_test['Fare'].apply(fare_to_category)

df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 0 if x=='S' else 1 if x=='C' else 2)

df_test['Sex'] = df_test['Sex'].apply(lambda x: 0 if x=='male' else 1)


# In[ ]:


df_test = df_test[['PassengerId', 'Pclass', 'Sex', 'Age', 'FamilyMember', 'Fare', 'Embarked']]


# In[ ]:


df_test['Survived'] = clf.predict(df_test.drop('PassengerId', axis=1))


# In[ ]:


# Generate submission file
df_test[['PassengerId', 'Survived']].to_csv('submission_cv.csv', encoding='utf8', index=False)

