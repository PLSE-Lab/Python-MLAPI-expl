#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (16, 10)


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train['dataset'] = 'train'
df_test = pd.read_csv('../input/test.csv')
df_test['dataset'] = 'test'
df = pd.concat([df_train, df_test], sort=True, copy=False)


# We concate train and test dataset so every transformation on Feature Engineering will change both of them. But for EDA (Exploratory Data Analysis) we will using train dataset.

# In[ ]:


df_train.info()


# 11 predictor features (minus Survived and dataset). Age, Cabin and Embarked seems contain null value, we will take care of them later.

# In[ ]:


df_train.nunique().sort_values()


# No constant feature (feature with 1 unique value except dataset, it wont affect model so we want remove it first)

# In[ ]:


num_features = df_train.select_dtypes(['float64', 'int64']).columns.tolist()
cat_features = df_train.select_dtypes(['object']).columns.tolist()
print('{} numerical features:\n{} \nand {} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))


# We will remove PassengerId because it doesnt mean anything. Now lets analyse 6 numerical features.

# In[ ]:


num_features.remove('PassengerId')
num_features = sorted(num_features)
num_features


# In[ ]:


df_train[num_features].describe()


# ## Survived

# In[ ]:


print('{:.2f}% survival rate, {} out of {} survived'.format(df_train.Survived.sum()/len(df_train)*100, df_train.Survived.sum(), len(df_train)))


# In[ ]:


corrplot = sns.heatmap(df_train[num_features].corr(), cmap=plt.cm.Reds, annot=True)


# In[ ]:


abs(df_train[num_features].corr()['Survived']).sort_values(ascending=False)


# Seems only Pclass and Fare that has promising correlation to Survived, the other has low correlation.

# ## Pclass

# In[ ]:


g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'Pclass')


# In[ ]:


df_train.groupby('Pclass').agg(['mean', 'count'])['Survived']


# From above we see that Survived rate more likely related to Pclass. Pclass 3 has the worst survival rate, the second 2 and the best is Pclass 1. So for safety purpose just buy class 1 ticket okay. But it must cost more money i think.

# In[ ]:


sns.boxplot(data=df, x='Fare', y='Pclass', orient='h')


# Yep! safety proportional to cost we pay.

# ## Age

# In[ ]:


age_plot = sns.distplot(df_train[df_train.Age.notnull()].Age)


# Normal distribution, outlier not detected. Nothing suspicious here.

# In[ ]:


df_train.Age.isnull().sum()


# 177 null value. There is some way to handle missing data, like delete it, replace with central value (mean / median), and interpolate it. But since we use title *Simple ML*, we just replace all missing data with mean of Age on all dataset.

# In[ ]:


# remember to use df instead of df_train to transform dataset
df['Age'] = df[['Age']].applymap(lambda x: df.Age.mean() if pd.isnull(x) else x)


# ## SibSp & Parch

# In[ ]:


print(df_train.groupby('SibSp').agg(['mean', 'count'])['Survived'])
print(df_train.groupby('Parch').agg(['mean', 'count'])['Survived'])


# This is survival rate each SibSp and Parch, how about join them together

# In[ ]:


df['Family'] = df.SibSp + df.Parch
print(df[df.dataset == 'train'].groupby('Family').agg(['mean', 'count'])['Survived'])


# Family feature makes difference more obvious. We will use Family and remove SubSp and Parch feature

# In[ ]:


df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# ## Fare

# In[ ]:


fare_plot = sns.distplot(df_train.Fare)


# Very clearly there is outlier > 200. Lets confirm with Z-score.

# In[ ]:


df_train['Fare_std'] = df_train[['Fare']].apply(lambda x: abs(x-x.mean())/x.std())
df_train[['Fare', 'Fare_std']].sort_values('Fare_std', ascending=False).head(25)


# We will remove Z-score > 3 thats mean Fare > 200

# In[ ]:


# condition we want to remove is dataset == 'train' and Fare > 200
# so we use negation
df = df[(df.dataset != 'train') | (df.Fare < 200)]


# ## Name & Ticket

# In[ ]:


df_train[['Name', 'Ticket']].head(20)


# Actually, we can construct useful feature from Name and Ticket. But again, for simple ML we just remove this two feature and see the model performance.

# In[ ]:


df.drop(['Name', 'Ticket'], axis=1, inplace=True)


# ## Sex

# In[ ]:


g = sns.FacetGrid(df_train, col='Survived').map(sns.countplot, 'Sex')
df_train.groupby('Sex').agg(['mean', 'count'])['Survived']


# Around 74% female survived but male just 18%. This big difference make it key feature to predict Survived.
# Some ML algorithm wont do better if there is some non-numerical data, so we must convert it to numerical data.

# In[ ]:


df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})


# ## Cabin

# In[ ]:


print(df_train.Cabin.isnull().sum())
print(df_train.Cabin.nunique())
df_train.Cabin[df_train.Cabin.notnull()].head(10)


# Cabin has 687 NaN value and lot of 147 category. We will simplify with take first letter on Cabin and change NaN value to 'Z'

# In[ ]:


df['Cabin'] = df[['Cabin']].applymap(lambda x: 'Z' if pd.isnull(x) else x[0])


# In[ ]:


# Lets create stacked plot
pivoted = df.groupby(['Cabin', 'Survived']).size().reset_index().pivot(index='Cabin', columns='Survived')
stackedplot = pivoted.plot.bar(stacked=True)


# the scale goes to high because too many Z value. So we must exclude Z value to see proportion of other values.

# In[ ]:


# without Z
stackedplot_withoutZ = pivoted.drop('Z').plot.bar(stacked=True)


# Now we create convert Cabin to numerical feature.
# For unique member k > 2 we can create feature with values 1, 2, 3,... but we cant assume Cabin A > Cabin B > Cabin C ...
# So its better to use dummy values.

# In[ ]:


df = pd.get_dummies(df, columns=['Cabin'], prefix='Cabin')


# In[ ]:


df.head()


# ## Embarked

# In[ ]:


pivoted = df.groupby(['Embarked', 'Survived']).size().reset_index().pivot(index='Embarked', columns='Survived')
stackedplot = pivoted.plot.bar(stacked=True)


# Same as Cabin, we will create dummies values.

# In[ ]:


df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')


# In[ ]:


df.head()


# # Training Model

# For training model, we use scikit-learn package

# In[ ]:


# for splitting train and validate dataset
from sklearn.model_selection import train_test_split

# machine learning classifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X = df[df.dataset == 'train'].drop(['PassengerId', 'dataset', 'Survived'], axis=1)
y = df[df.dataset == 'train']['Survived']
X_test = df[df.dataset == 'test'].drop(['PassengerId', 'dataset', 'Survived'], axis=1)


# X contain all predictor feature from training dataset, 
# y only contain dependent feature, thats what we want to predict,
# X_test same as X except it from test dataset which we want to submit

# In[ ]:


# Split train and tes set from train dataset
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2)


# we will split training dataset to 80% for training model and 20% for validate model

# In[ ]:


# Train model
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)


# Here is the public score so far:
# 
# svm.SVC() => 0.66028
# 
# linear_moel.LogisticRegression() => 0.74641
# 
# ensemble.RandomForestClassifier() => 0.75119
# 
# ensemble.RandomForestClassifier(n_estimators=1000) => 0.77033

# In[ ]:


model.score(X_validate, y_validate)


# Thats accuracy of our model. If you not satisfied with the result, you just need to re-run training model with different classifier or tuning some parameter. You can see all available parameter each classifier on scikit-learn documentation.
# 
# Okay, now create submission file. But first lets check if any value NaN on test feature

# In[ ]:


X_test.isnull().sum()


# We must fill Fare NaN value with median of Fare. We dont use mean because it contain outlier, and mean is sensitive to outlier value.

# In[ ]:


X_test['Fare'] = X_test[['Fare']].applymap(lambda x: df.Fare.median() if pd.isnull(x) else x)


# In[ ]:


# Create submission file
y_test = model.predict(X_test)
submission = pd.DataFrame(np.c_[df[df.dataset == 'test'].PassengerId, y_test.astype(int)], columns=['PassengerId','Survived'])
submission.to_csv('submission.csv', index=False)

