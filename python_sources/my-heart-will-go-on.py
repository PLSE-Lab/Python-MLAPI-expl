#!/usr/bin/env python
# coding: utf-8

# ## Predicting survivals for the famous 1912 Titanic Disaster
# In this notebook, I'm showing the step-by-step process that I've taken while trying to predict survivals on Kaggle's Titanic competition.

# In[ ]:


import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt


# **Import datasets**

# In[ ]:


data = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data.tail()


# In[ ]:


data.info()


# **Processing strings**
# 
# Here I'm taking only a piece of the original strings in order to later (maybe) use them as inputs

# In[ ]:


data['ticket_type'] = data.Ticket.apply(lambda x : x.replace('.','').split(' ')[0] if len(x.split(' ')) > 1 else '')
test['ticket_type'] = test.Ticket.apply(lambda x : x.replace('.','').split(' ')[0] if len(x.split(' ')) > 1 else '')

data['cabin_type'] = data.Cabin.apply(lambda x : str(x)[0] if pd.notnull(x) else 'X')
test['cabin_type'] = test.Cabin.apply(lambda x : str(x)[0] if pd.notnull(x) else 'X')


# **Training and validation**
# 
# Using tran_test_split from scikit-learn to create the traning and validation sets

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='Survived'), 
                                                    data.Survived, test_size=0.2, 
                                                    stratify=data[['Survived', 'Sex', 'Pclass']]
                                                   )

X_train['Survived'] = y_train


# ### Exploratory data analysis
# 
# In the following plots, I'm trying to make some sense of the data in order to think of new features to create

# Age distribuition

# In[ ]:


sns.distplot(data.Age.dropna(), bins=20)


# Fare distribuition

# In[ ]:


sns.distplot(data.Fare.dropna(), bins=100)


# Finding what a high Fare value would be using quantile.

# In[ ]:


data.Fare.dropna().quantile(.75)


# Survival by Embarked feature

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=data.groupby('Embarked')['Survived'].mean().reset_index())


# Survival by generated cabin_type feature

# In[ ]:


sns.barplot(x='cabin_type', y='Survived', data=data.groupby('cabin_type')['Survived'].mean().reset_index())


# Survival by generated ticket_type feature

# In[ ]:


ax, fig = plt.subplots(figsize=(18,5))
sns.barplot(x='ticket_type', y='Survived', data=data.groupby('ticket_type')['Survived'].mean().reset_index())


# Boxplot of Survival (categorical) x Fare (continuous)

# In[ ]:


sns.boxplot(x='Survived', y='Fare', data=data)


# Survival by Pclass feature

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=data.groupby('Pclass')['Survived'].mean().reset_index())


# ### Feature Engineering
# 
# The following function creates most of the used features in my final model, and it uses a few of the EDA discoveries above. The features not inserted in this functions have a special behavior that makes it necessary for them to be created later (so far only one feature - *"surname_pct"*).

# In[ ]:


def feature_engineering(df):
    df = df.merge(X_train.groupby('cabin_type')['Survived'].mean().reset_index().rename(columns={'Survived':'cabin_pct'}), on='cabin_type', how='left')
    df['cabin_pct'] = df.cabin_pct > .7
#     df = df.merge(X_train.groupby('Embarked')['Survived'].mean().reset_index().rename(columns={'Survived':'embarked_pct'}), on='Embarked', how='left')
    df['cherbourg'] = df.Embarked.apply(lambda e : e == 'C')
#     df = df.merge(X_train.groupby('Pclass')['Survived'].mean().reset_index().rename(columns={'Survived':'pclass_pct'}), on='Pclass', how='left')
    df = df.merge(X_train.groupby('ticket_type')['Survived'].mean().reset_index().rename(columns={'Survived':'ticket_pct'}), on='ticket_type', how='left')
    df['ticket_pct'] = df.ticket_pct > .6
    df['title'] = df.Name.apply(lambda x : x.split('.')[0].split(',')[1].strip())
    df['sex'] = df.Sex.apply(lambda x : 1 if x == 'male' else 0)
    df['child'] = df.Age.apply(lambda x : x < 13)
#     df['teenager'] = df.Age.apply(lambda x : (x > 12) & (x < 20))
#     df['young_adult'] = df.Age.apply(lambda x : (x > 19) & (x < 36))
#     df['adult'] = df.Age.apply(lambda x : (x > 35) & (x < 60))
    df['elder'] = df.Age.apply(lambda x : x > 59)
    df['mother'] = df.Parch.apply(lambda x : x > 0) & df.title.apply(lambda x : x == 'Mrs')
    df['single'] = df.SibSp.apply(lambda x : x == 0) & df.Parch.apply(lambda x : x == 0)
    df['fare>30'] = df.Fare.apply(lambda x : x > 30)
    df['fare<10'] = df.Fare.apply(lambda x : x < 10)
    df['first_class'] = df.Pclass == 1
    df['big_fam'] = (df.Parch + df.SibSp).apply(lambda x : x > 3)
      
    return df

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)
test = feature_engineering(test)


# In[ ]:


X_train['surname'] = X_train.Name.apply(lambda name : name.split(',')[0])
X_train = X_train.merge(X_train.groupby('surname')['Survived'].mean().reset_index().rename(columns={'Survived':'surname_pct'}), on='surname', how='left')

X_test['surname'] = X_test.Name.apply(lambda name : name.split(',')[0])
X_test = X_test.merge(X_train.groupby('surname')['Survived'].mean().reset_index().rename(columns={'Survived':'surname_pct'}), on='surname', how='left')

test['surname'] = test.Name.apply(lambda name : name.split(',')[0])
test = test.merge(X_train.groupby('surname')['Survived'].mean().reset_index().rename(columns={'Survived':'surname_pct'}), on='surname', how='left')


# Dropping useless columns (non-numerical and/or already processed to create other features)

# In[ ]:


X_train = X_train.drop(columns=['PassengerId', 'Age', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                                'Cabin', 'Embarked', 'title', 'surname', 'cabin_type', 'ticket_type']).fillna(X_train.mean()).astype('float64')

X_test = X_test.drop(columns=['PassengerId', 'Age', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                                'Cabin', 'Embarked', 'title', 'surname', 'cabin_type', 'ticket_type']).fillna(X_train.mean()).astype('float64')

test_num = test.drop(columns=['PassengerId', 'Age', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                                'Cabin', 'Embarked', 'title', 'surname', 'cabin_type', 'ticket_type']).fillna(X_train.mean()).astype('float64')


# A view of the final training matrix

# In[ ]:


X_train.head()


# Correlation matrix of the final features

# In[ ]:


import numpy as np

corr = X_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Pipeline
# 
# Pipeline used to predict survival. It contains:
# 
# 1. A StandardScaler and a Normalizer (to normalize all data)
# 2. Polynomial feature generator (to capture possibly revelant non-linear relations)
# 3. PCA (to reduce dimensionality while keeping the more relevant components)
# 4. MinMaxScaler (to keep data between 0 and 1 before entering ML model)
# 5. Model (to do the cool Machine Learning stuff)

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest

model = Pipeline(steps=[
                        ('scaler', StandardScaler()),
                        ('norm', Normalizer()),
                        ('poly', PolynomialFeatures(3)),
                        ('PCA', PCA(6)),
                        ('minmax', MinMaxScaler()),
#                         ('logreg', RidgeClassifier())
                        ('rfc', RandomForestClassifier(1000, max_depth=5)),
#                         ('xgb', xgb.XGBRFClassifier(1000, depth=2)),
#                         ('svm', SVC(kernel='rbf', gamma='scale'))
                       ])

model.fit(X_train.drop(columns=['Survived']), y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)


# **Precision**

# In[ ]:


precision_score(y_pred, y_test)


# **Recall**

# In[ ]:


recall_score(y_pred, y_test)


# ### Conclusion
# 
# I was able to surpass 80% on evaluation data a few times when using the final pipeline, I've tried a few different classifiers, but used Random Forests in the end. When submitting the result to Kaggle, the accuracy would go below 80%, which I believe can be solved/tackled with a little more feature engineering.
# 
# I only spent a couple days on this challenge, so I hope I will come back to it in the near future to make some improvements.

# #### Predicting test.csv and saving submission
# Prediction for Kaggle unlabeled *test.csv*

# In[ ]:


y_pred = model.predict(test_num)
print(y_pred)


# In[ ]:


sum(y_pred) / len(y_pred)


# In[ ]:


submission = pd.read_csv("../input/titanic/gender_submission.csv")


# In[ ]:


submission['Survived'] = y_pred


# In[ ]:


submission.head(3)


# In[ ]:


submission.to_csv("submission.csv", index=False)

