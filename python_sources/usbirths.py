#!/usr/bin/env python
# coding: utf-8

# # Can we predict if an infant will be born with low birthweight?
# 
# Low birth weigth (LBW) of an infant can have serious health effects on it in its later life. World Health Organization defines Low birth weight (LBW) as a birth weight of an infant weighing 2,499 g or less. The birth weight of a baby is notable because very low birth weight babies are 100 times more likely to die compared to normal birth weight babies. Lower gestational age, lower number of prenaltal care visits, maternal tobacco smoking habits, air pollution, maternal race, maternal stress, etc. have been identified as few of the causes for LBW. In this notebook, we attempt to predict the weight of a baby, based on various features recorded after the birth of an infant. 
# 
# References:
# * https://www.childtrends.org/?indicators=low-and-very-low-birthweight-infants

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/us-births-2018/US_births(2018).csv', 
                 low_memory=False)


# In[ ]:


df.shape


# Approximately 4 million babies were born in 2018.

# In[ ]:


df.isnull().sum()


# In[ ]:


plt.hist(df.DBWT, bins=10**2)
plt.xlim(0, 6 * 1e3)
plt.vlines(2500, 0, 3*1e5)
plt.show()


# In[ ]:


df['LBW'] = df['DBWT'].apply(lambda x: 0 if x > 2500 else 1)


# In[ ]:


1e2 * df['LBW'].value_counts()/len(df)


# Approximately, 10% of babies are born as "Low Birth Weight" babies.

# In[ ]:


plt.scatter(df.WTGAIN, df.DBWT, alpha=0.01)
plt.xlabel('Mother\'s Weight Gain')
plt.ylabel('Baby\'s Weight')
plt.ylim(0, 6000)
plt.show()


# In[ ]:


df.groupby('SEX').mean()['DBWT'].plot(kind='barh')
plt.show()


# What is the relationship between the **Mother's race** and the **babie's weight**?

# In[ ]:


(1e2 * df['MRACE15'].value_counts()/len(df)).plot(kind='bar', figsize=(12, 4))
plt.xticks(rotation=0);


# In[ ]:


plt.figure(figsize=(12, 10))
sns.heatmap(pd.crosstab(df[df['FRACE6']!=9]['FRACE6'], df[df['FRACE6']!=9]['MRAVE6'])/len(df[df['FRACE6']!=9]) * 1e2, 
            cmap='Blues', annot=True, fmt='.2f', 
            linewidths=1, linecolor='black');


# In[ ]:


df.groupby('MRACE15').mean()['DBWT'].sort_values().plot(kind='barh')
plt.show()


# In[ ]:


df.SEX.value_counts(dropna=False)/len(df)


# In[ ]:


plt.figure(figsize=(8,4))
df[(df.PRIORLIVE==0)&(df.MEDUC!=9)].groupby('MEDUC').median()['MAGER'].plot(kind='barh')
plt.yticks(np.arange(8), ('8th', '9th-12th', 'High School', 'College', 
                          'Associate', 'Bachelors', 'Masters', 'PhD'))
plt.ylabel('Mother\'s Education')
plt.xlabel('Mother\'s Age at First Birth')
plt.xlim(15, 35);


# In[ ]:


pd.DataFrame(df.DOB_WK.value_counts()).sort_index(ascending=False).plot(kind='barh')
plt.yticks(np.arange(7), ('Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon', 'Sun'))
plt.show()


# In[ ]:


plt.hist(df.MAGER, bins=100, alpha=0.7, density=True)
plt.hist(df.FAGECOMB[df.FAGECOMB<70], bins=100, alpha=0.6, density=True)
plt.xlim(0, 70);


# In[ ]:


df = df.rename(columns={"FRACE6": "Fathers_Race", "MRAVE6": "Mothers_Race"})


# In[ ]:


round(pd.crosstab(df[df.Fathers_Race!=9].Mothers_Race, 
                  df[df.Fathers_Race!=9].Fathers_Race)\
      /len(df[df.Fathers_Race!=9])*1e2, 2)


# In[ ]:


plt.figure(figsize=(8,4))
plt.scatter(df.MAGER[:10**6], df.PRIORLIVE[:10**6], alpha=0.5, s=10, c='black')
plt.ylim(-1, 15)
plt.xlabel('Mother\'s Age')
plt.ylabel('Prior Births')
plt.show()


# In[ ]:


plt.hist(df.PRIORLIVE, bins=200, density=True)
plt.xlim(0, 20)
plt.show()


# ### Figuring out the hour and minutes when the baby is likely to be born.

# In[ ]:


plt.figure(figsize=(10,5))
plt.hist(df.DOB_TT, bins=2*10**3)
plt.xlim(0, 2400)
plt.xlabel('Time (HHMM)')
plt.ylabel('# Count')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
plt.hist(df.DOB_TT.apply(lambda x:x%100), bins=2*10**2)
plt.xlim(0, 60)
plt.show()


# In[ ]:


#df.DOB_TT.apply(lambda x:x%100).value_counts()


# ## A classifier that predicts if an infannt is underweight

# In[ ]:


df.columns


# In[ ]:


df['DBWT'] = df.DBWT.apply(lambda x: 0 if x > 2500 else 1)


# In[ ]:


df_ml = df.sample(2*10**4)
df_ml = df_ml.drop(columns=['FRACE15', 'Fathers_Race', 'MRACE15', 'Mothers_Race'])
X = df_ml.drop(columns='DBWT')
y = df_ml.DBWT


# In[ ]:


X = pd.get_dummies(X, columns=['ATTEND', 'BFACIL', 'DMAR', 'FEDUC', 'FHISPX', 
                               'FRACE31', 'IMP_SEX', 'IP_GON', 'LD_INDL', 
                               'MAGE_IMPFLG', 'MAR_IMP', 'MBSTATE_REC', 'MEDUC', 
                               'MHISPX', 'MM_AICU', 'MRACE31', 'MRACEIMP', 
                               'MTRAN', 'NO_INFEC', 'NO_MMORB', 'NO_RISKS', 
                               'PAY', 'PAY_REC', 'PRECARE', 'RDMETH_REC', 
                               'RESTATUS', 'RF_CESAR', 'RF_CESARN', 'SEX'], 
                   drop_first=True)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, random_state=42)


# In[ ]:


from sklearn import model_selection 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, GradientBoostingClassifier, 
                              VotingClassifier) 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

start = time.time()
for model in [
    DummyClassifier, 
    LogisticRegression, 
    SGDClassifier,
    DecisionTreeClassifier, 
    KNeighborsClassifier,
    GaussianNB, 
    QuadraticDiscriminantAnalysis,
    # SVC, # Takes a long time
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier, 
    GradientBoostingClassifier,
    XGBClassifier,
    LGBMClassifier
]:
    cls = model()
    kfold = model_selection.KFold(n_splits=10, random_state=1)
    start_t = time.time()
    s = model_selection.cross_val_score(cls, X_train, y_train, 
                                        scoring='roc_auc', cv=kfold, n_jobs=-1)
    training_time = time.time() - start_t 
    print("{:32}  AUC:{:.3f} STD: {:.2f} Time: {:.2f}".format(model.__name__, 
                                                              s.mean(), 
                                                              s.std(), 
                                                              training_time))


# In[ ]:


# Perhaps should use GPUs!
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
# folds = KFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)
# param_grid = {
#     'num_leaves': [30, 120],
#     'reg_alpha': [0.1, 0.5],
#     'min_data_in_leaf': [30, 50, 100, 300, 400],
#     'lambda_l1': [0, 1, 1.5],
#     'lambda_l2': [0, 1]
#     }
# lgb_estimator = LGBMClassifier(boosting_type='gbdt',
#                                objective='binary', 
#                                num_boost_round=2000, 
#                                learning_rate=0.01, 
#                                metric='auc')
# grid = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=folds)
# lgb_model = grid.fit(X_train, y_train)

# print(lgb_model.best_params_, lgb_model.best_score_)


# In[ ]:


lgbmc = LGBMClassifier()
lgbmc.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test, lgbmc.predict(X_test)))


# In[ ]:


print(confusion_matrix(y_test,lgbmc.predict(X_test)))

