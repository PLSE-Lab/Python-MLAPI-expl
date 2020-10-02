# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split, StratifiedKFold #,StratifiedShuffleSplit
from sklearn.metrics import log_loss, make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
###################### Importing Data ######################################
path2data = '/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw'
df_train = pd.read_csv(path.join(path2data,'train.csv'))
print(df_train.shape)

df_test = pd.read_csv(path.join(path2data,'test.csv'))
print(df_test.shape)

df_test['target'] = np.nan

df = pd.concat([df_train, df_test])
print(df.shape)

df_tmp = df.loc[df['target'].notna()].groupby(['education'])['target'].agg(['mean', 'std'])
df_tmp = df_tmp.rename(columns={'mean': 'target_mean', 'std': 'target_std'}).fillna(0.0).reset_index()

df = df.merge(df_tmp, how = 'left', on = ['education'])

df['target_mean'] = df['target_mean'].fillna(0.0)
df['target_std'] = df['target_std'].fillna(0.0)
#df['target_median'] = df['target_median'].fillna(0.0)
DT_df = pd.get_dummies(df,columns=[c for c in df_train.columns if df_train[c].dtype == 'object'])

#################### Simple tree with 1 new ######################
DT_model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
    min_samples_split=42,
    min_samples_leaf=17
)

####################### Testing and validating on training data
DT_df_train = DT_df[DT_df['target'].notna()].drop(['uid','target'], axis=1)
DT_target_train = DT_df.loc[DT_df['target'].notna(),'target']

DT_df_test = DT_df[DT_df['target'].isna()].drop(['uid','target'], axis=1)

fit_DT_model_base = DT_model.fit(DT_df_train, DT_target_train)

DT_p = fit_DT_model_base.predict_proba(DT_df_test)[:,1]
##################### Fitting KNN Data ###########################################
knn_df_train = pd.read_csv(path.join(path2data,'train.csv'))
print(knn_df_train.shape)

knn_df_test = pd.read_csv(path.join(path2data,'test.csv'))
print(knn_df_test.shape)

knn_df_test['target'] = np.nan

knn_df = pd.concat([knn_df_train, knn_df_test])
print(knn_df.shape)

#################### Start Pre-Processing the Data ###########################
cats = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

'''
As most of the features are Categorical and KNN is not very suitable for categorical,
We will calculate mean of the numerical variable at various level of combinations of categorica variables.
'''
import itertools as it

for i in range(1, 4):
    print(i)
    for g in it.combinations(cats, i):
        knn_df = pd.concat([knn_df,
                knn_df.groupby(list(g))[nums].transform('mean').rename(
                    columns=dict([(s, ':'.join(g) + '__' + s + '__mean') for s in nums])
                )
            ],
            axis=1
        )

##### As we have trasformed the numerical columns to extract info from categorical variables, we will drop the categoricals
knn_df.drop(columns=cats, inplace=True)
knn_df.shape
### Filtering the columns for Features
cols = [c for c in knn_df.columns if c != 'uid' and c != 'target']
### Standardising the Feature columns
from sklearn.preprocessing import StandardScaler
knn_df[cols] = StandardScaler().fit_transform(knn_df[cols])

### Calculate correlation and filtering columns with more than 0.5 correlation.
knn_df_m = knn_df[cols].corr()

cor = {}

for c in cols:
    cor[c] = set(knn_df_m.loc[c][knn_df_m.loc[c] > 0.5].index) - {c}

for c in cols:
    if c not in cor:
        continue
    for s in cor[c]:
        if s in cor:
            cor.pop(s)

cols = list(cor.keys())
####################
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=100,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
)

##################### Testing and validating on training data
knn_df_train = knn_df.loc[knn_df['target'].notna(), cols]
knn_df_target = knn_df.loc[knn_df['target'].notna()]['target']

knn_df_test = knn_df.loc[knn_df['target'].isna(), cols]

knn_model = knn_model.fit(knn_df_train, knn_df_target)

knn_p = knn_model.predict_proba(knn_df_test)[:,1]
