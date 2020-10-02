#!/usr/bin/env python
# coding: utf-8

# **Author**:       Miran T.  
# **Last Updated**: 2016-08-14
# 
# If you have any suggestions, critiques, etc. - please write them below in the comments, I would love to hear them. Thank you!  
# If you fork/like the script, please upvote it.
# 
# This is a second notebook in two-part series. Data preparation in this notebook is a result of the exploratory analysis in the first notebook.
# 
#  - [Part I - exploratory analysis](https://www.kaggle.com/narimiran/titanic/exploring-titanic) 
#  - [Part II - machine learning](https://www.kaggle.com/narimiran/titanic/democracy-in-ml-ensembles) (this notebook)
# 
# Some decisions about choosing hyperparameter values might seem strange/illogical, based on provided grid search, but all computations were done on my local machine, where they were making sense :)

# # Data preparation

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
sns.set_style('whitegrid')


# In[ ]:


titanic_train = pd.read_csv('../input/train.csv', index_col='PassengerId')
titanic_test = pd.read_csv('../input/test.csv', index_col='PassengerId')


# In[ ]:


titanic_train.info()
print('\n')
titanic_test.info()


# In[ ]:


combined = pd.concat((titanic_train, titanic_test), axis=0)
combined.info()


# In[ ]:


Title_Dictionary = {
                    "Capt":       "Officer", #
                    "Col":        "Officer",
                    "Major":      "Officer", #
                    "Jonkheer":   "Royalty", ##
                    "Don":        "Royalty", #
                    "Sir" :       "Royalty", #
                    "Dr":         "Officer",
                    "Rev":        "Officer", 
                    "the Countess":"Royalty", ##
                    "Dona":       "Royalty", ##
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty" ##
                    } 

combined['Title'] = combined['Name'].apply(lambda x: Title_Dictionary[x.split(',')[1].split('.')[0].strip()])


# In[ ]:


ages_mean = combined.pivot_table('Age', index=['Title'], columns=['Sex', 'Pclass'], aggfunc='mean')
ages_mean


# In[ ]:


ages_std = combined.pivot_table('Age', index=['Title'], columns=['Sex', 'Pclass'], aggfunc='std')
ages_std


# In[ ]:


def age_guesser(person):
    gender = person['Sex']
    mean_age = ages_mean[gender].loc[person['Title'], person['Pclass']]
    std = ages_std[gender].loc[person['Title'], person['Pclass']]
    persons_age = np.random.randint(mean_age - std, mean_age + std)
    return persons_age

unknown_age = combined['Age'].isnull()
people_w_unknown_age = combined.loc[unknown_age, ["Age", "Title", "Sex", "Pclass"]]

people_w_unknown_age['Age'] = people_w_unknown_age.apply(age_guesser, axis=1)

known_age = combined['Age'].notnull()
people_w_known_age = combined.loc[known_age, ["Age", "Title", "Sex", "Pclass"]]

combined['new_age'] = pd.concat((people_w_known_age['Age'], people_w_unknown_age['Age']))
combined.head(7)


# In[ ]:


combined['Embarked'].fillna('S', inplace=True)
combined['Fare'].fillna(value=combined['Fare'].mean(), inplace=True)

combined['kid'] = 0
combined.loc[combined.new_age <= 12, 'kid'] = 1

combined['parent'] = 0
combined.loc[(combined.Parch > 0) & (combined.new_age >= 18), 'parent'] = 1

combined['child'] = 0
combined.loc[(combined.Parch > 0) & (combined.new_age < 18), 'child'] = 1

combined.tail(5)

combined['family'] = combined['SibSp'] + combined['Parch']
combined.loc[combined.family > 0, 'family'] = 1

combined['male'] = (~combined['Sex'].str.contains('fe')).astype(int)

combined.info()


# In[ ]:


not_needed = ['Age', 'Cabin', 'Name', 'Sex', 'Ticket', 'Parch', 'SibSp']
combined.drop(not_needed, axis=1, inplace=True)


# In[ ]:


categorical = ['Embarked', 'Title', 'Pclass']

for column in categorical:
    dummy = pd.get_dummies(combined[column], prefix=column).astype(int)
    combined = combined.join(dummy)
    combined.drop(column, axis=1, inplace=True)
    
combined.head()


# In[ ]:


df = combined.loc[:len(titanic_train), :]

df_test = combined.loc[len(titanic_train)+1:, :].copy()
df_test.drop('Survived', axis=1, inplace=True)

df.shape, df_test.shape


# In[ ]:


corr_mat = np.corrcoef(df.values.T)

ax = sns.heatmap(corr_mat, annot=True, fmt='.2f',
                 xticklabels=df.columns, yticklabels=df.columns,
                )

_ = (ax.set_title('Correlation Matrix'))


# # Training

# In[ ]:


from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.decomposition import KernelPCA

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score


# In[ ]:


X_train = df.drop('Survived', 1)
y = df['Survived']

X_test_ = df_test.copy()


# In[ ]:


seed = 2016
folds = 10
scoring = 'accuracy'

kfold = StratifiedKFold(y=y, n_folds=folds, random_state=seed)


# ## Preprocessing, feature selection

# In[ ]:


robust = RobustScaler()
robust.fit(X_train[['Fare', 'new_age']])

X = X_train.copy()
X_test = X_test_.copy()
X[['Fare', 'new_age']] = robust.transform(X_train[['Fare', 'new_age']])
X_test[['Fare', 'new_age']] = robust.transform(X_test_[['Fare', 'new_age']])


# In[ ]:


rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=50, random_state=seed), 
                 cv=kfold, scoring='accuracy')

rfecv.fit(X, y)

plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_, 'o-')
plt.title('Accuracy depending on number of features')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')


# In[ ]:


rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=seed), 
          n_features_to_select=7
         )

rfe.fit(X, y)


# In[ ]:


X_redux = X.loc[:, X.columns[rfe.ranking_ == 1].values]
X_test_redux = X_test.loc[:, X_test.columns[rfe.ranking_ == 1].values]

X_redux.columns


# In[ ]:


corr_mat = np.corrcoef(X_redux.values.T)

ax = sns.heatmap(corr_mat, annot=True, fmt='.2f',
                 xticklabels=X_redux.columns, yticklabels=X_redux.columns,
                )

_ = (ax.set_title('Correlation Matrix'))


# In[ ]:


pca = KernelPCA(n_components=6, kernel='linear')
model = SVC(random_state=seed)

pipeline = Pipeline(
    [('pca', pca),
     ('model', model)
    ]
)
param_grid = dict(pca__n_components=[2, 3, 4, 5, 6],
                  pca__kernel=['linear', 'rbf', 'poly'],
                  pca__degree=[2, 3, 4]
                 )

grid = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
grid.fit(X, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)


# # Basic estimators    
# 
# 

# In[ ]:


models = [('logreg', LogisticRegression(random_state=seed)), 
          ('knn', KNeighborsClassifier()),
          ('nb', GaussianNB()),
          ('svm', SVC(random_state=seed)),
          ('rf', RandomForestClassifier(n_estimators=50, random_state=seed)), 
          ('ada', AdaBoostClassifier(random_state=seed)),
          ('gbm', GradientBoostingClassifier(random_state=seed)), 
         ]


# In[ ]:


training_sets = [
    ('full X', X),
    ('reduced X', X_redux),
    ('pca X', X_pca)
]

for traning_set in training_sets:
    print('\n', traning_set[0], '\n===========')

    for name, model in models:
        result = cross_val_score(model, traning_set[1], y, cv=kfold, scoring=scoring)
        print("{0}:\t ({1:.4f}) +/- ({2:.4f})".format(name, result.mean(), result.std()))


# # Hyperparameter tuning  
# 
# 

# ## Logistic Regression
# 
# 
# 
# 

# In[ ]:


model = LogisticRegression(random_state=seed)

parameters = dict(
    C=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


logreg_best = LogisticRegression(C=0.5)


# ## KNN 
# 
# 

# In[ ]:


model = KNeighborsClassifier()

parameters = dict(
    n_neighbors=[2, 3, 4, 5, 6, 8, 10, 12, 14, 20],
    weights=['uniform', 'distance']
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X_redux, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:15]


# In[ ]:


knn_best = Pipeline([
        ('rfe', rfe),
        ('knn', KNeighborsClassifier(n_neighbors=8))   
    ])


# ## Support Vector Classifier ##

# In[ ]:


model = SVC(random_state=seed, probability=True)

parameters = dict(
    C=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
    gamma=['auto', 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X_pca, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


model = SVC(random_state=seed, probability=True)

parameters = dict(
    C=[0.5, 1, 2, 5, 10, 15, 20, 25, 30, 50],
    gamma=['auto', 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X_pca, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


svm_best = Pipeline([
        ('pca', pca),
        ('svm', SVC(C=10, gamma=0.1, random_state=seed, probability=True))
    ])


# ## Random Forest
# 
# 

# In[ ]:


model = RandomForestClassifier(random_state=seed)

parameters = dict(
    n_estimators=[50, 100, 150, 200, 250, 300, 350, 400],
    min_samples_split=[2, 4, 8, 12, 16, 20, 24]
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X_redux, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


rf_best = Pipeline([
        ('rfe', rfe),
        ('rf', RandomForestClassifier(n_estimators=300, min_samples_split=16,
                                      random_state=seed))
        
    ])


# ## Ada boost 

# In[ ]:


model = AdaBoostClassifier(random_state=seed)

parameters = dict(
    n_estimators=[100, 200, 400, 600, 800, 1000, 1200, 1500],
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


ada_best = AdaBoostClassifier(n_estimators=1000, random_state=seed)


# ## Gradient boost  

# In[ ]:


model = GradientBoostingClassifier(random_state=seed)

parameters = dict(
    n_estimators=[100, 200, 300, 400, 500, 600, 700, 800],
    learning_rate=[0.01, 0.05, 0.1, 0.2, 0.5, 1],
    max_depth=[1, 2, 3, 4]
)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, scoring=scoring, n_jobs=-1)
grid.fit(X_redux, y)

grid.grid_scores_.sort(key=lambda x: x[1], reverse=True)
grid.grid_scores_[:10]


# In[ ]:


gbm_best = Pipeline([
        ('rfe', rfe),
        ('gbm', GradientBoostingClassifier(n_estimators=700, learning_rate=0.05, max_depth=2,
                                           random_state=seed))
    ])


# ## Scoring metrics 

# In[ ]:


estimators = [
    ('Logistic Regression', logreg_best),
    ('KNN', knn_best),
    ('SVC', svm_best),
    ('Random Forest', rf_best),
    ('Ada boost', ada_best),
    ('Gradient boost', gbm_best)
]


# In[ ]:


plt.figure(figsize=(12,9))
for name, model in estimators:
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_score = roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, label='{} ({:.4f})'.format(name, auc_score))
_ = plt.legend(loc=4)


# In[ ]:


for name, model in estimators:
    model.fit(X, y)
    y_pred = model.predict(X)
    print(name)
    print(confusion_matrix(y, y_pred))
    print()


# In[ ]:


for name, model in estimators:
    model.fit(X, y)
    y_pred = model.predict(X)
    print(name)
    print('Score: {:.4f}'.format(f1_score(y, y_pred)))
    print()


# In[ ]:


for name, model in estimators:
    result = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    print("{0:<20} ({1:.4f}) +/- ({2:.4f})".format(name, result.mean(), result.std()))


# # Voting time

# In[ ]:


voters = [
    ('Random Forest', rf_best),
    ('Gradient boost', gbm_best),
    ('Ada boost', ada_best),
    ('KNN', knn_best) 
]


# In[ ]:


voting_ensemble = VotingClassifier(voters, voting='soft')

results = cross_val_score(voting_ensemble, X, y, cv=kfold, scoring=scoring)
print("({0:.4}) +/- ({1:.4f})".format(results.mean(), results.std()))


# In[ ]:


voting_ensemble.fit(X, y)
predictions = voting_ensemble.predict(X_test)

submission = pd.DataFrame()
submission["PassengerId"] = df_test.index
submission["Survived"] = predictions.astype(int)

submission.to_csv("submission_001.csv", index=False)

