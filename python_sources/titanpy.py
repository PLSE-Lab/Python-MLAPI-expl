#!/usr/bin/env python3

""" script to predict titanic survival """

import numpy as np
import pandas as pd
import re
import sklearn as sk
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as m
import sklearn.svm as svm
import sklearn.neighbors as n
import sklearn.naive_bayes as nb
import sklearn.ensemble as en
import sklearn.neural_network as nn
import matplotlib.pyplot as plt

__author__ = 'Justin Logan'
__copyright__ = 'Copyright 2017'
__credits__ = ['Justin Logan']

__license__ = 'CC-BY-SA'
__version__ = '1.0.0'
__maintainer__ = 'Justin Logan'
__email__ = 'Justin.A.Logan@gmail.com'
__status__ = 'Development'

# Import data
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data = [data_train, data_test]

# Clean test
data_test.loc[:, 'Fare'] = data_test.Fare.replace(np.nan, 0)

# Feature engineering
for d in data:
    # Gender dummy
    d.loc[:, 'Female'] = (d.Sex == 'female')

    # Class dummies
    d.loc[:, 'MidClass'] = (d.Pclass == 2)
    d.loc[:, 'UpClass'] = (d.Pclass == 1)

    # Null age
    d.loc[:, 'NullAge'] = d.Age.isnull()

    # Child and Elder dummies
    d.loc[:, 'Child'] = (d.Age < 15)
    d.loc[:, 'Elder'] = (d.Age < 60)

    # Embarked dummies
    d.loc[:, 'Embarked_C'] = (d.Embarked == 'C')
    d.loc[:, 'Embarked_Q'] = (d.Embarked == 'Q')

    # Cabin
    d.loc[:, 'CabinNA'] = (d.Cabin.isnull())
    d.loc[:, 'CabinLetter'] = d.Cabin.astype(str).str[0]
    d.loc[:, 'CabinLetterC'] = (d.CabinLetter == 'C')
    d.loc[:, 'CabinLetterB'] = (d.CabinLetter == 'B')
    d.loc[:, 'CabinLetterD'] = (d.CabinLetter == 'D')
    d.loc[:, 'CabinLetterE'] = (d.CabinLetter == 'E')
    d.loc[:, 'CabinLetterA'] = (d.CabinLetter == 'A')
    d.loc[:, 'CabinLetterF'] = (d.CabinLetter == 'F')

# Dataframes
model_vars = ['Female', 'MidClass', 'UpClass', 'NullAge', 'Child', 'Elder',
              'Embarked_C', 'Embarked_Q', 'SibSp', 'Parch', 'Fare',
              'CabinLetterC', 'CabinLetterB', 'CabinLetterD', 'CabinLetterE',
              'CabinLetterA', 'CabinLetterF', ]

y = data_train.Survived
X = data_train[model_vars]

# Validation set
X_train, X_val, y_train, y_val = ms.train_test_split(X, y, test_size=.1,
                                                     random_state=370294753)
X_test = data_test[model_vars]

# Naive model comp
models = [lm.LogisticRegression(), lm.LogisticRegressionCV(),
          svm.SVC(probability=True),
          n.KNeighborsClassifier(), nb.GaussianNB(),
          en.RandomForestClassifier(), en.ExtraTreesClassifier(),
          en.GradientBoostingClassifier(), nn.MLPClassifier()]

labels = ['LR', 'CV', 'SVC', 'KNN', 'NB', 'RF', 'ET', 'GB', 'NN']
length = len(models)
auc = [None] * length
probs = [None] * length

for mod, i in zip(models, np.arange(length)):
    clf = mod.fit(X_train, y_train)
    prob = clf.predict_proba(X_val)[:, 1]
    auc[i] = m.roc_auc_score(y_val, prob)
    probs[i] = prob

# Bar comp
y_pos = np.arange(length)
plt.bar(y_pos, auc, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('AUC')
plt.title('AUC by Classifier')
plt.show()

# Bagging
l = len(probs[0])
avg = np.zeros(l)
for p in probs:
    avg = avg + p
avg = avg/l
auc_avg = m.roc_auc_score(y_val, avg)

# Tuning model
gb = en.GradientBoostingClassifier()

# Gridsearch parameters
loss = ['deviance']
lr = np.arange(0.1, 1, .3)
est = np.arange(1000, 5000, 2000)
depth = np.arange(1, 6, 2)
sub = np.arange(0.4, 1.2, .3)
params = {'loss': loss, 'learning_rate': lr, 'n_estimators': est,
          'max_depth': depth, 'subsample': sub}

# Gridsearch
#clf = ms.GridSearchCV(gb, params).fit(X_train, y_train)
#print(clf.best_params_)
#prob = clf.predict_proba(X_val)[:, 1]
#auc = m.roc_auc_score(y_val, prob)

# Final Model
clf_final = en.GradientBoostingClassifier(learning_rate=.05,
                                          n_estimators=1000, subsample=.5,
                                          max_depth=3)\
                                          .fit(X_train, y_train)
prob = clf_final.predict_proba(X_val)[:, 1]
auc_final = m.roc_auc_score(y_val, prob)
print(auc_final)

# # ROC
# fpr, tpr, thresh = m.roc_curve(y_val, prob)
# plt.plot(thresh, tpr)
# plt.show()

# Optimal cutoff
pred = (prob > .6)
a = m.accuracy_score(y_val, pred)

# Prediction
prob_out = clf_final.predict_proba(X_test)[:, 1]
pred_out = (prob_out > .6).astype(int)
d_out = {'PassengerId': data_test.PassengerId, 'Survived': pred_out}
df_out = pd.DataFrame(d_out)
df_out.to_csv('gbc_submission.csv', index=False)
