#!/usr/bin/env python
# coding: utf-8

# This is a fun Halloween competition. We have some characteristics of monsters and the goal is to predict the type of monsters: ghouls, goblins or ghosts.
# 
# At first I do data exploration to get some insights. Then I try various models for prediction.

# In[ ]:


#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB


# ## Data exploration

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# So there are 4 numerical variables and 1 categorical. And no missing values, which is nice!

# In[ ]:


train.describe(include='all')


# Numerical columns are either normalized or show a percentage, so no need to scale them.

# In[ ]:


train.head()


# In[ ]:


plt.subplot(1,4,1)
train.groupby('type').mean()['rotting_flesh'].plot(kind='bar',figsize=(7,4), color='r')
plt.subplot(1,4,2)
train.groupby('type').mean()['bone_length'].plot(kind='bar',figsize=(7,4), color='g')
plt.subplot(1,4,3)
train.groupby('type').mean()['hair_length'].plot(kind='bar',figsize=(7,4), color='y')
plt.subplot(1,4,4)
train.groupby('type').mean()['has_soul'].plot(kind='bar',figsize=(7,4), color='teal')


# It seems that all numerical features may be useful.

# In[ ]:


sns.factorplot("type", col="color", col_wrap=4, data=train, kind="count", size=2.4, aspect=.8)


# Funny, but many colors are evenly distributes among the monsters. So they maybe nor very useful for analysis.

# In[ ]:


#The graphs look much better with higher figsize.
fig, ax = plt.subplots(2, 2, figsize = (16, 12))
sns.pointplot(x="color", y="rotting_flesh", hue="type", data=train, ax = ax[0, 0])
sns.pointplot(x="color", y="bone_length", hue="type", data=train, ax = ax[0, 1])
sns.pointplot(x="color", y="hair_length", hue="type", data=train, ax = ax[1, 0])
sns.pointplot(x="color", y="has_soul", hue="type", data=train, ax = ax[1, 1])


# In most cases color won't "help" other variables to improve accuracy.

# In[ ]:


sns.pairplot(train, hue='type')


# This pairplot shows that data is distributed normally. And while most pairs are widely scattered (in relationship to the type), some of them show clusters: hair_length and has_soul, hair_length and bone_length. I decided to create new variables with multiplication of these columns and it worked great!

# ## Data preparation

# In[ ]:


train['hair_soul'] = train['hair_length'] * train['has_soul']
train['hair_bone'] = train['hair_length'] * train['bone_length']
test['hair_soul'] = test['hair_length'] * test['has_soul']
test['hair_bone'] = test['hair_length'] * test['bone_length']
train['hair_soul_bone'] = train['hair_length'] * train['has_soul'] * train['bone_length']
test['hair_soul_bone'] = test['hair_length'] * test['has_soul'] * test['bone_length']


# In[ ]:


#test_id will be used later, so save it
test_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


# In[ ]:


#Deal with 'color' column
col = 'color'
dummies = pd.get_dummies(train[col], drop_first=False)
dummies = dummies.add_prefix("{}#".format(col))
train.drop(col, axis=1, inplace=True)
train = train.join(dummies)
dummies = pd.get_dummies(test[col], drop_first=False)
dummies = dummies.add_prefix("{}#".format(col))
test.drop(col, axis=1, inplace=True)
test = test.join(dummies)


# In[ ]:


X_train = train.drop('type', axis=1)
le = LabelEncoder()
Y_train = le.fit_transform(train.type.values)
X_test = test


# In[ ]:


clf = RandomForestClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train)
indices = np.argsort(clf.feature_importances_)[::-1]

# Print the feature ranking
print('Feature ranking:')

for f in range(X_train.shape[1]):
    print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]],
                                      clf.feature_importances_[indices[f]]))


# Graphs and model show that color has little impact, so I won't use it. In fact I tried using it, but the result got worse.
# And three features, which I created, seem to be important!

# In[ ]:


best_features=X_train.columns[indices[0:7]]
X = X_train[best_features]
Xt = X_test[best_features]


# In[ ]:


#Splitting data for validation
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y_train, test_size=0.20, random_state=36)


# Tune the model. Normally you input all parameters and their potential values and run GridSearchCV. My PC isn't good enough so I divide parameters in two groups and repeatedly run two GridSearchCV until I'm satisfied with the result. This gives a balance between the quality and the speed.

# In[ ]:


forest = RandomForestClassifier(max_depth = 100,                                
                                min_samples_split =7,
                                min_weight_fraction_leaf = 0.0,
                                max_leaf_nodes = 60)

parameter_grid = {'n_estimators' : [10, 20, 100, 150],
                  'criterion' : ['gini', 'entropy'],
                  'max_features' : ['auto', 'sqrt', 'log2', None]
                 }

grid_search = GridSearchCV(forest, param_grid=parameter_grid, scoring='accuracy', cv=StratifiedKFold(5))
grid_search.fit(Xtrain, ytrain)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


forest = RandomForestClassifier(n_estimators = 150,
                                criterion = 'entropy',
                                max_features = 'auto')
parameter_grid = {
                  'max_depth' : [None, 5, 20, 100],
                  'min_samples_split' : [2, 5, 7],
                  'min_weight_fraction_leaf' : [0.0, 0.1],
                  'max_leaf_nodes' : [40, 60, 80],
                 }

grid_search = GridSearchCV(forest, param_grid=parameter_grid, scoring='accuracy', cv=StratifiedKFold(5))
grid_search.fit(Xtrain, ytrain)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# Calibrated classifier gives probabilities for each class, so to check the accuracy at first I chose the most probable class and convert it to values. Then I compare it to values of validation set.

# In[ ]:


#Optimal parameters
clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, criterion = 'entropy', max_features = 'auto',
                             min_samples_split=7, min_weight_fraction_leaf=0.0,
                             max_leaf_nodes=40, max_depth=20)
#Calibration improves probability predictions
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_clf.fit(Xtrain, ytrain)
y_val = calibrated_clf.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val, columns=le.classes_).idxmax(axis=1).values
                                   == le.inverse_transform(ytest))/len(ytest))


# I used the best parameters and validation accuracy is ~68-72%. Not bad. But let's try something else.

# In[ ]:


svc = svm.SVC(kernel='linear')
svc.fit(Xtrain, ytrain)
y_val_s = svc.predict(Xtest)
print("Validation accuracy: ", sum(le.inverse_transform(y_val_s)
                                   == le.inverse_transform(ytest))/len(ytest))


# Much better! Usually RandomForest requires a lot of data for good performance. It seems that in this case there was too little data for it.

# In[ ]:


#The last model is logistic regression
logreg = LogisticRegression()

parameter_grid = {'solver' : ['newton-cg', 'lbfgs'],
                  'multi_class' : ['ovr', 'multinomial'],
                  'C' : [0.005, 0.01, 1, 10, 100, 1000],
                  'tol': [0.0001, 0.001, 0.005]
                 }

grid_search = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(5))
grid_search.fit(Xtrain, ytrain)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


log_reg = LogisticRegression(C = 1, tol = 0.0001, solver='newton-cg', multi_class='multinomial')
log_reg.fit(Xtrain, ytrain)
y_val_l = log_reg.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l, columns=le.classes_).idxmax(axis=1).values
                                   == le.inverse_transform(ytest))/len(ytest))


# It seems that regression is better. The reason? As far as I understand, the algorithms are similar, but with different loss function. And most importantly: SVC is a hard classifier and LR gives probabilities.
# 
# And then I received an advise to try ensemble or voting. Let's see.
# Voting can be done manually or with sklearn classifier. For manual voting I need to make predictions for each classifier and to take the most common one. Advantage is that I may use any classifier I want, disadvantage is that I need to do it manually. Also, if some classifiers give predictions as classed and others as probability distribution, it complicates things.
# Or I can use sklearn.ensemble.VotingClassifier. Advantage is that it is easier to use. Disadvantage is that it may use only sklearn algorithms (or more precisely - algorithms with method "get_param") and only those which can give probability predictions (so no SVC and XGBooost). Well, SVC can be used if correct parameters are set.
# I tried both ways and got the same accuracy as a result.

# In[ ]:


clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'gini', max_features = 'sqrt',
                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                             max_leaf_nodes=40, max_depth=100)

calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

log_reg = LogisticRegression(C = 1, tol = 0.0001, solver='newton-cg', multi_class='multinomial')

gnb = GaussianNB()


# In[ ]:


calibrated_clf1 = CalibratedClassifierCV(RandomForestClassifier())

log_reg1 = LogisticRegression()

gnb1 = GaussianNB()


# As far as I can understand, while using hard voting, it is better to used unfitted estimators. Hard voting uses predicted class labels for majority rule voting. Soft voting predicts the class label based on the argmax of the sums of the predicted probalities, which is recommended for an ensemble of well-calibrated classifiers.
# And with soft voting we can use weights for models.

# In[ ]:


Vclf1 = VotingClassifier(estimators=[('LR', log_reg1), ('CRF', calibrated_clf1),
                                     ('GNB', gnb1)], voting='hard')
Vclf = VotingClassifier(estimators=[('LR', log_reg), ('CRF', calibrated_clf),
                                     ('GNB', gnb)], voting='soft', weights=[1,1,1])


# In[ ]:


hard_predict = le.inverse_transform(Vclf1.fit(X, Y_train).predict(Xt))
soft_predict = le.inverse_transform(Vclf.fit(X, Y_train).predict(Xt))


# In[ ]:


#Let's see the differences:
for i in range(len(hard_predict)):
    if hard_predict[i] != soft_predict[i]:
        print(i, hard_predict[i], soft_predict[i])


# There are differences, but both predictions give the same result on leaderboard. I think that some ensemble of voting classifiers could improve score. For example use different classifiers for several VotingClassifiers and them make a majority voting on these VotingClassifiers.

# In[ ]:


submission = pd.DataFrame({'id':test_id, 'type':hard_predict})


# In[ ]:


submission.to_csv('GGG_submission.csv', index=False)


# Previous solution gave me score of 0.73724. Current one - 0.74291.
