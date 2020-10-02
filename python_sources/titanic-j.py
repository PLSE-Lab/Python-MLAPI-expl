#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_all = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#print(train.columns.values)
#train.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 891 entries, 0 to 890
#Data columns (total 12 columns):
#PassengerId    891 non-null int64
#Survived       891 non-null int64
#Pclass         891 non-null int64
#Name           891 non-null object
#Sex            891 non-null object
#Age            714 non-null float64
#SibSp          891 non-null int64
#Parch          891 non-null int64
#Ticket         891 non-null object
#Fare           891 non-null float64
#Cabin          204 non-null object
#Embarked       889 non-null object
#dtypes: float64(2), int64(5), object(5)
#memory usage: 83.6+ KB


#train_all
train_all.head()
#train.index
#test.info()
#train is the only data that has a label. Test doens't have.


# In[3]:


#combine train and test set to check all the data categories
all = [train_all, test]
all = pd.concat(all)
#all.index
#all.head()
all.Pclass.value_counts()
#Pclass has 3 categories
all.PassengerId.value_counts()
all.Sex.value_counts()
#sex has 2 catg
all.SibSp.value_counts()
#SibSp has 7 values, what does this mean?
all.Parch.value_counts()
#Parch has 8 values, meaning?
all.Cabin.value_counts()
#Cabin has A-G values, need to get the first letter and see again
all.Embarked.value_counts()
#Embarked has 3 values


# In[4]:


print(train_all.columns.values)
len(np.unique(train_all['PassengerId']))
train_all.index


# In[5]:


#create numerical features
#train_all delete 'Name', 'Ticket', 'PassengerId'
#train_all = train_all[['Survived','PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

#gender
def gender(sex):
    if sex.lower().strip().replace(" ","") in 'male':
        return 1
    else:
        return 0
 
train_all['Gender']=train_all['Sex'].map(gender)


#Cabin
#by manual check, first letter will be sufficient, NaN will be N
train_all['Cabin_level']=train_all['Cabin'].astype(str).str[0]

#train_all.Cabin_level.value_counts()

#filling with Missing values
#Age: 177 is NaN, impute it with the mean (29.699)
#Embarked has 3 nan, impute with the most freq 'S'
train_all["Age"].fillna(train_all["Age"].mean(), inplace=True)
train_all['Embarked'].fillna(train_all['Embarked'].value_counts().index[0])
train_all = train_all[['Survived','Pclass','Gender','Age','SibSp','Parch','Fare','Cabin_level','Embarked']]

#train_all
train_all.info()
test.info()
#train_all.describe(include='all')
#train_all.head()


# In[6]:


#correlation
train_all.corr()
#Correlation: Survival v.s Gender> Pclass> Fare
# Fare vs Pclass -0.55 [class=1 is most wealthy]
train_all.groupby('Survived').mean()


# In[7]:


# create dataframes with an intercept column and dummy variables for
# Pclass Cabin_level Embarked
#reference: http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
from patsy import dmatrices
y, X = dmatrices('Survived ~ C(Pclass) + C(Cabin_level) + C(Embarked) + Gender +                   Age + SibSp + Parch + Fare',
                  train_all, return_type="dataframe")
X.columns
# fix column names of X
X = X.rename(columns = {'C(Pclass)[T.2]':'class_2',
                        'C(Pclass)[T.3]':'class_3',
                        'C(Cabin_level)[T.B]':'cabin_B',
                        'C(Cabin_level)[T.C]':'cabin_C',
                        'C(Cabin_level)[T.D]':'cabin_D',
                        'C(Cabin_level)[T.E]':'cabin_E',
                        'C(Cabin_level)[T.F]':'cabin_F',
                        'C(Cabin_level)[T.G]':'cabin_G',
                        'C(Cabin_level)[T.T]':'cabin_T',
                        'C(Cabin_level)[T.n]':'cabin_n',
                        'C(Embarked)[T.Q]':'Embarked_Q',
                        'C(Embarked)[T.S]':'Embarked_S'})
# flatten y into a 1-D array
y = np.ravel(y)
#y


# In[8]:


# evaluate the model by splitting into train and test sets
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
model1 = LogisticRegression()
model1.fit(X_train, y_train)


# In[9]:


# check the accuracy on the training set
print(model1.score(X_train, y_train))
# examine the coefficients
#pd.DataFrame(zip(X_train.columns, np.transpose(model1.coef_)))
#print(model1.coef_)
#print(list(zip(model1.coef_, X_train.columns)))
#X_train.columns
coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(model1.coef_))], axis = 1)
print(coefficients)


# In[10]:


# predict class labels for the test set
predicted = model1.predict(X_test)
#print(predicted)
# generate class probabilities
probs = model1.predict_proba(X_test)
#print(probs)


# In[11]:


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# In[12]:


# start working with train_all data so that categorical levels of one varialbe still be one variable:
X2 = train_all.drop('Survived', axis='columns')
y2 = train_all['Survived']
features = ['Pclass','Gender','Age','SibSp','Parch','Fare','Cabin_level','Embarked']
X3 = pd.get_dummies(X2[features])
X3_train, X3_test, y2_train, y2_test = train_test_split(X3, y2, test_size=0.3, random_state=0)
X3.head()
#X2_train.shape (623,8)


# In[13]:


##### 1.Logistic regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X3_train, y2_train)
print(logit.score(X3_train, y2_train))
print(logit.score(X3_test, y2_test))


# In[14]:


###### 2. Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X3_train, y2_train)
print(tree.score(X3_train, y2_train))
print(tree.score(X3_test, y2_test))
features = X3_train.columns
pd.Series(tree.feature_importances_, index=features).plot.bar(figsize=(18,7))


# In[15]:


###### 3. Random forest
from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier(n_estimators=10, random_state=0)
rnd.fit(X3_train, y2_train)
print('Training {}'.format(rnd.score(X3_train, y2_train)))
print('Testing  {}'.format(rnd.score(X3_test, y2_test)))
pd.Series(rnd.feature_importances_, index=X_train.columns).plot.bar(figsize=(18,7))


# In[16]:


##### 4. Gradiant Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(max_depth=8, learning_rate=0.1, n_estimators=20, random_state=0,max_features='sqrt')
gb.fit(X3_train, y2_train)
print('Training {}'.format(gb.score(X3_train, y2_train)))
print('Testing  {}'.format(gb.score(X3_test, y2_test)))

#predictors = [x for x in train.columns if x not in [target, IDcol]]

#predictors = [x for x in train.columns if x not in [target, IDcol]]
#param_test1 = {'n_estimators':range(20,81,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(train[predictors],train[target])

#param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
#param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch2.fit(train[predictors],train[target])
#gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[17]:


from sklearn.grid_search import GridSearchCV
param_test1 = {'n_estimators':[20,81,10]}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X3_train, y2_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print('Training {}'.format(gsearch1.score(X3_train, y2_train)))
print('Testing  {}'.format(gsearch1.score(X3_test, y2_test)))


# In[18]:


###### 5. support vector machines
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X3_train, y2_train)
print('Training {}'.format(svc.score(X3_train, y2_train)))
print('Testing  {}'.format(svc.score(X3_test, y2_test)))


# In[19]:



from sklearn.svm import SVC
svc2 = SVC()
svc2.fit(X3_train, y2_train)
print('Training {}'.format(svc2.score(X3_train, y2_train)))
print('Testing  {}'.format(svc2.score(X3_test, y2_test)))


# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.cross_validation import cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
#     
# pipe1 = Pipeline([['clf', LogisticRegression(random_state=0)]])
# pipe2 = Pipeline([
#                   ['clf', DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)]])
# pipe3 = Pipeline([
#                   ['clf', KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')]])
# pipe4 = Pipeline([
#                   ['clf', RandomForestClassifier(n_estimators=10, criterion='entropy',n_jobs=2)]])
# pipe5 = Pipeline([
#                   ['clf', SVC(kernel='linear', C=1.0, random_state=0)]])
# clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'SVM']
# 
# for clf, label in zip([pipe1, pipe2, pipe3, pipe4, pipe5], clf_labels):
#     #clf.fit(X=keep_char, y=y)
#     #print('Accuracy: %0.2f' % clf.score(keep_char,y))
#     
#     scores = cross_val_score(estimator=clf, X=X3, y=y2, cv=2, scoring='accuracy')
#     print('Accuracy: %0.2f [%s]' % (scores.mean(),label))
#     
# #Accuracy: 0.78 [Logistic Regression]
# #Accuracy: 0.79 [Decision Tree]
# #Accuracy: 0.65 [KNN]
# #Accuracy: 0.78 [Random Forest]
# #Accuracy: 0.78 [SVM]

# ###################MODEL TUNING
# from sklearn.grid_search import GridSearchCV
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# ##LR
# param_grid1 = [{'clf__C':param_range,
#                 'clf__penalty': ['l1']},
#                {'clf__C':param_range,
#                 'clf__penalty': ['l2']}]
# pipe1_gs = GridSearchCV(estimator=pipe1, param_grid=param_grid1, scoring='accuracy', 
#                         cv=5, n_jobs=-1)
# pipe1_gs = pipe1_gs.fit(X3, y2)
# print(pipe1_gs.best_score_)
# ##DT
# #pipe2_gs = GridSearchCV(estimator=pipe2, param_grid=[{'clf__max_depth':[np.arange(1,10)]}],
# #                        scoring='accuracy', cv=2, n_jobs=-1)
# #pipe2_gs = pipe2_gs.fit(X3, y2)
# #print(pipe2_gs.best_score_)
# ##KNN
# #pipe3_gs = GridSearchCV(estimator=pipe3, param_grid=[{'clf__n_neighbors':[1,3,5]}],
# #                        scoring='accuracy', cv=2, n_jobs=-1)
# #pipe3_gs = pipe3_gs.fit(X3, y2)
# #print(pipe3_gs.best_score_)
# ##RF
# pipe4_gs = GridSearchCV(estimator=pipe4, param_grid=[{'clf__n_estimators':[2,5,10,20]}],
#                         scoring='accuracy', cv=5, n_jobs=-1)
# pipe4_gs = pipe4_gs.fit(X3, y2)
# print(pipe4_gs.best_score_)
# ##SVM
# param_grid2 = [{'clf__C': param_range,
#                'clf__kernel': ['linear']},
#               {'clf__C': param_range,
#                'clf__gamma': param_range,
#                'clf__kernel': ['rbf']}] #list of dictioniary
# pipe5_gs = GridSearchCV(estimator=pipe5, param_grid=param_grid2, scoring='accuracy',cv=5,
#                   n_jobs=-1)
# pipe5_gs = pipe5_gs.fit(X3, y2)
# print(pipe5_gs.best_score_)
# 
# #for gs, label in zip([pipe1_gs, pipe2_gs, pipe3_gs, pipe4_gs, pipe5_gs], clf_labels):
# #    print('Accuracy: %0.2f [%s]' % (gs.best_score_, label))
# 

# In[20]:


#after evaluating all models above, i choose gradiant boosting
#model is gsearch1
#print(test)
test2 = test
test2['Gender']=test2['Sex'].map(gender)
test2['Cabin_level']=test2['Cabin'].astype(str).str[0]
test2["Age"].fillna(test2["Age"].mean(), inplace=True)
test2['Embarked'].fillna(test2['Embarked'].value_counts().index[0])
test2["Fare"].fillna(test2["Fare"].mean(), inplace=True)

features = ['Pclass','Gender','Age','SibSp','Parch','Fare','Cabin_level','Embarked']
test_dummy = pd.get_dummies(test2[features])
test_dummy['Cabin_level_T'] = 0
#df1['e'] = Series(np.random.randn(sLength), index=df1.index)
#y_pred = gsearch1.predict(test_dummy)
test_dummy.info()


# In[21]:


y_pred = gsearch1.predict(test_dummy)


# In[22]:


test2['Survived'] = y_pred
#test2.info()
submission = test2[['PassengerId', 'Survived']]
submission


# In[23]:


submission.to_csv("submission.csv")

