#!/usr/bin/env python
# coding: utf-8

# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import math
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.preprocessing import  LabelEncoder ,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score


# In[58]:


""""
PATH1 = os.path.join(os.getcwd(), os.path.join('data', 'train.csv'))
PATH2= os.path.join(os.getcwd(), os.path.join('data', 'test.csv'))
PATH3=os.path.join(os.getcwd(), os.path.join('data', 'gender_submission.csv'))
train = pd.read_csv(PATH1, delimiter=',')
test_df = pd.read_csv(PATH2, delimiter=',')
gender_submission = pd.read_csv(PATH3, delimiter=',')
train.head()
"""
test_df = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[59]:


train.head()


# In[60]:


#train.plot(kind='scatter', x='Fare', y='Survived')


# In[61]:




#test_dfWithAge = test_df[pd.isnull(test_df['Age']) == False]
#test_dfWithoutAge = test_df[pd.isnull(test_df['Age'])]


# In[62]:


all_data = [train, test_df]


# In[63]:


for dataset in all_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()


# In[64]:



title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in all_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[65]:


train.head(5)


# In[66]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[67]:


bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Title')


# In[68]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
all_data = [X_train, test_df]


# In[69]:


for data in all_data:
    data['family_size'] = data['SibSp'] + data['Parch'] +1


# In[ ]:





# In[70]:


all_data_train=X_train.drop(["Cabin","SibSp","Parch","Name","PassengerId","Ticket"], axis=1)
all_data_test=test_df.drop(["Cabin","SibSp","Parch","Name","Ticket","PassengerId"], axis=1)


# In[71]:


all_data_train.shape,all_data_test.shape


# In[72]:


all_data_train["Fare"] = all_data_train["Fare"].fillna(all_data_train["Fare"].mean())
all_data_test["Fare"] = all_data_test["Fare"].fillna(all_data_test["Fare"].mean())
#all_data_train["Age"] = all_data_train["Age"].fillna(all_data_train["Age"].median())
#all_data_test["Age"] = all_data_test["Age"].fillna(all_data_test["Age"].median())
mod = all_data_train.Embarked.value_counts().argmax()
all_data_train.Embarked.fillna(mod, inplace=True)


# In[73]:



LE2 = LabelEncoder()
all_data_train.Sex = LE2.fit_transform(all_data_train.Sex)
all_data_test.Sex = LE2.fit_transform(all_data_test.Sex)


# In[74]:


dumies= pd.get_dummies(all_data_train.Embarked)
dumies1=pd.get_dummies(all_data_test.Embarked)


# In[75]:


all_data_train=pd.concat([all_data_train,dumies],axis='columns')
all_data_test=pd.concat([all_data_test,dumies1],axis='columns')


# In[76]:


all_data_train=all_data_train.drop(['Embarked'],axis=1)
all_data_test=all_data_test.drop(['Embarked'],axis=1)


# In[77]:


all_data_test.head(2)


# # Dealing with missing values  

# In[79]:


trainWithAge = all_data_train[pd.isnull(all_data_train['Age']) == False]
trainWithoutAge = all_data_train[pd.isnull(all_data_train['Age'])]
testWithAge = all_data_test[pd.isnull(all_data_test['Age']) == False]
testWithoutAge = all_data_test[pd.isnull(all_data_test['Age'])]


# In[80]:


trainAgeTarget=trainWithoutAge.drop(["Age"], axis=1)
testAgeTarget=testWithoutAge.drop(["Age"], axis=1)


# In[81]:


xtrainWithAge = trainWithAge.drop(["Age"], axis=1)
ytrainWithAge=trainWithAge["Age"]
xtestWithAge = testWithAge.drop(["Age"], axis=1)
ytestWithAge=testWithAge["Age"]


# In[28]:


xtrainWithAge.shape,ytrainWithAge.shape


# In[82]:


xtestWithAge.shape,ytestWithAge.shape


# In[83]:


rfModel_age = RandomForestRegressor()
rfModel_age1 = RandomForestRegressor()
rfModel_age.fit(xtrainWithAge,ytrainWithAge)
rfModel_age1.fit(xtestWithAge,ytestWithAge)


# In[84]:


generatedAgeValues = rfModel_age.predict(trainAgeTarget)


# In[85]:


generatedAgeValues1 = rfModel_age1.predict(testAgeTarget)


# In[86]:


generatedAgeValues1.size


# In[87]:


generatedAgeValues.size


# In[32]:


all_data_train['Age']=np.array(all_data_train['Age'])


# In[88]:


all_data_test['Age'].isnull().sum()


# In[89]:


for i in  range(all_data_train['Age'].isnull().sum()):
    all_data_train['Age']=all_data_train['Age'].replace(np.NaN, generatedAgeValues[i])


# In[90]:


for i in  range(all_data_test['Age'].isnull().sum()):
    all_data_test['Age']=all_data_test['Age'].replace(np.NaN, generatedAgeValues1[i])


# In[91]:


all_data_train=pd.DataFrame(all_data_train)


# In[92]:


all_data_test=pd.DataFrame(all_data_test)


# In[93]:


all_data_train.isnull().sum()


# In[38]:


#all_data_test=all_data_test.dropna()
#all_data_test["Age"] = all_data_test["Age"].fillna(all_data_test["Age"].median())


# In[94]:


all_data_test.isnull().sum()


# In[95]:


sc_X = MinMaxScaler()
all_data_train_normalized = sc_X.fit_transform(all_data_train)
all_data_test_normalized = sc_X.transform(all_data_test)


# In[ ]:





# In[96]:


all_data_train_normalized=pd.DataFrame(all_data_train_normalized)
all_data_test_normalized=pd.DataFrame(all_data_test_normalized)


# In[44]:


""""
p_test = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
            param_grid = p_test, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(all_data_train_normalized,Y_train)
""""


# In[45]:


# tuning.best_params_, tuning.best_score_


# In[ ]:


#GB=GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10, 
         # learning_rate= 0.01, n_estimators= 1750)


# In[ ]:


#GB.fit(all_data_train_normalized,Y_train)


# In[ ]:


"""
pred = GB.predict(all_data_test_normalized)
pred =pd.DataFrame(pred,columns=['Survived'])

sub0 = pd.concat([test_df['PassengerId'],pred],axis=1)
sub0.to_csv('sub0.csv',index=False)"""


# In[97]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier


# In[ ]:





# In[47]:


"""
logi_clf = LogisticRegression(solver='lbfgs', max_iter=500)
logi_parm = {"C": [0.1, 0.5, 1, 5, 10, 50],
            'random_state': [0,1,2,3,4,5]}

svm_clf = SVC(probability=True)
svm_parm = {'kernel': ['rbf', 'poly'], 
            'C': [1, 5, 50, 100, 500, 1000,1500,2000], 
            'degree': [3, 5, 7], 
       'gamma':[0.01,0.04,.1,0.2,.3,.4,.6],
           'random_state': [0,1,2,3,4,5]}

dt_clf = DecisionTreeClassifier()
dt_parm = {'criterion':['gini', 'entropy'],
          'random_state': [0,1,2,3,4,5]}

knn_clf = KNeighborsClassifier()
knn_parm = {'n_neighbors':[5, 10, 15, 20], 
            'weights':['uniform', 'distance'], 
            'p': [1,2]}

gnb_clf = GaussianNB()
gnb_parm = {'priors':['None']}

clfs = [logi_clf, svm_clf, dt_clf, knn_clf]
params = [logi_parm, svm_parm, dt_parm, knn_parm] 
clf_names = ['logistic', 'SVM', 'DT', 'KNN', 'GNB']
"""


# In[48]:


"""
clfs_opt = []
clfs_best_scores = []
clfs_best_param = []
for clf_, param in zip(clfs, params):
    clf = RandomizedSearchCV(clf_, param, cv=5)
    clf.fit(all_data_train_normalized,Y_train)
    clfs_opt.append(clf.best_estimator_)
    clfs_best_scores.append(clf.best_score_)
    clfs_best_param.append(clf.best_params_)
"""


# In[49]:


max(clfs_best_scores)


# In[50]:


"""
arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]
"""


# In[ ]:


"""
all_Clfs_dict = {}
all_Clfs_list = []
for name, clf in zip(clf_names, clfs_opt):
    all_Clfs_dict[name] = clf
    all_Clfs_list.append((name, clf))
"""


# In[98]:


svm_clf1 = SVC(probability=True,random_state= 2, kernel='poly', gamma= 0.2, degree= 3, C= 5)


# In[99]:


svm_clf1.fit(all_data_train_normalized,Y_train)


# In[100]:


pred = svm_clf1.predict(all_data_test_normalized)
pred =pd.DataFrame(pred,columns=['Survived'])

sub2 = pd.concat([test_df['PassengerId'],pred],axis=1)
sub2.to_csv('sub2.csv',index=False)


# # Ensempling Methods

# # ## Voting Ensembling

# # Hard Voting
# 

# In[ ]:


#import sklearn.ensemble as ens 


# In[ ]:


"""
hard_voting_clf = ens.VotingClassifier(all_Clfs_list, voting='hard')
hard_voting_clf.fit(all_data_train_normalized,Y_train)
cross_val_score(hard_voting_clf,all_data_train_normalized,Y_train, cv=5).mean()
"""


# # Soft voting

# In[ ]:


"""
soft_voting_clf = ens.VotingClassifier(all_Clfs_list, voting='soft', weights=clfs_best_scores)
soft_voting_clf.fit(all_data_train_normalized,Y_train)
cross_val_score(soft_voting_clf,all_data_train_normalized,Y_train, cv=5).mean()
"""


# # Bagging Ensembling

# In[ ]:


"""
clf = ens.BaggingClassifier(base_estimator=clfs_opt[arg])
param = {'n_estimators':[10,50,100,500,100],
        'max_samples':[1.0, 0.9, 0.8],
        'bootstrap_features':[False, True],
        'random_state': [0,1,2,3,4,5]}
best_est_bagging = RandomizedSearchCV(clf, param, cv=5)
best_est_bagging.fit(all_data_train_normalized,Y_train)"""


# # Random Forest

# In[ ]:


"""
clf = ens.RandomForestClassifier()
param = {'n_estimators':[10,50,100,500,100],
         'criterion': ['gini', 'entropy'],}
RF = RandomizedSearchCV(clf, param, cv=5)
RF.fit(X_train_sc, y_train_df)
RF.best_score_
"""

