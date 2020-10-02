#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # setting seaborn default for plots
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# In[3]:


train_df.columns


# In[4]:


train_df.head()


# * Categorical: Survived, Sex, and Embarked. Ordinal: Pclass. <br>
# * Continous: Age, Fare. <br>
# * Discrete: SibSp, Parch. <br>

# In[5]:


train_df.describe()


# In[6]:


train_df.info()


# In[7]:


test_df.info()


# In[8]:


train_df.isnull().sum()


# In[9]:


test_df.isnull().sum()


# # Data Visualization

# In[10]:


train_df['Survived'].value_counts()


# In[11]:


def bar_chart(feature):
    survived = train_df[train_df['Survived']==1][feature].value_counts()
    dead = train_df[train_df['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[12]:


bar_chart('Sex')


# In[13]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[14]:


bar_chart('Pclass')


# In[15]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[16]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[17]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[18]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# # Feature engineering

# # Data Preprocessing

# In[19]:


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.columns


# In[20]:


train_test_data = [train_df, test_df] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    


# In[21]:


train_df.columns


# In[22]:


# train_df['Has_family'] = train_df['SibSp'] + train_df['Parch']
# test_df['Has_family'] = test_df['SibSp'] + test_df['Parch']
# train_df[train_df['Has_family']>0] = 1
# test_df[test_df['Has_family']>0] = 1


# In[23]:


X_train_df = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
X_test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])


# In[24]:


y_train_df = train_df['Survived']


# In[25]:


X_train_df.isnull().sum()


# In[26]:


X_test_df.isnull().sum()


# In[27]:


age_med = train_df.Age.median()
X_train_df.Age.fillna(age_med, inplace=True)
X_test_df.Age.fillna(age_med, inplace=True)


# In[28]:


# X_train_df.loc[ X_train_df['Age'] <= 5, 'Age']= 0
# X_train_df.loc[(X_train_df['Age'] > 5) & (X_train_df['Age'] <= 16), 'Age'] = 1
# X_train_df.loc[(X_train_df['Age'] > 16) & (X_train_df['Age'] <= 32), 'Age'] = 2
# X_train_df.loc[(X_train_df['Age'] > 32) & (X_train_df['Age'] <= 48), 'Age'] = 3
# X_train_df.loc[(X_train_df['Age'] > 48) & (X_train_df['Age'] <= 64), 'Age'] = 4
# X_train_df.loc[ X_train_df['Age'] > 64, 'Age'] = 5
# X_train_df


# In[29]:


# X_test_df.loc[ X_test_df['Age'] <= 5, 'Age']= 0
# X_test_df.loc[(X_test_df['Age'] > 5) & (X_test_df['Age'] <= 16), 'Age'] = 1
# X_test_df.loc[(X_test_df['Age'] > 16) & (X_test_df['Age'] <= 32), 'Age'] = 2
# X_test_df.loc[(X_test_df['Age'] > 32) & (X_test_df['Age'] <= 48), 'Age'] = 3
# X_test_df.loc[(X_test_df['Age'] > 48) & (X_test_df['Age'] <= 64), 'Age'] = 4
# X_test_df.loc[ X_test_df['Age'] > 64, 'Age'] = 5


# In[30]:


mod = X_train_df.Embarked.value_counts().argmax()
X_train_df.Embarked.fillna(mod, inplace=True)


# In[31]:


fare_med = train_df.Fare.median()
X_test_df.Fare.fillna(fare_med, inplace=True)


# In[32]:


X_train_df.isnull().sum()


# In[33]:


X_test_df.isnull().sum()


# # Lable Encoding

# In[ ]:


X_train_df.columns


# In[34]:


X_train_df.replace({"male": 0, "female": 1}, inplace=True)
X_test_df.replace({"male": 0, "female": 1}, inplace=True)
X_train_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
X_test_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)


# In[35]:


X_train_df.head()


# # OneHot Encoding

# In[36]:


X_train_df = pd.get_dummies(X_train_df, columns=['Pclass', 'Embarked', 'Title'], drop_first=True)
X_test_df = pd.get_dummies(X_test_df, columns=['Pclass', 'Embarked', 'Title'], drop_first=True)
X_train_df.head()


# In[37]:


X_test_df.head()


# In[38]:


# X_test_df['Age_5.0'] = 0


# In[39]:


X_train_df.shape, X_test_df.shape


# ## Data Scaling

# In[40]:


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train_sc = sc_X.fit_transform(X_train_df)
X_test_sc = sc_X.transform(X_test_df)


# # Algorithms Training

# In[41]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[42]:


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


# In[43]:


clfs_opt = []
clfs_best_scores = []
clfs_best_param = []
for clf_, param in zip(clfs, params):
    clf = RandomizedSearchCV(clf_, param, cv=5)
    clf.fit(X_train_sc, y_train_df)
    clfs_opt.append(clf.best_estimator_)
    clfs_best_scores.append(clf.best_score_)
    clfs_best_param.append(clf.best_params_)


# In[44]:


max(clfs_best_scores)


# In[45]:


arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]


# In[ ]:


gnb_score = cross_val_score(gnb_clf,X_train_sc, y_train_df, cv=5).mean()
gnb_clf.fit(X_train_sc, y_train_df)
clfs_opt.append(gnb_clf)
clfs_best_scores.append(gnb_score)


# In[46]:


all_Clfs_dict = {}
all_Clfs_list = []
for name, clf in zip(clf_names, clfs_opt):
    all_Clfs_dict[name] = clf
    all_Clfs_list.append((name, clf))


# In[47]:


max(clfs_best_scores)


# In[49]:


arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]


# In[50]:


clf = clfs_opt[arg]


# In[51]:


clf


# In[52]:


""""
pred = clf.predict(X_test_sc)
test_df = pd.read_csv(TEST_PATH)
y_test_df = test_df['PassengerId']
cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)
submit_df.to_csv('submission_best_est.csv', index=False)""""


# In[ ]:


submit_df.head()


# # Ensempling Methods

# ## Voting Ensembling

# ### Hard Voting

# In[53]:


import sklearn.ensemble as ens 


# In[54]:


hard_voting_clf = ens.VotingClassifier(all_Clfs_list, voting='hard')
hard_voting_clf.fit(X_train_sc, y_train_df)
cross_val_score(hard_voting_clf,X_train_sc, y_train_df, cv=5).mean()


# In[55]:


""""
clf = hard_voting_clf
pred = clf.predict(X_test_sc)
test_df = pd.read_csv(TEST_PATH)
y_test_df = test_df['PassengerId']
cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)
submit_df.to_csv('submission_hard_voting_clf.csv', index=False)""""


# ### Soft Voting

# In[56]:


soft_voting_clf = ens.VotingClassifier(all_Clfs_list, voting='soft', weights=clfs_best_scores)
soft_voting_clf.fit(X_train_sc, y_train_df)
cross_val_score(soft_voting_clf,X_train_sc, y_train_df, cv=5).mean()


# In[57]:


""""
clf = soft_voting_clf
pred = clf.predict(X_test_sc)
test_df = pd.read_csv(TEST_PATH)
y_test_df = test_df['PassengerId']
cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)
submit_df.to_csv('submission_soft_voting_clf.csv', index=False)""""


# ## Bagging Ensembling

# ### Bagging Meta-estimator

# In[58]:


clf = ens.BaggingClassifier(base_estimator=clfs_opt[arg])
param = {'n_estimators':[10,50,100,500,100],
        'max_samples':[1.0, 0.9, 0.8],
        'bootstrap_features':[False, True],
        'random_state': [0,1,2,3,4,5]}
best_est_bagging = RandomizedSearchCV(clf, param, cv=5)
best_est_bagging.fit(X_train_sc, y_train_df)


# In[59]:


best_est_bagging.best_score_


# In[62]:


clf = best_est_bagging.best_estimator_
pred = clf.predict(X_test_sc)
test_df = pd.read_csv(TEST_PATH)
y_test_df = test_df['PassengerId']
cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)
submit_df.to_csv('submission_bagging_best_clf.csv', index=False)


# ### Random Forest

# In[61]:


""""
clf = ens.RandomForestClassifier()
param = {'n_estimators':[10,50,100,500,100],
         'criterion': ['gini', 'entropy'],}
RF = RandomizedSearchCV(clf, param, cv=5)
RF.fit(X_train_sc, y_train_df)
RF.best_score_


# In[ ]:


""""
clf = RF.best_estimator_
pred = clf.predict(X_test_sc)
test_df = pd.read_csv(TEST_PATH)
y_test_df = test_df['PassengerId']
cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)
submit_df.to_csv('submission_RandomForest_clf.csv', index=False)""""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#xtest=pd.concat([xtest, xtrain['Survived']])


# In[ ]:





# In[ ]:





# In[ ]:




