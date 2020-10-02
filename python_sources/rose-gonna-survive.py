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


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


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

# ## Name

# In[19]:


train_test_data = [train_df, test_df] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'].value_counts()


# In[20]:


test_df['Title'].value_counts()


# In[21]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[22]:


bar_chart('Title')


# # Data Preprocessing

# In[23]:


X_train_df = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
X_test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])


# In[24]:


y_train_df = train_df['Survived']
y_test_df = test_df['PassengerId']


# In[25]:


X_train_df.isnull().sum()


# In[26]:


X_test_df.isnull().sum()


# In[27]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[28]:


X_train_df['Age'] = X_train_df[['Age','Pclass']].apply(impute_age,axis=1)
X_test_df['Age'] = X_test_df[['Age','Pclass']].apply(impute_age,axis=1)


# In[29]:


def Age_cat(x):
    if x <=4 :
        return 1
    elif x>4 and x<=14:
        return 2
    elif x>14 and x<=30:
        return 3
    else:
        return 4


# In[30]:


X_train_df['Age'] = X_train_df['Age'].apply(Age_cat)
X_test_df['Age'] = X_test_df['Age'].apply(Age_cat)


# In[31]:


X_train_df.Age.unique()


# In[32]:


X_train_df['With_someone'] = X_train_df['SibSp'] | X_train_df['Parch']
X_test_df['With_someone'] = X_test_df['SibSp'] | X_test_df['Parch']
X_train_df['Family'] = X_train_df['SibSp'] + X_train_df['Parch']+1
X_test_df['Family'] = X_test_df['SibSp'] + X_test_df['Parch']+1


# In[33]:


X_train_df['With_someone'] =X_train_df['With_someone'].apply(lambda x:1 if x >=1 else 0)
X_test_df['With_someone'] =X_test_df['With_someone'].apply(lambda x:1 if x >=1 else 0)


# In[34]:


X_train_df['With_someone'].unique()


# In[35]:


X_train_df.head()


# In[36]:


mod = X_train_df.Embarked.value_counts().argmax()
X_train_df.Embarked.fillna(mod, inplace=True)


# In[37]:


fare_med = train_df.Fare.median()
X_test_df.Fare.fillna(fare_med, inplace=True)


# In[38]:


X_train_df.isnull().sum()


# In[39]:


X_test_df.isnull().sum()


# # Lable Encoding

# In[40]:


X_train_df.columns


# In[41]:


X_train_df.replace({"male": 0, "female": 1}, inplace=True)
X_test_df.replace({"male": 0, "female": 1}, inplace=True)
X_train_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
X_test_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)


# In[42]:


X_train_df.head()


# # OneHot Encoding

# In[43]:


X_train_df = pd.get_dummies(X_train_df, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)
X_test_df = pd.get_dummies(X_test_df, columns=['Pclass', 'Embarked','Age','Title'], drop_first=True)
X_train_df.head()


# In[44]:


X_train_df = X_train_df.drop(columns=['SibSp','Parch'])
X_test_df = X_test_df.drop(columns=['SibSp','Parch'])


# In[45]:


X_train_df.shape, X_test_df.shape


# ## Data Scaling

# In[46]:


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train_df[['Fare','Family']] = sc_X.fit_transform(X_train_df[['Fare','Family']])
X_test_df[['Fare','Family']] = sc_X.transform(X_test_df[['Fare','Family']])


# In[47]:


X_train_df.head()


# # Algorithms Training

# In[48]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[49]:


logi_clf = LogisticRegression(random_state=0)
logi_parm = {"penalty": ['l1', 'l2'], "C": [0.1, 0.5, 1, 5, 10, 50]}

svm_clf = SVC(random_state=0)
svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50], 'degree': [3, 5, 7], 
            'gamma': ['auto', 'scale']}

dt_clf = DecisionTreeClassifier(random_state=0)
dt_parm = {'criterion':['gini', 'entropy']}

knn_clf = KNeighborsClassifier()
knn_parm = {'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], 'p': [1,2]}

gnb_clf = GaussianNB()
gnb_parm = {'priors':['None']}

clfs = [logi_clf, svm_clf, dt_clf, knn_clf]
params = [logi_parm, svm_parm, dt_parm, knn_parm] 


# In[ ]:


clf1 = RandomForestClassifier()
clf1.fit(X_train_df,y_train_df)
rf_rand = GridSearchCV(clf1,{'n_estimators':[50,100,200,300,500],'max_depth':[i for i in range (2,11)]},cv=10)
rf_rand.fit(X_train_df,y_train_df)
print(rf_rand.best_score_)
print(rf_rand.best_params_)


# In[ ]:


clf2 = GradientBoostingClassifier()
clf2.fit(X_train_df,y_train_df)
gb_rand = GridSearchCV(clf2,{'n_estimators':[50,100,200,300,500],'learning_rate':[0.01,0.1,1],'max_depth':[i for i in range (2,11)]},cv=10)
gb_rand.fit(X_train_df,y_train_df)
print(gb_rand.best_score_)
print(gb_rand.best_params_)


# In[50]:


clf3 = SVC(gamma='auto')
clf3.fit(X_train_df,y_train_df)
svc_rand = GridSearchCV(clf3,{'C':[5,10,15,20],'degree':[i for i in range(1,11)]},cv=10)
svc_rand.fit(X_train_df,y_train_df)
print(svc_rand.best_score_)
print(svc_rand.best_params_)


# In[ ]:


clf1 = RandomForestClassifier(max_depth=6,n_estimators=200)
clf1.fit(X_train_df,y_train_df)
clf2 = GradientBoostingClassifier(n_estimators=300,learning_rate=0.01,max_depth=4,random_state=0)
clf2.fit(X_train_df,y_train_df)
clf3 = SVC(C=5,degree=1,gamma='auto',probability=True)
clf3.fit(X_train_df,y_train_df)


# In[ ]:


eclf = VotingClassifier(estimators=[('rf',clf1),('gb',clf2),('svc',clf3)],voting='soft',weights=[2.5,2.5,2])


# In[ ]:


eclf.fit(X_train_df,y_train_df)


# In[ ]:


#clfs_opt = []
#clfs_best_scores = []
#clfs_best_param = []
#for clf_, param in zip(clfs, params):
#    clf = RandomizedSearchCV(clf_, param, cv=5)
#    clf.fit(X_train_sc, y_train_df)
#    clfs_opt.append(clf)
#    clfs_best_scores.append(clf.best_score_)
#    clfs_best_param.append(clf.best_params_)


# In[ ]:


#max(clfs_best_scores)


# In[ ]:


#arg = np.argmax(clfs_best_scores)
#clfs_best_param[arg]


# In[ ]:


#clf = clfs_opt[arg]


# In[ ]:


#pred = clf.predict(X_test_sc)


# In[ ]:


#Grad_clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)
#Grad_clf.fit(X_train_df,y_train_df)


# In[ ]:


#rand = RandomizedSearchCV(Grad_clf,{'learning_rate':[0.01,0.1,1],'max_depth':[1,5,10],
                                    #'n_estimators':[50,100,200,500]},n_iter=15,cv=10)


# In[ ]:


#rand.fit(X_train_df,y_train_df)


# In[ ]:


#print(rand.best_score_)
#print(rand.best_params_)


# In[ ]:


#Grad_clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.01,max_depth=5)
#Grad_clf.fit(X_train_df,y_train_df)


# In[ ]:


pred = eclf.predict(X_test_df)


# In[ ]:


cols = ['PassengerId', 'Survived']
submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 
                         columns=cols)


# In[ ]:


submit_df.to_csv('submission.csv', index=False)


# In[ ]:


submit_df.head()

