#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split,RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.feature_selection import RFECV

import lightgbm as lgb

import re

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.describe()
# Missing Value in train['Age']


# In[ ]:


train.head()


# In[ ]:


test.describe()


# In[ ]:


data = train.append(test)
data.reset_index(inplace = True, drop = True)
data


# In[ ]:


sns.countplot(data['Survived'])
#The data is balanced


# In[ ]:


sns.countplot(data['Pclass'],hue = data['Survived'])


# In[ ]:


sns.countplot(data['Embarked'],hue = data['Survived'])


# In[ ]:


sns.countplot(data['Sex'],hue = data['Survived'])


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'Age', kde= False)
#younger people were apt to survive


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'Fare', kde= False)
#higher mortablity rate for lower priced passengers


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'Fare', kde= False)


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'Parch', kde= False)
#People who didn't bring their parents or children tend to have lower survival rate


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'SibSp', kde= False)
#People who didn't bring their brothers or sisters tend to have lower survival rate
#This feature can be combined with 'Parch', called family_size


# In[ ]:


data['Family_size'] = data['SibSp'] + data['Parch']+1


# In[ ]:


g = sns.FacetGrid(data, col = 'Survived')
g.map(sns.distplot, 'Family_size', kde= False)

#It can be more obvious that the bigger the families are the high survival rate they have


# In[ ]:


data['Name'].str.split(', ',expand = True)[1].str.split('.', expand = True)[0].unique()


# In[ ]:


#Combine titles according to the social status and genders

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty",
    "Dona" : "Royalty"
}


# we extract the title from each name
data['Title_status'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
data['Title_status'] = data['Title_status'].map(Title_Dictionary)

data['Title_status'].unique()


# In[ ]:


#one-hot encoding of 'Title_status'

titles_dummies = pd.get_dummies(data['Title_status'], prefix='Title')
data = pd.concat([data, titles_dummies], axis=1)
data.head()


# In[ ]:


data['Ticket'].head(10)


# In[ ]:


data['Ticket_alpha'] = data['Ticket'].apply(lambda x:x.replace(".","").replace("/", "").strip().split(" ")[0] if not x.isdigit() else 'X')
len(data['Ticket_alpha'].unique())


# In[ ]:


#Hard to engineer this feature
data['Ticket_number']=data['Ticket'].apply(lambda x:re.sub('[^0-9]','', x))
len(data['Ticket_number'].unique())


# In[ ]:


#One-hot encoding the 'Ticket_alpha'

ticket_dummies = pd.get_dummies(data['Ticket_alpha'], prefix='Ticket')
data = pd.concat([data, ticket_dummies], axis=1)
data.head()


# In[ ]:


#Dealing with missing value
data['Embarked'] = data['Embarked'].fillna('S')
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Cabin'] = data['Cabin'].apply(lambda x:str(x)[0] if not pd.isnull(x) else 'NoCabin')
data['Embarked'].unique()


# In[ ]:


#One-hot encoding the 'Embarked'

embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
data = pd.concat([data, embarked_dummies], axis=1)

data.head()


# In[ ]:


#One-hot encoding the 'Cabin'

cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')    
data = pd.concat([data, cabin_dummies], axis=1)

data.head()


# In[ ]:


#One-hot encoding the 'Cabin'

pclass_dummies = pd.get_dummies(data['Pclass'], prefix="Pclass")
data = pd.concat([data, pclass_dummies],axis=1)

data.head()


# In[ ]:


fig, ax = plt.subplots( figsize = (18,7) )
data['Log_Fare'] = (data['Fare']+1).map(lambda x : np.log10(x) if x > 0 else 0)
sns.boxplot(y='Pclass', x='Log_Fare',hue='Survived',data=data, orient='h'
                ,ax=ax,palette="Set1")
ax.set_title('Log_Fare & Pclass vs Survived ',fontsize = 20)
pd.pivot_table(data,values = ['Fare'], index = ['Pclass'], columns= ['Survived'] ,aggfunc = 'median').round(3)


# In[ ]:


# Making Bins
data['FareBin_4'] = pd.qcut(data['Fare'], 4)
data['FareBin_5'] = pd.qcut(data['Fare'], 5)
data['FareBin_6'] = pd.qcut(data['Fare'], 6)
data['FareBin_7'] = pd.qcut(data['Fare'], 7)

label = LabelEncoder()
data['FareBin_Code_4'] = label.fit_transform(data['FareBin_4'])
data['FareBin_Code_5'] = label.fit_transform(data['FareBin_5'])
data['FareBin_Code_6'] = label.fit_transform(data['FareBin_6'])
data['FareBin_Code_7'] = label.fit_transform(data['FareBin_7'])

# plots
fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4,sharey=True)
fig.set_figwidth(18)
for axi in [ax1, ax2, ax3, ax4]:
    axi.axhline(0.5,linestyle='dashed', c='black',alpha = .3)
g1 = sns.factorplot(x='FareBin_Code_4', y="Survived", data=data,kind='bar',ax=ax1)
g2 = sns.factorplot(x='FareBin_Code_5', y="Survived", data=data,kind='bar',ax=ax2)
g3 = sns.factorplot(x='FareBin_Code_6', y="Survived", data=data,kind='bar',ax=ax3)
g4 = sns.factorplot(x='FareBin_Code_7', y="Survived", data=data,kind='bar',ax=ax4)
# close FacetGrid object
plt.close(g1.fig)
plt.close(g2.fig)
plt.close(g3.fig)
plt.close(g4.fig)


# In[ ]:


data['Sex'] = data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
# splits again beacuse we just engineered new feature
df_train = data[:len(train)]
df_test = data[len(train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
# show columns
X.columns


# In[ ]:


Y.isnull().value_counts()


# In[ ]:


#5 fold got best oob score -> choose this
b4, b5, b6, b7 = ['Sex', 'Pclass','FareBin_Code_4'], ['Sex','Pclass','FareBin_Code_5'],['Sex','Pclass','FareBin_Code_6'], ['Sex','Pclass','FareBin_Code_7']
b4_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b4_Model.fit(X[b4], Y)
b5_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b5_Model.fit(X[b5], Y)
b6_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b6_Model.fit(X[b6], Y)
b7_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b7_Model.fit(X[b7], Y)
print('b4 oob score :%.5f' %(b4_Model.oob_score_),)
print('b5 oob score :%.5f '%(b5_Model.oob_score_),)
print('b6 oob score : %.5f' %(b6_Model.oob_score_),)
print('b7 oob score : %.5f' %(b7_Model.oob_score_),)


# In[ ]:


#one-hot encoding engineered Fare feature
cabin_dummies = pd.get_dummies(data['FareBin_Code_5'], prefix='Fare_group')    
data = pd.concat([data, cabin_dummies], axis=1)

data.head()


# In[ ]:


df_train['Ticket'].describe()


# In[ ]:


duplicate_ticket = []
for tk in data.Ticket.unique():
    tem = data.loc[data.Ticket == tk, 'Fare']
    if tem.count() > 1:
        duplicate_ticket.append(data.loc[data.Ticket == tk,['Name','Ticket','Fare','Cabin','Family_size','Survived']])
duplicate_ticket = pd.concat(duplicate_ticket)
duplicate_ticket.head(40)


# In[ ]:


group_ticket = duplicate_ticket.loc[(duplicate_ticket.Family_size == 1) & (duplicate_ticket.Survived.notnull())].head(7)
family_ticket = duplicate_ticket.loc[(duplicate_ticket.Family_size > 1) & (duplicate_ticket.Survived.notnull())].head(7)

print('people keep the same ticket: %.0f '%len(duplicate_ticket))
print('friends: %.0f '%len(duplicate_ticket[duplicate_ticket.Family_size == 1]))
print('families: %.0f '%len(duplicate_ticket[duplicate_ticket.Family_size > 1]))


# In[ ]:


# the same ticket family or friends
data['Connected_Survival'] = 0.5 # default 
for _, grp in data.groupby('Ticket'):
    if (len(grp) > 1):
        for ind, row in grp.iterrows():
            smax = grp.drop(ind)['Survived'].max()
            smin = grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin==0.0):
                data.loc[data['PassengerId'] == passID, 'Connected_Survival'] = 0
#print
print('people keep the same ticket: %.0f '%len(duplicate_ticket))
print("people have connected information : %.0f" 
      %(data[data['Connected_Survival']!=0.5].shape[0]))
data.groupby('Connected_Survival')[['Survived']].mean().round(3)


# In[ ]:


pd.set_option("display.max_columns", None)
data


# In[ ]:


sns.countplot(data['Cabin'], hue = data['Survived'])
#People in th cabin had high chance to survived


# In[ ]:


sns.countplot(data['Embarked'], hue = data['Survived'])
#People from S port got high survival rate


# In[ ]:


data['Sex'] = data['Sex'].astype('category').cat.codes


# In[ ]:


data.columns


# In[ ]:


#cleaning features
data.drop(['Cabin', 'Embarked', 'Fare', 'Name', 'Parch','Pclass','SibSp','Ticket','Ticket_alpha', 'Ticket_number',
          'Log_Fare', 'FareBin_4', 'FareBin_5','FareBin_6', 'FareBin_7', 'FareBin_Code_4','FareBin_Code_5','FareBin_Code_6', 'FareBin_Code_7','Title_status']
          ,axis =1 ,inplace = True)


# In[ ]:


data.head()


# In[ ]:


data['Age'].describe()


# In[ ]:


#About 20% of missing values in age column
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]

X = dataAgeNotNull.drop(['Survived','Age'], axis = 1, inplace = False)
y = dataAgeNotNull["Age"]
rf_age = RandomForestRegressor(n_estimators=2000,random_state=42,verbose = 1)
rf_age.fit(X, y)


dataAgeNull_2 = dataAgeNull.drop(['Survived','Age'], axis = 1, inplace = False)
ageNullValues = rf_age.predict(dataAgeNull_2)

dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)
sns.distplot(data['Age'])


# In[ ]:


#people with smaller age has higher survivial rate
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Age', kde=False)


# In[ ]:


#below certain age, the survivial rate is obviouly higher
data['Age'] = data['Age'].apply(lambda x : 1 if x<=16 else 0) 
data
# data['AgeBin_6'] = pd.qcut(data['Age'], 6)
# label = LabelEncoder()
# data['Age_Code_6'] = label.fit_transform(data['AgeBin_6'])

# pclass_dummies = pd.get_dummies(data['Age_Code_6'], prefix="Age")
# data = pd.concat([data, pclass_dummies],axis=1)
# data


# In[ ]:


data_train = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
data_test = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_train.columns


# In[ ]:


X = data_train.drop(['Survived','PassengerId'], axis = 1)
y = data_train['Survived']
X_test = data_test.drop(['Survived','PassengerId'], axis = 1)
random_state = 42
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = random_state)


# In[ ]:


X.head(10)


# In[ ]:


X_test.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "lgbm_clf_test = lgb.LGBMClassifier()\nlgbm_clf_test.fit(X, y)\nscores=cross_val_score(lgbm_clf_test, X, y, scoring='accuracy', cv=5)\nprint('{:.5f}'.format(np.mean(scores)))")


# In[ ]:


features = pd.DataFrame()
features['feature'] = X.columns
features['importance_lgbm_test'] = lgbm_clf_test.feature_importances_
features.head()


# In[ ]:


features.sort_values(by=['importance_lgbm_test'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 25))


# In[ ]:


#feature selection
model = SelectFromModel(lgbm_clf_test, prefit=True)
X_reduced = model.transform(X)
print (X_reduced.shape)


# In[ ]:


test_reduced = model.transform(X_test)
print(test_reduced.shape)


# In[ ]:


X_train_reduced, X_valid_reduced, y_train_reduced, y_valid_reduced = train_test_split(X_reduced, y, test_size = 0.2, random_state = random_state)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_params = {"early_stopping_rounds" : 50, \n             "eval_metric" : \'auc\', \n             "eval_set" : [(X_valid_reduced,y_valid_reduced)],\n             \'eval_names\': [\'valid\'],\n             \'verbose\': 0,\n             \'categorical_feature\': \'auto\'}\n\nparam_test = {\'learning_rate\' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],\n              \'n_estimators\' : [200, 300, 400, 500, 600, 800, 1000, 1500, 2000],\n              \'feature_fraction\': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],\n              \'bagging_fraction\': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],\n              \'bagging_freq\': [1, 2, 4, 5, 6, 8],\n              \'num_leaves\': [5, 10, 15, 20, 25, 30, 35, 40, 50, 60], \n              \'min_child_samples\': sp_randint(5,40), \n              \'min_sum_hessian_in_leaf\':[1e-5, 1e-4,1e-3,1e-2,1e-1],\n              \'min_child_weight\': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],\n              \'max_depth\': [-1],\n              \'reg_alpha\': [0, 0.1, 0.4, 0.5, 0.6],\n              \'reg_lambda\': [0, 0.1, 1, 5, 10, 15, 35, 40],\n              \'gamma\': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],\n              \'min_gain_to_split\':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}\n\n#number of combinations\nn_iter = 1000\n\n#intializing lgbm and lunching the search\nlgbm_clf = lgb.LGBMClassifier(boosting =\'rf\', random_state=random_state, silent=True, metric=\'None\', n_jobs=-1)\ngrid_search = RandomizedSearchCV(\n    estimator=lgbm_clf, param_distributions=param_test, \n    n_iter=n_iter,\n    scoring=\'accuracy\',\n    cv=5,\n    refit=True,\n    random_state=random_state,\n    verbose=True)\n\ngrid_search.fit(X_reduced, y, **fit_params)\nprint(\'Best score reached: {} with params: {} \'.format(grid_search.best_score_, grid_search.best_params_))\n\nopt_parameters =  grid_search.best_params_')


# In[ ]:


print(opt_parameters)


# In[ ]:


get_ipython().run_cell_magic('time', '', "lgbm_clf_tuned = lgb.LGBMClassifier(**opt_parameters)\nlgbm_clf_tuned.fit(X_reduced, y)\nscores_lgbm_tuned=cross_val_score(lgbm_clf_tuned, X_reduced, y, scoring='accuracy', cv=5)\nprint('{:.5f}'.format(np.mean(scores_lgbm_tuned)))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "import xgboost as xgb\n\nxgb_clf_test = xgb.XGBClassifier()\nxgb_clf_test.fit(X, y)\nscores=cross_val_score(xgb_clf_test, X, y, scoring='accuracy', cv=5)\nprint('{:.5f}'.format(np.mean(scores)))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import xgboost as xgb\n\nfit_params2 = {"early_stopping_rounds" : 50, \n             "eval_metric" : \'auc\', \n             "eval_set" : [(X_valid_reduced, y_valid_reduced)],\n             \'verbose\': 0}\n\nparam_test2 = {\'learning_rate\' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],\n               \'max_depth\': [3, 4, 5, 6, 7, 8, 9, 10],\n              \'n_estimators\' : [200, 300, 400, 500, 600, 800, 1000, 1500, 2000],\n              \'feature_fraction\': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,1],\n              \'bagging_fraction\': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,1],\n              \'bagging_freq\': [2, 4, 5, 6, 8],\n              \'min_child_samples\': sp_randint(10, 70), \n              \'min_sum_hessian_in_leaf\':[1e-5, 1e-4,1e-3,1e-2,1e-1],\n              \'min_child_weight\': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\n              \'reg_alpha\': [0, 0.1, 0.4, 0.5, 0.6],\n              \'reg_lambda\': [0, 0.1, 1, 5, 10, 15, 35, 40],\n              \'gamma\': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],\n              \'min_gain_to_split\':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}\n\n#number of combinations\nn_iter = 800\n\n#intializing lgbm and lunching the search\nxgb_clf = xgb.XGBClassifier(random_state=random_state, silent=True, metric=\'None\', n_jobs=-1)\ngrid_search = RandomizedSearchCV(\n    estimator=xgb_clf, param_distributions=param_test2, \n    n_iter=n_iter,\n    scoring=\'accuracy\',\n    cv=5,\n    refit=True,\n    random_state=random_state,\n    verbose=True)\n\ngrid_search.fit(X_reduced, y, **fit_params2)\nprint(\'Best score reached: {} with params: {} \'.format(grid_search.best_score_, grid_search.best_params_))\n\nopt_parameters2 =  grid_search.best_params_')


# In[ ]:


print(opt_parameters2)


# In[ ]:


get_ipython().run_cell_magic('time', '', "xgb_clf_tuned= xgb.XGBClassifier(**opt_parameters2)\nxgb_clf_tuned.fit(X_reduced, y)\nscoresXGB=cross_val_score(xgb_clf_tuned, X_reduced, y, scoring='accuracy', cv=5)\nprint('{:.5f}'.format(np.mean(scoresXGB)))\nscoresXGB")


# In[ ]:


get_ipython().run_cell_magic('time', '', "RF=RandomForestClassifier(random_state=1)\nRF.fit(X, y)\nscores_RF1=cross_val_score(RF,X,y,scoring='accuracy',cv=5)\nprint('{:.5f}'.format(np.mean(scores_RF1)))")


# In[ ]:


# %%time

param_test3 = {'bootstrap': [True],
               'max_features': ['sqrt', 'auto', 'log2'],
               'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
               'max_features': [2, 3, 4, 5, 6, 7, 10],
               'min_samples_leaf': [1, 3, 4, 5, 7, 10],
               'min_samples_split': [0.1 ,0.3, 0.5, 0.7, 0.9],
               'n_estimators': [200, 300, 400, 500, 600, 800, 1000, 1500, 2000]}

#number of combinations
n_iter = 800

#intializing lgbm and lunching the search
RF_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
grid_search = RandomizedSearchCV(estimator=RF_clf, 
                                 param_distributions=param_test3,
                                 n_iter=n_iter,
                                 scoring='accuracy',
                                 cv=5,
                                 refit=True,
                                 random_state=random_state,
                                 verbose=True)

grid_search.fit(X_reduced, y)
print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))

opt_parameters3 =  grid_search.best_params_


# In[ ]:


get_ipython().run_cell_magic('time', '', "RF_clf_tuned = RandomForestClassifier(**opt_parameters3)\nRF_clf_tuned.fit(X_reduced, y)\nscoresRF=cross_val_score(RF_clf_tuned, X_reduced, y, scoring='accuracy', cv=5)\nprint('{:.5f}'.format(np.mean(scoresRF)))")


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
svc=make_pipeline(StandardScaler(),SVC(random_state=1, probability=True))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=5)
GSSVM.fit(X_reduced, y)
scores_svm=cross_val_score(GSSVM, X_reduced.astype(float), y, scoring='accuracy', cv=5)

print('{:.5f}'.format(np.mean(scores_svm)))


# In[ ]:


from sklearn.linear_model import LogisticRegression
lo_clf = LogisticRegression(random_state=43, solver='lbfgs',multi_class='multinomial')
lo_clf.fit(X_reduced, y)
scores_lo=cross_val_score(lo_clf, X_reduced.astype(float), y, scoring='accuracy', cv=5)
print('{:.5f}'.format(np.mean(scores_lo)))


# In[ ]:


submit1 = pd.read_csv('../input/gender_submission.csv')
y_test1 = lgbm_clf_tuned.predict(test_reduced )
submit1['Survived'] = y_test1
submit1['Survived'] = submit1['Survived'].astype(int)
submit1.to_csv('submit_lgbm_tuned_reduced_training.csv', index= False)
submit1.head()


# In[ ]:


submit2 = pd.read_csv('../input/gender_submission.csv')
y_test2 = xgb_clf_tuned.predict(test_reduced )
submit2['Survived'] = y_test2 
submit2['Survived'] = submit2['Survived'].astype(int)
submit2.to_csv('submit_xgb_tuned_reduced_training.csv', index= False)
submit2


# In[ ]:


submit3 = pd.read_csv('../input/gender_submission.csv')
y_test3 = RF_clf_tuned.predict(test_reduced)
submit3['Survived'] = y_test3
submit3['Survived'] = submit3['Survived'].astype(int)
submit3.to_csv('submit_RF_baseline.csv', index= False)


# In[ ]:


submit4 = pd.read_csv('../input/gender_submission.csv')
y_test4 = GSSVM.predict(test_reduced)
submit4['Survived'] = y_test4
submit4['Survived'] = submit4['Survived'].astype(int)
submit4.to_csv('submit_svm.csv', index= False)


# In[ ]:


submit5 = pd.read_csv('../input/gender_submission.csv')
y_test5 = lo_clf.predict(test_reduced )
submit5['Survived'] = y_test5
submit5['Survived'] = submit5['Survived'].astype(int)
submit5.to_csv('submit_logist.csv', index= False)


# In[ ]:


#ensembling_averaging
trained_models = [lgbm_clf_tuned, xgb_clf_tuned, RF_clf_tuned, GSSVM, lo_clf]
predictions = []
for model in trained_models:
    predictions.append(model.predict_proba(test_reduced )[:,1])
    
submit6 = pd.read_csv('../input/gender_submission.csv')
predictions_df = pd.DataFrame(predictions).T
submit6['Survived'] = predictions_df.mean(axis=1)
submit6['PassengerId'] = submit6['PassengerId']
submit6['Survived'] = submit6['Survived'].map(lambda s: 1 if s >= 0.5 else 0)
submit6.to_csv('submit_ensemble_mean.csv', index= False)


# In[ ]:


#ensembling_voting
predictions2 = []
for model in trained_models:
    predictions2.append(model.predict(test_reduced))

submit7 = pd.read_csv('../input/gender_submission.csv')
predictions_df = pd.DataFrame(predictions).T
submit7['Survived'] = predictions_df.mode(axis=1)
submit7['PassengerId'] = submit7['PassengerId']
submit7['Survived'] = submit7['Survived'].map(lambda s: 1 if s >= 0.5 else 0)
submit7.to_csv('submit_ensemble_vote.csv', index= False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




