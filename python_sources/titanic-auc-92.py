#!/usr/bin/env python
# coding: utf-8

# #                                                                            TITANIC: Machine Learning from Disaster

# ![](http://blog.ecocentro.es/wp-content/uploads/2015/07/5b8be34e82bee5b84719daeec62cd868_large.jpeg)

# **Welcome!** 
# 
# I've done here all the basic steps you might need to solve a ML problem, such as EDA, feature engineering, encoding, simple model baseline, model assembly, parameter tunning and model evaluation. I hope this will help you with your further classification projects of this kind. If you like this, please *up-vote* and of course, I'll be glad if you drop me a comment in the section below.

# I watched these two short videos to get some "business understanding" background:
# *     Sinking of the Titanic: https://youtu.be/b0L_2jKEbA4
# *     50 Insane Facts: https: https://youtu.be/Rqbsrj6-FgM

# ## LOAD LIBRARIES

# In[ ]:


# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Common Tools
from sklearn.preprocessing import LabelEncoder
from collections import Counter

#Algorithms
from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Model
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
#from sklearn.ensemble import VotingClassifier

#Configure Defaults
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Check versions

# In[ ]:


pd.__version__


# In[ ]:


np.__version__


# In[ ]:


sns.__version__


# ## LOAD DATA

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.shape


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.shape


# ### Check for imbalance class problem

# In[ ]:


sns.countplot(x='Survived', data=train)
print("Survival rate: ", train.Survived.sum()/train.Survived.count())


# ## EDA

# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


# Describe categorical features
train.describe(include=['O'])


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


sns.pairplot(train, hue="Survived")


# In[ ]:


a = sns.FacetGrid(train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0, train['Age'].max()))
a.add_legend()


# In[ ]:


sns.boxplot(x="Pclass", y="Fare",data=train)


# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## DATA WRANGLING

# ### Outlier removal

# In[ ]:


q = train.Fare.quantile(0.99)
q


# In[ ]:


train = train[train['Fare'] < q]


# ### Concat data 

# In[ ]:


#Save Id for the submission at the very end.
Id = test['PassengerId']


# In[ ]:


#Get split marker
split = len(train)


# In[ ]:


#Merge into one dataset
data =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[ ]:


#We don't need the Id anymore now.
data.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


data.shape


# ### Handle Nulls 

# #### Age 

# In[ ]:


sns.distplot(data['Age'].dropna())


# In[ ]:


median = data["Age"].median()
std = data["Age"].std()
is_null = data["Age"].isnull().sum()
rand_age = np.random.randint(median - std, median + std, size = is_null)
age_slice = data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data["Age"] = age_slice
data["Age"] = data["Age"].astype(int)


# In[ ]:


#Check
sns.distplot(data['Age'])


# #### Others

# In[ ]:


data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)


# In[ ]:


data["Embarked"].isnull().sum()


# In[ ]:


data['Fare'].fillna(data['Fare'].mean(), inplace = True)


# #### Cabin

# In[ ]:


data["CabinBool"] = (data["Cabin"].notnull().astype('int'))


# In[ ]:


sns.barplot(x="CabinBool", y="Survived", data=data)


# ##### Deck

# In[ ]:


data['Deck'] = data.Cabin.str.extract('([a-zA-Z]+)', expand=False)
data[['Cabin', 'Deck']].sample(10)
data['Deck'] = data['Deck'].fillna('Z')
data = data.drop(['Cabin'], axis=1)


# #### Embarked

# In[ ]:


data.groupby(['Embarked'])['Survived'].count()


# #### Family

# ##### Family Size

# In[ ]:


data['FamilySize'] = data['SibSp'] + data['Parch']


# ##### Is Alone

# In[ ]:


data['IsAlone'] = 1 #default value


# In[ ]:


data['IsAlone'].loc[data['FamilySize'] > 0] = 0


# In[ ]:


sns.factorplot(x="IsAlone", y="Survived", data=data, kind="bar")


# #### Age

# In[ ]:


a = sns.FacetGrid(train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0, train['Age'].max()))
a.add_legend()


# In[ ]:


# Bucketize
bins = [-1, 13, 31, 60, 80]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
data['AgeBin'] = pd.cut(data["Age"], bins, labels = labels).astype('object')


# In[ ]:


#data['AgeBand'] = pd.cut(data['Age'], 5)
#data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


# Plot
sns.factorplot(x="AgeBin", y="Survived", data=data, kind="bar")


# ##### Is Baby

# In[ ]:


data['IsBaby'] = 0 #default value


# In[ ]:


data['IsBaby'].loc[data['Age'] <= 5] = 1


# In[ ]:


sns.factorplot(x="IsBaby", y="Survived", data=data, kind="bar")


# ##### Title

# In[ ]:


data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


data['Title'] = data['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')  
data['Title'] = data['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')


# In[ ]:


data['Title'] = data['Title'].astype('object')


# In[ ]:


data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# #### Fare 

# In[ ]:


f = sns.FacetGrid(train, hue = 'Survived', aspect=4 )
f.map(sns.kdeplot, 'Fare', shade= True )
f.set(xlim=(0, train['Fare'].max()))
f.add_legend()


# In[ ]:


# Bucketize
bins = [-np.inf, 20, 30, 110, np.inf]
labels = ['Low', 'Mid', 'High', 'Extreme']
data['FareBin'] = pd.cut(data["Fare"], bins, labels = labels).astype('object')


# In[ ]:


#data['FareBand'] = pd.qcut(data['Fare'], 4)
#data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


data.columns


# ## Baseline model

# In[ ]:


# To tincker around a bit
df = data


# In[ ]:


# Drop high cardinality
df = df.drop(['Ticket', 'Name', 'Fare'], axis=1)


# In[ ]:


from catboost import Pool, CatBoostClassifier, cv

#Split data
train = df[:split]
test = df[split:]

# Get variables for a model
x = train.drop(["Survived"], axis=1)
y = train["Survived"]

#Do train data splitting
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#We will predict this value for a submission
test.drop(["Survived"], axis = 1, inplace=True)

cat_features = np.where(x.dtypes != float)[0]

cat = CatBoostClassifier(one_hot_max_size=7, iterations=21, random_seed=42, use_best_model=True, eval_metric='Accuracy', loss_function='Logloss')

cat.fit(X_train, y_train, cat_features = cat_features, eval_set=(X_test, y_test))
pred = cat.predict(X_test)

pool = Pool(X_train, y_train, cat_features=cat_features)
cv_scores = cv(pool, cat.get_params(), fold_count=10, plot=True)
print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))
print('The test accuracy is :{:.6f}'.format(accuracy_score(y_test, cat.predict(X_test))))


# ## Correlation

# In[ ]:


def correlation_heatmap(df, method):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(method=method),
        cmap = colormap,
        square=True, 
        annot=True, 
        annot_kws={'fontsize':9 }
    )
    
    plt.title('Correlation Matrix', y=1.05, size=15)


# In[ ]:


correlation_heatmap(data, 'pearson')


# In[ ]:


# Drop low corrlations 
to_drop = ['Age', 'AgeBin', 'SibSp', 'Parch', 'FamilySize', 'Embarked', 'Title']
df = df.drop(to_drop, axis=1, inplace=False)


# In[ ]:


#Check
df.info()


# In[ ]:


#Check
data.columns


# # Feature encoding

# In[ ]:


# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df[categorical_cols].head()


# In[ ]:


data.info()


# ## Model 

# In[ ]:


#Split data
train = df[:split]
test = df[split:]

# Get variables for a model
x = train.drop(["Survived"], axis=1)
y = train["Survived"]

#Do train data splitting
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#We will predict this value for a submission
test.drop(["Survived"], axis = 1, inplace=True)


# ### Train&Score 

# In[ ]:


MLA = [
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    gaussian_process.GaussianProcessClassifier(),
    linear_model.LogisticRegressionCV(),
    linear_model.RidgeClassifierCV(),
    linear_model.Perceptron(),
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    neighbors.KNeighborsClassifier(),
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    xgb.XGBClassifier()
    ]


# In[ ]:


#Do some preperation for the loop
col = []
algorithms = pd.DataFrame(columns = col)
idx = 0

#Train and score algorithms
for a in MLA:
    
    a.fit(X_train, y_train)
    pred = a.predict(X_test)
    acc = accuracy_score(y_test, pred) #Other way: a.score(X_test, y_test)
    f1 = f1_score(y_test, pred)
    cv = cross_val_score(a, X_test, y_test).mean()
    
    Alg = a.__class__.__name__
    
    algorithms.loc[idx, 'Algorithm'] = Alg
    algorithms.loc[idx, 'Accuracy'] = round(acc * 100, 2)
    algorithms.loc[idx, 'F1 Score'] = round(f1 * 100, 2)
    algorithms.loc[idx, 'CV Score'] = round(cv * 100, 2)

    idx+=1


# In[ ]:


#Compare invidual models
algorithms.sort_values(by = ['CV Score'], ascending = False, inplace = True)    
algorithms.head()


# In[ ]:


#Plot them
g = sns.barplot("CV Score", "Algorithm", data = algorithms)
g.set_xlabel("CV score")
g = g.set_title("Algorithm Scores")


# ### Evaluation

# ### Parameter Tunning

# In[ ]:


kfold = StratifiedKFold(n_splits=10) #-> library from sklearn.model_selection import StratifiedKFold


# In[ ]:


# XGBoost Classifier
XGB = XGBClassifier()
xgb_param = {
    'loss' : ["deviance"],
     'n_estimators' : [100,200,300],
     'learning_rate': [0.1, 0.05, 0.01],
     'max_depth': [4, 8],
     'min_samples_leaf': [100,150],
     'max_features': [0.3, 0.1] 
    }

gsXGB = GridSearchCV(XGB, param_grid = xgb_param, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsXGB.fit(X_train,y_train)
XGB_best = gsXGB.best_estimator_

# Best score
gsXGB.best_score_


# In[ ]:


# SVC Classifier
SVC = svm.SVC(probability=True)
svc_param = {
    'kernel': ['rbf'], 
    'gamma': [ 0.001, 0.01, 0.1, 1],
    'C': [1, 10, 50, 100,200,300, 1000]
    }

gsSVC = GridSearchCV(SVC, param_grid = svc_param, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVC.fit(X_train,y_train)
SVC_best = gsSVC.best_estimator_

# Best score
gsSVC.best_score_


# In[ ]:


# Gradient Boosting Classifier
GB = ensemble.GradientBoostingClassifier()
gb_param = {
        'loss' : ["deviance"],
        'n_estimators' : [100,200,300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [4, 8],
        'min_samples_leaf': [100,150],
        'max_features': [0.3, 0.1] 
        }

gsGB = GridSearchCV(GB, param_grid = gb_param, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGB.fit(X_train,y_train)
GB_best = gsGB.best_estimator_

# Best score
gsGB.best_score_


# ### Model Assembly 

# In[ ]:


vc = ensemble.VotingClassifier(
    estimators = [('xgb', XGB_best), ('gbc',GB_best), ('svc', SVC_best)],
    voting='soft', n_jobs=4)


# In[ ]:


vc = vc.fit(X_train, y_train)
pred = vc.predict(X_test)
acc = accuracy_score(y_test, pred) #Other way: vc.score(X_test, y_test)
f1 = f1_score(y_test, pred)
cv = cross_val_score(vc, X_test, y_test).mean()

print("Accuracy: ", round(acc*100,2), "\nF1-Score: ", round(f1*100,2), "\nCV Score: ", round(cv*100,2))


# In[ ]:


ada = ensemble.AdaBoostClassifier()
ada.fit(X_train, y_train)

lg = linear_model.LogisticRegressionCV()
lg.fit(X_train, y_train)

vc2 = ensemble.VotingClassifier(
    estimators = [('ada', ada), ('lg',lg), ('VotingClassifier', vc)],
    voting='soft', n_jobs=4)
vc2.fit(X_train, y_train)


# ### AUROC 

# #### Curve 

# In[ ]:


y_scores = vc2.predict_proba(X_test)
y_scores = y_scores[:,1]


# In[ ]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_scores)


# In[ ]:


def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


# In[ ]:


plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)


# #### Score 

# In[ ]:


auroc = roc_auc_score(y_test, y_scores)
print("ROC-AUC Score:", auroc)


# ## Submission

# In[ ]:


pred = vc2.predict(test).astype(int)
target = pd.Series(pred, name='Survived')

output = pd.concat({'PassengerId':Id, 'Survived':target}
                   ,axis='columns')

output.to_csv('submission.csv', index=False, header=True)


# ### **If you like the notebook, please up-vote!**
