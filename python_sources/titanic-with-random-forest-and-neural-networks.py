#!/usr/bin/env python
# coding: utf-8

# This notebook documents my attempt to tackle the Titanic data set. I have spent some time reading and learning from others kernels and discussions at Kaggle. Hence most of the ideas here is not fully original and I don't claim I've invented the methods presented here. The purpose of this notebook is to map what I found most useful and interesting, and to record some useful tricks (mostly pandas and sklearn).

# Some kernels I've found particularly useful (there are many others):
# - Li-Yen Hsu: Titanic - Neural Network https://www.kaggle.com/liyenhsu/titanic-neural-network
# - Niklas Donges: End to End Project with Python https://www.kaggle.com/niklasdonges/end-to-end-project-with-python

# In[ ]:


import pandas as pd
import numpy as np
import random as rnd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# read data
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")


# ### Basic data overview

# In[ ]:


train_raw.head()


# In[ ]:


print(train_raw.columns)


# In[ ]:


train_raw.describe(include = "all")


# In[ ]:


train_raw.info()


# In[ ]:


test_raw.describe(include = "all")


# In[ ]:


test_raw.info()


# Missing values need to be fixed. **Ticket** has large number of duplicates in both train and test set. I will explore this a further below.

# ### Missing values

# In[ ]:


# missing values in train
train_raw.isnull().sum()


# In[ ]:


# missing values in test
test_raw.isnull().sum()


# We observe that:
# -  in the **training set**, some values of *Age*, *Cabin*, and *Embarked* are missing
# -  in the **test set**, some values of *Age*, *Fare*, and *Cabin* are missing
# -  for **Cabin**, more than half of values is missing in each set and therefore I will remove this feature. This might be something to work on in future -- explore the *Cabin* feature more in detail instead just dropping it.
# -  the rest of the missing values will be populated with estimates derived from the valid data

# **Working copies of the datasets**:
# Before removing and modifying features, I will make copies of the origianl datasets

# In[ ]:


# deep copy to copy data and indeces
train = train_raw.copy(deep=True)
test = test_raw.copy(deep=True)

# dropping Passenger's IDs and Cabin
train = train.drop(['PassengerId', 'Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)


# For the missing values of **Age**, I'll deal with those below, where I'll bin the age data and estimate the missing values from the other data. For now, let's set the NaN age to -1.

# In[ ]:


train['Age'] = train['Age'].fillna(-1)
test['Age'] = test['Age'].fillna(-1)


# Two values are missing in the **Embarked** column of the train set. 

# In[ ]:


train[train['Embarked'].isnull()]


# Let's have a look on the distribution of possible values of Embarked in the combined dataset.

# In[ ]:


(pd.concat([train,test])).groupby('Embarked').count()['Sex']


# Embarking in Southampton is by far the most common one and I will populate the missing values in the train set by 'S'.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')


# One value is missing in the **Fare** column of the test set.

# In[ ]:


test[test['Fare'].isnull()]


# Let's have a look if Fare is correlated with any other feature.

# In[ ]:


sns.pairplot(train, hue="Survived", dropna=True)


# In[ ]:


corr = train.corr()
sns.heatmap(corr, square=True, annot=True, center=0)


# *Pclass* is strongly correlated with *Fare* -- the lower the class, the lower is the fare. Let's populate the single NaN *Fare* in the test set by the mean *Fare* for its *Pclass* of 3.

# In[ ]:


mean_fare_3 = np.nanmean(pd.concat([train[train['Pclass']==3]['Fare'],test[test['Pclass']==3]['Fare']]).values)
print("Mean Fare of passengers with Pclass==3:", mean_fare_3)
test['Fare'] = test['Fare'].fillna(mean_fare_3)


# There are still missing values for *Age* and we will deal with those below.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ### Feature engineering
# **Ticket** -- Above analysis showed that more than half of the values in Ticket are duplicated. Let's have a closer look on the examples.

# In[ ]:


train["Ticket"][:20]


# Duplicated examples together:

# In[ ]:


pd.concat(g for _, g in train.groupby("Ticket") if len(g) > 1)[:20]


# It seems that the same Ticket is associated with people traveling together -- they often have same names (married couple, family) and same cabin. Is this correlated with survival?

# In[ ]:


train['Alone'] = 1
train.loc[train.duplicated(subset='Ticket'),'Alone'] = 0


# In[ ]:


sns.factorplot(x="Alone", y="Survived", data=train)


# It looks like the new feature **Alone** is correlated with the survival. So I will keep it and I will drop the feature Ticket now.

# In[ ]:


test['Alone'] = 1
test.loc[test.duplicated(subset='Ticket'),'Alone'] = 0


# I'll drop the *Ticket* feature for now. There are probably ways to use it further though (e.g. exploring the first letters) -- something to try in future.

# In[ ]:


train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)


# **Name** -- The feature does not bring much insight as it is. A popular (and logical) method among Kaggle kernels is to extract the title of each person from their names. The title corresponds to social and economic status and might be correlated with survival.

# In[ ]:


# the following is adopted from 
# https://www.kaggle.com/niklasdonges/end-to-end-project-with-python

data = [train, test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


pd.concat([train,test]).groupby('Title').count()['Sex']


# In[ ]:


sns.factorplot(x="Title", y="Survived", data=train)


# In[ ]:


train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)


# In[ ]:


sns.factorplot(x="SibSp", y="Survived", data=train)


# In[ ]:


sns.factorplot(x="Parch", y="Survived", data=train)


# In quite a few Kaggle kernels, I saw people using a variable mapping the family size of each passenger: Family size = number of children / parents + number of siblings = *Parch* + *SibSp*. 

# In[ ]:


train['FamSize'] = train["Parch"].values+train["SibSp"]
sns.factorplot(x='FamSize', y="Survived", data=train)


# In[ ]:


corr = train.corr()
sns.heatmap(corr, square=True, annot=True, center=0)


# From the above plot it is not obvious to me that *FamSize* should be more correlated than *Parch* and *SibSp* separately. Probably best test it in production though.

# In[ ]:


test['FamSize'] = test["Parch"].values+test["SibSp"]


# ### Categorical variables into dummy variables
# Categorical variables (*Sex*, *Title*, *Alone*, and *Embarked*) are better transformed to dummy variable (see e.g. [here](http://pbpython.com/categorical-encoding.html)). I will also transform *Pclass*, because its integer represantation doesn't map the variable well -- e.g. *Pclass* of 2 is not twice better than of 1. Dummy variable representation might work better than the integer numerical value for *Pclass*.
# 
# First, let's have look how many categories are there for each of those features and how many examples per each category.

# In[ ]:


pd.concat([train,test]).groupby('Sex').count()['Age']


# In[ ]:


pd.concat([train,test]).groupby('Title').count()['Age']


# In[ ]:


pd.concat([train,test]).groupby('Alone').count()['Age']


# In[ ]:


pd.concat([train,test]).groupby('Embarked').count()['Age']


# In[ ]:


pd.concat([train,test]).groupby('Pclass').count()['Age']


# The dummy variables can be easily created with pandas. I'll keep the original features as well.

# In[ ]:


data = [train, test]
dummies = ['Pclass', 'Sex', 'Embarked', 'Alone', 'Title']

# keeps the original variable as well
train = pd.concat([train, pd.get_dummies(train[dummies], columns=dummies)], axis=1)
test = pd.concat([test, pd.get_dummies(test[dummies], columns=dummies)], axis=1)

# replaces the original variable by the dummy columns
#train = pd.get_dummies(train, columns=dummies)
#test = pd.get_dummies(test, columns=dummies)


# In[ ]:


train.columns


# ### Continuous variables binning
# We are left with five continuous variables: *Age*, *Rate*, *SibSp*, *Parch*, and the newly introduced *FamSize*. 
# 
# **Age** has a substantial number of examples with a missing value. Let's have a look on some distributions. First just the distribution (histogram) of ages in the training set and how does it relate to survival.

# In[ ]:


plt.hist([train.loc[train['Survived']==0,'Age'].values, train.loc[train['Survived']==1,'Age'].values], 
         color=['r','b'], 
         alpha=0.5,
         label=['survived=0','survived=1'])
plt.legend()
plt.xlabel('Age')
plt.ylabel('#')


# We see that the fractions of survivals differ for different ages. This correlation could be more pronounced if we introduce binning in *Age*. Binning can also help to prevent overfitting (which I experienced as a major issue in my earlier modeling). Let's explore the binning choice.

# In[ ]:


f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True, figsize=(15,5))
data_plot = [train.loc[train['Survived']==0,'Age'].values, train.loc[train['Survived']==1,'Age'].values]
hist_arg = {'color': ['r','b'], 
            'alpha': 0.5,
            'label':['survived=0','survived=1']}
for bin_size,ax in zip([5,10,15,20],[ax1,ax2,ax3,ax4]):
    ax.hist(data_plot, 
            bins=range(-bin_size,90,bin_size),
            **hist_arg)
plt.legend()
plt.xlabel('Age')
plt.ylabel('#')


# In[ ]:


# this is binning used by Li-Yen Hsu in his/her kernel: Titanic - Neural Network
# in https://www.kaggle.com/liyenhsu/titanic-neural-network

plt.hist(data_plot, 
         bins=[ 0, 4, 12, 18, 30, 50, 65, 100],
         **hist_arg)


# In[ ]:


age_0 = train.loc[train['Survived']==0,'Age'].values
age_1 = train.loc[train['Survived']==1,'Age'].values

for bin_size in [5,10,15,20]:
    bins_0 = np.histogram(age_0,bins=range(-bin_size,90,bin_size), range=(-bin_size,80))
    bins_1 = np.histogram(age_1,bins=range(-bin_size,90,bin_size), range=(-bin_size,80))
    print(bins_1[0]/(bins_0[0]+bins_1[0]))

# and Li-Yen Hsu binning again
bins = [ 0, 4, 12, 18, 30, 50, 65, 100]
bins_0 = np.histogram(age_0,bins=bins)
bins_1 = np.histogram(age_1,bins=bins)
print(bins_1[0]/(bins_0[0]+bins_1[0]))


# I am not sure what is the best bins choice. This is definitely something to test / work on in the future.
# I will go with the last option, mostly because it is a nonlinear binning which combines smaller bins for younger ages, where the survival rate is higher, and larger bins for older passengers.
# 

# In[ ]:


bins_age = [ 0, 4, 12, 18, 30, 50, 65, 100]
lab_ages = [0,1,2,3,4,5,6]


# But what about the missing values of *Age*? Perhaps they could be inferred from the other data. Let's have a look on some correlations.

# In[ ]:


train.columns


# In[ ]:


cols_for_age_corr = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'Alone', 'FamSize']
corr = (train[cols_for_age_corr]).corr()
sns.heatmap(corr, square=True, annot=True, center=0)


# It looks that *Age* has quite a strong correlation with *Pclass*. What about *Titles*? Let's plot age distribution for different titles and calculate the mean age and its STD.

# In[ ]:


grid = sns.FacetGrid(train, col='Title', sharey=False)
grid.map(plt.hist, 'Age', bins=range(0,105,5))
plt.show()


# In[ ]:


data_all = pd.concat([train,test])
print('mean', data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).mean())
print('std', data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).std())


# In[ ]:


grid = sns.FacetGrid(train, col='Pclass', sharey=False)
grid.map(plt.hist, 'Age', bins=range(0,105,5))
plt.show()


# In[ ]:


plt.scatter(train['Age'].values, train['Fare'])


# In[ ]:


# counts of missing age values for different titles
train.loc[train['Age']<0].groupby('Title').count()['Sex']


# In[ ]:


plt.scatter(train['Age'].values, train['SibSp'])


# Hmmm, it looks like that there are some correlations. The strongest one is with *Title* and that's what I'll use. I will simply fill the missing age with a random value with Gaussian distribution of the mean and STD of given *Title*. Future work here could be invested in more sophisticated estimate of missing ages (e.g. through some ML models, see https://www.kaggle.com/liyenhsu/titanic-neural-network).

# In[ ]:


means = data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).mean()
stds = data_all.loc[data_all['Age']>0,['Title', 'Age']].groupby(['Title']).std()

np.random.seed(seed=666)
data = [train,test]
for data_i in data:
    ages_i = []
    for index, row in data_i[data_i['Age']<0].iterrows():
        mu = means.loc[means.index==row['Title'],'Age'].values
        std = stds.loc[stds.index==row['Title'],'Age'].values
        ages_i.append(np.random.normal(mu,std)[0])
    #print(ages_i)
    data_i.loc[data_i['Age']<0,'Age'] = ages_i


# Now when the missing values for *Age* are filled, we can bin them.

# In[ ]:


for data_i in data:
    data_i['Age_bin'] = pd.cut(data_i.Age,bins_age,labels=lab_ages)
train = pd.concat([train, pd.get_dummies(train['Age_bin'], columns=['Age_bin'], prefix='Age_bin', prefix_sep='_')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Age_bin'], columns=['Age_bin'], prefix='Age_bin', prefix_sep='_')], axis=1)


# Lastly, I want to come back and have a brief look on **Fare** data. The original (and current, since I haven't changed anything about *Fare* yet) correlation with Survived is 0.25. However during reading through various Kaggle discussions, I came across the information that *Fare* was actually charged for the whole group of passengers on the same ticket. Let's try recalculating *Fare* and see if the correlation is stronger.

# In[ ]:


n_on_ticket = []
for index, row in train_raw.iterrows():
    n_on_ticket.append(1.*sum(train_raw["Ticket"]==row['Ticket']))
fare_per_person = train_raw['Fare'].values/np.array(n_on_ticket)
train_raw['Fare_pp'] = fare_per_person


# In[ ]:


print(train_raw.corr())


# The correlation of *Fare_pp* (fare per person) and Survived is basically the same. The correlation of *Fare_pp* and *Pclass* and *Age* increased wrt to those with *Fare*; while with *SibSp* and *Parch* decreased. That make sense, because the number of people on a ticket should be related to *SibSp* and *Parch* and hence this dependency was removed by normalizing *Fare* by the number of people on the ticket. 
# 
# I'm not sure if *Fare_pp* can help, but let's add it to our working datasets.

# In[ ]:


train['Fare_pp'] = train_raw['Fare_pp']


# In[ ]:


# and now for test set
n_on_ticket = []
for index, row in test_raw.iterrows():
    n_on_ticket.append(1.*sum(test_raw["Ticket"]==row['Ticket']))
test['Fare_pp'] = test['Fare'].values/np.array(n_on_ticket)


# ## ML Models

# First let's select features that will be used in actual modeling.

# In[ ]:


# list of all features
train.columns


# In[ ]:


features_to_use = ['Pclass_1', 'Pclass_2', 'Pclass_3', 
                   'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S', 
                   'Alone_0', 'Alone_1', 
                   'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 
                   'Age_bin_0', 'Age_bin_1', 'Age_bin_2', 'Age_bin_3', 'Age_bin_4', 'Age_bin_5', 'Age_bin_6', 
                   'Fare_pp', 'SibSp', 'Parch']

# training data
X_all = train[features_to_use]
y_all = train['Survived']

# test data
X_test = test[features_to_use]
# test ID to save result
pr_id = test_raw['PassengerId']


# ### Random Forest

# I will start with Random Forest (RF, mostly because I already experimented with it in the past). RF has several hyper-parameters that can be tuned. I will use random search utility implemented in sklearn for this.

# In[ ]:


from scipy.stats import randint

# Randomized search on RF hyper parameters
# specify parameters and distributions to sample from
param_dist_rf = {"max_depth": randint(5, 30),
                 "min_samples_split": randint(2, 11),
                 "min_samples_leaf": randint(1, 6),
                 "max_leaf_nodes": randint(10, 50),
                 "criterion": ["gini", "entropy"],
                 "n_estimators": randint(100, 1000)}

# Utility function to report best scores
# see http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# wrap-up function for hyper-parameter tuning of general ML classifier
def model_random_search(cl, param_dist, n_iter, X, y):
    random_search = RandomizedSearchCV(cl, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       random_state=666,
                                       cv=4)
                                       #verbose=2)
    random_search = random_search.fit(X, y)
    report(random_search.cv_results_, n_top=3)
    return random_search


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# less iterations used for shorter running time
#random_search_rf1 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 30, X_all, y_all)
random_search_rf1 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 5, X_all, y_all)


# In[ ]:


# some helper functions

def save_prediction(answer, file_out):
    np.savetxt(file_out, answer, header='PassengerId,Survived', delimiter=',', fmt='%i', comments='')
    
def apply_cl(cl, xtest, pr_id):
    pr_test_data = cl.predict(xtest)
    
    answer = np.array([pr_id,pr_test_data]).T
    #print(np.shape(answer))
    return answer

def recall(y_hat, y_obs):
    true_pos = y_obs*y_hat
    return np.sum(true_pos)/np.sum(y_obs)

def precission(y_hat, y_obs):
    true_pos = y_obs*y_hat
    return np.sum(true_pos)/np.sum(y_hat)
    
def train_cl_param(cla, X, y, param):
    cl = cla(**param)
    cl.fit(X,y)
    try: score = cl.oob_score_
    except: score = cl.score(X,y)
    r = recall(cl.predict(X), y)
    p = precission(cl.predict(X), y)
    f = 2.*(p*r)/(p+r)
    print('precission =', p, 'recall =', r, 'f-score =', f, 'score', score)
    return cl

def write_result(cl, file_out, X_test, pr_id):
    an = apply_cl(cl, X_test, pr_id)
    save_prediction(an, file_out)


# Let's use the set of parameters that ranked highest in the random search.

# In[ ]:


par = {'max_leaf_nodes': 25, 
       'min_samples_split': 9, 
       'min_samples_leaf': 3, 
       'criterion': 'entropy', 
       'max_depth': 28, 
       'n_estimators': 557,
       'n_jobs':1,
       'oob_score':True}
rf = train_cl_param(RandomForestClassifier, X_all, y_all, par)
#write_result(rf, 'random_forest_3.csv', X_test, pr_id)


# This resulted in a submission with the public score of ~0.79. This is better than my earlier attempts using crudely cleaned data. Most importantly the RF model seems to generalize better. While my previous attempts with oob_score around 0.84 resulted in public score of about 0.77, the current model has oob_score of 0.82 and public score of ~0.79. The newly engineered features such as *Title* and more sophisticated filling of NaN values probably also helped.

# Let's have a look on **feature importances** of the above RF (see e.g. [here](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/) for more details).

# In[ ]:


def get_importances(cl, features):
    importances = cl.feature_importances_
    std = np.std([cl.feature_importances_ for tree in cl.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    for i in range(len(features)):
        print(features_to_use[indices[i]], importances[indices[i]], std[indices[i]])
get_importances(rf, features_to_use)


# It looks like some features are not as importand as others (as expected). Particularly *Age_bin_* and *Alone_* (this might be due to its possible correlation with *Parch* and *SibSp*?). Let's try a different *Age* binning and get rid of *Alone*.

# In[ ]:


# based on some discussions on Kaggle 
# (e.g. Oscar Takeshita https://www.kaggle.com/c/titanic/discussion/49105#279477), 
# I'll try two age bins ~ young and others
bins_age_2 = [ 0, 15, 100]
lab_ages_2 = [20,21]

for data_i in data:
    data_i['Age_bin_2'] = pd.cut(data_i.Age,bins_age_2,labels=lab_ages_2)
train = pd.concat([train, pd.get_dummies(train['Age_bin_2'], columns=['Age_bin_2'], prefix='Age_bin_2', prefix_sep='_')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Age_bin_2'], columns=['Age_bin_2'], prefix='Age_bin_2', prefix_sep='_')], axis=1)


# In[ ]:


features_to_use_2 = ['Pclass_1', 'Pclass_2', 'Pclass_3', 
                   'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S',
                   'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                   'Age_bin_2_0', 'Age_bin_2_1',
                   'Fare_pp', 'SibSp', 'Parch']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_search_rf2 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 5, train[features_to_use_2], y_all)
# random_search_rf2 = model_random_search(RandomForestClassifier(n_jobs=1), param_dist_rf, 100, train[features_to_use_2], y_all)


# In[ ]:


par =  {'max_depth': 12, 'criterion': 'gini', 'min_samples_leaf': 4, 'max_leaf_nodes': 17, 'min_samples_split': 2, 'n_estimators': 472,
       'n_jobs':1,
       'oob_score':True}
rf2 = train_cl_param(RandomForestClassifier, train[features_to_use_2], y_all, par)
#write_result(rf, 'random_forest_5.csv', test[features_to_use_2], pr_id)


# This actually results in even better public score of ~0.8.

# In[ ]:


get_importances(rf2, features_to_use_2)


# ### Neural network

# In[ ]:


# helper functions to generate distributions of hyper-parameters for NN

# learning rate of 0.0001--1, list of n_max floats 10**random_uniform
def rand_learning_rate(n_max=1000):
    return list(10.**np.random.uniform(-3,0,n_max))

# hidden layers: generates list of n_max tuples with 
# n_l_min--n_l_max integers, each between n_a_min and n_a_max
def rand_hidden_layer_sizes(n_l_min,n_l_max,n_a_min,n_a_max,n_max=1000):
    n_l = np.random.randint(n_l_min,n_l_max,n_max)
    list_hl = []
    for nl_i in n_l:
        list_hl.append(tuple(np.random.randint(n_a_min,n_a_max,nl_i)))
    return list_hl


# In[ ]:


# NN hyper parameters to test
param_dist_nn = {"activation": ["tanh", "relu"],
                 "learning_rate_init": rand_learning_rate(),
                 "hidden_layer_sizes": rand_hidden_layer_sizes(2,15,5,20),
                 "alpha": [0.00001,0.000006,0.000003,0.000001]
                }


# In[ ]:


from sklearn.neural_network import MLPClassifier

random_search_nn1 = model_random_search(MLPClassifier(batch_size=256), param_dist_nn, 5, X_all, y_all)
#random_search_nn1 = model_random_search(MLPClassifier(batch_size=256), param_dist_nn, 50, X_all, y_all)


# In[ ]:


par = {'learning_rate_init': 0.020871090748067426, 
       'alpha': 1e-06, 
       'activation': 'tanh', 
       'hidden_layer_sizes': (5, 9, 15, 10),
       'batch_size':256
       }
nn = train_cl_param(MLPClassifier, X_all, y_all, par)
#write_result(nn, 'nn_1.csv', X_test, pr_id)


# The public score of the above submission is 0.76, which is not better than the results of RF. It seems that NN (with the above parameters / parameters distributions) does not generalize as well as RF.

# ### Conclusions

# I will stop here because I'm quite happy with the public score of ~0.8. Possible improvements:
# - **Cabin**: Use this feature somehow instead of just dropping it. 
# - **Ticket**: Infer more from ticket numbers (not just *Alone* feature based on duplicity of *Ticket* values).
# - missing values of **Age**: More sophisticated estimate of the missing values (e.g. a decision tree or other ML method).
# - ML **hyper-parameters** could probably be tuned some more.
# - Experiment with **different binning**, mostly for *Age* and *Fare* (*Fare_pp*).
# - Detailed exploration of **feature importances** which might help prevent overfitting.
# - Try **different ML methods**.

# In[ ]:




