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

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from fancyimpute import mice

#Functions

#Data Loading
def load(train_path, test_path):
    df_train_x = pd.read_csv(train_path)
    df_test_x = pd.read_csv(test_path)
    
    m_all_y = df_train_x.loc[:, 'Survived']
    df_train_x = df_train_x.drop('Survived', axis = 1)
    
    df_all_x = pd.concat([df_train_x, df_test_x], ignore_index = True)
    
    l_train = len(df_train_x)
    print(df_all_x.info())
    return df_all_x, l_train, m_all_y

#Write to file
def write(df_pred, l_train):
    m_temp = np.zeros((len(df_pred),2))
    for i in range(len(m_temp)):
        m_temp[i,0] = l_train + 1 + i
        m_temp[i,1]  = df_pred[i]
        
    df_sample = pd.DataFrame(m_temp,columns=['PassengerId', 'Survived']).astype(int)
    
    del m_temp, i
    
    df_sample.to_csv("submission.csv", index = False)
    return print("Write Completed")

#Data Wrangling
def wrangle(df_all_x, m_all_y):    
    #Sex
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by ='Survived', ascending = False)

    df_all_x['Sex'] = df_all_x['Sex'].map({'male' : 0, 'female' : 1})
    
    #Title
    df_all_x.loc[:,'Title'] = df_all_x['Name'].str.extract('(?<=, )(.*?)(?=\.)', expand = True).values
    df_all_x.loc[df_all_x['Title'].isin(['Mme', 'Mrs']), 'Title'] = '0'
    df_all_x.loc[df_all_x['Title'].isin(['Mlle', 'Ms', 'Miss']), 'Title'] = '1'
    df_all_x.loc[df_all_x['Title'].isin(['Don', 'Sir', 'Rev', 'Dr']) , 'Title'] = '5'
    df_all_x.loc[df_all_x['Title'].isin(['Dona', 'Jonkheer', 'Lady', 'the Countess']) , 'Title'] = '2'
    df_all_x.loc[df_all_x['Title'].isin(['Major', 'Capt', 'Col']) , 'Title'] = '4'
    df_all_x.loc[df_all_x['Title'].isin(['Master']), 'Title'] = '3'
    df_all_x.loc[df_all_x['Title'].isin(['Mr']), 'Title'] = '6'
    
    df_all_x['Title'] = df_all_x['Title'].astype(int)
    
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Title', 'Survived']].groupby(['Title'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Surname
    df_all_x.loc[:,'Surname'] = df_all_x['Name'].str.extract('(.*?)(?=\,)', expand = True).values
    df_all_x['SameFamily'] = df_all_x['Surname'].duplicated(keep = False).astype(int)
    
    #Tickets
    df_all_x['Ticket'] = (df_all_x['Ticket'].str.extract('.*?([0-9]*)$', expand = True))
    df_all_x.loc[df_all_x['Ticket'] == '', 'Ticket'] = '370160'
    df_all_x['Ticket'] = df_all_x['Ticket'].astype(int)

    df_all_x['TicketNumLength'] = df_all_x['Ticket'].apply(lambda x: len(str(x).split(' ')[-1])).astype(int)

    df_all_x['JointTicket'] = 1
    df_all_x.loc[df_all_x['Ticket'].duplicated(keep = False) == False, 'JointTicket'] = 0
    
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['JointTicket', 'Survived']].groupby(['JointTicket'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    df_visual[['TicketNumLength', 'Survived']].groupby(['TicketNumLength'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Fare
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    
    #Cabin
    df_all_x.loc[pd.isnull(df_all_x['Cabin']) == True, 'Cabin'] = 'X'
    df_all_x['Cabin'] = df_all_x['Cabin'].astype(str).str[0]
    df_all_x.loc[df_all_x['Cabin'] == "X", 'Cabin'] = '0'
    df_all_x.loc[df_all_x['Cabin'] == "D", 'Cabin'] = '1'
    df_all_x.loc[df_all_x['Cabin'] == "E", 'Cabin'] = '2'
    df_all_x.loc[df_all_x['Cabin'] == "B", 'Cabin'] = '3'
    df_all_x.loc[df_all_x['Cabin'] == "F", 'Cabin'] = '4'
    df_all_x.loc[df_all_x['Cabin'] == "C", 'Cabin'] = '5'
    df_all_x.loc[df_all_x['Cabin'] == "G", 'Cabin'] = '6'
    df_all_x.loc[df_all_x['Cabin'] == "A", 'Cabin'] = '7'
    df_all_x.loc[df_all_x['Cabin'] == "T", 'Cabin'] = '8'
    df_all_x['Cabin'] = df_all_x['Cabin'].astype(int)
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Cabin', 'Survived']].groupby(['Cabin'], as_index = False).mean().sort_values(by ='Survived', ascending=False)
    
    #Embarked
    df_all_x.loc[pd.isnull(df_all_x['Embarked']) == True, 'Embarked'] = 'S'
    df_all_x['Embarked'] = df_all_x['Embarked'].map({'C' : 0, 'Q' : 1, 'S' : 2})
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by ='Survived', ascending=False)

    #Family size
    df_all_x['FamilySize'] = df_all_x['Parch'] + df_all_x['SibSp'] + 1
    
    #Dummifying Embarkation and Title
    #embarked_dummies = pd.get_dummies(df_all_x['Embarked']).astype(int)
    #title_dummies = pd.get_dummies(df_all_x['Title']).astype(int)
    #cabin_dummies = pd.get_dummies(df_all_x['Cabin']).astype(int)
    
    #Drop the Categorical
    df_all_x = df_all_x.drop('PassengerId', axis = 1)
    df_all_x = df_all_x.drop('Name', axis = 1)
    df_all_x = df_all_x.drop('Surname', axis = 1)
    #df_all_x = df_all_x.drop('Embarked', axis = 1)
    #df_all_x = df_all_x.drop('Title', axis = 1)
    #df_all_x = df_all_x.drop('Cabin', axis = 1)
    
    return df_all_x

def impute(df_all_x):
    df_all_x.columns
    df_filled_x = mice.MICE().complete(df_all_x.as_matrix())
    df_filled_x = pd.DataFrame(df_filled_x, columns = df_all_x.columns)
    df_filled_x['Age'] = np.round(df_filled_x['Age'])
    return df_filled_x

#New Features
def new_feat(df_all_x):
    df_all_x['ServentMisstress'] = 0
    df_all_x.loc[(df_all_x['JointTicket'] == 1) & (df_all_x['FamilySize'] == 1), 'ServentMisstress'] = 1
    df_all_x['FamilySize'] += df_all_x['ServentMisstress']
    
    df_all_x['FarePerPerson'] = df_all_x['Fare'] / df_all_x['FamilySize']
    
    df_all_x['ClassAge'] = df_all_x['Age'] * df_all_x['Pclass']
    df_all_x['ClassFare'] = df_all_x['Fare'] ** df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})
    df_all_x['ClassFamily'] = df_all_x['FamilySize'] ** df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})
    df_all_x['ClassSex'] = (df_all_x['Sex'] + 1) * df_all_x['Pclass'].map({1 : 3, 2 : 2, 3 : 1})

    df_all_x['Child'] = 0
    df_all_x.loc[df_all_x['Age'] < 18,'Child'] = 1
    
    df_all_x['Mother'] = 0
    df_all_x.loc[(df_all_x['Sex'] == 1) & (df_all_x['Parch'] > 0 ) & (df_all_x['Age'] >= 18 ) & (df_all_x['Title'] != 1 ), 'Mother'] = 1
    
    df_all_x.loc[df_all_x['FamilySize'] == 4, 'FamilySize'] = '4'
    df_all_x.loc[df_all_x['FamilySize'] == 3, 'FamilySize'] = '3'
    df_all_x.loc[df_all_x['FamilySize'] == 2, 'FamilySize'] = '3'
    df_all_x.loc[df_all_x['FamilySize'] == 7, 'FamilySize'] = '2'
    df_all_x.loc[df_all_x['FamilySize'] == 1, 'FamilySize'] = '2'
    df_all_x.loc[df_all_x['FamilySize'] == 5, 'FamilySize'] = '1'
    df_all_x.loc[df_all_x['FamilySize'] == 6, 'FamilySize'] = '1'
    df_all_x.loc[df_all_x['FamilySize'] == 8, 'FamilySize'] = '0'
    df_all_x.loc[df_all_x['FamilySize'] == 11, 'FamilySize'] = '0'
    df_all_x['FamilySize'] = df_all_x['FamilySize'].astype(int)
    df_visual = pd.concat([df_all_x.iloc[0:891, :], pd.DataFrame(m_all_y)], axis = 1)
    df_visual[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by ='Survived', ascending=False)

    return df_all_x

#End-training
def final(df_all_x, l_train, m_all_y):
    end_train_x = []
    end_train_y = []
    end_test_x = []
    end_train_x = pd.DataFrame(df_all_x[0:l_train].as_matrix()[:, 1:],
                               columns = df_all_x.columns[1:]
                               )
    end_train_y = m_all_y
    end_test_x = pd.DataFrame(df_all_x[l_train:].as_matrix()[:, 1:],
                              columns = df_all_x.columns[1:]
                              )

    return end_train_x, end_train_y, end_test_x

#Centering & Normalizing
def centernorm(end_train_x, end_test_x):
    end_means = end_train_x.mean(axis = 0)
    end_std = end_train_x.std(axis = 0)
    
    end_train_x -= end_means
    end_train_x /= end_std
    end_test_x -= end_means
    end_test_x /= end_std
    
    return end_train_x, end_test_x


#Drop the correlated Vars
def dropcor(end_train_x, end_test_x):
    print(pd.DataFrame(end_train_x).corr() > 0.6)
    end_train_x = end_train_x.drop('SibSp', axis = 1)
    end_test_x = end_test_x.drop('SibSp', axis = 1)
    
    end_train_x = end_train_x.drop('Parch', axis = 1)
    end_test_x = end_test_x.drop('Parch', axis = 1)

    end_train_x = end_train_x.drop('Fare', axis = 1)
    end_test_x = end_test_x.drop('Fare', axis = 1)

    end_train_x = end_train_x.drop('ClassFare', axis = 1)
    end_test_x = end_test_x.drop('ClassFare', axis = 1)

    end_train_x = end_train_x.drop('Sex', axis = 1)
    end_test_x = end_test_x.drop('Sex', axis = 1)
    
    return end_train_x, end_test_x


#Feature Selection
def feat_select(end_train_x, end_test_x)    :
    from sklearn.ensemble import RandomForestClassifier as rfc
    
    model_randfor = rfc(n_estimators = 1000,
                        max_features = None,
                        bootstrap = False,
                        oob_score = False,                                                
                        warm_start = True
                        )
    
    model_randfor.fit(end_train_x, end_train_y)
        
    importance = pd.DataFrame(model_randfor.feature_importances_,
                              columns = ['Importance'],
                              index = end_train_x.columns
                              )
    
    importance['Std'] = np.std([tree.feature_importances_
                                for tree in model_randfor.estimators_], axis = 0)
    
    importance = importance.sort_values(by = 'Importance',
                                        axis = 0,
                                        ascending = False
                                        )
        
    plt.bar(range(importance.shape[0]), 
            importance.loc[:, 'Importance'], 
            yerr = importance.loc[:, 'Std'], 
            align = 'center',
            
            )
    
    #Drop the Irrelevant Features
    end_train_x = end_train_x[importance.loc[importance['Importance'] > 0.01,].index]
    end_test_x = end_test_x[importance.loc[importance['Importance'] > 0.01,].index]
    
    return end_train_x, end_test_x, importance

def ensembl(end_train_x, end_test_x):
    #Models
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier as rfc
    
    model_randfor = rfc(n_estimators = 200,
                        max_features = None,
                        max_depth = 15,
                        min_samples_split = 10,
                        bootstrap = True,
                        oob_score = False,
                        warm_start = True
                        )
    
    model_randfor.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_randfor, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
     
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p0 = model_randfor.predict(end_test_x)
    
    
    #Adaptive Boosting
    from sklearn.ensemble import AdaBoostClassifier as abc
    from sklearn.tree import DecisionTreeClassifier as ctree
    model_abctree = abc(base_estimator = ctree(criterion = 'gini',
                                               splitter = 'best',
                                               max_features = 0.9,
                                               max_depth = 2,
                                               min_samples_split = 10,
                                               min_samples_leaf = 10,
                                               class_weight = None,
                                               max_leaf_nodes = 5,
                                               min_impurity_split = 10e-7,
                                               presort = True
                                               ),
                        n_estimators = 200,
                        learning_rate = 0.06,
                        )
    
    model_abctree.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_abctree, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p1 = model_abctree.predict(end_test_x)
    
    
    #KNN with bagging
    from sklearn.neighbors import KNeighborsClassifier as knn
    from sklearn.ensemble import BaggingClassifier as bc
    model_knnbc = bc(knn(n_neighbors = 2,
                         weights = 'uniform',
                         algorithm = 'auto',
                         p = 2,
                         n_jobs = 1
                         ),
                     n_estimators = 200,
                     max_samples = 0.71,
                     max_features = 0.71,
                     bootstrap = False,
                     bootstrap_features = True,
                     warm_start = True,
                     n_jobs = 1
                     )
    
    model_knnbc.fit(end_train_x, end_train_y) 
    
    scores = cross_val_score(estimator = model_knnbc, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p2 = model_knnbc.predict(end_test_x)
    
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier as ctree
    model_tree = ctree(criterion = 'gini',
                       splitter = 'best',
                       max_features = 0.9,
                       max_depth = 15,
                       min_samples_split = 2,
                       min_samples_leaf = 1,
                       class_weight = None,
                       max_leaf_nodes = 15,
                       min_impurity_split = 10e-3,
                       presort = True
                       )
    
    model_tree.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_tree, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
     
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p3 = model_tree.predict(end_test_x)
    
    
    #Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier as gbc
    model_gbctree = gbc(loss = 'deviance',
                        learning_rate = 0.3,
                        n_estimators = 300,
                        max_depth = 8,
                        max_leaf_nodes = 8,
                        min_samples_split = 2,
                        min_samples_leaf = 1,
                        max_features = None,
                        min_impurity_split = 1e-3,
                        presort = True
                        )
    
    model_gbctree.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_gbctree, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p4 = model_gbctree.predict(end_test_x)
    
    
    #Extremely Randomized Trees
    from sklearn.ensemble import ExtraTreesClassifier as etc
    model_exttree = etc(n_estimators = 200,
                        max_features = None,
                        max_depth = 15,
                        max_leaf_nodes = 10,
                        bootstrap = False,
                        oob_score = False
                        )
    
    model_exttree.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_exttree, 
                             X = end_train_x, 
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p5 = model_exttree.predict(end_test_x)
    
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression as lr
    model_lr = lr(penalty = 'l2', 
                  C = 0.1,
                  fit_intercept = True, 
                  intercept_scaling = 1, 
                  class_weight = None, 
                  solver = 'newton-cg', 
                  multi_class = 'multinomial',
                  max_iter = 200,
                  n_jobs = 1,
                  warm_start = True
                  )
    
    model_lr.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_lr, 
                              X = end_train_x, 
                              y = end_train_y,
                              cv = 5
                              )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p6 = model_lr.predict(end_test_x)
    
    
    #Support Vector Machines
    from sklearn import svm
    
    #SVC
    model_svc = svm.SVC(C = 0.75,
                        kernel = 'rbf',
                        gamma = 0.15
                        )
    
    model_svc.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_svc,
                             X = end_train_x,
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p7 = model_svc.predict(end_test_x)
    
    #NuSVC
    model_nusvc = svm.NuSVC(nu = 0.41,
                          kernel = 'rbf',
                          gamma = 0.09
                          )
    
    model_nusvc.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_nusvc,
                             X = end_train_x,
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p8 = model_svc.predict(end_test_x) 
    
    #LinearSVC
    model_linearsvc = svm.LinearSVC(penalty = 'l2',
                                    loss = 'squared_hinge',
                                    dual = False,
                                    C = 1e-2
                                    )
    
    model_linearsvc.fit(end_train_x, end_train_y)
    
    scores = cross_val_score(estimator = model_linearsvc,
                             X = end_train_x,
                             y = end_train_y,
                             cv = 5
                             )
    
    print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std() * 2))
    
    end_test_p9 = model_svc.predict(end_test_x)
       
    #Ensemble
    end_test_p = 1*end_test_p0 + 1*end_test_p1 + 1*end_test_p2 + 1*end_test_p3 + 1*end_test_p4 + 1*end_test_p5 + 1*end_test_p6 + 1*end_test_p7 + 1*end_test_p8 +1*end_test_p9
    end_test_p = np.round(end_test_p / 10)
    
    return end_test_p


#Script
np.random.seed(117)
df_all_x, l_train, m_all_y = load("../input/train.csv", "../input/test.csv")

df_all_x = wrangle(df_all_x, m_all_y)
df_all_x = impute(df_all_x)
df_all_x = new_feat(df_all_x)

end_train_x, end_train_y, end_test_x = final(df_all_x, l_train, m_all_y)
end_train_x, end_test_x = centernorm(end_train_x, end_test_x)

end_train_x, end_test_x = dropcor(end_train_x, end_test_x)
#end_train_x, end_test_x, importance = feat_select(end_train_x, end_test_x)

end_test_p = ensembl(end_train_x, end_test_x)

write(end_test_p, l_train)