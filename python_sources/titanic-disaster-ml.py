#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# Thanks to this amazing [reference](http://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook#Step-5:-Model-Data)

# In[ ]:


data_train = pd.read_csv('../input/titanic/train.csv')
data_test  = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


data_train.head(5)


# Data cleaning

# In[ ]:


data_train.describe(include = 'all')


# In[ ]:


for dataset in [data_train, data_test]:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    drop_column = ['PassengerId','Cabin', 'Ticket', 'Name']
    dataset.drop(drop_column, axis=1, inplace = True)

data_train.describe(include = 'all')


# In[ ]:


for dataset in [data_train, data_test]:
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    dataset['Sex_Code'] = 0
    dataset['Sex_Code'].loc[dataset['Sex'] == 'male'] = 1
data_train.describe(include = 'all')


# In[ ]:


for dataset in [data_train, data_test]:
    dataset['Embarked_C'] = 0
    dataset['Embarked_Q'] = 0
    dataset['Embarked_S'] = 0
    dataset['Embarked_C'].loc[dataset['Embarked'] == 'C'] = 1
    dataset['Embarked_Q'].loc[dataset['Embarked'] == 'Q'] = 1
    dataset['Embarked_S'].loc[dataset['Embarked'] == 'S'] = 1
    
data_train.describe(include = 'all')


# In[ ]:


print(data_train.info())
print(data_train.isnull().sum())


# In[ ]:


print(data_test.info())
print(data_test.isnull().sum())


# In[ ]:


data_columns = ['Pclass','Sex_Code','Age', 'SibSp', 'Parch','Fare', 'FamilySize','IsAlone','Embarked_C','Embarked_Q','Embarked_S']
target = ['Survived']

#split train and test data 75/25
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data_train[data_columns], data_train[target], random_state = 0)

print("Data1 Shape: {}".format(data_train.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))


# Exploratory Analysis

# In[ ]:


# variable correlation
for x in data_columns:
    if data_train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data_train[[x, target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')


# In[ ]:


#pairplots
pp = sns.pairplot(data_train, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])


# In[ ]:


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data_train)


# In[ ]:


#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data_train, ax = qaxis[0])

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data_train, ax  = qaxis[1])

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data_train, ax  = qaxis[2])


# In[ ]:


#histogram comparison of sex, class, and age by survival
a = sns.FacetGrid( data_train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data_train['Age'].max()))
a.add_legend()


# Model data

# In[ ]:


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]


# In[ ]:


MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data_train[target]
# run model 10x with 60/30 split intentionally leaving out 10%
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 
#index through MLA and save performance to table
row_index = 0
for alg in MLA:
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data_train[data_columns], data_train[target], cv  = cv_split, return_train_score = True, )
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions - see section 6 for usage
    alg.fit(data_train[data_columns], data_train[target])
    MLA_predict[MLA_name] = alg.predict(data_train[data_columns])
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


# In[ ]:


#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# In[ ]:


grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

#extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990
submit_xgb = XGBClassifier()
submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
submit_xgb.fit(data_train[data_columns], data_train[target])
print('Best Parameters: ', submit_xgb.best_params_) 
#Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}
data_test['Survived'] = submit_xgb.predict(data_test[data_columns])


# In[ ]:


data_test.to_csv("../working/submit.csv", index=False)

print('Validation Data Distribution: \n', data_test['Survived'].value_counts(normalize = True))
data_test.sample(10)

