#!/usr/bin/env python
# coding: utf-8

# #### This work is largely inspired by the following notebooks:
# - [Titanic - Advanced Feature Engineering Tutorial by Gunes Evitan](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial)
# - [A Data Science Framework: To Achieve 99% Accuracy by LD Freeman](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)

# In[ ]:


# import necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
import matplotlib.pyplot as plt # plot
import os
import warnings
warnings.filterwarnings('ignore')


# # 1. Data importation

# #### Import the data and clean the dataframe

# In[ ]:


from sklearn.model_selection import train_test_split
# read csv file and EDA
titanic_train = pd.read_csv('../input/titanic/train.csv') ; titanic_test = pd.read_csv('../input/titanic/test.csv')
titanic_train = titanic_train.reset_index(drop=True)
Ntrain = titanic_train.shape[0] ; Ntest = titanic_test.shape[0]
titanic_all = pd.concat([titanic_train, titanic_test], sort=True).reset_index(drop=True)


# #### Show the original training dataset

# In[ ]:


titanic_train.head(5) 


# The survival rate seems to depend on the following features: sex, cabin, embarkment, and class of the cabin that are categorical variables and on the age, number of siblings and spouses, number of parents and children that are continuous variables (or floats/integers).
# 
# Let's clean a bit the dataframe by removing useless features (ticket), by categorising the age into different groups, and by simplifying the name and the cabin (what's really matter is the first letter).

# # 2. Data cleaning and feature engineering

# #### Add new categories from continuous variables (age, fare) or from feature extraction (cabin, name)

# In[ ]:


# clean_dataframe function   
def clean_dataframe(df):
    #
    # Age: replace NaN values by median of each class
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    #
    # Embarked category: Nan value embarked at Southampton according to google
    #mf_imputer = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = df['Embarked'].fillna('S')
    #
    # Fare
    df['Fare'].replace(np.nan, df.Fare.median(), inplace=True) # replace NaN values by median age
    #
    return df


# In[ ]:


# replace NaN values
#
titanic_train_clean = clean_dataframe(titanic_train)
titanic_test_clean = clean_dataframe(titanic_test)
titanic_all_clean = clean_dataframe(titanic_all)
#


# In[ ]:


titanic_all_clean.info()


# In[ ]:


def feature_engineering(df):
    #
    # Create deck feature from the first letter of the Cabin column (M stands for Missing)
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    idx = df[df['Deck'] == 'T'].index
    df.loc[idx, 'Deck'] = 'A' # Passenger in the T deck is changed to A
    df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
    df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
    #
    # Create Title and IsMarried features
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Family'] = df.Name.apply(lambda x: x.split(',')[0])
    df['IsMarried'] = 0
    df['IsMarried'].loc[df['Title'] == 'Mrs'] = 1
    df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss')
    df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Rare')
    #
    # Create other features
    df['FamilySize'] = df['SibSp']+df['Parch']+1
    df['FamilySize'] = df['FamilySize'].astype('int')
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    df['TicketFrequency'] = df.groupby('Ticket')['Ticket'].transform('count')
    #
    # Age: categorize age
    df['Age_category'] = pd.qcut(df['Age'], 10)
    #
    # categorize fare
    df['Fare_category'] = pd.qcut(df['Fare'], 13)
    #
    # categorize integer features
    list_int2cat = ['Pclass', 'SibSp', 'Parch']
    df[list_int2cat] = df[list_int2cat].astype('object')
    #
    # simplify name
    df.Name = df.Name.apply(lambda x: x.split(' ')[0][:-1])
    #
    # remove useless features
    #df = df.drop(['Ticket', 'Cabin'], axis=1)  #, 'Embarked'
    #
    return df


# In[ ]:


# modify and create features
#
titanic_all_clean = feature_engineering(titanic_all_clean)
#
titanic_train_clean = titanic_all_clean[:Ntrain]
titanic_train_clean['Survived'] = titanic_train['Survived']
titanic_test_clean = titanic_all_clean[Ntrain:]


# #### Label-encode and OneHot-encode the features.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#
# Label Encoding of non-numerical features
list_le = ['Age_category', 'Fare_category'] #'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Embarked', 'Cabin_simp', 'Title']
le = LabelEncoder()
titanic_all_encoded = titanic_all_clean.copy()
for feature in list_le:
    titanic_all_encoded[feature] = le.fit_transform(titanic_all_encoded[feature].astype('category'))


# In[ ]:


# One-Hot encoding of all features but Age_category and Fare_category that are ordinal 
list_oh = ['Pclass', 'Sex', 'Embarked', 'Deck', 'Title']#, 'Embarked'] #['Sex', 'Embarked', 'Deck', 'Title', 'FamilySize']
titanicoh = pd.get_dummies(titanic_all_clean[list_oh].astype('category')) 
titanic_all_encoded = titanic_all_encoded.join(titanicoh)


# In[ ]:


titanic_train_encoded = titanic_all_encoded[:Ntrain]
titanic_train_encoded['Survived'] = titanic_train['Survived']
titanic_test_encoded = titanic_all_encoded[Ntrain:]


# In[ ]:


titanic_train_encoded.info()


# # 3. EDA

# ### Plot the survival rate for different categories

# In[ ]:


plt.figure(0,figsize=[15,12])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
list_features = ['Age_category', 'Fare_category', 'Deck', 'Pclass', 'FamilySize', 'Parch', 'Title']
#
plt.subplot(3,3,1)
barplot = sns.barplot(x='Sex', y="Survived", data=titanic_train_clean)
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)
for ifeat, feature in enumerate(list_features):
    plt.subplot(3,3,ifeat+2)
    barplot = sns.barplot(x=feature, y="Survived", hue="Sex", data=titanic_train_clean)
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)


# Looking at the barplots, it seems that the sex has the strongest influence on the survival probability, followed by age, class, ticket fare. Number of siblings and parents seems to be less important. 

# # 4. Evaluate performance of various classifiers

# In[ ]:


# import the necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier


# #### Define the classifiers and their hyperparameters.

# Four classifiers are used here: Random Forest, Adaboost, Gradient Boosting, and XGBoost. For each classifier, a (small) grid search is performed through GridSearchCV with two or three hyper-parameters to tune. It takes about 30 minutes to run the four grids with 4 CPUs. 

# In[ ]:


# name and parameters of different classifiers
SEED = 59
#
model_models  = [# Random Forest: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
                 ('randomforest', 
                  RandomForestClassifier(criterion='gini',min_samples_split=6,min_samples_leaf=6,
                                         max_features='auto',oob_score=True,random_state=SEED,n_jobs=-1,verbose=1)),
                 # Adaboost: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                 ('adaboost', 
                  AdaBoostClassifier(random_state=SEED)),
                 # Gradient Boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
                 ('gradientboosting',  
                 GradientBoostingClassifier(random_state=SEED,verbose=1,learning_rate=0.1)),
                 # XGBoost: http://xgboost.readthedocs.io/en/latest/parameter.html
                 ('xgboost', 
                  xgb.XGBClassifier(objective='binary:logistic',min_child_weight= 2,eta=0.8, subsample=0.8,colsample_bytree=0.8,
                                    scale_pos_weight=1,nthread=-1))
                 ]
#
model_params = [# Random forest
                {'max_depth' : np.arange(2,14,2), #[2,4,6,8,10,12], 
                 'n_estimators' : np.logspace(1,3,9).astype(int), #[10, 30, 100, 300, 1000], #
                 #'min_samples_split' : np.arange(4,14,2), #[1,2,4,6,8,10]
                 #'min_samples_leaf' : np.arange(4,14,2)
                 }, 
                # Adaboost
                {'learning_rate' : [0.1,0.2,0.5,1,1.2,1.5,2], #np.logspace(-2,1,7), 
                 'n_estimators' : np.logspace(1,3,9).astype(int)
                }, 
                # Gradient boosting
                {'max_depth' : [2,4,6,8,10,12], 
                 'n_estimators' : np.logspace(1,4,7).astype(int),
                 'learning_rate' : [0.1,0.5,1,2], #np.logspace(-2,1,7)}, 
                },
                # XGBoost
                {'max_depth' : [2,4,6,8,10,12,14,16,18], #[4,5,6,7,8], #
                 'n_estimators' : np.logspace(0,3,7).astype(int), #[2,3,4,5,6],#
                # 'eta' : [0,0.2,0.4,0.6,0.8,1.], #[0.7,0.75,0.8,0.85,0.9],#
                # 'gamma' : [0], #[0,1,10,100,1000],
                # 'min_child_weight' : [2,4,6,8,10,12] #[4,5,6,7,8], #
                }, 
                ]                  


# #### Prepare the training, test, and validation sets.

# In[ ]:


# define training and test sets
list_drop = ['Age', 'Fare', 'Name', 'Parch', 'PassengerId', 'Pclass', 'SibSp', 'Deck', 'Title', 'Sex', 
             #'TicketSurvivalRate', 'TicketSurvivalRateNA', 'FamilySurvivalRate', 'FamilySurvivalRateNA', 
             'Embarked', 'Ticket', 'Cabin', 'Family']
num_test = 0.20
#
X_all = StandardScaler().fit_transform(titanic_train_encoded.drop(list_drop, axis=1).drop('Survived',axis=1)) #
y_all = titanic_train['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
X_test = StandardScaler().fit_transform(titanic_test_encoded.drop(list_drop, axis=1).drop('Survived',axis=1))


# #### Look for hyper-parameters that give the best predictions through GridSearchCV with a cross validation. 

# In[ ]:


# Run a GridSearchCV on a specified dataset with a given model and return the model object
def score_model(model_tupple, model_param, X, y, verbose):
    modelname, model = model_tupple
    steps = [('scaler', RobustScaler()), #StandardScaler()), #
             (modelname, model)]
    parameters = model_param 
    acc_scorer = make_scorer(accuracy_score)
    pipeline = Pipeline(steps)
    grid_obj = GridSearchCV(model, param_grid=parameters,cv=5,                            scoring='accuracy',
                            return_train_score=True,verbose=verbose,n_jobs=4) #, 
    grid_obj.fit(X, y)
    return grid_obj 


# In[ ]:


# run a GridSearchCV for a list of models with a given list of parameters
def rungrid(list_models, list_params, X, y):
    list_results = [] ; list_bestestimator = [] ; list_bestparam = [] ; list_bestscore = []
    for im, model in enumerate(list_models):
        print('Running %s model' % (model[0]))
        model_tupple = model 
        model_param = list_params[im] 
        grid_obj = score_model(model_tupple, model_param, X, y, 1)
        result = pd.DataFrame(grid_obj.cv_results_)
        bestest = grid_obj.best_estimator_
        bestparam = grid_obj.best_params_
        bestscore = grid_obj.best_score_ 
        list_results.append(result) ; list_bestestimator.append(bestest)
        list_bestparam.append(bestparam) ; list_bestscore.append(bestscore)
        list_results[-1].to_csv('results_'+model_tupple[0]+'.csv')
        print("%s score: %.3f obtained for parameter value(s)" % 
              (model_tupple[0], list_bestscore[-1]))
        print(list_bestparam[-1])
    return list_results, list_bestestimator, list_bestparam, list_bestscore


# In[ ]:


# run the grids for the models defined above
model_results = [] ; model_bestestimator = [] ; model_bestparam = [] ; model_bestscore = []
model_results, model_bestestimator, model_bestparam, model_bestscore =     rungrid(model_models, model_params, X_train, y_train)


# #### Create a table summarising the performances of all models

# In[ ]:


# compute the predictions for models defined above and create summary table
MLA_columns = ['Model', 'Parameters','Training Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy 3*STD' , 'Validation Accuracy', 'Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
for im, model2 in enumerate(model_models):
    namemod = model2[0]
    model = model_bestestimator[im]
    #
    titanic_test['Survived_pred('+namemod+')'] = (model.predict(X_test))
    testcsv = titanic_test.copy()
    testcsv['Survived'] = testcsv['Survived_pred('+namemod+')']
    testcsv[['PassengerId', 'Survived']].set_index('PassengerId').to_csv('predictions_'+str(namemod)+'.csv')
    #
    pred_valid = model.predict(X_valid)
    score_valid = accuracy_score(y_valid, pred_valid)
    #
    results = model_results[im].sort_values(by='mean_test_score', axis=0, ascending=False)#['params'].tolist()[0]
    MLA_compare.loc[im,'Model'] = namemod
    MLA_compare.loc[im,'Parameters'] = str(model_bestparam[im])
    MLA_compare.loc[im,'Training Accuracy Mean'] = float(results['mean_train_score'].tolist()[0])
    MLA_compare.loc[im,'Test Accuracy Mean'] = results['mean_test_score'].tolist()[0]
    MLA_compare.loc[im,'Test Accuracy 3*STD'] = 3.*results['std_train_score'].tolist()[0]
    MLA_compare.loc[im,'Validation Accuracy'] = score_valid
    MLA_compare.loc[im,'Time'] = results['mean_fit_time'].tolist()[0]


# In[ ]:


MLA_compare.sort_values(by='Validation Accuracy',ascending=False)


# #### Plot the model scores as function of the hyper-parameter values.

# In[ ]:


# plot the score of the models as function of hyperparameter
def plot_score_1d(plotname, plottitle, list_results):
    plt.figure(0,figsize=[15,10])
    plt.subplots_adjust(wspace=0.35, hspace=0.5)
    for idf, df in enumerate(list_results):
        Nparams = len(df.params[0])
        param_name = list(df.params[0].keys())
        #
        plt.subplot(2,int(len(list_results)/2)+1,idf+1)
        plt.title(plottitle[idf]) #clf_name[im])
        #plt.ylim(0,0.06)
        plt.xscale('log')
        plt.ylabel('Score')
        if Nparams == 1:
            plt.xlabel(param_name[0]) #[x for x in model_param.keys()][0])
            param_value = ([([x for x in df['params'][i].values()][0]) for i in range(len(df['params']))])
            plt.plot(param_value, df['mean_train_score'],ls='-',color='b',label='Train cont')
            plt.plot(param_value, df['mean_test_score'],ls='-',color='r',label='Test cont')
        elif Nparams == 2:
            plt.xlabel(param_name[0]) #[x for x in model_param.keys()][0])
            list_param2 = sorted(set([row.params[param_name[1]] for ir, row in df.iterrows()])) 
            for ip2, param2 in enumerate(list_param2):
                list_param1 = sorted(set([row.params[param_name[0]] for ir, row in df.iterrows()])) 
                mask = [row.params[param_name[1]] == param2 for ir, row in df.iterrows()]
                plt.plot(list_param1, df[mask]['mean_train_score'],ls='-',color='b',label='Train cont '+str(param2))
                plt.plot(list_param1, df[mask]['mean_test_score'],ls='-',color='r',label='Test cont '+str(param2))
        #plt.legend()
    plt.show()
    #plt.savefig(plotname+'.png',bbox_inches='tight',transparent=True)
    plt.close(0)


# In[ ]:


# plot the scores for the computed models
list_models = list(zip([model[0] for model in model_models], model_bestestimator)) ; model_score = []
plot_score_1d('score_classifiers', [i[0] for i in list_models], model_results)


# #### Plot the feature importances derived by the models.

# In[ ]:


# plot the feature importances derived by the best estimators of each model
list_models = list(zip([model[0] for model in model_models], model_bestestimator)) ; model_score = []
list_usedfeat = titanic_train_encoded.drop(list_drop, axis=1).drop('Survived',axis=1).columns.tolist()
#
plt.figure(0,figsize=[15,25])
plt.subplots_adjust(hspace=0.2)
for im, bestest in enumerate(model_bestestimator):
    plt.subplot(len(model_bestestimator),1,im+1)
    importances = pd.DataFrame(bestest.feature_importances_, index=list_usedfeat, columns=['Importance']).sort_values(by='Importance',ascending=False)
    sns.barplot(x='Importance', y=importances.index, data=importances)
    plt.title(model_models[im][0])
plt.show()
#plt.savefig('feature_importance.png',bbox_inches='tight',transparent=True)
plt.close(0)


# #### Give predictions on the test set by ensembling the results of the different used classifiers with a "soft" voting. 

# In[ ]:


# create voting classifier 
namemod = 'voting'
list_models = list(zip([model[0] for model in model_models], model_bestestimator)) ; model_score = []
#
vc = VotingClassifier(estimators=list_models, voting='soft', n_jobs=-1)
vc.fit(X_train, y_train) 
#
titanic_test['Survived_pred('+namemod+')'] = (vc.predict(X_test))
#
testcsv = titanic_test.copy()
testcsv['Survived'] = testcsv['Survived_pred('+namemod+')']
testcsv[['PassengerId', 'Survived']].set_index('PassengerId').to_csv('predictions_'+str(namemod)+'.csv')
#
MLA_compare.loc[im+1,'Model'] = namemod
#
y_valid_pred = vc.predict(X_valid)
score_valid = accuracy_score(y_valid, y_valid_pred)
MLA_compare.loc[im+1,'Validation Accuracy'] = score_valid


# In[ ]:


# compare predictions from voting
MLA_compare.sort_values(by='Validation Accuracy',ascending=False)

