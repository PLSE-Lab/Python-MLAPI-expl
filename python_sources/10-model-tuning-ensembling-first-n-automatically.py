### This script is going to tune 10 models and ensemmble 1 to n number 
### of them, among the best n models according to their cross validation scores:
### It is the prodcution level script version  of my notebook: https://www.kaggle.com/berkanacar/full-ml-work-flow-tuning-10-models-and-ensembling


## Importing Libraries
# data analysis libraries:
import numpy as np
import pandas as pd

# to ignore warnings:
import sys
if not sys.warnoptions:
    import os, warnings
    warnings.simplefilter("ignore") 
    os.environ["PYTHONWARNINGS"] = "ignore" 
# to display all columns:
pd.set_option('display.max_columns', None)

#timer
import time
from contextlib import contextmanager

# Importing modelling libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} done in {:.0f}s".format(title, time.time() - t0))


## r is the random state number for having the same results in case of replications.
r=1309
## Models and their string names to be used
models = [LogisticRegression(random_state=r),GaussianNB(), KNeighborsClassifier(),
            SVC(random_state=r,probability=True),DecisionTreeClassifier(random_state=r),
            RandomForestClassifier(random_state=r), GradientBoostingClassifier(random_state=r),
            XGBClassifier(random_state=r), MLPClassifier(random_state=r),
            CatBoostClassifier(random_state=r,verbose = False)]
names = ["LogisticRegression","GaussianNB","KNN","SVC",
         "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]
#Data Preparation
def data_preprocessing():
    print("\nDATA PREPROCESSING PROCESS HAS BEEN STARTED" "\n")
    ## Loading Data
    train = pd.read_csv("../input/titanic/train.csv")
    test = pd.read_csv("../input/titanic/test.csv")
    print('Train and test data is read from kaggle input file''\n\n')

    train.drop(['Ticket','Embarked'], axis = 1,inplace=True)   
    test.drop(['Ticket','Embarked'], axis = 1,inplace=True)
    
    #Outlier treatment
    #Defining the upper limit as 99% of all data for winsoring its above
    full_data=pd.concat([train, test], ignore_index=True)
    upper_limit = full_data['Fare'].quantile(0.99)
    print('Outlier treatment: \n')
    print('Repress the Fare variable at maximum to %99 value:','%.2f'% upper_limit )
    print('\n')
    for d in [train,test,full_data]:
        d.loc[d['Fare'] > upper_limit,'Fare'] = upper_limit

    #Missing Value Treatment
    print('Missing value treatment: ''\n')
    print('Number of missing values and their percentage for Train and Test samples respectively', end = "\n\n")
    for df in [train,test]:
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        print(pd.concat([total,percent], axis=1, keys=['Total','Percent']), end = "\n\n")
    
    print('Set the median Fare of each passenger class for the missing Fare values.''\n')
    test["Fare"].fillna(full_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    
    print("\nDATA PREPROCESSING PROCESS HAS BEEN FINISHED \n")
    
    return train, test

def feature_engineering(train, test):
    full_data=pd.concat([train, test], ignore_index=True)
    print("\nFEATURE ENGINEERING PROCESS HAS BEEN STARTED \n")
    #Extraction of Title and Nicknamed fron Name
    for d in [train,test,full_data]:
        d['Nicknamed']=d['Name'].apply(lambda x: 1 if '''"''' in x else 0)
        d["Title"] = d["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    print('New variables are created using Name variable: Title and Nicknamed.''\n')
          
    #Clustering Title variable    
    for d in [train,test,full_data]:
        d['Title'] = d['Title'].replace(['Lady','Mme','Mlle','Don','Col', 'Major', 'Dona','Countess', 'Sir'], 'Noble')
        d['Title'] = d['Title'].replace('Ms', 'Mrs')
        d['Title'] = d['Title'].replace(['Capt','Jonkheer','Rev'], 'Mr')
        d.loc[d['Sex']=='female', 'Title']=d.loc[d['Sex']=='female', 'Title'].replace('Dr', 'Mrs')
        d.loc[d['Sex']=='male', 'Title']=d.loc[d['Sex']=='male', 'Title'].replace('Dr', 'Mr')
        d.drop(['Name'], axis = 1,inplace=True)
    print('Rare titles are clustered in the new Rare category and Name variable is dropped.''\n')

    print('Cabin_dummy variable is created using Cabin variable.''\n')
    print('Cabin variable is dropped')
    for d in [train,test]:
        d["Cabin_dummy"] = d["Cabin"].notnull().astype('int')
        d.drop(['Cabin'], axis = 1, inplace=True)
    
    #Generating small_family, dropping family size and sex   
    for d in [train,test]:
        d["Familysize"] = d["SibSp"] + d["Parch"] + 1
    
    for d in [train,test]:
        d["small_family"] = [1 if 1<i < 5 else 0 for i in d["Familysize"] ]
        d.drop(["Familysize"], axis = 1, inplace=True)
        d.drop(["Sex"], axis = 1, inplace=True)
    
    #Categorical Variables' One hot encoding
    print('\nOne hot encoding for the categorical variables is done.''\n')    
    train, test= [ pd.get_dummies(data, columns = ['Title','Pclass']) for data in [train, test]]
    
    print('Nedian ages of each title are set for the missing Age values''\n')
    train["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)
    test["Age"].fillna(full_data.groupby("Title")["Age"].transform("median"), inplace=True)
    
    print("FEATURE ENGINEERING PROCESS HAS BEEN FINISHED \n")   
    return train, test


# Modeling, Evaluation and Model Tuning
def modelling(train):
    print("\n\nMODELLING PROCESS HAS BEEN STARTED \n")
    ##inner train and validation set split of the given train set
    predictors = train.drop(['Survived', 'PassengerId'], axis=1)
    target = train["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
    
    print('Default model validation accuracies for the train data:', end = "\n\n")
    for name, model in zip(names, models):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val) 
        print(name,':',"%.3f" % accuracy_score(y_pred, y_val))
    
    results = []
    print('\n10 fold Cross validation accuracy and std of the default models for the train data:', end = "\n\n")
    for name, model in zip(names, models):
        kfold = KFold(n_splits=10, random_state=1001)
        cv_results = cross_val_score(model, predictors, target, cv = kfold, scoring = "accuracy")
        results.append(cv_results)
        print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))
    
    print("\nMODEL TUNING STARTED\n")   
   # Possible hyper parameters for Model Tuning
    logreg_params= {"C":np.logspace(-1, 1, 10),
                        "penalty": ["l1","l2"], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], "max_iter":[1000]}

    NB_params = {'var_smoothing': np.logspace(0,-9, num=100)}
    knn_params= {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                     "weights": ["uniform","distance"],
                     "metric":["euclidean","manhattan"]}
    svc_params= {"kernel" : ["rbf"],
                     "gamma": [0.001, 0.01, 0.1, 1, 5, 10 ,50 ,100],
                     "C": [1,10,50,100,200,300,1000]}
    dtree_params = {"min_samples_split" : range(10,500,20),
                    "max_depth": range(1,20,2)}
    rf_params = {"max_features": ["log2","Auto","None"],
                    "min_samples_split":[2,3,5],
                    "min_samples_leaf":[1,3,5],
                    "bootstrap":[True,False],
                    "n_estimators":[50,100,150],
                    "criterion":["gini","entropy"]}
    gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
                 "n_estimators": [100,500,100],
                 "max_depth": [3,5,10],
                 "min_samples_split": [2,5,10]}
    gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
                 "n_estimators": [100,500,100],
                 "max_depth": [3,5,10],
                 "min_samples_split": [2,5,10]}

    xgb_params ={
            'n_estimators': [50, 100, 200],
            'subsample': [ 0.6, 0.8, 1.0],
            'max_depth': [1,2,3,4],
            'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5],
            "min_samples_split": [1,2,4,6]}

    mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
                  "hidden_layer_sizes": [(10,10,10),
                                         (100,100,100),
                                         (100,100),
                                         (3,5), 
                                         (5, 3)],
                  "solver" : ["lbfgs","adam","sgd"],
                  "activation": ["relu","logistic"]}
    catb_params =  {'depth':[2, 3, 4],
                  'loss_function': ['Logloss', 'CrossEntropy'],
                  'l2_leaf_reg':np.arange(2,31)}
    classifier_params = [logreg_params,NB_params,knn_params,svc_params,dtree_params,rf_params,
                         gbm_params, xgb_params,mlpc_params,catb_params]       
    
    # Tuning by Cross Validation  
    cv_result = {}
    best_estimators = {}
    for name, model,classifier_param in zip(names, models,classifier_params):
        with timer(">model tuning"):
            clf = GridSearchCV(model, param_grid=classifier_param, cv =10, scoring = "accuracy", n_jobs = -1,verbose = False)
            clf.fit(x_train,y_train)
            cv_result[name]=clf.best_score_
            best_estimators[name]=clf.best_estimator_
            print(name,'cross validation accuracy : %.3f'%cv_result[name])
    accuracies={}
    print('Validation accuracies of the tuned models for the train data:', end = "\n\n")
    for name, model_tuned in zip(best_estimators.keys(),best_estimators.values()):
        y_pred =  model_tuned.fit(x_train,y_train).predict(x_val)
        accuracy=accuracy_score(y_pred, y_val)
        print(name,':', "%.3f" %accuracy)
        accuracies[name]=accuracy
    print("\n MODELLING PROCESS HAS BEEN FINISHED \n")
    return accuracies,best_estimators,x_train, x_val, y_train, y_val 

    #Ensemble first n (e.g. 6) models
def ensembling(accuracies,best_estimators,x_train, x_val, y_train, y_val,i ):   
    accu=sorted(accuracies, reverse=True, key= lambda k:accuracies[k])[:i]
    firstn=[[k,v] for k,v in best_estimators.items() if k in accu]

    votingC = VotingClassifier(estimators = firstn, voting = "soft", n_jobs = -1)
    votingC = votingC.fit(x_train, y_train)
    print(accuracy_score(votingC.predict(x_val),y_val))
    return votingC
    
    
def submission(votingC, test,j):    
    ids = test['PassengerId']
    x_test=test.drop('PassengerId', axis=1)
    predictions = votingC.predict(x_test)

    #set the output as a dataframe and convert to csv file named submission.csv
    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
    output.to_csv('first{}.csv'.format(j), index=False)
    print("Submission file has been created: ",'first{}.csv'.format(j))

def main():
   
    n=6
    print('This model is going to tune and ensemmble maximum',n,
          'models among the following models according to their cross validation scores:' '\n' )
    j=1
    for i in names:
        print(j,'-',i)
        j+=1
        
    with timer("Data Preprocessing"):
        train, test = data_preprocessing()
    
    with timer("Feature Engineering"):
        train, test = feature_engineering(train, test)
     
    with timer("ModelLing"):
        accuracies,best_estimators,x_train, x_val, y_train, y_val=modelling(train)
        
    ##n results by ensembling 1 to n models  
    print("\n\n ENSEMBLING AND SUBMISSION PROCESSES HAVE BEEN STARTED \n")
    with timer("Ensembling and submission"):
        for i in np.arange(1,n+1)[::-1]:
            votingC=ensembling(accuracies,best_estimators,x_train, x_val, y_train, y_val,i)         
            submission(votingC, test,i)
        
if __name__ == "__main__":
    with timer("Full model run"):
        main()
