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


#Project : Titanic: Machine Learning from Disaster
#Data : https://www.kaggle.com/c/titanic/data



import pandas as pd
import numpy as np
import timeit
import os
from datetime import datetime
from statistics import mean 
from statistics import stdev
from sklearn.svm import LinearSVC
from functools import reduce
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
DATA_PATH='/kaggle/input/titanic/'
DISABLE_SCALE=True
MISSING_AGE=29.12345
TITANIC_DROP=['Embarked','Ticket','Fare','Name','SibSp','Cabin']
def load_any_csv(fl_name,nth_sample=1):
    full_path=os.path.join(DATA_PATH,fl_name+'.csv')
    return pd.read_csv(full_path,skiprows=lambda i: i % nth_sample != 0)
def cleanup_titanic_data(data):
    def wt_sum(x1, x2): return x1*1000.0 + x2*1.0
    # index data
    data.set_index('PassengerId',inplace=True)
    # text data    
    data['Cabin']=data['Cabin'].fillna(0).map(lambda x: x if (x == 0) else 1)
    data.Sex=data.Sex.map(lambda x:0 if(x=='female') else 1)
    data['Name']=data['Name'].map(lambda x: x.split(',')[0])
    data['Name']=data['Name'].map(lambda x:  reduce(wt_sum,(ord(ch) for ch in x)))
    # missing data 
    data.loc[(data.Age.isnull()) & (data.Parch>0) ,['Age']]=9
    data.loc[(data.Age.isnull()) ,['Age']]=data['Age'].mean()
    return data
def make_x_y_sets_any(t_set,label_field):
    x_t=t_set.drop(columns=[label_field]);
    y_t=t_set[label_field];
    return x_t,y_t
def scale_any_train_set(data):
    if DISABLE_SCALE:
        return data    
    scaler = StandardScaler()
    cols=data.columns
    rt=pd.DataFrame(scaler.fit_transform(data),columns=cols)
    return rt
def remove_any_unrelated(set,fields):
    w= set.copy()
    w.drop(columns=fields,inplace=True)
    return w
def test_titanic_model(clf,X_live,out_csv):
    X_live=scale_any_train_set(X_live)
    y_prob = clf.predict(X_live)
    dindex = simple_set_test.index.values
    df=pd.concat([pd.Series(dindex),pd.Series(y_prob)], axis=1)
    df.columns = ['PassengerId','Survived']
    result_csv_path=os.path.join(DATA_PATH,out_csv)
    #df.to_csv(index=False,path_or_buf=result_csv_path)
    df.head(4)
def any_clf_gini(): 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
    return clf_gini  
def any_clf_entropy(): 
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
    return clf_entropy 
def any_clf_Linear_svc(): 
    # LinearSVC 
    clf_linearSVC=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    return clf_linearSVC 
def any_clf_random_forest(): 
    return RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
import sklearn
print(sklearn.__version__)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}
def any_grid_search(x,y,num_validation=20):
    rfc=any_clf_random_forest()
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= num_validation)
    CV_rfc.fit(x,y)
    return CV_rfc.best_estimator_
def now():
    return datetime.now().strftime("%b %d %Y %H:%M:%S")
def pprint(a,b):
    if(type(b) is str):
        print("%s %-23s%8s" %(now(),a,b))  
    else:
        print("%s %-23s%8.0f%%" %(now(),a,b))
        return a+'_'+str(int(b))+"_submission.csv"
def any_clf_knn(): 
    return KNeighborsClassifier(n_neighbors=3)
def get_whole_XY(train,drop_cols,label_field):
    simple_set=remove_any_unrelated(cleanup_titanic_data(load_any_csv(train)),drop_cols)
    simple_set=simple_set[simple_set.Age!=MISSING_AGE]
    x_t,y_t=make_x_y_sets_any(simple_set,label_field)
    return simple_set,scale_any_train_set(x_t),y_t
def get_whole_X(test,drop_cols):
    simple_set=remove_any_unrelated(cleanup_titanic_data(load_any_csv(test)),drop_cols)
    return simple_set,scale_any_train_set(simple_set)
def accuracy(classifier, X, y,label_field, num_validations=20):
    learn=simple_set.copy()
    classifier.fit(X,y)
    accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_validations)
    perc=accuracy.mean()*100
    y_prob = classifier.predict(X)
    learn['pred']=y_prob
    return pprint(str(idx)+'_'+name_model[idx],perc),learn[learn[label_field]>learn.pred],learn[learn[label_field]<learn.pred]
name_model=['DecisionTree-gini','Linear_svc','random_forest','K-Nearest_Neighbor','DecisionTree-entropy','grid_search_csv']
many_models=[any_clf_gini(),any_clf_Linear_svc(),any_clf_random_forest(),any_clf_knn(),any_clf_entropy()]
#----Ready-----------------
print(now(),'comienzo')    
simple_set,trainX,trainY=get_whole_XY('train',TITANIC_DROP,'Survived')
simple_set_test,X_live=get_whole_X('test',TITANIC_DROP)
print(now(),'data loaded')    
many_models.append(any_grid_search(trainX,trainY))
print(now(),'model created')    
#----Go-----------------
for idx in range(len(many_models)): 
    out_csv,wrong_guess_as_dead,wrong_guess_as_alive=accuracy(many_models[idx],trainX,trainY,'Survived') 
    test_titanic_model(many_models[idx],X_live,out_csv)  
print(now(),'terminar')    
print(wrong_guess_as_dead.shape)
print(wrong_guess_as_alive.shape)
wrong_guess_as_dead
wrong_guess_as_alive