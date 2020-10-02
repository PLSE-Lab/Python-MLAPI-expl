# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd

import csv as csv

import matplotlib.pyplot as plt

###### FONCTIONS #######


def MultiClasse(data,variable,classe1,val1,classe2,val2,classe3,val3,
                classe4,val4,classe5,val5,classe6,val6,classe7,val7,
                classe8,val8,classe9,val9,classe10,val10):
                    
    i_max=data.shape[0]
    for i in range(0,i_max):
        if data[variable].values[i]==classe1:
            data[variable].values[i]=val1
        elif data[variable].values[i]==classe2:
            data[variable].values[i]=val2
        elif  data[variable].values[i]==classe3:
            data[variable].values[i]=val3
        elif data[variable].values[i]==classe4:
            data[variable].values[i]=val4
        elif  data[variable].values[i]==classe5:
            data[variable].values[i]=val5
        elif  data[variable].values[i]==classe6:
            data[variable].values[i]==val6
        elif  data[variable].values[i]==classe7:
            data[variable].values[i]=val7
        elif  data[variable].values[i]==classe8:
            data[variable].values[i]=val8
        elif  data[variable].values[i]==classe9:
            data[variable].values[i]=val9
        elif data[variable].values[i]==classe10:
            data[variable].values[i]=val10

###### PREPARATION DES DONNÉES #######

main_df=pd.read_csv("../input/Iris.csv")

#### Transformer les 'Species' en 3 classes #### 
#### Iris-setosa=1 , Iris-versicolor=2 , Iris-verginica=3 ####

MultiClasse(main_df,'Species','Iris-setosa',1,'Iris-versicolor',2,
                        'Iris-virginica',3,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                        
#print(main_df.head())

#### Definir les variables predictives et cibles #### 

X=main_df.drop(['Species','Id'],axis=1)
y=main_df['Species']

#### Separation en test/train ####

from sklearn import cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,
                              test_size=0.35)


X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

y_train=np.array(y_train,dtype=np.float64)
y_test=np.array(y_test,dtype=np.float64)


####### PREPROCESSING #######

#### Centrer reduire les données ####


from sklearn.preprocessing import scale

X_train_np=scale(X_train)
X_test_np=scale(X_test)

X_train=pd.DataFrame(X_train_np)
X_test=pd.DataFrame(X_test_np)


####### ALGORITHME #######


from sklearn import svm

clf=svm.SVC()

clf.fit(X_train,y_train)


#### Grid search ####


from sklearn.grid_search import GridSearchCV

param_grid=[{'kernel':['rbf'],'gamma':[0.1,0.5,1],'C':[0.1,0.5,1,10]},
            {'kernel':['poly'],'gamma':[0.1,0.5,1],'C':[0.1,0.5,1,10],'degree':[2,3]},
            {'kernel':['linear'],'C':[0.1,0.5,1,10]},
            {'kernel':['sigmoid'],'C':[0.1,0.5,1,10]}]
            

grid=GridSearchCV(clf, param_grid)

grid.fit(X_train,y_train)

BestPara=grid.best_params_

print("Best Para = " , BestPara)

### Validation croisée ###

from sklearn.cross_validation import cross_val_score

scores=cross_val_score(clf,X_train,y_train,cv=3,scoring='accuracy')

print("moyenne des scores CV = %f" %scores.mean())

### Prediction ###

prediction=clf.predict(X_test)

size=prediction.shape[0]
print("taille de y_prediction = %i" %size)

from sklearn.metrics import accuracy_score 

a=accuracy_score(y_test,prediction)

print("Score final = %f" %a)
