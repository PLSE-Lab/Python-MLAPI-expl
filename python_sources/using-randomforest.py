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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


df = pd.read_csv('../input/train.csv',index_col=None) #load the training set
df1 = pd.read_csv('../input/test.csv',index_col=None)
#df1=df1.drop('Survived',axis=1)
df["Age"].fillna(df["Age"].median(), inplace=True) #fill missing age with mean values
df1["Age"].fillna(df1["Age"].median(), inplace=True)
#to improve speed of training the model remove unnecessary data
df["HasCabin"]=df["Cabin"]
df.HasCabin.loc[df.Cabin.notnull()]=1
df.HasCabin.loc[df.Cabin.isnull()]=0
df.Sex=df.Sex.astype('category').cat.codes #male=1 female=0
df.Embarked=df.Embarked.astype('category').cat.codes #s=2 c=0 q=1


df1["HasCabin"]=df1["Cabin"]
df1.HasCabin.loc[df1.Cabin.notnull()]=1
df1.HasCabin.loc[df1.Cabin.isnull()]=0
df1.Sex=df1.Sex.astype('category').cat.codes #male=1 female=0
df1.Embarked=df1.Embarked.astype('category').cat.codes #s=2 c=0 q=1

#drop these columns they do not help in prediction
dropcolumns = ['Ticket','Name','Cabin','Fare']
df.drop(dropcolumns,axis=1,inplace=True)
df1.drop(dropcolumns,axis=1,inplace=True)

#drop empty rows
df = df.dropna(axis=0)
x = df
x = x.drop('Survived', axis=1)

y = df.Survived
predictors = ["SibSp", "Parch","Pclass","HasCabin","Embarked", "Age", "Sex",]
LR= LogisticRegression()
LR.fit(x,y)
Y_pred1=LR.predict(df1)
print(LR.score(x,y))

random_forest = RandomForestClassifier(n_estimators= 10, max_features='log2', min_samples_leaf= 3, max_depth= 4, min_samples_split= 10, bootstrap=True)
random_forest.fit(x,y)
Y_pred = random_forest.predict(df1)
random_forest.score(x,y)
acc_random_forest = round(random_forest.score(x,y) * 100, 2)
print(acc_random_forest)


# parameter_grid = {
#                  'max_depth' : [4, 6, 8],
#                  'n_estimators': [50, 10,100,150,75],
#                  'max_features': ['sqrt', 'auto', 'log2'],
#                  'min_samples_split': [1.0, 3, 10],
#                  'min_samples_leaf': [1, 3, 10],
#                  'bootstrap': [True, False],
#                  }
#
# cross_validation = StratifiedKFold(y, n_folds=5)
#
# grid_search = GridSearchCV(random_forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(x,y)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
#
# print("finished")
pred_df = pd.DataFrame({
        "PassengerId": df1["PassengerId"],
        "Survived": Y_pred
    })
pred_df.to_csv('kaggle-titanic-competition.csv', index=False)