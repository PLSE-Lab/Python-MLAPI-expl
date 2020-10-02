# =============================================================================
# The purpose of this script is to compare catboost and xgboost models with 
# defualt parameters on the titanic dataset. The xgboost model had slightly 
# better accuracy and dramatically faster training time.
# =============================================================================

################################################################################
# import packages

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import catboost as cb
import xgboost as xgb
import time

################################################################################
# Read in data and combine them

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['set'] = 'train'
test['set'] = 'test'

train_test = pd.concat([train, test], sort = False)

################################################################################
# Handle missing values

train_test.Embarked.fillna('C', inplace = True)

med_fare = train[(train.Pclass == 3) & (train.Embarked == 'S')].Fare.median()

train_test['Fare'].fillna(med_fare, inplace = True)

################################################################################
# Feature engineering

train_test['Deck'] = train_test.Cabin.str[0]
train_test.Deck.fillna('U', inplace = True)

train_test['FamSize'] = train_test.SibSp + train_test.Parch + 1

train_test['NameLength'] = train_test['Name'].apply(lambda x: len(x))

train_test['Title'] = train_test.Name.str.extract('([A-Za-z]+)\.')
rare_title = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 
              'Rev', 'Sir', 'Jonkheer', 'Dona']
train_test.Title = train_test.Title.replace(rare_title, 'Rare')
train_test.Title = train_test.Title.replace({'Mlle': 'Miss',
                                             'Ms': 'Miss',
                                             'Mme': 'Mrs'})

bins = [0, 13, 19, 25, 35, 50, 81]
bin_names = ['child', 'teen', 'young adult', 'adult', 'mid age', 'sr']
age_cats = pd.cut(train_test.Age, bins, labels = bin_names)
train_test['age_cats'] = age_cats.astype('object').fillna('U')

train_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 
                'Age'], axis = 1, inplace = True)

################################################################################
# One hot encode categorical variables and split to train and test set

cat_var = ['Sex', 'Embarked', 'Deck', 'Title', 'age_cats']
dummy = pd.get_dummies(train_test[cat_var])
train_test2 = pd.concat([train_test, dummy], axis = 1).drop(cat_var, axis = 1)

train2 = train_test2[train_test2.set == 'train'].drop('set', axis = 1)
test2 = train_test2[train_test2.set == 'test'].drop('set', axis = 1)

X = train2.drop(['Survived'], axis = 1)
Y = train2.Survived

################################################################################
# Catboost model CV accuracy and run time

kfold = KFold(n_splits = 10, random_state = 8)

start = time.time()

cat_mod = cb.CatBoostClassifier(eval_metric='Accuracy', iterations = 100, 
                                random_seed = 88)

results_cb = cross_val_score(cat_mod, X, Y, cv = kfold)

results_cb.mean() # Accuracy: 0.8305

print(time.time() - start) #  Run time: 9.4249 seconds                

################################################################################
# xgboost model CV accuracy and run time

start = time.time()

xgb_mod = xgb.XGBClassifier()

results_xgb = cross_val_score(xgb_mod, X, Y, cv = kfold)

results_xgb.mean() # Accuracy: 0.8316

end = time.time()
print(end - start) # Run time: 0.9265 seconds
