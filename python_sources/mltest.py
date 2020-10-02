import pandas as pd
import numpy as np
import collections, re
import copy


from cycler import cycler
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%matplotlib inline  

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.grid_search import GridSearchCV

pd.set_option('display.max_columns', 500)


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.info()
test.info()

max_emb = np.argmax(train['Embarked'].value_counts())
train['Embarked'].fillna(max_emb, inplace=True)

indz = test['Fare'].index[test['Fare'].apply(np.isnan)].tolist
print(indz)
pclass = test['Pclass'][152]
fare_test = test[test['Pclass']==pclass].Fare.dropna()
fare_train = train[train['Pclass']==pclass].Fare
fare_med = (fare_test + fare_train).median()
print(fare_med)
test.loc[152,'Fare'] = fare_med

ages = np.concatenate((test['Age'].dropna(), train['Age'].dropna()), axis=0)
std_ages = ages.std()
mean_ages = ages.mean()
train_nas = np.isnan(train["Age"])
test_nas = np.isnan(test["Age"])
np.random.seed(122)
impute_age_train  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = train_nas.sum())
impute_age_test  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = test_nas.sum())
train["Age"][train_nas] = impute_age_train
test["Age"][test_nas] = impute_age_test
ages_imputed = np.concatenate((test["Age"],train["Age"]), axis = 0)

train['Age*Class'] = train['Age']*train['Pclass']
test['Age*Class'] = test['Age']*test['Pclass']

ax = pd.DataFrame({'age':ages}).plot(kind = 'kde')
pd.DataFrame({'age':ages_imputed}).plot(kind = 'kde', ax=ax)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)  

test['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = 'True')
test['Title'].replace(['Countess','Dona','Lady'], 'Mrs', inplace = 'True')
#test['Title'].replace(['Master'], 'Mr', inplace = 'True')
test['Title'].replace(['Mme'], 'Mrs', inplace = 'True')
test['Title'].replace(['Mlle','Ms'], 'Miss', inplace = 'True')
    
train['age_cat'] = None
train.loc[(train['Age'] <= 13), 'age_cat'] = 'young'
train.loc[ (train['Age'] > 13), 'age_cat'] = 'adult'

train_label = train['Survived']
test_pasId = test['PassengerId']
drop_cols = ['Name','Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId']
train.drop(drop_cols + ['Cabin'], 1, inplace = True)
test.drop(drop_cols, 1, inplace = True)

train['Pclass'] = train['Pclass'].apply(str)
test['Pclass'] = test['Pclass'].apply(str)

train.drop(['Survived'], 1, inplace = True)
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0)
dataset = pd.get_dummies(dataset)
train = copy.copy(dataset[:train_objs_num])
test = copy.copy(dataset[train_objs_num:])

droppings = ['Embarked_Q', 'Age']
#droppings += ['Sex_male', 'Sex_female']

test.drop(droppings, 1, inplace = True)
train.drop(droppings,1, inplace = True)

train.head(5)

def prediction(model, train, label, test, test_pasId):
    model.fit(train, label)
    pred = model.predict(test)
    accuracy = cross_val_score(model, train, label, cv = 5)

    sub = pd.DataFrame({
            "PassengerId": test_pasId,
            "Survived": pred
        })    
    return [model, accuracy, sub]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train['Fare'].values.reshape(-1, 1))
train['Fare'] = scaler.transform(train['Fare'].values.reshape(-1, 1)) 
test['Fare'] = scaler.transform(test['Fare'].values.reshape(-1, 1))  

scaler = StandardScaler().fit(train['Age*Class'].values.reshape(-1, 1))
train['Age*Class'] = scaler.transform(train['Age*Class'].values.reshape(-1, 1)) 
test['Age*Class'] = scaler.transform(test['Age*Class'].values.reshape(-1, 1))  



lr  = LogisticRegression(random_state=110)
acc = prediction(lr, train, train_label, test, test_pasId)
print(acc[1])

test_predictions = acc[0].predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })
submission.to_csv("titanic_submission_logregres.csv", index=False)

#train.columns.tolist()
print(list(zip(acc[0].coef_[0], train.columns.tolist())))