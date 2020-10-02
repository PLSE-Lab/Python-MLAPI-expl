import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

def map_unique(df1,df2, cols):
    #return maped df1,df2 where unique values are maped to int 
    for c in cols:
        unique = set(df1[c].unique()).union(set(df2[c].unique()))
        unique_dict = {key: value for key, value in zip(unique, range(len(unique)))}
        df1[c] = df1[c].map(unique_dict).astype('int')
        df2[c] = df2[c].map(unique_dict).astype('int')
    return df1, df2
        
train = pd.read_csv('../input/titanic/train.csv')
train.drop(columns=['Name', 'PassengerId'], inplace=True)
test = pd.read_csv('../input/titanic/test.csv')
test.drop(columns=['Name', 'PassengerId'], inplace=True)
submission = pd.read_csv('../input/titanic/gender_submission.csv')

train, test = map_unique(train, test, ['Sex', 'Ticket', 'Cabin', 'Embarked'])

target = train['Survived']
train.drop(columns=['Survived'], inplace =True)

fill_dict = train.mean().to_dict()
train = train.fillna(fill_dict)
test = test.fillna(fill_dict)

pol = PolynomialFeatures(degree = 3, include_bias= False)
train = pol.fit_transform(train)
test = pol.fit_transform(test)

scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

cv1  = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2020)

prediction = np.zeros((len(test), 2))
y_pred = np.zeros(len(train))
fold =1
for train_ind, test_ind in cv1.split(train, target):
    x_train, x_val = train[train_ind], train[test_ind]
    y_train, y_val = target[train_ind], target[test_ind]
    
    reg = XGBClassifier().fit(x_train, y_train)
    y_pred[test_ind] = reg.predict(x_val)
    print('Accuracy %.2f in fold %i' % (reg.score(x_val, y_val), fold))
    fold += 1
    prediction += reg.predict_proba(test)
    
print('Out-of-sample accuracy: %.3f' % accuracy_score(target, y_pred))
print('confusion matrix')
print(confusion_matrix(target, y_pred)/len(train))

submission['Survived'] = prediction.argmax(axis=1)
submission.to_csv('submission.csv', index=False)