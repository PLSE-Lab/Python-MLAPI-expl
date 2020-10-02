# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score, make_scorer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print(df_train.head(n=4))

#No survivers submission
#df_test['Survived']=0
#df_test[['PassengerId','Survived']].to_csv('no_survivors.csv',index=False)

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1,0,5,12,18,25,35,60,120)
    group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
    categories = pd.cut(df.Age,bins,labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1,0,8,15,31,1000)
    group_names = ['Unknown','1_quartile','2_quartile','3_quartile','4_quartile']
    categories = pd.cut(df.Fare,bins,labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname']=df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket','Name','Embarked'],axis=1)
    
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

df_train = transform_features(df_train)
df_test = transform_features(df_test)
print(df_train.head())

def encode_features(df_train,df_test):
    features = ['Fare', 'Cabin','Age','Sex','Lname','NamePrefix']
    df_combined = pd.concat([df_train[features],df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

df_train, df_test = encode_features(df_train,df_test)
print(df_train.head())

X_all = df_train.drop(['Survived','PassengerId'],axis=1)
y_all = df_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,test_size=num_test,random_state=23)

acc_scorer = make_scorer(accuracy_score)

print(X_all.shape)

seed=42
#encoder = LabelEncoder()
#encoder.fit(y_all)
#y_tr = encoder.transform(y_train)
#y_te = encoder.transform(y_test)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal',activation='relu'))
    model.add(Dense(10,kernel_initializer='normal',activation = 'sigmoid'))
    model.add(Dense(15,kernel_initializer='normal',activation = 'relu'))
    model.add(Dense(15,kernel_initializer='normal',activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

pipeline.fit(X_all,y_all)

X_testing = df_test.drop(['PassengerId'],axis=1)
predictions = pipeline.predict(np.array(X_testing))

ids = df_test['PassengerId']
predictions = pd.DataFrame(predictions).iloc[:,0]
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions})
output.to_csv('titanic-predictions.csv',index=False)
print(output.head())