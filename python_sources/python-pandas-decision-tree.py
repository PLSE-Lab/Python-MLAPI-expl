import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(pandas_df, extract_Y = True):
     # ['SibSp', 'Parch', 'Ticket', 'Name', 'Cabin'] not used columns

    gender_columns =  (pd.get_dummies(pandas_df['Sex'], dummy_na=False))  
    gender_columns.columns = ['Female', 'Male']
    
    class_columns =  (pd.get_dummies(pandas_df['Pclass'], dummy_na=False))
    class_columns.columns = ['First_class', 'Second_class', 'Third_class']
    
    embarked_columns =  (pd.get_dummies(pandas_df['Embarked'], dummy_na=True, prefix='Port'))
    
    age_temp =  pandas_df['Age'].apply(lambda x: int(x/5) if pd.notnull(x) else np.nan)
    age_columns =  (pd.get_dummies(age_temp, dummy_na=True, prefix='Age'))

    fare_temp =  pandas_df['Fare'].apply(lambda x: int(x/10) if pd.notnull(x) else np.nan)
    fare_columns =  (pd.get_dummies(fare_temp, dummy_na=True, prefix='Fare'))
    
    parch = pandas_df['Parch']
    sibsp = pandas_df['SibSp']
    
    final_data = pd.concat([class_columns, gender_columns, embarked_columns, age_columns, fare_columns, parch, sibsp], axis=1)

    if extract_Y:
        return final_data, pandas_df['Survived']
    else:
        return final_data


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64, "Fare": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64, "Fare": np.float64}, )


df_X, df_Y = preprocess_data(train)
df_test    = preprocess_data(test, extract_Y=False)

for c in df_test.columns:
    if c not in df_X.columns:
        df_X[c] = 0 

for c in df_X.columns:
    if c not in df_test.columns:
        df_test[c] = 0

X_train = df_X.as_matrix()
Y_train = df_Y.as_matrix()
X_test = df_test.as_matrix()


for i in [0.01]:
    fs = LinearSVC(C=i, penalty="l1", dual=False)
    X_train_new = fs.fit_transform(X_train, Y_train)
    X_test_new = fs.transform(df_test)

    print (X_train_new.shape)
    
    # for nt in range(1, 200):
    #     ntrees = nt
    #     model = RandomForestClassifier(n_estimators=ntrees)
    #     scores = cross_validation.cross_val_score(model, X_train_new, Y_train, cv=5)
    #     print ("Accuracy for %i trees: %0.2f (+/- %0.2f)" % (ntrees, scores.mean(), scores.std() * 2))

    model = RandomForestClassifier(n_estimators=6)
    model.fit(X_train_new, Y_train)
    outcome = model.predict(X_test_new)
    result = pd.concat([test['PassengerId'], pd.Series(outcome)], axis = 1)
    result.columns = ['PassengerId', 'Survived']
    result.to_csv('submission.csv', index=False)

train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)
