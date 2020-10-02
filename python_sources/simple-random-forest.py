import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


training = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
testing = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

def create_features(df):
    df['Name_title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    df["NameLength"] = df["Name"].apply(lambda x: len(x))
    df['Family_size'] = df['Parch'] + df['SibSp'] + 1
    df['Is_alone'] = 0
    df.loc[df['Family_size'] == 1, 'Is_alone'] = 1
    return df

training = create_features(training)
testing = create_features(testing)

def preprocess_data(df, isTest=False):
    dfFeatures = df.set_index('PassengerId')
    sex_dict = {'male':0, 'female':1}
    embarked_dict = {'C':1, 'Q':2, 'S':3}
    name_titles = dfFeatures.Name_title.unique()
    name_title_dict = dict(zip(name_titles, range(len(name_titles))))
    dfFeatures = dfFeatures.replace({'Sex': sex_dict, 'Embarked':embarked_dict, 'Name_title':name_title_dict})
    dfFeatures.fillna(dfFeatures.median(), inplace=True)
    dfFeatures.loc[dfFeatures['Age'] <= 16, 'Age'] = 0
    dfFeatures.loc[(dfFeatures['Age'] > 16) & (dfFeatures['Age'] <= 32), 'Age'] = 1
    dfFeatures.loc[(dfFeatures['Age'] > 32) & (dfFeatures['Age'] <= 48), 'Age'] = 2
    dfFeatures.loc[(dfFeatures['Age'] > 48) & (dfFeatures['Age'] <= 64), 'Age'] = 3
    dfFeatures.loc[dfFeatures['Age'] >= 64, 'Age'] = 4
    if isTest:
        dfFeatures = dfFeatures.drop(['Name', 'Ticket', 'Cabin', 'Family_size'], 1)
        return dfFeatures                 
    else:
        dfLabels = dfFeatures.Survived
        dfFeatures = dfFeatures.drop(['Name', 'Ticket', 'Cabin', 'Survived', 'Family_size'], 1)
        return dfFeatures, dfLabels
                                     
df_trainX, df_trainY = preprocess_data(training)
df_testX = preprocess_data(testing, isTest=True)

trainX = df_trainX.as_matrix()
trainY = df_trainY.as_matrix()
testX = df_testX.as_matrix()
#trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.1)

cls = RandomForestClassifier(n_estimators=50, min_samples_split=4, min_samples_leaf=2)
cls = cls.fit(trainX, trainY)
#print('Validation score:', cls.score(validX, validY))

predictions = cls.predict(testX)


ids = testing['PassengerId'].as_matrix()
submission = np.column_stack((ids.T, predictions.T))
f_handle = open('submission.csv', 'ab')
np.savetxt(f_handle, np.array([['PassengerID', 'Survived']]), delimiter=',', fmt='%s')
np.savetxt(f_handle, submission, delimiter=',', fmt='%i')
f_handle.close()