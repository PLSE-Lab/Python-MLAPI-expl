import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def oneOfNEncoding(featureVectors):
    categoricalFeatures = [feature for feature in featureVectors.columns if featureVectors[feature].dtype == 'object' and feature != 'Id']
    if len(categoricalFeatures) > 0:
        vec = DictVectorizer()
        vec_data = pd.DataFrame(vec.fit_transform(featureVectors[categoricalFeatures].to_dict(orient='records')).toarray())
        vec_data.columns = vec.get_feature_names()
        vec_data.index = featureVectors.index

    featureVectors = featureVectors.drop(categoricalFeatures, axis=1)
    featureVectors = featureVectors.join(vec_data)
    return featureVectors


def prepareData(data):

    #New features
    data['Title'] = data.Name.apply(lambda x: x.split(',')[1].split(" ")[1])
    data['CabinLetter'] = data.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'No data')

    #drop relevant columns
    data = data.drop(['PassengerId','Name','Cabin','Ticket'],1)

    for col in ['Pclass','Sex','Embarked']:
        data[col] = data[col].astype(str)

    #One Of N encoding for data
    data = oneOfNEncoding(data)

    return data

###########################################################
###########################################################

trainData = pd.read_csv("train.csv")
testData = pd.read_csv('test.csv')
label = 'Survived'

###########################################################
## Prepare train data and build model
###########################################################

trainData = prepareData(trainData)
colsInTrainingData = [col for col in trainData.columns if col != label]

#Fill in missing ages
data_NoNan = trainData[~np.isnan(trainData.Age)]
knnModel = KNeighborsRegressor().fit(data_NoNan.drop(['Age','Survived'],1),data_NoNan.Age)
trainData.loc[:,'Age'] = trainData.apply(lambda x: x.Age if ~np.isnan(x.Age) else knnModel.predict(x.drop(['Age','Survived']))[0],1)

#train and test models
#clf = xgb.XGBClassifier()
clf = RandomForestClassifier()
clf.fit(trainData.drop(label,1),trainData[label])

colsInFitDat = trainData.drop(label,1).columns

###########################################################
## Prepare test data and make predictions
###########################################################

testData = prepareData(testData)

colsToDrop = [col for col in testData.columns if col not in colsInTrainingData]
testData = testData.drop(colsToDrop,1)

colsToAdd = [col for col in colsInTrainingData if col not in testData.columns]
for col in colsToAdd:
    testData[col] = 0

testData.loc[:,'Age'] = testData.apply(lambda x: x.Age if ~np.isnan(x.Age) else knnModel.predict(x.drop('Age'))[0],1)

#Fill nan in fare with median
testData['Fare'] = testData['Fare'].fillna(testData['Fare'].median())

#Make predictions
predictions = pd.Series(data = clf.predict(testData), index = testData.index)

print(predictions)

predictions.to_csv("RF_predictions.csv")