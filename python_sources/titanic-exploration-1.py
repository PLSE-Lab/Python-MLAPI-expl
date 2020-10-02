import numpy as np
import pandas as pd
import pylab as P

from sklearn import ensemble

#Set pandas precision
pd.set_option('precision',1)

#Read in input data into dataframes
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Save a copy of the training data to output
train.to_csv('copy_of_the_training_data.csv', index=False)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

#Copy training and test data and clense
clensedTrain = train.copy()
clensedTest = test.copy()

#Check survival percentage
print(clensedTrain['Survived'].sum())

#Check unique values
print('Unique values in Emarked')
print(clensedTrain.Embarked.unique())

#Encode text variables
embarkedMapping = {'S': 0, 'C': 1, 'Q': 2}
clensedTrain['Embarked'] = clensedTrain.Embarked.apply(lambda x: x if not pd.isnull(x) else -1)
clensedTrain.replace({'Embarked' : embarkedMapping }, inplace=True)

clensedTest['Embarked'] = clensedTest.Embarked.apply(lambda x: x if not pd.isnull(x) else -1)
clensedTest.replace({'Embarked' : embarkedMapping }, inplace=True)

#Delete nuisance variables
del clensedTrain['Ticket']
del clensedTest['Ticket']
#del clensedTrain['Embarked']
#del clensedTest['Embarked']
#Delete potentially nuisance variables
del clensedTrain['Name']
del clensedTest['Name']
del clensedTrain['Cabin']
del clensedTest['Cabin']

#Encode text variables
clensedTrain['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
clensedTest['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#Fill empty variables
print(clensedTrain['Age'].isnull().sum())

#Compute mean class ages
meanClassAges = clensedTrain.groupby('Pclass').apply(lambda x: pd.Series(dict(mean=(x.Age).median())))
print(meanClassAges)

meanClassSexAges = clensedTrain.groupby(['Pclass','Sex']).apply(lambda x: pd.Series(dict(mean=(x.Age).median())))
print(meanClassSexAges)

for i in clensedTrain['Age'].index:
    currentAge = clensedTrain['Age'][i]
    if pd.isnull(currentAge):
        pClass = clensedTrain['Pclass'][i]
        pSex = clensedTrain['Sex'][i]
        currentAge = meanClassSexAges['mean'][pClass][pSex]
        clensedTrain.ix[i,'Age'] = currentAge

for i in clensedTest['Age'].index:
    currentAge = clensedTest['Age'][i]
    if pd.isnull(currentAge):
        pClass = clensedTest['Pclass'][i]
        pSex = clensedTest['Sex'][i]
        currentAge = meanClassSexAges['mean'][pClass][pSex]
        clensedTest.ix[i,'Age'] = currentAge

#Compute mean class fares
meanClassFares = clensedTrain.groupby('Pclass').apply(lambda x: pd.Series(dict(mean=(x.Fare).median())))
print(meanClassFares)

#clensedTrain['Age'] = clensedTrain.Age.apply(lambda x: x if not pd.isnull(x) else -1)
#clensedTest['Age'] = clensedTest.Age.apply(lambda x: x if not pd.isnull(x) else -1)
#clensedTest['Fare'] = clensedTest.Faire.apply(lambda x: x if not pd.isnull(x) else -1)

for i in clensedTest['Fare'].index:
    currentFare = clensedTest['Fare'][i]
    if pd.isnull(currentFare):
        pClass = clensedTest['Pclass'][i]
        currentFare = meanClassFares['mean'][pClass]
        clensedTest.ix[i,'Fare'] = currentFare
        
#Normalize fares
#print(clensedTest['Fare'].max())
#maxFare = clensedTrain['Fare'].max()

#clensedTrain['Fare'] = clensedTrain['Fare'].apply(lambda x: x*0.1)
#clensedTest['Fare'] = clensedTest['Fare'].apply(lambda x: x*(1/maxFare))

#Create new engineered features
clensedTrain['FamilySize'] = clensedTrain['SibSp'] + clensedTrain['Parch']
clensedTest['FamilySize'] = clensedTest['SibSp'] + clensedTest['Parch']

clensedTrain['WomenAndChildren'] = clensedTrain['Sex'] * clensedTrain['Parch']
clensedTest['WomenAndChildren'] = clensedTest['Sex'] * clensedTest['Parch']

clensedTrain['Virile'] = 0
clensedTest['Virile'] = 0

for i in clensedTrain['Virile'].index:
    pAge = clensedTrain['Age'][i]
    pClass = clensedTrain['Pclass'][i]
    pSex = clensedTrain['Sex'][i]
    pFare = clensedTrain['Fare'][i]
    if pAge >= 15 and pAge <= 45 and pSex == 0:
        clensedTrain.ix[i,'Virile'] = pClass * pFare
        
for i in clensedTest['Virile'].index:
    pAge = clensedTest['Age'][i]
    pClass = clensedTest['Pclass'][i]
    pSex = clensedTest['Sex'][i]
    pFare = clensedTest['Fare'][i]
    if pAge >= 15 and pAge <= 45 and pSex == 0:
        clensedTest.ix[i,'Virile'] = pClass * pFare

#Dropout redundant columns
del clensedTrain['SibSp']
del clensedTest['SibSp']
del clensedTrain['Parch']
del clensedTest['Parch']

#Output clensed data summary
print("\n\nSummary statistics of training data")
print(clensedTrain.describe())

#Output clensed data summary
print("\n\nSummary statistics of test data")
print(clensedTest.describe())

train_data = clensedTrain.values
trainIds = train_data[:,0]
target = train_data[:,1]
xTrainData = train_data[:,2:]

test_data = clensedTest.values
testIds = test_data[:,0]
xTestData = test_data[:,1:]

# Set a random seed
#clf = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1)
#clf = ensemble.RandomForestClassifier(n_estimators = 500, min_samples_split=4, class_weight={0:0.616,1:0.384})
clf = ensemble.RandomForestClassifier(n_estimators = 500)
clf.fit(xTrainData, target)

#check prediction error    

predictionAccuracy = clf.score(xTrainData, target)
print("Prediction Accuracy: {0:.4f}%".format(predictionAccuracy * 100))

trainingOutputFile = "training_output.csv"
 
predictionError = 0;
with open(trainingOutputFile, "w") as outfile:
    outfile.write("Id,Survival,Truth\n")
    for e, val in enumerate(list(clf.predict_proba(xTrainData))):
        predictedSurvival = 0
        if val[1] > 0.5:
            predictedSurvival = 1
            
        #if predictedSurvival == 1 and xTrainData[e,7] == 1 and val[1] < 0.51:
        #    print('adjusted one')
        #    predictedSurvival = 0
        
        if predictedSurvival != target[e]:
            print(val[1])
            outfile.write("%s,%s,%s,%s,%s,%s,%s\n"%(trainIds[e],xTrainData[e,0],xTrainData[e,1],xTrainData[e,2],xTrainData[e,7],val[1],target[e]))
            predictionError = predictionError + 1
        
adjustedPredictionAccuracy = predictionError / 891 * 100
print("Adjusted Prediction Accuracy: {0:.4f}%".format(100 - adjustedPredictionAccuracy))

#create submission
submissionFile = "submission.csv"
 
with open(submissionFile, "w") as outfile:
    outfile.write("PassengerId,Survived\n") 
    for e, val in enumerate(list(clf.predict(xTestData))):
        outfile.write("%d,%d\n"%(testIds[e],val))