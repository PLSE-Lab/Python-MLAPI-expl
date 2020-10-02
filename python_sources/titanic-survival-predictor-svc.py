import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import copy
#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#read in the training and test data
trainData = pd.read_csv("../input/train.csv", index_col=0)
testData = pd.read_csv("../input/test.csv", index_col=0)

together = [trainData, testData]
fullData = pd.concat(together)

def titles(x):
	z = x.split()
	for i in z:
		if (i=='Mr.'):
			return 'Mr'
		elif (i=='Mrs.'):
			return 'Mrs'
		elif (i=='Master.'):
			return 'Master'
		elif (i=='Miss.'):
			return 'Miss'
		elif (i=='Ms.'):
			return 'Miss'
		elif (i=='Rev.'):
			return 'Sir'
		elif (i=='Col.'):
			return 'Sir'
		elif (i=='Capt.'):
			return 'Sir'
		elif (i=='Dr.'):
			return 'Sir'
		elif (i=='Major.'):
			return 'Sir'
		elif (i=='M.'):
			return 'Sir'
		elif (i=='Mme.'):
			return 'Lady'
		elif (i=='Mlle.'):
			return 'Lady'
		elif (i=='Don.'):
			return 'Sir'
		elif (i=='Lady.'):
			return 'Lady'
		elif (i=='Sir.'):
			return 'Sir'
		elif (i=='Countess.'):
			return 'Lady'
		elif (i=='Jonkheer.'):
			return 'Lady'
	return 'Other'

def travelsWith(x, ticketNumberCounts):
	return ticketNumberCounts[x]-1

def cabins(x):
	if pd.isnull(x):
		return 'N'
	z = x.split()[0]
	if len(z)>1:
		return z.split()[0][0]
	else:
		return z[0]

#train = trainData.drop('Cabin', axis = 1)
fullData['Titles'] = fullData['Name'].apply(titles)
fullData['Relationships'] = fullData['Parch']+fullData['SibSp']
fullData['Cabs'] = fullData['Cabin'].apply(cabins)

a = fullData['Ticket'].value_counts()
fullData['TravelsWith'] = fullData['Ticket'].apply(travelsWith, args = [a])
#found on the internet, these individuals bordered at Southampton
fullData['Embarked'] = fullData['Embarked'].fillna('S')


index = fullData[pd.isnull(fullData['Fare'])].index[0]
value = fullData[(fullData['Pclass']==3) & (fullData['Sex']=='male') & (fullData['Embarked']=='S') & (fullData['Titles']=='Mr') & (fullData['Relationships']==0) & (fullData['TravelsWith']==0)]['Fare'].median()
fullData.set_value(index,'Fare',value)


fullData['Titles'] = fullData['Titles'].astype('category', categories = ['Mr','Mrs', 'Master', 'Miss', 'Sir', 'Lady','Other'])
titleCats = fullData['Titles'].cat.categories
fullData['Titles'].cat.rename_categories(np.linspace(1,len(titleCats),len(titleCats), dtype='int'), inplace = True)
fullData['Embarked'] = fullData['Embarked'].astype('category')
embarkedCats = fullData['Embarked'].cat.categories
fullData['Embarked'].cat.rename_categories(np.linspace(1,len(embarkedCats),len(embarkedCats), dtype='int'), inplace = True)
fullData['Sex'] = fullData['Sex'].astype('category')
sexCats = fullData['Sex'].cat.categories
fullData['Sex'].cat.rename_categories(np.linspace(1,len(sexCats),len(sexCats), dtype='int'), inplace = True)
fullData['Cabs'] = fullData['Cabs'].astype('category')
cabCats = fullData['Cabs'].cat.categories
fullData['Cabs'].cat.rename_categories(np.linspace(1,len(cabCats),len(cabCats), dtype='int'), inplace = True)


#preprocessing the data
#Sex
xTrain = fullData.values[:,7:8]
encSex = OneHotEncoder()
encodedArraySex = encSex.fit_transform(xTrain).toarray()

xTrain = fullData.values[:,2:3]
encEmbarked = OneHotEncoder()
encodedArrayEmbarked = encEmbarked.fit_transform(xTrain).toarray()

xTrain = fullData.values[:,11:12]
encTitles = OneHotEncoder()
encodedArrayTitles = encTitles.fit_transform(xTrain).toarray()


xTrain = fullData.values[:,13:14]
encCabs = OneHotEncoder()
encodedArrayCabs = encCabs.fit_transform(xTrain).toarray()

for i in range(len(sexCats)):
	string = 'Sex_' + sexCats[i]
	fullData[string] = encodedArraySex[:,i]

for i in range(len(embarkedCats)):
	string = 'Embarked_' + embarkedCats[i]
	fullData[string] = encodedArrayEmbarked[:,i]

for i in range(len(titleCats)):
	string = 'Titles_' + titleCats[i]
	fullData[string] = encodedArrayTitles[:,i]

for i in range(len(cabCats)):
	string = 'Cab_' + cabCats[i]
	fullData[string] = encodedArrayCabs[:,i]

fullData = fullData.drop('Cabin',axis = 1)
fullData = fullData.drop('Cabs',axis = 1)
fullData = fullData.drop('Name',axis = 1)
fullData = fullData.drop('Embarked',axis = 1)
fullData = fullData.drop('Sex',axis = 1)
fullData = fullData.drop('Ticket',axis = 1)
fullData = fullData.drop('Parch',axis = 1)
fullData = fullData.drop('SibSp',axis = 1)
fullData = fullData.drop('Titles',axis = 1)
fullData = fullData.drop('Survived',axis = 1)


fullData = fullData.drop('Sex_male',axis = 1)
fullData = fullData.drop('Embarked_Q',axis = 1)
fullData = fullData.drop('Titles_Other',axis = 1)
fullData = fullData.drop('Cab_T',axis = 1)

AgeNAs = fullData[pd.isnull(fullData['Age'])]
ageTable = fullData.iloc[:trainData.shape[0],:].groupby(['Pclass','Sex_female']).median()['Age']
for i in range(len(AgeNAs)):
	#print(AgeNAs.iloc[i,:].name)
	name = AgeNAs.iloc[i,:].name
	fullData.loc[name,'Age'] = ageTable.xs((AgeNAs.loc[name,'Pclass'],AgeNAs.loc[name,'Sex_female']), level = ('Pclass','Sex_female'))[0]

#apply standard scaler
for i in range(5):
	scaler = StandardScaler()
	fullData.iloc[:,i] = scaler.fit_transform(fullData.iloc[:,i].values.reshape((1309,1)))

for i in range(5,22):
	fullData.iloc[:,i] = fullData.iloc[:,i].apply(lambda x: -1 if x == 0 else 1)

#then split back up into original train and test data sets
trainX = fullData.iloc[:trainData.shape[0],:]
trainY = trainData['Survived']
testX = fullData.iloc[trainData.shape[0]:,:]


svc = SVC(gamma = 0.0491, C = 0.8174)

svc.fit(trainX,trainY)
svc.score(trainX,trainY)

predictedSurv = svc.predict(testX)
submission = pd.DataFrame(predictedSurv.astype('int'))
submission.index = testX.index
submission.index.name = 'PassengerId'
submission.columns = ['Survived']
submission.to_csv('titanic_submission.csv')
