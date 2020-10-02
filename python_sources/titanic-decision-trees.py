from sklearn.metrics import accuracy_score
import csv
from sklearn import tree

features_train = []
labels_train = []
features_test = []
labels_test = []
	
with open('../input/train.csv') as training_data:
	reader = csv.DictReader(training_data)
	for row in reader:
		features_train.append([0 if row['Sex'] == 'male' else 1, int(row['Pclass']), int(row['SibSp'])])
		labels_train.append(int(row['Survived']))
		#print row['Survived'], row['Age'], row['Sex'], row['Pclass'], row['SibSp'], row['Parch']
		
with open('../input/test.csv') as test_data:
	reader = csv.DictReader(test_data)
	for row in reader:
		features_test.append([0 if row['Sex'] == 'male' else 1, int(row['Pclass']), int(row['SibSp'])])			
		
with open('../input/gendermodel.csv') as gender_model:
	reader = csv.DictReader(gender_model)
	for row in reader:
		labels_test.append(int(row['Survived']))			
	

clf = tree.DecisionTreeClassifier(min_samples_split=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(accuracy_score(pred, labels_test))
