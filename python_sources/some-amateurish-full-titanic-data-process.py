from sklearn.ensemble import RandomForestClassifier as rfc;
from collections import Counter;
import re;
import csv as csv;
import numpy as np;
import pandas as pd;
import operator;

csv_data = csv.reader(open('../input/train.csv', 'r'));
header = csv_data.__next__();

#Convert the CSV data into a python list
def csv_to_npArray(csv_data):
	data = []
	for k in csv_data:
		data.append(k);
	data = np.array(data);
	return data;

data = csv_to_npArray(csv_data);
del csv_data;

#Raw data
"""
number_of_passengers = np.size(data[0::,1].astype(np.float));
number_survived = np.sum(data[0::,1].astype(np.float));
proportion = number_survived/number_of_passengers; 

print 'No. passengers:\t'+str(number_of_passengers)+'\tNo. survived:\t'+str(number_survived);
print 'The proportion of the survivors is:\t' + str(round(proportion, 3));


del (number_of_passengers);
del (number_survived);
del (proportion);
"""

#Gender issues
#Verify the diff between males and females
"""
pass_female = data[0::,4] == 'female';
pass_male = data[0::,4] != 'female';

survived_female = np.sum(data[pass_female, 1].astype(np.float));
survived_male = np.sum(data[pass_male, 1].astype(np.float));

print '\nGender\t\tTotal:\t\tSurvived:\tRate:'
print 'Female:\t\t' + str(np.sum(pass_female)) + '\t\t'+ str(int(survived_female))+'\t\t' \
	+ str(round(survived_female/np.sum(pass_female),5));
print 'Male:\t\t' + str(np.sum(pass_male)) + '\t\t'+ str(int(survived_male)) + '\t\t'\
	+ str(round(survived_male/np.sum(pass_male),5));
"""

#We already know that the title is extremely important. In fact, the "title" kinda 
#merge the gender information with the class and the age, important factors of
#prediction in this situation. # Let's extract then.

def title_preprocess(data, name_col):
	titles = [];
	for k in data[0::,name_col]:
		titles.append(re.split(r'\,\ (.*?)\ ', k)[1]);

	#counter_obj = Counter(titles);
	"""
	print '\nHidden variable "Class" check-up:';
	for k in counter_obj:
		print '%s: %s' % (k, counter_obj[k]);
	"""
	"""
		Miss.: 182
		Mme.: 1
		Rev.: 6
		Jonkheer.: 1
		Sir.: 1
		Mlle.: 2
		Mrs.: 125
		Capt.: 1
		Col.: 2
		Ms.: 1
		Mr.: 517
		Lady.: 1
		Dr.: 7
		the: 1
		Master.: 40
		Major.: 2
		Don.: 1
	"""
	#Some titles seens to be 'weird', like the 'the' one. 
	#Thus, some of then seens likely to be merged into a single one more numerous.

	#1. 'the'?
	#print '\nTitle "the" analysis:\n' + str(data[titles.index('the')]) + '\n';
	#Seens like the right title should be 'Countess'. Switch it to Mistress.
	try:
		titles[titles.index('the')] = 'Mrs.';
	except:
		None;

	#Append the new information on the data matrix
	#print str(np.shape(data)) + '/'+ str(np.shape(titles));
	titles = np.array(titles)[..., None];
	data1 = np.append(data, titles, axis=1);
	del titles;

	#2. Rev and Dr... Those titles has a small yet expressive number of passengers. 
	#Whats the proportion of survival compared to misters, in general?
	passenger_doc = (data1[0::, -1] == 'Dr.');
	passenger_rev = (data1[0::, -1] == 'Rev.');
	"""
	print '"Doctor" and "Reverend" class lookup:\nDr.: %lf%%\tRev.: %lf%%\n' % \
		(100*np.sum(data1[passenger_doc, 1].astype(np.float))/counter_obj['Dr.'], 
		100*np.sum(data1[passenger_rev, 1].astype(np.float))/counter_obj['Rev.']);
	"""
	#Bad news for reverends, but still a good chance for doctors.
	#Let's merge revereds into Misters.

	data1[passenger_rev, -1] = 'Mr.';
	del(passenger_rev);

	#3. Merge every title that is not "mr", "miss", "master", "mrs" on a single one called "other".
	other_titles = np.logical_and(np.logical_and(data1[0::, -1] != 'Mrs.', data1[0::, -1] != 'Mr.'), 
		np.logical_and(data1[0::, -1] != 'Master.', data1[0::, -1] != 'Miss.'));
	data1[other_titles, -1] = 'Other.';

	#Check new title data.
	#counter_obj = Counter(data1[0::, 12]);
	"""
	print 'New "Title" data:';
	for k in counter_obj:
		print '%s: %s' % (k, counter_obj[k]);
	"""
	#Everything seens OK.

	#Check up the class thing.
	"""
	fst_class_passenger = (data1[0::, 2] == '1');
	snd_class_passenger = (data1[0::, 2] == '2');
	trd_class_passenger = (data1[0::, 2] == '3');

	print '\nRelative proportion of survival between the classes:'
	print '1st: %.4lf\t2nd: %.4lf\t3rd: %.4lf\n' % \
		(np.sum(data1[fst_class_passenger, 1].astype(np.float))/np.size(data1[fst_class_passenger, 1]), 
		(np.sum(data1[snd_class_passenger, 1].astype(np.float)))/np.size(data1[snd_class_passenger, 1]), 
		(np.sum(data1[trd_class_passenger, 1].astype(np.float)))/np.size(data1[trd_class_passenger, 1]));

	del fst_class_passenger;
	del snd_class_passenger;
	del trd_class_passenger;
	"""
	#Indeed, larger class passengers has higher probabilities of survival.
	return data1;

data1 = title_preprocess(data, 3);
del data;

#Let's find out possible missing values on data.
def full_preprocess(my_data):
	try:
		data_frame = pd.DataFrame(data=my_data[0::,1::],
		index=my_data[0::, 0],
		columns=('Survived', 'PClass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title'));

		data_frame[['Survived', 'PClass', 'Age', 'SibSp', 'Parch', 'Fare']] = \
			data_frame[['Survived', 'PClass', 'Age', 'SibSp', 'Parch', 'Fare']].apply(lambda x: pd.to_numeric(x, errors='coerce'));
	except:
		data_frame = pd.DataFrame(data=my_data[0::,1::],
		index=my_data[0::, 0],
		columns=('PClass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title'));

		data_frame[['PClass', 'Age', 'SibSp', 'Parch', 'Fare']] = \
			data_frame[['PClass', 'Age', 'SibSp', 'Parch', 'Fare']].apply(lambda x: pd.to_numeric(x, errors='coerce'));
	#print data_frame.head(10);
	#data_frame.info();


	#Now, let's examine a bit further.
	"""
	print '\nClass x Sex survivability:\n\t\tMale [S/T/P]:\t\tFemale [S/T/P]'
	for i in range(1,4):
		male_fc_total = np.size(data_frame[np.logical_and(data_frame['Sex'] == 'male', data_frame['PClass'] == i)][['Survived']]);
		male_fc_survived = np.sum(data_frame[np.logical_and(data_frame['Sex'] == 'male', data_frame['PClass'] == i)][['Survived']]);
		female_fc_total = np.size(data_frame[np.logical_and(data_frame['Sex'] == 'female', data_frame['PClass'] == i)][['Survived']]);
		female_fc_survived = np.sum(data_frame[np.logical_and(data_frame['Sex'] == 'female', data_frame['PClass'] == i)][['Survived']]);
		print 'Class %d:\t%u/%u/%.4lf\t\t%u/%u/%.4lf' % (i, male_fc_survived, male_fc_total, 100*(male_fc_survived/male_fc_total),
			female_fc_survived, female_fc_total, 100*(female_fc_survived/female_fc_total));
	"""
	#Women on 1st or 2nd class has almost a guaranteed survivability. 
	#Men, in general, has low probability of survival.

	#Let's look at the 'Master' title.
	"""
	print '\nClass x Master survivability:'
	for i in range(1,4):
		master_fc_total = np.size(data_frame[np.logical_and(data_frame['Title'] == 'Master.', data_frame['PClass'] == i)][['Survived']]);
		master_fc_survived = np.sum(data_frame[np.logical_and(data_frame['Title'] == 'Master.', data_frame['PClass'] == i)][['Survived']]);
		print 'Class %d:\t%u/%u/%.4lf' % (i, master_fc_survived, master_fc_total, 100*(master_fc_survived/master_fc_total));
	"""
	#Masters on 1st and 2nd class always survives, however, on 3rd class, the odds are low.

	#import pylab as pl;
	#data_frame['Age'].hist();
	#pl.show();

	#Transform the 'Sex' and 'Embarked' thing onto a numeral representation.
	#First, check NaN values:

	#print 'Checking for NA values on "Sex" and "Embarked" data:'
	#print data_frame[data_frame['Sex'].isnull()];
	#print data_frame[data_frame['Embarked'] == ''];
	#There's two NAs on "Embarked". "Sex" seens already fine.
	data_frame['Sex'] = data_frame['Sex'].map({'female': 0, 'male': 1}).astype(int);

	#Get the most frequent of "Embarked" and assume this value on NAs.
	data_frame.loc[data_frame['Embarked'] == '', 'Embarked'] = Counter(data_frame['Embarked']).most_common(1)[0][0];

	#Same thing with the embarked values, this time S: 0, C: 1, Q:2;
	#print '\nUnique "Embarked" options: %s' % pd.unique(data_frame['Embarked'].values.ravel());
	data_frame['Embarked'] = data_frame['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int);

	#Check if there's a real correlation between the Embarked port and the survivability.
	#WORK HERE!!!!

	#Again, but now with the Title thing.
	data_frame['Title'] = data_frame['Title'].map({\
		'Master.': 0, 'Mrs.': 1, 'Miss.': 2, 'Mr.': 3, 'Other.': 4}).astype(int);

	#Now let's get rid of the NA's values on all the age data,
	#assuming the value of then is the median value.
	#median = np.median(data_frame['Age']);

	data_frame.loc[data_frame['Age'].isnull(), 'Age'] = data_frame['Age'].median(axis='rows', skipna = True);

	#Now with the "Fare" thing.
	data_frame.loc[data_frame['Fare'].isnull(), 'Fare'] = data_frame['Fare'].median(axis='rows', skipna = True);

	#Take a look on 'SibSp' and 'Parch' stuff.
	#print data_frame[np.logical_or(data_frame['SibSp'].isnull(), data_frame['Parch'].isnull())];
	#No missing values. Maybe its a nice idea merge those two information into one called "Family Size".
	data_frame['Fsize'] = data_frame['SibSp'] + data_frame['Parch'] + 1;

	#Check if all stuff worked.
	#print '\nNew "Sex", "Age", "Embarked", "Title" and "Fsize" numeral representation:\n' + \
	#	str(data_frame[['Sex', 'Age', 'Embarked', 'Fsize', 'Title']].head(15));

	#Final touchs
	#print data_frame.dtypes[data_frame.dtypes.map(lambda x: x=='object')];
	#Remove "Cabin", "Ticket", "Name", "SibsSp", "Parch", and "Sex" aswell (because its implicit on 
	#the "title" data, and keeping those two together is redundant). 
	data_frame = data_frame.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Sex'], axis = 1);
	#data_frame = data_frame.dropna();
	#print data_frame.describe();
	return data_frame.values;

#Convert the pandas.dataFrame into a numpy array

train_data = full_preprocess(data1);
del data1;
#print train_data[0:10, 0::];

#Everything seens OK. Let's get the test_data.
test_file = open('../input/test.csv', 'r');
test_data_csv = csv.reader(test_file);
header = test_data_csv.__next__();

#Clean and make changes on the test_data.
test_data = csv_to_npArray(test_data_csv);
del test_data_csv;
test_data1 = title_preprocess(test_data, 2);
del test_data;
test_data_final = full_preprocess(test_data1);

test_file.close();

#Time to do some predictions
rd_forest = rfc(n_estimators = 1200);
rd_forest = rd_forest.fit(train_data[0::,1::], train_data[0::, 0]);

final_result = rd_forest.predict(test_data_final);

#Generate the desired output
print ('\"PassengerId","Survived\"');
for i in range(892,1310):
	print (i, int(final_result[i - 892]));
