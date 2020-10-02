# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('../input/titanic-prediction/train.csv')
data_test = pd.read_csv('../input/titanic-prediction/test.csv')

def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1,0,6,12,18,26,40,60,80,120)
	group_names = ['unknown','baby','child','student','teenager','adult','middle-aged','senior','old']
	categories = pd.cut(df.Age,bins,labels=group_names)
	df.Age = categories
	#print 'simplify_ages_finish'
	return categories

def simplify_cabins(df):
	df.Cabin = df.Cabin.fillna('N')
	df.Cabin = df.Cabin.apply(lambda x : x[0])
	#print 'simplify_cabins_finish'
	return df


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    #print 'simplify_fares_finish'
    return df

def format_embarked(df):
	df.Embarked = df.Embarked.fillna('N')
	return df

def format_name(df):
	#df['Lname'] = df.Name.apply(lambda x:re.split(" |\, " ,x)[0])
	#df['NamePrefix'] = df.Name.apply(lambda x:re.split(" |\. " ,x)[1])

	df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
	df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
	#print 'format_name_finish'
	return df

def has_parch(df):
	df.loc[df['Parch'] > 0,'Parch'] = 1
	return df



def drop_features(df):
	df.drop(['Ticket', 'Name'], axis=1,inplace=True)
	#print 'drop_features_finish'
	return df


def transform_features(df):
	#print data_train.shape
	simplify_ages(df)
	#print data_train.shape
	simplify_cabins(df)
	simplify_fares(df)
	format_embarked(df)
	format_name(df)
	has_parch(df)
	drop_features(df)
	return df


data_train = transform_features(data_train)
data_test = transform_features(data_test)

print (data_train)

def encode_features(df_train,df_test):
	features = ['Fare','Cabin','Age','Sex','Lname','NamePrefix','Embarked','Parch']
	#print df_train.columns
	#print df_test.columns
	df_combined = pd.concat([df_train[features],df_test[features]])
	#print df_combined

	for feature in features:

		le = preprocessing.LabelEncoder()
		le = le.fit(df_combined[feature])
		df_train[feature] = le.transform(df_train[feature])
		df_test[feature] = le.transform(df_test[feature])
		
	return df_train,df_test

data_train,data_test = encode_features(data_train,data_test)


x_all = data_train.drop(['Survived','PassengerId','Lname'],axis = 1)
y_all = data_train['Survived']
num_test = 0.4
X_train,X_test,y_train,y_test = train_test_split(x_all,y_all,test_size=num_test,random_state=23)



'''
#--------------logistic regression----------------#

lr = LogisticRegression(C=10000,random_state=0)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print (accuracy_score(y_test,predictions))
'''


#--------------random forest----------------#
clf = RandomForestClassifier()

parameters = {
			  'n_estimators': [4, 6, 9], 
              'max_features': ['log2'], 
              'criterion': ['entropy'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]

}



# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


