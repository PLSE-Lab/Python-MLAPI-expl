from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation 
from keras.optimizers import SGD 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas 
import numpy as np 

def create_model(): 
	model = Sequential() 
	model.add(Dense(64,input_dim=8,init='uniform',activation='relu'))
	model.add(Dense(64,activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
	return model
train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")
train_data = train_data.drop('Cabin',axis=1)
train_data = train_data.drop('Ticket',axis=1)
train_data = train_data.drop('Name',axis=1)


test_data = test_data.drop('Cabin',axis=1)
test_data = test_data.drop('Ticket',axis=1)
test_data = test_data.drop('Name',axis=1)
Y_train = train_data['Survived']
train_data = train_data.drop('Survived',axis=1)
mean_age = train_data['Age'].mean() 
train_data.loc[:,('Age')]= train_data['Age'].fillna(mean_age)
test_data.loc[:,('Age')] = test_data['Age'].fillna(mean_age)
embarked_occurence = train_data['Embarked'].value_counts().idxmax() 
train_data.loc[:,('Embarked')] = train_data['Embarked'].fillna(embarked_occurence)
test_data.loc[:,('Embarked')] = test_data['Embarked'].fillna(embarked_occurence)
le = preprocessing.LabelEncoder()
le.fit(train_data['Sex'])
train_data.loc[:,('Sex')] = le.transform(train_data['Sex'])
test_data.loc[:,('Sex')] = le.transform(test_data['Sex'])
le.fit(train_data['Embarked'])
train_data.loc[:,('Embarked')] = le.transform(train_data['Embarked'])
test_data.loc[:,('Embarked')] = le.transform(test_data['Embarked'])
print(train_data)
X_train = train_data.as_matrix() 
X_test = test_data.as_matrix()
print(len(X_train[0]))
model = create_model()
model.fit(X_train,Y_train,batch_size = len(train_data),nb_epoch = 1000)
predictions = model.predict_classes(X_test,batch_size=len(test_data))
outputFrame = pandas.DataFrame()
outputFrame['PassengerId'] = test_data['PassengerId']
outputFrame['Survived'] = predictions
outputFrame.to_csv("output.csv",index =False)