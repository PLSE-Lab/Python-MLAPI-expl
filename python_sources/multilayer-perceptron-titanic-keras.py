# Loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Multilayer perceptron Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# importing data
traindata = pd.read_csv('/kaggle/input/titanic/train.csv')
testdata = pd.read_csv('/kaggle/input/titanic/test.csv')

# encoding string data columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
traindata['Sex'] = le.fit_transform(traindata['Sex'])
testdata['Sex'] = le.fit_transform(testdata['Sex'])
traindata['Embarked'] = le.fit_transform(traindata['Embarked'].astype(str))
testdata['Embarked'] = le.fit_transform(testdata['Embarked'].astype(str))

# feature selection
# some groups of people were more likely to survive than others, 
# such as women, children, and the upper-class.
X = traindata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values
y = traindata.iloc[:,1].values
X_real_test = testdata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values

# handling missing values
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer()
X_transformed = imputer.fit_transform(X)
X_real_test = imputer.fit_transform(X_real_test)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

labels = np_utils.to_categorical(y)

def mlp_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, init='normal', activation='relu'))
	model.add(Dense(2, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = mlp_model()
# Fit the model
model.fit(X_transformed, labels, nb_epoch=200, batch_size=5, verbose=2)

pred = model.predict(X_real_test)
y_pred_test = pred.argmax(1)
pid = testdata[['PassengerId']].values
res = np.expand_dims(y_pred_test,axis=1)
f = np.hstack((pid,res))
df = pd.DataFrame(f, columns = ['PassengerId', 'Survived']) 
df.to_csv('gender_submission.csv', index=False)