# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=400000)


# Importing the dataset
dataset = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#list(dataset)
dataset = dataset[['Attrition',
                   'Age',
                   'BusinessTravel',
                   'DailyRate',
                   'Department',
                   'DistanceFromHome',
                   'Education',
                   'EducationField',
                   'EmployeeCount',
                   'EnvironmentSatisfaction',
                   'Gender',
                   'HourlyRate',
                   'JobInvolvement',
                   'JobLevel',
                   'JobRole',
                   'JobSatisfaction',
                   'MaritalStatus',
                   'MonthlyIncome',
                   'MonthlyRate',
                   'NumCompaniesWorked',
                   'Over18',
                   'OverTime',
                   'PercentSalaryHike',
                   'PerformanceRating',
                   'RelationshipSatisfaction',
                   'StandardHours',
                   'StockOptionLevel',
                   'TotalWorkingYears',
                   'TrainingTimesLastYear',
                   'WorkLifeBalance',
                   'YearsAtCompany',
                   'YearsInCurrentRole',
                   'YearsSinceLastPromotion',
                   'YearsWithCurrManager']]
dataset['JobInvolment_On_Salary']= dataset['JobInvolvement'] / dataset['MonthlyIncome'] * 1000
dataset['WifeAndBad_Worklife_Balance'] = np.where(dataset['MaritalStatus']=='Married', 
       dataset['WorkLifeBalance']-2,
       dataset['WorkLifeBalance']+1)
dataset['DistanceFromHome_rootedTo_JobSatisfaction'] = dataset['DistanceFromHome']**(1/dataset['JobSatisfaction'])
dataset['TotalJobSatisfaction'] = dataset['EnvironmentSatisfaction'] + dataset['JobSatisfaction'] + dataset['RelationshipSatisfaction'] + dataset['JobLevel'] 
dataset['OldLowEmployeeTendToStay'] = dataset['YearsAtCompany'] / dataset['JobLevel'] 


X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_6= LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_9= LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_13= LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])
labelencoder_X_15= LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
labelencoder_X_19= LabelEncoder()
X[:, 19] = labelencoder_X_19.fit_transform(X[:, 19])
labelencoder_X_20= LabelEncoder()
X[:, 20] = labelencoder_X_20.fit_transform(X[:, 20])
X = X.astype(float)

labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)
#no dummy for male/female because only 2 categ
onehotencoder = OneHotEncoder(categorical_features = [1,3,6,13,15])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




layer1Neurones = 16
dropout = 0.1
epochs = 2000
batch_size = 30
optimizer = 'adadelta'#['adam', 'rmsprop', 'nadam', 'adamax','adadelta','adagrad', 'SGD']


#-------------------------- Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(layer1Neurones, kernel_initializer="truncated_normal", activation = 'relu', input_shape = (X.shape[1],))) #to add layer
    classifier.add(Dropout(dropout)) #to add layer
    classifier.add(Dense(1, kernel_initializer="truncated_normal", activation = 'sigmoid', )) #outputlayer
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, epochs = epochs)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
mean = accuracies.mean()
variance = accuracies.std()

print('variance', variance)
print('mean', mean)
print('accuracy',accuracy)
print('accuracies',accuracies)

