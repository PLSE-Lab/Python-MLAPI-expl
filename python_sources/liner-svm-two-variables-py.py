### My first Kaggle step: A simple script 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv ('../input/test.csv')

################# Deal with Missing Data ###############################

#identify columns that have missing values
# train_df.describe (include = 'all')
# Columns with missing values: Age, Cabin, Embarked have missing values

# test_df.describe (include = 'all')
# Fare, cabin and Embarked have missing values 
# I tentatively plan to drop Embarked and Cabin to keep it simple

# Replace missing Age data with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(train_df[['Age']])
train_df['Age'] = imputer.transform(train_df[['Age']]).ravel()
test_df['Age'] = imputer.transform(test_df[['Age']]).ravel()

#################  Data Normalization ##############################

# Ref: https://www.kaggle.com/sinakhorami/titanic-best-working-classifier/notebook

full_data = [train_df, test_df]

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
  
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4


#################  Feature Selection #######################
selected_features = ['Sex', 'Age']

X_train=train_df.loc[:, selected_features].values
y_train=train_df.loc[:, ['Survived']].values

X_test=test_df.loc[:, selected_features].values

##################  SVM Linear #####################

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Save output

ids = test_df['PassengerId']

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_pred })
output.to_csv('titanic-predictions.csv', index = False)

output.head()


