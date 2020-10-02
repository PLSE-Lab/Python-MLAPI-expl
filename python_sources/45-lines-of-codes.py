# importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
# importing datasets
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
data = [train, test]
# COMPLETING: complete or delete missing values in train and test/validation dataset
for dataset in data:    
    # preprocessing     
    dataset['Name_length'] = dataset['Name'].apply(len)
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    dataset['Sex'] = dataset['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    number = LabelEncoder()
    dataset['Embarked_Code'] = number.fit_transform(dataset['Embarked'])
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    #delete the Ticket feature/column and others previously stated to exclude in train dataset
    drop_column = ['Name', 'Cabin', 'Ticket', 'Embarked']
    dataset.drop(drop_column, axis=1, inplace = True)
print(train.isnull().sum())
print(test.isnull().sum())
# spliting the train data into explainatory(X) and response(y) variables 
X_train = train.drop(['Survived', 'PassengerId'], axis = 1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis = 1)
###################### Machine Learning Techniques ############################ 
# K-Nearest Neighbors
knn  = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
knn_train_pred = knn.predict(X_train)
# Obtaining confusion matrix and classification report
print(confusion_matrix(y_train, knn_train_pred))
print(classification_report(y_train, knn_train_pred))
knn_test_pred = knn.predict(X_test)
test['knn_test_pred'] = knn_test_pred
knn_prediction = test[['PassengerId', 'knn_test_pred']]
print(knn_prediction)
# here we got 100% Accuracy for training data.