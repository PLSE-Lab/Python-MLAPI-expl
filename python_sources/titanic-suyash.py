
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data and seperate labels
data = pd.read_csv('../input/train.csv')
labels_train = data['Survived']
features_train = data.drop(['Survived'], axis=1)
print(features_train.head())
# axis set to 0 as we are talking about the columns

def accuracy_measure(labels_test, features_test):
    predicted_survival = classify(features_test)
    true_classified = float(sum((labels_test == predicted_survival)))/len(labels_test)
    return true_classified
def classify(features_set):
    predicted_survival = []
    for _,feature_data in features_set.iterrows():
        if(feature_data['Sex'] == 'male'):
            if((feature_data['Age']<10)and(feature_data['Pclass']!=3)):
                predicted_survival.append(1)
            else:
                predicted_survival.append(0)
        else:
            if(feature_data['Embarked'] == 'S') and (feature_data['Pclass'] == 3):
                predicted_survival.append(0)
            else:
                predicted_survival.append(1)
    return predicted_survival
data_test = pd.read_csv('../input/test.csv')
#labels_test = data_test['Survived']
#features_test = data_test.drop(['Survived'], axis=1)
print(accuracy_measure(labels_train, features_train))
# On the train set

def makeFinalAnswer(test_set):
    predictions = classify(test_set)
    a = pd.DataFrame({'PassengerId':test_set['PassengerId'], 'Survived':predictions})
    return(a)
    
makeFinalAnswer(data_test)