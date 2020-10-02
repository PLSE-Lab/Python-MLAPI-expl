import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
from collections import OrderedDict
import random


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 
#'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
#starting with training data
#look at the trends in each of the data variable
#Survived vs Pclass
plt.figure(0)
train.groupby('Pclass')['Survived'].mean().plot(kind='bar',title='Pclass vs Survived')
plt.figure(1)
# gender vs Survived
train.groupby('Sex')['Survived'].mean().plot(kind='bar',title='Pclass vs Gender')
plt.figure(2)
# age vs survived 
train.groupby('Age')['Survived'].mean().plot(kind='bar',title='Pclass vs Age')
plt.figure(3)
# Sibsp vs survived
train.groupby('SibSp')['Survived'].mean().plot(kind='bar',title='Pclass vs SibSp')
plt.figure(4)
# Parch vs survived
train.groupby('Parch')['Survived'].mean().plot(kind='bar',title='Pclass vs Parch')

plt.show()

# removing rows with NaN ages 
# that gives up nearly 20% of the test data
df_train=train[np.isfinite(train['Age'])]

Y=df_train['Survived']
df_train.replace({'male':0,'female':1},inplace=True)
X=df_train[['Pclass', 'Sex', 'Age', 'SibSp','Parch']]
# spliting into testing and training data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


age_range=range(int(test['Age'].min()),int(int(test['Age'].max())))
test['Age'].fillna(age_range[int(random.random()*len(age_range))],inplace=True)
df_test=test
df_test.replace({'male':0,'female':1},inplace=True)
df_test_X=df_test[['Pclass', 'Sex', 'Age', 'SibSp','Parch']]

def run_one_model(func_model,func_scoring):
    '''
    func_model = model to be used
    func_scoring = scoring method to me be used
    '''
    try:
        logreg = func_model()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        #print(func.__name__,' Accuracy Score :',logreg.score(df_train_X,df_train_Y))
        return func_scoring(y_test,y_pred)
        
    except:
        return 0.0

# trying Linear Models
def run_all_models(cls_list,scoring_func):
    results={}
    for item in cls_list:
        for methods in dir(item):
            if not methods.startswith('_'):
                results[item.__name__+':'+methods]=run_one_model(getattr(item,methods),scoring_func)    
    return results

# list of models 
cls_list=[sklearn.linear_model,sklearn.svm,sklearn.ensemble,sklearn.neighbors,sklearn.naive_bayes]
# list of scoring methods
scoring_functions=[f1_score,accuracy_score]

# sorting the results
results_f1=run_all_models(cls_list,scoring_functions[0])
sorted_results_f1=OrderedDict(sorted(results_f1.items(), key=lambda t: t[1],reverse=True))

results_accuracy=run_all_models(cls_list,scoring_functions[1])
sorted_results_accuracy=OrderedDict(sorted(results_accuracy.items(), key=lambda t: t[1],reverse=True))

print('Results based on f1-scoring')
print(sorted_results_f1)

print('Results based on accuracy_score')
print(sorted_results_accuracy)

clf=sklearn.ensemble.GradientBoostingClassifier()
clf.fit(X,Y)
Y_pred=clf.predict(df_test_X)

submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
