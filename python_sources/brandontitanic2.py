import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pandas import get_dummies
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

testID = test['PassengerId'].values

pClassDum = pd.get_dummies(train['Pclass'])
pTestDum = pd.get_dummies(test['Pclass'])
# Setup the pipeline steps: steps
steps = [('imp',Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)
X_train = train[['Fare','Age']]
X_test = test[['Fare','Age']]
y_train = train['Survived'].values
frames = [X_train,pClassDum]
testFrames = [X_test,pTestDum]
X_train = pd.concat(frames, axis = 1)
X_test = pd.concat(testFrames, axis = 1)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

finalSub = pd.DataFrame({'PassengerId': testID,'Survived': y_pred})
finalSub.shape
finalSub.head()
finalSub.to_csv( 'titanicPred.csv' , index = False )
#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)