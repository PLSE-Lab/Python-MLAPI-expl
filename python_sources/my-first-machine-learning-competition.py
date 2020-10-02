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
##We can see that age has a lot of missing values. We will use mean imputation 

sns.set()

plt.hist(train['Fare'])
plt.show()
## Make Pclass a dummy variable
#trainDF = get_dummies(train)
#print(trainDF.shape)
#THIS DIDNT WORK BECAUSE IT MADE 1700 Columns

##Split train data into X and label
###Use .values to get a numpy array instead of a pandas series
y_train = train['Survived'].values
X_train = train['Fare'].values.reshape(-1,1)
#y_test = test['Survived'].values
X_test = test['Fare'].values.reshape(-1,1)

testID = test['PassengerId'].values

#### Model 1: Fare
#Instantiate knn model 

steps = [('imp',Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
pipeline.score(X_train, y_train)
#Score is .746

cvScores = cross_val_score(pipeline,X_train,y_train,cv = 5)
mod1 = np.mean(cvScores)

y_pred = pipeline.predict(X_test)






finalSub = pd.DataFrame({'PassengerId': testID,'Survived': y_pred})
finalSub.shape
finalSub.head()


finalSub.to_csv( 'titanicPred.csv' , index = False )
