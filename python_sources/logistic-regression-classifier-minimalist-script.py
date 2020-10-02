#!/usr/bin/env python
# coding: utf-8

# This is a minimalist script which applies the logistic regression classifier from scikit-learn, to the 'Titanic' data set. It produces a score of around 0.77033 (i.e. 322 out of 418 are correctly classified), but this is not good, nor is it the point: the purpose of this script is to serve as a basic starting framework from which you can launch your own exploratory data analysis, perform feature engineering, parameter selection, etc.

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a minimal script to perform a classification 
# using the logistic regression classifier from scikit-learn 
# Carl McBride Ellis (18.IV.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')

#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch"]

#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train       = pd.get_dummies(train_data[features])
y_train       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])

#===========================================================================
# perform the classification
#===========================================================================
from sklearn.linear_model import LogisticRegression
# we use the default Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
classifier = LogisticRegression(solver='lbfgs')
classifier.fit(X_train, y_train)

#===========================================================================
# use the model to predict 'Survived' for the test data
#===========================================================================
predictions = classifier.predict(final_X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 
                       'Survived': predictions})
output.to_csv('submission.csv', index=False)

