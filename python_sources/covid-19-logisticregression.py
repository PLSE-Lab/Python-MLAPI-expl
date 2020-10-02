#importing all packages

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# importing our database
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.describe()
# It seems that a lot of data is missing, so let's choose our variable based on quality and quantity relevance.


# Separating only the features that we will use
df1 = df.iloc[:,:20]


# As the lines that has hematocrit as an not null value are the most complete in variables, we will use it to define our database.

df1 = df1[df1.Hematocrit.notnull()]


# We will also fill with 0 the fields that had no value. As they are not many, it will not impact our model.

df1f = df1.fillna(0)


# Creating a column named "testresult" with the coronavirus results
df1f['testresult'] = df['SARS-Cov-2 exam result']

# Creating binary dummies for that column.

df1f['testresult'] = df1f.testresult.map({'negative': 0, 'positive': 1 })


#Dropping the columns that will not be good features because of their irrelevance to the analysis at hand
df_drop = df1f.drop(["Patient ID",'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)',
    'SARS-Cov-2 exam result'
], axis=1)


# Making datasets to train and test

y = df_drop['testresult']
drop = df_drop.drop('testresult', 1)
X = drop


# Separating datasets for the execution
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# Fitting for prediction method
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# Applying Logistic Regression Method to the Test dataset
y_pred=logreg.predict(X_test)


# Printing Accuracy, precision and Recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

