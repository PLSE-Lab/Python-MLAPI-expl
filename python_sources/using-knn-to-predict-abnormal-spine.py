import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing, cross_validation, neighbors #(KNN Validation)

# import the file and check output
df = pd.read_csv('../input/Dataset_spine.csv')

# Only picking data related columns and 
# and converting strings to numerics for our class column.
df['Class_Ord'] = pd.Categorical(df.Class_att).codes 

X = df[['Col1',
'Col2',
'Col3',
'Col4',
'Col5',
'Col6',
'Col7',
'Col8',
'Col9',
'Col10',
'Col11',
'Col12',
'Class_Ord'
]] 



# dropping the class column as we don't want the distance measuring for the class
# and then filling in the numpy arrays
X_mod = np.array(X.drop(['Class_Ord'], 1))
y = np.array(df['Class_Ord'])

# creating test and training datasets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_mod, y, test_size = 0.2)

#preparing the classifier
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

#calculating the predict accuracy and printing it
predict_accuracy = clf.score(X_test, y_test)
print(predict_accuracy)