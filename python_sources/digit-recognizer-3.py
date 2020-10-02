import pandas as pd
import csv as csv

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

n_samples = train.shape[0]
digits_train = train

# Extract feature (X) and target (y) columns
feature_cols = list(digits_train.columns[1:])  # all columns but first are features
target_col = digits_train.columns[0]  # first column is the target/label

X_train = digits_train[feature_cols]  # feature values 
y_train = digits_train[target_col]  # corresponding targets/labels

# Extract feature (X) columns
feature_cols = list(test.columns[0:])  # all columns are features

X_test = test[feature_cols]  # feature values 

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

pca = PCA(n_components=100)
X_transformed = pca.fit_transform(X_train)

clf = SVC(kernel='poly', degree=2)
clf.fit(X_transformed, y_train)
X_test_transformed = pca.transform(X_test)
pred = clf.predict(X_test_transformed)

prediction_file = open('pred.csv', 'w')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(['ImageId', 'Label'])

for number, i in enumerate(pred):
    prediction_file_object.writerow([number + 1, i])
    
prediction_file.close()








