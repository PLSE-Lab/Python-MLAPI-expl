import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
df = pd.read_csv("../input/train.csv",header=0)
tf  = pd.read_csv("../input/test.csv",header=0)

print("Obtaining train data")
X_train = df.iloc[:,1:].values
X_train = X_train.reshape(X_train.shape[0], 28, 28) #reshape to rectangular
X_train = X_train/255 #pixel values are 0 - 255 - this makes puts them in the range 0 - 1
X_flat = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) #flat the images

print("Obtaining train labels")
y_train = df["label"].values

print("Obtaining test data")
X_test = tf.values
X_test = X_test.reshape(X_test.shape[0], 28, 28) #reshape to rectangular
X_test = X_test/255 #pixel values are 0 - 255 - this makes puts them in the range 0 - 1
X_flat_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) #flat the images

# Import the random forest package
print("Training forest")
forest = RandomForestClassifier(n_estimators = 100) #'n_estimators': 210, 'criterion': 'gini', 'max_depth': 4
forest = forest.fit(X_flat,y_train)
prediction = forest.predict(X_flat_test)

#Export the results to an output file
ids = range(1, len(prediction)+1)
submission = pd.DataFrame({ "ImageId": ids, "Label": prediction })
submission.to_csv('output.csv', index=False)
print("File generated ")