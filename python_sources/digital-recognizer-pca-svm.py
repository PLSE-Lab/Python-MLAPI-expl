import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# Extract the train set and the test set:
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)

# Separate samples and labels:
label_train = train.label.values
sample_train = train.drop(["label"], axis = 1).values
sample_test = test.values

# Do PCA and reduce dimension to 50:
dimension = 50
pca = PCA(n_components = dimension)
sample_train_pca = pca.fit_transform(sample_train)
sample_test_pca = pca.transform(sample_test)

# Normalize the data:
sc = StandardScaler()
sample_train_pca_norm = sc.fit_transform(sample_train_pca)
sample_test_pca_norm = sc.transform(sample_test_pca)

# Use svm:
clf = svm.SVC()
clf.fit(X = sample_train_pca_norm, y = label_train, sample_weight = None)
label_predict = clf.predict(sample_test_pca_norm)

# Print out the result and write to the result csv file:
print("The PCA+SVM results give us: {}".format(label_predict))
prediction_file = pd.DataFrame({"ImageId": range(1, len(label_predict) + 1), "Label": label_predict})
prediction_file.to_csv("prediction.csv", index = False, header = True)

