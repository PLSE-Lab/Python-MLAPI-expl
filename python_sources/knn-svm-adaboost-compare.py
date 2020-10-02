import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

'''R#######################################################################'''
'''U#######################################################################'''
'''N#######################################################################'''
import numpy as np
import pandas as pd
from time import time

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
## init
start = time()
# import
train_data = train
test_data  = test
# split
trainData = train_data.drop('label', 1)
trainLabel = train_data[['label']]
header = trainData.columns
X_train, X_test, y_train, y_test = \
    train_test_split(trainData, trainLabel, test_size=0.30, random_state=1)
# norm
norm = Normalizer().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)
testData = norm.transform(test_data)
# prepare
X_train = pd.DataFrame(X_train, columns = header)
X_test = pd.DataFrame(X_test, columns = header)
testData = pd.DataFrame(testData, columns = header)
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()
# end init
end = time()
print("\n***Loading Done: %.2f ***\n" % (end-start))
########################## train & test ##########################
start = time()

# dim reduce
component = 30 #30/40/50
pca = PCA(n_components=component).fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
testData = pca.transform(testData)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
testData = pd.DataFrame(testData)
print("PCA - component : {}".format(component))

## train
neighbors = 5
CLF1 = KNeighborsClassifier(n_neighbors=neighbors).fit(X_train, np.ravel(y_train))
print('CLF | KNN-neighbors : {}'.format(neighbors))

penalty_C = 10.0
CLF2 = SVC(C=penalty_C, gamma=0.1, kernel='rbf').fit(X_train, np.ravel(y_train))
print('CLF | SVM-penalty_C : {}'.format(penalty_C))

#max_depth = 15
#CLF3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
#                         n_estimators=1000, learning_rate=1.0, 
#                         algorithm="SAMME.R").fit(X_train, np.ravel(y_train))
#print('CLF | Adaboost-max_depth : {}'.format(max_depth))

## test
print("\n---CLF1---")
print("ACC: %f.4" % CLF1.score(X_test, y_test))
print("\n---CLF2---")
print("ACC: %f.4" % CLF2.score(X_test, y_test))
#print("\n---CLF3---")
#print("ACC: %f.4" % CLF3.score(X_test, y_test))

end = time()
print("\n***Training Done: %.2f ***\n" % (end-start))
########################## submission ##########################
predLabel1 = CLF1.predict(testData)
predLabel2 = CLF2.predict(testData)
#predLabel3 = CLF3.predict(testData)

submission = pd.DataFrame({"ImageId": np.arange(1, predLabel1.shape[0] + 1),
                           "Label": predLabel1})
submission.to_csv("submission1_KNN.csv", index=False)

submission = pd.DataFrame({"ImageId": np.arange(1, predLabel2.shape[0] + 1),
                           "Label": predLabel2})
submission.to_csv("submission2_SVM.csv", index=False)

#submission = pd.DataFrame({"ImageId": np.arange(1, predLabel3.shape[0] + 1),
#                           "Label": predLabel3})
#submission.to_csv("submission3_ADA.csv", index=False)