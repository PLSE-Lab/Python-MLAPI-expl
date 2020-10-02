import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Get Train and Test data. 
train_x = train.values[:,1:]
train_y = train.ix[:,0]
test_x = test.values

# Train PCA
pca = PCA(n_components=0.99,whiten=True)
train_x = pca.fit_transform(train_x)

# Train SVM
svc = svm.SVC(kernel='rbf',C=2)
svc.fit(train_x, train_y)

# Predict class:
test_y = svc.predict(pca.transform(test_x))

pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)
                