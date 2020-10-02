import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")


train_x = train.values[:,1:]
train_y = train.ix[:,0]
test_x = test.values

pca = PCA(n_components=0.85,whiten=True)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


svc = svm.NuSVC()
svc.fit(train_x, train_y)

test_y = svc.predict(test_x)

svc = svm.NuSVC(probability=True)
svc.fit(train_x, train_y)

test_y_proba = svc.predict_proba(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)
np.save('test_y_proba', test_y_proba)