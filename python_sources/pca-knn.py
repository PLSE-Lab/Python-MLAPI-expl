import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# f-1 score: 0.91 for n_train=3000, n_cross=2500, without pca
# f-1 score: 0.91 for n_train=3000, n_cross=2500, with pca
# out of memory with 1GB ram for n_train=7500
# score from kaggle (https://www.kaggle.com/wonjohnchoi/digit-recognizer/knn)
print('Input train, cross, test')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_x = train.ix[:,1:].values.astype('uint8')
train_y = train.ix[:,0].values.astype('uint8')
test_x = test.ix[:,:].values.astype('uint8')

print('PCA reduction')
pca = PCA(n_components=36, whiten=True)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)


print('KNN classifier')
clf = KNeighborsClassifier()
clf.fit(train_x, train_y)

print('Output')
test_y = clf.predict(test_x)
pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('out.csv', index=False, header=True)