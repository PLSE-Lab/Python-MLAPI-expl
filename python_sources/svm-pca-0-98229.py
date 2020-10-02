import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA

dataset = pd.read_csv('train.csv')
label = dataset[[0]].values.ravel()
traindata = dataset.ix[:,1:].values
test = pd.read_csv('test.csv')
#pca
pca = PCA(n_components=50,whiten=True)
pca.fit(traindata)
traindata_pca = pca.transform(traindata)
test_pca = pca.transform(test)
#svc
svc = SVC()
svc.fit(traindata_pca,label)
pred = svc.predict(test_pca)

np.savetxt('submission_SVM.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments='',fmt='%d')

#accuracy = 0.98229