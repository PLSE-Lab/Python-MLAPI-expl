from sklearn import svm
from sklearn.externals import joblib
from sklearn import svm, grid_search
import numpy as np

from sklearn.externals import joblib

dataset = np.loadtxt(open('../input/train.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)
testset = np.loadtxt(open('../input/test.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)

joblib.dump(dataset, 'training_set.pkl')
joblib.dump(testset, 'test_set.pkl')

dataset = joblib.load('training_set.pkl')
testset = joblib.load('test_set.pkl')

target = [x[0] for x in dataset[:42000]]
train = [x[1:] for x in dataset[:42000]]
test = [x[0:] for x in testset[:28000]]

GridParam = {'kernel': ['rbf'], 'gamma': [1e-4],'C': [1]}
model_init = svm.SVC(max_iter = 100)
model = grid_search.GridSearchCV(model_init, GridParam, n_jobs = -1) 
model.fit(train, target)

result = model.predict(test)

np.savetxt('Submit_SVM.csv', result, delimiter=',', comments = '', header = 'ImageId,Label', fmt='%d')