from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
data = datasets.load_digits()
cf = svm.LinearSVC()
X = data.data
Y = data.target
train_x,test_x,train_y,test_y = train_test_split(X, Y)
cf.fit(train_x,train_y)
print ('The score of the train data: %s'%(str(round(cf.score(train_x,train_y)))))
print ('The score of the test data: %s'%(str(round(cf.score(test_x,test_y)))))
