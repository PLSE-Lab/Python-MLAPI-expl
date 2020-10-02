import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
COMPONENT_NUM = 50

def eliminateFirstLine(sample_data):
    return sample_data.split("\n")[1:-1]

def eliminateComma(sample_data):
    return [i.split(",") for i in sample_data]

def randomForestClassify(X_train,y_train,X_test):
	#accuracy 0.93857 -- 10 estimators 
	#         0.96003 -- 20 estimators
	#         0.96443 -- 100 estimators
	clf = RFC(n_estimators=100)
	clf = clf.fit(X_train,y_train)
	return clf.predict(X_test)

def SVMSVC(X_train,y_train,X_test):
	clf = SVC()
	clf.fit(X_train,y_train)
	return clf.predict(X_test)

def MLP(X_train,y_train,X_test):
	clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-5,
						solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    	learning_rate_init=.1)
	clf.fit(X_train,y_train)
	return clf.predict(X_test)

def pcaSVC(X_train,y_train,X_test):
	train_label = y_train
	train_data = X_train
	pca = PCA(n_components=COMPONENT_NUM, whiten=True)
	pca.fit(train_data)
	train_data = pca.transform(train_data)
	svc = SVC()
	svc.fit(train_data, train_label)
	test_data = X_test
	test_data = pca.transform(test_data)
	return svc.predict(test_data)

#tensorflow related
## weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

## convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#load csv to numpy arrays
train_sample = open('../input/train.csv').read()
test_sample = open('../input/test.csv').read()
#eliminate first line
train_sample = eliminateFirstLine(train_sample)
test_sample = eliminateFirstLine(test_sample)
#eliminate comma
train_sample = eliminateComma(train_sample)
test_sample = eliminateComma(test_sample)

#transfer string to int
X_train = np.array([[int(i[j]) for j in range(1,len(i))]for i in train_sample])
y_train = np.array([int(i[0]) for i in train_sample])
X_test = np.array([[int(i[j]) for j in range(0,len(i))]for i in test_sample])
#use algorithm to retreive result
result = pcaSVC(X_train,y_train,X_test)
print(result)
print(result.shape)
out_file = open("./prediction.csv","w")
out_file.write("ImageId,Label\n")
enum = 1
for i in result:
    out_file.write(str(enum) + "," + str(i) + "\n")
    enum += 1


