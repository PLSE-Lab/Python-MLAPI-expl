import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

x = []
file = 'EEG data.csv'
with open('../input/'+file) as f:
	x = f.readlines()

train = []
test = []
traininput = []
trainoutput = []
testinput = []
testoutput = []

for i, a in enumerate(x):
	if i < len(x) - 102:
		train.append(list(int(b) for b in a.split(',')))
	else:
		test.append(list(int(b) for b in a.split(',')))


for i, a in enumerate(train):
	toappend = []
	for c, b in enumerate(a):
		if 1000 not in a and len(a) == 6:
			if c > 2 and c < 13:
				toappend.append(b)
			elif c == 5:
				trainoutput.append(b)
	if(len(toappend) > 0):
		traininput.append(toappend)

for i, a in enumerate(test):
	toappend = []
	for c, b in enumerate(a):
		if 1000 not in a and len(a) == 6:
			if c > 2 and c < Z:
				toappend.append(b)
			elif c == 5:
				testoutput.append(b)
	if(len(toappend) > 0):
		testinput.append(toappend)

X = np.array(traininput)
y = np.array(trainoutput)

#SVM Classification Training
svm = SVC(C = .5)
svm.fit(X, y)

#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)

#Artificial Neural Network
ann = MLPClassifier(learning_rate = 'adaptive')
ann.fit(X, y)

#K-Nearest Neighbor Classifier
knn = KNeighborsClassifier()
knn.fit(X, y)

correct = [0, 0, 0, 0]
incorrect = [0, 0, 0, 0]
svms = []
gnbs = []
anns = []
knns = []

for i, a in enumerate(testinput):
	if svm.predict([a])[0] == testoutput[i]:
		correct[0] += 1
		svms.append(0)
	else:
		incorrect[0] += 1
		svms.append(1)
	if gnb.predict([a])[0] == testoutput[i]:
		correct[1] += 1
		gnbs.append(0)
	else:
		incorrect[1] += 1
		gnbs.append(1)
	if ann.predict([a])[0] == testoutput[i]:
		correct[2] += 1
		anns.append(0)
	else:
		incorrect[2] += 1
		anns.append(1)
	if knn.predict([a])[0] == testoutput[i]:
		correct[3] += 1
		knns.append(0)
	else:
		incorrect[3] += 1
		knns.append(1)

sums = [0, 0, 0, 0, 0]
tests = [0, 0]

print(correct)
print(incorrect)
print(sums)

#Only about 80% Accurate