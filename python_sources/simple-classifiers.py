#Learner for predicting voice
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.linear_model import Perceptron, SGDClassifier

def sex(x):
    return {
		'"male"': 1,
		'"female"': 2,
    }.get(x, 0)

x = []
y = []
with open("../input/voice.csv") as textFile:
	data = textFile.read().splitlines()
	data = data[1:]
	for line in data:
		#print line
		arr = line.split(',')
		arr[20] = sex(arr[20])
		for i in range(20):
			arr[i] = float(arr[i])
		y.append(arr[-1])
		arr.pop()
		x.append(arr)
		#print arr
		
textFile.close()

x = np.asarray(x)

x_train = x[:-630]
y_train = y[:-630]
x_test = x[-630:]
y_test = y[-630:]


clf1 = MultinomialNB()
clf1.fit(x_train,y_train)
res1 = clf1.predict(x_test)
diff1 = 0
for i in range(len(res1)):
	diff1 += abs(res1[i] - y_test[i])
print (1 - float(diff1)/len(y_test))

clf2 = SGDClassifier(penalty='l2')
clf2.fit(x_train,y_train)
res2 = clf2.predict(x_test)
diff2 = 0
for i in range(len(res2)):
	diff2 += abs(res2[i] - y_test[i])
print (1 - float(diff2)/len(y_test))

clf3 = linear_model.LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
clf3.fit(x_train,y_train)
res3 = clf3.predict(x_test)
diff3 = 0
for i in range(len(res3)):
	diff3 += abs(res3[i] - y_test[i])
print (1 - float(diff3)/len(y_test))

clf4 = Perceptron(penalty='l1', n_iter=3)
clf4.fit(x_train,y_train)
res4 = clf4.predict(x_test)
diff4 = 0
for i in range(len(res4)):
	diff4 += abs(res4[i] - y_test[i])
print (1 - float(diff4)/len(y_test))
