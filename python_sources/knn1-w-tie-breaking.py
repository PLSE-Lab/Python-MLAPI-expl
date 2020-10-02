''' 
Apache V2.0 Dual-licensed with GPL V3
KNN1 with 1-level tie-break v2.0
variables are self-explanatory
'''

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class ScrappyKNN:
	def fit(self, X_train, y_train):
		self.X_train=X_train
		self.y_train=y_train

	def predict(self, X_test):
		prediction = []
		for row in X_test:
			prediction.append(self.closest(row))
		return prediction

	def closest(self, row):
		best_dist=euc(row, self.X_train[0])
		best_index=0
		for i in range(1, len(self.X_train)):
			dist = euc(row,self.X_train[i])
			if(dist < best_dist):
				best_dist = dist
				best_index = i
			if(dist == best_dist):
				second_best_dist=euc(row, self.X_train[0])
				second_best_index=0
				for j in range(1, len(self.X_train)):
					second_dist = euc(row,self.X_train[j])
					if((second_dist < second_best_dist) and (second_dist < best_dist)):
						second_best_dist = second_dist
						second_best_index = second_best_index
				if(euc(self.X_train[second_best_index], self.X_train[best_index]) > euc(self.X_train[i],self.X_train[second_best_index])):
					best_dist = dist
					best_index = i
		return self.y_train[best_index]

from sklearn import datasets

iris = datasets.load_iris()

X=iris.data
y=iris.target
knn = ScrappyKNN()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

total=0
for i in range(1,11):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=datetime.now().microsecond)
	knn.fit(X_train, y_train)
	predictions = knn.predict(X_test)
	total+=accuracy_score(y_test, predictions)

print(total/10)