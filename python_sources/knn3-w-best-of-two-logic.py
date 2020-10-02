'''
Apache V2.0 dual-licensed with GPL V3
KNN3 with best-of-two logic v1.1
some variables optimized for iris dataset
variables are self-explanatory
'''

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class ScrappyKNN3:
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
			dist=euc(row,self.X_train[i])
			if(dist < best_dist):
				best_dist=dist
				best_index=i
		best_dist_2=euc(row, self.X_train[0])
		best_index_2=0
		for j in range(1, len(self.X_train)):
			dist2=euc(row,self.X_train[i])
			if((dist2 < best_dist_2) and (dist2 <= best_dist) and (j != best_index)):
				best_dist_2 = dist2
				best_index_2 = j
		best_dist_3=euc(row, self.X_train[0])
		best_index_3=0
		for k in range(1, len(self.X_train)):
			dist3=euc(row,self.X_train[i])
			if((dist3 < best_dist_3) and (dist3 <= best_dist_2) and (k != best_index_2)):
				best_dist_3=dist3
				best_index_3=k
		if((self.y_train[best_index]==self.y_train[best_index_2]) or (self.y_train[best_index]==self.y_train[best_index_3])):
			return self.y_train[best_index]
		elif((self.y_train[best_index_2]==self.y_train[best_index_3]) and (best_dist_2 - best_dist < 0.3) and (best_dist_3 - best_dist_2 < 2.0)):
			return self.y_train[best_index_2] # modify the distances according based on dataset
		else:
			return self.y_train[best_index]

from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
y=iris.target
knn=ScrappyKNN3()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

total=0
for i in range(1,11):
	X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.25, random_state=datetime.now().microsecond)
	knn.fit(X_train, y_train)
	predictions=knn.predict(X_test)
	total+=accuracy_score(y_test, predictions)

print(total/10)