# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from timeit import default_timer as timer
from multiprocessing.dummy import Pool as ThreadPool

import os
cwd = os.getcwd()
print(cwd)


start= timer()

labeled_images = pd.read_csv('../input'+'/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
X=images
y=labels

#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.9,test_size=0.1, random_state=21)#0.945238095238


# Create feature and target arrays
'''X = digits.data
y = digits.target'''


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size = 0.2, random_state=42, stratify= y)


X_test[X_test>0]=1
X_train[X_train>0]=1

y_train= y_train.values.ravel()
y_test= y_test.values.ravel()

# Create a k-NN classifier with 7 neighbors: knn
'''knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))'''


n_neighbors= 9
neighbors = np.arange(1, n_neighbors)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

def fetch_optimum_accuracy(k):

		knn = KNeighborsClassifier(n_neighbors=k,n_jobs=10)

		# Fit the classifier to the training data
		knn.fit(X_train,y_train)

		#Compute accuracy on the training set
		train_accuracy[k-1]=(knn.score(X_train, y_train))

		#Compute accuracy on the testing set
		test_accuracy[k-1]=(knn.score(X_test, y_test))


# Setup arrays to store train and test accuracies

pool= ThreadPool(2)

pool.map(fetch_optimum_accuracy,neighbors)

pool.close()
pool.join()




end= timer()

print(end-start)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('accuracy.png')