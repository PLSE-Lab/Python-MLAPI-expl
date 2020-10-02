from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn import datasets 
from sklearn import svm
import time
data_iris = datasets.load_iris() 

X_train, X_test, y_train, y_test = train_test_split(data_iris.data, data_iris.target, test_size=0.3, random_state=0) 
start_training_time = time.time()
#Using SVM classifier 
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) 
end_training_time = time.time()
print("Score of Testing Data Set (Metric) : ",clf.score(X_test, y_test)) #Using testing data set

scores = cross_val_score(clf, data_iris.data, data_iris.target, cv=5) 
print('Training Time : ', end_training_time-start_training_time)
print("Accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))