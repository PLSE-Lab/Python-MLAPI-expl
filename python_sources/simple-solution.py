import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,confusion_matrix ,classification_report
cactus_train = pd.read_csv('..input/cactus/train_RGB.csv')
cactus_train =cactus_train.drop(['id'],axis =1)
cactus_test = pd.read_csv('..input/cactus/test_RGB.csv')

cactus_test =cactus_test.drop(['id'],axis =1)


X_train, X_test, y_train, y_test = train_test_split(cactus_train, cactus_train['has_cactus'], test_size=0.33) 

from sklearn.neighbors import KNeighborsClassifier
start = time.time()



knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train) 
score = knn.score(X_test, y_test)

end = time.time()
print("The run time of KNN is {:.3f} seconds".format(end-start))
print("KNN alogirthm's test score is: {:.3f}".format(score))

##clf = LinearSVC(multi_class = 'ovr')
##clf.fit(X_train,y_train)
##
##y_p = clf.predict(X_test)
##i = accuracy_score(y_test,y_p)
##endi = time.time()
##print("The run time of KNN is {:.3f} seconds".format(endi-start))
##print("SVM alogirthm's test score is: {:.3f}".format(i))



from sklearn.svm import LinearSVC

start = time.time()
linearKernel = LinearSVC().fit(X_train, y_train)
score = linearKernel.score(X_test,y_test)
end = time.time()

print("The run time of Linear SVC is {:.3f} seconds".format(end-start))
print("Linear SCV alogirthm's test score is: {:.3f}".format(score))


##scaler = preprocessing.StandardScaler()
##scaler.fit(X_train_flatten)
##X_test_normalized = scaler.transform(X_test_flatten)
##X_train_normalized = scaler.transform(X_train_flatten)
##test_flatten = imageToFeatureVector(test_img)
##test_normalized = scaler.transform(test_flatten)
##linearKernel = LinearSVC().fit(X_train_normalized, y_train)
##predictions = linearKernel.predict(test_normalized)
##sample['has_cactus'] = predictions
##sample.head()

predictions = linearKernel.predict(cactus_test)
cactus_test['has_cactus'] = predictions

sub = pd.read_csv('..input/cactus/sample_submission.csv')

sub.to_csv("samplesubmission.csv", index = False)
