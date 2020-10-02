import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
#submission = pd.read_csv("../input/sample_submission.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

print (train.head())

labels= train['label'].values

print (len(labels) )

del train['label']

train_features=[]
for i in range(train.shape[0]):
    train_features.append(train.iloc[i].values)
    
print ("train ", len(train_features))

test_features=[]
for i in range(test.shape[0]):
    test_features.append(train.iloc[i].values)
    
print (" test: ", len(test_features))

from skimage.feature import hog
from sklearn.svm import LinearSVC

''' 
list_hog_fd_train= []
for feature in train_features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd_train.append(fd)
    
hog_features_train = np.array(list_hog_fd_train, 'float64')

list_hog_fd_test= []
for feature in test_features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd_test.append(fd)
    
hog_features_test = np.array(list_hog_fd_test, 'float64')



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(hog_features,labels,test_size=0.2)


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(x_train, y_train)

print ("Training Sucessful")

predict = clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, predict)
print (accuracy)
'''


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(train_features, labels)

print ("Training Sucessful")
    
predication= clf.predict(test_features)

print ("Predication Succesussful")
submission= pd.DataFrame(columns=['ImageId','Label'])
submission['ImageId'] = np.arange(1,28001)
submission['Label'] = predication
submission.to_csv('submission_4.csv',index=False)