import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

mnist = pd.read_csv(os.path.join("../input/", "train.csv"))
print("Dataset size =", mnist.shape)

# Prepare features(X) and target(y)
X = mnist.drop("label", axis=1)
y = mnist["label"]

# Split dataset into train and test set
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(mnist, test_size=0.2, random_state=42)
print("Length of train_set =", train_set.shape[0], "AND test_set =", test_set.shape[0])

# Now shuffle the train dataset
split_idx = len(train_set)
shuffle_index = np.random.permutation(split_idx)
train_set = train_set.iloc[shuffle_index]
print("Length of shuffled train_set =", train_set.shape[0])

# Extract features(X) and target(y) from train and test set
X_train = train_set.drop("label", axis=1)
y_train = train_set["label"]

X_test = test_set.drop("label", axis=1)
y_test = test_set["label"]

# Check CV accuracy
from sklearn.model_selection import cross_val_score
def checkCrossValAccuracy(model, X=X_train, y=y_train, cv=3, scoring="accuracy"):
    cv_score = cross_val_score(model, X, y, cv, scoring)
    print("CV Accuracy =", cv_score)
    return cv_score


# Check test accuracy
def checkTestAccuracy(model, X=X_test, y=y_test):
    y_test_pred = model.predict(X)
    test_accuracy = np.mean((y_test_pred == y)) * 100
    print("Test Accuracy % =", test_accuracy)
    return test_accuracy

##################################################################################
# Add new code to test after this #
##################################################################################
start_time = time.time()
def shiftImage(X, shift_by=5, shift_dir='right'):
    X = X.reshape(28,-1)
    X_shifted = np.zeros(X.shape)
    if shift_dir == 'right':
        X_shifted[:, shift_by:] = X[:,:-shift_by]
    elif shift_dir == "left":
        X_shifted[:, :-shift_by] = X[:,shift_by:]
    X_shifted = X_shifted.ravel()
    return X_shifted

X_shifted_right, X_shifted_left = np.zeros(X_train.shape), np.zeros(X_train.shape)
#for idx, X in enumerate(X_train.values):
#    X_shifted_right[idx] = shiftImage(X, shift_dir='right')
#   X_shifted_left[idx] = shiftImage(X, shift_dir='left')

X_train_augmented = np.r_[X_train, X_shifted_right, X_shifted_left]
y_train_augmented = np.r_[y_train, y_train, y_train]

print("Shape of augmented data =", X_train_augmented.shape, y_train_augmented.shape)

start_time = time.time()

# Standardize data
#X_train = X_train/255
#X_test = X_test/255

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=4, weights='distance') 
print("Training ....")
knn_clf.fit(X, y)

#forest_cv_score = checkCrossValAccuracy(forest_clf, X=X_train_augmented, y=y_train_augmented)
print("Predicting on train data ....")
knn_train_score = checkTestAccuracy(knn_clf, X=X_train, y=y_train)

print("Predicting on test data ....")
knn_test_score = checkTestAccuracy(knn_clf)

print("Execution time in Seconds =", time.time() - start_time)

# Evaluate feature importance
#feature_importance = forest_clf.feature_importances_
#print("Feature importance =", forest_clf.feature_importances_)
#n_px = int(np.sqrt(X_train.shape[1]))
#print("Number of pixels =", n_px)
#feature_importance = feature_importance.reshape(n_px, -1)
#fig = plt.figure(figsize=(8,8))
#plt.imshow(feature_importance, cmap=plt.cm.binary)
#plt.show()

##################################################################################
# Test on Kaggle test dataset and prepare submission file
##################################################################################
mnist_test = pd.read_csv(os.path.join("../input/", "test.csv"))

test_data_m = mnist_test.shape[0]
submission_out_path = os.path.join(".", 
                                   "Submission_01Feb19_knn_clf.csv")

knn_test_pred = knn_clf.predict(mnist_test)

my_submission = pd.DataFrame({ \
            'ImageId': np.arange(1, test_data_m+1), \
            'Label': knn_test_pred})
my_submission.to_csv(submission_out_path, index=False)