"""
Created on Wed Oct  4 19:55:27 2017

@author: bruceokallau
"""

###############
## WARNING!!!##
###############
# This algorithim is computationaly costly, each fit and predict function may take 30+ minutes.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

#############
print( "\nReading data from disk ...")
#############
dig_its = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#############
print( "\nSplit data sets to train/validation ..." )
#############
train, holdout = train_test_split(dig_its, test_size = 0.15, random_state = 1234)

y_weight = holdout['label']
x_weight = holdout.drop(['label'],axis=1)

y_train = train['label']
x_train = train.drop(['label'],axis=1)

#############
print( "\nScale data ..." )
#############
x_weight = preprocessing.scale(x_weight)
x_train = preprocessing.scale(x_train)
test = preprocessing.scale(test)

#############
print( "\nCreate and fit KNN ..." )
# init knn model
#############
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
#############
# fit the model
#############
knn.fit(x_train, y_train)

############
print( "\nPredict validation set ..." )
############
pred_valid = knn.predict(x_weight)
# for ensembling use knn_valid_probs = knn.predict_proba(x_weight)

##########
print( "\nScore validation predictions ..." )
##########
incorrect = np.asarray(np.where(pred_valid != y_weight))
print((6300-incorrect.size)/6300)
#0.9385714285714286 not bad for first try!

############################
# Look at confusion matrix## 
############################

# code from Yassine Ghouza "Introduction to CNN Keras - 0.997 (top 6%)" kernel https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# compute the confusion matrix
confusion_mtx = confusion_matrix(y_weight, pred_valid) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

# model performed best on 1's, 3's, and 7's, worst on 8's and 5's
# if we could find models that perform better on 8's and 5's we could ensemble them

######################################
print( "\nPredicting test set ..." )##
######################################
test_pred = knn.predict(test)

#write submission
results = pd.Series(test_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

print( submission.head() )

###########
print( "\nWriting results to disk ..." )
###########
submission.to_csv('knn_1a{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")

##########
#version tracking with leaderboard score
##########
#v1a knn with 5nn and default settings -- 0.93814