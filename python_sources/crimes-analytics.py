import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pylab as plt
import itertools
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

crimes = pd.read_csv('../input/Data.csv',error_bad_lines=False)

crimes.index = pd.DatetimeIndex(crimes.Date)

# separate crime data into training data and test data
crimes_2012_2016 = crimes.loc['2012':'2016']
crimes_2017 = crimes.loc['2017']
print('Number of observations in the training data:', len(crimes_2012_2016))
print('Number of observations in the test data:',len(crimes_2017))

# separate features and output of training data
features_train = crimes_2012_2016[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
# transform to categorical encoding
y = crimes_2012_2016["Location Description Number"]
y_train = keras.utils.to_categorical(y, num_classes=4)
#print(features.head())

# separate features and output of training data
features_test = crimes_2017[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
y_2017 = crimes_2017["Location Description Number"]
y_test = keras.utils.to_categorical(y_2017, num_classes=4)
time_start = time.clock()

#build neural network model structure1#
model = Sequential()
model.add(Dense(128, activation='tanh', input_dim=6))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(np.array(features_train), np.array(y_train),
          epochs=5,
          batch_size=128)
#print(model.evaluate(np.array(features_test), np.array(y_test), batch_size=32, verbose=0))
#validation_data=(np.array(features_test), np.array(y_test))
# score = model.evaluate(features_test, y_test, batch_size=4, verbose=0)
# print(score)
y_pred = model.predict_classes(np.array(features_test), batch_size=32, verbose=0)
import pickle
with open("model.pickle","wb") as f:
    pickle.dump(model,f)
#y_pred = np.array([y_pred])
print(y_pred.shape)
print(collections.Counter(y_pred))
y_pred = np.array([y_pred])
y_pred = y_pred.transpose()

time_elapsed = (time.clock() - time_start)
print("time to build a model is:", time_elapsed)
# print('before to category:', y_pred)
# y_pred = pd.DataFrame(y_pred, index=pd.DatetimeIndex(crimes_2017.Date), dtype='object')
# print(y_pred.head())
# print('after to category:', y_pred)
# aggfunc=','.join
# comparison = pd.crosstab(y_2017, y_pred)
# print(comparison)

#plot confusion matrix of the results#
class_names = [1,2,3]


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
        cm = np.around(cm, decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_2017, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig("plot1.pdf")
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig("plot2.pdf")


# compute classification report
classificationReport = classification_report(y_2017, y_pred, digits=5)
# classificationReport = classificationReport.to_array()
# np.set_printoptions(precision=2)
print(classificationReport)


def plot_classification_report(cm, classes,title='Classification Report',cmap=plt.cm.Blues):
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names2 = [1,2,3,'ave']
plt.figure()
plot_classification_report(classificationReport, classes=class_names2,title='Classification Report')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png')
plt.show()

plt.savefig("plot3.pdf")