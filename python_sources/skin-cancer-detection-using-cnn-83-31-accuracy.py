#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


print(os.listdir('../input/data/'))


# In[ ]:


folder_benign_train = '../input/data/train/benign'
folder_malignant_train = '../input/data/train/malignant'

folder_benign_test = '../input/data/test/benign'
folder_malignant_test = '../input/data/test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load in training pictures 
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')

# Load in testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])


# Merge data 
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]


# In[ ]:


print(X_test[1])
plt.imshow(X_test[1], interpolation='nearest')
plt.show()


# In[ ]:


X_train = X_train/255
X_test = X_test/255
print(X_test[1])


# In[ ]:


model = SVC()
model.fit(X_train.reshape(X_train.shape[0],-1), y_train)


# In[ ]:


y_pred = model.predict(X_test.reshape(X_test.shape[0],-1))


# In[ ]:


print(accuracy_score(y_test, y_pred))


# In[ ]:


logreg = LogisticRegression(C= 1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
logreg.fit(X_train.reshape(X_train.shape[0],-1), y_train)


# In[ ]:


y_pred = logreg.predict(X_test.reshape(X_test.shape[0],-1))


# In[ ]:


print(accuracy_score(y_test, y_pred))


# In[ ]:


import tensorflow as tf
X_Train = tf.keras.utils.normalize(X_train)
X_Test = tf.keras.utils.normalize(X_test)


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128,(3,3), input_shape = X_Train.shape[1:] ,activation = tf.nn.relu ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=None))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))


# In[ ]:


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


X_Train.shape


# In[ ]:


model.fit(X_Train, y_train, epochs = 20, verbose=2, batch_size=32, validation_split = 0.1)


# In[ ]:


y_pred = model.predict(X_Test)


# In[ ]:


yp =[]
for i in range(0,660):
    if y_pred[i][0] >= 0.5:
        yp.append(0)
    else:
        yp.append(1)


# In[ ]:


print(accuracy_score(y_test, yp))


# In[ ]:


#Model 2 starts from here


# In[ ]:


import os
import numpy as np 
import pandas as pd 
import itertools
import tensorflow as tf 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


train_path = '../input/data/train/' 
test_path =  '../input/data/test'


# In[ ]:


generate = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[ ]:


train_data = generate.flow_from_directory( train_path,
                                             target_size = (200, 200),
                                             batch_size = 2637,
                                             classes = ["malignant","benign"],
                                             class_mode = "binary")
test_data = generate.flow_from_directory( test_path,
                                             target_size = (200, 200),
                                             batch_size = 660,
                                             classes = ["malignant","benign"],
                                             class_mode = 'binary')


# In[ ]:


X_train,y_train = train_data.next()
X_test,y_test = test_data.next()


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=(200,200,3), activation=tf.nn.relu, padding="valid"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))

model.add(tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, padding="same"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))

model.add(tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, padding="same"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.30))

model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))


# In[ ]:


model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


Model = model.fit(X_train, y_train, epochs = 10, verbose=2, batch_size=32, validation_split = 0.1)


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


plt.plot(Model.history['acc'])
plt.plot(Model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(Model.history['val_loss'])
plt.plot(Model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred,axis = 1) 
#y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(y_test, y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(2)) 

