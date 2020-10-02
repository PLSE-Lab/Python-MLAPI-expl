#!/usr/bin/env python
# coding: utf-8

# **Each training and test example is assigned to one of the following labels:**
# 
# * 0 T-shirt/top
# * 1 Trouser
# * 2 Pullover
# * 3 Dress
# * 4 Coat
# * 5 Sandal
# * 6 Shirt
# * 7 Sneaker
# * 8 Bag
# * 9 Ankle boot

# In[ ]:


# Import librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Load the data
train_dataset = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_dataset = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


# In[ ]:


# Read data
X_train = train_dataset.iloc[:, 1:].values.astype("float32")
Y_train = train_dataset.iloc[:, 0].values.astype("int32")
X_test = test_dataset.iloc[:, 1:].values.astype("float32")
Y_test = test_dataset.iloc[:, 0].values.astype("int32")


# In[ ]:


# Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# In[ ]:


# Feature Scaling
X_train /= 255.0
X_test /= 255.0


# In[ ]:


# One hot encoding
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)
Y_test = to_categorical(Y_test, num_classes = 10)


# In[ ]:


# Divide the dataset into training and validation
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)


# In[ ]:


# Setting hyperparemeters
epochs = 30
batch_size = 64
input_dim = (28, 28, 1)


# In[ ]:


# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range = 10,  
                             zoom_range = 0.1, 
                             width_shift_range = 0.1, 
                             height_shift_range = 0.1,  
                             horizontal_flip = False, 
                             vertical_flip = False)

datagen.fit(X_train)


# In[ ]:


# Learning Rate Decay
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001)


# In[ ]:


# Build the Model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    Conv2D(32, (5, 5), padding = 'same', activation = 'relu', input_shape = input_dim),
    Conv2D(32, (5, 5), padding = 'same', activation = 'relu'),
    MaxPool2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    MaxPool2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation = 'relu'),
    Dropout(0.25),
    Dense(10, activation = 'softmax')
])


# In[ ]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])


# In[ ]:


model.summary()


# In[ ]:


# Without Data Augmentation
history = model.fit(X_train, 
                    Y_train, 
                    epochs = epochs, 
                    batch_size = batch_size, 
                    validation_data = (X_val, Y_val))


# In[ ]:


# With data augmentation on training set
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),
                              epochs = epochs,
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              validation_data = (X_val, Y_val), 
                              callbacks = [learning_rate_reduction])


# In[ ]:


# Evaluating the model
score = model.evaluate(X_test, Y_test)
print('Loss: {:.4f}'.format(score[0]))
print('Accuracy: {:.4f}'.format(score[1]))


# In[ ]:


# Plotting the learning graph
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label = "Training Accuracy")
plt.plot(epochs, val_acc, "b", label = "Validation Accuracy")
plt.legend(loc='upper right')
plt.title("Training and Validation Accuracy")
plt.figure()

plt.plot(epochs, loss, "r", label = "Training Loss")
plt.plot(epochs, val_loss, "b", label = "Validation Loss")
plt.legend(loc='upper right')
plt.title("Training and Validation Loss")
plt.figure()


# In[ ]:


# Submitting Predictions to Kaggle
preds = model.predict_classes(X_test)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "cnn_mnist_datagen.csv")


# In[ ]:


# Look at confusion matrix 
from sklearn.metrics import confusion_matrix
import itertools

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

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# In[ ]:


# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# In[ ]:




