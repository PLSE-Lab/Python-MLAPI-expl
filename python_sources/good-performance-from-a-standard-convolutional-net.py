#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# How well can we classify MNIST with a neural network, without having to think about it very hard or spend a long time experimenting?
# 
# In this notebook, we will set up a typical convolutional net, and spend a while training it. We won't do any hyperparameter tuning.
# 
# We'll see that it performs competitively (> 99.5% accuracy), with no apparent danger of overfitting.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.constraints import max_norm

sns.set(style="white", context="talk")


# # Get the data and prepare it

# In[ ]:


train = pd.read_csv("../input/train.csv")

image_length = 28
image_size = image_length**2

y_train_full = train["label"].values
y_train_full = to_categorical(y_train_full, num_classes=10)
X_train_full = train.drop(labels="label", axis="columns").values
X_train_full = X_train_full.reshape((-1, image_length, image_length, 1))

test = pd.read_csv("../input/test.csv")
X_test = test.values
X_test = X_test.reshape((-1, image_length, image_length, 1))

input_shape = (image_length, image_length, 1)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                  test_size=.1, random_state=0)


# # Set up data augmentation

# We will use basic data augmentation, applying modest linear transformations.

# In[ ]:


datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=.1,
                             height_shift_range=.1,
                             shear_range=10,
                             zoom_range=.1)
datagen.fit(X_train)


# # Create model

# We will use a standard architecture for our model, following established rules of thumb:
# * Short sequences (just 2) of convolutional layers with small (size 3) kernels
# * Padding the convolutions so that the size of the image is not changed
# * Max pooling (size 2) after each sequence of convolutional layers
# * A substantial number (64) of kernels in the first layer, doubling after every pool
# * Once the image is quite small (7 by 7), finishing with a dense final layer with a large number of units (256)
# * Dropout after every pooling layer and after the dense final layer, with probability of 1/2 of dropping out
# * Constrain every weight vector to have a max norm of 3
# * ReLU activations throughout, with a softmax for the output

# In[ ]:


convolution_constraint = max_norm(3, axis=[0, 1, 2])
dense_constraint = max_norm(3)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu", input_shape=input_shape))

model.add(Conv2D(filters=64, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(.5))
model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))

model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(units=256, kernel_constraint=dense_constraint,
                activation="relu"))
model.add(Dropout(.5))
model.add(Dense(units=10, activation='softmax'))


# # Train model
# 
# We will use a standard loss function (categorical cross-entropy).
# 
# We will use the Adam optimiser because it performs well without fine-tuning.
# 
# Note that I used a Kaggle GPU, which makes the process much faster.

# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


# In[ ]:


epochs = 150
batch_size = 378
history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size
)


# # Results

# In[ ]:


epochs = np.array(history.epoch) + 1
train_accuracies = history.history["acc"]
validation_accuracies = history.history["val_acc"]

# Add a trend line for the validation accuracies based on the reciprocal of the epoch number.
regression = LinearRegression()
epoch_features = np.array([1/epochs]).T
trend = regression.fit(epoch_features, validation_accuracies).predict(epoch_features)
asymptote = regression.intercept_

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(epochs, train_accuracies, label="Training")
ax.axhline(asymptote, color="grey", linestyle=":")
ax.plot(epochs, trend, color="grey", linestyle="--")
ax.plot(epochs, validation_accuracies, label="Validation")

ax.legend(loc="lower right")
ax.set_ylim([.96, 1])
ax.set_xlim([0, max(epochs)])
ax.set_title("Learning curve")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epoch")
plt.show()


# In[ ]:


final_train_accuracy = np.mean(train_accuracies[-10:])
final_validation_accuracy = np.mean(validation_accuracies[-10:])
print("Final training accuracy:\t{:.4f}".format(final_train_accuracy))
print("Final validation accuracy:\t{:.4f}".format(final_validation_accuracy))


# The final validation accuracy is around 99.5% to 99.6%.
# 
# Note that the training accuracy is lower than the validation accuracy because dropout is applied during training. I'm impressed that the net still does a good job in this situation, approaching a 99% accuracy when half of its neurons are randomly going missing! It must be learning some robust features.
# 
# The validation accuracy has plateaued and shows no sign of falling due to overfitting -- I experimented with training for a few hundred more epochs and this remained the case, even as the training loss levelled off to its minimum. I noticed that the validation accuracy was almost perfectly fit by a reciprocal trend line, which I've included on the graph as a dashed line, along with a dotted line indicating the asymptote of this trend line. This further corroborates the validation accuracy having plateaued at its final maximum value. It appears that dropout is doing an excellent job of preventing overfitting, and we don't need to worry about training the net too much.
# 
# Finally, we retrain the net using the full training data set, before submitting our predictions. This should boost the final test accuracy a little. At this point we could also make an ensemble of several such nets, randomly initialised; this should boost the final accuracy a little more, approaching a value of 99.6% to 99.7%.

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu", input_shape=input_shape))

model.add(Conv2D(filters=64, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(.5))
model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))

model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                 kernel_constraint=convolution_constraint,
                 activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(units=256, kernel_constraint=dense_constraint,
                activation="relu"))
model.add(Dropout(.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


# In[ ]:


epochs = 150
batch_size = 420
model.fit_generator(
    datagen.flow(X_train_full, y_train_full, batch_size=batch_size),
    epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size
)

predictions = model.predict(X_test).argmax(axis=1)

predictions_count = predictions.size
submission = pd.DataFrame({"Label": predictions, "ImageId": range(1, predictions_count + 1)}, columns=["ImageId", "Label"])
submission.to_csv("submission.csv", index=False)

