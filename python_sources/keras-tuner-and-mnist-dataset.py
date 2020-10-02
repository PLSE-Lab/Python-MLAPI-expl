#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install keras-tuner


# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#make float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# input image dimensions
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)
print(input_shape)


# In[ ]:


NUM_CLASSES = 10
INPUT_SHAPE = input_shape


# In[ ]:


from kerastuner import HyperModel
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)

class MNIST_CNN_HyperModel(HyperModel):
  def __init__(self, input_shape, num_classes):
    self.input_shape = INPUT_SHAPE
    self.num_classes = NUM_CLASSES
  
  def build(self,hp):
    model = keras.Sequential()
    model.add(
        Conv2D(
            filters = 32,
            kernel_size = 3,
            activation = 'relu',
            input_shape = self.input_shape
        )
    )

    model.add(
        Conv2D(
            filters = 64,
            kernel_size = 3,
            activation = 'relu'
        )
    )

    model.add(MaxPooling2D(pool_size = 2))

    model.add(
        Dropout(rate = hp.Float(
            'dropout_1',
            min_value = 0.0,
            max_value = 0.5,
            default = 0.25,
            step = 0.05
        ))
    )

    model.add(Flatten())

    model.add(
        Dense(
            units = hp.Int(
                'units',
                min_value = 32,
                max_value = 512,
                step = 32,
                default = 128
            ),
            activation = hp.Choice(
                'dense_activation',
                values = ['relu','tanh','sigmoid'],
                default = 'relu'
            )
        )
    )

    model.add(
        Dropout(
            rate = hp.Float(
                'dropout_3',
                min_value = 0.25,
                max_value = 0.75,
                default = 0.5,
                step = 0.05
            )
        )
    )

    model.add(
        Dense(
            self.num_classes,
            activation = 'softmax'
        )
    )

    model.compile(
        optimizer = keras.optimizers.Adadelta(
            hp.Float(
                'learning_rate',
                min_value = 1e-4,
                max_value = 1e-2,
                sampling = 'LOG',
                default = 1e-3
            )
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model


# In[ ]:


hypermodel = MNIST_CNN_HyperModel(input_shape = INPUT_SHAPE, num_classes=NUM_CLASSES)


# In[ ]:


HYPERBAND_MAX_EPOCHS = 20
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
SEED = 4
INPUT_SHAPE = (28,28,1)
NUM_CLASSES = 10


# In[ ]:


# Choosing Tuner from Keras Tuner 
from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    seed=SEED,
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTION_PER_TRIAL
)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


N_EPOCH_SEARCH = 20

tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_data = (x_test,y_test))


# In[ ]:


# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model)

# Evaluate the best model.
loss, accuracy = best_model.evaluate(x_test, y_test)
print(loss)
print(accuracy)


# In[ ]:


from kerastuner.tuners import Hyperband


tuner = Hyperband(
    hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_accuracy',
    seed=SEED,
    executions_per_trial=EXECUTION_PER_TRIAL,
    project_name = 'MNIST')


# In[ ]:


N_EPOCH_SEARCH = 20

tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_data = (x_test,y_test))


# In[ ]:


# Show a summary of the search
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model)

# Evaluate the best model.
loss, accuracy = best_model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

