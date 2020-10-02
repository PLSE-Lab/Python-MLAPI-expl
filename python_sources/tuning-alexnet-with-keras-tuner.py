#!/usr/bin/env python
# coding: utf-8

# ### By doing Hyper Parameter tuning from hands is a very time consuming + tedious task.
# ### So, here comes keras Tuner to save us.It automatically finds the best hyperparameter for your model.
# 
# 

# Keras Tuner: The Keras Tuner is a library that helps you pick the optimal set of hyperparameters for your TensorFlow program.
# 
# Tuning Hyperparameters are very much essisential for any ML development cycle. These are the constant variables that governs the training process. Right choise of hyperparameters leads to early convergence and also increases accuracy of the model.
# 
# It's basically two types:
# 1. Model hyperparameters which influence model selection such as the number and width of hidden layers
# 2. Alogrithm hyperparameters : which influence the speed and quality of the learning algorithm such as the learning rate of optimizer.
# 

# So, in this kernel first we'll be going make a ```baseline_alexnet``` model(Alexnet) . 
# Then we'll train this model on fashion mnist datasets.
# 
# And after that I'll tell you, how to use keras tuner.
# Then we'll be making ```modified_alexnet``` model in which we'll be tuning
# ```Model hyperparameters``` and after that we'll use this model to tune the 
# ```Algorithm hyperparameters```(learning rate).
# 
# And finally we'll compare the results of the ```baseline_alexnet``` and the modified hyper parameter tuned ```modified_alexnet``` .

# Let's first build the ```baseline_alexnet```  and train it on ``` fashion_mnist``` dataset.

# In[ ]:


import tensorflow as tf
import IPython
import tensorflow_datasets as tfds


# In[ ]:


get_ipython().system('pip install -q -U keras-tuner')
import kerastuner as kt


# In[ ]:


# loding the datasets from tensorflow_datasets
train, train_info = tfds.load('fashion_mnist', split='train', 
                              as_supervised=True, 
                              with_info=True)
val, val_info = tfds.load('fashion_mnist', 
                          split='test', 
                          as_supervised=True, 
                          with_info=True)


# In[ ]:


def resize(img, lbl):
  img_size = 96
  return (tf.image.resize(img, [img_size, img_size])/255.) , lbl

train = train.map(resize)
val = val.map(resize)

train = train.batch(32, drop_remainder=True)
val = val.batch(32, drop_remainder=True)     


# In[ ]:


def baseline_alexnet():
     return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# In[ ]:


#training on gpu if available
def try_gpu(i=0): 
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')
device_name = try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)


# In[ ]:


with strategy.scope():
  model = baseline_alexnet()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    
                    callbacks = [callback])


# In[ ]:


# storing the result on baseline_val_acc so that we can use it to compare later.
baseline_val_acc = max(history.history['val_accuracy'])


# In[ ]:


baseline_val_acc


# Now, let's make our  ```modified_alexnet``` function, this function will be used for  model-building. It takes an arguement ```hp``` from which we can sample
# hyperparameter. \
# For example if you want to sample units in a dense layer we can do simply by,
# ```hp.Int('units', min_value=32, max_value=512, step=32)``` here you'll notice that 
# we're using ```hp.Int``` because the units in dense layer are alwayes ```Integers```.  \
# Use ```hp.Int``` where the hyperparameters are integer type and use ```hp.Float``` where the hyperparamers are of Float type. \
# Say, for dropout units ```hp.Float('dropout`, min_values=0, max_value=0.5, step=0.1)```
# 
# This ```hp.Int``` or ```hp.Float``` takes the following arguements:
# 1. ```name``` It's the required arguement, make sure to name them unquely.
# 2. ``` min_value``` It's the minimum value from which the keras tuner starts finding the best hyperparametrs
# 3. ```max_value``` It's the maximum value upto which the keras tuner finds the best 
# hyperparameter.
# 4. ```step``` it's the number of steps skips in order to find the best hyperparameter values.\
# 5.```default``` It's an optional arguement which you can set.
# 
# 

# In[ ]:


def modified_alexnet(hp):
  alexnet = tf.keras.models.Sequential()
  # filter size from 96-256
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_1',
                                                      min_value=96,
                                                       max_value=256,
                                                       default=96,
                                                       step=32),
                                      kernel_size=11, strides=4, activation='relu'))
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  # filter size from 256-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_2',
                                                       min_value=256,
                                                       max_value=512,
                                                       default=256,
                                                       step=32),
                                      kernel_size=5, padding='same', activation='relu'))
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  # filter size from 384-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_3',
                                                       min_value=384,
                                                       max_value=512,
                                                       default=384,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
  # filter size from 384-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_4',
                                                       min_value=384,
                                                       max_value=512,
                                                       default=384,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
   # filter size from 256-512
  alexnet.add(tf.keras.layers.Conv2D(filters = hp.Int(name='conv_block_5',
                                                       min_value=256,
                                                       max_value=512,
                                                       default=256,
                                                       step=32),
                                      kernel_size=3, padding='same', activation='relu'))
  
  alexnet.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
  alexnet.add(tf.keras.layers.Flatten())
  # dense unit from 4096 to 8192
  alexnet.add(tf.keras.layers.Dense(units = hp.Int(name='units_1',
                                                  min_value=4096,
                                                  max_value=8192,
                                                  default=4096,
                                                  step=256),
                                                  activation='relu'))
  # dropout value from 0-0.5
  alexnet.add(tf.keras.layers.Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1, default=0.5)))
  alexnet.add(tf.keras.layers.Dense(units = hp.Int(name='units_2',
                                                   min_value=4096,
                                                    max_value=8192,
                                                    default=4096,
                                                    step=256),
                                                    activation='relu'))
  # dropout value from 0-0.5
  alexnet.add(tf.keras.layers.Dropout(hp.Float('dropout_2', 0, 0.5, step=0.1, default=0.5)))
  alexnet.add(tf.keras.layers.Dense(10, activation='softmax'))
  # choice for the learning rate, i.e 0.01, 0.001, 0.0001 
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
  
  alexnet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                metrics = ['accuracy'])

  return alexnet


# Now, we've to instantiate the tuner and perform hypertuning.\
# Here, we're having for choices(algorithms) for instantiating the tuner.\
# These are, ```RandomSearch```, ```Hyperband```, ```BayesianOptimization```, and  ```sklearn```.\
# For this kernel we're going to use ```Hyperband``` algorithm.
# 
# To instantiate the Hyperband tuner, you must specify the hypermodel, the objective to optimize and the maximum number of epochs to train (max_epochs).
# 

# In[ ]:


tuner = kt.Hyperband(modified_alexnet,
                     objective = 'val_accuracy', 
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                      distribution_strategy=tf.distribute.OneDeviceStrategy(device_name),
                     project_name ='intro_to_kt')    


# In[ ]:


#callback to clear the training output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)


# In[ ]:


tuner.search(train, epochs = 10, validation_data = (val), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


# In[ ]:


# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


# In[ ]:


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
callback = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(train, 
                    epochs=100, 
                    validation_data=val,
                    callbacks = [callback])


# In[ ]:


# storing the result on modified_alexnet_val_acc baseline_val_acc so that we can use it to compare baseline_val_acc.
modified_alexnet_val_acc = max(history.history['val_accuracy'])


# In[ ]:


modified_alexnet_val_acc


# In[ ]:


print("The validation accuracy of the baseline alexnet model is {} VS The validation accuracy of the baseline alexnet model is {}".format(baseline_val_acc, modified_alexnet_val_acc))


# Things you can do beyond this kernel is try to do ```hp.choice('sgd','adam')``` for the optimizer and try out on diffrent DL tasks, say image segmentation.

# PS: If You belive that you've learn anything from this kernel, then please upvote this kerenel and follow me on kaggle.
