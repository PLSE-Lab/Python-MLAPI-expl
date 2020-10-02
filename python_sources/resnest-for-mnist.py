#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
print("tf.__version__: ", tf.__version__)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd # for load MNIST
from sklearn.model_selection import train_test_split # for Validation


# # import ResNeSt Module

# In[ ]:


get_ipython().system('pip install StealthFlow==0.0.9')
from stealthflow.resnest import ResNeStBlock
from stealthflow.tf_layers import MyBlock


# # Set Dataset

# In[ ]:


class MNIST():
    def __init__(self):

        train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
        test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

        X_train = (train.iloc[:,1:].values).reshape(-1, 28, 28, 1).astype(np.float32) / 255.0 
        y_train = train.iloc[:,0].values.astype('int32') #tf.keras.utils.to_categorical()
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        self.X_test = test.values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

mnist = MNIST()
print(mnist.X_train.max(), mnist.X_train.min())
print(mnist.X_train.shape)
print(mnist.y_train.shape)
print(mnist.X_val.shape)
print(mnist.y_val.shape)
print(mnist.X_test.shape)
del mnist


# # Define ResNeSt class

# In[ ]:


class ResNeSt:
    def __init__(self, params):
        self.params = params
        self.myblock = MyBlock()        
        self.build_model()

    def build_model(self):
        self.model = self.define_resnest()
        self.model.compile(loss = self.params.LOSS, optimizer = self.params.OPTIMIZER, metrics = self.params.METRICS)
        tf.keras.utils.plot_model(self.model, to_file='architecture.png', show_shapes=True)

    def define_resnest(self):
        x_in = x = tf.keras.layers.Input(shape=self.params.INPUT_SHAPE)

        x = self.myblock.conv_3x3_batch_relu(x, 16)
        x = ResNeStBlock(radix=1, cardinality=1, bottleneck=16, ratio=4)(x)
        
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = self.myblock.conv_3x3_batch_relu(x, 64)
        x = ResNeStBlock(radix=2, cardinality=2, bottleneck=32, ratio=4)(x)

        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = self.myblock.conv_3x3_batch_relu(x, 128)
        x = ResNeStBlock(radix=2, cardinality=4, bottleneck=64, ratio=4)(x)
        
        y = self.myblock.classification_MLP(x, num_classes=10, num_units=1024, dropout=0.5)
        #y = self.myblock.classification_GAP(x, num_classes=10)
        
        model = tf.keras.Model(inputs=x_in, outputs=y)
        return model


# # Set Parameters

# In[ ]:


class Params():
    def __init__(self):
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.BATCH_SIZE_test = 32

        self.LR_base = 0.05 * (self.BATCH_SIZE / 256)
        self.scheduling_on = [int(self.EPOCHS*0.1), int(self.EPOCHS*0.6), int(self.EPOCHS*0.8)]
        self.scheduling_decay = [0.2, 0.04]
        self.LR_scheduler = tf.keras.callbacks.LearningRateScheduler(self.schedule, verbose=1)
        
        self.EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_sparse_categorical_accuracy', min_delta=0.0001, patience=10, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=True
                                )

        self.OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=self.LR_base, momentum=0.9, nesterov=True, name="SGD_momentum") # and momentum of 0.9,(5.1 Implementation Details)
        self.LOSS = ['sparse_categorical_crossentropy']
        self.METRICS = [tf.keras.metrics.SparseCategoricalAccuracy()]

        self.INPUT_SHAPE = (28, 28, 1)

    def schedule(self, epoch):
        if(epoch < self.scheduling_on[0]):
            return self.LR_base * (epoch+1)/self.scheduling_on[0]
        elif(epoch < self.scheduling_on[1]):
            return self.LR_base
        elif(epoch < self.scheduling_on[2]):
            return self.LR_base * self.scheduling_decay[0]
        else:
            return self.LR_base * self.scheduling_decay[1]


# # Run

# In[ ]:


class Trainer():
    def __init__(self):
        self.mnist = MNIST()
        self.params = Params()
        self.resnest = ResNeSt(self.params)
    
    def train(self):

        self.resnest.model.fit(self.mnist.X_train, self.mnist.y_train
                          , validation_data=(self.mnist.X_val, self.mnist.y_val)
                          , epochs=self.params.EPOCHS
                          , batch_size=self.params.BATCH_SIZE
                          , callbacks=[self.params.LR_scheduler, self.params.EarlyStopping]
                         )
        
        val_loss, val_acc = self.resnest.model.evaluate(self.mnist.X_val, self.mnist.y_val, batch_size=self.params.BATCH_SIZE_test)
        print(val_loss, val_acc)

    def predict(self, output_file):
        
        y_hat = self.resnest.model.predict(self.mnist.X_test, batch_size=self.params.BATCH_SIZE_test)
        y_pred = np.argmax(y_hat,axis=1)
        
        submissions = pd.DataFrame({
                                  "ImageId": list(range(1, len(y_pred)+1))
                                , "Label": y_pred
                                })
        submissions.to_csv("submission.csv", index=False, header=True)


# In[ ]:


trainer = Trainer()
trainer.train()
trainer.predict(output_file="submission.csv")


# In[ ]:




