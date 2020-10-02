#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils, nn
from sklearn.model_selection import train_test_split
import time


# In[ ]:


Dig_MNIST = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

Dig_MNIST.shape, train.shape, test.shape


# In[ ]:


X = train.iloc[:, 1:].values.astype('float32') / 255.0
y = train['label'].values.ravel()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)
X_train = nd.array(X_train).reshape((-1, 1, 28, 28))
X_valid = nd.array(X_valid).reshape((-1, 1, 28, 28))

X_train_val = nd.array(X).reshape((-1, 1, 28, 28))
y_train_val = y

batch_size = 100
train_data = mx.io.NDArrayIter(X_train, y_train, batch_size=batch_size, shuffle=True)
val_data = mx.io.NDArrayIter(X_valid, y_valid, batch_size=batch_size)
train_val_data = mx.io.NDArrayIter(X_train_val, y_train_val, batch_size=batch_size, shuffle=True)


# In[ ]:


net = nn.Sequential()
net.add(
    # first
    nn.Conv2D(64, kernel_size=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.Conv2D(64, kernel_size=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.Conv2D(64, kernel_size=5, padding=2),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.MaxPool2D(pool_size=(2,2)),
    nn.Dropout(0.2),
    # second
    nn.Conv2D(128, kernel_size=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.Conv2D(128, kernel_size=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.Conv2D(128, kernel_size=5, padding=2),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.MaxPool2D(pool_size=(2,2)),
    nn.Dropout(0.2),
    # third
    nn.Conv2D(256, kernel_size=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.MaxPool2D(pool_size=(2,2)),
    nn.Dropout(0.2),
    # fourth
    nn.Flatten(),
    nn.Dense(256),
    nn.BatchNorm(),
    nn.Dense(128),
    nn.BatchNorm(),
    nn.Dense(10, activation='sigmoid')
)

# set the context on GPU is available otherwise CPU
ctx = [mx.gpu(0)]
net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': 0.001})

def evaluate_accuracy(val_data, net, ctx):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    # Reset the validation data iterator.
    val_data.reset()
    # Loop over the validation data iterator.
    for batch in val_data:
        # Splits validation data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gutils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits validation label into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gutils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        # Updates internal evaluation
        metric.update(label, outputs)
        name, acc = metric.get()
    metric.reset()
    return acc

get_ipython().run_line_magic('time', '')
epoch = 20
# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = gloss.SoftmaxCrossEntropyLoss()

for i in range(epoch):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gutils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gutils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with autograd.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropogate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    test_acc = evaluate_accuracy(val_data, net, ctx)
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc at epoch %d: %s=%f, test acc: %f'%(i, name, acc, test_acc))


# In[ ]:


epoch = 25
for i in range(epoch):
    # Reset the train data iterator.
    train_val_data.reset()
    # Loop over the train data iterator.
    for batch in train_val_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gutils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gutils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with autograd.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropogate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc at epoch %d: %s=%f'%(i, name, acc))

X_test = test.drop('id', axis=1).iloc[:,:].values.astype('float32') / 255.0
X_test = nd.array(X_test, ctx=ctx[0]).reshape((-1, 1, 28, 28))
preds = net(X_test).argmax(axis=1).astype(int).asnumpy()
preds.shape


# In[ ]:


sample_submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sample_submission['label'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([sample_submission['id'], sample_submission['label']], axis=1)
submission.to_csv('submission.csv', index=False)


# In[ ]:




