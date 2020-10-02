#!/usr/bin/env python
# coding: utf-8

# # Multilayer Neural Network in Tensorflow
# 
# This notebook is based on the Deep Learning course from the Master Datascience Paris Saclay. Materials of the course can be found [here](https://github.com/m2dsupsdlclass/lectures-labs).
# 
# ### Goal of the notebook
# 
# * Introduce the basics of `Tensorflow`.
# * Computation  of auto-differentiation with `Tensorflow`.
# * Implement the digit classifier using the low level `Tensorflow` API without Keras abstraction.
# 
# ## Introduction to Tensorflow
# 
# `Tensorflow` is a dynamic graph computation engine that allows differentiation of each node. This library is the default computational backend of the `Keras` library. It can also be used directly from Python to build deep learning models. Check out this link:
# * https://www.tensorflow.org/
# * https://www.tensorflow.org/tutorials/quickstart/advanced
# 
# It builds on nodes where nodes may be:
# * **constant**: constants tensors, such as training data;
# * **Variable**: any tensor tht is meant to be updated when training, such as parameters of the models.
# 
# **Note:** we are going to use the version 2.0 of `Tensorflow`. This version cleaned the old cluttered API and uses by default dynamic graph of operations to make it natural to design a model interactively in Jupyter. Previously, you defined the graph statically once, and then needed to evaluate it by feeding it some data. Now, it is dynamically defined when executing imperative Python instructions which means that you can `print` any tensor at any moment or even use `pdb.set_trace()` to inspect intermediary values.

# In[ ]:


# Display figure in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[ ]:


# Define some functions
def plot_mnist(data, index, label=None):
    """Plot one image from the mnist dataset."""
    fig = plt.figure(figsize=(3, 3))
    if type(data) == pd.DataFrame:
        plt.imshow(np.asarray(data.iloc[index, 1:]).reshape((HEIGHT, WIDTH)),
                   cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.title(f"Image label: {data.loc[index, 'label']}")
    else:
        plt.imshow(data[index].reshape((HEIGHT, WIDTH)),
                   cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.title(f"Image label: {label}")
    
    plt.axis('off')
    return fig


# In[ ]:


# Define a constant tensor
a = tf.constant(3)
a


# In[ ]:


# Define a Variable tensor
c = tf.Variable(0)

b = tf.constant(2)

# Sum of two tensors
c = a + b
c


# In[ ]:


# Define a constant tensor in 2 dimensions
A = tf.constant([[0, 1], [2, 3]], dtype=tf.float32)
A


# In[ ]:


# Convert tf.Tensor as numpy array
A.numpy()


# In[ ]:


# Define a Variable tensor
b = tf.Variable([1, 2], dtype=tf.float32)
b


# In[ ]:


# Reshape a Variable tensor
tf.reshape(b, (-1, 1))


# In[ ]:


# Perform matrix multiplication
tf.matmul(A, tf.reshape(b, (-1, 1)))


# Write a function that computes the squared Euclidean norm of an 1D tensor input `x`:
# * Use element wise arithmetic operations `(+, -, *, /, **)`.
# * Use `tf.reduce_sum` to compute the sum of the element of a Tensor.

# In[ ]:


def squared_norm(x):
    return tf.reduce_sum(x ** 2)


# In[ ]:


x = tf.Variable([1, -4], dtype=tf.float32)
x


# In[ ]:


squared_norm(x)


# In[ ]:


squared_norm(x).numpy()


# ### Autodifferentiation and Gradient Descent

# In[ ]:


with tf.GradientTape() as tape:
    result = squared_norm(x)
    
variables = [x]
gradients = tape.gradient(result, variables)
gradients


# We can apply a gradient step to modify `x` in place by taking one step of gradient descent.

# In[ ]:


x.assign_sub(0.1 * gradients[0])
x.numpy()


# Execute the following gradient descent step many times consecutively to watch the decrease of the objective function and the values of `x` converging to the minimum of the `squared_norm` function.

# In[ ]:


with tf.GradientTape() as tape:
    objective = squared_norm(x)

x.assign_sub(0.1 * tape.gradient(objective, [x])[0])
print(f"Objective = {objective.numpy():e}")
print(f"x = {x.numpy()}")


# ### Device-aware Memory Allocation
# 
# To explicitely place tensors on a device, we use context managers.

# In[ ]:


# On CPU
with tf.device("CPU:0"):
    x_cpu = tf.constant(3)
x_cpu.device

# On GPU
#with tf.device("GPU:0"):
#    x_gpu = tf.constant(3)
#x_gpu.device


# ## Building a digits classifier in Tensorflow
# 
# Dataset:
#  * The MNIST dataset ([Kaggle link](https://www.kaggle.com/c/digit-recognizer/overview))

# In[ ]:


# Load the data
digits_train = pd.read_csv('../input/digit-recognizer/train.csv')
digits_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


# Define some global parameters
HEIGHT = 28 # Height of an image
WIDTH = 28 # Width of an image
PIXEL_NUMBER = 784 # Number of pixels in an image
PIXEL_VALUE = 255 # Maximum pixel value in an image


# In[ ]:


# Print an image
sample_index = 42

plot_mnist(digits_train, sample_index)
plt.show()


# ### Preprocessing
# 
# * Normalization
# * Train / Validation split

# In[ ]:


# Extract and convert the pixel as numpy array with dtype='float32'
train = np.asarray(digits_train.iloc[:, 1:], dtype='float32')
test = np.asarray(digits_test, dtype='float32')

train_target = np.asarray(digits_train.loc[:, 'label'], dtype='int32')


# In[ ]:


# Split and scale the data
X_train, X_val, y_train, y_val = train_test_split(
    train, train_target, test_size=0.15, random_state=42)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test)


# `Tensorflow` provides dataset abstraction which makes possible to iterate over the data batch by batch.

# In[ ]:


def gen_dataset(x, y, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)) # Create the dataset
    dataset = dataset.shuffle(buffer_size=10000, seed=42) # Shuffle the dataset
    dataset = dataset.batch(batch_size=batch_size) # Combine consecutive elements of dataset into batches.
    return dataset


# In[ ]:


# Create the dataset
dataset = gen_dataset(X_train, y_train)
dataset


# In[ ]:


# Get the first batch
batch_x, batch_y = next(iter(dataset))
print(f"Size batch_x: {batch_x.shape} / Size batch_y: {batch_y.shape}")


# ### Build a model using Tensorflow
# 
# * Using `Tensorflow`, build a MLP with one hidden layer.
# * The input will be a batch coming from `X_train`, and the output will be a batch of integer.
# * The output do not need be normalized as probabilities, the softmax will be moved to the loss function.

# In[ ]:


# Helper function
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))

def accuracy(y_pred, y):
    return np.mean(np.argmax(y_pred, axis=1) == y)

def test_model(model, x, y):
    dataset = gen_dataset(x, y)
    preds, targets = [], []
    
    for batch_x, batch_y in dataset:
        preds.append(model(batch_x).numpy())
        targets.append(batch_y.numpy())
        
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return accuracy(preds, targets)


# In[ ]:


# Hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZE = 15
LEARNING_RATE = 0.5
NUM_EPOCHS = 10
INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = 10
LAMBDA = 10e-4
GAMMA = 0.9

# Build the model
class MyModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_h = init_weights(shape=[input_size, hidden_size])
        self.b_h = init_weights([hidden_size])
        self.W_o = init_weights(shape=[hidden_size, output_size])
        self.b_o = init_weights([output_size])
        
    def __call__(self, inputs):
        h = tf.sigmoid(tf.matmul(inputs, self.W_h) + self.b_h)
        return tf.matmul(h, self.W_o) + self.b_o


# In[ ]:


# Define the model
model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


# In[ ]:


# Run the model on the validation set
print(f"Accuracy on the validation set on the untrained model: {test_model(model, X_val, y_val)}")


# The following implements a training loop in Python. Note the use of `tf.GradientTape` to automatically compute the gradients of the loss with respect to the different parameters of the model.

# In[ ]:


losses = []
train_acc = [test_model(model, X_train, y_train)]
test_acc = [test_model(model, X_val, y_val)]
# Loop over the epochs
for e in range(NUM_EPOCHS):
    train_dataset = gen_dataset(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Loop over the batches
    for batch_x, batch_y in train_dataset:
        # tf.GradientTape records the activation to compute the gradients.
        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, logits))
            losses.append(loss.numpy())
        
        # Compute the gradient of the loss with respect to W_h, b_h, W_o and b_o
        dW_h, db_h, dW_o, db_o = tape.gradient(
            loss, [model.W_h, model.b_h, model.W_o, model.b_o])
        
        # Update the weights as a SGB would do
        model.W_h.assign_sub(LEARNING_RATE * dW_h)
        model.b_h.assign_sub(LEARNING_RATE * db_h)
        model.W_o.assign_sub(LEARNING_RATE * dW_o)
        model.b_o.assign_sub(LEARNING_RATE * db_o)
    
    train_acc_e = test_model(model, X_train, y_train)
    test_acc_e = test_model(model, X_val, y_val)
    train_acc.append(train_acc_e)
    test_acc.append(test_acc_e)
    print(f"Epoch {e}: train accuracy = {round(train_acc_e, 4)}, test accuracy = {round(test_acc_e, 4)}")


# In[ ]:


# Plot of the losses
plt.plot(losses)
plt.show()


# In[ ]:


# Plot of the accuracy
fig, ax = plt.subplots()
ax.plot(train_acc, label='Train')
ax.plot(test_acc, label='Test')
ax.legend()
plt.show()


# * Add $L_2$ regularization with $\lambda = 10^{-4}$.
# 
# With the regularization, the cost function is the negative likelihood of the model computed on the full training set (for i.i.d. samples):
# $$L_S(\theta) = -\frac{1}{\lvert S \rvert}\sum_{s \in S}\log f(x^s; \theta)_{y^s} + \lambda\Omega(\theta)$$
# where $\Omega(\theta) = \| W^h \|^2 + \| W^o \|^2$.

# In[ ]:


# Define the model
model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


# In[ ]:


# Run the model on the validation set
print(f"Accuracy on the validation set on the untrained model: {test_model(model, X_val, y_val)}")


# In[ ]:


losses = []
train_acc = [test_model(model, X_train, y_train)]
test_acc = [test_model(model, X_val, y_val)]
# Loop over the epochs
for e in range(NUM_EPOCHS):
    train_dataset = gen_dataset(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Loop over the batches
    for batch_x, batch_y in train_dataset:
        # tf.GradientTape records the activation to compute the gradients.
        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, logits)) +                    LAMBDA * (tf.nn.l2_loss(model.W_h) + tf.nn.l2_loss(model.W_o))
            losses.append(loss.numpy())
        
        # Compute the gradient of the loss with respect to W_h, b_h, W_o and b_o
        dW_h, db_h, dW_o, db_o = tape.gradient(
            loss, [model.W_h, model.b_h, model.W_o, model.b_o])
        
        # Update the weights as a SGB would do
        model.W_h.assign_sub(LEARNING_RATE * dW_h)
        model.b_h.assign_sub(LEARNING_RATE * db_h)
        model.W_o.assign_sub(LEARNING_RATE * dW_o)
        model.b_o.assign_sub(LEARNING_RATE * db_o)
    
    train_acc_e = test_model(model, X_train, y_train)
    test_acc_e = test_model(model, X_val, y_val)
    train_acc.append(train_acc_e)
    test_acc.append(test_acc_e)
    print(f"Epoch {e}: train accuracy = {round(train_acc_e, 4)}, test accuracy = {round(test_acc_e, 4)}")


# In[ ]:


# Plot of the losses
plt.plot(losses)
plt.show()


# In[ ]:


# Plot of the accuracy
fig, ax = plt.subplots()
ax.plot(train_acc, label='Train')
ax.plot(test_acc, label='Test')
ax.legend()
plt.show()


# In[ ]:


# Do prediction on the test set
pred = np.argmax(model(X_test).numpy(), axis=1)


# In[ ]:


# Submision file
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sub.loc[:,'Label'] = pred

sub.to_csv('submission.csv', index=False)


# * Implementation of the momentum
# 
# This idea of the momentum is to accumulate gradients across successive updates:
# $$m_t = \gamma m_{t-1} + \eta \nabla_{\theta}L_{B_t}(\theta_{t-1})$$
# $$\theta_t = \theta_{t-1} - m_t$$
# $\gamma$ is typically set to $0.9$.

# In[ ]:


# Define the model
model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


# In[ ]:


# Run the model on the validation set
print(f"Accuracy on the validation set on the untrained model: {test_model(model, X_val, y_val)}")


# In[ ]:


losses = []
train_acc = [test_model(model, X_train, y_train)]
test_acc = [test_model(model, X_val, y_val)]

# Define the momentum
m_W_h = np.zeros((INPUT_SIZE, HIDDEN_SIZE))
m_W_o = np.zeros((HIDDEN_SIZE, OUTPUT_SIZE))
# Loop over the epochs
for e in range(NUM_EPOCHS):
    train_dataset = gen_dataset(X_train, y_train, batch_size=BATCH_SIZE)

    # Loop over the batches
    for batch_x, batch_y in train_dataset:
        # tf.GradientTape records the activation to compute the gradients.
        with tf.GradientTape() as tape:
            logits = model(batch_x)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, logits)) +                    LAMBDA * (tf.nn.l2_loss(model.W_h) + tf.nn.l2_loss(model.W_o))
            losses.append(loss.numpy())
        
        # Compute the gradient of the loss with respect to W_h, b_h, W_o and b_o
        dW_h, db_h, dW_o, db_o = tape.gradient(
            loss, [model.W_h, model.b_h, model.W_o, model.b_o])
        
        # Update the momentum
        m_W_h = GAMMA * m_W_h + LEARNING_RATE * dW_h
        m_W_o = GAMMA * m_W_o + LEARNING_RATE * dW_o
        
        # Update the weights as a SGB would do
        model.W_h.assign_sub(m_W_h)
        model.b_h.assign_sub(LEARNING_RATE * db_h)
        model.W_o.assign_sub(m_W_o)
        model.b_o.assign_sub(LEARNING_RATE * db_o)
    
    train_acc_e = test_model(model, X_train, y_train)
    test_acc_e = test_model(model, X_val, y_val)
    train_acc.append(train_acc_e)
    test_acc.append(test_acc_e)
    print(f"Epoch {e}: train accuracy = {round(train_acc_e, 4)}, test accuracy = {round(test_acc_e, 4)}")


# In[ ]:


# Plot of the losses
plt.plot(losses)
plt.show()


# In[ ]:


# Plot of the accuracy
fig, ax = plt.subplots()
ax.plot(train_acc, label='Train')
ax.plot(test_acc, label='Test')
ax.legend()
plt.show()

