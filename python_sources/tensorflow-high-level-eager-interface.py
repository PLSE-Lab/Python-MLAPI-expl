#!/usr/bin/env python
# coding: utf-8

# Note: this notebook is a translated version of https://www.tensorflow.org/get_started/eager.
# 
# Import Tensorflow and turn on eager execution. Eager execution performs NN tasks right away, instead of waiting for two-step graph initialization and runtime, as in the (notably more awkward) low-level API utilization.

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/Iris.csv")
df.Species = df.Species.astype('category').cat.codes
df.set_index('Id').to_csv("foo.csv")


# In[ ]:


def parse_csv(line):
    """Parse a single line of a CSV file into constituent features and labels."""
    # Tensors are typed, and TF expects those types at declare time.
    # You can feed them in using typed examples.
    # The value you select may also be used as a default in case of NaN values!
    example_defaults = [[0.], [0.], [0.], [0.], [0.], [0]]
    
    # Decoding is linewise.
    parsed_line = tf.decode_csv(line, example_defaults)
    
    # Subselect the features as one tensor.
    features = tf.reshape(parsed_line[1:-1], shape=(4,))
    
    # Subselect the label as a unary tensor.
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


# Configure the dataset iterator. Results in a randomized 32-item-per-batch draw of a 1000-item subset of the original dataset.
train_dataset = tf.data.TextLineDataset("foo.csv")
train_dataset = train_dataset.skip(1) 
train_dataset = train_dataset.map(parse_csv)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(32)


# View a single example entry from a batch.
features, label = iter(train_dataset).next()
print("example features:", features[0])


# Next we create a three-layer Keras model. The last layer, the output layer, has as many nodes as classes in this classification problem. The hidden layers can have as many nodes as you would like, and you can have as many hidden layers as you would like. The input layer is not declared explicitly, but instead configured via the `input_shape` parameter on the first hidden layer's declaration, which is required.
# 
# The `Dense` layer is fully interconnected. In other words, every node in a successor layer relies on every node in the previous layer. 
# 
# Tensors passed between nodes in the hidden layers have a functor applied to them. The `Dense` layer uses the following functor by default:
# 
#     output = activation(dot(input, kernel) + bias)
#     
# The `input` is the input tensor. The `kernel` is a group of weights shared over the input space (basically feature importance). The dot product of the kernel and the input is a transformed feature vector. The bias is a linear transform applied to the entire input tensor. The activation function is a function that is applied to the result of the rest of the operations right before the whole thing is handed on down to the next neuron in the set.

# In[ ]:


# Create a two-layer ten node dense Keras layer.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(3)
])


# Training a neural network model means optimizing the kernel (the weights) and the bias. To do so, we must define a loss to minimize, and a gradient function to use to "seek out" that loss minimum. In this next code block we define a loss functor.

# In[ ]:


def loss(model, x, y):
    """Defines a loss function on a batch of predictions. Uses cross-entropy loss."""
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


# Here's a configured optimizer.

# In[ ]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


# Eager execution introduces a gradient tape, or `GradientTape`. This object is a context manager that "records operations for automatic differentiation". Variables (persistant tensors) are automatically recorded, whilst regular tensors can be recorded using an explicit `GradientTape.watch` function call.
# 
# The tape is a space for performing automatic differentiation. Graphs defined within the tape gain a `gradient` object method that can be used to take the gradient. This gradient can then be given to an optimizer, which will determine, based on the gradient and other factors, how best to tune the model parameters.
# 
# For example, here is a gradient computation for the function `x * x` at the value 3:

# In[ ]:


x = tf.constant(3.)
with tfe.GradientTape() as g:
    g.watch(x)
    y = x * x
grad = g.gradient(y, [x])
grad


# We will define a `grad` method which, given the model inputs and a `loss` factory function, returns the gradient applied to that loss function within that input space, using a fresh gradient tape to do it.

# In[ ]:


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


# In[ ]:


train_loss_results = []
train_accuracy_results = []

num_epochs = 200

for epoch in range(num_epochs + 1):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop. Recall that we are using a batch size of 32.
    for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

