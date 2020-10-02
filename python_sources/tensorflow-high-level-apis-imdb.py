#!/usr/bin/env python
# coding: utf-8

# # Classifying movie reviews with tf.keras, tf.data and eager execution
# Code: https://github.com/tensorflow/workshops

# In[ ]:


import json
import numpy as np
import tensorflow as tf

# enable eager execution
tf.enable_eager_execution()


# ## Get the dataset
# The "IMDB dataset" is a set of around 50,000 positive or negative reviews for movies from the Internet Movie Database. Since we can not directly use tf.keras.datasets because of kaggle, we will use the `load_data` and `get_word_index` functions from the keras implementations.

# In[ ]:


def load_data(path, num_words=None, skip_top=0, seed=113):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    
    if not num_words:
        num_words = max([max(x) for x in xs])

    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    
    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
    
    return (x_train, y_train), (x_test, y_test)


# In[ ]:


def get_word_index(path):
    with open(path) as f:
        return json.load(f)


# In[ ]:


(train_data, train_labels), (test_data, test_labels) = load_data('../input/imdb.npz', num_words=10000)


# ## Exploring the dataset
# The dataset comes preprocessed. Each example is an array of integers representing the words of movie review. Each label is either a "0" for negative or "1" for positive.

# In[ ]:


print('train_data shape:', train_data.shape)
print('train_labels shape:', train_labels.shape)
print('a train_data sample:', train_data[0])
print('a train_label sample:', train_labels[0])


# ## Word indexing
# To prove that we have limited ourselves to the 10,000 most frequent words, we will iterate over all the reviews and check the maximum value

# In[ ]:


print(max([max(review) for review in train_data]))


# ## Converting the integers back to words
# We can get the dictionary by using `get_word_index` which hashes the words to the corresponding integers. Let's try and convert a review from integer back into it's original text by first reversing this dictionary and then iterating over a review and converting the integerst to strings

# In[ ]:


# dictionary that hashes words to their integer
word_to_integer = get_word_index('../input/imdb_word_index.json')

# print out the first ten keys in the dictionary
print(list(word_to_integer.keys())[0:10])

integer_to_word = dict([(value, key) for (key, value) in word_to_integer.items()])

# demonstrate how to find the word from an integer
print(integer_to_word[1])
print(integer_to_word[2])

# we need to subtract 3 from the indices because 0 is 'padding', 1 is 'start of sequence' and 2 is 'unknown'
decoded_review = ' '.join([integer_to_word.get(i - 3, 'UNK') for i in train_data[0]])
print(decoded_review)


# ## Format the data
# Unfortunately, we cannot just feed unformatted arrays into our neural network. We need to standardize the input. Here we are going to multi-hot-encode our review which is an array of integers into a 10,000 dimensional vector. We will place 1's in the indices of word-integers that occur in the review and 0's everywhere else.

# In[ ]:


def vectorize_sequences(sequences, dimension=10000):
    # creates an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, sequence in enumerate(sequences):
#         print(i, sequence)
        results[i, sequence] = 1. # set specific indices of results[i] to be 1s (float)
    return results

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)

print(train_data.shape) # length is same as before
print(train_data[0]) # now, multi-hot encode

# vectorize the labels as well and reshape from (N, ) to (N, 1)
train_labels = np.reshape(np.asarray(train_labels, dtype=np.float32), (len(train_data), 1))
test_labels = np.reshape(np.asarray(test_labels, dtype=np.float32), (len(test_data), 1))


# ## Create the model
# Now that we have vectorized input, we are ready to build our neural network. We are going to use multiple hidden layers, each with 16 hidden units. The more layers and hidden units you have, the more complicated patterns the network can recognize and learn. Beware of overfitting though. We will be using a sigmoid activation function which will take the output of the first two layers and "squish" it into a number between 0 and 1 which will form our prediction.

# In[ ]:


# create model
model = tf.keras.Sequential()

# input shape here is the length of our movie review vector
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(10000, )))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

model.summary()


# ## Create a validation set
# We want to test our model on data it hasn't seen before, but before we get our final test accuracy. This dataset is generally called validation set. In practice, it's used to tune parameters like learning rate or the number of layers/units in the model.

# In[ ]:


VAL_SIZE = 10000

val_data = train_data[:VAL_SIZE]
partial_train_data = train_data[VAL_SIZE:]

val_labels = train_labels[:VAL_SIZE]
partial_train_labels = train_labels[VAL_SIZE:]


# ## Create a tf.data Dataset
# We are working with a small in-memory dataset, so converting to tf.data isn't essential, but it's a good practice.

# In[ ]:


BATCH_SIZE = 512
SHUFFLE_SIZE = 1000

training_set = tf.data.Dataset.from_tensor_slices((partial_train_data, partial_train_labels))
training_set = training_set.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)


# ## Training the model
# Now we are going to train the model. As we go, we will keep track of training loss, training accuracy, validation loss and validation accuracy. This may take a while.

# In[ ]:


EPOCHS = 10

# store list of metric values for plotting later
training_loss_list = []
training_accuracy_list = []
validation_loss_list = []
validation_accuracy_list = []

for epoch in range(EPOCHS):
    for reviews, labels in training_set:
        # calculate training loss and accuracy
        training_loss, training_accuracy = model.train_on_batch(reviews, labels)
        
    # calculate validation loss and accuracy
    validation_loss, validation_accuracy = model.evaluate(val_data, val_labels)
    
    # add to the lists
    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    validation_loss_list.append(validation_loss)
    validation_accuracy_list.append(validation_accuracy)
    
    print(('EPOCH %d\t Training Loss: %.2f\t Training Accuracy: %.2f\t Validation Loss: %.2f\t Validation Accuracy: %.2f') % (epoch + 1, training_loss, training_accuracy, validation_loss, validation_accuracy))
    


# ## Plotting loss and accuracy
# using `matplotlib` to plot our training and validation metrics

# In[ ]:


import matplotlib.pyplot as plt

epochs = range(1, EPOCHS + 1)

# "bo" specifies "blue dot"
plt.plot(epochs, training_loss_list, 'bo', label='Training loss')
# b spcifies "solid blue line"
plt.plot(epochs, validation_loss_list, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf() # clear plot

plt.plot(epochs, training_accuracy_list, 'bo', label='Training accuracy')
plt.plot(epochs, validation_accuracy_list, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ## Testing the model
# Not that the model is ready and our training accuracy is above 90%, we need to test it on entirely different dataset. We are going to run our model on the test set. The test accuracy is a better evalutation metric for how our model will perform in the real world.

# In[ ]:


loss, accuracy = model.evaluate(test_data, test_labels)
print('Test accuracy: %.2f' % (accuracy))


# ### Congratulations
# You have successfully trained a model on the IMDB dataset.
