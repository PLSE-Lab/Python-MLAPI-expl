#!/usr/bin/env python
# coding: utf-8

# Install Tensorflow datasets Tensorflow2.0

# In[ ]:


get_ipython().system('pip install tensorflow_datasets')


# In[ ]:



import tensorflow as tf
import tensorflow_datasets as tfds
import os


# Reference Code
# https://www.tensorflow.org/tutorials/load_data/text

# The dataset used for analysis is the clean version as preperared using the following kernel 
# https://www.kaggle.com/brunnoricci/personalitytypeclassification with the numerical categories added for the MBTI 
# Type . Can be downloaded from www.kaggle.com/uplytics/mbti-clean-with-categories

# In[ ]:


mbti_dataset_line = tf.data.TextLineDataset("../input/mbti-clean-with-categories/mbti_clean.csv")


# In[ ]:


for ex in mbti_dataset_line.take(5):
  print(ex)


# In[ ]:


def label(line):
  label =  tf.strings.substr([line],[-10],[1])
  if label[0]==',':
    label = tf.strings.substr([line],[-9],[1])
  else:
    label = tf.strings.substr([line],[-10],[2])
  labelnum=tf.strings.to_number(label,tf.int64)
  line= tf.strings.substr([line],[6],(tf.strings.length([line])-17))
  return line[0], labelnum[0]


# In[ ]:



mbti_dataset_line = mbti_dataset_line.skip(1).map(lambda line: label(line))


# 

# In[ ]:


for ex in mbti_dataset_line.take(5):
  print(ex)


# In[ ]:


BUFFER_SIZE = 100000
BATCH_SIZE = 64
TAKE_SIZE = 5000


# In[ ]:


mbti_dataset_line = mbti_dataset_line.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)


# ##Encode text lines as numbers
# First, build a vocabulary by tokenizing the text into a collection of individual unique words. 
# 1.Iterate over each example's numpy value.
# 2. Use tfds.features.text.Tokenizer to split it into tokens.
# 3.Collect these tokens into a Python set, to remove duplicates.
# 4.Get the size of the vocabulary for later use.

# In[ ]:


tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in mbti_dataset_line:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size


# Create an encoder by passing the vocabulary_set to tfds.features.text.TokenTextEncoder. The encoder's encode method takes in a string of text and returns a list of integers.

# In[ ]:


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# In[ ]:


example_text = next(iter(mbti_dataset_line))[0].numpy()
print(example_text)


# In[ ]:


encoded_example = encoder.encode(example_text)
print(encoded_example)


# Now run the encoder on the dataset by wrapping it in tf.py_function and passing that to the dataset's map method.

# In[ ]:


def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

all_encoded_data = mbti_dataset_line.map(encode_map_fn)


# Split the dataset into test and train batches

# In[ ]:


train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))


# 

# In[ ]:


sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]


# Add the padding (0) token to vocabulary

# In[ ]:


vocab_size += 1


# Start Building the Tensorflow2.0 Model

# In[ ]:


model = tf.keras.Sequential()


# The first layer converts integer representations to dense vector embeddings. 

# In[ ]:


model.add(tf.keras.layers.Embedding(vocab_size+1, 64))


# The next layer is a Long Short-Term Memory layer, which lets the model understand words in their context with other words. A bidirectional wrapper on the LSTM helps it to learn about the datapoints in relationship to the datapoints that came before it and after it.

# In[ ]:


model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))


# Finally we'll have a series of one or more densely connected layers, with the last one being the output layer. The output layer produces a probability for all the labels. The one with the highest probability is the models prediction of an example's label.

# In[ ]:


# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(16, activation='softmax'))


# Finally, compile the model. For a softmax categorization model, use sparse_categorical_crossentropy as the loss function. You can try other optimizers, but adam is very common.

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(train_data, epochs=1, validation_data=test_data)


# In[ ]:


eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))


# In[ ]:


exit()

