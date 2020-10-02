#!/usr/bin/env python
# coding: utf-8

# # 0 Introduction to the Notebook
# 
# ## How to use this notebook 
# 
# In order to use this notebook and run the code, follow the next steps: 
# 1. Sign up to kaggle.com and log in 
# 2. Click on the following [link]() and click Copy and Edit in the upper right corner (a new tab opens with an editable version of this notebook) 
# 3. Run each cell with Shift + Enter
# 
# ## Motivation and Disclaimer
# 
# This is a notebook introducing Recurrent Neural Networks(RNN) written by Sheraz Ali, Diana Alexandra Gaiu and Jonas Kowalick for the seminar project "Social Media and Business Analytics" in WS19/20 at Potsdam University. 
# The practical example in section 3 is inspired by the notebook 'RNN for Spam Detection' which can be found under the following [link](https://www.kaggle.com/kentata/rnn-for-spam-detection) and makes use of the Kaggle database found at [link](https://www.kaggle.com/team-ai/spam-text-message-classification)
# 
# ## Outline 
# 
# 1. [Theoretical Background](#theoretical-background)
# 2. [Practical Applications](#practical-applications)
# 3. [Coding Example - RNN Implementation in Python using Keras](#coding-example)
# 4. [References](#references)

# # <a name='theoretical-background'></a> 1 Theoretical Background 

# ## What are RNNs?
# <img src="https://www.researchgate.net/profile/Weijiang_Feng/publication/318332317/figure/fig1/AS:614309562437664@1523474221928/The-standard-RNN-and-unfolded-RNN.png"/>
# 
# 
# Source: [ResearchGate](https://www.researchgate.net/figure/The-standard-RNN-and-unfolded-RNN_fig1_318332317)
# 
# ## Unfolded RNN
# 
# Recurrent neural networks involve loops where the output of the networks is fed back as input for further processing. RNNs can use their internal state (memory) to process sequences of inputs.
# In a recurrent neural network we store the output activations from one or more of the layers of the network. Often these are hidden later activations. Then, the next time we feed an input example to the network, we include the previously-stored outputs as additional inputs.
# 
# $x^{(t)}$ - current input at time $t$ 
# 
# $x^{(t-1)}$ - input at time $(t-1)$
# 
# $x^{(t+1)}$  - input at time $(t+1)$
# 
# $o^{(t)}$  - output at time $t$ 
# 
# $h^{(t)}$  - hidden state at time $t$
# 
# $U, V, W$ - weight matrices (hyperparameters) 
# 
# 
# ## Forward Pass
# 
# <img src="https://miro.medium.com/max/419/1*55c3opV_tqm3wUwcj0m-jg.png"/>
# 
# Source: [Deep Learning, Goodfellow](http://www.deeplearningbook.org/contents/rnn.html)
# 
# $a^{(t)}$ - bias vector $b$ + weight matrix $W$ * previous hidden state (at time $(t-1)$ ) + weight matrix $U$ * current input $x$ (at time $t$)
# 
# $h^{(t)}$ - hyperbolic tangent activation function for hidden state at time $t$ 
# 
# $o^{(t)}$ - output: bias vector $c$ + weight matrix $V$ * hidden state at time $t$ 
# 
# $\hat{y}^{(t)}$ - predicted probability using softmax 

# # <a name='practical-applications'></a>2 Practical Applications
# 
# <img src="https://miro.medium.com/max/1223/1*XosBFfduA1cZB340SSL1hg.png">
# 
# Source: [Medium](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912)
# 
# 1) Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification)
# 
# 2) Sequence output (e.g. image captioning takes an image and outputs a sentence of words)
# 
# 3) Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment)
# 
# 4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French)
# 
# 5) Synced sequence input and output (e.g. video classification where we wish to label each frame of the video)
# 
# Source: [Medium](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912)
# 
# #### Other example: Time Series Data e.g. predicting stock prices
# #### Other example: Anticipating car routes in autonomous driving
# #### Other example: Generating new art [MAGENTA](https://magenta.tensorflow.org/) Link to Video: [Making music with RNN](https://www.youtube.com/watch?time_continue=82&v=iTXU9Z0NYoU&feature=emb_title)
# 
# 

# # <a name='coding-example'></a>3 Coding Example - RNN Implementation in Python using Keras

# In[ ]:


#Importing the useful methods from the RNN library keras (sublibrary of tensorflow)
from keras.layers import SimpleRNN, Embedding, Dense, LSTM, GRU
from keras.models import Sequential

#Importing th basic math,plotting and data manipulation libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()


# In[ ]:


#Reading the data from the csv file and taking a look at it
data = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
data.head(5)


# In[ ]:


#Generating the training/test data for the model and assigning numerical labels 0(nonspam), 1(spam).

emails = []
labels = []
for i, label in enumerate(data['Category']):
    emails.append(data['Message'][i])
    if label == 'ham':
        labels.append(0)
    else:
        labels.append(1)

emails = np.asarray(emails)
labels = np.asarray(labels)


print("number of emails :" , len(emails))
print("number of labels: ", len(labels))


# In[ ]:


#from keras.layers import SimpleRNN, Embedding, Dense, LSTM
#from keras.models import Sequential

#Import libraries for preprocessing text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# number of words used as features
max_features = 10000
# cut off the words after seeing 500 words in each document(email)
maxlen = 500


# we will use 80% of data as training, 20% as validation data
training_samples = int(len(emails) * .8)
test_samples = int(len(emails) - training_samples)

print("The number of training {0}, validation {1} ".format(training_samples, test_samples))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)


word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

first_email = []
for i in sequences[0]:
    first_email.append([word for word, index in word_index.items() if index == i][0]);
print(first_email)

data = pad_sequences(sequences, maxlen=maxlen)

print("data shape: ", data.shape)

np.random.seed(23)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


emails_train = data[:training_samples]
y_train = labels[:training_samples]
emails_test = data[training_samples:]
y_test = labels[training_samples:]


# In[ ]:


#Define the RNN model

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model.fit(emails_train, y_train, epochs=10, batch_size=60, validation_split=0.2)


# In[ ]:


acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training loss')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


pred = model.predict_classes(emails_test)
acc = model.evaluate(emails_test, y_test)
proba_rnn = model.predict_proba(emails_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))


# ## LTSM

# In[ ]:


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_ltsm = model.fit(emails_train, y_train, epochs=10, batch_size=30, validation_split=0.2)


# In[ ]:


acc = history_ltsm.history['acc']
val_acc = history_ltsm.history['val_acc']
loss = history_ltsm.history['loss']
val_loss = history_ltsm.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training loss')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


pred = model.predict_classes(emails_test)
acc = model.evaluate(emails_test, y_test)
proba_ltsm = model.predict_proba(emails_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))


# # <a name='references'></a> 4 References

# (1) [NLP from basics to using RNN and LSTM, vibhor nigam](https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66)
# 
# (2) [Recurrent Neural Networks(RNNs), Javaid Nabi](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)
# 
# (3) [Sequence Modeling: Recurrentand Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html)
# 
# (4) [An Introduction to Recurrent Neural Networks, Suvro Banerjee](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912)
# 
# (4) [Illustrated Guide to Recurrent Neural Networks, Michael Nguyen](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)
# 
# (5) [How Recurrent Neural Networks work, Simeon Kostadinov](https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7)

# In[ ]:




