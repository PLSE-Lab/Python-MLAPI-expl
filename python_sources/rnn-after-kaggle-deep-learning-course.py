#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I recently finished the Deep Learning Kaggle Course and quickly realised how incredible it is! It covers plenty of deep learning techniques, but I want to build on what I learned. 
# 
# The next step mastering deep learning after doing the course seems to be Recurrent Neural Networks (RNN)... so here are my brief notes explaining what RNNs are and how to work with them using Keras.

# # What are Recurring Neural Networks? 
# 
# Recurring Neural Networks (RNNs) are neural networks that are good at modelling sequence data. Think of it this way: if you see a picture of a ball like the image below, how can you predict which direction it will go in next? 
# 
# ![image.png](attachment:image.png)
# Image taken from: https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9
# 
# Without knowing where the ball has been, you can't really tell where the ball is going.
# 
# RNNs are good at processing things that a in a sequence, like stock predictions, speech recognition and to describe content in pictures. 
# 
# RNNs have sequential memory, an example of this within humans is the alphabet. We can easily recite the alphabet in sequence: 
# abcdefghijk... 
# But if we try it backwards (without practicing) then it is more difficult. 
# 
# *How do RNNs replicate sequential memory?*
# 
# Feed Forward Neural Networks work in a flow, like this:

# ![image.png](attachment:image.png)

# Image taken from: https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9
# 
# RNNs loop back in the hidden state:

# ![image.png](attachment:image.png)
# Image taken from: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470

# An example of this is when we ask the neural network 'what time is it?' The RNN breaks the input up into a sequence, because that is how RNNs work: 'what', 'time', 'is', 'it', '?'
# 
# So the first step would be to produce an output for the word 'what', this would be done in the hidden state. Then the RNN remembers this output and produces an output for the word 'time', this is still in the hidden state.

# ![image.png](attachment:image.png)
# Image taken from: https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9
# 
# This continues until all steps of the input have ran, these loops are all in the hidden layer. A final output is then produced. This can be seen in the image above. 
# 
# With each loop through the hidden layer, the RNN forgets a little more about what happened at the start of the sequence. This is because RNNs have **short term memory**. This leads us onto...
# 
# # The vanishing gradient problem
# 
# Neural networks have three major steps: 
# 1. a forward pass to make a prediction
# 2. comparing the prediction to the ground truth using the loss function - producing an error value to show how well the model is performing
# 3. uses the error value from step 2 to do backpropagation and improve the model
# 
# To understand more about backpropagation, check out the amazing video: https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=711s
# 
# Backpropagation works by using the gradient to adjust the weights of each layer, it works back through the network, i.e. in the direction of the output to the hidden layers to the input layer. As the adjustments start at the end, each adjustmet after gets smaller and smaller, so the layers at the end of the backpropagation have the least learning. 
# 
# This means that the earlier layers fail to do a lot of learning because the internal weights are barely being adjusted due to small gradients, this is the **vanishing gradient problem**.
# 
# Linking back to the example: the RNN forgets the start of the statement, and the neural network is trying to find an output with the input 'is it?'. This would be very difficult, even for a human!
# 
# LSTM and GRU are more commonly used methods that combat the memory problem...

# # LSTM and GRU
# 
# Long Short Term Memory and Gated Recurrent Unit
# 
# 
# They are a form of RNN but are capable of learning long-term dependencies using something called 'gates'. The RNNs described above are commonly called 'vanilla' RNNs. LSTMs and GRUs are used more often as they tackle the vanishing gradient problem, as they are useful for modelling longer sequences and long-term dependencies. 
# 
# The following brief explanation and images are taken from the video:
# https://www.youtube.com/watch?v=8HyCNIVRbSU 
# 
# As you can see from the image below, both LSTMs and GRUs have 'gates' which regulate the flow of information, these gates learn what information to keep and what to discard. So like in the example mentioned earlier: 'what time is it?', the gate may put the most weight on the word 'time' and forget other filler words like 'is' or 'it'. Sort of like how a human would remember the question. 
# 
# Tanh activation:
# 
# Tanh activation, shown in the image below, is a way to regulate the values flowing through the network. As the values assigned to the inputs undergo mathematical operations like multiplication, the values can increase exponentially. So running the values through the tanh function keeps the values between 1 and -1 so that the algorithm can have a better understanding of how important an input variable is. 
# 
# Sigmoid activation: 
# 
# Works similarly to tanh activation as it regulates the values so that they are between 0 and 1. The values that are closer to 0 get forgotten as they are so small and the values closer to 1 are remembered as they travel through the algorithm.
# 
# **LSTMs:**
# 
# As you can see from the image below, LSTMs have three different types of gates: the forget gate, input gate and output gate. The forget gate decides what information should be thrown away or kept. 
# 
# In the input gate the information from the previous cell/hidden state and information from the current input is passed through the sigmoid function to figure out if the information is important. The same information is passed through the tanh function to help regulate the network.
# 
# The cell state gets updated as it has enough information from the forget and input gates, it basically drops more irrelevant information. 
# 
# The outputs from the cell state and the input gate are now fed to the output gate which decides what the next hidden state should be - what should be passed onto the next vector. 
# 
# The diagram shows one part of the process, there are plenty of these vectors that are strung together, one after another.
# 
# ![image.png](attachment:image.png)
# Image taken from: https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
# 
# **GRUs:**
# 
# Newer generation of RNNs and is pretty similar to an LSTM, but it got rid of the cell state and hidden state to transfer information. It has 2 gates: a reset gate and an update gate. These can be seen in the diagram above. 
# 
# The update gate decides what information to remember or forget - like combining the LSTM forget and input gates.
# 
# The reset gate is another gate that decides what information to forget. 
# 
# GRUs have fewer tensor operations - so they are quicker to train than LSTMs.

# # Conclusion
# 
# It makes more sense to ignore vanilla RNNs and just use LSTMs and GRUs, but RNNs have the key advantage of being able to train faster and use less computational resources as there are fewer tensor operations to compute. 
# 
# LSTMs and GRUs are standard and people generally try both and see which one works better for the data.

# # Tensorflow Example Code
# 
# Just to help out with understanding the theory, I have wrote some super basic Keras LSTM and GRU examples. I am sure there are ways to get better results, like those mentioned on the 'Deep Learning' course, but this is a useful starting point.

# The input into every LSTM layer myst be thre-dimensional. The three dimensional inputs are: 
# 
# * Samples - one or more samples
# * Time steps - one time step is one point of observation in the sample
# * Features - one feature is one observation at a time step
# 
# When defining the LSTM, the network assumes you have one or more samples, and requires that you specify the number of time steps and features. You can do this by specifying a tuple to the "input_shape" argument. E.g. model.add(LSTM(32, input_shape = (50, 2))) means that there are 1 or more samples, 50 time steps and 2 features.
# 
# As we need a 3D array for the LSTM, we can use the *reshape()* function in NumPy:

# In[ ]:


from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data = data.reshape((1, 10, 1))
print(data.shape)


# We can print the array to check that it is 3D.
# 
# This data is now ready to be used and has input_shape = (10, 1).
# 
# Now, we shall move onto a more complex example of the MNIST Fashion data that was used in the 'Deep Learning' course

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Flatten

img_rows, img_cols = 28, 28 
num_classes = 10

def prep_data(raw): 
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:] #gives the data as a numpy array
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)
print("Data Loaded")


# In[ ]:


print(fashion_data.shape)
print(x.shape)
print("x.shape[1:]", x.shape[1:])


# In[ ]:


model_LSTM = Sequential()

model_LSTM.add(LSTM(units = 128, activation = 'relu', input_shape = (x.shape[1:]), return_sequences = False)) 
#units = 1 for a vanilla RNN as it is only one layer of the vectors described above
#can have return_sequences=True in case we want to continue onto another LSTM, but False is the default
model_LSTM.add(Dropout(0.2))
#we are adding dropout to ignore randomly chosen nodes, this reduces overfitting
#like in the Deep Learning course

model_LSTM.add(Dense(12, activation = 'relu'))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(num_classes, activation = 'softmax')) #prediction layer

model_LSTM.compile(loss = keras.losses.categorical_crossentropy,
                   optimizer = 'adam',
                   metrics = ['accuracy'])

model_LSTM.summary()


# In[ ]:


model_LSTM.fit(x,y, batch_size = 100, epochs = 4, validation_split = 0.2) #validation_split = 0.2 means that we set 
#20% of the data aside for validation


# It should be noted that CuDNNLSTM is a much faster way to run LSTM.
# 
# 
# GRU Example and some useful tips:
# 
# Decay: 
# 
# You want to take those larger steps in your learning rate to get to the minima, but if we took smaller steps as time goes by, then we could get to the minima faster. Decay essentially decays the learning rate a little bit.
# 
# We keep the the optimizer as Adam in this case.

# In[ ]:


model_GRU = Sequential()

model_GRU.add(GRU(128, activation = 'relu', input_shape = (x.shape[1:]), return_sequences = False)) 
model_GRU.add(Dropout(0.2))

model_GRU.add(Flatten())

model_GRU.add(Dense(12, activation = 'relu'))
model_GRU.add(Dropout(0.2))

model_GRU.add(Dense(num_classes, activation = 'softmax')) #prediction layer

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model_GRU.compile(loss = keras.losses.categorical_crossentropy,
                   optimizer = opt,
                   metrics = ['accuracy'])

model_GRU.summary()


# In[ ]:


model_GRU.fit(x,y,batch_size = 100, epochs = 4, validation_split = 0.2)


# # References:
# https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9
# 
# https://www.youtube.com/watch?v=8HyCNIVRbSU 
# 
# https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
# 
# https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/
# 
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
