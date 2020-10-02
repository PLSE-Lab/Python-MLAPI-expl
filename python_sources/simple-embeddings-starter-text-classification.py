#!/usr/bin/env python
# coding: utf-8

# Most of the time when looking at a Kaggle competition (especially when trying to learn something new) it can be extremely intimidating to see kernels that are tagged as 'beginner' or have 'simple starter' pasted across the headline and when exploring these kernels,  there are ensembled neural nets with bidirectional LSTM's, pretrained embeddings etc. 
# 
# I also find that on kaggle it is extremely satisfying getting from loading the data to a useable submission as fast as possible because then one has a "working model", thus reducing the intimidation barrier regarding the goal of producing something functional.
# 
# So I thought I would take a shot at creating a starter kernel that I would like to read when starting a competition as a beginner, in the case of the Quora Insincere Questions Classification competition that would be one that would allow someone to train a neural net with embeddings as fast and easily as possible, using best practice. Word embeddings are not necessarily a "beginner" concept, however the competition will no doubtedly depend heavily on them. Here it goes:

# # Contents
# 
# * Import libraries and read in data
# * Turn text data into a vector
# * Make vectors uniform length
# * Split training data into training and validation sets
# * Create simple multi-layer perceptron (MLP) neural network with embedding layer
# * What is the F1 score?
# * Optimize F1 score for submission
# * Make submission file using optimized predictions
# * Going Further

# # Import libraries and read in data

# First we import libraries that we are likely to use:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plot figures in the notebook wihtout the need to call plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn-ticks") #set default plotting style for matplotlib

import time
import os
print(os.listdir("../input")) #Print directories/folders in the directory: current_working_directory/input/embeddings


# Next let us read in the train and test data from the above directories using pandas. The training data has three columns: 'qid' , 'question_text' and 'target'. The qid is a unique identifier for each question, the question_text is a string of a question/sentence and the target is a signifier of insincerity where 0 is not insincere and 1 is insincere. 

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

train.head()


# The test data only has the qid and question_text columns as the goal is to predict the target for the questions in the test set:

# In[ ]:


test.head()


# We can confirm that the qid's are unique by comparing the number of unique values to that of the size of the dataframe in both the training data and the test data. We find that they are the same so we will focus only on the question_text column:

# In[ ]:


print(train.shape,train.qid.nunique())
print(test.shape,test.qid.nunique())


# # Turn text data into vector

# Now we need to turn the text from each question into something that a neural network will understand, specifically a vector for each question where each word has a unique integer identifier based on the number of words in a text corpus, here being all the words in our training set. To do this we will use the Tokenizer class from keras. Specifially we will make use of the tensorflow.keras implementation:

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
print(Tokenizer.__doc__)


# So we will no choose a number of possible words/tokens to keep amongst all of our possible words. This will govern the ultimate dimensionality of our training matrix. We must fit the Tokenizer class to our training data and then transform both our training data and our test data:

# In[ ]:


num_possible_tokens=10000 #At this stage this was chosen arbitrarily

tokenizer=Tokenizer(num_words=num_possible_tokens) #Instantiate tokenizer class with number of possible tokens
tokenizer.fit_on_texts(train.question_text) #Fit the tokenizer to training data
sequences_train=tokenizer.texts_to_sequences(train.question_text) #Convert training data to vectors
sequences_test=tokenizer.texts_to_sequences(test.question_text) #Convert test data to vectors


# In[ ]:


sequences_train[0:5]


# The only problem now before going into the neural network is that the network requires a matrix (equal length vectors), we will cover that next.

# # Make vectors uniform length

# To convert our vectors all to the same length, let us use a list comprehension to get the length of the longest vector amongst all vectors in the training and test sets:

# In[ ]:


max_len=np.max([len(i) for i in sequences_train]+[len(i) for i in sequences_test])
print(max_len)


# Next let us use the pad_sequences function from keras to either add zeros to the beginning of each vector or add trailing zeros to each vector to make the all the maximu length calculated above. It can be seen in the docstring below that the default is to pre-pad the input. We need to pass the maximum length required to the function to know how far to pad:

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
print(pad_sequences.__doc__)


# In[ ]:


X=pad_sequences(sequences_train,maxlen=max_len) #Pad the training data, later to be split into a smaller training set and a validation set
X_test=pad_sequences(sequences_test,maxlen=max_len) #Pad the test data

y=train.target.values #Make and independent target variable from the training target. Also to be split into a smaller training set and validation set.

print(X[0:10,:]) # Print first ten rows of the training data
print(X.shape)


# We now see that our input is a matrix with a dimension equal to the length of the longest vector in the training and test data. The leading zeros (pre-padded) can also be seen.

# # Split training data into training and validation sets

# Seeing as we do not have the targets for the test data (obviously) we need a validation set. If the dataset is too small one can use cross validation, however with approximately 1.3 million questions let us use a single validation set. We split the training data and training target above (variable X) into smaller training sets and validation sets. The test size is arbitrarily chosen to be 20 percent.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y, test_size=0.2, random_state=42)


# # Create simple multi-layer perceptron (MLP) neural network with embedding layer

# Before getting into the network let us discuss the embedding layer for the neural network. The embedding layer is much like a normal layer of a neural network that transforms an input matrix into a learned representation at each node, in this case each node in the layer is a word and the parameters of the layer will depict where that word sits in vector space relative to other words.
# 
# From the docstring below we can see that the embedding layer receives a 2D vector with the shape (sample size,input length) where the input length is the length of our questions modified above. The output of the embedding layer is then (sample size,input length, embedding dimension).  From the docstring, it is important to note that if we are to use a MLP, the input_length must be specified in order to flatten the embedding layer as input for a dense layer

# In[ ]:


import tensorflow.keras as keras
from tensorflow.keras import layers

print(layers.Embedding.__doc__)


# For our embedding layer:
# * input_dim is the size of our vocabulary + 1 for unknowns
# * output dim is a chosen dimensionality
# * input_length is the dimension of the input vector specified as the length of the longest question in our input vector
# 
#  Let us make a small 3 layer MLP where the first layer is the embedding layer and the remaing two layers are conventional Dense layers:

# In[ ]:


embedding_dimension=32 # Arbitraily choose an embedding dimension,the 157 dimension input vector will be compressed down to this dimension

model=keras.models.Sequential() # Instantiate the Sequential class

model.add(layers.Embedding(num_possible_tokens+1,embedding_dimension,input_length=max_len)) # Creat embedding layer as described above
model.add(layers.Flatten()) #Flatten the embedding layer as input to a Dense layer
model.add(layers.Dense(32, activation='relu')) # Dense layer with relu activation
model.add(layers.Dense(1,activation='sigmoid')) # Dense layer with sigmoid activation for binary target
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) #binary cross entropy is used as the loss function and accuracy as the metric 
model.summary() # print out summary of the network


# In[ ]:


batch_size=1024 # Choose a batch size
epochs=3 #Choose number of epochs to train

history=model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_val,y_val])


# # What is the F1 score?

# The F1 score is defined as:
# 
# *  2 * (precision x recall) / (precision + recall)
# 
# which is effectively a weighted average of precision and recall where:
# * precision is true positive/(true positive+false positive)
# * recall is true positive/(true positive+false negative)
# 
# We can caluclate it directly using ScikitLearn, where we first need to convert our predicted values to a binary target on order to calculate the precision and recall, initially chosen to be zero below and equal to 0.5 and 1 above 0.5. Remember that the neural network outputs a probability distribution, so we can choose the probability threshold at which our binary target is split.
# 
# The f1 score varies between 0 (worst) and 1 (best).

# In[ ]:


from sklearn.metrics import f1_score

val_pred=model.predict(X_val,batch_size=batch_size).ravel() # predict the values in the validation set which the neural net has not seen

f1_score(y_val,val_pred>0.5) #Predict the f1 score at a threshold of 50%, the point at which our binary target is split in our neural networks output probability distribution


# # Optimize F1 score for submission

# Above we chose a probability threshold of 50% to split our binary target, but is this always the best predictor? We can sweep a range of thresholds between 0 and 50% and recalculate the f1 score based on each threshold:

# In[ ]:


Threshold=[] # List ot store tested thresholds
f1=[] # List to store associated f1 score for threshold

for i in np.arange(0.1, 0.501, 0.01):
    Threshold.append(i)
    temp_val_pred=val_pred>i # convert to True or False Boolean based on threshold
    temp_val_pred=temp_val_pred.astype(int) # Convert Boolean to integer
    score=f1_score(y_val,temp_val_pred) #Calculate f1 score at threshold
    f1.append(score) #store f1 score
    print("Threshold: {} \t F1 Score: {}".format(np.round(i,2),score))


# From the stored thresholds and lists calculate the optimum threshold:

# In[ ]:


best_threshold=Threshold[np.argmax(f1)] #Get threshold at index of largest f1 score.
best_threshold


# # Make submission file using optimized predictions

# Finally we calculate our predictions on the test data and put the data into a dataframe in the required format, where our predictions are converted to a binary target based on the best threshold calculated above. 

# In[ ]:


test_pred=model.predict(X_test,batch_size=4096).ravel() #Predict test data

df=pd.DataFrame({'qid':test.qid.values,'prediction':test_pred}) #Create dataframe of unique id's and predicted target 
df.prediction=(df.prediction>best_threshold).astype(int) #Convert target to binary based on best f1 threshold
df.head()


# Write csv for submission!

# In[ ]:


df.to_csv("submission.csv", index=False)


# # Going Further

# Obvioulsy this is an extremely naive model having never looked at the data but its purpose was to get from input to working submission in a minimal manner in order to understand the process. Now with a working model one can quite easily start to:
# 
# * Perform Exploratory Data Analysis (EDA) knowing how to convert it into something useable
# * Clean the data knowing the required format
# * Vary the network structure by making small changes to a working network
# * Change network parameters etc,
# 
# without the looming intimidation of getting through the process.  Let me know if this helps if you are new to keras and embeddings :) 
# 
