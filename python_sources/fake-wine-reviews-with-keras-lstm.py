#!/usr/bin/env python
# coding: utf-8

# LSTM are a very effective way to mimic sequences of characters (See [this][1] amazing blog post for more information on the principle). And since professional wine reviewers might be ["faking it"][2], I wanted to try out, how good a simple LSTM would be at faking wine-reviews.
# 
#   [1]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#   [2]: https://www.youtube.com/watch?v=bdcG7PlkAg0

# The first thing I need to do is to import the data in a way that is usable for the LSTM. The descriptions from the original CSV are selected, concatenated with a separator between reviews and put into lowercase to reduce the total "character vocabulary" the LSTM has to learn. There is obviously a lot of improvement potential here, which is left as an exercise to the reader ;)
# 
# The output is to test if the import worked as intended and to give some metrics about the training dataset.

# In[ ]:


#Imports for the entire Kernel
import pandas as pd
import numpy as np
import sklearn.preprocessing
import keras as ke

separator_char = "|"
sample_length = 64
batch_size = 64

df = pd.read_csv("../input/winemag-data_first150k.csv")
df = df["description"]

train_string = df.str.cat(sep=separator_char)

train_string = train_string.lower()



vocab_size = len(set(train_string))

print ("Data Sanity Check:")
print (train_string[:1000])
print("\n\n")
print ("Training Set Total Lenght:", len(train_string))
print ("Total Unique Characters:", vocab_size)


# Next, the training string has to be brought into a format usable for the LSTM. This requires multiple steps. First, a sample of sample_length characters is taken from the training set. This will be encoded as the input for the training datapoint. The character following this sample is the output. 
# 
# An example from the beginning of the train_string (see above):
# > IN: "this tremendous 100% varietal wine hail"  OUT: "s"
# 
# After this sample has been created, the characters have to be individually mapped to integers with a dictionary.  These integers are then one hot encoded since they are categorical variables.
# 
# The entire preprocessing is done within a generator function that can later be used with keras.model.fit_generator(). Doing the preprocessing on the entire dataset would highly increase the memory requirements per datapoint (every character except the first 256 is mapped to 257*vocab_size bools), which is unviable for a dataset as big as this one.

# In[ ]:


#Mapping Vocabulary characters to integers and also creating a reverse dict for later:
mapping_dict = {}
mapping_dict_rev = {}
for i, c in enumerate(set(train_string)): 
    mapping_dict[c] = i
    mapping_dict_rev[i] = c 

#Preprocessing the Data inside of a Generator
def training_batch_generator(trainstring, mapping_dict, batchsize, sample_length, vocab_size):
    enc = sklearn.preprocessing.OneHotEncoder(n_values=vocab_size)
    
    fitarray = np.array(list(trainstring[:batchsize*1000]), dtype=np.str)
    fitarray = np.vectorize(mapping_dict.__getitem__)(fitarray)
    fitarray = fitarray.reshape((-1, 1))
    enc.fit(fitarray)
    del fitarray
    
    trainlength = len(trainstring)-1
    
    while True:
        seed = np.random.randint(0, trainlength-sample_length-batchsize-1)
        
        batch_x = []
        batch_y = []
        
        for i in range(batchsize):
            sample_in =  np.array(list(trainstring[seed+i:seed+i+sample_length]), dtype=np.str) 
            sample_in = np.vectorize(mapping_dict.__getitem__)(sample_in)
            sample_in = sample_in.reshape((-1, 1))
            x = enc.transform(sample_in)
            
            sample_out = np.array(list(trainstring[seed+i+sample_length]), dtype=np.str) 
            sample_out = np.vectorize(mapping_dict.__getitem__)(sample_out)
            sample_out = sample_out.reshape((-1, 1))
            y= enc.transform(sample_out)
            
            batch_x.append(x.toarray())
            batch_y.append(y.toarray())
        
        batch_x = np.array(batch_x, dtype=np.bool)
        batch_x = batch_x.reshape((batchsize, sample_length, vocab_size))
        
        batch_y = np.array(batch_y, dtype=np.bool)
        batch_y = batch_y.reshape((batchsize, vocab_size))
        
        yield (batch_x, batch_y)
        
#Generator object for later use
generator_object = training_batch_generator(train_string, mapping_dict, batch_size, sample_length, vocab_size)


# Next, the LSTM model itself is defined. Due to Kernel restrictions, the model has only a few memory units and can obviously be expanded upon. Notice the big learning rate, which is possible do use when training LSTM.

# In[ ]:


#Define the LSTM
print("Generating Model")
lstm_model = ke.models.Sequential()
lstm_model.add(ke.layers.LSTM(256, return_sequences=True, input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.LSTM(128, return_sequences=True, input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.LSTM(128,input_shape=(sample_length, vocab_size)))
lstm_model.add(ke.layers.Dropout(0.3))
lstm_model.add(ke.layers.Dense(vocab_size, activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer=ke.optimizers.Adam(lr=0.001))

lstm_model.summary()


# The next step is to use the model to generate new reviews. I defined a Keras Callback for the review generation, so I get n reviews after every epoch. This is, of course, unviable for deployment of the model, but a good choice for demonstration of training progress.
# 
# The generation itself start on a random part of the training set (pay special attention to how the random generator function doubles as an initiator here). It then proceeds to generate character by character, until it generates the review separator that marks the beginning of a review. The model keeps generating until the next separator to mark the end of the current review. This while loop needs a fallback, in case the model never generates the separator or gets stuck.

# In[ ]:


class LSTMGeneration(ke.callbacks.Callback):
    def __init__(self, generator):
        #The generator should be the same as the one used for training in order to have consistent encoding
        self.generator = generator 
        
    
    def on_epoch_end(self, epoch, logs={}):
        #Settings for the Generator
        file = "lstm_generator.txt"
        generate_n = 1
        
        for n in range(generate_n):
            start = next(self.generator)         
            current_predict = start[0][0]
            
            predicted = 0
            fallback_string = ""
            predict_string = ""
            started = False
            finished = False
            
            while not finished:
                x_predict = np.reshape(current_predict, (1, sample_length, vocab_size))
                pred_raw = self.model.predict(x_predict)
                
                pred = np.zeros((1, vocab_size))
                pred[0, np.argmax(pred_raw)] = 1
                
                
                current_predict = np.vstack([current_predict, pred])
                current_predict = current_predict[1:]
                
                predict_char = mapping_dict_rev[np.argmax(pred_raw)]
                
                fallback_string += predict_char
                if started:
                    predict_string += predict_char
                    
                if predict_char == separator_char:
                    if not started:
                        started = True
                    else:
                        finished = True
                
                predicted += 1
                if (predicted >= 1000 and not started) or (predicted >= 3000):
                    break
                
            with open(file, "a+") as f:
                f.write("Epoch "+str(epoch)+" , Text "+str(n)+"\n")
                if len(predict_string) > 0:
                    f.write(predict_string+"\n\n")
                    print(predict_string+"\n\n")
                else:
                    f.write(fallback_string+"\n\n")
                    print(fallback_string+"\n\n")


# The last remaining step is to actually train the model. Kernel restrictions make good results pretty much impossible, but feel free to copy my code to your local machine and go a bit crazy on model architecture and number of epochs. A lot of the results are quite fun, but I'm not going to post them here, to encourage you to play around.
# 
# Additional Suggestions for Improvement:
# 
# 1: Reduce the vocabulary, there are a lot of weird symbols in there.
# 2:  Increase the Sample Lenght and try random instead of continuous batches
# 3: Experiment with different optimizers, learning rates and learning rate schedules.
# 4: Use Words instead of letters as vocabulary. With many words repeating in most of the reviews, this might be a valid road to take.
# 5: Implement a validation generator and early stopping to reduce overfitting in late stages of training
# 6: Try different model architectures or more memory units
# 
#   [1]: http://friedrich-duge.de/?p=75

# In[ ]:


#Change Steps per Epoch if you want to have denser or sparser monitoring of learning progress 
lstm_model.fit_generator(generator_object, steps_per_epoch=1000, epochs=1, callbacks=[LSTMGeneration(generator_object)], verbose=1)


# Lessons I learned from writing this Kernel:
# 
# 1. LSTM are really useful for replicating character sequences
# 2. Keras fit_generator with a custom generator function is an amazing way to deal with large datasets, especially when preprocessing would drastically increase memory requirements
# 3. Custom callbacks are really useful for keeping track of the models learning process in hacky ways
# 
# Please leave feedback in the comments, and if you get nice results with the Kernel Code, please do share.
