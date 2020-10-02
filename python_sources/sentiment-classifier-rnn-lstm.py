import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#this program is for the NLP task of Classification //

#this function will return the one hot coded inputs as per the 


def one_hot_encode_output(inp):
  i=0
  data_list=[0,0]
  if inp==1:
     data_list==[0,1]
  else:
     data_list=[1,0]
  return np.array(data_list)      

def one_hot_encode(sentence,vocab_list):
  #in this function we will be getting a sentence which needs to be one hot encode
  i=0
  j=0
  input_data=[]
  terms=sentence.split(" ")
  while i<len(terms):
        final_data=[]
        j=0
        while j<200:
           final_data.append(0)
           j=j+1
        j=0
        while j<200:
           if terms[i]==vocab_list[j]:
              final_data[j]=1
              break
           j=j+1
        input_data.append(final_data)
        i=i+1
  input_data=np.array(input_data)
  input_data=input_data.reshape(1,len(terms),200)      
  return input_data
  # if we look at the above output that is being returned by the RNN
  # 1 is the size of the batch that is being returned, lenght of the terms is the mail that is being broken into the words and are one hot encoded
  # 5000 is the size of vocabulary that is being returned, though   
  
  

input_data=pd.read_csv('emails.csv')
x=input_data['text']
y=input_data['spam']
print(x)
print(y)
# the import has been successful

# first we will convert all the data into lower form //
i=0
while i<len(x):
	x[i].lower()
	i=i+1

# we have converted all the data to the lower case #
print(x[0])
# now we will be building up the dictionary #
vocab_words=[]
i=0
while i<len(x):
   j=0
   word_list=str(x[i])
   word_list=word_list.split(" ")
   while j<len(word_list):
    	if (word_list[j]=="Subject:" or word_list[j]==":" or word_list[j]=="," or word_list[j]=="." or word_list[j]=="?" or word_list[j]=="/" or word_list[j]=="-" or word_list[j]=="@" or word_list[j]==">" or word_list[j]=="<"or word_list[j].isnumeric()or word_list[j]==""or word_list[j]==")"or word_list[j]=="(" ):

    		j=j+1
    	else:
    	    vocab_words.append(word_list[j])
    	    j=j+1
   print(i)
   i=i+1

print(len(vocab_words))

#now we will be removing the duplicates from the vocab_words#
vocab_list = [] 
[vocab_list.append(x) for x in vocab_words if x not in vocab_list] 
print(len(vocab_list))

#now we will be making the design of the neural network #
import tensorflow as tf
from tensorflow.keras import layers
model = tf.keras.Sequential()
model.add(layers.LSTM(2048,input_shape=(None,200),dropout=0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print('Our Neural Net has been initialised')
i=0
loss=[]
acc=[]
while i<200:
    print(i)
    x_inp=one_hot_encode(x[i],vocab_list)
    y_out=y[i]
    y_out=y_out.reshape(1,1)
    model.fit(x_inp,y_out)
    i=i+1

model.summary()
y_test=np.array(y[1560])
y_test=y_test.reshape(1,1)
res=model.evaluate(one_hot_encode(x[1560],vocab_list),y_test)



