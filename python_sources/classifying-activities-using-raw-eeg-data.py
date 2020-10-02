#!/usr/bin/env python
# coding: utf-8

# In my other script, I like at using EEG Power [bands][1]. Here, we use the Raw EEG data. There is more data, but it is completely unstructured. How will this tradeoff allow us to classify what activity the subject is engaged in?
# 
# 
#   [1]: http://support.neurosky.com/kb/development-2/eeg-band-power-values-units-amplitudes-and-meaning

# Imports:

# In[ ]:


import json
import random
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

df = pd.read_csv("../input/eeg-data.csv")


# In[ ]:



def prepare_individual_data(df,individual):
	# drop unused features. just leave eeg_power and the label
	df = df.drop('Unnamed: 0', 1)
	df = df.drop('indra_time', 1)
	df = df.drop('browser_latency', 1)
	df = df.drop('reading_time', 1)
	df = df.drop('attention_esense', 1)
	df = df.drop('meditation_esense', 1)
	df = df.drop('signal_quality', 1)
	df = df.drop('createdAt', 1)
	df = df.drop('updatedAt', 1)
	df = df.drop('eeg_power',1)
	df['raw_values'] = df.raw_values.map(json.loads) #must perform, or else we won't be able to split cell by commas 
	# separate eeg power to multiple columns
	to_series = pd.Series(df['raw_values']) # df to series
	raw_data=pd.DataFrame(to_series.tolist()) #series to list and then back to df
	df = pd.concat([df,raw_data], axis=1, join='outer') # concatenate the create columns
	df = df.drop('raw_values', 1) # drop comma separated cell
	df=df.loc[df['id'] == individual]
	return df


# In[ ]:


individual_data=prepare_individual_data(df,3)
print(individual_data.shape)
print(individual_data.head())
# now we have all raw values for id 3. and labels


# Create function to clean labels, so that labels like "math1" = "math2" = math...

# In[ ]:


def clean_labels(dd):
    #Thanks Alexandru
	dd["label"] = dd["label"].str.replace("\d|Instruction|-|ver", "")
	return dd

cleaned_individual_data = clean_labels(individual_data)


# Drop labels that I don't fully understand and therefore don't care to classify. 

# In[ ]:


def drop_useless_labels(df):
	# drop unlabeled and everyone paired.
	df = df[df.label != 'unlabeled']
	df = df[df.label != 'everyone paired']
	return df

final_individual_full_data= drop_useless_labels(cleaned_individual_data)


# We can see how many labels we have and their frequency.

# In[ ]:


print(final_individual_full_data['label'].value_counts())


# Set aside a test set:

# In[ ]:


# set aside training
def set_aside_test_data(d):
	label=d.pop("label") # pop off labels to new group
	x_train,x_test,y_train,y_test = train_test_split(d,label,test_size=0.2)
    
x_train, x_test, y_train, y_test = set_aside_test_data(final_individual_full_data)


# Check out test and train data after split.

# In[ ]:


print(x_train.shape)
print(x_train.head())
print(y_test.shape)
print(y_test.head())


# Now, since we have limited training data, let's expand it.

# In[ ]:


full_train = x_train.append(y_train)
print(full_train.head())


#for i in range(5):
    # merge x_train and y_train back together
    # copy with noise
    # append
#	copy = x_train
#	copy[0]=copy[0]+random.gauss(1,.1) # add noise to mean freq var
#	final_individual_full_data=final_individual_full_data.append(copy,ignore_index=True) # make voice df 2x as big
#	print("shape of df after {0}th intertion of this loop is {1}".format(i,final_individual_full_data.shape))


# Split back into features and labels. 

# Now we need to convert these pd dataframes to np arrays for tensorflow. 

# In[ ]:


#	train_labels = pd.get_dummies(label) #make labels into one hot vector
#	train_labels = train_labels.values # convert to np array
#	df=individualdata.values # convert features to np array
#	x_train,x_test,y_train,y_test = train_test_split(df,train_labels,test_size=0.2)
#	#so now we have predictors and y values, separated into test and train
#	x_train,x_test,y_train,y_test = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'),np.array(y_train,dtype='float32'),np.array(y_test,dtype='float32')
#	return x_train, x_test, y_train, y_test


# Create mini batch creator function.

# In[ ]:


def get_mini_batch(x,y):
	rows=np.random.choice(x.shape[0], 50)
	return x[rows], y[rows]


# Train NN. 

# In[ ]:


def trainNN(x_train, y_train,x_test,y_test,number_trials):
	# there are 8 features
	# place holder for inputs. feed in later
	x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
	# # # take 20 features  to 10 nodes in hidden layer
	w1 = tf.Variable(tf.random_normal([x_train.shape[1], 1000],stddev=.5,name='w1'))
	# # # add biases for each node
	b1 = tf.Variable(tf.zeros([1000]))
	# # calculate activations 
	hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)
	w2 = tf.Variable(tf.random_normal([1000, y_train.shape[1]],stddev=.5,name='w2'))
	b2 = tf.Variable(tf.zeros([y_train.shape[1]]))
	# # placeholder for correct values 
	y_ = tf.placeholder("float", [None,y_train.shape[1]])
	# # #implement model. these are predicted ys
	y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)
	# loss and optimization 
	loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))
	opt = tf.train.AdamOptimizer(learning_rate=.0009)
	train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])
	# start session
	sess = tf.Session()
	# init all vars
	init = tf.initialize_all_variables()
	sess.run(init)
	ntrials = number_trials
	for i in range(ntrials):
	    # get mini batch
	    a,b=get_mini_batch(x_train,y_train)
	    # run train step, feeding arrays of 100 rows each time
	    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})
	    if i%500 ==0:
	    	print("epoch is {0} and cost is {1}".format(i,cost))
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))


#trainNN(x_train,y_train,x_test,y_test,10000)

