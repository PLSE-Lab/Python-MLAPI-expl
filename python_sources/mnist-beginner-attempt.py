# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import math
import pandas as  pd







def get_batches(data,batch_size,num_epochs,shuffle=True):



    data=np.array(data)

    data_size=len(data)

    num_batches_per_epoch=int((len(data)-1)/batch_size)+1

    for epoch in range(num_epochs):

        if shuffle:

            shuffle_indices=np.random.permutation(np.arange(data_size))

            shuffled_data=data[shuffle_indices]



        else:

            shuffled_data=data

        for batch_num in range(num_batches_per_epoch):

            start_index=batch_num*batch_size

            end_index=min((batch_num+1)*batch_size,data_size)

            yield shuffled_data[start_index:end_index]



data=pd.read_csv('../input/train.csv').values

y=data[:,0]

x=data[:,1:]

x=x/255.

y_one_hot= np.zeros(shape=(len(y),10))

for i in range(len(data)):

    y_one_hot[i][y[i]]=1.

    

    

indices=np.random.permutation(np.arange(len(y)))



x=x[indices]

y_one_hot=y_one_hot[indices]



dev_sample_index=-1 * int(0.20 * float(len(y)))



x_train,x_dev=x[:dev_sample_index],x[dev_sample_index:]

y_train,y_dev=y_one_hot[:dev_sample_index],y_one_hot[dev_sample_index:]



image_len=int(math.sqrt(x_train.shape[1]))

image_width=int(float(x_train.shape[1])/float(image_len))



x_data=tf.placeholder(shape=[None,784],dtype=tf.float32,name='x')

y_data=tf.placeholder(shape=[None,10],dtype=tf.float32,name='y')



image=tf.reshape(x_data,shape=[-1,image_len,image_width,1])



def get_weights(name,shape):

    return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())



def get_biases(name,shape):

    return tf.zeros(shape=shape,name=name)



def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')



def max_pool(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




with tf.variable_scope('conv_weights1', reuse = tf.AUTO_REUSE) as scope:
    W_conv1=get_weights('W_conv1',[5,5,1,32])

    b_conv1=get_biases('b_conv1',[32])



    h_conv1=tf.nn.relu(conv2d(image,W_conv1)+b_conv1)



    h_pool1=max_pool(h_conv1)

with tf.variable_scope("conv_weights2",reuse=tf.AUTO_REUSE) as scope:

    W_conv2=get_weights('W_conv1',[5,5,32,64])

    b_conv2=get_biases('b_conv1',[64])



    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)



    h_pool2=max_pool(h_conv2)
    
   



W_fc1=get_weights('W_fc',[7*7*64,1024])

b_fc1=get_biases('b_fc',[1024])



h_pool2_flat=tf.reshape(h_pool2,shape=[-1,7*7*64])



h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)



keep_prob=tf.placeholder('float',name='keep_prob')

h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)



W_fc2=get_weights('W_fc1',[1024,10])

b_fc2=get_weights('b_fc1',[10])



scores=tf.matmul(h_fc1_drop,W_fc2)+b_fc2



loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=y_data))



train=tf.train.AdamOptimizer(0.001).minimize(loss)



prediction=tf.argmax(tf.nn.softmax(scores),axis=1,name='prediction')

correct_prediction=tf.equal(prediction,tf.argmax(y_data,axis=1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

init=tf.initialize_all_variables()

saver = tf.train.Saver()

sess=tf.Session()





batches=get_batches(list(zip(x_train,y_train)),batch_size=64,num_epochs=25)





sess.run(init)



for i,batch in enumerate(batches):

    x_batch,y_batch=zip(*batch)

    x_batch=np.array(x_batch)

    y_batch=np.array(y_batch)



    sess.run(train,feed_dict={x_data:x_batch,y_data:y_batch,keep_prob:0.5})



    if(i!=0 and i%100==0):

        print("Loss: {}".format(sess.run(loss,feed_dict={x_data:x_dev,y_data:y_dev,keep_prob:1.0})))

        print("Accuracy: {}".format(sess.run(accuracy,feed_dict={x_data:x_dev,y_data:y_dev,keep_prob:1.0})))



test_data=pd.read_csv('./data/test.csv')



test_data=(test_data.values)/255.



predictions=sess.run(prediction,feed_dict={x_data:test_data,keep_prob:1.0})

indexes=np.linspace(1,len(predictions),len(predictions))

predictions=np.hstack((indexes,predictions))

np.savetxt('test_predictions.csv',predictions,delimiter=',')







# Any results you write to the current directory are saved as output.