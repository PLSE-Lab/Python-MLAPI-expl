#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/training/training.csv')  
test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test/test.csv')
lookid_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')


# In[ ]:


train_data.head().T


# In[ ]:


import matplotlib.pyplot as plt

def show_img(image,loc,y_min,y_max):
    plt.imshow(image.reshape(96,96),cmap='gray')
    plt.scatter((loc[0::2]*(y_max-y_min))+y_min, (loc[1::2]*(y_max-y_min))+y_min , marker='x', s=10)
    plt.show()


# In[ ]:


def data_preprocess(train_data,is_test):
    train_data.isnull().any().value_counts()
    train_data=train_data.fillna(method = 'ffill')
    train_data.isnull().any().value_counts()#removing None
    imgs=[]
    for i in range(len(train_data)):#preparing X 
       img=train_data['Image'][i].split(' ')
       img=[0 if x=='' else int(x) for x in img ]
       imgs.append(img)
    imgs=np.array(imgs,dtype = 'float')
    images=imgs.reshape([-1,96,96,1])
    X_train=images/255
    if is_test==True:
        return X_train
    else:
       training=train_data.drop('Image',axis=1)#prepearing y
       y_train=[]
       for i in range(0,len(train_data)):
          y=training.iloc[i,:]
          y_train.append(y)
       y_train=np.array(y_train,dtype = 'float')
       y_min=y_train.min()
       y_max=y_train.max()
       y_train=(y_train-y_train.min())/(y_train.max()-y_train.min())
       #print(y_train.min(),y_train.max())
       return X_train,y_train ,y_min,y_max
    
X_train,y_train,y_min,y_max=data_preprocess(train_data,False)
X_test=data_preprocess(test_data,is_test=True)
show_img(X_train[1],y_train[1],y_min,y_max)


# In[ ]:


import tensorflow as tf
tf.reset_default_graph()

train_graph = tf.Graph()
with train_graph.as_default():
    
    Weights = {
    
    'Wc1' : tf.get_variable('W0', shape = (3, 3, 1, 32), initializer = tf.contrib.layers.xavier_initializer()),
    'Wc2' : tf.get_variable('W1', shape = (3, 3, 32, 64), initializer = tf.contrib.layers.xavier_initializer()),
    'Wc3' : tf.get_variable('W2', shape = (3, 3, 64, 128), initializer = tf.contrib.layers.xavier_initializer()),
    'Wc4' : tf.get_variable('W3', shape = (3, 3, 128, 256), initializer = tf.contrib.layers.xavier_initializer()),
    'Wc5' : tf.get_variable('W4', shape = (3, 3, 256, 512), initializer = tf.contrib.layers.xavier_initializer()),

    #'Wc6' : tf.get_variable('W5', shape = (3, 3, 512, 1024), initializer = tf.contrib.layers.xavier_initializer()),
    #'Wc7' : tf.get_variable('W6', shape = (3, 3, 1024, 2048), initializer = tf.contrib.layers.xavier_initializer()),        
        
    'Wd1' : tf.get_variable('W7', shape = (3 * 3 * 512, 256), initializer = tf.contrib.layers.xavier_initializer()),
    'Wd2' : tf.get_variable('W8', shape = (256, 128), initializer = tf.contrib.layers.xavier_initializer()),

    'Wd3' : tf.get_variable('W9', shape = (128, 68), initializer = tf.contrib.layers.xavier_initializer()),

                            
    'Wd4' : tf.get_variable('W10', shape = (68, 30), initializer = tf.contrib.layers.xavier_initializer())

    }

    Biases = {
    'bc1': tf.get_variable('B0', shape = (32), initializer = tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape = (64), initializer = tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape = (256), initializer = tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable('B4', shape = (512), initializer = tf.contrib.layers.xavier_initializer()),
    
    #'bc6': tf.get_variable('B5', shape = (1024), initializer = tf.contrib.layers.xavier_initializer()),
    #'bc7': tf.get_variable('B6', shape = (2048), initializer = tf.contrib.layers.xavier_initializer()),    
        
    'bd1': tf.get_variable('B7', shape = (256), initializer = tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable('B8', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),

    'bd3': tf.get_variable('B9', shape = (68), initializer = tf.contrib.layers.xavier_initializer()),
        
    
    'bd4': tf.get_variable('B10', shape = (30), initializer = tf.contrib.layers.xavier_initializer()) 
    }

    '''
    
    
    Weights = {   
    
    'Wc1' : tf.Variable(tf.random_normal([3,3,1,32])),
    'Wc2' : tf.Variable(tf.random_normal([3,3,32,64])),
    'Wc3' :tf.Variable(tf.random_normal([3,3,64,128])),
    'Wc4' :tf.Variable(tf.random_normal([3,3,128,256])),
    'Wc5' :tf.Variable(tf.random_normal([3,3,256,512])),

    #'Wc6' : tf.get_variable('W5', shape = (3, 3, 512, 1024), initializer = tf.contrib.layers.xavier_initializer()),
    #'Wc7' : tf.get_variable('W6', shape = (3, 3, 1024, 2048), initializer = tf.contrib.layers.xavier_initializer()),        
            
    'Wd1' : tf.Variable(tf.random_normal([3*3*512,256])),
    'Wd2' : tf.Variable(tf.random_normal([256,128])),

    'Wd3' :tf.Variable(tf.random_normal([128,68])),

                            
    'Wd4' : tf.Variable(tf.random_normal([68,30]))

    }

    Biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([512])),
    
    #'bc6': tf.get_variable('B5', shape = (1024), initializer = tf.contrib.layers.xavier_initializer()),
    #'bc7': tf.get_variable('B6', shape = (2048), initializer = tf.contrib.layers.xavier_initializer()),    
        
    'bd1': tf.Variable(tf.random_normal([256])),
    'bd2': tf.Variable(tf.random_normal([128])),

    'bd3': tf.Variable(tf.random_normal([68])),
        
    
    'bd4': tf.Variable(tf.random_normal([30])) 
    }
    '''
    
    
    
    
    
    
    
def conv2d(input_,W,b,stride=1):
    with train_graph.as_default():

       out=tf.nn.conv2d(input_,W,[1,stride,stride,1],'SAME')
       out=tf.nn.bias_add(out, b)
       out=tf.nn.relu(out)
    
    return out
def maxPooling2d(input_,k=2):
    with train_graph.as_default():

     return tf.nn.max_pool(input_,ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')

def batch_normalization(input_,is_training):
   with train_graph.as_default():

    return tf.compat.v1.layers.batch_normalization(input_, training=is_training)

def conv_net(input_,W,b,stride,k,dropout,is_training):
   with train_graph.as_default():

    conv1=conv2d(input_,W['Wc1'],b['bc1'],stride)
    conv1=tf.nn.dropout(conv1,dropout)
    conv1=batch_normalization(conv1,is_training) 
    conv1=tf.nn.leaky_relu(conv1,alpha=0.1)
    conv1=maxPooling2d(conv1,k)
    conv2=conv2d(conv1,W['Wc2'],b['bc2'],stride)
    conv2=tf.nn.dropout(conv2,dropout)
    conv2=batch_normalization(conv2,is_training) 
    conv2=tf.nn.leaky_relu(conv2,alpha=0.1)
    conv2=maxPooling2d(conv2,k)
    conv3=conv2d(conv2,W['Wc3'],b['bc3'],stride)
    conv3=tf.nn.dropout(conv3,dropout)
    conv3=batch_normalization(conv3,is_training) 
    conv3=tf.nn.leaky_relu(conv3,alpha=0.1)
    conv3=maxPooling2d(conv3,k)
    conv4=conv2d(conv3,W['Wc4'],b['bc4'],stride)
    conv4=tf.nn.dropout(conv4,dropout)
    conv4=batch_normalization(conv4,is_training) 
    conv4=tf.nn.leaky_relu(conv4,alpha=0.1)
    conv4=maxPooling2d(conv4,k)
    conv5=conv2d(conv4,W['Wc5'],b['bc5'],stride)
    conv5=tf.nn.dropout(conv5,dropout)
    conv5=batch_normalization(conv5,is_training) 
    conv5=tf.nn.leaky_relu(conv5,alpha=0.1)
    conv5=maxPooling2d(conv5,k)
    '''
    conv6=conv2d(conv5,W['Wc6'],b['bc6'],stride)
    conv6=tf.nn.dropout(conv6,dropout)
    conv6=batch_normalization(conv6,is_training) 
    
    conv7=conv2d(conv6,W['Wc7'],b['bc7'],stride)
    conv7=tf.nn.dropout(conv7,dropout)
    conv7=batch_normalization(conv7,is_training) 
    '''
    fc1_input=tf.reshape(conv5,[-1,W['Wd1'].shape.as_list()[0]])
    fc1=tf.add(tf.matmul(fc1_input,W['Wd1']),b['bd1'])
    fc1=batch_normalization(fc1,is_training) 
    fc1 = tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,dropout)
    fc2=tf.add(tf.matmul(fc1,W['Wd2']),b['bd2'])
    fc2=batch_normalization(fc2,is_training) 
    fc2 = tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,dropout)
    
    fc3=tf.add(tf.matmul(fc2,W['Wd3']),b['bd3'])
    fc3= tf.nn.relu(fc3)
    fc3=tf.nn.dropout(fc3,dropout)
    
    fc4=tf.add(tf.matmul(fc3,W['Wd4']),b['bd4'])
    
    
    return fc4


# In[ ]:


def generate_batches(X_train,y_train,batch_size):
    if X_train.shape[0]%batch_size==0:
        no_batches=X_train.shape[0]/batch_size
    else :
        no_batches=X_train.shape[0]//batch_size+1
    for batch in range(int(no_batches)):
        
       if batch<no_batches-1:
          yield X_train[batch*batch_size:(batch*batch_size)+batch_size,:],                              y_train[batch*batch_size:(batch*batch_size)+batch_size,:]
       elif batch==no_batches-1 :
           yield X_train[batch*batch_size:,:,:,:],                              y_train[batch*batch_size:,:]  


# In[ ]:


# parameters
learning_rate = 0.001
epochs = 300
batch_size = 256

# network Parameters
no_output = 30  # number of the output neurons
dropout = 0.90 # dropout (probability to keep units)   
#train_graph = tf.Graph()
with train_graph.as_default():
     X_=tf.placeholder(tf.float32,[None,96,96,1],name='input_') 
     y_=tf.placeholder(tf.float32,[None,30],name='out')
     keep_prob = tf.placeholder(tf.float32,name='drop')
     is_training = tf.placeholder(tf.bool,name='training')
     logits=conv_net(X_,Weights,Biases,1,2,keep_prob,is_training)
     cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=logits, labels=y_))
     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
     init=tf.global_variables_initializer()


# In[ ]:


with train_graph.as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        train_losses=[]
        val_losses=[]
        
        for e in range(epochs):
          #batch_no=0
          train_epoch_loss=0
          val_epoch_loss=0
            
          for X,y in generate_batches(X_train,y_train,batch_size):
              #print(X.shape,y.shape)
              dict_= { X_:np.array(X[:-5,:],dtype=np.float32).reshape([-1,96,96,1]),
                       y_:np.array(y[:-5,:],dtype=np.float32).reshape([-1,30]),
                       keep_prob: dropout,
                       is_training:True
                      }
              sess.run(train_opt, feed_dict=dict_)
              train_loss = sess.run(cost, feed_dict=dict_)
            
              dict_= { X_:np.array(X[-5:,:],dtype=np.float32).reshape([-1,96,96,1]),
                       y_:np.array(y[-5:,:],dtype=np.float32).reshape([-1,30]),

                       keep_prob: dropout,
                       is_training:False
                      }
              val_loss = sess.run(cost, feed_dict=dict_)            
            
              train_epoch_loss+=train_loss
              val_epoch_loss+=val_loss
                
          train_losses.append(train_epoch_loss)
          val_losses.append(val_epoch_loss)

          print('Epoch {:>2} - '
              'train Loss: {:>10.4f} '
              'val Loss: {:>10.4f} '.format(
                e + 1,
                train_epoch_loss,
                val_epoch_loss  
                ))
              #batch_no=batch_no+1
        save_path = saver.save(sess, "model_")        


# In[ ]:


import matplotlib.pyplot as plt
      
plt.plot(train_losses,linewidth=1,label="train:")
plt.legend()
plt.grid()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.show()

plt.plot(val_losses,linewidth=1,label="train:")
plt.legend()
plt.grid()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.show()


# In[ ]:


with train_graph.as_default():
    saver = tf.train.Saver()

    with tf.Session() as sess:
       saver.restore(sess, tf.train.latest_checkpoint('./'))
       dict_= { X_:X_test.reshape([-1,96,96,1]),
                    keep_prob: 1.0,
                    is_training: False
                   }
       pred=sess.run(logits, feed_dict=dict_)


# In[ ]:


show_img(X_test[0],pred[0],y_min,y_max)
show_img(X_test[1],pred[1],y_min,y_max)
show_img(X_test[2],pred[2],y_min,y_max)
show_img(X_test[3],pred[3],y_min,y_max)


# In[ ]:


landmark_dict={}
lookid_list = list(lookid_data['FeatureName'])

for f in list(lookid_data['FeatureName']):
    landmark_dict.update({f:lookid_list.index(f)})


# In[ ]:


ImageId = lookid_data["ImageId"]
FeatureName = lookid_data["FeatureName"]
RowId = lookid_data["RowId"]
pred=(pred*(y_max-y_min))+y_min


for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
         if pred[i][j]>96 :
            pred[i][j]=96
         elif pred[i][j]<0:
            pred[i][j]=-pred[i][j]            

            
submit = []
for rowId,irow,landmark in zip(RowId,ImageId,FeatureName):
    submit.append([rowId,pred[irow-1][landmark_dict[landmark]]])
    
submit = pd.DataFrame(submit,columns=["RowId","Location"])
    ## adjust the scale 
print(submit.shape)

submit.to_csv("submision1.csv",index=False) 


# In[ ]:


from IPython.display import FileLink
FileLink(r'submision1.csv')

