"""
Spyder Editor

Author: Muhammad Rifayat Samee(Sanzee)

Current Structure :
    [
            input_layer -----> conv1 ----> conv2 ------> conv3 ----> fullyconnected ----> output_layer
            optimizaer : SGD (stochastic gradient descent)
            regularization: L2 and dropout with keep_probability 0.75
            image has been normalized from [0:255] ---> [0.0:1.0]
            NO max pooling
            flatten the last conv layer before fully connected layer
            L2 regularizatin on every layer
            dropout on fullyconnected layer
    ]
"""

import numpy as np
import pandas as pd
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected,flatten
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

image_size = 28
num_class = 10
convfilter_size_1 = 6
conv_filter_Size_2 = 12
conv_filter_size_3 = 24
fully_size_1 = 200

train_file_dir = '../input/train.csv'
test_fie_dir = '../input/test.csv'
    
def Convolutional_neural_network(input_size,ch):
    
    input_layer = input_data(shape=[None,input_size,input_size,ch],name='input_layer')
    conv1 = conv_2d(input_layer,nb_filter=convfilter_size_1,filter_size=6,strides=1,activation='relu',regularizer='L2')
    #conv1 = max_pool_2d(conv1,2)
    
    conv2 = conv_2d(conv1,nb_filter=conv_filter_Size_2,filter_size=5,strides=2,activation='relu',regularizer='L2')
    #conv2 = max_pool_2d(conv2,2)
    conv3 = conv_2d(conv2,nb_filter=conv_filter_size_3,filter_size=4,strides=2,activation='relu',regularizer='L2')
    
    full_layer1 = fully_connected(flatten(conv3),fully_size_1,activation='relu',regularizer='L2')
    full_layer1 = dropout(full_layer1,0.75)
    
    out_layer = fully_connected(full_layer1,10,activation='softmax')
    
        
    sgd = tflearn.SGD(learning_rate=0.1,lr_decay=0.096,decay_step=100)
    
    top_k = tflearn.metrics.top_k(3)
    
    network = regression(out_layer,optimizer=sgd,metric=top_k,loss='categorical_crossentropy')
    return tflearn.DNN(network,tensorboard_dir='tf_CNN_board',tensorboard_verbose=3)

def make_the_data_ready_conv(Data,Labels):
    Data = Data.reshape(-1,image_size,image_size,1).astype(np.float32)
    Labels = (np.arange(num_class) == Labels[:,None]).astype(np.float32)
    #Data = np.multiply(Data,1.0/255.0)
    return Data,Labels


def get_train_and_validation_data(filename):
    Data = pd.read_csv(filename)
    data = np.array(Data)
    data = np.random.permutation(data)
    X = data[:,1:785].astype(np.float32)
    Y = data[:,0].astype(np.float32)
    return train_test_split(X, Y, test_size=0.20, random_state=42)

def get_accuracy(P,L):
    return (100 * np.sum(np.argmax(P,1) == np.argmax(L,1)))/P.shape[0]

train_x,valid_x,train_y,valid_y = get_train_and_validation_data(train_file_dir)

#print("training shape",train_x.shape)
#print("valid shape",valid_x.shape)
train_x,train_y = make_the_data_ready_conv(train_x,train_y)
valid_x,valid_y = make_the_data_ready_conv(valid_x,valid_y)

test_x = np.array(pd.read_csv(test_fie_dir))
test_x = test_x.reshape(-1,image_size,image_size,1)
#test_x = np.multiply(test_x,1.0/255.0)

#define model

model = Convolutional_neural_network(input_size=28,ch=1)
#training the model
model.fit(train_x,train_y,batch_size=128,validation_set=(valid_x,valid_y),n_epoch=1,show_metric=True)


#Now predict

P = model.predict(test_x)

index = [i for i in range(1,len(P)+1)]
prediction = []
for i in range(len(P)):
    prediction.append(np.argmax(P[i]).astype(np.int))

res = pd.DataFrame({'ImageId':index,'Label':prediction})
res.to_csv("result_cnn.csv",index=False)
