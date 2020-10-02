
"""
Spyder Editor

Author : Muhammad Rifayat Samee (Sanzee)
"""

import numpy as np
import pandas as pd
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.cross_validation import train_test_split
#train_data,train_labels,test_data,test_labels = mnist.load_data(one_hot = True)
    
#print(train_data.shape())
image_size = 28
num_class = 10

def neural_network_model(input_size):
    
    if input_size <= 0:
        raise ValueError("No input layer size given!!")
    
    input_layer = input_data(shape=[None,input_size],name='input_signal')
    
    full_layer1 = fully_connected(input_layer,512,activation='relu',regularizer='L2')
    full_layer1 = dropout(full_layer1,0.8)
    
    full_layer2 = fully_connected(full_layer1,128,activation='relu',regularizer='L2')
    full_layer2 = dropout(full_layer2,0.8)
    
    out_layer = fully_connected(full_layer2,10,activation='softmax')
    
    sgd = tflearn.SGD(learning_rate=0.1,lr_decay=0.096,decay_step=1000)
    top_k = tflearn.metrics.top_k(3)
    
    network = regression(out_layer,optimizer=sgd,metric=top_k,loss='categorical_crossentropy')
    return tflearn.DNN(network,tensorboard_verbose=0)


def make_the_data_ready(Data,Labels):
    Data = Data.reshape(-1,image_size*image_size).astype(np.float32)
    Labels = (np.arange(num_class) == Labels[:,None]).astype(np.float32)
    return Data,Labels


def get_train_and_validation_data(filename):
    Data = pd.read_csv(filename)
    data = np.array(Data)
    data = np.random.permutation(data)
    X = data[:,1:785].astype(np.float32)
    Y = data[:,0].astype(np.float32)
    return train_test_split(X, Y, test_size=0.10, random_state=42)



train_x,valid_x,train_y,valid_y = get_train_and_validation_data("../input/train.csv")

#print("training shape",train_x.shape)
#print("valid shape",valid_x.shape)
train_x,train_y = make_the_data_ready(train_x,train_y)
valid_x,valid_y = make_the_data_ready(valid_x,valid_y)

test_x = np.array(pd.read_csv("../input/test.csv"))
test_x = test_x.reshape(-1,image_size*image_size).astype(np.float32)

#define model
model = neural_network_model(input_size=image_size*image_size)

#training the model
model.fit(train_x,train_y,validation_set=(valid_x,valid_y),n_epoch=20,show_metric=True)

#Now predict
P = model.predict(test_x)

index = [i for i in range(1,len(P)+1)]
prediction = []
for i in range(len(P)):
    prediction.append(np.argmax(P[i]).astype(np.int))

res = pd.DataFrame({'ImageId':index,'Label':prediction})
res.to_csv("result.csv",index=False)
