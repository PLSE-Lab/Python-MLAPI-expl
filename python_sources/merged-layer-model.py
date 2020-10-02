import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import dropout,input_data,fully_connected,flatten
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.merge_ops import merge,merge_outputs
from scipy.ndimage.filters import gaussian_filter

#importing sets
zalando_mnist_train=pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
zalando_mnist_test=pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

#sorting sets
X_pd,Y=zalando_mnist_train[zalando_mnist_train.columns[1:]],zalando_mnist_train[zalando_mnist_train.columns[0]]
X_pd,Y=zalando_mnist_train[zalando_mnist_train.columns[1:]],zalando_mnist_train[zalando_mnist_train.columns[0]]
x_test_pd,y_test=zalando_mnist_test[zalando_mnist_test.columns[1:]],zalando_mnist_test[zalando_mnist_test.columns[0]]

#reshaping
Y=np.array(Y.values)
y_test=y_test.values

X=np.array([X_pd.values])
X=X.reshape([-1,28,28,1])/255
X-=np.mean(X,axis=0)
X/=np.std(X,axis=0)
#blurr for training
Xblurr=gaussian_filter(X, sigma=0.3)

x_test=x_test_pd.values
x_test=x_test.reshape([-1,28,28,1])/255
x_test-=np.mean(x_test,axis=0)
x_test/=np.std(x_test,axis=0)


#I'm using 2 small convolutional nets and let a third one evaluate on their concatenated output
with tf.Graph().as_default():
    #unforunately tflearns image augmentation and preprocessing can't be used with this structure for some reason
    input_net=input_data([None,28,28,1])
    
    net1=conv_2d(input_net,32,2,activation="relu",weights_init="xavier",bias_init="xavier")
    net1=dropout(net1,0.7)
    net1=fully_connected(net1,256,activation="relu",weights_init="xavier",bias_init="xavier")
    net1=dropout(net1,0.7)
    net1=batch_normalization(net1)
    net1=fully_connected(net1,10,activation="softmax")
    net1= regression(net1, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',to_one_hot=True,n_classes=10, name='targets')
    
    net2=conv_2d(input_net,32,2,activation="relu",weights_init="xavier",bias_init="xavier")
    net2=dropout(net2,0.7)
    net2=fully_connected(net2,256,activation="relu",weights_init="xavier",bias_init="xavier")
    net2=dropout(net2,0.7)
    net2=batch_normalization(net2)
    net2=fully_connected(net2,10,activation="softmax")
    net2= regression(net2, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',to_one_hot=True,n_classes=10, name='targets')
    
    merged_net=merge([net1,net2],mode="concat")
    merged_net=fully_connected(merged_net,10,activation="softmax",weights_init="xavier",bias_init="xavier",regularizer="L2")
    merged_net = regression(merged_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',to_one_hot=True,n_classes=10, name='targets')
    
    
    
    
                      
    m=tflearn.DNN(merged_net)
    m.load("../input/merged-layer-model/m.model")
    #Do not use a validation set on this, it will crash
    #for each net another Y has to be added
    '''
    m.fit(Xblurr, [Y,Y,Y], n_epoch=30,batch_size=256,snapshot_step=10000, show_metric=True)
    '''
    print("evaluation on the training set" ,m.evaluate(X,[Y,Y,Y]))
    print("evaluation on the test set" ,m.evaluate(x_test,[y_test,y_test,y_test]))
    m.save("m.model")
    
    #training the set will result in a overfit on the training set allowing it to score 100% on it, however it's still able to score 92.25% on the test set
    #while the performance on the test set is about 2% lower than a normal convolutional, the rather interesting thing 
    #about it is that it's quite cheap in terms of computing power and able to be trained very quickly even when the 256 node fully connected
    #layers are reduced to 128 it's still quite efficient and even faster
    
    #recommondations on how to reduce the overfitting are welcome
    
    
    
    
    
