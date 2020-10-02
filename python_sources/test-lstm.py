
print("Started..")
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

num_classes=1000

batch_size=3

print("Loaded")

def get_data():

    x1=[x1 for x1 in range(1,1001)]
    x2=[x2 for x2 in range(2,1002)]  
    x3=[x3 for x3 in range(3,1003)]  
    y=[y for y in range(0,1000)] 


    x=list(zip(x1,x2,x3))
    x=np.array(x)
    y=np.array(y)
    x,y=shuffle(x, y, random_state=0)
    return x,y  

def get_network(n_cols, num_classes):
                                   # b_size,n_col,
    net = tflearn.input_data(shape=[None, n_cols] ,name='input')
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net 


X, Y = get_data()
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=0)
n_cols=trainX.shape[1]
print(n_cols)


# Data preprocessing
# Sequence padding

# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)

trainY = to_categorical(trainY,num_classes)
testY = to_categorical(testY,num_classes)


# Training
net=get_network(n_cols,num_classes)

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)


pred=model.predict(np.array([5,6,7]))

print(np.argmax(pred))

# Save it.
# model.save('checkpoints/rnn.tflearn')