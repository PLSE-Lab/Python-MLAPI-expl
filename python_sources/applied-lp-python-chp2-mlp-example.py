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

import tensorflow as tf, numpy as np, pandas as pan
import pandas_datareader as data
#tf.reset_default_graph()

def load_data():
    tickers = ["F", "SPY","DIA", "HAL", "MSFT", "SWN", "SJM", "SLG"]
    raw_data = pan.DataFrame()
    
    for i in range(0, len(tickers)):
        print(str((i/float(len(tickers)))*100) + ' percent complete with loading training data...')
        raw_data = pan.concat([raw_data, data.DataReader(tickers[i], data_source = 'yahoo', 
                                                             start = '2010-01-01', 
                                                             end = '2017-01-01')['Close']], axis=1)
        
    #Renaming Coliumns
    raw_data.columns = tickers
    
    #Calculating returns on stocks 
    stock_returns = np.matrix(np.zeros([raw_data.shape[0], raw_data.shape[1]]))
    for j in range(0, raw_data.shape[1]):
        for i in range(0, len(raw_data)-1):
            stock_returns[i,j] = raw_data.ix[i+1,j]/raw_data.ix[i,j] - 1
    
    return stock_returns

train_data = load_data()
print(pan.DataFrame(train_data[0:10, 1:]))

def mlp_model(train_data = train_data, learning_rate = 0.01, num_hidden = 256, epochs = 100):
    n = len(train_data)
    train_x, train_y = train_data[:round(n*0.67),1:], train_data[:round(n*0.67),0]  
    test_x, test_y = train_data[round(n*0.67):,1:], train_data[round(n*0.67):,0]
    X = tf.placeholder('float', shape=(None,7))
    Y = tf.placeholder('float', shape=(None,1))
    nrows = train_x.shape[1]
    weights = {'input':tf.Variable(tf.random_normal([nrows, num_hidden])),
               'hidden1':tf.Variable(tf.random_normal([num_hidden,num_hidden])),
               'output':tf.Variable(tf.random_normal([num_hidden,1]))
              }
    bias = {'input':tf.Variable(tf.random_normal([num_hidden])),
            'hidden1':tf.Variable(tf.random_normal([num_hidden])),
            'output':tf.Variable(tf.random_normal([num_hidden]))
    }
    input_layer = tf.add(tf.matmul(X,weights['input']), bias['input'])
    input_layer = tf.nn.sigmoid(input_layer)
    input_layer = tf.nn.dropout(input_layer,0.2)
    
    hidden_layer = tf.add(tf.multiply(input_layer,weights['hidden1']), bias['hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    hidden_layer = tf.nn.dropout(hidden_layer,0.2)
    
    output_layer = tf.add(tf.multiply(hidden_layer,weights['output']), bias['output'])
    
    error = tf.reduce_sum(tf.pow(output_layer-Y,2)) / len(X)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for _train_x, _train_y in zip(train_x,train_y):
                _, _error = sess.run([optimizer,error],feed_dict={X:_train_y,Y:_train_y})
                #Printing Logging Information 
            print('Epoch ' +  str((i+1)) + ' Error: ' + str(_error))
                    
        #Predicting out of sample
        test_error = []
        for _test_x, _test_y, in zip(test_x, test_y):
            test_error.append(sess.run(error, feed_dict={X:_test_x, Y:_test_y}))
        print('Test Error: ' + str(np.sum(test_error)))
 

if __name__ == '__main__':
    mlp_model() 



    
    
    
    
    
    
