# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle,sys,os
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split



sns.set(style="darkgrid", palette="deep", font_scale=1.5)
rcParams['figure.figsize']= 14,8
RANDOM_SEED = 42

columns = ["user", "activity","timestamp","x-axis", "y-axis", "z-axis"]
df=pd.read_csv("WISDMData.txt", header=None, names=columns)
df.head()


df["activity"].value_counts().plot(kind="bar", title="Training examples by activity types")
df["user"].value_counts().plot(kind="bar", title="Training examples by user types")

#Removing ";" from z-axis
df=df.applymap(lambda x: str(x).lstrip(";").rstrip(';'))
df['z-axis']= df['z-axis'].str.replace(";","")
df['x-axis']=df['x-axis'].astype(str).astype(float)
df['y-axis']=df['y-axis'].astype(str).astype(float)
df['z-axis']=df['z-axis'].astype(str).astype(float)       
df['timestamp']=df['timestamp'].astype(str).astype(float)

def plot_activity(activity, df):
    data=df[df['activity']==activity][['x-axis', 'y-axis', 'z-axis']][:200]
    
    axis= data.plot(subplots=True, figsize =(16,12), title = activity)
    
    for ax in axis:
            ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5) )
            
    plot_activity("Sitting", df)    
    plot_activity("Standing", df)
    plot_activity("Walking", df)
    plot_activity("Jogging", df)
    
    df.activity.unique()
    df['z-axis'].count()
    len(df.columns)
    df.groupby('activity').apply(lambda x: x.sort_values('user'))
    df.count()
    
    #How much of your data is missing
    df.isnull().sum().sort_values(ascending=False).head()
    
    #impute missing values using imputer in sklearn.preprocessing
   # from sklearn.preprocessing import Imputer
    #imp=Imputer(missing_values='NaN', strategy='most_frequent',axis=0)
    #imp.fit(df.values[:, 3:5])
    #df.values[:, 3:5]=imp.transform(df.values[:, 3:5])
       
    df.dtypes #Listing data types
        
    
    N_TIME_STEPS =200
    N_FEATURES =3
    step =20
    segments = []
    labels= []
    
    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = df['x-axis'].values[i: i+N_TIME_STEPS]
        ys= df['y-axis'].values[i: i+N_TIME_STEPS]
        zs= df['z-axis'].values[i: i+N_TIME_STEPS]
        label= stats.mode(df['activity'][i: i+N_TIME_STEPS])[0][0]
        segments.append([xs,ys,zs])
        labels.append(label)
        
np.array(segments).shape      
reshaped_segments=np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS,N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)    
 
reshaped_segments.shape   
labels[0]
    

#Spliting dataset into training and test set

x_train,x_test,y_train,y_test=train_test_split(reshaped_segments,labels,
                            test_size=0.2, random_state=RANDOM_SEED)

    
#Building the model
N_CLASSES=6
N_HIDDEN_UNITS=64

def create_LSTM_model(inputs):
    W={
       'hidden':tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
       'output':tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
       }    
    biases={
            'hidden':tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
            'output':tf.Variable(tf.random_normal([N_CLASSES]))
            
          }
    x=tf.transpose(inputs, [1,0,2])
    x=tf.reshape(x, [-1, N_FEATURES])
    hidden=tf.nn.relu(tf.matmul(x, W['hidden'])+
    biases['hidden'])
    hidden=tf.split(hidden,N_TIME_STEPS,0)                
            
            # Stack 2 LSTM layers

    lstm_layers=[tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS,
                                        forget_bias=1.0) for _ in range(2)]
    lstm_layers=tf.contrib.rnn.MultiRNNCell(lstm_layers)
    
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
    
    #get output for the last time steps
    lstm_last_output=outputs[-1]
    return tf.matmul(lstm_last_output, W['output'])+biases['output']
    
    
    #Creating placeholder for model
    tf.reset_default_graph()
    x=tf.placeholder(tf.float32, [None, N_TIME_STEPS,N_FEATURES],
                     name='input')
    y=tf.placeholder(tf.float32, [None, N_CLASSES])
    
    
    #Creating Model
    pred_y=create_LSTM_model(x)
    pred_softmax=tf.nn.softmax(pred_y, name="y_")
    
    
    
    L2_LOSS=0.0015
    l2=L2_LOSS * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred_y, labels=y))+l2
    
    LEARNING_RATE=0.0025
    optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    correct_pred=tf.equal(tf.argmax(pred_softmax,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    
    
    #training
    N_EPOCHS=50
    BATCH_SIZE=1024
    
    saver=tf.train.Saver()
    
    history= dict(train_loss=[],
                  train_acc=[],
                  test_loss=[],
                  test_acc=[])
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    train_count = len(x_train)
    for i in range(1, N_EPOCHS+1):
        for start, end in zip(range(0, train_count, BATCH_SIZE), 
                                    range(BATCH_SIZE, train_count+
                                          1, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={x:x_train[start:end],
                                           y: y_train[start:end]})
    _, acc_test, loss_test, sess.run([pred_softmax, accuracy, loss], feed_dict={
            x: x_test, y: y_test})
    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)
    
    
    while i !=1 and i% 10!= 0:
       continue
    printf("epoch: {i} test accuracy: {acc_test} loss: {loss_test}")
    predictions, acc_final, loss_final, = sess.run([pred_softmax,
                accuracy,loss], feed_dict={x: x_test, y:y_test} )
    print()
    printf('final results: accuracy: {acc_final}, loss:{loss_final}')
    
    
    #Storing Model to disc
    
    pickle.dump(predictions, open("predictions.p", "wb"))
    pickle.dump(history, open("history.p", "wb"))
    tf.train.write_graph(Sess.graph_def, '.', './har.pbtxt')
    saver.save(sess, save_path="./har.ckpt")
    sess.close()