import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



train=pd.read_csv('../input/train.csv')
ytr=train.values[:,0]
xtr=train.values[:,1:]

xtr=xtr.astype('float')
xtr=xtr/255
batch=64
learning_rate=0.01
num_units=1024

x=tf.placeholder(tf.float32,[None,28*28])
y_=tf.placeholder(tf.int32,shape=((None,)))
y=tf.one_hot(y_,depth=10)

Weight=tf.Variable(tf.random_normal(shape=[28*28,num_units]),dtype=tf.float32)
Biasis=tf.Variable(tf.zeros(shape=[num_units],dtype=tf.float32))

h1=tf.matmul(x,Weight)+Biasis
h1=tf.nn.sigmoid(h1)
out_weight=tf.Variable(tf.random_normal([num_units,10]))
out_biasis=tf.Variable(tf.zeros([10],dtype=tf.float32)+0.1)
out=tf.matmul(h1,out_weight)+out_biasis

softmax=tf.nn.softmax(out)

loss=tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y)
accuracity=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(softmax,1)),dtype=tf.float32))
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)


for step in range(2000):
    index=np.arange(len(xtr))
    np.random.shuffle(index)
    index=index[:batch]
    sess.run(train,{x:xtr[index],y_:ytr[index]})
    if step%100==0:
        print(sess.run(accuracity,{x:xtr[index],y_:ytr[index]}))