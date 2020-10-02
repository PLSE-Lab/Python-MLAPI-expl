import numpy as np
import tensorflow as tf
def identityblock(x,w1,b1,w2,b2,w3,b3):
    out=tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b1)
    out=tf.nn.relu(out)
    out=tf.nn.conv2d(out,w2,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b2)
    out=tf.nn.relu(out)
    out=tf.nn.conv2d(out,w3,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b3)
    return tf.nn.relu(tf.add(x,out))


def convblock(x,w1,b1,w2,b2,w3,b3,w,b):
    out=tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b1)
    out=tf.nn.relu(out)
    out=tf.nn.conv2d(out,w2,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b2)
    out=tf.nn.relu(out)
    out=tf.nn.conv2d(out,w3,strides=[1,1,1,1],padding='SAME')
    out=tf.nn.bias_add(out,b3)
    
    direct=tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    direct=tf.nn.bias_add(direct,b)
    return tf.nn.relu(tf.add(out,direct))
    
    
    

def filter_init(shape):
    return np.random.randn(*shape)*np.sqrt(2/np.prod(shape[:-1]))

batch_sz=300

X=tf.placeholder(np.float32,(batch_sz,28,28,3))
Y=tf.placeholder(np.int32,(batch_sz,10))

#conv
wc1=tf.Variable(filter_init((5,5,3,32)).astype(np.float32))
bc1=tf.Variable(np.zeros(32).astype(np.float32))
#cb
w1c=tf.Variable(filter_init((5,5,32,64)).astype(np.float32))
b1c=tf.Variable(np.zeros(shape=64).astype(np.float32))
w2c=tf.Variable(filter_init((5,5,64,128)).astype(np.float32))
b2c=tf.Variable(np.zeros(shape=128).astype(np.float32))
w3c=tf.Variable(filter_init((5,5,128,128)).astype(np.float32))
b3c=tf.Variable(np.zeros(shape=128).astype(np.float32))

w1d=tf.Variable(filter_init((5,5,32,128)).astype(np.float32))
b1d=tf.Variable(np.zeros(128).astype(np.float32))
#id
w1i=tf.Variable(filter_init((5,5,128,128)).astype(np.float32))
b1i=tf.Variable(np.zeros(shape=128).astype(np.float32))
w2i=tf.Variable(filter_init((5,5,128,254)).astype(np.float32))
b2i=tf.Variable(np.zeros(shape=254).astype(np.float32))
w3i=tf.Variable(filter_init((5,5,254,128)).astype(np.float32))
b3i=tf.Variable(np.zeros(shape=128).astype(np.float32))


w1c2=tf.Variable(filter_init((5,5,128,256)).astype(np.float32))
b1c2=tf.Variable(np.zeros(shape=256).astype(np.float32))
w2c2=tf.Variable(filter_init((5,5,256,512)).astype(np.float32))
b2c2=tf.Variable(np.zeros(shape=512).astype(np.float32))
w3c2=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b3c2=tf.Variable(np.zeros(shape=512).astype(np.float32))

w1d2=tf.Variable(filter_init((5,5,128,512)).astype(np.float32))
b1d2=tf.Variable(np.zeros(512).astype(np.float32))


w1i2=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b1i2=tf.Variable(np.zeros(shape=512).astype(np.float32))
w2i2=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b2i2=tf.Variable(np.zeros(shape=512).astype(np.float32))
w3i2=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b3i2=tf.Variable(np.zeros(shape=512).astype(np.float32))




w1c3=tf.Variable(filter_init((5,5,512,1024)).astype(np.float32))
b1c3=tf.Variable(np.zeros(shape=1024).astype(np.float32))
w2c3=tf.Variable(filter_init((5,5,1024,512)).astype(np.float32))
b2c3=tf.Variable(np.zeros(shape=512).astype(np.float32))
w3c3=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b3c3=tf.Variable(np.zeros(shape=512).astype(np.float32))

w1d3=tf.Variable(filter_init((5,5,512,512)).astype(np.float32))
b1d3=tf.Variable(np.zeros(512).astype(np.float32))

m=500
k=10

conv1=tf.nn.conv2d(X,wc1,strides=[1,1,1,1],padding='SAME')
conv1=tf.nn.bias_add(conv1,bc1)
maxpool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

cb1=convblock(maxpool1,w1c,b1c,w2c,b2c,w3c,b3c,w1d,b1d)
id1=identityblock(cb1,w1i,b1i,w2i,b2i,w3i,b3i)
id2=identityblock(id1,w1i,b1i,w2i,b2i,w3i,b3i)
cb2=convblock(id2,w1c2,b1c2,w2c2,b2c2,w3c2,b3c2,w1d2,b1d2)
id3=identityblock(cb2,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id4=identityblock(id3,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id5=identityblock(id4,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
cb3=convblock(id5,w1c3,b1c3,w2c3,b2c3,w3c3,b3c3,w1d3,b1d3)
id6=identityblock(cb3,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id7=identityblock(id6,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id8=identityblock(id7,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id9=identityblock(id8,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id10=identityblock(id9,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
cb4=convblock(id10,w1c3,b1c3,w2c3,b2c3,w3c3,b3c3,w1d3,b1d3)
id11=identityblock(cb4,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
id12=identityblock(id11,w1i2,b1i2,w2i2,b2i2,w3i2,b3i2)
finalpool=tf.nn.avg_pool(id12,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
sh=finalpool.get_shape().as_list()
resout=tf.reshape(finalpool,[sh[0],np.prod(sh[1:])])

last=resout.get_shape().as_list()[-1]
wd1=tf.Variable((np.random.randn(last,m)*np.sqrt(2/last)).astype(np.float32))
bd1=tf.Variable(np.zeros(m).astype(np.float32))
wd2=tf.Variable((np.random.randn(m,k)*np.sqrt(2/m)).astype(np.float32))
bd2=tf.Variable(np.zeros(k).astype(np.float32))

logits=tf.nn.relu(tf.matmul(resout,wd1)+bd1)
logits=tf.matmul(logits,wd2)+bd2


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))

opt=tf.train.AdamOptimizer(0.001).minimize(cost)




