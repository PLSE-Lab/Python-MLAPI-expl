
import numpy as np
import pandas as pd

import tensorflow as tf

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


dataTR = pd.read_csv('../input/train.csv')

images = dataTR.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
labels_flat = dataTR.iloc[:,0].values
labels = dense_to_one_hot(labels_flat, 10)
labels = labels.astype(np.uint8)

dataTE = pd.read_csv('../input/test.csv')
images2 = dataTE.iloc[:,0:].values
images2 = images2.astype(np.float)

sess = tf.InteractiveSession()
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


index =0   
def lote(batch):
  global index
  imas=images[index:index+batch]
  hotVect= labels[index:index+batch]
  index=index+batch
  return imas,hotVect  

x = tf.placeholder(tf.float32, shape=[None, 784]) # definicao do input
y_ = tf.placeholder(tf.float32, shape=[None, 10])#deficao de output para sistema com 10 simbolos
W = tf.Variable(tf.zeros([784,10]))#definicao de pesos
b = tf.Variable(tf.zeros([10]))# definicao de bia ou vies sei la
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b #rede neural em si e so isso mesmo esse modelo
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#definicao de otimizador ,taxa de aprendizado
for _ in range(2000):#quantidade de treinos
  batch =lote(100) # lote de treino/ prepara uma imagem em vetor e um vetor de 10 posicees pra cada caso de treino
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})#treino em si
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(y,y_,1)
predict = tf.argmax(y,1)
print(accuracy.eval(feed_dict={x: images[30001:25555], y_: labels[30001:25555]}))
res=predict.eval(feed_dict={x: images2})
np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(images2)+1),res], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')