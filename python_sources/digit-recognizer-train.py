import tensorflow as tf
import numpy as np
from numpy import genfromtxt

train_file='../input/train.csv'
input_data=genfromtxt(train_file, delimiter=',')
#print(input_data)

label=input_data[1:,0].astype("float32")
image=input_data[1:,1:].astype("float32")
label_processed=np.zeros([len(label),10])
for i in range(len(label)):
	label_processed[i,label[i]]=1.0
print(label_processed.shape,image.shape)

def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

# avoid dead neurons by giving slightly positive init value
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape) 
  return tf.Variable(initial,name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)

#ksize is window size, strides is [batch height width channel]
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x =tf.placeholder(tf.float32, shape=[None,784],name='x')
y_=tf.placeholder(tf.float32, shape=[None,10],name="y_")
x_image = tf.reshape(x, [-1,28,28,1])

# 1st convolution 3x3 with 16 feature maps
W_conv1 = weight_variable([3, 3, 1, 32],name="W_conv1")
b_conv1 = bias_variable([32],name="b_conv1")

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 2nd convolution 3x3 with 32 feature maps
W_conv2 = weight_variable([5, 5, 32, 64],name="W_conv2")
b_conv2 = bias_variable([64],name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# Pooling
h_pool1 = max_pool_2x2(h_conv2)	#28x28 -> 14x14

# 3rd convolution 3x3 with 64 feature maps
W_conv3 = weight_variable([3, 3, 64, 128],name="W_conv3")
b_conv3 = bias_variable([128],name="b_conv3")

h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

# 4th convolution 3x3 with 16 feature maps
W_conv4 = weight_variable([5, 5, 128, 256],name="W_conv4")
b_conv4 = bias_variable([256],name="b_conv4")

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

# Pooling
h_pool2 = max_pool_2x2(h_conv4)

# Fully connected layer_1 3136 -> 1024
W_fc1 = weight_variable([7 * 7 * 256, 1024],name="W_fc1")
b_fc1 = bias_variable([1024],name="b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*256]) # 7*7*64 = 3136
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected layer_2 1024 -> 10
W_fc2 = weight_variable([1024, 10],name="W_fc2")
b_fc2 = bias_variable([10],name="b_fc2")
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

param_list=[W_conv1,b_conv1,W_conv2,b_conv2, W_conv3,b_conv3,W_conv4,b_conv4,W_fc1,b_fc1,W_fc2,b_fc2]
saver=tf.train.Saver(param_list)

batch_size=200


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for i in range(100):
		#acc and accuracy should be named differently to avoid error
		
		avg_acc=0
		j=0
		while j<len(image):
			#print(j)
			batch_image=image[j:j+batch_size]
			batch_label=label_processed[j:j+batch_size]

			feed_dict={x:batch_image,y_:batch_label,keep_prob:0.5}
			
			_,error,acc=sess.run([train_step,cross_entropy,accuracy],feed_dict=feed_dict) 
			
			avg_acc+=acc
			j+=batch_size
			
			if (avg_acc/210)>=0.990:
				saver.save(sess,"/home/gunho/LearnPython/Kaggle/Digit_Recognizer/model/model-%s"%(avg_acc/210))
				print("-------over 99.0%-------")
				print("i: %i"%i)
				print("error: %f"%error)
				print("accuracy: %f"%(avg_acc/210))
		print(i,avg_acc/210)