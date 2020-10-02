__author__ = 'Hanzhou Wu'

import numpy  as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

weights = {
	'wc1':tf.Variable(tf.truncated_normal([3,3,1,64], stddev=0.1)),
	'wc2':tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1)),

	'wc3':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1)),
	'wc4':tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1)),

	'wc5':tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1)),
	'wc6':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1)),
	'wc7':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1)),
	'wc8':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1)),

	'wc9':tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.1)),
	'wc10':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),
	'wc11':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),
	'wc12':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),

	'wc13':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),
	'wc14':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),
	'wc15':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),
	'wc16':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1)),

	'wd1':tf.Variable(tf.truncated_normal([512,1024], stddev=0.1)),
	'wd2':tf.Variable(tf.truncated_normal([1024,1024], stddev=0.1)),
	'out':tf.Variable(tf.truncated_normal([1024,10], stddev=0.1)),
}

biases = {
	'bc1':tf.Variable(tf.constant(0.1, shape=[64])),
	'bc2':tf.Variable(tf.constant(0.1, shape=[64])),

	'bc3':tf.Variable(tf.constant(0.1, shape=[128])),
	'bc4':tf.Variable(tf.constant(0.1, shape=[128])),

	'bc5':tf.Variable(tf.constant(0.1, shape=[256])),
	'bc6':tf.Variable(tf.constant(0.1, shape=[256])),
	'bc7':tf.Variable(tf.constant(0.1, shape=[256])),
	'bc8':tf.Variable(tf.constant(0.1, shape=[256])),

	'bc9':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc10':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc11':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc12':tf.Variable(tf.constant(0.1, shape=[512])),

	'bc13':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc14':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc15':tf.Variable(tf.constant(0.1, shape=[512])),
	'bc16':tf.Variable(tf.constant(0.1, shape=[512])),

	'bd1':tf.Variable(tf.constant(0.1, shape=[1024])),
	'bd2':tf.Variable(tf.constant(0.1, shape=[1024])),
	'out':tf.Variable(tf.constant(0.1, shape=[10])),
}

def single_conv(x, id):
	return tf.nn.relu(tf.nn.conv2d(x, weights['wc'+str(id)], [1,1,1,1], padding='SAME')+ biases['bc'+str(id)])

def single_pool(x):
	return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='SAME')

def vgg19(input_data, keep_prob):
	x = tf.reshape(input_data, shape=[-1,28,28,1])
	
	x = single_conv(x, 1)
	x = single_conv(x, 2)
	x = single_pool(x)
	print (x.get_shape())

	x = single_conv(x, 3)
	x = single_conv(x, 4)
	x = single_pool(x)
	print (x.get_shape())

	x = single_conv(x, 5)
	x = single_conv(x, 6)
	x = single_conv(x, 7)
	x = single_conv(x, 8)
	x = single_pool(x)
	print (x.get_shape())

	x = single_conv(x, 9)
	x = single_conv(x, 10)
	x = single_conv(x, 11)
	x = single_conv(x, 12)
	x = single_pool(x)
	print (x.get_shape())

	x = single_conv(x, 13)
	x = single_conv(x, 14)
	x = single_conv(x, 15)
	x = single_conv(x, 16)
	x = single_pool(x)
	print (x.get_shape())

	x = tf.reshape(x, [-1, 512])
	x = tf.nn.relu(tf.matmul(x, weights['wd1']) + biases['bd1'])
	x_drop = tf.nn.dropout(x, keep_prob)
	print (x_drop.get_shape())

	x = tf.nn.relu(tf.matmul(x_drop, weights['wd2']) + biases['bd2'])
	x_drop = tf.nn.dropout(x, keep_prob)
	print (x_drop.get_shape())

	x = tf.matmul(x_drop, weights['out']) + biases['out']
	print (x.get_shape())

	return x

X = tf.placeholder('float', shape=[None, 28*28])
y = tf.placeholder('float', shape=[None, 10])
keep_prob = tf.placeholder('float')

pred = vgg19(X, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optm = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

indx = tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1)) # ofcourse, you can use softmax, however, it does not affect the result
accuracy____ = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) # count the total number of correct samples

print ('Reading csv file ...')
df_train = pd.read_csv('../input/train.csv')
df_train = shuffle(df_train)
print (df_train.shape)

df_train_labels = np.array(df_train.pop('label'))
df_train_labels = LabelEncoder().fit_transform(df_train_labels)[:,None]
df_train_labels = OneHotEncoder().fit_transform(df_train_labels).todense()
print (df_train_labels.shape)
df_train_values = df_train.values / 255.0
print (df_train_values.shape)

VALID = 2000
train_data, valid_data = df_train_values[:-VALID,:], df_train_values[-VALID:,:]
train_labels, valid_labels = df_train_labels[:-VALID,:], df_train_labels[-VALID:,:]
print (train_data.shape, valid_data.shape)
print (train_labels.shape, valid_labels.shape)
print ('train data shape = ' + str(train_data.shape) + ' = (TRAIN, WIDTH, WIDTH, CHANNELS)')
print ('train labels shape = ' + str(train_labels.shape) + ' = (TRAIN, LABELS)')

BATCH_SIZE      = 500
TRAIN_BATCH_LEN = int(train_data.shape[0] / BATCH_SIZE)
VALID_BATCH_LEN = int(valid_data.shape[0] / BATCH_SIZE)
print ('---', TRAIN_BATCH_LEN, VALID_BATCH_LEN)
DROPOUT         = 0.5
EPOCHS          = 8

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(EPOCHS):
		for i in range(TRAIN_BATCH_LEN):
			batch_Xs, batch_ys = train_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:], train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
			sess.run(optm, feed_dict={X:batch_Xs, y:batch_ys, keep_prob:DROPOUT})
			now_cost = sess.run(cost, feed_dict={X:batch_Xs, y:batch_ys, keep_prob:1.0})
		if (epoch + 1) % 1 == 0:
			acc = 0.0
			for i in range(VALID_BATCH_LEN):
				batch_Xs, batch_ys = valid_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:], valid_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
				tmp  = accuracy____.eval(feed_dict={X:batch_Xs, y:batch_ys, keep_prob:1.0})
				acc += tmp
			print ('Validation %d: %.6f'%(epoch+1, acc / valid_data.shape[0]))

	print ('Rading test data ...')
	df_test = pd.read_csv('../input/test.csv')
	print (df_test.shape)
	test_data = df_test.values / 255.0
	TEST_BATCH_LEN = int(test_data.shape[0] / BATCH_SIZE)
	test_labels = np.array([])
	for i in range(TEST_BATCH_LEN):
		batch_Xs    = test_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
		cur_labels  = sess.run(indx, feed_dict={X:batch_Xs,keep_prob:1.0})
		test_labels = np.append(test_labels, cur_labels)
	test_labels = test_labels.astype(int)
	submission  = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
	submission.to_csv('../output/tf_vgg.csv', index=False)
	print (submission.tail())
