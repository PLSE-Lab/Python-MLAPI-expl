

import tensorflow as tf
import numpy as np


def variable_summaries(var, name):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar( name + '/mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.summary.scalar( name + '/sttdev' , stddev)
		tf.summary.scalar( name + '/max' , tf.reduce_max(var))
		tf.summary.scalar( name + 'min/' , tf.reduce_min(var))
		tf.summary.histogram(name, var)

def fc(tensor, output_dim, IsTrainingMode, name,KP_dropout, act=tf.nn.relu):
	with tf.name_scope(name):
		input_dim = tensor.get_shape()[1].value
		Winit = tf.truncated_normal([input_dim, output_dim], stddev=np.sqrt(2.0/input_dim))
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.constant(0.0, shape=[output_dim])
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.matmul(tensor, W) + B
		tensor = act(tensor)
		if KP_dropout != 1.0:
			tensor = tf.cond(IsTrainingMode,lambda: tf.nn.dropout(tensor, KP_dropout), lambda: tf.identity(tensor))	

	return tensor

def conv(tensor, outDim, filterSize, stride, IsTrainingMode, name,KP_dropout, act=tf.nn.relu):
	with tf.name_scope(name):
		inDimH = tensor.get_shape()[1].value
		inDimW = tensor.get_shape()[2].value
		inDimD = tensor.get_shape()[3].value
		Winit = tf.truncated_normal([filterSize, filterSize, inDimD, outDim], stddev=np.sqrt(2.0/(inDimH*inDimW*inDimD)))
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.constant(0.0, shape=[outDim])
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.nn.conv2d(tensor, W, strides=[1, stride, stride, 1], padding='SAME') + B
		tf.summary.image(name+'_Filtre1', tensor[:,:,:,0:1], 5)
		tensor = act(tensor)
		if KP_dropout != 1.0:
			tensor = tf.cond(IsTrainingMode,lambda: tf.nn.dropout(tensor, KP_dropout), lambda: tf.identity(tensor))	

	return tensor

def maxpool(tensor, poolSize, name):
	with tf.name_scope(name):
		tensor = tf.nn.max_pool(tensor, ksize=(1,poolSize,poolSize,1), strides=(1,poolSize,poolSize,1), padding='SAME')
	return tensor
	
def flat(tensor):
	inDimH = tensor.get_shape()[1].value
	inDimW = tensor.get_shape()[2].value
	inDimD = tensor.get_shape()[3].value
	tensor = tf.reshape(tensor, [-1, inDimH * inDimW * inDimD])
	print ('flat output  ',tensor)
	return tensor

def unflat(tensor, outDimH,outDimW,outDimD):
	tensor = tf.reshape(tensor, [-1,outDimH,outDimW,outDimD])
	tf.summary.image('input', tensor, 5)
	print ('unflat output  ',tensor)
	return tensor	