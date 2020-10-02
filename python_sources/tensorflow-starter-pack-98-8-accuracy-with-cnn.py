import tensorflow as tf
import numpy as np

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


# load the datasets using your corresponding path
DATASET_TRAIN_PATH = 'yourpath/data/train.csv'
DATASET_TEST_PATH = 'yourpath/data/test.csv'

dataset_features_train, dataset_labels_train = get_dataset_in_np(DATASET_TRAIN_PATH)    # implementing this function is straightforward, so skipping to maintain focus on TensorFlow and CNN
dataset_features_train = dataset_features_train / 255.0 # normalize
dataset_features_test = get_test_dataset_in_np(DATASET_TEST_PATH)   # implementing this function is straightforward, so skipping to maintain focus on TensorFlow and CNN
dataset_features_test = dataset_features_test / 255.0   # normalize

# divide dataset into training and validation sets
dataset_features_validation = dataset_features_train[36000:dataset_features_train.shape[0], :]
dataset_labels_validation = dataset_labels_train[36000:dataset_labels_train.shape[0], :]
dataset_features_train = dataset_features_train[0:36000, :]
dataset_labels_train = dataset_labels_train[0:36000, :]

# reshape dataset_features_train and dataset_features_test to such that can be used by tf.nn.conv2d
dataset_features_train = dataset_features_train.reshape((-1, 28, 28, 1))
dataset_features_validation = dataset_features_validation.reshape((-1, 28, 28, 1))
dataset_features_test = dataset_features_test.reshape((-1, 28, 28, 1))
print('dataset_features_train.shape:', dataset_features_train.shape)

# define some hyperparameters
NUM_EXAMPLES = dataset_features_train.shape[0]
DIM_INPUT_FLAT = dataset_features_train.shape[1]
DIM_INPUT = [28, 28, 1]
DIM_OUTPUT = 10
NUM_EPOCHS = 20
BATCH_SIZE = 36

# placeholders for input and true label output
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, DIM_OUTPUT])

# define the weights with respective shapes
weights = {
	'w1': tf.Variable(tf.random_normal([3, 3, 1, 32])),		#[filter_dim_1, filter_dim_2, num_input_channels, num_output_channels]
	'w2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
	'w3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
	'wfc1': tf.Variable(tf.random_normal([7*7*128, 1024])),
	'wfc2': tf.Variable(tf.random_normal([1024, DIM_OUTPUT])),
}

# define the biases with respective shapes
biases = {
	'b1': tf.Variable(tf.random_normal([32])),
	'b2': tf.Variable(tf.random_normal([64])),
	'b3': tf.Variable(tf.random_normal([128])),
	'bfc1': tf.Variable(tf.random_normal([1024])),
	'bfc2': tf.Variable(tf.random_normal([DIM_OUTPUT]))
}

# construct the convolutional neural network architecture
def convolutional_neural_network(input):
	layer_1_output = tf.nn.conv2d(input, weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b1']
	layer_1_output = tf.nn.relu(layer_1_output)
	layer_1_output = tf.nn.max_pool(layer_1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer_1_output = tf.contrib.layers.batch_norm(layer_1_output)
	layer_2_output = tf.nn.conv2d(layer_1_output, weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b2']
	layer_2_output = tf.nn.relu(layer_2_output)
	layer_2_output = tf.nn.max_pool(layer_2_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer_2_output = tf.contrib.layers.batch_norm(layer_2_output)
	layer_3_output = tf.nn.conv2d(layer_2_output, weights['w3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b3']
	layer_3_output = tf.nn.relu(layer_3_output)
	layer_3_output = tf.nn.max_pool(layer_3_output, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

	layer_3_output = tf.contrib.layers.batch_norm(layer_3_output)
	layer_3_output = tf.reshape(layer_3_output, [-1, 7*7*128])	# flatten layer_4_output
	layer_4_output = tf.add(tf.matmul(layer_3_output, weights['wfc1']), biases['bfc1'])

	layer_5_output = tf.add(tf.matmul(layer_4_output, weights['wfc2']), biases['bfc2'])

	return layer_5_output

logits = convolutional_neural_network(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		# initialize all global variables, which includes weights and biases

	# training start
	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x = get_batch(dataset_features_train, i, BATCH_SIZE)	# get batch of features of size BATCH_SIZE
			batch_y = get_batch(dataset_labels_train, i, BATCH_SIZE)	# get batch of labels of size BATCH_SIZE

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})	# train on the given batch size of features and labels
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		y_predicted = tf.nn.softmax(logits)
		correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
		accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy_validation = accuracy_function.eval({x:dataset_features_validation, y:dataset_labels_validation})
		print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation)
	# training end

	# testing start
	y_predicted = tf.nn.softmax(logits)
	batch_x = get_batch(dataset_features_test, 0, BATCH_SIZE)
	y_predicted_labels = sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})
	i_= 0
	for i in range(1, int(dataset_features_test.shape[0]/BATCH_SIZE)):
		batch_x = get_batch(dataset_features_test, i, BATCH_SIZE)
		y_predicted_labels = np.concatenate((y_predicted_labels, sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})), axis=0)
		i_ = i
	batch_x = get_batch(dataset_features_test, i_+1, BATCH_SIZE)
	y_predicted_labels = np.concatenate((y_predicted_labels, sess.run(tf.argmax(y_predicted, 1), feed_dict={x: batch_x})), axis=0)
	# testing end

# writing predicted labels into a csv file
y_predicted_labels = np.array(y_predicted_labels)
with open('run1.csv','w') as file:	
	file.write('ImageId,Label')
	file.write('\n')

	for i in range(0, y_predicted_labels.shape[0]):
		file.write(str(i+1) + ',' + str(int(y_predicted_labels[i])))
		file.write('\n')

# done with CNN


