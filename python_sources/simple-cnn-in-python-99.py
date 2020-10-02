import numpy as np
import pandas as pd
import tensorflow as tf


################################
## Part 1
#### Preprocessing
################################

TRAINING_FILE_NAME = '../input/train.csv'

IMG_SIZE = 28 # square images (28x28)
PERCENT_VALIDATION = 0.1 # 90% training, 10% validation (Set to 0.0 for best results) 
CLASS_COUNT = 10 # 10 classes representing digits: [0->9]

def get_image_inputs(csv_data):
    # Get the pixel data from every row of columns 2 -> 785 of the csv
    img_data = csv_data.iloc[:,1:].values 

    # Convert pixel intensity from [0->255] to [0->1]
    return np.multiply(img_data.astype(np.float32), 1./255.)

def get_image_labels(csv_data):
    # Get the first column of every row from the csv
    labels = csv_data.iloc[:,0].values.ravel()
    label_count = labels.shape[0]
    
    # Convert numeric labels to 1-hot vectors
    # Ex: 4 => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    one_hot_vector = np.zeros(label_count*CLASS_COUNT)
    index_offset = np.arange(label_count) * CLASS_COUNT
    one_hot_vector[index_offset + labels.ravel()] = 1
    one_hot_vector = one_hot_vector.reshape(label_count,CLASS_COUNT)
    return one_hot_vector.astype(np.uint8)

def train_data_from_csv(file_name):
    training_data = pd.read_csv(file_name)
    imgs = get_image_inputs(training_data)
    img_labels = get_image_labels(training_data)
    return (imgs, img_labels)

def split_train_validate(all_data, percent_validation):
    input_count = all_data.shape[0]
    split = int(input_count*(1. - percent_validation))
    training_set = all_data[:split]
    validation_set = all_data[split:]
    return training_set, validation_set

all_imgs, all_labels = train_data_from_csv(TRAINING_FILE_NAME)
train_inputs, validate_inputs = split_train_validate(all_imgs, PERCENT_VALIDATION)
train_labels, validate_labels = split_train_validate(all_labels, PERCENT_VALIDATION)


#################################
## Part 2
#### Building the Network
#################################

PATCH_SIZE = 5
CONV_1_SIZE = 32
CONV_2_SIZE = 64
FULL_CONNECT_SIZE = 1024

def weight_variable(shape):
    return tf.Variable( tf.truncated_normal(shape, stddev=0.1) )

def bias_variable(shape):
    return tf.Variable( tf.constant(0.1, shape=shape) )

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Placeholder nodes for inputs 
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE*IMG_SIZE])
# Reshape flat list of pixel values back into "images" 
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 1])

# Variable nodes for 1st convolutional layer 
W_conv1 = weight_variable([PATCH_SIZE,PATCH_SIZE,1,CONV_1_SIZE])
b_conv1 = bias_variable([CONV_1_SIZE])
# 1st convolutional/pooling layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Variable nodes for 2nd convolutional layer 
W_conv2 = weight_variable([PATCH_SIZE,PATCH_SIZE,CONV_1_SIZE,CONV_2_SIZE])
b_conv2 = bias_variable([CONV_2_SIZE])
# 2nd convolutional/pooling layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Variable nodes for 1st fully connected layer (images are now 7x7)
W_fc1 = weight_variable([7 * 7 * CONV_2_SIZE, FULL_CONNECT_SIZE])
b_fc1 = bias_variable([FULL_CONNECT_SIZE])
# 1st fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*CONV_2_SIZE])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add a dropout layer to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Variable nodes for 2nd (final) fully connected layer
W_fc2 = weight_variable([FULL_CONNECT_SIZE, CLASS_COUNT])
b_fc2 = bias_variable([CLASS_COUNT])
# Final (output) layer
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Placeholder nodes for correct labels (used for training)
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_COUNT])


#################################
## Part 3
#### Loss Function, Training Step & Accuracy
#################################

DIGIT_LABEL_AXIS = 1
LEARNING_RATE = 1e-4

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

prediction = tf.argmax(y_conv, DIGIT_LABEL_AXIS)
correct_answer = tf.argmax(y_, DIGIT_LABEL_AXIS)
correct_prediction = tf.equal(prediction, correct_answer)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#################################
## Part 4
#### Training and Evaluation
#################################

BATCH_SIZE = 50
TRAINING_STEPS = 6000 # Increase to ~19k for best results
DROPOUT_KEEP_PROB = 0.5
STEPS_BETWEEN_LOGGING = 100
train_input_len = len(train_inputs)

# Required condition for shuffling to work correctly (see below)
assert train_input_len % BATCH_SIZE == 0

def next_train_batch(training_data, iteration):
    data_count = len(training_data)
    start_index = (iteration*BATCH_SIZE) % data_count 
    return training_data[start_index:start_index+BATCH_SIZE]

# Shuffle 2 lists, keeping them in the same relative order
def shuffle_same(inputs, labels):
    perm = np.random.permutation(len(inputs))
    return inputs[perm], labels[perm]

def model_accuracy(inputs,labels):
    return accuracy.eval(feed_dict={x:inputs, y_:labels, keep_prob:1.0})

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train the model we defined in Parts 2 & 3
for i in range(TRAINING_STEPS):
    batch_inputs = next_train_batch(train_inputs, i)
    batch_labels = next_train_batch(train_labels, i)
    train_step.run(feed_dict={x:batch_inputs, y_:batch_labels, keep_prob:DROPOUT_KEEP_PROB})
    # Shuffle training data after each time we've trained on all of it  
    if (i*BATCH_SIZE) % train_input_len  == 0:
        train_inputs, train_labels = shuffle_same(train_inputs, train_labels)
    if i % STEPS_BETWEEN_LOGGING == 0 and PERCENT_VALIDATION != 0:
        print('Step: %d | Training Accuracy %.10f | Validation Accuracy: %.10f' % 
                (i, model_accuracy(batch_inputs,batch_labels), model_accuracy(validate_inputs,validate_labels)))


##################################
## Part 5
#### Predicting Test Image Labels
##################################

TEST_FILE_NAME = '../input/test.csv'
SUBMISSION_FILE_NAME = 'mnist_test_predictions.csv'

def test_data_from_csv(file_name):
    test_data = pd.read_csv(file_name).values
    # Convert pixel intensity from [0->255] to [0->1]
    return np.multiply(test_data.astype(np.float32), 1./255.)

def write_submission_file(predictions, file_name):
    results = pd.DataFrame(predictions)
    # Index column starts at 0 by default; increment so ImageId goes from 1 to 28000:
    results.index = results.index + 1
    results.to_csv(file_name, header=['Label'], index_label='ImageId')


test_inputs = test_data_from_csv(TEST_FILE_NAME)
test_predictions = prediction.eval(feed_dict={x:test_inputs, keep_prob:1.0})

write_submission_file(test_predictions, SUBMISSION_FILE_NAME)

sess.close()
