# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

# Ratio of training to test data split
TRAIN_TEST_SPLIT_RATIO = 0.8
# Initializer seed for weight and biases
INIT_SEED = 1
# Number of neurons in each layer
INPUT_LAYER = 28 * 28
HIDDEN_LAYER_1 = 100
HIDDEN_LAYER_2 = 50
OUTPUT_LAYER = 10

# Returns train dataset as numpy array
def get_train_dataset():
    print("Loading training dataset ....")
    train = pd.read_csv("../input/train.csv")
    train = np.array(train).astype(int)
    np.random.shuffle(train)
    return train[:, 1:], train[:, 0]

# Returns test dataset as numpy array
def get_test_dataset():
    print("Loading test dataset .....")
    test = pd.read_csv("../input/test.csv")
    return np.array(test).astype(int)

# Takes in np array and outputs to csv
def write_predictions(predictions, file_name):
    print("Writing predictions to file ", file_name, "....")
    submission = pd.DataFrame(predictions, columns = ['Label'])
    submission['ImageId'] = np.arange(1, predictions.shape[0] + 1)
    submission = submission[['ImageId', 'Label']]
    submission.head()
    
    submission.to_csv(file_name, index=False)
    return

# Normalises pixel data
def normalise(data):
    data = np.multiply(data, 1.0/255)
    return data.T

# One hot encodes labels into the 10 classes
def one_hot_encoding(labels):
    with tf.Session() as sess:
        one_hot_matrix = tf.one_hot(labels, 10, axis = 0)
        return sess.run(one_hot_matrix)

# Placeholders for input and output
def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=(n_x, None))
    Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None))
    return X, Y

# Initialises weights and biases for all neurons for training
def initialize_parameters():
    W1 = tf.get_variable("W1", [HIDDEN_LAYER_1, INPUT_LAYER],
                         initializer = tf.contrib.layers.xavier_initializer(seed=INIT_SEED))
    W2 = tf.get_variable("W2", [HIDDEN_LAYER_2, HIDDEN_LAYER_1],
                         initializer = tf.contrib.layers.xavier_initializer(seed=INIT_SEED))
    W3 = tf.get_variable("W3", [OUTPUT_LAYER, HIDDEN_LAYER_2],
                         initializer = tf.contrib.layers.xavier_initializer(seed=INIT_SEED))
    b1 = tf.get_variable("b1", [HIDDEN_LAYER_1,1], initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [HIDDEN_LAYER_2,1], initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [OUTPUT_LAYER,1], initializer = tf.zeros_initializer())
    parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3
    }
    return parameters

# One forward propagation step - predicts output based on current parameters and inputs
def forward_propagation(X, parameters, keep_prob):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    Z1_drop = tf.nn.dropout(Z1, keep_prob)
    A1 = tf.nn.relu(Z1_drop)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    Z2_drop = tf.nn.dropout(Z2, keep_prob)
    A2 = tf.nn.relu(Z2_drop)
    
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3

# Loss from prediction
def compute_cost(predicted_y, actual_y):
    logits = tf.transpose(predicted_y)
    labels = tf.transpose(actual_y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def shuffle(X, Y, m):
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10,m))
    return shuffled_X, shuffled_Y

def random_mini_batches(X, Y, mini_batch_size, seed):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    X, Y = shuffle(X, Y, m)
    
    batch_start = 0
    batch_end = mini_batch_size
    
    while batch_start < m:
        batch_x = X[:, batch_start:batch_end]
        batch_y = Y[:, batch_start:batch_end]
        mini_batch = (batch_x, batch_y)
        mini_batches.append(mini_batch)
        batch_start += mini_batch_size
        batch_end += mini_batch_size
        if batch_end > m:
            batch_end = m
    
    return mini_batches

def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
        
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [784, None])
    
    z3 = forward_propagation(x, params, 1)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

# Returns a model ready for prediction
def train(train_x, train_y, test_x, test_y, learning_rate = 0.0001,
          epochs = 300, minibatch_size = 32, keep_prob = 0.95):
    ops.reset_default_graph()
    
    print("Beginning to train ....")
        
    assert train_x.shape[1] == train_y.shape[1]
    assert test_x.shape[1] == test_y.shape[1]
        
    m = train_x.shape[1]
    n_x = train_x.shape[0]
    n_y = train_y.shape[0]
        
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
        
    predicted_y = forward_propagation(X, parameters, keep_prob)
    cost = compute_cost(predicted_y, Y)
    # Back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        
    init = tf.global_variables_initializer()
    
    seed = INIT_SEED
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(epochs + 1):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(train_x, train_y, minibatch_size, seed)
            
            for minibatch in minibatches:
                (batch_X, batch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost],
                                              feed_dict={X: batch_X, Y: batch_Y})
                epoch_cost += minibatch_cost
                
            if epoch%2 == 0:
                print("Epoch cost", epoch, epoch_cost)
                
        parameters = sess.run(parameters)
        print("Model training completed ....")
        
        correct_prediction = tf.equal(tf.argmax(predicted_y), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: train_x, Y: train_y}))
        print ("Test Accuracy:", accuracy.eval({X: test_x, Y: test_y}))
        
        return parameters

# Main script
print("3, 2, 1 .... Let it RIP!")
train_input, train_labels = get_train_dataset()
train_input, validation_input, train_labels, validation_labels = train_test_split(train_input,
                                                                                  train_labels,
                                                                                  train_size=TRAIN_TEST_SPLIT_RATIO)
train_input = normalise(train_input)
train_labels = one_hot_encoding(train_labels)
validation_input = normalise(validation_input)
validation_labels = one_hot_encoding(validation_labels)
print("Input data ready for model training ....")

model = train(train_input, train_labels, validation_input, validation_labels)

print("Running on actual tests...")
test = get_test_dataset()
predictions = predict(test.T, model)
write_predictions(predictions, 'first-run.csv')