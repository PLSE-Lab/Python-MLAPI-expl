import tensorflow as tf
import numpy as np
import pandas as pd

LEARNING_RATE = 0.5
GRADIENT_STEPS = 1000

############ HELPER METHODS #######################
def split_data(train):
    num_samples = train.shape[0]
    train_size = (int) (num_samples * 0.9)
    train_samples = train[:train_size]
    validation_samples = train[train_size:]
    train_x = train_samples.drop("label", axis=1)
    train_y = train_samples["label"]
    validation_x = validation_samples.drop("label", axis=1)
    validation_y = validation_samples["label"]
    return train_x, train_y.values, validation_x, validation_y.values

def main():
    ########### INPUT CODE##################
    train = pd.read_csv("../input/train.csv")
    test_x  = pd.read_csv("../input/test.csv")

    #print(train.head()) ###
    #print(train.head()["label"]) ###

    #print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    #print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

    train_x, train_labels, validation_x, validation_labels = split_data(train)
    
    # One hot encode labels
    train_y = np.zeros(shape=(train_labels.size, 10))
    for i in range(train_labels.size):
        train_y[i, train_labels[i]] = 1

    #print("Training features has {0[0]} rows and {0[1]} columns".format(train_x.shape))
    #print("Training labels has {0[0]} rows and {0[1]} columns".format(train_y.shape))

    validation_y = np.zeros(shape=(validation_labels.size, 10))
    for i in range(validation_labels.size):
        validation_y[i, validation_labels[i]] = 1
    #print("Validation features has {0[0]} rows and {0[1]} columns".format(validation_x.shape))
    #print("Validation labels has {0[0]} rows and {0[1]} columns".format(validation_y.shape))
    
    
    ########### TF CODE##################
    # Model code
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
        
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
        
    # Training
    for i in range(GRADIENT_STEPS):
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})
    print("TRAINING DONE!")
        
    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("TRAINING RESULTS!", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    print("VALIDATION RESULTS!", sess.run(accuracy, feed_dict={x: validation_x, y_: validation_y}))

    # Testing
    #test_feed_dict = {x: test_x}
    output = sess.run(tf.argmax(y,1), feed_dict={x: test_x})
    print("OUTPUT:", output)
    print(len(output))
    f = open("output.csv", 'w')
    f.write("ImageId,Label\n")
    for i in range(len(output)):
        f.write(str(i+1) + "," + str(output[i]) + "\n")
    
    # Output
    sess.close()
    print("DONE!!!")

if __name__ == '__main__':
    main()