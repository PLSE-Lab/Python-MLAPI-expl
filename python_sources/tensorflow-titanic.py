import csv
import numpy as np
import pandas as pd
import tensorflow as tf

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# We see that 2 passengers embarked data is missing, we fill those in as the most common port, S
train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'

# First we convert some inputs into ints so we can perform mathematical operations
train['gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train['emb'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test['gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['emb'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# There are passengers with missing age data, so we need to fill that in
# We want to get median ages by gender and class
classes = [1, 2 ,3]
genders = [0, 1]
for _class in classes:
    for gender in genders:
        # We locate every passenger of given class and gender whose age is null, and assign it to the median age
        median_age = train[(train.Pclass==_class) & (train.gender==gender)].Age.dropna().median()
        train.loc[(train.Age.isnull()) & (train.gender==gender) & (train.Pclass==_class), 'Age'] = median_age

# Now we have all the data we want. We will not do any feature engineering, 
# as we want the neural net to do that for us
# Now we will format the data so we can use it in tensorflow
dataset = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Survived'], axis=1).values
test_dataset = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1).values

# The labels need to be 1hot encoded
raw_labels = train.Survived.values
labels = np.zeros((dataset.shape[0], 2))
labels[np.arange(dataset.shape[0]), raw_labels] = 1

# Function to print the accuracy of prediction
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# We are going to break up the data into a training and validation set
# As the size of entire dataset is 891, we'll use 267 as validation ~30%
# In this case, data is already randomized, so we don't need to do that
valid_dataset = dataset[:267:, ::]
train_dataset = dataset[267::, ::]
valid_labels = labels[:267]
train_labels = labels[267:]

# Initializing the tf graph
num_possibilities = 2
input_parameters = dataset.shape[1]
batch_size = train_dataset.shape[0]
num_nodes = 1024
num_steps = 3001
graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_parameters))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_possibilities))
    
    tf_valid_dataset = tf.cast(tf.constant(valid_dataset), tf.float32)
    tf_test_dataset = tf.cast(tf.constant(test_dataset), tf.float32)
    
    weights1 = tf.Variable(tf.truncated_normal([input_parameters, num_nodes]))
    biases1 = tf.Variable(tf.zeros([num_nodes]))
    weights2 = tf.Variable(tf.truncated_normal([num_nodes, num_possibilities]))
    biases2 = tf.Variable(tf.zeros(num_possibilities))
    
    lay_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    logits = tf.matmul(lay_1, weights2) + biases2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_lay_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_lay_1, weights2) + biases2)
    test_lay_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(tf.matmul(test_lay_1, weights2) + biases2)
    
    
with tf.Session(graph=graph) as session:
    print('Training Model')
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        batch_data = train_dataset
        batch_labels = train_labels
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        # if step%500 == 0:
        #     print('{} steps complete. Loss = {}'.format(step, l))
        #     print('Batch accuracy = {:.2f}'.format(accuracy(prediction, batch_labels)))
        #     print('Validation accuracy = {:.2f}'.format(accuracy(valid_prediction.eval(), valid_labels)))
    print('Predicting on test data')
    final_prediction = test_prediction.eval()
    
print('Outputting csv')
with open ('titanic_submission.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['PassengerId', 'Survived'])
    for pid, pred in zip(test.PassengerId.values, np.argmax(final_prediction,1)):
        writer.writerow([pid, pred])
    
    
    

    
    
    
    

































