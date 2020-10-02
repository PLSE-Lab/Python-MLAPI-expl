# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Let’s import all the required modules
import pylab
#import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import imageio
from sklearn.metrics import accuracy_score
import tensorflow as tf

# To stop potential randomness
seed1 = 128
rng = np.random.RandomState(seed1)

#
root_dir = os.path.abspath('../input')
data_dir = os.path.join(root_dir,'av_identifying the digits','test and train image label')
sub_dir = os.path.join(root_dir,'av_identifying the digits','Images')


# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

#Now let us read our datasets.
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

#Let us see what our data looks like! We read our image and display it.
img_name = rng.choice(train.filename)
filepath = os.path.join(sub_dir,'train',img_name)

img = imread(filepath, flatten=True)
#img1 =imageio.imread(filepath)
img = img.astype('float32')

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

#For easier data manipulation, let’s store all our images as numpy arrays
temp = []
for img_name in train.filename:
    image_path = os.path.join(sub_dir,'train',img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
train_x = np.stack(temp)

temp = []
for img_name in test.filename:
    image_path = os.path.join(sub_dir,'test',img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)

#As this is a typical ML problem, to test the proper functioning of our model we create a validation set. Let’s take a split size of 70:30 for train set vs validation set
split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]


#Now we define some helper functions, which we use later on, in our programs
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

# # # Now comes the main part! Let us define our neural network architecture. We define a neural network with 3 layers;  input, hidden and output. The number of neurons in input and output are fixed, as the input is our 28 x 28 image and the output is a 10 x 1 vector representing the class. We take 500 neurons in the hidden layer. This number can vary according to your need. We also assign values to remaining variables. Read the article on fundamentals of neural network to know more in depth of how it works.
   
### set all variables

# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed1)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed1))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed1)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed1))
}

#Now create our neural networks computational graph
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

#Also, we need to define cost of our neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y))

#And set the optimizer, i.e. our backpropogation algorithm. Here we use Adam, which is an efficient variant of Gradient Descent algorithm. There are a number of other optimizers available in tensorflow (refer here)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#After defining our neural network architecture, let’s initialize all the variables
init = tf.global_variables_initializer()

#Now let us create a session, and run our neural network in the session. We also validate our models accuracy on validation set that we created
with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print ("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))
    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

#To test our model with our own eyes, let’s visualize its predictions
img_name = rng.choice(test.filename)
image_path = os.path.join(sub_dir,'test',img_name)

img = imread(image_path, flatten=True)

test_index = int(img_name.split('.')[0]) - 49000

print ("Prediction is: ", pred[test_index])

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

#We see that our model performance is pretty good! Now let’s create a submission
sample_submission.filename = test.filename

sample_submission.label = pred

sample_submission.to_csv( 'sub01.csv', index=False)
