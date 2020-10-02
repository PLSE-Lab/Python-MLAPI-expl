#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Task: Identify digit in the Image

# MNIST data is one of the databases readily available to Machine Learning community for training solutions for various image processing systems. It consists of 60000 training images and 10000 test image. Each record is a 28*28 pixel image of handwritten digit. This kernel gives an overview of some of the ways available in Tensorflow to classify the images available in mnist database. Since there are 10 digits, we have 10 classes in which each image needs to be classified. The Softmax regression approach will compute the probabily of an image belonging to each of these classes. We will then mark the image to be of the digit corresponding to highest probability. This kernel while solving this problem gives a preliminary exposure of the core concepts of Tensorflow including variables, placeholders, graphs, activation functions as well as the estimator API. DNNClassifier of Estimator API uses dense neural network and comparison of various methods will help us to understand why Convolution Neural Networks are the recommended way to solve an image Classification problem. We will compare the various possible approaches from the perspective of performance, time taken to train the model and code required to get the job done. So lets code along and have fun!

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.model_selection as sks
import pandas as pd
import sklearn.metrics as skm
import warnings
import tensorflow.examples.tutorials.mnist as mnist
import timeit

warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


# ## Summary of Findings

# In[ ]:


results = {"Model":["Basic Tensorflow Model","Estimator Linear Regression",
                    "Estimator Dense Neural Network", "Convolutional Neural Network"],
 "Coding Ease":[3,5,5,2], "Accuracy and Speed Of Training":[2,3,1,5], "Speed of Training":[2,3,1,5], 
           "Big Data Performance":[4,2,3, 5], 
           "Color":[(0.7, 0.2, 0.6, 0.6),(0.2, 0.4, 0.6, 0.6),(0.7, 0.7, 0.7, 0.6), (0.4, 0.7, 0.1, 0.6)],
          "param":["Coding Ease","Accuracy and Speed Of Training","Speed of Training"]}
fig = plt.figure(figsize=(17,5), facecolor='white')

for i in [1]: #range(3):
    for k in range(4):
        if (k==0):
           
            plt.plot(np.linspace(0.6, 3.6, 4), results["Accuracy and Speed Of Training"])
            #plt.subplot(1,3,i+1)
            plt.title(results["param"][i], size=20)
            plt.xlabel("Model")
            plt.ylabel("Rating")
            plt.xticks([0,1,2,3,4])
            plt.yticks([1,2,3,4,5,6])
            plt.subplots_adjust(  hspace=5.5)
        plt.bar([k+0.6], results[results["param"][i]][k],width=0.8, color=results["Color"][k])

leg = plt.legend(results["Model"],loc="upper left", bbox_to_anchor=(0.4,0.9))
for i in range(4):
    leg.legendHandles[i].set_color(results["Color"][i])


# ## Load and Visualize Data

# Let us start with creating a function that will load for us training and test data for our sample problem. MNIST data package in tensorflow provides several ready made methods to deal with the data set but I am trying to avoid using them so as to be able to get a grip on how to deal with any image database. I have converted mnist data into a format that we will work with if dealing with any image classification task. The 28X28 pixels of images are flattened out to generate a feature set of length 784. Once we do this image classification task reduces to any other classification task with 784 features. However it is important to note that flattening takes away from us the capability to identify features that relate to contigous  pixels in the two dimesion space. This realization actually paves the ways for CNNs and we can sense intuitively that CNNs should give better results for image classification.

# In[ ]:


def get_data_for_classification():
    mnist_data = mnist.input_data.read_data_sets("../input/", one_hot=True)
    x_train, y_train = mnist_data.train.next_batch(60000)
    x_test, y_test = mnist_data.test.next_batch(10000)
    num_rows = x_train.shape[0] 
    num_features = x_train.shape[1]
    num_classes = y_train.shape[1]
    return x_train, x_test, y_train, y_test, num_rows, num_features, num_classes


# Next it will be good to have a function that gives visual view of the data that we just loaded and see how output specifies class of the corresponding image. For this let us get some random numbers corresponding to image indxes in our train data set. Let us get flattened pixels for these sample images and reshape them to 28X28 dimension space. That is it, we can then plot the reshaped pixel sets to see our handcoded digits that we are learning to read.

# In[ ]:


def plot_data(x_data, y_data, num_rows, num_features):
    #Pi
    num_sample_rows = 9
    img_idx = np.random.randint(0, num_rows, num_sample_rows)
    img_data = x_data[img_idx]
    class_data = y_data[img_idx]

    fig = plt.figure(figsize=(15,15))
    fig.suptitle("Train data set")
    for i in range(num_sample_rows):
        plotname = i + 1 
        plt.subplot(3,3,plotname)
        img = img_data[i].reshape(28,28)
        img_class = np.argmax(class_data[i])
        plt.imshow(img, cmap="Greys")
        plt.title("Number" + str(img_class))


# Let us execute our functions and get an understanding of our data set. 

# In[ ]:


x_train, x_valid, y_train, y_valid, num_rows, num_features, num_classes = get_data_for_classification()


# Let us execute plot_data function couple times to see a variety of images and get an understanding of the data.

# In[ ]:


plot_data(x_train, y_train, num_rows, num_features)


# We can see some pixels are grey, so we expect to see not just 0 or 1 as pixel values but values in between as well. Let us view non zero pixels and validate our understanding. 

# In[ ]:


def get_pixel_values(x_data, num_rows):
    img_idx = np.random.randint(num_rows)
    img = x_data[img_idx]
    img = img.reshape(28,28)
    plt.imshow(img, cmap="Greys")
    img = img[img>0]
    print(img)
    
    
get_pixel_values(x_train, num_rows)


# ## Method 1 - Tensorflow Graph with Gradient Descent optimizing Softmax Cross Entropy

# We will apply several approaches to execute this machine learning task and compare results. The first method is to define and run a graph where in we perform following steps:
# 1. Take as input a batch of image data and define place holders to feed this input.
# 2. Take some initial set of weights. Since input feature set is of dimension 784 and number of clsses is 10, weights shape will need to be (784,10). Bias shape will be (1,10). 
# 3. Define a loss function for gradient descent optimizer. In this case we want to first compute probability of the image belonging to each of the 10 classes. This is what softmax function does.That is instead of saying that the image if of particular digit, it will  return 10 probabilities each corresponding to each of the 10 digits and the total of these probabilities wil be 1. Then we will cmpute the distance of these probabilities from the actual classification wihich is a one hot encoded vector of size 10. 
# 4. Define a graph that puts all the above together and declares an optimizer that minimizes loss.
# 5. Execute graph in tensorflow session.
# 6. Compute accuracy and plot results. 

# Typically any realistic regression problem will have large amount of data and so it would be best to feed data in batches while training the model in each iteration. Having a small batch size does not really impact model performance adversely. It does however however boost training time considerably.

# In[ ]:


def get_data_batch(x, y, batch_size):
    rand_ind = np.random.randint(len(x),size=batch_size)
    return x[rand_ind,:],y[rand_ind,:]


# Placeholders are tensor objects that get values during execution of the graph. They dont have initial values but only know what is their data type and shape. As graph executes, it gets a batch of data on which computation is performed in each iteration. Hence feature data and label data for each batch are declared as placeholders.

# In[ ]:


def get_placeholders(num_features, num_classes, batch_size):
    ph_x = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_features))
    ph_y = tf.placeholder(dtype=tf.float32, shape=(batch_size,num_classes))
    return ph_x, ph_y


# In[ ]:


def get_weights_and_bias(num_features, num_classes):
    weights = np.random.randn(num_features,num_classes)
    bias = np.random.randn(1,10)
    #print("Initial weights:", " ".join(list(weights.flatten().astype(str))), "Initial bias:", bias)
    var_w = tf.Variable(initial_value=weights, dtype=tf.float32)
    var_b = tf.Variable(bias, dtype=tf.float32)
    print(var_w.shape)
    print(var_b.shape)
    return var_w, var_b

Gradient descent minimizes the loss function that we specify. Let us code a loss function that can take as input y_true and y_pred and compute the softmax entropy.
# In[ ]:


def get_loss(y_true, y_pred):
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(entropy)
    return entropy, loss


# We need to define a graph in Tensorflow which after definiton is executed in a tensorflow session. A graph is is essentially a series of connected nodes. A node could be a tensor object or an operation. It will be very good to see the graph on tensor board as it will visually plot what is actually going on.

# In[ ]:


def create_graph(var_w, var_b, ph_x, ph_y, learning_rate):
    y_pred = tf.add(tf.matmul(ph_x, var_w), var_b)
    entropy, loss = get_loss(ph_y, y_pred)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    return train, loss, y_pred, entropy
    

Now that we have all the functions in place, let us call them to define a custom tensor flow model that will create placeholders for the batch data, define the graph, define global variables initializer, and then execute gradient descent optimizer in session to minimize the loss. Global variables initializer is something that must be called before variables can get initialized with the initial value specified for them at the time of declaration. One epoch corresponds to the number of iterations required to loop thught the entire train data. After going to the specified number of iterations we get values of the tained weight variables. These are then used to compute predicted value. Accuracy is computed as the ratio of correct predictions to total size of test data set.
# In[ ]:


def tensorflow_base_model(x_train, y_train, x_valid, y_valid, num_features, num_classes,                               batch_size, num_epochs, learning_rate):
    var_w, var_b = get_weights_and_bias(num_features, num_classes)
    ph_x, ph_y = get_placeholders(num_features, num_classes, batch_size)
    train, loss, y_pred, entropy = create_graph(var_w, var_b, ph_x, ph_y, learning_rate)
    init = tf.global_variables_initializer()
    cur_loss = 0
    num_iter = (num_epochs * len(x_train)) // batch_size
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iter):
            x_batch, y_batch = get_data_batch(x_train, y_train, batch_size)
            feed = {ph_x:x_batch,ph_y:y_batch}
            sess.run(train,feed_dict=feed)
            
        cur_var_w, cur_var_b = sess.run([var_w, var_b])
        cur_y_pred = tf.add(tf.matmul(x_valid, var_w), var_b)
        cur_y_pred = sess.run(cur_y_pred)
        y_valid_argmax = np.argmax(y_valid, 1)
        y_pred_argmax = np.argmax(cur_y_pred, 1)
        y_correct = np.equal(y_valid_argmax, y_pred_argmax)
        acc = y_correct.sum()/y_pred_argmax.shape[0]
    return cur_var_w, cur_var_b, cur_y_pred, acc, y_valid_argmax,y_pred_argmax 


# Next let us have a function ready that will plot predicted values vs actual values and help us validate accuracy of our model.

# In[ ]:


def plot_y_true_vs_y_pred(x, y_true, y_pred):
    pred_data = {"SNo": range(len(y_true)), "Y_true": y_true, "Y_pred": y_pred}
    x = list(x)
    pred_data["X"] = x
    pred_data["Correct"] = np.equal(pred_data["Y_true"], pred_data["Y_pred"] )
    df_pred = pd.DataFrame(pred_data)
    df_pred_summary = df_pred.groupby(["Correct"]).count()
    print(df_pred_summary.head(10))
    num_sample_rows = 9
    df_pred = df_pred[df_pred["Y_true"]==df_pred["Y_pred"]]
    df_pred = df_pred.sample(num_sample_rows)
    i = 1
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("True Value vs Predicted Value")
    for idx,row in df_pred.iterrows():
        plt.subplot(3,3,i)
        img = row["X"].reshape(28,28)
        img_class_true = row["Y_true"]
        img_class_pred = row["Y_pred"]
        plt.imshow(img, cmap="Greys")
        plt.title("Number " + str(img_class_true) + "Prediction:" + str(img_class_pred))
        i = i+1
    


# Executing this base model gives an accuracy of 90.06 percent. The time taken to train this model for 50 epochs is around 3 minutes.

# In[ ]:


def exec_tensorflow_base_model():
    batch_size = 5
    num_epochs = 1 #50
    learning_rate=0.1

    final_w, final_b, final_pred_base_model, acc,y_valid_argmax,y_pred_argmax  =     tensorflow_base_model(x_train, y_train, x_valid, y_valid, num_features, num_classes, batch_size, num_epochs, learning_rate)  

    print("Num Epoch:", num_epochs, " Accuracy:", acc) #, \
           #   " Weights:", " ".join(list(final_w.astype(str).flatten())), " Bias:", final_b)
    plot_y_true_vs_y_pred(x_valid, y_valid_argmax.reshape(len(y_valid)), y_pred_argmax.reshape(len(y_valid)))

timeit.timeit(exec_tensorflow_base_model, number=1)


# The plot of predicted vs actual values for this model shows,  we have done decently well on a data with six features! Quite encouraging indeed. The loss is only 0.001 for 500 epochs and the weights and bias are almost equal to the actual weights. Time take in 20 s. Yes we did have code a bit but the results have been very exciting. We can play with increasing data size by both increasing number of rows as well as number of features. Let us see what happens if we take a case of 10,000 rows and 50 features.

# ## Method 2: Linear Classifier in Tensorflow Estimator API

# Let us now code the required methods for using estimaor API and see whether or not it makes our life easier! Here all we need to do is define feature columns, define input functions for traing and prediction and call the train method to train the model followed by predict method to get the prediction! That is it. Find it quite simple as compared to handcoding the graph for Linear Regression. But what about the performance. Need to execute to figure this out.

# In[ ]:


def get_feat_cols(x_train, x_valid, num_features):
    feat_cols = []
    dic_features_train = {}
    dic_features_valid = {}
    for i in range(num_features):
        col_name = "f" + str(i+1)
        feat_cols.append(tf.feature_column.numeric_column(col_name, shape=(1,)))
        dic_features_train[col_name] = x_train[:,i]
        dic_features_valid[col_name] = x_valid[:,i]
    return feat_cols, dic_features_train, dic_features_valid


# In[ ]:


def get_input_func(dic_features_train, dic_features_valid, y_train, batch_size, num_epochs):
    print(y_train.shape)
    input_fn_train = tf.estimator.inputs.numpy_input_fn(dic_features_train, y_train,                                 batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
    input_fn_valid_pred = tf.estimator.inputs.numpy_input_fn(dic_features_valid, num_epochs=1, shuffle=False)
    return input_fn_train, input_fn_valid_pred


# In[ ]:


def train_and_pred_estimator(estimator,input_fn_train, input_fn_valid_pred, y_valid):
    estimator.train(input_fn_train)
    print("*****************************")
    arr_weights = []
    y_valid_pred = list(estimator.predict(input_fn_valid_pred))
    arr_pred = []
    for pred in y_valid_pred:
        arr_pred.append(pred["class_ids"][0])
    cur_y_pred = np.array(arr_pred)
    y_correct = np.equal(y_valid, cur_y_pred)
    acc = y_correct.sum()/cur_y_pred.shape[0]
    return cur_y_pred, acc


# In[ ]:


def tensorflow_estimator_lr_model(x_train, y_train, x_valid, y_valid, num_features,num_classes,                                batch_size, num_epochs, learning_rate):
    feat_cols, dic_features_train, dic_features_valid = get_feat_cols(x_train, x_valid, num_features)
    input_fn_train, input_fn_valid_pred = get_input_func(dic_features_train, dic_features_valid,                                                          y_train, batch_size, num_epochs)
    print(num_classes)
    estimator = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=num_classes)
    cur_pred,acc = train_and_pred_estimator(estimator,input_fn_train, input_fn_valid_pred, y_valid)
    return cur_pred,acc


# In[ ]:


num_rows = 1000
num_features = 6

x_train, x_valid, y_train, y_valid,num_rows, num_features, num_classes = get_data_for_classification()
print(y_train)
y_train = np.argmax(np.array(y_train), axis=1)
y_valid = np.argmax(np.array(y_valid), axis=1)
print(y_train.shape)
batch_size = 5
num_epochs = 1 #10 I have just set it to one so that I dont have to wait while deploying
learning_rate=0.001
def exec_tensorflow_estimator_lr_model():
    final_pred_elr_model,acc =       tensorflow_estimator_lr_model(x_train, y_train, x_valid, y_valid,                                          num_features, num_classes,batch_size, num_epochs, learning_rate)  
    print("Num Epoch:", num_epochs, " Accuracy:", acc)
    plot_y_true_vs_y_pred(x_valid, y_valid, final_pred_elr_model)
    
timeit.timeit(exec_tensorflow_estimator_lr_model, number=1)


# The model took around an hour to train for 10 epochs! Performance is much better at 92.41 percent!

# ## Method 3: Dense Neural Network Classifier in Tensorflow Estimator API

# This method in terms of codig is very somilar to the Linear Classification method coded above bt under the hood what goes on is very different. I have taken three dense layers each with neurans 7, 14 and 7. As you will see this is very time intensive and practically almost impossible to use on a non GPU machine. 

# In[ ]:


def tensorflow_estimator_dnnlr_model(x_train, y_train, x_valid, y_valid, num_features,num_classes,                                batch_size, num_epochs, learning_rate):
    feat_cols, dic_features_train, dic_features_valid = get_feat_cols(x_train, x_valid, num_features)
    input_fn_train, input_fn_valid_pred = get_input_func(dic_features_train, dic_features_valid,                                                          y_train, batch_size, num_epochs)
    print(num_classes)
    estimator = tf.estimator.DNNClassifier(hidden_units=[7,14,7], feature_columns=feat_cols, 
                                           n_classes=num_classes,
                                          optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=learning_rate
    ), activation_fn=tf.nn.softmax)
    cur_pred,acc = train_and_pred_estimator(estimator,input_fn_train, input_fn_valid_pred, y_valid)
    return cur_pred,acc


# In[ ]:



num_rows = 1000
num_features = 6

x_train, x_valid, y_train, y_valid,num_rows, num_features, num_classes = get_data_for_classification()
print(y_train)
y_train = np.argmax(np.array(y_train), axis=1)
y_valid = np.argmax(np.array(y_valid), axis=1)
print(y_train.shape)
batch_size = 5
num_epochs = 1 #10 I have just set it to 1 so that I dont have to wait while deploying
learning_rate=0.0001
def exec_tensorflow_estimator_dnnlr_model():
    final_pred_elr_model,acc =       tensorflow_estimator_dnnlr_model(x_train, y_train, x_valid, y_valid,                                          num_features, num_classes,batch_size, num_epochs, learning_rate)  
    print("Num Epoch:", num_epochs, " Accuracy:", acc)
    plot_y_true_vs_y_pred(x_valid, y_valid, final_pred_elr_model)
    
timeit.timeit(exec_tensorflow_estimator_dnnlr_model, number=1)


# ## Method 4: Convolutional Neural Network

# As we can see there are two problems with the approaches that we have tried earlier. Since we flatten out the pixels, we lose the ability to extract features related to pixels located close to one another in 2D space. Secondly as the image size grows, the time taken to train is so large that is becomes practically impossible to train the model.

# In[ ]:


def convolutional_layer(x,filter_shape):
    w_init = tf.truncated_normal(filter_shape, stddev=0.1)
    var_w = tf.Variable(initial_value=w_init,dtype=tf.float32)
    #Batch, Height, Width, Channel
    strides_shape = [1,1,1,1]
    conv1 = tf.nn.conv2d(x, var_w, strides=(1,1,1,1), padding="SAME")
    bias_shape = [filter_shape[3]]
    bias_init = tf.truncated_normal(bias_shape, stddev=0.1)
    var_bias = tf.Variable(initial_value=bias_init,dtype=tf.float32)
    conv1_output = tf.nn.relu(conv1 + var_bias)
    return conv1_output


# In[ ]:


def normal_layer(x, num_input_features, num_output_features):
    weights_shape = [num_input_features,num_output_features]
    w_init = tf.truncated_normal(weights_shape, stddev=0.1)
    var_w = tf.Variable(initial_value = w_init, dtype=tf.float32)
    
    bias_shape = [num_output_features]
    bias_init = tf.truncated_normal(bias_shape, stddev=0.1)
    var_bias = tf.Variable(initial_value=bias_init, dtype=tf.float32)
    
    return (tf.add(tf.matmul(x, var_w), var_bias))


# In[ ]:


def tensorflow_cnn_model(x_train, y_train, x_valid, y_valid, num_features, num_classes,                               batch_size, num_epochs, learning_rate):
    ph_x = tf.placeholder(dtype=tf.float32, shape=(None, 28,28,1))
    ph_y = tf.placeholder(dtype=tf.float32, shape=(None,num_classes))
    print(ph_x.shape)
    #Height, Width, inchannels, outchannels
    filter_shape = [6,6,1,32]
    conv1 = convolutional_layer(ph_x, filter_shape)
    print(conv1.shape)
    max_pool1 = tf.nn.max_pool(conv1,ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")
    #Output shape should be (batch_size, 14,14,32)
    print(max_pool1.shape)
    
    filter_shape = [6,6,32,64]
    conv2 = convolutional_layer(max_pool1, filter_shape)
    print(conv2.shape)
    max_pool2 = tf.nn.max_pool(conv2,ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")
    #Output shpe should be (batch_size, 7,7, 64)
    print(max_pool2.shape)
    
    flat_layer = tf.reshape(max_pool2, (-1, 7*7*64))
    print(flat_layer.shape)
    
    norm1 = tf.nn.relu(normal_layer(flat_layer, 7*7*64, 1024))
    print(norm1.shape)
    
    ph_hold_prob = tf.placeholder(tf.float32)
    drop1 = tf.nn.dropout(norm1, keep_prob=ph_hold_prob)
    print(drop1.shape)
    
    y_pred = normal_layer(drop1, 1024, 10)
    print(y_pred.shape)
    
    entropy, loss = get_loss(ph_y, y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    cur_loss = 0
    num_iter = (num_epochs * len(x_train)) // batch_size
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iter):
            x_batch, y_batch = get_data_batch(x_train, y_train, batch_size)
            x_batch = x_batch.reshape(-1,28,28,1)
            feed = {ph_x:x_batch,ph_y:y_batch, ph_hold_prob:0.95}
            sess.run(train,feed_dict=feed)
        
        
        x_batch = x_valid.reshape(-1,28,28,1)
        cur_y_pred = sess.run(y_pred, feed_dict={ph_x:x_batch, ph_hold_prob:1})
        print(cur_y_pred.shape)
        y_valid_argmax = np.argmax(y_valid, 1)
        y_pred_argmax = np.argmax(cur_y_pred, 1)
        y_correct = np.equal(y_valid_argmax, y_pred_argmax)
        acc = y_correct.sum()/y_pred_argmax.shape[0]
    return cur_y_pred, acc, y_valid_argmax,y_pred_argmax 


# In[ ]:


print(y_valid[1])


# In[ ]:


def exec_tensorflow_cnn_model():
    batch_size = 50
    num_epochs = 1 #50
    learning_rate=0.0001
    x_train, x_valid, y_train, y_valid, num_rows, num_features, num_classes = get_data_for_classification()
    final_pred_base_model, acc,y_valid_argmax,y_pred_argmax  =         tensorflow_cnn_model(x_train, y_train, x_valid, y_valid, num_features, num_classes, batch_size, num_epochs, learning_rate)  

    print("Num Epoch:", num_epochs, " Accuracy:", acc) #, \
           #   " Weights:", " ".join(list(final_w.astype(str).flatten())), " Bias:", final_b)
    plot_y_true_vs_y_pred(x_valid, y_valid_argmax.reshape(len(y_valid)), y_pred_argmax.reshape(len(y_valid)))

timeit.timeit(exec_tensorflow_cnn_model, number=1)


# ### Now this is pretty amazing! 97.7 % accuracy in less than 3 minutes! CNN is a undisputed winner!

# In next series we will also experiment with sklearn as well as pre trained models like Inception V3 etc and the likes. It is not over yet!

# In[ ]:




