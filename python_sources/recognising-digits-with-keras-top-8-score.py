#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer
# ## Table of contents
# <b>- Introduction</b><br>
# <b>- Step 1: </b> Load packages and data<br>
# 
# <b>- Step 2:</b> Data preparation<br>
#     <i>- Normalisation</i><br>
#     <i>- Reshape</i><br>
#     <i>- Label encode</i><br>
#     <i>- Splitting data</i><br>
#     
# <b>- Step 3:</b> Modelling<br>
#     <i>- Define model architecture</i><br>
#     <i>- Data augmentation</i><br>
#     <i>- Callback</i><br>
#     <i>- Model fitting and prediction</i><br>
#     
# <b>- Step 4:</b> Submission and conclusion<br>
# 

# ## Introduction
# In this kernel I am going to be taking on the Computer Vision challenge of classifying digits into one of 10 categories, from 0-9. As with nearly all challenges including image data, I will be deploying a Neural Network in order to solve it, specifically, a convolutional neural network. I'll provide a running commentary as I work through this project so it's hopefully easy to follow, and can serve as a helpful for newbies to Deep Learning. 
# 
# I hope you enjoy the read, and please feel free to leave any questions that you may have.
# 
# <i>Additional note: At the time of posting, this model scores in the top 8% of the Leaderboard. Getting into the top 5% would be nice so maybe one day I'll come back and attempt to crack it!</i>

# ## 1. Load packages and data

# In[ ]:


# Data handling, processing and visualisations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Couple of sklearn operations
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

# Deep learning tools from the keras library
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import models
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


# Read in data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Before taking on full data preparation steps, the X_train & y_train variables can firstly be set up. Let's also take a sneak peak of labels in the y_train file. 

# In[ ]:


# Set up y_train
y_train = train["label"]

# Plot graph of labels
g = sns.countplot(y_train)

# Get value counts
y_train.value_counts()

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# Free up memory by deleting the train file
del train


# ## 2. Data preparation
# When using traditional Machine learning techniques, most of the features need to be identified by an domain expert in order to reduce the complexity of the data and make patterns more visible to learning algorithms to work. This is commonly known as the feature engineering step.
# 
# The biggest advantage of Deep Learning algorithms is that they will attempt to learn high-level features from data in an incremental manner. This eliminates the need of domain expertise and hard core feature extraction, which for the researcher can equate to a huge time-saving! The below image illustrates this:

# ![Image of ML Workflow](https://qph.fs.quoracdn.net/main-qimg-11a4efc2c131fe7edf5b4ce5b03d5cfb)

# There are however still some (relatively simple) preparation steps to consider - let's walk through these one by one next.

# ### Normalisation
# Another standard preparation step is to normalise data, this means dividing by the pixel range (in this case 255) so that all data becomes scaled between 0-1. The risk of not normalising data fed into a neural network is that large gradient updates become more likely, and these will typically prevent the network from later converging. Thus, it's best practice to convert data where needed to take on small values, like 0-1, and ensuring all data is homogenous in range (e.g. roughly the same scale is being used).

# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# ### Reshape
# The MNIST digits dataset with structure (nb_samples, 28, 28) i.e. with 2 dimensions per example representing a greyscale image 28x28.
# 
# The Convolution2D layers in Keras however, are designed to work with 3 dimensions per example. They have 4-dimensional inputs and outputs. This covers colour images (nb_samples, nb_channels, width, height), but more importantly, it covers deeper layers of the network, where each example has become a set of feature maps i.e. (nb_samples, nb_features, width, height).
# 
# The greyscale image for MNIST digits input would either need a different CNN layer design (or a param to the layer constructor to accept a different shape), or the design could simply use a standard CNN and you must explicitly express the examples as 1-channel images. The Keras team chose the latter approach, which needs the re-shape.

# In[ ]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# ### Label encode
# Transforming the target variable into a one-hot vector is another typical step taken to support model performance. This allows the network to understand relationships between digits, for instance, that 3 is greater than 2, but smaller than 4. This is completed with the 'to_categorical' function below.

# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = to_categorical(y_train, num_classes = 10)


# ### Splitting data
# As familiar with Machine Learning, data is split into separate sets prior to modelling. When building networks it's usually most helping to create 3 data sets: Training, Validation & Test. The test set will be completely held back until the final prediction is made. Data will be fit to the training data as usual, and then continually monitored through the validation set, which can be useful in helping to detect and mitigate issues with overfitting as the model grows deeper and more complex. Without a validation set it can be very easy to compile a model that fails to generalise well to new, unseen data.
# 

# In[ ]:


# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=random_seed)


# ## 3. Modelling
# In relatively few steps the data is set up and it's now time to build a network to predict the target (number classification). Neural networks come in all shapes and sizes, with different broad families of networks performing better for certain tasks rather than others. To comprehend the scale of recognised networks commonly used to solve complex data problems, see the below image:

# ![Image of ML Workflow](https://cdn-images-1.medium.com/max/2000/1*cuTSPlTq0a_327iTPJyD-Q.png)

# When working on Computer Vision, Convolutional Neural Networks are widely regarded as a top performing network. Reference to the above image, this could be a Deep Convolutional Network. 
# 
# The fundamental difference between a more standard densely connected layer and a convolution layer is this: Dense layers learn global patterns in their input feature space (for example, for a MNIST digit, patterns involving all pixels), whereas convolution layers learn local patterns: in the case of images, patterns found in small 2D windows of the inputs. 
# 
# This key characteristic gives convnets two interesting properties:
# 1. <b>The patterns they learn are translation invariant:</b> After learning a certain pattern in the lower-right corner of a picture, a convolutional network can recognize it anywhere: for example, in the upper-left corner. A densely connected network would have to learn the pattern anew if it appeared at a new location. This makes convolutional networks data efficient when processing images (because the visual world is fundamentally translation invariant): they need fewer training samples to learn representations that have generalization power.
# 2. <b>They can learn spatial hierarchies of patterns:</b> A first convolution layer will learn small local patterns such as edges, a second convolution layer will learn larger patterns made of the features of the first layers, and so on. This allows the network to efficiently learn increasingly complex and abstract visual concepts (because the visual world is fundamentally spatially hierarchical).
# 
# I won't spend any more time dicsussing the model architecture here, however if you are interested in learning more, find below a few articles that really helped to cement my understanding of how Convolutional Networks specifically help to solve Computer Vision problems:
# 1. https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8
# 2. https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050
# 3. https://towardsdatascience.com/components-of-convolutional-neural-networks-6ff66296b456

# ### Define model architecture
# I'll now walk through my final chosen Convolutional Neural Network step by step.
# 
# A common framework for CNN's is as follows:
# <b>Conv -> Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC -> Softmax</b>
# 
# I aim to follow this guideline as much as possible, with a bit of flexing which basically results from trial and error.
# 
# The very first step prior to defining a convolutional layer is to define a sequential model in Keras. Sequential, because we'll be adding layers one at a time.

# In[ ]:


# Set up sequential model
model = models.Sequential()


# Next up, the first convolutional layer is added. To explain each part:
# - Filters: How many observations to make on each digit
# - Kernel_size: How large each filter should (5 by 5 or 3 by 3 are pretty standard sizes)
# - Padding: To account for border effects in a given image. 'Same' means the input image ought to have zero padding so that the output in convolution doesnt differ in size
# - Activation: The activation function for this network, set to 'relu' which is a common choice for such networks
# - Input_shape: Shape of the input, which is already known. This only needs to be specified in the first layer.

# In[ ]:


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))


# Let's add in the second convolutional layer, same as before.

# In[ ]:


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))


# Next up I will add in a Max Pooling layer (the first of two). Max Pooling is a method of downsampling feature maps, which reduces the amount of parameters within the network which thus controls for model <b>overfitting</b>.
# 
# For more information on Max Pooling, check out the below link for a clear and succint summary of this operation in practice:
# https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks

# In[ ]:


model.add(layers.MaxPooling2D((2, 2)))


# I will be adding in further Convolutional layers shortly, however before I will define a Dropout layer of 0.25 (or 25%). The premise of Dropout layers is: <b>'Learning less to learn better'</b>. By adding in a dropout layer, a specified chunk of neurons will be chosen at random and ignored (or 'dropped out') of the training phase. This is another method of overfitting control. It works in a similar fashion to Regularisation within Machine Learning, such that it seeks to reduce interdependent learning amongst neurons within the network. Dropout therefore forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

# In[ ]:


model.add(Dropout(0.25))


# A third convolutional layer will now be added, this time with a larger filter size combined with smaller kernel size. This is in attempt to extract more local features from the input data. The pattern of increasing feature map depth (filters) and decreasing feature map size (kernel_size) is one seen in almost all Convolutional Networks.

# In[ ]:


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))


# While the framework outlined previously did not include a 4th convolutional layer, I am adding one here as a result of trial and error with the current data set. Simply put, my model performed better with this 4th layer. It seems there is still more to learn at this stage!

# In[ ]:


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))


# A second Max Pooling & Dropout layer will now be added.

# In[ ]:


model.add(layers.MaxPooling2D((2, 2)))


# In[ ]:


model.add(Dropout(0.25))


# The next step is to feed the last output tensor (of shape (3, 3, 64)) into a densely connected classifier network: a stack of Dense layers. These classifiers process vectors, which are 1D, whereas the current output is a 3D tensor. Therefore we have to flatten the 3D outputs to 1D, and then add a Dense layer on top. It's not uncommon to add multiple dense layers at this stage - for now I'm sticking with one.

# In[ ]:


model.add(Flatten())
model.add(Dense(256, activation = "relu"))


# Before initiating the final classification layer, a batch normalisation layer will be added. Batch normalization is a type of layer. It works by internally maintaining an exponential moving average of the batch-wise mean and variance of the data seen during training. The main effect of batch normalization is that it helps with gradient propagation, and thus allows for deeper networks.

# In[ ]:


model.add(BatchNormalization())


# Last up, because the problem is multi-class classification, the network will end according to the number of outcome possibilities (in this case, a Dense layer of size 10), followed by a sigmoid activation, which is the most appropriate for multi-class classification problems.

# In[ ]:


model.add(Dense(10, activation = "softmax"))


# With the model set-up, there's just one more step to add before all of the individual elements can be compiled, and that's to add the Optimizer. The model's optimizer uses the Backpropagation algorithm (which is probably the single most important algorithm within Deep Learning) is to use the model's score at each iteration to adjust the weights in a direction that improves the model's ability to predict the outcome. So, the Optimizer is pretty important. There are different optimizers to choose from - I am using RMSprop which is a pretty common choice for classification problems. 

# In[ ]:


my_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Let's now bring it all together!

# In[ ]:


model.compile(optimizer=my_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ### Data Augmentation
# At the current stage the model could be ready for fitting and prediction, however there is one more crucial trick that can be deployed, one that usually yields big gains in model performance: Data Augmentation. This is another technique introduced to combat the ever-present challenge of overfitting, and it comes from the angle that overfitting occurs because of a lack of data to learn from. Thus, augmenting data is the process of randomly generating new training data from the existing samples and transforming in ways that yield different, yet still believable observations. This means when the model is trained, it won't be presented with the same image twice despite copies of the original data being included. The model can then learn about more aspects of the data and ultimately generalise better.
# 
# The below code is the data augmentation set-up, with the desired transformations specified as parameters (e.g. rotation range=10 means a 10 degree rotation at the most will be deployed). Lastly, the augmented data is fit to the training data.**

# In[ ]:


datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             fill_mode='nearest')  

datagen.fit(X_train)


# ### Callbacks
# This is the final step to implement before model fitting. A callback is a method of stopping or altering model training when the validation loss stops improving. Callbacks are handy because they are operations deployed during the model training process, taking actions that could not be predicted prior to the model training process. Examples of callbacks include interrupting training, saving a model or loading a different weight set.
# 
# This callback is a simple to alter the model's learning rate when the validation loss has stopped improving. It's an effective method for helping a model to escape a 'local minimum' when training, thus supporting the goal of reaching the global minimum. For more info on this, read up about the gradient descent process. 
# 
# Below I will initiate the ReduceLROnPlateau function and define some key parameters. Other Kaggle kernel's helped me to arrive at the specific parameter weights in this instance.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# ### Model fit & prediction
# The model is ready, so let's get training. The model will be fit using the 'fit_generator' function since it will include additional augmented data. I'll start with a batch size of 86 (feel free to choose another). I'd also like the model to run over 100 epochs (or iterations), although it is quite likely to converge before this limit. 

# In[ ]:


history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=86),
                              epochs=100, validation_data = (X_val, y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction]) 


# In[ ]:


# Cheking the model's performance
test_loss, test_acc = model.evaluate(X_val, y_val)

test_acc


# Some other result metrics will help to determine the model's accuracy and most importantly, the ability it has to generalise to new data. Below is code to produce a confusion for the model's ability to accurately predict each outcome, followed by plots for accuracy and loss to determine the model's performance at every epoch, and to also check for any signs of overfitting.

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
### End of function definition
    
# Predict the values from the validation dataset
y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))


# In[ ]:


# Plotting accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# No signs of overfitting and pretty accurate model, all told! Let's waste no time and make our predictions on the test data.

# In[ ]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# ## 4. Submission and conclusion
# Let's bundle the newly generated prediction into a submission file for Kaggle.

# In[ ]:


# Concat prediction labels with the Image ID
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

# Create final csv file for submission
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# With this, I have my data ready for submission the Kaggle leaderboard! 
# 
# This kernel has been a demonstration of how with relatively little code, highly accurate networks can be constructed to predict on image data. With minimal set-up, a relatively standard convolutional model archictecture and some added gizmo's in the shape of data augmentation and a learning rate callback, the current model is able to predict handwritten 0-9 digits to ~99% accuracy. In terms of the model's performance relative to others on Kaggle, it features in/around the <b>top 8%</b> on the Leaderboard, which i'm really pleased with. There are more ways in which the model could be improved even further, perhaps i'll come back to these in the future.
# 
# In the meantime, I hope you've enjoyed the read and found the contents helpful. Share an upvote or comment if it did help you, it's always nice to hear. And of course for any questions or suggested improvements, i'm more than happy to hear about these too, so please fire away if you have them! Thanks again for reading and happy Kaggling.
