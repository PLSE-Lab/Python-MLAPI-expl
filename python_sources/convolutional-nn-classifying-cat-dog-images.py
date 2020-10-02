#!/usr/bin/env python
# coding: utf-8

# **Context: **
# The following project outlines the implementation of a standard Convolutional Neural Network using Python and its popular Deep Learning library, Keras:
# 
#     "Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. It allows for easy and fast prototyping (through user friendliness, modularity, and extensibility), supports both convolutional networks and recurrent networks, as well as combinations of the two, and runs seamlessly on CPU and GPU." - Keras Homepage <https://keras.io/>
#     
# Our goal during this project will be to create a machine that can predict, with at least 80% certainty, whether a given image is that of a cat or a dog. We will train our machine on a set of images containing dogs, as well as a separate set of images containing cats. We will then use this machine to predict whether each image in our test set should be classified as a "Dog" or "Cat" and we will take a look at the accuracy that our machine was able to accomplish in order to try to improve the model.
# 
# The four(4) attached dataset files are as follows: one training set containing 4,000 images of cats, one training set containing 4,000 images of dogs, one test set containing 1,000 images of cats, and one test set containing 1,000 images of dogs. This format will allow us to easily differentiate between the sets and allow our methods/functions to easily reference them when needed. We will also use the .flowFromDirectory() method from Keras' Image Data Generator class to augment the images in the training set. This will prevent the model from being overfitted to this data and generate very similar prediction results for the training set and the test set.

# **Content:  **
# Our first step in building our CNN is to import the necessary packages from Keras. We will discus the significance of each one as they are initialized throughout the project. Do not forget to execute the following cell or future ones will not compile.
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


# Sequential() will be used to specify the type of model that we want to use and initialize the Neural Network with a *sequence* of layers. We will create an object of the Sequential class and refer to it as 'classifier' (because its purpose is to *classify* whether or not a given image is a cat or a dog):

# In[ ]:


classifier = Sequential()


# Now it's time to add the layers. We'll start with the convolution layer, and given that we are working with two-dimensional images (as opposed to videos which have the additional dimension of time), we will be using the Convolution2D class. This allows for the creation of a Feature Map via 3x3 feature detector matrices. The many maps, together, that the class creates result in an entire convolution layer. The parameters of the layer will be as follows: (num_filters, num_rows in feature detector, num_columns in feature detector, input_shape(pixels_per_row, pixels_per_column, num_channels), activation_function_type). Most of the numerical values applied in the following cell are generally industry standard but the reasons why are outside the scope of this project. The number of filters is typically a power of two between 16 and 256 (however, the more filters applied, the more computation will be required). The rectifier activation function is specifically chosen for the categorical parameter in order to make sure that we have all positive values in our feature maps, thus removing any linearity from the image by turning *gradual* color/shade transitions into very sharp, abrupt changes. This make it easier for the machine to detect where an image's features begin and end.
# 
# For more information on Convolution2D, feel free to uncomment the help() function in the following cell.

# In[ ]:


classifier.add(Convolution2D(32, (3, 3), 
                      input_shape=(64,64,3),
                      activation='relu'))
#help(Convolution2D)


# The next layer of the CNN model involves reducing the size of the feature maps by 'pooling' the existing slides into a more condensed version of itself in order to reduce the number of nodes that we will have to flatten in the next step. Otherwise, the model will create one massive one-dimensional vector which would be far too computationally intensive to parse in a reasonable amount of time. Pooling allows us to reduce the time complexity of the model without sacrificing its performance. A pooling size of 2x2 is industry standard:

# In[ ]:


classifier.add(MaxPooling2D(pool_size = (2,2)))


# The next layer, Flattening, involves taking our pooled feature map matrices and literally flattening them into a single vector which contains all of the individual cells within all of the feature maps. This will put our information into a form that that the CPU can easily process; it is essential preparation for the next step which will be the application of a classic Artificial Neural Network composed of fully-connected layers.

# In[ ]:


classifier.add(Flatten())


# The Full Connection step consists of taking our vector (that now contains the image's spacial structure and pixel pattern information) and applying it to an ANN as its input layer. We must now establish a hidden layer ("Fully Connected Layer") by calling the Dense() function with the following parameters: (output_dimensions, activation function). The former must be a number that is sufficiently high enough to make an accurate prediction, but not so high that the model becomes too computationally intensive. The best dimensions are generally somewhere between the number of input nodes and the number of output nodes. We will use 128 hidden nodes in the Fully Connected Layer this time but experimentation with this number can lead to improved results. We will also be using the Rectifier Activation Function again, as we did in the convolution step.

# In[ ]:


classifier.add(Dense(activation = 'relu', 
                    units = 128,))


# The final layer of our Neural Network is the output layer, which is also created with the Dense() function in a very similar manner. However, we will have to change the values of the relevant parameters. We will now be applying the Sigmoid function since we are dealing with a binary outcome (image = cat or dog) and our number of output nodes will only be one since we will expect to output a single probability (the probability of a predicted class, either the dog or the cat).

# In[ ]:


classifier.add(Dense(activation = 'sigmoid',
                    units = 1))


# Now that we're done creating our Neural Network, it is time to compile the model by choosing a stochastic gradient descent algorithm, a loss function, and a performance metric. The optimizer we will use here is called Adam algorithm but other common SGD algorithms can be used, as well, such as AdaGrad and RMSProp. Binary cross-entropy is appropriate here because it corresponds to the logarithmic loss and because we will have a binary outcome (otherwise we would use categorical cross-entropy). The most relevant and most effective measurement of performance for this situation is the very common 'accuracy' metric. 

# In[ ]:


classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


# The final step before getting our result is the Image Preprocessing phase where we will fit our CNN to all of our images. Keras documentation provides templates for this image augmentation process via the aforementioned .flowFromDirectory() method from Keras' Image Data Generator class. As stated in the introduction, augmenting the image files will prevent the model from being overfitted to this data and so that both the training set and the test set generate nearly identical prediction results. We can also use this template to create more sample images and increase our overall number of image files to train on and test with. Make sure to update your path to the dataset if you are using different files from the ones exploited in this project. 

# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../input/training_set/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = validation_generator = test_datagen.flow_from_directory(
        '../input/test_set/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# The following piece of code has been separated from the previous cell because upon executing, it will begin running our machine. This could take minutes to hours, depending on your computing power and the number of epochs that you run. It is recommended that epochs usually be set to at least 25 (preferably 50) but the OP ran 5 here to reduce time for Commit/Publish.

# In[ ]:


classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=2000)


# **Conclusion:  **
# The following is a screenshot of the results when we run 25 epochs. We achieved an accuracy of 84.5% on the training set and 75% on the test set. These results are good but there is definitely room for improvement. The difference between the two is significant so the machine must be performing some overfitting. The solution to this problem is to make a *deeper* deep learning model. To make this happen, we can either add additional convolutional and pooling layers, or we can include additional fully-connected layers. Just one additional convolutional layer results in a test set accuracy of +80%.

# ![Screen%20Shot%202019-03-04%20at%2010.40.21%20PM.png](attachment:Screen%20Shot%202019-03-04%20at%2010.40.21%20PM.png)

# There are a number of other ways to improve the accuracy of the model with parameter tuning, as well. A GPU would be able to process this data much quicker than a regular CPU, which would give us an opportunity to maximize the number of filters, epochs, and layers implemented within the model. Nonetheless, we should be satisfied with an +80% accuracy rate given that we only had access to minimal computational power. 

# 
