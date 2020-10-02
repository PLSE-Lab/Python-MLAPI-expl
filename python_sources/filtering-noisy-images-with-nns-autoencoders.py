#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Auto Encoders are by far one of the most powerful deep learning models that can be used for a variety of purposes. Over time, many kinds of Auto Encoders have come out
# 
# 1. Standard AutoEncoders - Can be used to compress and decompress information (including images)
# 2. Variational AutoEncoders - Can be used to generate images like GANs.
# 3. Denoising AutoEncoders - Can be used to remove noise from images, making them clearer. (They have a much wider use case such as increasing resolution of images as well).
# 
# I was suprised to see a low mention of denoising autoencoders in the MNIST challenge on kaggle, so I decided to make a notebook on this topic using keras. 
# 
# # Why Denoising AutoEncoders
# 
# The obvious use case of denoising autoencoders is to, well, clean noisy images. 
# 
# However, the same model (with different training and testing data) can also be used to increase the resolution of blurry images. Then even photos taken by an amateur on a low resolution camera can be turned into stunning hig res images!
# 
# I have also mentioned another unique use case that I am working on at the end of the notebook
# 
# I am using a simple MNIST dataset as an example, so that it makes a simple tutorial, although it can (and should) be used on other datasets and real life problems as well. I hope you will learn how to implement your own Denoising Autoencoder through this notebook.
# 
# 
# # How Do AutoEncoders Work
# ![autoencoder.png](attachment:autoencoder.png)
# 
# (Image from towardsdatascience)
# 
# As seen from the image, high dimensional input data is fed to the model, which encodes it to a small number of neurons (in this case 3). This part of the network is called the encoder.
# The outputs of these 3 neurons is then fed to layers which increase the dimension of the data to the original size. This way, we are able to compress lots of information in just a few numbers, rather than needing data for every pixel (Think of this as a super compression method using deep learning!). This part is the decoder, which reconstructs the high dimensional data.
# 
# Loss Function: The loss function essentially wants the encoder and decoder to cooperate such that the data reconstructed at the end, is as close as possible to the original input image. This way, we don't need any labels, because the training data is the label.
# 
# And that's it!
# 
# 
# These autoencoders can be used for denoising images (we will using convolutional layers instead since they work better on image data).
# 
# # Implementation in Keras

# In[ ]:


#Importing necessary libraries. I will explain the different layers later.
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1337) #Setting random seed for reproducible results
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Activation, Flatten, Reshape, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as K
print("Tensorflow Version " + tf.__version__)


# First let's see if the GPU is enabled and if tensorflow is able to locate it.

# In[ ]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# As mentioned before, the training data is the target data as well, so when we load in the mnist, we will ignore the labels.

# In[ ]:


#Loading data - we don't want the training labels
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()


# Now, we want the model to filter noise, so the input images will be noisy. We will have to create these from the training data, and we will do that by adding random values to the data.

# In[ ]:


#Normalizing the data between 0 and 1
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal distribution centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

#Even after noise, we don't want values below 0 or more than 1, hence we will 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# Now let's move on to build the encoder and decoder. 
# 
# The encoder will have 3 convolutional layers, with increasing number of filters but decreasing image size (like a standard CNN). We will then flatten it and feed it into a Dense layer, which will finally compress the information into a smaller number of neurons.

# In[ ]:


#Defining how small the middle compression layer should be
latent_dim = 16


# # Encoder

# In[ ]:


#Building the Encoder - Same structure as standard CNN

enc_input = Input(shape=(28, 28, 1))

enc = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu')(enc_input)
enc = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(enc)
enc = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(enc)

enc_shape = K.int_shape(enc)

enc = Flatten()(enc)

#Note that here we want activation to be linear. This is because the denoiser will take in a linear input, and bring it back to image data.
#Using sigmoid or relu will lose more information making it harder for the model to learn.
enc = Dense(latent_dim)(enc)

encoder = Model(inputs=enc_input, outputs=enc)
encoder.summary()


# The denoiser, as expected, will be an exactly flipped version of the encoder. One difference is that we will be using Conv2DTranspose instead of Conv2D. For simplicity, a Conv2DTranspose can be thought of as an inverse to the Conv2D layer, so it can scale back the image and add image elements (instead of understanding image elements).
# 
# Note that the order of the filter sizes will also change. The encoder used 16 --> 32 --> 64. The decoder will use 64 --> 32 --> 16.
# 
# Also, the decoder will have an extra convolutional layer with sigmoid activation purely to reconstruct the image. Thus, there will only be one filter, and stride will also be 1.

# # Decoder

# In[ ]:


#Building the Decoder - We will use Conv2D Transpose instead. Otherwise, Model will be identical.

dec_input = Input(shape=(latent_dim,))

dec = Dense(enc_shape[1] * enc_shape[2] * enc_shape[3])(dec_input)
dec = Reshape((enc_shape[1], enc_shape[2], enc_shape[3]))(dec)

#Note the descending number of filters
dec = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(dec)
dec = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(dec)
dec = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu')(dec)

#Final conv2dtranspose layer to reconstruct image. We are using sigmoid activation since image is normalized between 0 and 1.
#If image was between -1 and 1, we would use tanh activation
dec = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='Same', activation='sigmoid')(dec)

decoder = Model(inputs=dec_input, outputs=dec)
decoder.summary()


# Now we will combine the two models to form the final Denoising AutoEncoder. In Keras this is simple, since we can easily pass the output of 1 model to the input of a second, and create another model out of that.

# # Final AutoEncoder

# In[ ]:


#Building Full Denoising AutoEncoder
#Output of encoder is input to decoder.
denoiser = Model(inputs=enc_input, outputs=decoder(encoder(enc_input)))

#Using Adam Optimizer with default values (learning_rate = 0.001)
from tensorflow.keras.optimizers import Adam
opt = Adam()

#Loss should be mean squared error, since we want the generated output to be as close to original image.
#We also want to penalize large deviations more than small deviations.
denoiser.compile(loss='mse', optimizer=opt)

denoiser.summary()


# Now that the model architecture is complete, we just need to train it. We will let it run for 30 epochs, although 20 - 25 should do good enough. 
# We don't want the model to overfit to noise in every update and make slow progress, so we will use a batch size of 500.

# In[ ]:


#Training Final AutoEncoder

denoiser.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test), epochs=30, batch_size=500)


# In[ ]:


#Retreving Model's predictions for the test data.
recovered = denoiser.predict(x_test_noisy)


# Now let's plot our model's output. However, to make it clear, let's first plot the original image, then the noisy image we generated, and then the filtered image outputed by the AutoEncoder.

# In[ ]:


c = 1
fig=plt.figure(figsize=(8, 8))

for i in range(10):  #This only plots the first 10 images. You can change it to a larget number for more results
    for j in range(3):
        if j == 0:
            img = x_test[i, :, :].reshape(28, 28)
        elif j == 1:
            img = x_test_noisy[i, :, :].reshape(28, 28)
        elif j == 2:
            img = recovered[i, :, :].reshape(28, 28)
        else:
            pass

        fig.add_subplot(10, 3, c)
        plt.imshow(img, cmap='gray')
        c += 1
print("  Real              Noisy              Denoised")
plt.show()


# Wow, these results are genuinely good. The images are extremely noisy, yet the autoencoder is able to clean them with no problem.
# 
# However, to be fair, this is bit of a stretch. The autoencdoer actually modifies the number sometimes (look at the 5 - second last picture - the outline has changed). If you do more digging, you will see that sometimes, a noise 4 will look like a 9 after being clean (the autoencoder adds a small bar connecting the two ends of the 4). Occasionally, the autoencoder also generates faded images, so the number is barely visible.
# 
# But given the amount of noise, this model is actually quite good even on unseen data!

# In[ ]:


#Saving Weights
denoiser.save("denoiser.h5")

#If you don't want to train the model again, you can just use the weights of this model (in the outputs section). Uncomment below line to load in these weights for fresh model.
#denoiser.load_weights(filepath='./denoiser.h5')


# # Further Applications
# 
# I am also experimenting with the idea of using denoising autoencoders to combat adversarial examples, since adversarial examples are also essentially noisy images (where the noise is selected in a carefull way). This is the link to the [github code for that project](https://github.com/Yushgoel/AdversarialAttack/blob/master/adversarial_attack.ipynb) (You should scroll to the bottom for the denoising autoencoder use case).

# Thanks a lot for going through my entire notebook.
# 
# I also used a lot of help from the implementation on the official keras documentation, although this model is deeper and has a few parameters changed.
# 
# Please upvote if you found it helpful.
