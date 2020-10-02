#!/usr/bin/env python
# coding: utf-8

# # Image Colorization
# 
# The purpose of this notebook is to attempt to create an Image Colorization convolutional neural network.
# The objective is to create quality colorization, while learning along the way the variables which affect the results the most
# and trying to optimize the net to the best of my ability.
# 
# 
# I am still just a beginner with neural nets, but will try to create a starting point for image colorization, and provide an explanation to my results and decisions throughout the process of attempting Image Colorization.
# 
# This example will only train for 30 epochs and show results for that. Some images from training the same model for much longer (around 10 hours) will be shown as well, to examplify what the model can achieve. The model trained for 10 hours was also trained on many more images than the model shown in this example.
# 
# Even after more training on a larger dataset, the results vary from image to image, some good and poor results will be shown to give a better idea of the net at that point. 

# # Imports

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Input, Concatenate, Activation, concatenate
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import PIL
from PIL import Image
import random
import h5py
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# # Creating the model
# 
# Originally the model was a MobileNet pretrained model as an encoder, with a Conv2DTranspose based decoder attached at the end. While having a couple successful images, the model mostly produced brown images, as it was the simplest solution to provide the average color throughout the image. The model did not seem to move past this local minimum despite more training. This appeared as a sudden drop in Loss overtime, followed by a 'flatline' showing little improvement.
# 
# After further research, this [research paper](https://arxiv.org/abs/1712.03400) was found which used a combination of a pretrained model (Inception-Resnet-v2) and a CNN encoder-decoder.
# The model shown in the paper seemed more successful, and it was decided to follow their concept and combined MobileNetV2 with a Convolutional encoder and decoder.
# Although Inception-resnet-v2 has greater accuracy than mobilenet on ImageNet, it also has many more parameters, so mobilenet was chosen, as the experiment is limited by GPU memory.
# 
# Further changes included using less conv2d layers, to limit my GPU usage. Conv2DTranpose layers were also use in place of Conv2D layers with UpSampling.
# 
# Skip layers were also added, since they have been found useful in Generative Adversial Networks, as demonstratred by [this paper](http://https://arxiv.org/pdf/1901.08954.pdf) where they claimed that "the findings indicate that skip connections provide more stable training."
# Dropout was also added to each skip layer, in an attempt to reduce the model's dependence on them, and prevent overfitting, hoping to encourage the network to rely more on mobilenet.
# 
# The model at this point had BatchNormalization between every layer, but despite increased training, continued to produce mainly brown images. Removing batchnormalization between the layers appeared to stabilize training, though adding some to few blocks might still improve the model.
# 
# In regards to the loss function, this model uses MSE with a tanh activation. There have been other examples online that went with a softmax activation and categorical crossentropy, but in this case it appeared to be unstable in training and showed little improvements in results.
# 
# Below is the code for the final model, which is also shown as a diagram further down.
# 

# In[ ]:


def create_model(image_shape):
    # Prepare the kernel initializer values
    weight_init = RandomNormal(stddev=0.02)
    # Prepare the Input layer
    net_input = Input((image_shape))
    # Download mobile net, and use it as the base.
    mobile_net_base = MobileNetV2(
        include_top=False,
        input_shape=image_shape,
        weights='imagenet'
    )
    mobilenet = mobile_net_base(net_input)
    
    # Encoder block #
    # 224x224
    conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(net_input)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    
    # 112x112
    conv2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    # 112x112
    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv2)
    conv3 =  Activation('relu')(conv3)

    # 56x56
    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv3)
    conv4 = Activation('relu')(conv4)

    # 28x28
    conv4_ = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv4)
    conv4_ = Activation('relu')(conv4_)

    # 28x28
    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv4_)
    conv5 = Activation('relu')(conv5)

    # 14x14
    conv5_ = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv5)
    conv5_ = Activation('relu')(conv5_)
    
    #7x7
    # Fusion layer - Connects MobileNet with our encoder
    conc = concatenate([mobilenet, conv5_])
    fusion = Conv2D(512, (1, 1), padding='same', kernel_initializer=weight_init)(conc)
    fusion = Activation('relu')(fusion)
    
    # Skip fusion layer
    skip_fusion = concatenate([fusion, conv5_])
    
    # Decoder block #
    # 7x7
    decoder = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_fusion)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    # Skip layer from conv5 (with added dropout)
    skip_4_drop = Dropout(0.25)(conv5)
    skip_4 = concatenate([decoder, skip_4_drop])
    
    # 14x14
    decoder = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_4)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    # Skip layer from conv4_ (with added dropout)
    skip_3_drop = Dropout(0.25)(conv4_)
    skip_3 = concatenate([decoder, skip_3_drop])
    
    # 28x28
    decoder = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_3)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    # 56x56
    decoder = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Dropout(0.25)(decoder)

    # 112x112
    decoder = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)

    # 112x112
    decoder = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
    decoder = Activation('relu')(decoder)
    
    # 224x224
    # Ooutput layer, with 2 channels (a and b)
    output_layer = Conv2D(2, (1, 1), activation='tanh')(decoder)

    model = Model(net_input, output_layer)
    model.compile(Adam(lr=0.0002), loss='mse', metrics=['accuracy'])
    
    return model


# # Model Diagram
# 

# In[ ]:


model = create_model((224, 224, 3))
plot_model(model, 'model_diagram.png')
plt.figure(figsize=(160, 60))
plt.imshow(Image.open('model_diagram.png'))


# # Utility functions

# In[ ]:


def graph_training_data(epochs, training_data, validation_data, y_label, title):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, epochs+1), mode='lines+markers', y=training_data,
            marker=dict(color="mediumpurple"), name="Training"))

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, epochs+1), mode='lines+markers', y=validation_data,
            marker=dict(color="forestgreen"), name="Validation"))

    fig.update_layout(title_text=title, yaxis_title=y_label,
                      xaxis_title="Epochs", template="plotly_white")
    fig.show()


# In[ ]:


# Get prediction from the model based of the 'L' grayscale image
def get_pred(model, image_l):
    # Repeat the L value to match input shape
    image_l_R = np.repeat(image_l[..., np.newaxis], 3, -1)
    image_l_R = image_l_R.reshape((1, 224, 224, 3))
    # Normalize the input
    image_l_R = (image_l_R.astype('float32') - 127.5) / 127.5
    # Make prediction
    prediction = model.predict(image_l_R)
    # Normalize the output
    pred = (prediction[0].astype('float32') * 127.5) + 127.5
    
    return pred


# In[ ]:


# Combine an 'L' grayscale image with an 'AB' image, and convert to RGB for display or use
def get_LAB(image_l, image_ab):
    image_l = image_l.reshape((224, 224, 1))
    image_lab = np.concatenate((image_l, image_ab), axis=2)
    image_lab = image_lab.astype("uint8")
    image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    image_rgb = Image.fromarray(image_rgb)
    return image_rgb


# In[ ]:


# Create some samples of black and white images combined, to show input/output
def create_sample(model, images_gray, amount):
    path = "/kaggle/working/"
    samples = []
    for i in range(amount):
        # Select random images
        r = random.randint(0, images_gray.shape[0])
        # Get the model's prediction
        pred = get_pred(model, images_gray[r])
        # Combine input and output to LAB image
        image = get_LAB(images_gray[r], pred)
        # Get number of images in output folder
        count = len(os.listdir(path))
        # Create new combined image and save it
        new_image = Image.new('RGB', (448, 224))
        gray_image = Image.fromarray(images_gray[r])
        new_image.paste(gray_image, (0,0))
        new_image.paste(image, (224, 0))
        # Saving the image with the current count of images (to make it unique)
        # and the index of the image, so that it can be found if needed
        new_image.save(path + str(count)+('_%i.png' % r))
        samples.append(new_image)
    return samples


# # Training function
# 
# Training is kept simple in this example, as it only uses ab1.
# The model could be greatly improved from this example by managing the training to include ab2 and ab3. 

# In[ ]:


def train(model, gray, ab, epochs, batch_size):
    # Setup the training input data (grayscale images)
    train_in = gray
    # Convert the shape from (224, 224, 1) to (224, 224, 3) by copying the value to match MobileNet's requirements
    train_in = np.repeat(train_in[..., np.newaxis], 3, -1)
    
    train_out = ab
    # Normalize the data
    train_in = (train_in.astype('float32') - 127.5) / 127.5
    train_out = (train_out.astype('float32') - 127.5) / 127.5

    history = model.fit(
        train_in,
        train_out,
        epochs=epochs,
        validation_split=0.05,
        batch_size=batch_size
    )
    
    return history


# # Main - Training the model
# 
# In this example the model is only trained using the first 3,000 images. It is recommended to use as many images as possible from the dataset to train the model. A script to manage that and RAM usage would be recommended. 
# 
# This example will only run on a small amount of data for few epochs, to facilitate the run in Kaggle, and simplify this notebook.

# In[ ]:


images_gray = np.load("../input/image-colorization/l/gray_scale.npy")
images_ab = np.load("../input/image-colorization/ab/ab/ab1.npy")

# Set batch size and epochs for training run
BATCH_SIZE = 32
EPOCHS = 30

# Create the model through the function above
model = create_model((224, 224, 3))

# Train the model and keep history for graphing
history = train(model, images_gray[:3000], images_ab[:3000], EPOCHS, BATCH_SIZE)


# # Graphing the training data
# 

# In[ ]:


graph_training_data(EPOCHS, history.history['loss'], history.history['val_loss'], 'Loss', "Loss while training")
graph_training_data(EPOCHS, history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', "Accuracy while training")


# # Results
# 
# A steady increase in Accuracy and a steady decrease in loss can be seen in the graphs above. This indicates the model is learning steadily.
# The validation accuracy should also increase, but may lag behind as the model has to learn to generalize the dataset. In this case it might not be since the model is overfitting due to the lack of training data, however, increasing the data the model sees should solve that issue.
# 
# While an increase in validation accuracy / decrease in loss is important, the model can have unstable values throughout the training while still providing successful images, success as determined by looking at the quality of the images manually.
# 
# Furthermore since the images differ greatly, there has been better results from a model with lower validation accuracy than one with higher accuracy, though it is correlated to the quality of the images, it should be taken as general feedback, not literally.

# # Creating Samples
# 
# Here are some images generated from the model through Kaggle, by the code shown above.

# In[ ]:


# Create 10 sample images. These images are both returned in a list, and saved.
samples = create_sample(model, images_gray, 5)
for image in samples:
    plt.figure()
    plt.imshow(np.array(image))


# # Understanding the samples
# 
# As you can see from the images above, the model is not very accurate at this point.
# The model mostly generates brown images, as brown represents the average color and thus is the first step in training.
# 
# It should be noted, however, that the model begins to notice the difference between objects, and may assign different colors to different parts of the image. This is a good sign that more training should prove successful, as the model is begining to differentiate the objects and assign them different colors.

# # Samples from better training
# 
# Here are some samples given by the same model but trained with more images from the dataset (using both ab1 and ab2), and trained for much longer.
# The model was trained for around 10 hours over time, and using varying parts of the image dataset.

# In[ ]:


path = '../input/imagecolorization-samplesmodel/ImageColorization_Samples'
success = os.listdir(path + '/Successful')
for img in success:
    plt.figure()
    plt.imshow(Image.open(path +'/Successful/' + img))


# # Poor Results
# 
# While the model provides some good images, it still fails to colorize some images. The ratio seems to be about 60-40, where 60% of the images look 'somewhat realistic,' and 40% would be considered unrealistic.
# 
# As can be seen in the images below, the model failed to create a realistic looking image for these inputs. While some colors have been added, the image is not complete.
# 

# In[ ]:


path = '../input/imagecolorization-samplesmodel/ImageColorization_Samples'
success = os.listdir(path + '/Failed')
for img in success:
    plt.figure()
    plt.imshow(Image.open(path +'/Failed/' + img))


# # Historical Photographs
# 
# Here are samples from using the model on historical photographs.

# In[ ]:


path = '../input/imagecolorization-samplesmodel/'
images = os.listdir(path)
for img in images:
    if (img == 'ImageColorization_Samples'): continue
    plt.figure()
    plt.imshow(Image.open(path + img))


# # Further Improvements
# 
# **Reimplementing BatchNormalization**
# 
# >BatchNormalization was removed in the final version of the model, this greatly improved the training. Though adding it back into strategic places could improve the training optimization, and could lead to better results. More experimentation with BatchNormalization is needed.
# 
# 
# 
# **Improved Training**
# 
# >Although the model trained outside of Kaggle saw many more images than the example given above, it had uneven training in regards to which part of the dataset it was trained on, and only 20,000 out of the 25,000 images were used. The images were handled 'manually' with slicing of the array before training sessions. A better way of managing the images, allowing the net to train on the entire dataset should greatly improve the model's results, perhaps a method using tf.keras.ImageDataGenerator would be successful.
# 
# 
# 
# **Better Fusion Layer**
# 
# >The 'fusion layer' in this case is a simple concatenate layer between mobilenet and the output of the encoder. Using a more specific and considered approach may yield to better results. An example of this is done in the [Koalarization paper](https://arxiv.org/pdf/1712.03400.pdf).
# 
# 
# 
# **Improved Normalization**
# 
# >It was difficult to find the exact details of how the data is represented in the dataset, and led to some confusion. In this example, and the values had 127.5 subtracted, and then were divided by 127.5. In RGB, this would lead to an interval of [-1, 1], though it is not the case here and leads to a slightly different interval for the AB values. A full understanding of this and a better normalization function could lead to better results.
# 
# 

# # Final Thoughts
# 
# If you have made it this far, thank you for reading, hope it was helpful
# 
# I am still a beginner with neural nets and CNNs in particular, and this is mainly for personal experimentations.
# I would appreciate any feedback or criticisms, particularly if I am making important mistakes that could easily be avoided.
# 
# If you have any questions, comments, or suggestions, do not hesitate to comment them below.
