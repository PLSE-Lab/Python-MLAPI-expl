#!/usr/bin/env python
# coding: utf-8

# # Intro to computer vision
# When Snapchat introduced a filter featuring a breakdancing hotdog, its stock price surged. But investors where less interested in the hotdogs handstand. What fascinated them was that Snap had built powerful computer vision technology. It's app could not only take pictures, it could find the surfaces in the pictures that a hotdog could breakdance on and then stick the hotdog there. Even when the user moved the phone, the hotdog kept dancing in the same spot.
# 
# The dancing hotdog might be one of the more silly applications of computer vision, but it shows the potential of the technology. In a world full of cameras, from the billions of smartphones to security cameras to satellites to IOT devices being able to interpret the images yields great benefit. Take self driving cars for instance, another benefiter of computer vision. Planting a bunch of cameras on a car does not help much, but once a computer can interpret and understand what is in the images taken it can use that information to steer the car.
# 
# In the second week, we already performed a small computer vision task. We classified images of hand written digits from the MNIST dataset. In this week, we will go much further. While still mainly focusing on image classification, figuring out what an image shows, we will work with larger, more complex images as you would encounter them in the real world. Later in the week, we will visit semantic segmentation, segmenting an image by its content.
# 
# ## Images in finance
# Remember the very first chapter on how ML can be used to distill valuable information that analysts can use? Computer vision is the prime example of this use case. The world is full of images. But analysts can not possibly look at all of them. Computers can help and automate much of the hard work. Say you are a commodity trader, trading oil. In this business it is quite important to have a good estimate how much oil is produced around the world. And luckily, the amount of production can be estimated from satellite images of production sites. Frequently updated satellite imagery of the whole world is now available for purchase but analysts would have to look at millions of images only to find the production sites and then at thousands more to estimate the production. A computer vision system that automatically finds and analyzes oil production site from space comes in handy. Another example of the same technology would be to count cars in the parking lots of retailers to estimate how well sales go. This kind is done by several firms and will probably find more usage in the future.
# 
# A slightly less fancy but never the less important application of computer vision in finance is insurance. Insurers might use drones to fly over roofs to spot issues before they become an expensive problem. Or they might inspect factories and equipment they insured. The applications are near endless.
# 
# Now, enough of the motivational talk, let's dive into the technology.
# 
# # ConvNets
# Convolutional Neural Networks, ConvNets for short, are the driving engine behind computer vision. ConvNets allow us to work with larger images while keeping the size of the network reasonable. The name Convolutional Neural Net comes from the mathematical operation that differentiates them from regular neural nets. Convolution is the mathematical correct term for sliding one matrix over another matrix. You will see in a minute why this is important for ConvNets but also why this is not the best name in the world. Actually, ConvNets should be called Filter Nets. Because what makes them work is the fact that they use filters. Let's go back to the MNIST dataset to see what this means.
# 
# ## Filters on MNIST
# What does a computer actually see when it sees an image? The value of the pixels are stored as numbers in the computer. So when the computer 'sees' a black and white image of a seven, what it actually sees is something like this:
# ![MNIST](https://storage.googleapis.com/aibootcamp/Week%203/assets/mnist_seven.png)
# 
# The larger numbers in the image have been highlighted to make the seven visible for humans, but for the computer an image is really just numbers. This means, we can perform all kinds of mathematical operations on the image. 
# 
# When detecting numbers, there are a few lower level features that make a number. A seven for example is a combination of one vertical straight line, one straight  line on the top and one straight line through the middle. A nine for contrast is made up of four rounded lines that form a circle at the top and a straight, vertical line.
# When detecting numbers, there are a few lower level features that make a number. A seven for example is a combination of one vertical straight line, one straight horizontal line on the top and one straight horizontal line through the middle. A nine for contrast is made up of four rounded lines that form a circle at the top and a straight, vertical line.
# 
# And now comes the central idea behind ConvNets (or Filter Nets): We can use  small filters that detect a certain kind of low level feature like a vertical line and then slide it over the entire image to detect all the vertical lines in there.
# This is how a vertical line filter would look like:
# 
# 
# ![Vertical line filter](https://storage.googleapis.com/aibootcamp/Week%203/assets/vertical_line_filter.png)
# 
# 
# 
# It is a 3 by 3 matrix. To detect vertical lines in our image, we slide this filter over the image. We start in the top left corner and slice out the most top left 3 by 3 grid of pixels (all zeros in this case). We then perform an element wise multiplication of all elements in the filter with all elements in the slice of the image. The nine products get then summed up and a bias is added. This value then forms the output of the filter and gets passed on as a new pixel to the next layer.
# 
# $$Z_1 = \sum A_0 * F_1 + b $$
# 
# ![ConvNetGif](https://storage.googleapis.com/aibootcamp/Week%203/assets/convnet.gif)
# 
# The output of our vertical line filter looks like this:
# 
# ![Output vertical line filter](https://storage.googleapis.com/aibootcamp/Week%203/assets/output_vert_line_filter.png)
# 
# Notice that the vertical lines are visible while the horizontal lines are gone. Only a few artifacts remain. Also notice how the filter captures the vertical line from one side. Since it responds to high pixel values on the left, and low pixel values on the right, only the right side of the output shows strong positive values while the left side of the line actually shows negative values. This is not a big problem in practice as there are usually different filters for different kinds of lines and directions.
# 
# ### Adding a second filter 
# Our vertical filter is cool, but we already noticed that we also need to filter our image for horizontal lines to detect a seven. Our vertical filter might look like this: 
# 
# ![Horizontal Filter](https://storage.googleapis.com/aibootcamp/Week%203/assets/horizontal_filter.png)
# 
# We can now slide this filter over our image the exact same way we did with the vertical filter. We obtain:
# 
# ![Output horizontal](https://storage.googleapis.com/aibootcamp/Week%203/assets/output_horizontal_filter.png)
# 
# See how this filter removes the vertical lines and pretty much only leaves the horizontal lines?
# 
# But what do we now pass on to the next layer? We stack the outputs of both filters on top of each other, creating a 3 dimensional cube.
# 
# ![MNIST conv overview](https://storage.googleapis.com/aibootcamp/Week%203/assets/mnist_conv.png)
# 
# ## Filters on Color images 
# Of course, our filter technique does not only work on black and white images. Let's have a look at color images. Any color image consists of 3 layers, or channels. One red channel, one blue channel and one green channel, RGB for short. When these 3 channels are laid on top of each other, they add up to the color image that we know. An image is therefore not flat, but actually a cube, a 3 dimensional matrix. When we want to apply a filter to the image, we have to apply it to all three channels at once. We will therefore perform an element wise multiplication between two, three dimensional cubes. Our three by three filter now also has a depth of 3 and thus 9 parameters plus the bias.
# 
# ![3 Layer](https://storage.googleapis.com/aibootcamp/Week%203/assets/img_times_filter.png)
# 
# This cube gets slided over the image just like the two dimensional matrix did before. The element wise products get then again summed up, the bias is added and the outcome represents a pixel in the next layer. Filters always capture the whole depth of the previous layer. They never get slided depth wise, only along the height and width of the image.
# 
# ## Activation functions in ConvNets
# Convolutional neural nets have actvation functions just like densly connected ones. The sum of the element wise products represents the input pixels to the activation function. The activation function can then be the same as in a dense network, for example a ReLu function. 
# $$Z_1 = \sum A_0 * F_1 + b $$
# $$A_1 = relu(Z_1)$$
# 
# ## Padding
# You might have already wondered: As a 3 by 3 grid results in one new pixel, how do we keep the image from shrinking towards the network? And furthermore, as we stride over the image, most pixels get included in the filter multiple times. But the pixels at the side get included less often. How can we use them better? The answer is padding. Padding ads a ring of zero-value pixels around the actual image. This makes the image larger. Therefore, even as the image shrinks from the convolutions, we can preserve its size by just adding pixels around it before shrinking it. This also helps get the outer pixels of the image get included in the filters as often as the inner pixels, since the outer pixels of the actual image are no longer the outer pixels of the padded image. 
# 
# In Keras, the padding can be either set to ``"valid"`` or ``"same"``. Valid padding ensures that the filter actually fits on the image and does not 'stand over' at some side. Same padding additionally insures that the output of the convolutional layer has the same size as the input.
# 
# ## Stride size
# So far we have assumed that the filters get moved one pixel at a time. Of course we can also move them faster. This hyperparameter is called the stride size. How many pixels the steps of the filters are.
# 
# In Keras, the stride size is usually specified as a list or [tuple](https://www.programiz.com/python-programming/tuple) containing two integers. This allows us to specify a different stride size over width than height. In practice, the stride size is usually equal to 1.
# 
# ## Summary
# In a ConvNet (or more precisely in the convolutional layers of a neural net), we use filters to detect certain patterns. The filters get slided (convolved) over the image. A filter captures a certain width and height, but always the entire depth of the previous layer. A filter consists of a cube of parameters and a bias parameter. Inputs to the convolutional layer can be padded, to prevent shrinking and improve usage of outer pixels.
# 
# For a convolutional layer in Keras, you have to know these hyperparameters:
# - The number of filters you want to use in the layer 
# - The size (width and height) of the filters 
# - The stride size 
# - The padding parameter 

# In[ ]:


from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Activation

# Images fed into this model are 512 x 512 pixels with 3 channels
img_shape = (512,512,3)

# Set up model
model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same',input_shape=img_shape))
# Add relu activation to the layer 
model.add(Activation('relu'))

# The same conv layer in a more common notation
model.add(Conv2D(3,(3,3),padding='same',activation='relu'))


# In[ ]:


# Give out model summary to show that both layers are the same
model.summary()


# In[ ]:




