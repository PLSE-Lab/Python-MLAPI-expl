#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import inception_v3
from keras import backend as K
import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.preprocessing import image
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # Deep Dream in Theory

# Deep Dream is a computer vision algorithm developed at google in 2014.
# It modify an image with some dream-like shapes.
# 
# DeepDream is a Generative model, as opposed to the classical predictive models.
# Generative models generate "something new" on their own. 
# 
# 
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/image/deep-dream.jpg" style="width:700px;height:400px;">

# For our Implementation we will use an existing Neural Network, called Inception.
# 
# Inception was trained on imagenet, a famous dataset of picture labeled with classes. 1000 classes exist in the dataset: cat, elephant, phone, house... but it is mostly cats and dogs...
# 
# Below a picture of the architecture of the model:
# 
# 
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/image/inception.png" style="width:800px;height:350px;">

# Another important consideration is that, each layer of a NN contains fitlers, and each filter capture a shape, a texture, a color... of an image.
# 
# Early layers in a NN capture simple shapes: vertical edges, simple texture... whereas later layers in a NN capture more complex shapes: ears, faces, fish...
# 
# This is important for our algorithm as we will leverage this property to modify our images.
# 
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/image/layers-observation.png">

# For the sake of explaination I show below a simplification of the Neural Network architecture during training:
# 
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/image/neural-network.png">

# For the Deep Dream algorithm we are going to hack this NN to achieve our goal: Modify the input picture!
# 1. First we remove the "top" of the network, the part of fully connected layers that compute the last activations.
# 2. Then we create a new LOSS function that average the values of several activations layers.
# 3. The gradients are computed with this LOSS function and regarding the input, not the parameters as it is usually the case.
# 4. The input values (our picture) are updated with the <b>gradients ascent</b> and another iteration begins with these new input.
# 
# The intuition here is that the input picture will be modify at every iteration and input again for the next iteration.
# The modification of the input try to increase the activations, by gradient ascent (we want to maximize the loss).
# By maximizing the LOSS it means that we increase the activations values, if we increase the activations it means that there is a lot of patterns matching the  filters of the layers.
# 
# In other words if there is filters that recognize eyes in the layers that we selected to calculate the LOSS function, the gradient ascent will try to add eyes to the input picture to maximize the loss, hence these weird pattern appear on our input.
# 
# i.e:
# 
# Gradient descent: w = w - lr * grad_w
# 
# Gradient ascent: w = w + lr * grad_w
# 
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/image/deep-dream-network.png">

# # Deep Dream in Practice

# ## Load the inception model
# We load the inception network trained with imagenet, and without the top of fully connected layers

# In[ ]:


K.set_learning_phase(0) # disable all training of the model weights
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)


# In[ ]:


# you can run this cell to see all the names of the layers
model.summary()


# In[ ]:


#######################################################################################
# UTILITY FUNCTIONS
# mostly image manipulation
# nothing interesting here to understant the Deep Dream algorithm
######################################################################################

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
    float(size[0]) / img.shape[1],
    float(size[1]) / img.shape[2],
    1)
    return scipy.ndimage.zoom(img, factors, order=1)
   
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
    return
   
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img
   
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
   
###################################################################################


# ## Choose layers to compute the loss
# It is better result to mix several layers in the algorithm
# 
# You can also modify the coefficient (weight) of each layer in the calculation of the Loss fucntion.
# 
# If you want to try it yourself the most interesting layers are the one named "conv" or "mixed"
# 

# In[ ]:


# four layer will be involve in calculation of the loss
# each layer will have a specific weight in the loss.
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
    }


# ## Define the loss function
# 

# In[ ]:


# We will define the loss as a weighted sum of L2 norm of all filters in several layers
layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.) # initialize loss to 0
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output  # get the activations corresponding to a layer
    scaling = K.prod(K.cast(K.shape(activation), 'float32')) # number of activation in a layer
    # avoid border effect by selecting 2:-2 
    loss += coeff * K.sum(K.square(activation[:,2:-2, 2:-2, :])) / scaling   # "average" the activations


# ## Define our gradient ascent

# In[ ]:


# Gradient ascent
dream = model.input
grads = K.gradients(loss, dream)[0]  # gradient of the input with regard to the loss.
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) # normalize gradient (like clipping)

outputs = [loss, grads]
# keras syntax use a K.function to interact with the backend, in a computational graph.
fetch_loss_and_grads = K.function([dream], outputs)  

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
   
# main function of the gradient ascent   
# the max_loss parameter "cap" the value of the loss we want to reach
def gradient_ascent(picture, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss_value, gradient_values = eval_loss_and_grads(picture)
        if (max_loss!=None) and (loss_value>max_loss):
            # Stop the algorithm after a Maximum loss threshold 
            break
        print('...Loss value at', i, ':', loss_value)
        # gradient ascent add the gradient instead of substracting it
        picture = picture + learning_rate*gradient_values
    return picture


# ## Generate the new images
# Below is the main function that essentialy run several iterations of gradient ascent on our input
# 
# Traditionnaly the iterations are run first on a smaller version of the image then on a bigger scaling until we reach the original size of the image. This way allow apparently to conserve the most details of the image. I'm not explaining this part as this is not essential to understand Deep Dream, but you can leave a comment if you want more information.

# In[ ]:


# running gradient ascent of different scales of image
def main(base_image_path):
    learning_rate = 0.01
    iterations = 20
    max_loss = 10
    
    # all the part below concern the rescaling of the image
    num_octave = 3 
    octave_scale = 1.4
    img = preprocess_image(base_image_path)
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0]) # srunk to first scale

    # we run the algorithm for each size of the image
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        # the most interesting call to the gradient ascent fucntion
        img = gradient_ascent(img,
                              iterations=iterations,
                              learning_rate=learning_rate,
                              max_loss=max_loss)
        # below is the upsizing of the image
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale.png')
    save_img(img, fname='final_dream.png')


# In[ ]:


# Run the generative model
main('../input/rome.jpg')


# You should observe that at each iteration the loss increase.
# 
# Although every time the image is rescaled the process is reset again and the loss restart from zero (but the image is already modified)
# 
# Below is our input image:
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/dream/rome.jpg">

# In[ ]:


# Now our Dreamed Image is:
Image(filename='final_dream.png')


# # Changing parameters
# If you want to play around with the algorithm here is a few advices:
# 
# - Pictures of homogenous view (sky, grass land, desert) are actually easier for the neural network to modify, has there is not much patterns already in place.
# - Number of iteration, max_loss and learning rate are interesting parameters to modify to observe how they influence the loss values and consequently the output
# - The contributing layers, and their coefficients are interesting to change to observe different patterns modifications of the image.
# 
# Remenber after each modification to re-run the previous cells to update the computational graph.

# In[ ]:


##############################
# Edit the contibuting layers

layer_contributions = {
    'conv2d_52': 25.,
    }


# In[ ]:


####################################
# RUN THIS CELL TO UPDATE THE GRAPH
# NO NEED TO CHANGE THIS CODE


layer_dict = dict([(layer.name, layer) for layer in model.layers]) # map layer_name -> instance

loss = K.variable(0.) # initialize loss to 0
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32')) # number of activation in a layer
    # avoid border effect by selecting 2:-2 
    loss += coeff * K.sum(K.square(activation[:,2:-2, 2:-2, :])) / scaling
    # Gradient ascent
    
dream = model.input
grads = K.gradients(loss, dream)[0]  # gradient of the input with regard to the loss.
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) # normalize gradient (like clipping)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
   
def gradient_ascent(picture, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss_value, gradient_values = eval_loss_and_grads(picture)
        if (max_loss!=None) and (loss_value>max_loss):
            # Stop the algorithm after a Maximum loss threshold 
            break
        print('...Loss value at', i, ':', loss_value)
        picture = picture + learning_rate*gradient_values
    return picture


# In[ ]:


############################################################
# Edit the learning rate to speed up the modification
# Edit the max_loss to increase the modification of the image


def main(base_image_path):
    learning_rate = 0.01
    num_octave = 3
    octave_scale = 1.4
    iterations = 20
    max_loss = 100

    img = preprocess_image(base_image_path)
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0]) # srunk to first scale

    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              learning_rate=learning_rate,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale.png')
    save_img(img, fname='final_dream2.png')


# In[ ]:


main('../input/sky.JPG')


# You can see that the loss take higher values now, making more dramatic changes to the input/
# 
# Below is our input image:
# <img src="https://raw.githubusercontent.com/flacout/deep-dream-demo/master/dream/sky.JPG">

# In[ ]:


# Now our Dreamed Image is:
Image(filename='final_dream2.png')


# Now we can start seing some animal shapes forming in the clouds.
# 
# Change with your own settings to obtain different images modifications!!

# ## Acknowledgement
