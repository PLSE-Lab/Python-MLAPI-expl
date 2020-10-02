#!/usr/bin/env python
# coding: utf-8

# ![](https://images.fastcompany.net/image/upload/w_937,ar_16:9,c_fill,g_auto,f_auto,q_auto,fl_lossy/fc/3048941-poster-p-1-why-google-deep-dreams-of-dogs.jpg)
# # Convolutional Neural Networks Visualization in Pytorch
# 
# In this kernel, we'll look into a convolutional network, to try and understand how they work by generating images that maximize the activation of the filters in the convolutional layers.
# To generate these images, we apply gradient ascent to the inputs (which will be images with random noise)

# ## Packages and utils functions
# 
# Some util functions to load and visualize images

# In[ ]:


import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models, transforms
import PIL
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import scipy.ndimage as ndimage

get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from io import BytesIO


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    
def showtensor(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    showarray(inp)
    clear_output(wait=True)

def plot_images(im, titles=None):
    plt.figure(figsize=(30, 20))
    
    for i in range(len(im)):
        plt.subplot(10 / 5 + 1, 5, i + 1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
        plt.imshow(im[i])
        
    plt.pause(0.001)
    
normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalise_resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def init_image(size=(400, 400, 3)):
    img = PIL.Image.fromarray(np.uint8(np.full(size, 150)))
    img = PIL.Image.fromarray(np.uint8(np.random.uniform(150, 180, size)))
    img_tensor = normalise(img).unsqueeze(0)
    img_np = img_tensor.numpy()
    return img, img_tensor, img_np

def load_image(path, resize=False, size=None):
    img = PIL.Image.open(path)
    
    if size is not None:
        img.thumbnail(size, PIL.Image.ANTIALIAS)
        
    if resize:
        img_tensor = normalise_resize(img).unsqueeze(0)
    else:
        img_tensor = normalise(img).unsqueeze(0)
    img_np = img_tensor.numpy()
    return img, img_tensor, img_np

def tensor_to_img(t):
    a = t.numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    inp = np.uint8(np.clip(inp, 0, 255))
    return PIL.Image.fromarray(inp)

def image_to_variable(image, requires_grad=False, cuda=False):
    if cuda:
        image = Variable(image.cuda(), requires_grad=requires_grad)
    else:
        image = Variable(image, requires_grad=requires_grad)
    return image


# ## Model Creation
# 
# Here we load a pretrained VGG-16 model

# In[ ]:


model = models.vgg16()
model.load_state_dict(torch.load("../input/vgg16/vgg16.pth"))


# In[ ]:


use_gpu = False
if torch.cuda.is_available():
    use_gpu = True

print(model)

for param in model.parameters():
    param.requires_grad = False

if use_gpu:
    print("Using CUDA")
    model.cuda()


# ## Octaver function
# 
# The octaver function is used to produce better output images. This procedure is taken from the [Deep Dream code](https://github.com/google/deepdream/blob/master/dream.ipynb). 
# We will use it not only for the deep dream algorithm, but also for visualizing the filters.
# The gradient ascent algorithm is run on multiple downscaled versions of the image, and the results are upscaled and merged together to get the final image.
# 
# ![](https://raw.githubusercontent.com/Hvass-Labs/TensorFlow-Tutorials/master/images/14_deepdream_recursive_flowchart.png)
# *Figure 1: Flowchart of the octaver function + deep dream, taken from https://github.com/Hvass-Labs/TensorFlow-Tutorials*

# In[ ]:


def octaver_fn(model, base_img, step_fn, octave_n=6, octave_scale=1.4, iter_n=10, **step_args):
    octaves = [base_img]
    
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)
        
        src = octave_base + detail
        
        for i in range(iter_n):
            src = step_fn(model, src, **step_args)

        detail = src.numpy() - octave_base

    return src


# ## Filter Visualization
# 
# This function produces an image that maximizes the activation of the filter at *filter_index* in the layer *layer_index*. 

# In[ ]:


def filter_step(model, img, layer_index, filter_index, step_size=5, display=True, use_L2=False):
    global use_gpu
    
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    
    model.zero_grad()
    
    img_var = image_to_variable(torch.Tensor(img), requires_grad=True, cuda=use_gpu)
    optimizer = SGD([img_var], lr=step_size, weight_decay=1e-4)
    
    x = img_var
    for index, layer in enumerate(model.features):
        x = layer(x)
        if index == layer_index:
            break

    output = x[0, filter_index]
    loss = output.norm() #torch.mean(output)
    loss.backward()
    
    if use_L2:
        #L2 normalization on gradients
        mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2) + 1e-5])
        if use_gpu:
            mean_square = mean_square.cuda()
        img_var.grad.data /= torch.sqrt(mean_square)
        img_var.data.add_(img_var.grad.data * step_size)
    else:
        optimizer.step()
    
    result = img_var.data.cpu().numpy()
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std)
    
    if display:
        showtensor(result)
    
    return torch.Tensor(result)

def visualize_filter(model, base_img, layer_index, filter_index, 
                     octave_n=6, octave_scale=1.4, iter_n=10, 
                     step_size=5, display=True, use_L2=False):
    
    return octaver_fn(
                model, base_img, step_fn=filter_step, 
                octave_n=octave_n, octave_scale=octave_scale, 
                iter_n=iter_n, layer_index=layer_index, 
                filter_index=filter_index, step_size=step_size, 
                display=display, use_L2=use_L2
            )


# Next, we define a helper function to visualize a number of filter for a given layer

# In[ ]:


def show_layer(layer_num, filter_start=10, filter_end=20, step_size=7, use_L2=False):
    filters = []
    titles = []
    
    _, _, img_np = init_image(size=(600, 600, 3))
    for i in range(filter_start, filter_end):
        title = "Layer {} Filter {}".format(layer_num , i)
        print(title)
        filter = visualize_filter(model, img_np, layer_num, filter_index=i, octave_n=2, iter_n=20, step_size=step_size, display=True, use_L2=use_L2)
        filter_img = tensor_to_img(filter)
        filter_img.save(title + ".jpg")
        filters.append(tensor_to_img(filter))
        titles.append(title)
        
    
    plot_images(filters, titles)
    return filters, titles


# In[ ]:


images, titles = show_layer(1, use_L2=True, step_size=0.05)


# In[ ]:


images, titles = show_layer(10, use_L2=True, step_size=0.05)


# In[ ]:


images, titles = show_layer(14, use_L2=True, step_size=0.05)


# In[ ]:


images, titles = show_layer(17, use_L2=True, step_size=0.05)


# In[ ]:


images, titles = show_layer(19, use_L2=True, step_size=0.05)


# In[ ]:


images, titles = show_layer(21, use_L2=True, step_size=0.05)


# ## Deep Dream
# 
# The Deep Dream function is similar to the filter visualization, but instead of starting from a random noise image, we start from an actual picture and try to maximize the network output. In this way, we're enhancing the features that the network recognizes in the image. Different layers yields different results; lower ones tend to procude geometric patterns and textures, while higher ones produce more abstract shapes that resemble what the network has seen during its training process.
# 
# The actual code is ported to pytorch from the [Google Deep Dream repository](https://github.com/google/deepdream) which runs under cafe.

# In[ ]:


def objective(dst, guide_features):
    if guide_features is None:
        return dst.data
    else:
        x = dst.data[0].cpu().numpy()
        y = guide_features.data[0].cpu().numpy()
        ch, w, h = x.shape
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        A = x.T.dot(y)
        diff = y[:, A.argmax(1)]
        diff = torch.Tensor(np.array([diff.reshape(ch, w, h)])).cuda()
        return diff

def make_step(model, img, objective=objective, control=None, step_size=1.5, end=28, jitter=32):
    global use_gpu
    
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    
    img = np.roll(np.roll(img, ox, -1), oy, -2)
    tensor = torch.Tensor(img) 
    
    img_var = image_to_variable(tensor, requires_grad=True, cuda=use_gpu)
    model.zero_grad()
      
    x = img_var
    for index, layer in enumerate(model.features.children()):
        x = layer(x)
        if index == end:
            break
    
    delta = objective(x, control)
    x.backward(delta)
    
    #L2 Regularization on gradients
    mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2)])
    if use_gpu:
        mean_square = mean_square.cuda()
    img_var.grad.data /= torch.sqrt(mean_square)
    img_var.data.add_(img_var.grad.data * step_size)
    
    result = img_var.data.cpu().numpy()
    result = np.roll(np.roll(result, -ox, -1), -oy, -2)
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean / std, (1 - mean) / std)
    showtensor(result)
    
    return torch.Tensor(result)
                                                             
def deepdream(model, base_img, octave_n=6, octave_scale=1.4, 
              iter_n=10, end=28, control=None, objective=objective, 
              step_size=1.5, jitter=32):
    
    return octaver_fn(
              model, base_img, step_fn=make_step, 
              octave_n=octave_n, octave_scale=octave_scale, 
              iter_n=iter_n, end=end, control=control,
              objective=objective, step_size=step_size, jitter=jitter
           )


# In[ ]:


input_img, input_tensor, input_np = load_image('../input/sampleimages/data/data/market1.jpg', size=[1024, 1024])
print(input_img.size)
input_img


# In[ ]:


dream = deepdream(model, input_np, end=14, step_size=0.06, octave_n=6)
dream = tensor_to_img(dream)
dream.save('dream00.jpg')
dream


# In[ ]:


dream = deepdream(model, input_np, end=20, step_size=0.06, octave_n=6)
dream = tensor_to_img(dream)
dream.save('dream01.jpg')
dream


# In[ ]:


dream = deepdream(model, input_np, end=28, step_size=0.06, octave_n=6)
dream = tensor_to_img(dream)
dream.save('dream03.jpg')
dream


# ## Controlling the dream
# 
# We can control the dream by trying to alter the image in order to maximize the filters that are activated by another image (which we'll call guide)

# In[ ]:


guide_img, guide_img_tensor, guide_img_np = load_image('../input/sampleimages/data/data/kitten2.jpg', resize=True)
plt.imshow(guide_img)


# In[ ]:


end = 26

guide_features = image_to_variable(guide_img_tensor, cuda=use_gpu)

for index, layer in enumerate(model.features.children()):
    guide_features = layer(guide_features)
    if index == end:
        break
    
dream = deepdream(model, input_np, end=end, step_size=0.06, octave_n=4, control=guide_features)
dream = tensor_to_img(dream)
dream.save('dream04.jpg')
dream


# In[ ]:


input_img, input_tensor, input_np = load_image('../input/sampleimages/data/data/face1.jpg', size=[1024, 1024])
print(input_img.size)
input_img


# In[ ]:


dream = deepdream(model, input_np, end=26, step_size=0.06, octave_n=6, control=guide_features)
dream = tensor_to_img(dream)
dream.save('dream05.jpg')
dream


# ## References
# 
# - How Convolutional Neural Networks see the world - Keras Blog https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# - Google Deep Dream code https://github.com/google/deepdream
# - Inceptionism: Going Deeper into Neural Networks https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
# - PyTorch DeepDream https://github.com/L1aoXingyu/Deep-Dream
# - DL06: DeepDream (with code) https://hackernoon.com/dl06-deepdream-with-code-5f735052e21f
# - https://www.youtube.com/watch?v=ws-ZbiFV1Ms
# - TensorFlow-Tutorials https://github.com/Hvass-Labs/TensorFlow-Tutorials
