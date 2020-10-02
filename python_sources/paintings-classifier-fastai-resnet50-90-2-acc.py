#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# ![Paintings-museum](https://i.imgur.com/0ZQMAJ8.png)
# 
# The following Kernel looks into developing a paintings classifier based on [best-artworks-of-all-time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time), a Kaggle public dataset with paintings and work of arts (drawings and sketches are included) from 50 of the best artists of all time. 
# 
# The obvious aim of the classifier is to output the name of the artist that painted the input painting. However, I also believe that an important part of the Kernel can be found in the model interpretation part as we dig deeper into how the model predicted a certain output. The techniques used to evauate and visualize the results of the CNN are Heatmaps and Guided Backpropagation.
# 
# In order to train the model I have used fastai, a free open source library for deep learning that sits on top of Pytorch, as it provides really good out-of-the-box techniques for image classification. After various experiments, the model reached 90.2% accuracy. I am overall happy with the result but I believe there is room for improvement. Get in touch if you have any advice to do so!
# 
# This Kernel is divided into four parts:
# 
# * Import Libraries and read data
# * Exploratory Data Analysis
# * Model And Learning
# * Model Evaluation and Visualization
# 

# ## Import Libraries and read data

# In this section we just read the data and import a few libraries that we are going to need.

# In[ ]:


from pathlib import Path
from fastai.vision import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fastai import basic_train
from fastai.callbacks import *
import shap
from fastai.callbacks.hooks import *
from PIL import *


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


path = Path("../input/best-artworks-of-all-time")
df = pd.read_csv(path/'artists.csv')
df.head()


# ## EDA

# In[ ]:


len(df)


# Let's look at the main nationalities and paintings' genres of the artists present in the datatset

# In[ ]:


by_nationality = df[['nationality', 'paintings']].groupby(['nationality'], as_index = False).sum()
nationality_top = by_nationality.sort_values('paintings', ascending = False)[:10]

count_names = df[['nationality', 'name']].groupby(['nationality']).count()
count_names = count_names.rename({'name' : 'number of artists'}, axis=1)

nationality_top = nationality_top.join(count_names, on = 'nationality')
print(nationality_top)


# In[ ]:


by_genre = df[['genre', 'paintings']].groupby(['genre'], as_index = False).sum()
genre_top = by_genre.sort_values('paintings', ascending = False)[:10]

count_names = df[['genre', 'name']].groupby(['genre']).count()
count_names = count_names.rename({'name' : 'number of artists'}, axis=1)

genre_top = genre_top.join(count_names, on = 'genre')
print(genre_top)


# Create a list of 20 artists with the most number of paintings in the dataset. These names will be used as the model's precitions' classes.

# In[ ]:


by_artist = df[['name', 'paintings']].groupby(['name'], as_index = False).sum()
name_top = by_artist.sort_values('paintings', ascending = False)[:20]
name_top


# We also define the show_random_paintings() function. The function takes n_artist and n_paintings (default 4 and 4) as parameters and outputs a n_artist x n_paintings grid of random paintings and random artists (from the 20 selected artists).

# In[ ]:


# set variables

images_dir = Path(path/'images/images')
artists = name_top['name'].str.replace(' ', '_').values
artists = np.delete(artists, 4)

#let's give a look at some of these beautiful paintings !

def show_random_paintings(n_artists = 4, n_paintings = 4):
  
    fig, axes = plt.subplots(n_artists, n_paintings, figsize=(20,10))

    for r in range(n_artists):
        random_artist = random.choice(artists)
        random_images = random.sample(os.listdir(os.path.join(images_dir, random_artist)), n_paintings)

        c=0
        for random_image in random_images:

          random_image_file = os.path.join(images_dir, random_artist, random_image)
          image = plt.imread(random_image_file)

          axes[r, c].imshow(image)
          axes[r, c].set_title("Artist: " + random_artist.replace('_', ' '))
          axes[r, c].axis('off')

          c+=1

    return plt.show()


# In[ ]:


show_random_paintings()


# ## Training And Learning

# Due to memory reason and convenience, the modelling and trainging part of the kernel has been done entirely on Google Colaboratory (Why google Colaboratory? Well, because it's free). You can look at the notebook [here](https://github.com/Attol8/paintings-classifier/blob/master/Paintings_Classifier_kaggle.ipynb). 
# 
# Some highlights from the modelling, learning and Trainging process:
# 
# * The deep  Neural Network model trained on [best-artworks-of-all-time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset uses `ResNet50` as the architecture
# * The model has been pretrained on Imagenet
# * fastai default [`get_transform`](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) performs Data Augmentation on the datataset: flip, rotate, zoom, warp, lighting transforms are applied to the paintings 
# * The final model is the result of many experiments with different architectures, learning rates, batch sizes and image sizes. 
# * First, we train only the last group of layers, then (after using `learn.unfreeze()`) we use different learning rates for different groups of layers in the network (`max_lr=slice(1e-5,1e-4)`). This technique is called disicriminative learning 
# * The biggest improvement in terms of accuracy seems to be yielded by setting image sizes to 256x256

# In[ ]:


artists


# In[ ]:


path = images_dir
path
get_ipython().system('ls {path}')


# In[ ]:


#bs,size=32, 128
#bs,size = 24,160
bs,size = 24,256
arch = models.resnet50


# In[ ]:


path = Path("../input/modelpaintings/valid.csv")
valid_df = pd.read_csv(path)
valid_df = valid_df['0'].str.replace('data/images', '')

i = 0
for vn in valid_df: 
    vn = os.path.basename(vn)
    valid_df[i] = vn
    i+=1
    
valid_df.to_csv("../working/valid.csv", index= False, index_label=False)


# In[ ]:


valid_df.head()


# In[ ]:


valid_names = loadtxt_str("../working/valid.csv")
valid_names


# In[ ]:


include = artists
path=images_dir

src = (ImageList.from_folder(path)
.filter_by_folder(include=include)
.filter_by_func(lambda fname: Path(fname).suffix == '.jpg')
.split_by_fname_file(path = Path('../working/'), fname= 'valid.csv')
.label_from_folder())


# In[ ]:


data= (src.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))


# In[ ]:


len(data.classes)==len(artists)


# In[ ]:


model_dir = Path("/kaggle/working")
learn = cnn_learner(data, arch, metrics=accuracy, model_dir=model_dir)


# In[ ]:


model_path = Path("../input")
get_ipython().system('ls {model_path}')


# In[ ]:


model_path = Path("../input/modelpaintings/")
dest_path = model_dir
get_ipython().system("cp {model_path}/'best-2.pth' {dest_path}")


# In[ ]:


get_ipython().system('ls {dest_path}')


# In[ ]:


model_path= Path(dest_path/'best-2')
learn.load(model_path)


# ## Model Evaluation

# In[ ]:


from prettytable import PrettyTable

def validate_withtext():
    val_loss, acc = learn.validate()
    t = PrettyTable(['val_loss', 'accuracy'])
    t.add_row([val_loss, round(float(acc), 6)])
    return print(t) 

validate_withtext()


# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=[20, 20])


# In[ ]:


interp.most_confused()


# ## CNN Visualization - Heatmap

# In[ ]:


learn.summary()


# In[ ]:


idx= np.random.randint(len(data.valid_ds))
x,y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]


# In[ ]:


x.shape


# In[ ]:


m = learn.model.eval();


# In[ ]:


m


# In[ ]:


m[0]


# In[ ]:


xb, _ = data.one_item(x)
xb_d, _ = data.one_item(x, denorm=True)
xb_im = vision.Image(xb_d[0])
xb = xb.cuda()


# In[ ]:


xb_im.shape


# In[ ]:


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g


# In[ ]:


hook_a,hook_g = hooked_backward()


# In[ ]:


acts  = hook_a.stored[0].cpu()
acts.shape


# In[ ]:


avg_acts = acts.mean(0)
avg_acts.shape


# In[ ]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,256,256,0),
              interpolation='bilinear', cmap='magma');


# In[ ]:


show_heatmap(avg_acts)


# ## More Visualization - Guided Backpropagation 

# Guided backpropagation is a technique to vetter visualize and evaluate CNNs. Backpropagation is used to visualize which parts of the input picture most activate the output prediction. In order to do so, after a forward pass, when backpropagating the output, we set negative gradients to 0 (see clamp_gradients_hook function). This way we only keep positive gradients that corresponds to the parts of the network that highly activated the output prediction we are visualizing.
# 
# ![Imgur](https://i.imgur.com/hBCJR8W.png)
# Kersner, M. (2018). CNN Visualization. [online] Available at: http://seoulai.com/presentations/CNN_Visualizations.pdf     

# In[ ]:


get_ipython().system('[Paintings-museum](https://imgur.com/YQPExLw)')


# In[ ]:


print(os.listdir("../usr/lib"))


# In[ ]:


print(os.listdir("../usr/lib/pytorch_cnn_visualization"))


# In[ ]:


print(os.listdir("../usr/lib/pytorch_cnn_visualization/src"))


# In[ ]:


# python 3

from src.misc_functions import *


# In[ ]:


def new_get_example_params(example_index):
    
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    prep_img, target_class = data.valid_ds[example_index]
    file_name_to_export = str(target_class)
    original_image = prep_img
    original_image,_ = data.one_item(original_image, denorm=True)
    original_image = vision.Image(original_image[0])
    prep_img, _ = data.one_item(prep_img)
    prep_img = prep_img.cuda()
    # Define model
    pretrained_model = m
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


# In[ ]:


target_example = np.random.randint(len(data.valid_ds))
(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =    new_get_example_params(target_example)


# In[ ]:


def minmax_norm(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


# In[ ]:


def clamp_gradients_hook(module, grad_in, grad_out):
    for grad in grad_in:
        torch.clamp_(grad, min=0.0)


# In[ ]:


def hooked_ReLU(pretrained_model,prep_img,target_class):
    relu_modules = [module[1] for module in m.named_modules() if str(module[1]) == "ReLU(inplace)"]
    with callbacks.Hooks(relu_modules, clamp_gradients_hook, is_forward=False) as _:
        preds = pretrained_model(prep_img)
        preds[0,int(target_class)].backward()


# In[ ]:


def guided_backprop(pretrained_model,prep_img,target_class):
    m = pretrained_model.eval();
    prep_img.requires_grad_();
    if not prep_img.grad is None:
        prep_img.grad.zero_(); 
    hooked_ReLU(m,prep_img,target_class);
    return prep_img.grad[0].cpu().numpy()


# In[ ]:


def get_backprop_image(pretrained_model,prep_img,target_class, show=True):
    '''
    Main function to get xb_grad for guided backprop
    '''
    m = pretrained_model.eval();
    prep_img_grad = guided_backprop(pretrained_model,prep_img,target_class) # (3,256, 256). Gradient of the output w.r.t. input image       
    #minmax norm the grad
    prep_img_grad = minmax_norm(prep_img_grad)
    
    # multiply xb_grad and hmap_scaleup and switch axis
    #xb_grad = np.einsum('ijk, jk->jki',xb_grad, hmap_scaleup) #(256,256,3)
    prep_img_grad = np.transpose(prep_img_grad, (1,2,0))
    if show == True:
        return plt.imshow(prep_img_grad)
    else:
        return prep_img_grad


# In[ ]:


get_backprop_image(pretrained_model,prep_img,target_class)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def plot_GBP_and_input():

    columns = 2
    rows = 5
    figsize = [20, 26]
    
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)
    
    for i in range(rows):
        
        #plot inputs
        idx= np.random.randint(len(data.valid_ds))
        (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =        new_get_example_params(idx)
        
        x,y = data.valid_ds[idx]
        x, _ = data.one_item(x, denorm=True)
        x = x.cpu().numpy()
        x = np.transpose(x[0], (1, 2, 0))
        ax[i, 0].imshow(x)
        ax[i, 0].set_title(F"INPUT, {y}")
        
        #plot gradients
        ax[i, 1].imshow(get_backprop_image(pretrained_model,prep_img,target_class, show=False))
        ax[i, 1].set_title("GBP")

    return plt.show()


# In[ ]:


plot_GBP_and_input()

