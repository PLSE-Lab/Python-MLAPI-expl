#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *


# In[ ]:


bs = 64


# In[ ]:


train=pd.read_csv('../input/labels.csv')
train.head()


# In[ ]:


train.id = train.id+'.jpg'


# In[ ]:


train.head()


# In[ ]:


tfms = get_transforms(max_rotate = 20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine = 1., p_lighting=1.)


# In[ ]:


src = (ImageList.from_df(train, '../input/', folder='train')
      .split_by_rand_pct()
      .label_from_df(cols = 'breed'))


# In[ ]:


def get_data(size, bs, padding_mode = 'reflection'):
    return(src.transform(tfms, size = size, padding_mode=padding_mode)
          .databunch(bs=bs, num_workers = 0).normalize(imagenet_stats))


# In[ ]:


data = get_data(224, bs, 'zeros')


# In[ ]:


def _plot(i,j,ax):
    x,y = data.train_ds[4]
    x.show(ax, y=y)
    
plot_multi(_plot, 3, 3, figsize=(8,8))


# ## Train the model

# In[ ]:


gc.collect()
learn = cnn_learner(data, models.resnet34, 
                    metrics = error_rate, bn_final = True, 
                    model_dir = "/tmp/model")


# In[ ]:


learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3), pct_start=0.8)


# In[ ]:


data = get_data(352,bs)
learn.data = data


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('352')


# ## Convolution kernel

# In[ ]:


data = get_data(352, 16)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir = '/tmp/model').load('352')


# Create a convolutional layer by hand. Take a standard 3 x 3 matrix that is going to look for lower right hand cornrers and turn it into a 3d convolutional layer with the .expand command. 
# 
# Can play around with these convolutinoal layers to see how they interact with the images
#  * Make sure to include decimal in first number to make the type of data in kernel floats
#  
#  [Good Resource for Convolution Visualization](http://setosa.io/ev/image-kernels/)

# In[ ]:


#Right Sobel kernel
k = tensor ([
    [-1., 0, 1],
    [-2., 0, 2],
    [-1., 0 , 1],
]).expand(1,3,3,3)/6


# In[ ]:


#Sharpen Kernel
k = tensor ([
    [0., -1, 0],
    [-1, 5, -1],
    [0, -1 , 0],
]).expand(1,3,3,3)


# In[ ]:


k


# In[ ]:


k.shape


# In[ ]:


idx = 2
t = data.valid_ds[idx][0].data; t.shape #Pull out a single image sample, 0th is image and 1st is label


# In[ ]:


t[None].shape #t[None] is a trick to get a mini-batch of a tensor of size 1, this also works in numpy


# In[ ]:


edge = F.conv2d(t[None], k)


# In[ ]:


show_image(edge[0], figsize = (5,5))


# In[ ]:


x,y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]


# This visualization of the convolutinoal layer shows how a convolution can identify an edge/corner, etc

# In[ ]:


#Number of categories
data.c


# In[ ]:


#Details of the resnet model 
learn.model
#theres a lot going on in the first Conv layer, but there are 64 chanels and a stride of 2 for the first layer
#When you stride by 2 you can double the number of chanels (this preserves the complexity of model/memory)


# In[ ]:


print(learn.summary())


# ## Create Heatmap
# 
# The images get boiled down into 11 'sections' of the image through various stride-ings and 512 kernel based channels of for instance (how fluffy is it, how long are ears, etc)
# * So for each 11x11 image, there's an activation for each part of the image for each of the 512 features
# * The network determines itself what are the features based on the optimization of the model
# * If we take the average of all the 512 features we can determine how activated each of the 11x11 parts of the images are

# In[ ]:





# In[ ]:


m = learn.model.eval();


# Before getting heatmap we need to:
#  * Put data into pytorch databunch (in this case a single item mini batch)
#  * Normalzie the image
#  * Put it on the GPU

# In[ ]:


xb,_ = data.one_item(x) #takes all the settings from our previously created data object
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


# In[ ]:


from fastai.callbacks.hooks import *


# Here we're taking the output of the convolutional layers found in ResNet34, which in the model we've created is found with the m[0] part of the model
# 
# hook_output is a fastai module that pulls the output of (in this case m[0]) out of the pytorch back end

# In[ ]:


get_ipython().run_line_magic('pinfo2', 'hook_output')


# In[ ]:


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: #Get activations from the convolution layers
        with hook_output(m[0], grad = True) as hook_g: #Get gradient from convolution layers
            preds = m(xb) #DO foreward pass through model
            preds[0,int(cat)].backward()
    return hook_a, hook_g


# In[ ]:


hook_a, hook_g = hooked_backward()


# In[ ]:


hook_a.stored


# In[ ]:


acts = hook_a.stored[0].cpu()
acts.shape #Now we see our 512 chanels over the 11x11 sections of the image


# Hook allows you to hook into the pytorch machinary itself and run any python code you want

# In[ ]:


avg_acts = acts.mean(0)
avg_acts.shape


# In[ ]:


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax) #fastai function to show the image
    ax.imshow(hm, alpha = 0.6, extent = (0,352,352, 0), #extent expands the 11x11 image to 352,352
             interpolation = 'bilinear', cmap = 'magma')


# In[ ]:


show_heatmap(avg_acts)

