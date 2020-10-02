#!/usr/bin/env python
# coding: utf-8

# 
# # Intro
# ## Disclaimer
# This is a beginners view and observation. I might missed some points in the framework or may have some wrong assumptions. So please double check if you have doubts and let me know if I got some things wrong.
# 
# ## What the kernel is about
# This kernel shows how you can speed up Image Processing in Fastai when you are using a small image dataset.
# 
# Because I didn't want to bother with converting the data into images, I started the competition with this nice [kernel](https://www.kaggle.com/melissarajaram/fastai-pytorch-with-best-original-mnist-arch). That kernel trains pretty fast (about 5 sec per epoch). Unfortunately the kernel doesn't convert the images into *Fastai Images* (which are usefull for ClassificationInterpretation), instead it passes the data directly as a Dataset to pytorch as discribed [here](https://docs.fast.ai/basic_data.html#Using-a-custom-Dataset-in-fastai). 
# 
# To overcome this shortcoming I tried another nice [kernel](https://www.kaggle.com/melissarajaram/fastai-mixup-training-aug-tta) which uses Fastai ImageList. Now I got the full functionalities I wanted. Unfortunately this time the training time want down (more then 2 minutes! per epoch).
# 
# ## Analysing
# After some time crawling through the Fastai code I figuered out, that when you set up a databunch with the the default ImageList the data is not beeing preprocessed (here: converting 784 valued into 28x28 Image) until you use it. Everytime you grab an ImageList entry the get()-function gets called, which furthermore calls open(). 
# 
# In [kernel 2](https://www.kaggle.com/melissarajaram/fastai-mixup-training-aug-tta)  the customized open() fetches the image data from a DataFrame and converts it into an Image. I suppose the frequent Dataframe lookup and preprocessing takes up a lot of time.
# 
# But why is [kernel 1](https://www.kaggle.com/melissarajaram/fastai-pytorch-with-best-original-mnist-arch) faster? In kernel 1 all the data gets preprocessed in advance and passed to the custom Dataset initialy. There wan't be fetching and preprocessing necessary while training.
# 
# ## Task
# Since I want the full functionality of Fastai Images and also decent runtime I'm writing my own customized ImageList with initial loading and preprocessing.
# 
# **Note:** I guess in the framework the initial loading is not done because of memory consumption. But for this small dataset it works fine.
# 
# ## Credits
# Please visit @melissarajaram kernels and upvote if you like them. There is a lot more in there.
# * [Pytorch Dataset kernel 1](https://www.kaggle.com/melissarajaram/fastai-pytorch-with-best-original-mnist-arch)
# * [ImageList kernel 2](https://www.kaggle.com/melissarajaram/fastai-mixup-training-aug-tta)
# 
# Others
# * [numpy to Image conversion](https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist)
# 
# 

# # Preprocessing

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks import *
DATAPATH = Path('/kaggle/input/Kannada-MNIST/')
from matplotlib import pyplot

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv(DATAPATH/'train.csv')
df_train['fn'] = df_train.index
df_train.head()


# I'm using the from_df() function to do the preprocessing and loading of the images. Therefore the init() needs a variable (myimages). It is also passed in as a new parameter otherwise the classmethodes *label_from_df* and *split_by_rand_pct* (used later on) would override the myimages dictionary.
# 
# I also added an additional channel with a blurred representation of the image.

# In[ ]:


class PixelImageItemList(ImageList):
    
    def __init__(self, myimages = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.myimages = myimages 
    
    def open(self,fn):
        return self.myimages.get(fn)
    
    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='', **kwargs)->'ItemList':
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."
        res = super().from_df(df, path=path, cols=cols, **kwargs)
        
        # full load of all images
        for i, row in df.drop(labels=['label','fn'],axis=1).iterrows():
            # Numpy to Image conversion from
            # https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist
            img_pixel = row.values.reshape(28,28)
            img_pixel = np.stack((img_pixel,)*1,axis=-1)
            
            # add channel with blured image
            x=pil2tensor(img_pixel,np.float32).div_(255).unsqueeze(0)
            
            ## Blur with gaussien kernel 
            b = torch.Tensor([[1, 1, 1],
                              [1, 2, 1],
                              [1, 1, 1]])
            b = b.view((1,1,3,3))
            blurred_ch = F.conv2d(x, b, padding=1).div_(2)
            
            # construct a 3 channel image (3rd channel doesn't contain additional info here. It is just added for the show case.)
            res.myimages[res.items[i]]=vision.Image(torch.cat((x.squeeze(0), blurred_ch.squeeze(0), x.squeeze(0))))

        return res


# In[ ]:


get_ipython().run_cell_magic('time', '', "piil = PixelImageItemList.from_df(df=df_train,path='.',cols='fn')")


# In[ ]:


piil.get(4)


# In[ ]:


src = (piil
      .split_by_rand_pct()
      .label_from_df(cols='label')
      )


# In[ ]:


src


# # Creating a databunch

# In[ ]:


data = (src.databunch(bs=128))


# In[ ]:


len(data.myimages)


# In[ ]:


data.show_batch(rows=3,figsize=(10,7))


# # Model Architecture
# The model is taken from [here](https://www.kaggle.com/melissarajaram/fastai-pytorch-with-best-original-mnist-arch).

# ## Model

# In[ ]:


leak = 0.1

best_architecture = nn.Sequential(
   
    conv_layer(3,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=1,ks=3,leaky=leak),
    conv_layer(32,32,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.2),
    
    conv_layer(32,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=1,ks=3,leaky=leak),
    conv_layer(64,64,stride=2,ks=5,leaky=leak),
    nn.Dropout(0.2),
    
    Flatten(),
    nn.Linear(3136,256), 
    relu(inplace=True,leaky=0.1),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256,10)
)


# ## Learner

# In[ ]:


learn = Learner(data, best_architecture, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] ).mixup()


# In[ ]:


#learn


# # Train

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-3
n_cycle = 5

reduceLR = ReduceLROnPlateauCallback(learn=learn, monitor = 'valid_loss', mode = 'auto', patience = 2, factor = 0.2, min_delta = 0)

learn.fit_one_cycle(n_cycle, slice(lr), callbacks=[reduceLR])


# **It worked! 8 sec per epoch!**

# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(6,6))


# In[ ]:


interp.plot_confusion_matrix()

