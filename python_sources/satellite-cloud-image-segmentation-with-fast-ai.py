#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
from time import time

import numpy as np
import pandas as opd

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

init_notebook_mode(connected=True)

import warnings
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# In[ ]:


from fastai.utils.show_install import show_install; show_install()


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


def fmt_now():
    return datetime.today().strftime('%Y%m%d-%H%M%S')


# ## EDA

# In[ ]:


path = Path('../input')
path.ls()


# In[ ]:


path_img = path/'train_images'

fnames_train = get_image_files(path_img)
for f in fnames_train[:3]:
    print(f)
print('\nTotal number of training images: {}'.format(len(fnames_train)))


# In[ ]:


path_test = path/'test_images'

fnames_test = get_image_files(path_test)
for f in fnames_test[:3]:
    print(f)
print('\nTotal number of test images: {}'.format(len(fnames_test)))


# In[ ]:


img_f = fnames_train[2]
img = open_image(img_f)
img.show(figsize=(10, 10))


# In[ ]:


def split_img_label(img_lbl):
    """Return image and label from filename """
    s = img_lbl.split("_")
    assert len(s) == 2
    return s[0], s[1]


# In[ ]:


train = pd.read_csv(f'{path}/train.csv')
train['Image'] = train['Image_Label'].apply(lambda img_lbl: split_img_label(img_lbl)[0])
train['Label'] = train['Image_Label'].apply(lambda img_lbl: split_img_label(img_lbl)[1])
del train['Image_Label']
train.head(5)


# In[ ]:


train_with_mask = train.dropna(subset=['EncodedPixels'])

#colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=train_with_mask['Label'].value_counts().index, 
                             values=train_with_mask['Label'].value_counts().values)])

fig.update_traces(hoverinfo="label+percent+name")

fig.update_layout(height=600, width=900, title = 'Class distribution')

fig.show()


# In[ ]:


class_counts = train.dropna(subset=['EncodedPixels']).groupby('Image')['Label'].nunique()

fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x = class_counts,
        xbins=dict(
        start=0.5,
        end=4.5,
        size=1
        ),
    )
)

fig.update_layout(height=450, width=900, title = 'Distribution of no. labels per image')

fig.update_layout(
    xaxis_title_text='No. Image Class Labels', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

fig.update_xaxes(tickvals=[1, 2, 3, 4])

fig.show()


# In[ ]:


train = train.pivot(index='Image', columns='Label', values='EncodedPixels')
assert len(train) == len(fnames_train)
train.head()


# ### Broken images

# In[ ]:


def show_img_fn(fname, figsize=(10, 10)):
    img = open_image(fname)
    img.show(figsize=figsize)


# In[ ]:


def show_img_info(fname):
    show_img_fn(path_img/fname)
    display(train.loc[[fname]])


# In[ ]:


unusual_imgs = ["1588d4c.jpg", "c0306e5.jpg", "c26c635.jpg", "fa645da.jpg", "41f92e5.jpg", "e5f2f24.jpg"]


# In[ ]:


for fname in unusual_imgs:
    img = open_image(path_img/fname)
    img.show(figsize=(5, 5), title=fname)     


# ### Convert masks from/to RLE

# In[ ]:


train_img_dims = (1400, 2100)


# In[ ]:


def rle_to_mask(rle, shape):
    mask_img = open_mask_rle(rle, shape)
    mask = mask_img.px.permute(0, 2, 1)
    return mask


# In[ ]:


def mask_to_rle(mask):
    """ Convert binary 'mask' to RLE string """
    return rle_encode(mask.numpy().T)


# In[ ]:


def test_mask_rle():
    """ test case for mask RLE encode/decode"""
    mask_rle = train.iloc[0]['Fish']
    mask = rle_to_mask(mask_rle, train_img_dims)
    mask_rle_enc = mask_to_rle(mask)
    assert mask_rle_enc == mask_rle
    
    print(mask.shape)
    Image(mask).show()
    
test_mask_rle()


# ## Load images

# In[ ]:


item_list = (SegmentationItemList
            .from_df(df=train.reset_index(), path=path_img, cols='Image')
             #.use_partial_data(sample_pct=0.1)
            .split_by_rand_pct(0.2)
            )


# In[ ]:


class MultiLabelImageSegment(ImageSegment):
    """Store overlapping masks in separate image channels"""

    def show(self, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        cmap:str='tab20', alpha:float=0.5, class_names=None, **kwargs):
        "Show the masks on `ax`."
             
        # put all masks into a single channel
        flat_masks = self.px[0:1, :, :].clone()
        for idx in range(1, self.shape[0]): # shape CxHxW
            mask = self.px[idx:idx+1, :, :] # slice tensor to a single mask channel
            # use powers of two for class codes to keep them distinguishable after sum 
            flat_masks += mask * 2**idx
        
        # use same color normalization in image and legend
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2**self.shape[0]-1)
        ax = show_image(Image(flat_masks), ax=ax, hide_axis=hide_axis, cmap=cmap, norm=norm,
                        figsize=figsize, interpolation='nearest', alpha=alpha, **kwargs)
        
     # custom legend, see https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
        cm = matplotlib.cm.get_cmap(cmap)
        legend_elements = []
        for idx in range(self.shape[0]):
            c = 2**idx
            label = class_names[idx] if class_names is not None else f"class {idx}"
            line = Line2D([0], [0], color=cm(norm(c)), label=label, lw=4)
            legend_elements.append(line)
        ax.legend(handles=legend_elements)
        
        # debug info
        # ax.text(10, 10, f"px={self.px.size()}", {"color": "white"})
        
        if title: ax.set_title(title)

    def reconstruct(self, t:Tensor): 
        return MultiClassImageSegment(t)
        


# In[ ]:


# source: https://forums.fast.ai/t/unet-how-to-get-4-channel-output/54674/4
def bce_logits_floatify(input, target, reduction='mean'):
    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)


# In[ ]:


class MultiLabelSegmentationLabelList(SegmentationLabelList):
    """Return a single image segment with all classes"""
    # adapted from https://forums.fast.ai/t/how-to-load-multiple-classes-of-rle-strings-from-csv-severstal-steel-competition/51445/2
    
    def __init__(self, items:Iterator, src_img_size=None, classes:Collection=None, **kwargs):
        super().__init__(items=items, classes=classes, **kwargs)
        self.loss_func = bce_logits_floatify
        self.src_img_size = src_img_size
        # add attributes to copy by new() 
        self.copy_new += ["src_img_size"]
    
    def open(self, rles):        
        # load mask at full resolution
        masks = torch.zeros((len(self.classes), *self.src_img_size)) # shape CxHxW
        for i, rle in enumerate(rles):
            if isinstance(rle, str):  # filter out NaNs
                masks[i] = rle_to_mask(rle, self.src_img_size)
        return MultiLabelImageSegment(masks)
    
    def analyze_pred(self, pred, thresh:float=0.0):
        # binarize masks
        return (pred > thresh).float()
    
    
    def reconstruct(self, t:Tensor): 
        return MultiLabelImageSegment(t)


# In[ ]:


class_names = ["Fish", "Flower", "Gravel", "Sugar"]


# In[ ]:


def get_masks_rle(img):
    """Get RLE-encoded masks for this image"""
    img = img.split("/")[-1]  # get filename only
    return train.loc[img, class_names].to_list()


# In[ ]:


img_size = (84, 132)  # use multiple of 4
img_size


# In[ ]:


classes = [0, 1, 2, 3] # no need for a "void" class: if a pixel isn't in any mask, it is not labelled
item_list = item_list.label_from_func(func=get_masks_rle, label_cls=MultiLabelSegmentationLabelList, 
                                      classes=classes, src_img_size=train_img_dims)


# In[ ]:


item_list = item_list.add_test_folder(path_test, label="")


# In[ ]:



batch_size = 8

# TODO add data augmentation
tfms = ([], [])
# tfms = get_transforms()

item_list = item_list.transform(tfms, tfm_y=True, size=img_size)


# In[ ]:


data = (item_list
        .databunch(bs=batch_size)
        .normalize(imagenet_stats) # use same stats as pretrained model
       )  
assert data.test_ds is not None


# In[ ]:


data.show_batch(2, figsize=(15, 10), class_names=class_names)


# ## Training

# In[ ]:


def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


# In[ ]:


metrics = [dice_metric]

callback_fns = [
    # update a graph of learner stats and metrics after each epoch
    ShowGraph,

    # save model at every metric improvement
    partial(SaveModelCallback, every='improvement', monitor='dice_metric', name=f"{fmt_now()}_unet_resnet18_stage1_best"),
    
    # stop training if metric no longer improve
    partial(EarlyStoppingCallback, monitor='dice_metric', min_delta=0.01, patience=2),
]

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=1e-2, callback_fns=callback_fns)
learn.model_dir = "/kaggle/working/"  # point to writable directory


# In[ ]:


learn.loss_func


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, max_lr=1e-4)


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.save(f"{fmt_now()}_unet_resnet18_stage1", return_path=True)


# In[ ]:


get_ipython().system('ls -lth {learn.model_dir}')


# ### unfreeze and differential learing rate

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, max_lr=slice(5e-7, 5e-6))


# In[ ]:


learn.save(f"{fmt_now()}_unet_resnet18_stage2", return_path=True)


# ## Further tuning with larger images

# In[ ]:


img_size = (336, 528)
item_list = item_list.label_from_func(func=get_masks_rle, label_cls=MultiLabelSegmentationLabelList, 
                                      classes=classes, src_img_size=train_img_dims)
item_list = item_list.add_test_folder(path_test, label="")
batch_size = 8
tfms = ([], [])
item_list = item_list.transform(tfms, tfm_y=True, size=img_size)


# In[ ]:


data = (item_list
        .databunch(bs=batch_size)
        .normalize(imagenet_stats) # use same stats as pretrained model
       )  
assert data.test_ds is not None


# In[ ]:


data.show_batch(2, figsize=(15, 10), class_names=class_names)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 2e-4


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


lr=5e-5


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save(f"{fmt_now()}_unet_resnet18_stage3", return_path=True)


# In[ ]:


learn.show_results(imgsize=8, class_names=class_names)

