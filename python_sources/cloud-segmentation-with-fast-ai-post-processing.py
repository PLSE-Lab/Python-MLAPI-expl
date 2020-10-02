#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
from time import time

import seaborn as sns
import numpy as np
import pandas as opd
import cv2

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


# In[ ]:


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(final_size, np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


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



def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)


    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)
        
    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)


    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)


# ## EDA

# In[ ]:


path = Path('../input/understanding_cloud_organization/')
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
    
    print(mask.shape)
    Image(mask).show()
    assert mask_rle_enc == mask_rle
    
    
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


test_items = ItemList.from_folder(path_test)
item_list = item_list.add_test(test_items)


# In[ ]:


item_list


# In[ ]:


get_ipython().system('ls -lth "../input/understanding_cloud_organization/test_images/"')


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


path_test


# In[ ]:


data.show_batch(4, figsize=(15, 10), class_names=class_names)


# > ## Training (loading fine-tuned model from other competition's mates)

# In[ ]:


def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


# In[ ]:


get_ipython().system('cp ../input/satellite-cloud-image-segmentation-with-fast-ai/*.pth /kaggle/working/')


# In[ ]:


data.test_ds


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

learn = unet_learner(data, models.resnet18, metrics=metrics, wd=1e-2, callback_fns=callback_fns)
learn.model_dir = "/kaggle/working/" # point to writable directory
learn.load("20190928-235953_unet_resnet18_stage2")


# In[ ]:


learn.validate(learn.data.valid_dl)


# In[ ]:


learn.loss_func


# In[ ]:


#valid_preds = learn.get_preds(ds_type=DatasetType.Valid)


# In[ ]:


#valid_dataset = zip(learn.data.valid_ds.x, learn.data.valid_ds.y)


# In[ ]:


final_size = (350, 525)
final_size_swap = (final_size[1], final_size[0])


# > # Get predictions

# In[ ]:


attempts = []
number_images = len(learn.data.valid_ds)

for i, (image, mask) in enumerate(tqdm(learn.data.valid_ds)):
    
    valid_masks = []
    predict_masks = []
    output = learn.predict(image)[2] # 2 means logits
    
    for m in mask.px:
        np_mask = m.numpy()
        if np_mask.shape != final_size:
            np_mask = cv2.resize(np_mask, dsize=final_size_swap, interpolation=cv2.INTER_LINEAR)
        valid_masks.append(np_mask)

        
    for j, prob in enumerate(output):
        probability = prob.numpy().astype('float32')
        if probability.shape != final_size:
            probability = cv2.resize(probability, dsize=final_size_swap, interpolation=cv2.INTER_LINEAR)
        predict_masks.append(probability)
        
    d = 1
    for class_id, (i, j) in enumerate(zip(valid_masks, predict_masks)):
        i_condition = i.sum() == 0
        for t in range(0, 100, 5):
            t /= 100
            for ms in [10000]:
                predict, _ = post_process(sigmoid(j), t, ms)
                if i_condition & (predict.sum() == 0):
                    d = 1
                else:
                    d = dice(i, predict)
                attempts.append((class_id, t, ms, d))
    
    del valid_masks
    del predict_masks
    del output


# In[ ]:


len(attempts)


# In[ ]:


attempts_df = pd.DataFrame(attempts, columns=['class_id', 'threshold', 'size', 'dice']).groupby(['class_id', 'threshold', 'size'], as_index=False).dice.mean()
attempts_df.head()


# In[ ]:


assert(attempts_df.shape[0] * number_images == len(attempts))


# In[ ]:


class_params = {}

for class_id in attempts_df.class_id.unique():
    class_df = attempts_df[attempts_df.class_id == class_id].sort_values('dice', ascending=False)
    print(class_df.head())
    best_threshold = class_df['threshold'].values[0]
    best_size = class_df['size'].values[0]
    class_params[class_id] = (best_threshold, best_size)
    


# In[ ]:


sns.lineplot(x='threshold', y='dice', data=attempts_df);
plt.title('Threshold and min size vs dice for one of the classes');


# In[ ]:


for i, (image, mask) in enumerate(learn.data.valid_ds):
    image_vis = image.px.numpy().transpose(1, 2, 0)
    mask = mask.px.numpy().astype('uint8').transpose(1, 2, 0)
    pr_mask = np.zeros((final_size[0], final_size[1], 4))
    output = learn.predict(image)[2] # 2 means logits
    output_tr = output.numpy().transpose(1, 2, 0).astype('float32')
    for j in range(4):
        probability = cv2.resize(output_tr[:, :, j], dsize=final_size_swap, interpolation=cv2.INTER_LINEAR)
        pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
    #pr_mask = (sigmoid(output.numpy()) > 0.5).astype('uint8').transpose(1, 2, 0) # TO BE REPLACED BY THE PREV LINE
    
        
    visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis, raw_mask=output_tr)
    
    if i >= 4:
        break


# # Compute final predictions using selected thresholds

# In[ ]:


import gc
torch.cuda.empty_cache()
gc.collect()


# In[ ]:


#class_params = {0: (0.5, 10000), 1: (0.5, 10000), 2: (0.45, 10000), 3: (0.55, 10000)}
class_params


# In[ ]:


def mask_to_rle_numpy(mask):
    """ Convert binary 'mask' to RLE string """
    return rle_encode(mask.T)


# In[ ]:


encoded_pixels = []
image_labels = []

image_id = 0

for i, (p, (image, mask)) in enumerate(tqdm(zip(learn.data.test_dl.items, learn.data.test_ds))):
    output = learn.predict(image)[2] # 2 means logits
    for labelpred, prob in enumerate(output):
        probability = prob.numpy().astype('float32')
        if probability.shape != final_size:
            probability = cv2.resize(probability, dsize=final_size_swap, interpolation=cv2.INTER_LINEAR)
        predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
        if num_predict == 0:
            encoded_pixels.append('')
        else:
            r = mask_to_rle_numpy(predict)
            encoded_pixels.append(r)
        image_labels.append(p.name + '_' + class_names[labelpred])
        del predict
        del probability
        image_id += 1
    del output


# In[ ]:


submission = pd.DataFrame({'Image_Label': image_labels, 'EncodedPixels': encoded_pixels}).drop_duplicates()
submission.head()


# In[ ]:


assert(submission.Image_Label.nunique() == submission.shape[0])


# In[ ]:


submission.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)

