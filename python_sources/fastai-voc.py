#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import fastai
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# In[ ]:


fastai.__version__


# In[ ]:


#!git clone https://github.com/meetshah1995/pytorch-semseg
#import sys
#sys.path.append('/kaggle/working/pytorch-semseg')


# In[ ]:


#borrowed from  https://github.com/meetshah1995/pytorch-semseg
def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


# In[ ]:


path = Path('../input/dataset')
path_2012 = path/'VOC2012'
path_img = path_2012/'JPEGImages'
path_lbl = path_2012/'SegmentationClass'
path_lbl_obj = path_2012/'SegmentationObject'


# In[ ]:


lbl_obj_names = get_image_files(path_lbl_obj)
lbl_obj_names[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3]


# In[ ]:


(path_img/f'{lbl_names[0].stem}.jpg').exists()


# In[ ]:


fnames = [path_img/f'{lbl.stem}.jpg' for lbl in lbl_names]
len(fnames), fnames[:3]


# In[ ]:


img = open_image(fnames[0])
img.show(figsize=(5, 5))


# In[ ]:


img = open_image(lbl_names[0])
img.show(figsize=(5, 5))


# In[ ]:


(img.data)[:, 190, 250]


# In[ ]:


mask = open_mask(lbl_names[0], convert_mode='RGB')
mask.show(figsize=(5,5), alpha=1)


# In[ ]:


mask.data.shape


# In[ ]:


mask.data.unique().tolist()


# In[ ]:


lbl_names[10]


# In[ ]:


mask = open_mask(lbl_names[10], convert_mode='RGB')
mask.data.dtype


# In[ ]:


mask = open_mask(lbl_names[10], convert_mode='RGB')
mask.show(alpha=1)
mask.data.unique()
print(mask.shape)

mask = mask.data.numpy().astype(int).transpose(1, 2, 0)
new_mask = encode_segmap(mask)
new_mask.shape, np.unique(new_mask)
#new_image = PIL.Image.fromarray(new_mask)


# In[ ]:


new_mask1 = new_mask.reshape(*new_mask.shape, 1)
new_mask1.shape


# In[ ]:


mask_path = Path('/kaggle/working/masks')
mask_path.mkdir(exist_ok=True)
#new_image = PIL.Image.fromarray(new_mask1.astype(np.uint8))
new_image = PIL.Image.fromarray(new_mask.astype(np.uint8))
new_image.save(mask_path/f'{lbl_names[10].name}')


# In[ ]:


#conver masks and save them
mask_path = Path('/kaggle/working/masks')
mask_path.mkdir(exist_ok=True)
for lname in lbl_names:
    mask = open_mask(lname, convert_mode='RGB')
    #mask.show(alpha=1)
    #mask.data.unique()

    mask = mask.data.numpy().astype(int).transpose(1, 2, 0)
    new_mask = encode_segmap(mask)
    new_image = PIL.Image.fromarray(new_mask.astype(np.uint8))
    new_image.save(mask_path/f'{lname.name}')
    


# In[ ]:


#ls masks/


# In[ ]:


mask_list = []
for ln in lbl_names:
    mask = open_mask(ln)
    mask_list.extend(mask.data.unique().tolist())
mask_list = sorted(list(set(mask_list)))
len(lbl_names), len(mask_list), mask_list


# In[ ]:


#classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#classes = [0, 128, 89, 33, 132, 37, 38, 72, 75, 108, 14, 112, 113, 147, 19, 52, 150, 57, 220, 94]
classes = list(range(21))
classes


# In[ ]:


def stem2file(stem, suffix='jpg'):
    return f'{stem}.{suffix}'

def df_stem2file(df, suffix='jpg'):
    for k in range(len(df)):
        df.at[k, 'file'] = stem2file(df.loc[k, 'file'], suffix)
    return df


# In[ ]:


path_info = path_2012/'ImageSets/Segmentation'
all_seg = pd.read_csv(path_info/'trainval.txt', header=None)
all_seg.columns = ['file']
all_seg = df_stem2file(all_seg, 'jpg')
#all_seg


# In[ ]:


val_seg = pd.read_csv(path_info/'val.txt', header=None)
val_seg.columns = ['file']
#val_seg.loc[0, 'file'] = path_img/f'{val_seg.loc[0, "file"]}.jpg'
#val_seg.at[0, 'file'] = stem2path(val_seg.loc[0, 'file'], path_img, 'jpg')
#val_seg.loc[0, 'file'].exists()


#val_seg = df_stem2path(val_seg, path_img, 'jpg')
#val_seg.head(5)

all_seg_list = list(range(len(all_seg)))
random.shuffle(all_seg_list)
val_idxes = all_seg_list[:400]
val_idxes[:10]


# In[ ]:


src_size = np.array(mask.shape[1:])
src_size,mask.data


# In[ ]:


size = src_size//2
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


size = 224


# In[ ]:


#path_lbl/f'{all_seg.iloc[0, 0][:-4]}.png'
#get_y_fn = lambda x: path_lbl/f'{Path(x).stem}.png'
get_y_fn = lambda x: mask_path/f'{Path(x).stem}.png'


# In[ ]:


src = (SegmentationItemList.from_df(all_seg, path=path_img)
       .split_by_idx(val_idxes)
       .label_from_func(get_y_fn, classes=classes)
      )


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs, num_workers=2)
       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(2, figsize=(10, 7))


# In[ ]:


data.show_batch(2, figsize=(10, 7), ds_type=DatasetType.Valid)


# In[ ]:


#name2id = {v:k for k,v in enumerate(codes)}
#void_code = name2id['Void']


# In[ ]:


def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics=acc_camvid
# metrics=accuracy


# In[ ]:


model_dir = '/kaggle/working/models'


# In[ ]:


#learn = unet_learner(data, models.resnet34, metrics=accuracy, wd=1e-2, model_dir='/kaggle/working/models')
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=1e-2, model_dir=model_dir)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr=2e-3


# In[ ]:


learn.fit_one_cycle(10, slice(lr), pct_start=0.9)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.show_results(rows=3, figsize=(8,9))


# In[ ]:


learn.unfreeze()


# In[ ]:


lrs = slice(lr/400,lr/4)


# In[ ]:


learn.fit_one_cycle(12, lrs, pct_start=0.8)


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.destroy() # uncomment once 1.0.46 is out

size = src_size

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 14200: bs=4
elif free > 8200: bs=3
else:           bs=1
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=metrics, wd=1e-2, model_dir=model_dir)


# In[ ]:


learn.load('stage-2')


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr=1e-3


# In[ ]:


learn.fit_one_cycle(10, slice(lr), pct_start=0.8)


# In[ ]:


learn.save('stage-1-big')


# In[ ]:


learn.load('stage-1-big')


# In[ ]:



learn.unfreeze()


# In[ ]:



lrs = slice(1e-6,lr/10)


# In[ ]:


learn.fit_one_cycle(10, lrs)


# In[ ]:


learn.save('stage-2-big')


# In[ ]:


learn.load('stage-2-big')


# In[ ]:


learn.show_results(rows=3, figsize=(10,10))

