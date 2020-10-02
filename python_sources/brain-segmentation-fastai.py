#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# In[ ]:


# we need to make pre-trained models available to this kernel, without internet access.
# output from creating the learner tells us that a pre-trained model is needed;
#   Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
# to make this work without internet;
# 1) manually download the model
# 2) use "+ Add Data" to make the model available
# 3) make the temp folder (that pytorch wants to use) if it doesn't exist already
Path('/tmp/.cache/torch/checkpoints').mkdir(parents=True, exist_ok=True)
# 4) copy the model from input to the temp folder
get_ipython().system('cp /kaggle/input/pytorch-models/resnet34-333f7ec4.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth')
# 5) then make sure we set path='/kaggle/working' when creating the learner - otherwise it'll use the dir of the model (which is read-only)
output_path = '/kaggle/working'


# In[ ]:


# get the Path of the mask for a given image
get_y_fn = lambda x: x.parents[0] / (x.stem + '_mask' + x.suffix)


# In[ ]:


data_path = Path('/kaggle/input/lgg-mri-segmentation/kaggle_3m/')
small_data_path = data_path/'TCGA_HT_7680_19970202'
# small_data_path.ls()


# In[ ]:


temp_img_file = small_data_path/'TCGA_HT_7680_19970202_6.tif'
temp_mask_file = get_y_fn(temp_img_file)
for f in [temp_img_file, temp_mask_file]:
    print('showing', f)
    if '_mask.tif' in f.name:
        mask = open_mask(f)
        print(mask.shape)
        mask.show()
    else:
        temp_img = open_image(f)
        print(temp_img.shape)
        temp_img.show()


# In[ ]:


# div=True changes mask.data.unique() from [0,255] to [0,1] (by dividing pixel values by 255)
mask = open_mask(get_y_fn(temp_img_file), div=True)
mask.show(figsize=(5,5), alpha=1)


# In[ ]:


src_size = np.array(mask.shape[1:])
src_size, mask.data


# # Dataseta

# In[ ]:


# use a fixed set of images for validation - partly to make this repeatable - but ...
# I think it makes sense to validate against images for patients that have not been seen during training
validation_folders = [
        'TCGA_HT_7694_19950404', 'TCGA_DU_5874_19950510', 'TCGA_DU_7013_19860523',
        'TCGA_HT_8113_19930809', 'TCGA_DU_6399_19830416', 'TCGA_HT_7684_19950816',
        'TCGA_CS_5395_19981004', 'TCGA_FG_6688_20020215', 'TCGA_DU_8165_19970205',
        'TCGA_DU_7019_19940908', 'TCGA_HT_7855_19951020', 'TCGA_DU_A5TT_19980318',
        'TCGA_DU_7300_19910814', 'TCGA_DU_5871_19941206', 'TCGA_DU_5855_19951217']


# In[ ]:


# v simple codes; n=nothing to see here, y=area of interest
codes = ['n', 'y']

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
bs=free//500
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# The mask files are single channel 256x256 images using pixel values 0 and 255.
# 
# If we're using grey scale; 0 is black, 255 is white.
# 
# Normally, the open_mask div=True option means divide pixel values by 255 to convert the range of pixel values from 0-255 to 0-1.
# 
# In this case, dividing by 255 gives us just 2 classes of pixels; 0=nothing to see here, 1=area of interest - and we need these to correspond to the codes (AKA classes) defined above

# In[ ]:


# we need to open the make files with the div=True option - which we can do with a custom label list / item list.
# thanks to https://www.kaggle.com/tanlikesmath/ultrasound-nerve-segmentation-with-fastai/data for showing how this can be done
class SegmentationLabelListWithDiv(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
class SegmentationItemListWithDiv(SegmentationItemList):
    _label_cls = SegmentationLabelListWithDiv


# In[ ]:


src = (SegmentationItemListWithDiv.from_folder(data_path, recurse=True)
       .filter_by_func(lambda x: not x.name.endswith('_mask.tif'))
       .split_by_valid_func(lambda x: x.parts[-2] in validation_folders)
       .label_from_func(get_y_fn, classes=codes))


# In[ ]:


# start by training on half size images
data = (src.transform(get_transforms(), size=src_size//2, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


data.train_ds


# In[ ]:


data.valid_ds


# In[ ]:


data.show_batch(2, figsize=(10,10))


# In[ ]:


data.show_batch(2, figsize=(10,10), ds_type=DatasetType.Valid)


# # Model

# In[ ]:


wd=1e-2


# In[ ]:


learn = unet_learner(data, models.resnet34, wd=wd, metrics=dice, path=output_path)


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


lr=3e-3


# In[ ]:


learn.fit_one_cycle(10, slice(lr), pct_start=0.9)


# In[ ]:


def save_and_show(name):
    saved_to = learn.save(name, return_path=True)
    print('Saved to', saved_to, 'Note: this will be lost unless we commit the kernel')
    learn.load(name) # free memory etc
    learn.show_results(rows=4, figsize=(32, 32))


# In[ ]:


save_and_show('stage-1')


# In[ ]:


learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(8, lrs, pct_start=0.8)


# In[ ]:


save_and_show('stage-2')


# # Train on full size images

# In[ ]:


data = (src.transform(get_transforms(), size=src_size, tfm_y=True)
        .databunch(bs=bs//2)
        .normalize(imagenet_stats))


# In[ ]:


# Export the minimal state of the data bunch for inference 
data.export('../../../working/big-databunch-export.pkl')


# In[ ]:


learn = unet_learner(data, models.resnet34, metrics=dice, wd=wd, path=output_path)
learn.load('stage-2')


# In[ ]:


lr_find(learn)
learn.recorder.plot()


# In[ ]:


# reduce learning rate for training on full size images
lr=1e-3


# In[ ]:


# we'll run 10 epochs before unfreeze - do just the 1st 3 so we can take a look at the results
learn.fit_one_cycle(3, slice(lr), pct_start=0.8)


# In[ ]:


save_and_show('stage-1-big')


# In[ ]:


learn.fit_one_cycle(7, slice(lr), pct_start=0.8)


# In[ ]:


save_and_show('stage-1-big') # overwrite the existing stage-1-big


# In[ ]:


learn.unfreeze()
lrs = slice(1e-6,lr/10)
learn.fit_one_cycle(10, lrs)


# In[ ]:


save_and_show('stage-2-big')


# In[ ]:


learn.export('stage-2-big-export.pkl')

