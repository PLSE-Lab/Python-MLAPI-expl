#!/usr/bin/env python
# coding: utf-8

# ## With fastai.
# 
# Issues,
# 1. Need more than 9 hr on kaggle p100.
# 1. `acc_steel` is nan. Debug it.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import time

from itertools import groupby

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# In[ ]:


import fastai; 
fastai.__version__


# We are pre created labels in https://www.kaggle.com/nikhilikhar/steel-create-labels?scriptVersionId=18627876
# 
# We want to access the output directly in this Kernel. We will unzip label mask in `../labels`

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


# !ls -R ../input/steel-create-labels/
get_ipython().system(' apt install  -y unzip ')
get_ipython().system(' mkdir -p ../labels/')
get_ipython().system(' unzip ../input/steel-create-labels/labels-img.zip -d ../labels/')


# In[ ]:


start = time.time()
path = Path('../input')
path_lbl = Path('../labels')

path_img = path/'severstal-steel-defect-detection/train_images'
path_test = path/'severstal-steel-defect-detection/test_images'
# path_lbl.ls(), path_img.ls()


# https://forums.fast.ai/t/unet-segmentation-mask-converter-to-help-against-common-errors-problems/42949

# # Data

# In[ ]:


fnames = get_image_files(path_img)
fnames[:3]


# In[ ]:


lbl_names = get_image_files(path_lbl)
lbl_names[:3]


# In[ ]:


img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))


# In[ ]:


def get_y_fn(x):
    x = Path(x)
    return path_lbl/f'{x.stem}.png'


# In[ ]:


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)


# In[ ]:


codes = ['0','1','2','3', '4'] # ClassId = codes + 1
free = gpu_mem_get_free_no_cache()
bs = 4
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


train_df = pd.read_csv(path/"severstal-steel-defect-detection/train.csv")
train_df[['ImageId', 'ClassId']] = train_df['ImageId_ClassId'].str.split('_', expand=True)
train_df.head()


# In[ ]:


image_df = pd.DataFrame(train_df['ImageId'].unique())
image_df.head()
# 12k
# image_df = image_df.iloc[:1000]


# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = 4
wd=1e-2

def acc_steel(input, target):
#     import pdb; pdb.set_trace()
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def iou(input:Tensor, targs:Tensor) -> Rank0Tensor:
    "IoU coefficient metric for binary target."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    return intersect / (union-intersect+1.0)

metrics = [acc_steel, iou]


# In[ ]:


size = 256#, 1600

def no_tfms(self, x, **kwargs): return x
EmptyLabel.apply_tfms = no_tfms

src = (SegmentationItemList.from_df(image_df, path_img,)
       .split_by_rand_pct(valid_pct=0.2, seed=33)
       .label_from_func(get_y_fn, classes=codes)
       .add_test_folder('../test_images')
      )
data = (src.transform(get_transforms(flip_vert=True, ), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize()
       )


# In[ ]:


print("TEST ==> {}\n VALID ==> {}\n ==> TRAIN {}".format(data.test_ds, data.valid_ds, data.train_ds))


# In[ ]:


# len(path_test.ls()) # => 1801


# In[ ]:


data.show_batch(2, figsize=(20,5))


# In[ ]:


data.show_batch(2, figsize=(20,5),ds_type=DatasetType.Valid)


# # Model

# In[ ]:


# learner, include where to save pre-trained weights (default is in non-write directory)
learn = unet_learner(data, models.resnet18, metrics=metrics, wd=wd, 
                     model_dir="/kaggle/working/models")


# In[ ]:


# print(learn.model)


# In[ ]:


# lr_find(learn)
# learn.recorder.plot(skip_end=15)


# In[ ]:


# Got nan with acc_steel at lr = 1e-3
lr=3e-4
epoch = 10
learn.fit_one_cycle(epoch, slice(lr), pct_start=0.9)


# In[ ]:


learn.save('stage-1')
learn.export("/kaggle/working/steel-1.pkl")


# In[ ]:


learn.show_results()


# In[ ]:


learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(epoch, lrs, pct_start=0.8)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.show_results()


# In[ ]:


learn.save('stage-2')
learn.export("/kaggle/working/steel-2.pkl")


# ## Finish of training for part-1
# Training with new learner is pending. See at, 

# In[ ]:


# learn.destroy()
# free = gpu_mem_get_free_no_cache()
# # the max size of bs depends on the available GPU RAM
# if free > 8200: bs=3
# else:           bs=1
# print(f"Using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


# data = (src.transform(get_transforms(), size=size, tfm_y=True)
#         .databunch(bs=bs)
#         .normalize())


# In[ ]:


# learn = unet_learner(data, models.resnet18, metrics=metrics, wd=wd, model_dir="/kaggle/working/models")
# learn.load("/kaggle/working/models/stage-2")


# In[ ]:


# lr_find(learn)
# learn.recorder.plot()


# In[ ]:


# lr=1e-3
# learn.fit_one_cycle(epoch, slice(lr), pct_start=0.8)


# In[ ]:


# learn.recorder.plot_losses()


# In[ ]:


# learn.recorder.plot_metrics()


# In[ ]:


# learn.show_results(rows=5, figsize=(20,5))


# In[ ]:


# learn.unfreeze()
# lrs = slice(1e-6,lr/10)
# learn.fit_one_cycle(epoch, lrs)


# In[ ]:


# learn.recorder.plot_losses()


# In[ ]:


# learn.recorder.plot_metrics()


# In[ ]:


# learn.show_results(rows=5, figsize=(20,5))


# In[ ]:


# learn.save('stage-2-big')
# learn.export("/kaggle/working/steel-2-big.pkl")


# ## Finish of training for part-2
# Training with new learner is pending. See at, 

# In[ ]:


learn.predict(open_image("../input/severstal-steel-defect-detection/test_images/38b9631df.jpg"))[1].data.numpy().flatten()


# In[ ]:


# def get_predictions(path_test, learn):
#     # predicts = get_predictions(path_test, learn)
#     learn.model.cuda()
#     files = list(path_test.glob("**/*.jpg"))    #<---------- HERE
#     test_count = len(files)
#     results = {}
#     for i, img in enumerate(files):
#         results[img.stem] = learn.predict(open_image(img))[1].data.numpy().flatten()
    
#         if i%20==0:
#             print("\r{}/{}".format(i, test_count), end="")
#     return results    

# results = get_predictions(path_test, learn)


# In[ ]:


def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def run_length(label_vec):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != len(codes)-1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]
    return class_dict

# https://www.kaggle.com/nikhilikhar/pytorch-u-net-steel-1-submission/output#Export-File
def get_predictions(path_test, learn):
    # predicts = get_predictions(path_test, learn)
    learn.model.cuda()
    files = list(path_test.glob("**/*.jpg"))    #<---------- HERE
    test_count = len(files)
    results = []
    for i, img in enumerate(files):
        img_name = img.stem + '.jpg'
        pred = learn.predict(open_image(img))[1].data.numpy().flatten()
        class_dict = run_length(pred)
        if len(class_dict) == 0:
            for i in range(4):
                results.append([img_name+ "_" + str(i+1), ''])
        else:
            for key, val in class_dict.items():
                results.append([img_name + "_" + str(key+1), " ".join(map(str, val))])
            for i in range(4):
                if i not in class_dict.keys():
                    results.append([img_name + "_" + str(i+1), ''])
        
        
        if i%20==0:
            print("\r{}/{}".format(i, test_count), end="")
    return results    

sub_list = get_predictions(path_test, learn)


# In[ ]:


submission_df = pd.DataFrame(sub_list, columns=['ImageId_ClassId', 'EncodedPixels'])
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv", index=False)


# In[ ]:


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution Time  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

