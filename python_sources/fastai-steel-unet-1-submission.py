#!/usr/bin/env python
# coding: utf-8

# # Using Fastai
# 
# Based on https://www.kaggle.com/nikhilikhar/fastai-steel-unet

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
category_num = len(codes)
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# In[ ]:


train_df = pd.read_csv(path/"severstal-steel-defect-detection/train.csv")
train_df[['ImageId', 'ClassId']] = train_df['ImageId_ClassId'].str.split('_', expand=True)
# train_df.head()


# In[ ]:


image_df = pd.DataFrame(train_df['ImageId'].unique())
image_df.head()


# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = 4
wd=1e-2

def acc_steel(input, target):
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


# In[ ]:


size = 256, 1600

src = (SegmentationItemList.from_df(image_df, path_img,)
       .split_by_rand_pct(valid_pct=0.2, seed=33)
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize()
       )


# In[ ]:


data.show_batch(2, figsize=(20,5))


# In[ ]:


data.show_batch(2, figsize=(20,5),ds_type=DatasetType.Valid)


# # Model

# In[ ]:





# # Create/Load Learner
# We will load our trained learner.
# 
# We are loading trained model from https://www.kaggle.com/nikhilikhar/fastai-steel-unet?scriptVersionId=18713214

# In[ ]:


# learner, include where to save pre-trained weights (default is in non-write directory)
# learn = unet_learner(data, models.resnet18, metrics=[acc_steel, iou], wd=wd, 
#                      model_dir="/kaggle/working/models")

learn = load_learner("../input/fastai-steel-unet/", file="steel-2.pkl")
learn.data.single_ds.tfmargs['size'] = None


# # Test

# In[ ]:


# def get_predictions(path_test, learn, size):
#     learn.model.cuda()
#     import pdb; pdb.set_trace()
#     files = list(path_test.glob("**/*.jpg"))    #<---------- HERE
#     test_count = len(files)
#     results = {}
#     for i, img in enumerate(files):
#         results[img.stem] = learn.predict(open_image(img))[1].data.numpy().flatten()
    
#         if i%20==0:
#             print("\r{}/{}".format(i, test_count), end="")
#     return results

# predicts = get_predictions(path_test, learn, size)
# len(predicts)


# In[ ]:


# https://www.kaggle.com/go1dfish/u-net-baseline-by-pytorch-in-fgvc6-resize
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

