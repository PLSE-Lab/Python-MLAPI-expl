#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# import fastai
# from fastai.vision import *
# from PIL import Image
# import zipfile
# import io
# import cv2
# import warnings
# warnings.filterwarnings("ignore")

# fastai.__version__


# In[ ]:


# nfolds = 1#4
# bs = 4
# n_cls = 4
# noise_th = 2000 #predicted masks must be larger than noise_th
# TEST = '../input/severstal-steel-defect-detection/test_images/'
# BASE = '../input/severstal-fast-ai-256x256-crops/'

# torch.backends.cudnn.benchmark = True


# In[ ]:


# #the code below modifies fast.ai functions to incorporate Hcolumns into fast.ai Dynamic Unet

# from fastai.vision.learner import create_head, cnn_config, num_features_model, create_head
# from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook, _hook_inner
# from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock

# class Hcolumns(nn.Module):
#     def __init__(self, hooks:Collection[Hook], nc:Collection[int]=None):
#         super(Hcolumns,self).__init__()
#         self.hooks = hooks
#         self.n = len(self.hooks)
#         self.factorization = None 
#         if nc is not None:
#             self.factorization = nn.ModuleList()
#             for i in range(self.n):
#                 self.factorization.append(nn.Sequential(
#                     conv2d(nc[i],nc[-1],3,padding=1,bias=True),
#                     conv2d(nc[-1],nc[-1],3,padding=1,bias=True)))
#                 #self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))
        
#     def forward(self, x:Tensor):
#         n = len(self.hooks)
#         out = [F.interpolate(self.hooks[i].stored if self.factorization is None
#             else self.factorization[i](self.hooks[i].stored), scale_factor=2**(self.n-i),
#             mode='bilinear',align_corners=False) for i in range(self.n)] + [x]
#         return torch.cat(out, dim=1)

# class DynamicUnet_Hcolumns(SequentialEx):
#     "Create a U-Net from a given architecture."
#     def __init__(self, encoder:nn.Module, n_classes:int, blur:bool=False, blur_final=True, 
#                  self_attention:bool=False,
#                  y_range:Optional[Tuple[float,float]]=None,
#                  last_cross:bool=True, bottle:bool=False, **kwargs):
#         imsize = (224,224)
#         sfs_szs = model_sizes(encoder, size=imsize)
#         sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
#         self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
#         x = dummy_eval(encoder, imsize).detach()

#         ni = sfs_szs[-1][1]
#         middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
#                                     conv_layer(ni*2, ni, **kwargs)).eval()
#         x = middle_conv(x)
#         layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

#         self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
#         hc_c = [x.shape[1]]
        
#         for i,idx in enumerate(sfs_idxs):
#             not_final = i!=len(sfs_idxs)-1
#             up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
#             do_blur = blur and (not_final or blur_final)
#             sa = self_attention and (i==len(sfs_idxs)-3)
#             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, 
#                 blur=blur, self_attention=sa, **kwargs).eval()
#             layers.append(unet_block)
#             x = unet_block(x)
#             self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
#             hc_c.append(x.shape[1])

#         ni = x.shape[1]
#         if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
#         if last_cross:
#             layers.append(MergeLayer(dense=True))
#             ni += in_channels(encoder)
#             layers.append(res_block(ni, bottle=bottle, **kwargs))
#         hc_c.append(ni)
#         layers.append(Hcolumns(self.hc_hooks, hc_c))
#         layers += [conv_layer(ni*len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]
#         if y_range is not None: layers.append(SigmoidRange(*y_range))
#         super().__init__(*layers)

#     def __del__(self):
#         if hasattr(self, "sfs"): self.sfs.remove()
            
# def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
#         norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, 
#         blur:bool=False, self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, 
#         last_cross:bool=True, bottle:bool=False, cut:Union[int,Callable]=None, 
#         hypercolumns=True, **learn_kwargs:Any)->Learner:
#     "Build Unet learner from `data` and `arch`."
#     meta = cnn_config(arch)
#     body = create_body(arch, pretrained, cut)
#     M = DynamicUnet_Hcolumns if hypercolumns else DynamicUnet
#     model = to_device(M(body, n_classes=data.c, blur=blur, blur_final=blur_final,
#         self_attention=self_attention, y_range=y_range, norm_type=norm_type, 
#         last_cross=last_cross, bottle=bottle), data.device)
#     learn = Learner(data, model, **learn_kwargs)
#     learn.split(ifnone(split_on, meta['split']))
#     if pretrained: learn.freeze()
#     apply_init(model[2], nn.init.kaiming_normal_)
#     return learn
# class SegmentationLabelList(SegmentationLabelList):
#     def open(self, fn): return open_mask(fn, div=True)
    
# class SegmentationItemList(SegmentationItemList):
#     _label_cls = SegmentationLabelList

# # Setting transformations on masks to False on test set
# def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
#     if not tfms: tfms=(None,None)
#     assert is_listy(tfms) and len(tfms) == 2
#     self.train.transform(tfms[0], **kwargs)
#     self.valid.transform(tfms[1], **kwargs)
#     kwargs['tfm_y'] = False # Test data has no labels
#     if self.test: self.test.transform(tfms[1], **kwargs)
#     return self
# fastai.data_block.ItemLists.transform = transform

# def open_mask(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=ImageSegment,
#         after_open:Callable=None)->ImageSegment:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", UserWarning)
#         x = PIL.Image.open(fn).convert(convert_mode)
#     if after_open: x = after_open(x)
#     x = pil2tensor(x,np.float32)
#     return cls(x)


# In[ ]:


# # Prediction with flip TTA
# def model_pred(learns, F_save,
#         ds_type:fastai.basic_data.DatasetType=DatasetType.Valid, 
#         tta:bool=True): #if use train dl, disable shuffling
#     for learn in learns: learn.model.eval();
#     dl = learn.data.dl(ds_type)
#     #sampler = dl.batch_sampler.sampler
#     #dl.batch_sampler.sampler = torch.utils.data.sampler.SequentialSampler(sampler.data_source)
#     name_list = [Path(n).stem for n in dl.dataset.items]
#     num_batchs = len(dl)
#     t = progress_bar(iter(dl), leave=False, total=num_batchs)
#     count = 0
#     with torch.no_grad():
#         for x,y in t:
#             x = x.cuda()
#             preds = []
#             for learn in learns:
#                 #i, hights, widths, classes
#                 py = torch.softmax(learn.model(x),dim=1).permute(0,2,3,1).detach()
#                 if tta:
#                     #you can comment some transfromations to save time
#                     flips = [[-1],[-2],[-2,-1]]
#                     for f in flips:
#                         py += torch.softmax(torch.flip(learn.model(torch.flip(x,f)),f),dim=1).permute(0,2,3,1).detach()
#                     py /= len(flips) + 1
#                 preds.append(py)
#             py = torch.stack(preds).mean(0).cpu().numpy() # taking average of all preds
#             batch_size = len(py)
#             for i in range(batch_size):
#                 taget = y[i].detach().cpu().numpy() if y is not None else None
#                 F_save(py[i],taget,name_list[count])
#                 count += 1
#     #dl.batch_sampler.sampler = sampler
    
# def save_img(data,name,out):
#     img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
#     out.writestr(name, img)
    
# #dice for threshold selection
# def dice_np(pred, targs, e=1e-7):
#     targs = targs[0,:,:]
#     pred = np.dstack([1.0 - pred.sum(-1), pred])
#     c = pred.shape[-1]
#     pred = np.argmax(pred, axis=-1)
#     dices = []
#     eps = 1e-7
#     for i in range(1,c):
#         intersect = ((pred==i) & (targs==i)).sum().astype(np.float)
#         union = ((pred==i).sum() + (targs==i).sum()).astype(np.float)
#         dices.append((2.0*intersect + eps) / (union + eps))
#     return np.array(dices).mean()


# In[ ]:


# def enc2mask(encs, shape=(1600,512)):
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for m,enc in enumerate(encs):
#         if isinstance(enc,np.float) and np.isnan(enc): continue
#         s = enc.split()
#         for i in range(len(s)//2):
#             start = int(s[2*i]) - 1
#             length = int(s[2*i+1])
#             img[start:start+length] = 1 + m
#     return img.reshape(shape).T

# def mask2enc(mask, n=n_cls):
#     pixels = mask.T.flatten()
#     encs = []
#     for i in range(1,n+1):
#         p = (pixels == i).astype(np.int8)
#         if p.sum() == 0: encs.append('')
#         else:
#             p = np.concatenate([[0], p, [0]])
#             runs = np.where(p[1:] != p[:-1])[0] + 1
#             runs[1::2] -= runs[::2]
#             encs.append(' '.join(str(x) for x in runs))
#     return encs


# In[ ]:


# stats = ([0.400,0.402,0.404], [0.178,0.181,0.175])
# #check https://www.kaggle.com/iafoss/256x256-images-with-defects for stats

# data = (SegmentationItemList.from_folder(TEST)
#         .split_by_idx([0])
#         .label_from_func(lambda x : str(x), classes=[0,1,2,3,4])
#         .add_test(Path(TEST).ls(), label=None)
#         .databunch(path=Path('.'), bs=bs)
#         .normalize(stats))


# In[ ]:


# rles,ids_test = [],[]
# learns = []
# for fold in range(nfolds):
#     learn = unet_learner(data, models.resnet34, pretrained=False)
#     learn.model.load_state_dict(torch.load(Path(BASE)/f'models/fold{fold}.pth')['model'])
#     learns.append(learn)

# with zipfile.ZipFile('pred.zip', 'w') as archive_out:
#     def to_mask(yp, y, id):
#         name = id + '.png'
#         save_img(yp[:,:,1:],name,archive_out)
#         yp = np.argmax(yp, axis=-1)
#         for i in range(n_cls):
#             idxs = yp == i+1
#             if idxs.sum() < noise_th: yp[idxs] = 0
#         encs = mask2enc(yp)
#         for i, enc in enumerate(encs):
#             ids_test.append(id + '.jpg_' + str(i+1))
#             rles.append(enc)
    
#     model_pred(learns,to_mask,DatasetType.Test)
    
# sub_df = pd.DataFrame({'ImageId_ClassId': ids_test, 'EncodedPixels': rles})
# sub_df.sort_values(by='ImageId_ClassId').to_csv('submission.csv', index=False)


# In[ ]:


import os
import cv2
import torch
import pandas as pd
import numpy as np
import glob
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Resize, Compose)
#from albumentations.torch import ToTensor
from albumentations.pytorch.transforms import ToTensor
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


# In[ ]:


class SteelDataset(Dataset):
    def __init__(self, df, augment=None):

        
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        

    def __len__(self):
        return len(self.fnames)


    def __getitem__(self, index):
        image_id = self.fnames[index]
        image = cv2.imread(test_data_folder + '/%s'%(image_id), cv2.IMREAD_COLOR)
        return image_id, image
    


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"


# In[ ]:


def null_collate(batch):
    batch_size = len(batch)

    input = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][1])
        infor.append(batch[b][0])

    input = np.stack(input).astype(np.float32)/255
    input = input.transpose(0,3,1,2)
    
    input = torch.from_numpy(input).float()
    
    return infor, input


# In[ ]:


df = pd.read_csv(sample_submission_path)
test_dataset = SteelDataset(df)

test_loader = DataLoader(
            test_dataset,
            batch_size  = 2,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
    )


# In[ ]:


#test time augmentation  -----------------------
def null_augment   (input): return input
def flip_lr_augment(input): return torch.flip(input, dims=[2])
def flip_ud_augment(input): return torch.flip(input, dims=[3])

def null_inverse_augment   (logit): return logit
def flip_lr_inverse_augment(logit): return torch.flip(logit, dims=[2])
def flip_ud_inverse_augment(logit): return torch.flip(logit, dims=[3])

augment = (
        (null_augment,   null_inverse_augment   ),
        (flip_lr_augment,flip_lr_inverse_augment),
        (flip_ud_augment,flip_ud_inverse_augment),
    )


# In[ ]:


TEMPERATE=0.5
######################################################################################
def probability_mask_to_probability_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 5)
    value, index = probability.max(1)
    probability = value[:,1:]
    return probability


def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b,c], min_size[c])
    return predict


# In[ ]:


def do_evaluate_segmentation(net, test_loader, augment=[]):

    #----

    #def sharpen(p,t=0):
    def sharpen(p,t=TEMPERATE):
        if t!=0:
            return p**t
        else:
            return p


    test_num  = 0
    test_id   = []
    #test_image = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = []
    test_truth_mask  = []

    #start = timer()
    for t, (fnames, input) in enumerate(tqdm(test_loader)):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  net(input)
                probability = torch.softmax(logit,1)

                probability_mask = sharpen(probability,0)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = net(torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = net(torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask += sharpen(probability)
                num_augment+=1

            #---
            probability_mask = probability_mask/num_augment
            probability_label = probability_mask_to_probability_label(probability_mask)

        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)
        probability_label = (probability_label.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i for i in fnames])

        test_probability_mask.append(probability_mask)
        test_probability_label.append(probability_label)
        
    test_probability_mask = np.concatenate(test_probability_mask)
    test_probability_label = np.concatenate(test_probability_label)
    
    
    return test_probability_label, test_probability_mask, test_id


# In[ ]:


get_ipython().system('ls ../input/henge5')


# In[ ]:


ckpt_file = '../input/henge5/trace_model_swa.pth'
net = torch.jit.load(ckpt_file).cuda()


# In[ ]:


probability_label, probability_mask, image_id = do_evaluate_segmentation(net, test_loader, augment=['null'])


# In[ ]:


del net
gc.collect()


# In[ ]:


#value = probability_mask*(value==probability_mask)
probability_mask = probability_mask[:,1:] #remove background class


# In[ ]:


threshold_label      = [ 0.70, 0.8, 0.50, 0.70,]
threshold_mask_pixel = [ 0.6, 0.8, 0.5, 0.6,]
threshold_mask_size  = [ 1,  1,  1,  1,]


# In[ ]:


predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)


# In[ ]:


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


image_id_class_id = []
encoded_pixel = []
for b in range(len(image_id)):
    for c in range(4):
        image_id_class_id.append(image_id[b]+'_%d'%(c+1))

        if predict_label[b,c]==0:
            rle=''
        else:
            rle = mask2rle(predict_mask[b,c])
        encoded_pixel.append(rle)


# In[ ]:


df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv('submission.csv', index=False)


# In[ ]:


df.head(50)


# In[ ]:


def summarise_submission_csv(df):


    text = ''
    df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
    df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
    num_image = len(df)//4
    num = len(df)

    pos = (df['Label']==1).sum()
    neg = num-pos


    pos1 = ((df['Class']==1) & (df['Label']==1)).sum()
    pos2 = ((df['Class']==2) & (df['Label']==1)).sum()
    pos3 = ((df['Class']==3) & (df['Label']==1)).sum()
    pos4 = ((df['Class']==4) & (df['Label']==1)).sum()

    neg1 = num_image-pos1
    neg2 = num_image-pos2
    neg3 = num_image-pos3
    neg4 = num_image-pos4


    text += 'compare with LB probing ... \n'
    text += '\t\tnum_image = %5d(1801) \n'%num_image
    text += '\t\tnum  = %5d(7204) \n'%num
    text += '\n'

    text += '\t\tpos1 = %5d( 128)  %0.3f\n'%(pos1,pos1/128)
    text += '\t\tpos2 = %5d(  43)  %0.3f\n'%(pos2,pos2/43)
    text += '\t\tpos3 = %5d( 741)  %0.3f\n'%(pos3,pos3/741)
    text += '\t\tpos4 = %5d( 120)  %0.3f\n'%(pos4,pos4/120)
    text += '\n'

    text += '\t\tneg1 = %5d(1673)  %0.3f  %3d\n'%(neg1,neg1/1673, neg1-1673)
    text += '\t\tneg2 = %5d(1758)  %0.3f  %3d\n'%(neg2,neg2/1758, neg2-1758)
    text += '\t\tneg3 = %5d(1060)  %0.3f  %3d\n'%(neg3,neg3/1060, neg3-1060)
    text += '\t\tneg4 = %5d(1681)  %0.3f  %3d\n'%(neg4,neg4/1681, neg4-1681)
    text += '--------------------------------------------------\n'
    text += '\t\tneg  = %5d(6172)  %0.3f  %3d \n'%(neg,neg/6172, neg-6172)
    text += '\n'

    if 1:
        #compare with reference
        pass

    return text


# In[ ]:


text = summarise_submission_csv(df)
print(text)


# In[ ]:


def rle2mask(mask_rle, shape=(1600,256)):
   '''
   mask_rle: run-length as string formated (start length)
   shape: (width,height) of array to return 
   Returns numpy array, 1 - mask, 0 - background

   '''
   s = mask_rle.split()
   starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
   starts -= 1
   ends = starts + lengths
   img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
   for lo, hi in zip(starts, ends):
       img[lo:hi] = 1
   return img.reshape(shape).T


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('submission.csv')[:60]
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])

for row in df.itertuples():
    img_path = os.path.join(test_data_folder, row.Image)
    img = cv2.imread(img_path)
    mask = rle2mask(row.EncodedPixels, (1600, 256))         if isinstance(row.EncodedPixels, str) else np.zeros((256, 1600))
    if mask.sum() == 0:
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 60))
    axes[0].imshow(img/255)
    axes[1].imshow(mask*60)
    axes[0].set_title(row.Image)
    axes[1].set_title(row.Class)
    plt.show()


# In[ ]:




