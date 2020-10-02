#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import importlib

sys.path.append('../input')
sys.modules['efficientnet_pytorch'] = importlib.import_module('efficientnet-pytorch')
sys.modules['rectified_adam'] = importlib.import_module('rectified-adam')


# In[ ]:


from pathlib import Path
import pandas as pd
from fastai.vision import *
from efficientnet_pytorch import EfficientNet
from rectified_adam.radam import *
from functools import partial
import io


# In[ ]:


data_dir = Path('../input/aptos2019-blindness-detection')
batch_size = 8
im_size = (528,528)

train = True
ssl = True
c = 1/3
model_dir = Path('../input/my-aptos2019-blindness-detection')


# In[ ]:


def preprocess(im):
    bbox = pil2tensor(im, dtype=np.uint8).nonzero().transpose(1,0)
    im = im.crop((bbox[2].min().item(), 
                    bbox[1].min().item(), 
                    bbox[2].max().item(), 
                    bbox[1].max().item()))

    return im
    
if train:
    df = pd.read_csv(data_dir/'train.csv')
    counts = df['diagnosis'].value_counts(normalize=True)
    w = 1 - tensor([counts[i] for i in range(5)])
    w = w.to(defaults.device)

    L = len(df)
    if ssl: df = df.sample(frac=1 - c)
    src = ImageList.from_df(df, data_dir/'train_images', suffix='.png', after_open=preprocess)
    if ssl:
        df = pd.read_csv(model_dir/'test.csv').sample(n=int(c * L))
        src.add(ImageList.from_df(df, data_dir/'test_images', suffix='.png', after_open=preprocess))

    data = src.split_none()        .label_from_df(classes=[0,1,2,3,4])        .transform(
            get_transforms(max_warp=None, 
                xtra_tfms=[cutout(length=(im_size[0]//8,im_size[0]//4))]), 
            size=im_size, 
            resize_method=ResizeMethod.SQUISH, 
            padding_mode='zeros')\
        .databunch(bs=batch_size)\
        .normalize(stats=imagenet_stats)


# In[ ]:


model_name = 'efficientnet-b6'
if train:
    model = EfficientNet.from_pretrained(model_name, num_classes=5)
    kappa = KappaScore()
    kappa.weights = 'quadratic'

    learn = Learner(data, model, 
        loss_func=partial(F.cross_entropy, weight=w), opt_func=RAdam, metrics=[kappa]).to_fp16()
    learn.layer_groups = split_model_idx(learn.model, idxs=[-1])
    learn.summary()
else:
    learn = load_learner(model_dir, model_name + '.pkl')
learn.path = Path('.')

if ssl:
    with io.open(model_dir/f'{model_name}.pth', 'rb') as f:
        learn = learn.load(f)


# In[ ]:


if train:
    if not ssl:
        learn.freeze()
        learn.fit(1, lr=1e-3)

    nepochs = 12
    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=nepochs, max_lr=1e-4, div_factor=10, pct_start=0)

    learn.export(model_name + '.pkl')
    learn.save(model_name)
else:
    learn.model.eval()
    mean = tensor(imagenet_stats[0])
    std = tensor(imagenet_stats[1])

    def diagnose(row):
        fn = row['id_code']
        im = open_image(data_dir/f'test_images/{fn}.png', div=True, after_open=preprocess)
        im = im.apply_tfms(None, size=im_size, resize_method=ResizeMethod.PAD, padding_mode='zeros')

        #tta
        x = torch.stack([
            im.data, 
            flip_lr(im).data, 
            im.zoom(scale=1.25).data
        ])
        x = (x - mean[None,:,None,None]) / std[None,:,None,None]
        x = x.to(defaults.device).half()
        pred = learn.model(x).mean(dim=0)
        pred = np.argmax(pred)

        return pred.item()
    
    df = pd.read_csv(data_dir/'test.csv')
    df['diagnosis'] = df.apply(diagnose, axis=1)
    df.to_csv('submission.csv', index=False, line_terminator='\n')

