#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install fastai2')


# In[ ]:


from fastai2.vision.all import *


# ## EDA

# In[ ]:


data = untar_data(URLs.CAMVID)


# In[ ]:


valid_fnames = (data/'valid.txt').read().split('\n')
valid_fnames


# In[ ]:


data.ls()


# In[ ]:


codes = (data/'codes.txt').read().split('\n')
codes


# In[ ]:


path_imgs = data/'images'
path_labels = data/'labels'


# In[ ]:


fnames = get_image_files(path_imgs)
lbl_names = get_image_files(path_labels)


# In[ ]:


lbl_names


# In[ ]:


img_fn = fnames[5]
img = PILImage.create(img_fn)
img.show()


# In[ ]:


get_mask = lambda o: data/'labels'/f'{o.stem}_P{o.suffix}'
mask = PILMask.create(get_mask(img_fn))
mask.show(alpha=1)


# In[ ]:


codes = np.loadtxt(data/'codes.txt', dtype=str)
codes


# ## Dataloaders

# In[ ]:


def FileSplitter(fname):
    valid = Path(fname).read().split('\n')
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner


# In[ ]:


size = mask.shape
size


# In[ ]:


half = tuple(int(x/2) for x in size)

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                  get_items= get_image_files,
                  get_y=get_mask,
                   splitter=FileSplitter(data/'valid.txt'),
                   batch_tfms = [*aug_transforms(size=half), Normalize.from_stats(*imagenet_stats)]
                  )


# In[ ]:


dls = camvid.dataloaders(data/'images', bs=8)


# In[ ]:


dls.show_batch(max_n=4)


# In[ ]:


dls.vocab = codes


# In[ ]:


name2id = {v:k for k,v in enumerate(codes)}
name2id
void_code=name2id['Void']


# In[ ]:


def acc_camvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != void_code
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()


# ## Segmentation model

# In[ ]:


config = unet_config(self_attention=True, act_cls=Mish)
opt = ranger

learn = unet_learner(dls, resnet18, metrics=acc_camvid, config=config, opt_func=opt)


# In[ ]:


learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


lr = 1e-3

learn.fit_flat_cos(3,slice(lr))
learn.save('stage-1')
learn.load('stage-1')


# In[ ]:


learn.show_results(max_n=4)


# In[ ]:


lrs = slice(lr/400,lr/4)
lr,lrs

learn.fit_flat_cos(5,lrs)


# In[ ]:


learn.save('model_1')
learn.show_results()


# ## INference

# In[ ]:


dl = learn.dls.test_dl(fnames[:5])
dl.show_batch()


# In[ ]:


preds = learn.get_preds(dl=dl)


# In[ ]:


pred_1 = preds[0][0]
pred_1.shape

pred_arx = pred_1.argmax(dim=0)
plt.imshow(pred_arx)


# In[ ]:


for i, pred in enumerate(preds[0]):
  pred_arg = pred.argmax(dim=0).numpy()
  rescaled = (255.0 / pred_arg.max() * (pred_arg - pred_arg.min())).astype(np.uint8)
  im = Image.fromarray(rescaled)
  im.save(f'Image_{i}.png')
    
####


# ## Weighted average

# In[ ]:


class CrossEntropyLossFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


# In[ ]:


weights = torch.tensor([[0.9]*31 + [1.1]]).cuda()
weights


# In[ ]:


# learn.loss_func = CrossEntropyLossFlat(weights=weights, axis=1)
loss_func = CrossEntropyLossFlat(weight=weights, axis=1)
learn = unet_learner(dls, resnet34, metrics=acc_camvid, loss_func=loss_func)


# In[ ]:


learn.fit_flat_cos(12,lrs)


# In[ ]:




