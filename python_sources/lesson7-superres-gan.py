#!/usr/bin/env python
# coding: utf-8

# ## Pretrained GAN

# In[ ]:


import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *


# In[ ]:


path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'crappy'


# ## Crappified data

# Prepare the input data by crappifying images.

# In[ ]:


from PIL import Image, ImageDraw, ImageFont
def crappify(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 96, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    w,h = img.size
    q = random.randint(10,70)
    ImageDraw.Draw(img).text((random.randint(0,w//2),random.randint(0,h//2)), str(q), fill=(255,255,255))
    img.save(dest, quality=q)


# Uncomment the first time you run this notebook.

# In[ ]:


try:
    il = ItemList.from_folder(path_hr)
    parallel(crappify, il.items)
except:
    print("Invalid Image Type")


# For gradual resizing we can change the commented line here.

# In[ ]:


bs,size=32, 128
# bs,size = 24,160
#bs,size = 8,256
arch = models.resnet34


# ## Pre-train generator

# Now let's pretrain the generator.

# In[ ]:


arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


# In[ ]:


def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


# In[ ]:


data_gen = get_data(bs,size)


# In[ ]:


data_gen.show_batch(4)


# In[ ]:


wd = 1e-3


# In[ ]:


y_range = (-3.,3.)


# In[ ]:


loss_gen = MSELossFlat()


# In[ ]:


def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)


# In[ ]:


learn_gen = create_gen_learner()


# In[ ]:


learn_gen.fit_one_cycle(1, pct_start=0.8)


# In[ ]:


learn_gen.unfreeze()


# In[ ]:


learn_gen.fit_one_cycle(1, slice(1e-6,1e-3))


# In[ ]:


learn_gen.show_results(rows=4)


# In[ ]:


learn_gen.save('gen-pre2')


# ## Save generated images

# In[ ]:


learn_gen.load('gen-pre2');


# In[ ]:


name_gen = 'image_gen'
path_gen = path/name_gen


# In[ ]:


# shutil.rmtree(path_gen)


# In[ ]:


path_gen.mkdir(exist_ok=True)


# In[ ]:


def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1


# In[ ]:


save_preds(data_gen.fix_dl)


# In[ ]:


PIL.Image.open(path_gen.ls()[0])


# ## Train critic

# In[ ]:


learn_gen=None
gc.collect()


# Pretrain the critic on crappy vs not crappy.

# In[ ]:


def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    print(ll)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data


# In[ ]:


data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)


# In[ ]:


data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)


# In[ ]:


loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


# In[ ]:


def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)


# In[ ]:


learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)


# In[ ]:


try:
    learn_critic.fit_one_cycle(2, 1e-3)
except:
    print("Error")


# In[ ]:


learn_critic.save('critic-pre2')


# ## GAN

# Now we'll combine those pretrained model in a GAN.

# In[ ]:


learn_crit=None
learn_gen=None
gc.collect()


# In[ ]:


data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)


# In[ ]:


learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')


# In[ ]:


learn_gen = create_gen_learner().load('gen-pre2')


# To define a GAN Learner, we just have to specify the learner objects foor the generator and the critic. The switcher is a callback that decides when to switch from discriminator to generator and vice versa. Here we do as many iterations of the discriminator as needed to get its loss back < 0.5 then one iteration of the generator.
# 
# The loss of the critic is given by `learn_crit.loss_func`. We take the average of this loss function on the batch of real predictions (target 1) and the batch of fake predicitions (target 0). 
# 
# The loss of the generator is weighted sum (weights in `weights_gen`) of `learn_crit.loss_func` on the batch of fake (passed throught the critic to become predictions) with a target of 1, and the `learn_gen.loss_func` applied to the output (batch of fake) and the target (corresponding batch of superres images).

# In[ ]:


switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))


# In[ ]:


lr = 1e-4


# In[ ]:


learn.fit(2,lr)


# In[ ]:


learn.save('gan-1c')


# In[ ]:


learn.data=get_data(16,192)


# In[ ]:


learn.fit(2,lr/2)


# In[ ]:


learn.show_results(rows=16)


# In[ ]:


learn.save('gan-1c')


# ## fin

# In[ ]:




