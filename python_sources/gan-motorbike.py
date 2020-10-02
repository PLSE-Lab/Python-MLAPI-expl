#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install gdown')
get_ipython().system('pip install torch==1.0.1')


# In[ ]:


get_ipython().system('pwd')
get_ipython().system('gdown https://drive.google.com/uc?id=1WzrREEwOtOqBjcDWlk5PSl4D27jayqm0')
get_ipython().system('gdown https://drive.google.com/uc?id=15nxPJ900Kf2mRKDcCuK-S13YdRYcdNb3')


# In[ ]:


import os

os.chdir("/kaggle/working/")
if not os.path.isdir("checkpoint"):
    get_ipython().system('mkdir checkpoint')
if not os.path.isdir("out_sample"):
    get_ipython().system('mkdir out_sample')
get_ipython().system('cp -rf "../input/10k-motorbike/utils/utils/utils.py" "../working/"')
get_ipython().system('cp -rf "../input/10k-motorbike/utils/utils/torch_utils.py" "../working/"')
# copy our file into the working directory (make sure it has .py suffix)
# copyfile(src = "../input/my_functions.py", dst = "../working/my_functions.py")
dataset_path = "../input/10k-motorbike/10k_mortorbike_dataset/motobike"
utils_path =  "./utils/utils.py"
checkpoint_path = './checkpoint'
restore_path = './'
out_path = "./out_sample"


# In[ ]:


# !ls ../input/10k-motorbike
get_ipython().system('ls')


# In[ ]:


# import sys
# sys.path.append("./utils")
import glob
import os
import warnings

import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# from utils.torch_utils import *
# from utils.utils import *
import random
from utils import *
from torch_utils import *


# In[ ]:


warnings.filterwarnings('ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {'DataLoader': {'batch_size': 64,
                         'shuffle': True},
          'Generator': {'latent_dim': 120,
                        'embed_dim': 32,
                        'ch': 64,
                        'num_classes': 1,
                        'use_attn': True},
          'Discriminator': {'ch': 64,
                            'num_classes': 1,
                            'use_attn': True},
          'sample_latents': {'latent_dim': 120,
                             'num_classes': 1},
          'num_iterations': 50000,
          'decay_start_iteration': 50000,
          'd_steps': 1,
          'lr_G': 2e-4,
          'lr_D': 4e-4,
          'betas': (0.0, 0.999),
          'margin': 1.0,
          'gamma': 0.1,
          'ema': 0.999,
          'seed': 42}


# In[ ]:


def _load_resized_image(file, root_images):
    img = cv2.imread(os.path.join(root_images, file))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = A.Compose([A.Resize(128, 128, interpolation=cv2.INTER_AREA),
                           A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform = A.Compose([A.Resize(64, 64, interpolation=cv2.INTER_AREA)])
        img = transform(image=img)['image']
    except:
        print("This file: {} is error".format(file))

    
    return img


# In[ ]:


seed_everything(config['seed'])
root_images = dataset_path
all_files = os.listdir(root_images)
# all_files = all_files[0:20]

all_images = []
for count, f in enumerate(all_files):
    if count % 1000 == 0: print(count)
    img = _load_resized_image(f, root_images)
    if img is not None:
        all_images.append(img)
        
# all_images = [_load_resized_image(f, root_images) for f in all_files]
all_images = np.array(all_images)
print("here1")
all_labels = np.zeros(len(all_images))
print("here2")
# train_dataiterator = get_dataiterator(all_images, all_labels, config['DataLoader'], device=device)


# ***Configure hyper parameter for training model***

# In[ ]:


train_dataiterator = get_dataiterator(all_images, all_labels, config['DataLoader'], device=device)
netG = Generator(**config['Generator']).to(device, torch.float32)
netD = Discriminator(**config['Discriminator']).to(device, torch.float32)
# Exponential moving average of generator weights works well.

netGE = Generator(**config['Generator']).to(device, torch.float32)
netGE.load_state_dict(netG.state_dict());
optim_G = Adam(params=netG.parameters(), lr=config['lr_G'], betas=config['betas'])
optim_D = Adam(params=netD.parameters(), lr=config['lr_D'], betas=config['betas'])
decay_iter = config['num_iterations'] - config['decay_start_iteration']
if decay_iter > 0:
    lr_lambda_G = lambda x: (max(0, 1 - x / decay_iter))
    lr_lambda_D = lambda x: (max(0, 1 - x / (decay_iter * config['d_steps'])))
    lr_sche_G = LambdaLR(optim_G, lr_lambda=lr_lambda_G)
    lr_sche_D = LambdaLR(optim_D, lr_lambda=lr_lambda_D)

# auxiliary classifier loss.
# this loss weighted by gamma (0.1) is added to adversarial loss.
# coefficient gamma is quite sensitive.

criterion = nn.NLLLoss().to(device, torch.float32)

step = 1

resume = False
if resume:
    state_G = load_checkpoint('{}/D_checkpoint_6000.ckpt'.format(restore_path))
    state_D = load_checkpoint('{}/D_checkpoint_6000.ckpt'.format(restore_path))

    netG.load_state_dict(state_G['model'])
    netD.load_state_dict(state_D['model'])

    optim_G.load_state_dict(state_G['optimizer'])
    optim_D.load_state_dict(state_D['optimizer'])
    step = state_G['iter']
    


# **Start training jobs**

# In[ ]:


# d_steps = config['d_steps'] - step
while True:
    # Discriminator
    for i in range(config['d_steps']):
        for param in netD.parameters():
            param.requires_grad_(True)

        optim_D.zero_grad()
        real_imgs, real_labels = train_dataiterator.__next__()
        # print(real_labels.shape)
        batch_size = real_imgs.size(0)

        latents, fake_labels = sample_latents(batch_size, **config['sample_latents'], device=device)
        # print(latents.shape)
        # print(fake_labels)
        # exit(0)
        fake_imgs = netG(latents, fake_labels).detach()
        preds_real, preds_real_labels = netD(real_imgs, real_labels)
        # print(preds_real_labels.shape)
        # print(real_labels.shape)
        preds_fake, _ = netD(fake_imgs, fake_labels)
        loss_D = calc_advloss_D(preds_real, preds_fake, config['margin'])
        loss_D += config['gamma'] * criterion(preds_real_labels, real_labels)
        loss_D.backward()
        optim_D.step()

        if (decay_iter > 0) and (step > config['decay_start_iteration']):
            lr_sche_D.step()

    # Generator
    for param in netD.parameters():
        param.requires_grad_(False)

    optim_G.zero_grad()

    real_imgs, real_labels = train_dataiterator.__next__()
    batch_size = real_imgs.size(0)

    latents, fake_labels = sample_latents(batch_size, **config['sample_latents'], device=device)
    fake_imgs = netG(latents, fake_labels)

    preds_real, _ = netD(real_imgs, real_labels)
    preds_fake, preds_fake_labels = netD(fake_imgs, fake_labels)

    loss_G = calc_advloss_G(preds_real, preds_fake, config['margin'])
    loss_G += config['gamma'] * criterion(preds_fake_labels, fake_labels)
    loss_G.backward()
    optim_G.step()

    if (decay_iter > 0) and (step > config['decay_start_iteration']):
        lr_sche_G.step()

    # Update Generator Eval
    for param_G, param_GE in zip(netG.parameters(), netGE.parameters()):
        param_GE.data.mul_(config['ema']).add_((1 - config['ema']) * param_G.data)
    for buffer_G, buffer_GE in zip(netG.buffers(), netGE.buffers()):
        buffer_GE.data.mul_(config['ema']).add_((1 - config['ema']) * buffer_G.data)

    # Save checkpoint
#     if step % 3000 == 0:
#         print("iter: {}, lr_G: {}".format(step, get_lr(optim_G)))
#         save_batch(fake_imgs, path=out_path, iter=step)
#         save_checkpoints("checkpoint", model=netG, optimizer=optim_G, iter=step, name_file="G_checkpoint")
#         save_checkpoints("checkpoint", model=netD, optimizer=optim_D, iter=step, name_file="D_checkpoint")
    if step % 3000 == 0:
        print("iter: {}, lr_G: {}".format(step, get_lr(optim_G)))
        save_batch(fake_imgs, path=out_path, iter=step)
        save_checkpoints(checkpoint_path, model=netG, optimizer=optim_G, iter=step, name_file="G_checkpoint")
        save_checkpoints(checkpoint_path, model=netD, optimizer=optim_D, iter=step, name_file="D_checkpoint")

    # stopping
    if step < config['num_iterations']:
        step += 1
    else:
        print('total step: {}'.format(step))
        break


# In[ ]:


get_ipython().system('touch out_sample/file1.txt')
get_ipython().system('ls ')
# create_download_link(filename='out_sample/file1.txt')
from IPython.display import FileLink, FileLinks
# FileLinks('checkpoint')
FileLinks('out_sample')
# FileLink('D_checkpoint_6000.ckpt')


# In[ ]:




