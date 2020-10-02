#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.layers import *
from fastai.vision.gan import *
from fastai.vision.learner import *


# # Out code

# In[ ]:


cd /kaggle/usr/lib/ugatit_network/


# In[ ]:


from ugatit_network import *


# In[ ]:


cd /kaggle/usr/lib/outcode/


# In[ ]:


import outcode as out


# In[ ]:


cd /kaggle/usr/lib/cyclegan/


# In[ ]:


import cyclegan as cy


# # Data

# In[ ]:


path  = Path('/kaggle/input/selfie2anime/selfie2anime')
path2 = Path('/kaggle/')


# In[ ]:


get_ipython().system("mkdir '/kaggle/trainA'")
get_ipython().system("mkdir '/kaggle/trainB'")
out.resize(path/'trainA',path2/"trainA",limit=10)
out.resize(path/'trainB',path2/"trainB",limit=10)


# In[ ]:


ls '/kaggle/input/selfie2anime/selfie2anime'


# In[ ]:


data = (cy.ImageTupleList.from_folders(path, 'trainA', 'trainB')
                      .split_none()
                      .label_empty()
                      .transform(get_transforms(), size=128)
                      .databunch(bs=2))


# In[ ]:


data.show_batch(rows=5)


# # Loss

# In[ ]:


def L1LossFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.MSELoss`, but flattens input and target."
    return FlattenedLoss(nn.L1Loss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)

class UGATIT_Dsc_Loss(nn.Module):
    def __init__(self,adv_weight:float=1, cycle_weight:float=10,identity_weight:float=100,cam_weight:float=1000):
        super().__init__() 
        self.adv_weight,self.cycle_weight,self.identity_weight,self.cam_weight = adv_weight,cycle_weight,identity_weight,cam_weight
        self.MSE_loss = MSELossFlat()
        
    def set_input(self, input):
        self.real_A,self.real_B = input
        
    def forward(self, output,target=None) :
        real_GA_logit, real_GA_cam_logit,real_LA_logit, real_LA_cam_logit, real_GB_logit, real_GB_cam_logit, real_LB_logit, real_LB_cam_logit, fake_GA_logit, fake_GA_cam_logit,fake_LA_logit, fake_LA_cam_logit, fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit = output
        
        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))
        
        D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        self.Discriminator_loss = D_loss_A + D_loss_B
        
        return self.Discriminator_loss
        

class UGATIT_Loss(nn.Module):
    def __init__(self, cgan:nn.Module,adv_weight:float=1, cycle_weight:float=10,identity_weight:float=10,cam_weight:float=1000):
        super().__init__() 
        self.cgan,self.adv_weight,self.cycle_weight,self.identity_weight,self.cam_weight = cgan,adv_weight,cycle_weight,identity_weight,cam_weight
        self.crit     = UGATIT_Dsc_Loss(adv_weight,cycle_weight,identity_weight,cam_weight)
        self.MSE_loss = MSELossFlat()
        self.BCE_loss = BCEWithLogitsFlat()
        self.L1_loss  = L1LossFlat()
        
    def set_input(self, input):
        self.real_A,self.real_B = input
        self.crit.set_input(input)
    
    def forward(self, output, target=None):   
        fake_A2B_cam_logit,fake_B2A_cam_logit,fake_A2B2A,fake_B2A2B,fake_A2A, fake_A2A_cam_logit,fake_B2B, fake_B2B_cam_logit,fake_GA_logit, fake_GA_cam_logit,fake_LA_logit, fake_LA_cam_logit, fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit = output
        
        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit))

        G_recon_loss_A = self.L1_loss(fake_A2B2A, self.real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, self.real_B)

        G_identity_loss_A = self.L1_loss(fake_A2A, self.real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, self.real_B)

        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit))
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit))

        G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
        G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

        self.Generator_loss = G_loss_A + G_loss_B
        
        return self.Generator_loss


# # UGATIT

# ## Module

# In[ ]:


from scipy import misc
import os, cv2, torch
import numpy as np
import sys

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# In[ ]:


class UGATIT_GAN(nn.Module):
    def __init__(self,ch = 64,n_res=4,img_size=256,light=False):
        super().__init__()
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res, img_size=img_size, light=light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res, img_size=img_size, light=light)
        self.disGA  = Discriminator(input_nc=3, ndf=ch, n_layers=7)
        self.disGB  = Discriminator(input_nc=3, ndf=ch, n_layers=7)
        self.disLA  = Discriminator(input_nc=3, ndf=ch, n_layers=5)
        self.disLB  = Discriminator(input_nc=3, ndf=ch, n_layers=5)

    def forward(self, real_A, real_B):        
        if not self.training : 
            fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap  = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap  = self.genB2A(real_B)
        
            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
            
            return torch.cat([fake_A2B[:,None],fake_B2A[:,None]], 1)
        
        fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap  = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap  = self.genB2A(real_B)
        
        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
        
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        
        return [fake_A2B_cam_logit,fake_B2A_cam_logit,fake_A2B2A,fake_B2A2B,fake_A2A, fake_A2A_cam_logit,fake_B2B, fake_B2B_cam_logit,fake_GA_logit, fake_GA_cam_logit,fake_LA_logit, fake_LA_cam_logit, fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit]


# In[ ]:


class UGATIT_GANTrainer(LearnerCallback):
    _order=-20
    
    def on_train_begin(self, **kwargs):
        self.genA2B, self.genB2A = self.learn.model.genA2B,self.learn.model.genB2A
        self.disGA, self.disGB   = self.learn.model.disGA,self.learn.model.disGB
        self.disLA, self.disLB   = self.learn.model.disLA,self.learn.model.disLB
        
        self.Rho_clipper = RhoClipper(0, 1)
        self.crit        = self.learn.loss_func.crit
        
        self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.genA2B), *flatten_model(self.genB2A))])
        self.opt_D = self.learn.opt.new([nn.Sequential(*flatten_model(self.disGA), *flatten_model(self.disGB),*flatten_model(self.disLA), *flatten_model(self.disLB))])
        
        self.learn.opt.opt = self.opt_G.opt
        self.names = ['Discriminator_loss', 'Generator_loss']
        self.learn.recorder.no_val=True
        self.learn.recorder.add_metric_names(self.names)
        self.smootheners = {n:SmoothenValue(0.98) for n in self.names}
        self.last_gen = [0,0,1]
        self.images,self.titles = [],[]
        self.opt_G.zero_grad()
    
    def on_batch_begin(self, last_input, **kwargs):
        self.learn.loss_func.set_input(last_input)
        self.last_gen[0] = last_input[0].detach().cpu()
        self.last_gen[1] = last_input[1].detach().cpu()
    
    # def on_backward_begin(self, last_loss, last_output, **kwargs):
                
    def on_batch_end(self, last_input, last_output, **kwargs):
        self.genA2B.apply(self.Rho_clipper)
        self.genB2A.apply(self.Rho_clipper)
        
        self.opt_D.zero_grad()
        
        real_A, real_B = last_input
        
        fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap  = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap  = self.genB2A(real_B)
            
        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        
        loss_D = self.crit([real_GA_logit, real_GA_cam_logit,real_LA_logit, real_LA_cam_logit, real_GB_logit, real_GB_cam_logit, real_LB_logit, real_LB_cam_logit, fake_GA_logit, fake_GA_cam_logit,fake_LA_logit, fake_LA_cam_logit, fake_GB_logit, fake_GB_cam_logit, fake_LB_logit, fake_LB_cam_logit])
        loss_D.backward()
        self.opt_D.step()
        
        self.last_gen[2] = fake_A2B.detach().cpu()
        metrics = [loss_D, self.learn.loss_func.Generator_loss]
        for n,m in zip(self.names,metrics): self.smootheners[n].add_value(m)
    
    def on_epoch_end(self, pbar, epoch, last_metrics, **kwargs):
        img_out = Image(self.last_gen[2][0]/2+0.5)
        self.images.append(img_out)
        self.titles.append(f'Epoch {epoch}')
        pbar.show_imgs(self.images, self.titles)
        learn.opt.lr -= (learn.opt.lr/50)
        return add_metrics(last_metrics, [s.smooth for k,s in self.smootheners.items()])


# ## Train

# In[ ]:


UGATIT_gan = UGATIT_GAN(light=True) #light=True
learn = Learner(data, UGATIT_gan, loss_func=UGATIT_Loss(UGATIT_gan), opt_func=partial(optim.Adam, betas=(0.5,0.99), weight_decay=1e-4, lr=1e-4),
               callback_fns=[UGATIT_GANTrainer])


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(5)


# In[ ]:


learn.show_results(ds_type=DatasetType.Train, rows=2)


# In[ ]:


learn.show_results(ds_type=DatasetType.Train, rows=2)


# In[ ]:


learn.save("/kaggle/working/UGATIT")


# In[ ]:


from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

