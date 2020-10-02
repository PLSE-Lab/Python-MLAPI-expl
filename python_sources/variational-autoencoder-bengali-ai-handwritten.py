#!/usr/bin/env python
# coding: utf-8

# This code is a fork of the [ignite/VAE.ipynb](https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb) (pytorch-ignite)  guide for training a variational autoencoder on the MNIST dataset. That code is primarily based on the [official PyTorch example](https://github.com/pytorch/examples/tree/master/vae). 
# 
# Their goal was to replicate the goal as to replicate  [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma and Welling.  This paper uses an encoder-decoder architecture to encode images to a vector and then reconstruct the images. Here I am going to run a similar experiment with BengaliAi handwritten characters. The ignite event handlers make debugging training bottle necks easier and the visualisation of the relative difficulty of resolving the finer details of graphemes.
#  
# It must be noted that I **have not** used the target classes (grapheme_root, consonant_diacritic, vowel_diacritic). For those looking for more information about how variational encoders work check out this great guide posted by Louis Tia: 
# 
# https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/. 
# 
# > Like all autoencoders, the variational autoencoder is primarily used for unsupervised learning of hidden representations

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import datetime as dt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from keras.preprocessing import image

import PIL

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

import cv2
from torchvision import datasets, transforms
import torchvision
from torchvision.utils import save_image, make_grid
# from torchvision.transforms import Compose, ToTensor
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Loss, RunningAverage


# In[ ]:


import torch
import random
from sklearn.model_selection import KFold


# # Load Raw Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "img_data = pd.read_parquet('../input/bengaliai-cv19/train_image_data_0.parquet').set_index('image_id')")


# # Image Processing Functions
# 
# Goal here was to crop out as much background as possible and down sizing image to match the MNIST data (28x28 pixels)

# In[ ]:


def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# In[ ]:


HEIGHT = 137
WIDTH = 236
#
# aiming for the same size as the MNIST data set
IMGSIZE=28
def process_image(img_):
#     
    img = img_[5:-5,3:-7]
    img_inv = 255-img
    img_invb = cv2.GaussianBlur(img_inv, (7,7) , 0)
    img_inv = cv2.addWeighted(img_inv, 1.0 + 4.0, img_invb, -4.0, 0) # im1 = im + 4.0*(im - im_blurred)

    mask = cv2.threshold( img_inv, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    img_inv=mask
    img_inv = img_inv[np.ix_(mask.any(1), mask.any(0))]

    kernel = np.ones((1,1),np.uint8)
    close = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(close,kernel,iterations = 2)
    aug = transforms.Compose([transforms.ToPILImage(),
                              expand2square,
                              transforms.Pad(5),
                              transforms.Resize((int(IMGSIZE),int(IMGSIZE)),PIL.Image.ANTIALIAS),
                              transforms.ToTensor(),
                             ])
    image = aug(dilation)
    return image


# ### torchvision transforms:
# * **ToPILImage** convert numpy array to PIL Image.
# * **Pad** avoid clipping after futher augmentation.
# * **Resize** to resize the image to target size.
# * **ToTensor** convert to PIL Image to tensor.
# 
# The high scale factor makes some feature impreceptible so this function also implements some morphological transforms(dialation) to increase line thickness and back ground cropping and thresholding in using cv2.

# In[ ]:


from tqdm import tqdm_notebook
tqdm_notebook().pandas()
image_tensor =torch.stack(img_data.progress_apply(lambda x:process_image(x.values.reshape(HEIGHT,WIDTH)),axis=1).values.tolist())


# # LABELS
# 
# These are not used in this notebook(we will be looking at the reconstruction loss and KL divergence) but are loaded into the dataset / dataloader and required by ignite during training and inference.

# In[ ]:


LABELS = '../input/bengaliai-cv19/train.csv'
df = pd.read_csv(LABELS).set_index('image_id')
nunique = list(df.nunique())[1:-1]
print(nunique)
df.head()


# # dataframe to tensors to train/val split pytorch dataset

# In[ ]:


label_tensor= torch.from_numpy( df.loc[img_data.index,['grapheme_root','vowel_diacritic','consonant_diacritic']].values)
dataset=torch.utils.data.TensorDataset(image_tensor,label_tensor)
train_data,val_data = torch.utils.data.random_split(dataset,lengths=[4*len(dataset)//5,len(dataset)//5])

print ('len(train_data) : ', len(train_data))
print ('len(val_data) : ', len(val_data))
print ('image.shape : ', image_tensor.shape)
print ('label.item() : ', label_tensor[0])


# In[ ]:


test_image = make_grid(image_tensor[:5].detach().cpu(), nrow=5)
plt.figure(figsize=(25, 10));
plt.imshow(test_image.permute(1, 2, 0));


# These downsampled images are missing finer detail but will be sufficient for our purposes.

# # Load DataLoader

# In[ ]:


SEED = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 40
INIT_LR = 5*1e-3
BS = int(128*6)

# CUDA memory
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BS, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BS, shuffle=True, **kwargs)

for batch in train_loader:
    x, y = batch
    break

print ('x.shape : ', x.shape)
print ('y.shape : ', y.shape)
fixed_images = x.to(device)


# In[ ]:


HEIGHT = 137
WIDTH = 236

class VAE(nn.Module):
    def __init__(self,input_size=IMGSIZE*IMGSIZE):
        self.input_size=input_size
        super(VAE, self).__init__()
        
        fc1out= int(2*input_size/3)
        embeddim =int(np.sqrt(fc1out)) 
        
#         input layer
        self.fc1 = nn.Linear(input_size, fc1out)

#         embedded vector    
        self.fc21 = nn.Linear(fc1out, embeddim)
        self.fc22 = nn.Linear(fc1out, embeddim)
        self.fc3 = nn.Linear(embeddim, fc1out)
        
#         output layer
        self.fc4 = nn.Linear(fc1out,input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kld_loss(x_pred, x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

bce_loss = nn.BCELoss(reduction='sum')


# In[ ]:


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters for {}: {:.2e}".format(model.__class__,total_num_params))
    


model = torchvision.models.resnext50_32x4d(pretrained=False)
print_num_params(model)
model = torchvision.models.resnet50(pretrained=False)
print_num_params(model)


# In[ ]:


model = VAE(IMGSIZE*IMGSIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=6e-3)
print_num_params(model)
model


# In[ ]:


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, np.prod(fixed_images.shape[-2:])), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# In[ ]:



def process_function(engine, batch):
    model.train()
    # clear the gradients from previous run
    optimizer.zero_grad()
    x, _ = batch
    x = x.to(device)
    x = x.view(-1,  np.prod(fixed_images.shape[-2:]))
    # get predictions    
    x_pred, mu, logvar = model(x)
    # calculate loss     
    BCE = bce_loss(x_pred, x)
    KLD = kld_loss(x_pred, x, mu, logvar)
    loss = BCE + KLD
    # apply gradients     
    loss.backward()
    optimizer.step()
    return loss.item(), BCE.item(), KLD.item()


# In[ ]:


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, _ = batch
        x = x.to(device)
        x = x.view(-1,  np.prod(fixed_images.shape[-2:]))
        x_pred, mu, logvar = model(x)
        kwargs = {'mu': mu, 'logvar': logvar}
        return x_pred, x, kwargs


# In[ ]:





# In[ ]:


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
training_history = {'bce': [], 'kld': [], 'mse': []}
validation_history = {'bce': [], 'kld': [], 'mse': []}


# In[ ]:



RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'bce')
RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'kld')


# In[ ]:



MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'mse')
Loss(bce_loss, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'bce')
Loss(kld_loss).attach(evaluator, 'kld')


# In[ ]:



def print_logs(engine, dataloader, mode, history_dict):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_mse = metrics['mse']
    avg_bce = metrics['bce']
    avg_kld = metrics['kld']
    avg_loss =  avg_bce + avg_kld
    print(
        mode + " Results - Epoch {} - Avg mse: {:.2f} Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
        .format(engine.state.epoch, avg_mse, avg_loss, avg_bce, avg_kld))
    for key in evaluator.state.metrics.keys():
        history_dict[key].append(evaluator.state.metrics[key])

trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training', training_history)
trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, val_loader, 'Validation', validation_history)


# In[ ]:



def compare_images(engine, save_img=False):
    epoch = engine.state.epoch
    reconstructed_images = model(fixed_images.view(-1, np.prod(fixed_images.shape[-2:])))[0].view(-1,1,*tuple(fixed_images.shape[-2:]))
    comparison = torch.cat([fixed_images, reconstructed_images])
    if save_img:
        save_image(comparison.detach().cpu(), 'reconstructed_epoch_' + str(epoch) + '.png', nrow=32)
    comparison_image = make_grid(comparison.detach().cpu(), nrow=32)
    fig = plt.figure(figsize=(25, 10));
    output = plt.imshow(comparison_image.permute(1, 2, 0));
    plt.title('Epoch ' + str(epoch));
    plt.show();
    
trainer.add_event_handler(Events.STARTED(every=3), compare_images, save_img=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=8), compare_images, save_img=False)


# This is an example of what the network starts with.

# In[ ]:



reconstructed_images = model(fixed_images.view(-1, np.prod(fixed_images.shape[-2:])))[0].view(-1,1,*tuple(fixed_images.shape[-2:]))
comparison = torch.cat([fixed_images, reconstructed_images])
comparison_image = make_grid(comparison.detach().cpu(), nrow=32)
fig = plt.figure(figsize=(25, 18));
output = plt.imshow(comparison_image.permute(1, 2, 0));
plt.title('Epoch ' + str(-1));
plt.show();


# # let the training begin!

# In[ ]:


BS,EPOCHS


# In[ ]:


e = trainer.run(train_loader, max_epochs=EPOCHS)


# In[ ]:


plt.plot(range(EPOCHS), training_history['bce'], 'dodgerblue', label='training')
plt.plot(range(EPOCHS), validation_history['bce'], 'orange', label='validation')
plt.xlim(0, EPOCHS);
plt.xlabel('Epoch')
plt.ylabel('BCE')
plt.title('Binary Cross Entropy on Training/Validation Set')
plt.legend();


# In[ ]:


plt.plot(range(EPOCHS), training_history['kld'], 'dodgerblue', label='training')
plt.plot(range(EPOCHS), validation_history['kld'], 'orange', label='validation')
plt.xlim(0, EPOCHS);
plt.xlabel('Epoch')
plt.ylabel('KLD')
plt.title('KL Divergence on Training/Validation Set')
plt.legend();


# In[ ]:




