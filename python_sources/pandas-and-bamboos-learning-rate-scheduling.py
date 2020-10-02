#!/usr/bin/env python
# coding: utf-8

# The Kernel uses the EfficientNet B4 to get a LB 0.6. The model is trained with three stages of decremented learning rate.
# I'm yet to add CV Folds and this Kernel runs for only one fold. 
# To add in the Future - 
# 1. 4-Fold CV
# 2. N_tiles
# 3. Try Cyclic learning rates
# 
# Hope this helps the viewer. It is not supposed to be score grabber but a simple Pytorch Implementation for beginners.
# Please Upvote if you like it.

# # Loading the Important Libraries

# In[ ]:


import os, sys, warnings, random, time, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from IPython.display import display
from tqdm import tqdm_notebook as tqdm

import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.functional import F 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
# from efficientnet_pytorch import model as enet

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


pd.plotting.register_matplotlib_converters()
pd.options.display.max_rows=50
pd.options.display.max_columns=100
plt.rcParams.update({'font.size':18})
sns.set_style('darkgrid')
plt.rcParams.update({'font.family':'Humor Sans'})
plt.xkcd();


# # Fixing Config

# In[ ]:


SEED = 69
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
package_path = '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)
from efficientnet_pytorch import model as enet

Progress_Bar = False
DEBUG = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


data_dir = '../input/prostate-cancer-grade-assessment/'
train_img_dir = os.path.join(data_dir, 'train_images')
train_df = pd.read_csv(data_dir+'train.csv')
train_df = train_df.sample(1000).reset_index(drop=True) if DEBUG else train_df

display(train_df.head())
len(train_df)


# # Create Folds

# In[ ]:


skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
train_df['fold'] = -1
for i, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df['isup_grade'])):
    train_df.loc[val_idx, 'fold'] = i
train_df.head()


# In[ ]:


train_df.drop(columns=['data_provider', 'gleason_score'], inplace=True)
train_df.head()


# # Building Dataset

# In[ ]:


class Build_Dataset(Dataset):
    '''Builds Dataset to be fed to Neural Network
       :param df: train_df or test_df
       :param resize: tuple, eg(256, 256)
       :param mode: string train or test 
       :param: augmentations: Image augmentations
    '''
    def __init__(self, df, resize=None, mode='train', augmentations=None):
        self.df = df
        self.resize = resize
        self.mode = mode
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = os.path.join(train_img_dir, self.df['image_id'].values[idx]) + '.tiff'
            image = skimage.io.MultiImage(img_path)
            label = self.df['isup_grade'].values[idx]
            
        if self.mode == 'test':
            img_path = os.path.join(test_img_dir, self.df['image_id'].values[idx]) + '.tiff'
            image = skimage.io.MultiImage(img_path)
            label = -1
            
        if self.resize is not None:
            image = cv2.resize(image[-1], (self.resize[0], self.resize[1]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        image = np.array(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        #pytorch expects CHW instead of HWC
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return image, label
            


# # Plotting some Images

# In[ ]:


def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(np.array(images[i][0]))
        ax.set_title('ISUP: '+str(images[i][1]))
        ax.axis('off')


# In[ ]:


N_IMAGES = 9

train_data = Build_Dataset(train_df, resize=(512, 512), mode='train')
images = [(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]] 
plot_images(images)


# # Processing The Images

# In[ ]:


#Image-net standard mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Defining train and test transforms
train_transforms = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Normalize(mean=mean, std=std, always_apply=True),
    albumentations.pytorch.ToTensorV2(),
])
test_transforms = albumentations.Compose([
    albumentations.Normalize(mean=mean, std=std, always_apply=True),
    albumentations.pytorch.ToTensorV2(),
])


# # Building Model

# In[ ]:


pretrainied_model = {
    'efficientnet-b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',
    'efficientnet-b4': '../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth'
}

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.enet.load_state_dict(torch.load(pretrainied_model[backbone]))
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()
    
    def extract(self, x):
        return self.enet(x)
    
    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


# model = enetv2('efficientnet-b0', 5).to(device)
# loss_criterion = nn.CrossEntropyLoss().to(device)
# optimizer=optim.Adam(model.parameters())

# print(f'The model has {count_parameters(model):,} trainable parameters')


# # Defining Training and Validation epochs

# In[ ]:


def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    model.train()
    bar = tqdm(iterator) if Progress_Bar else iterator
    
    for (x, y) in bar:
        
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        epoch_loss += loss_np
        if Progress_Bar:
            bar.set_description('Training loss: %.5f' % (loss_np))
        
    return epoch_loss/len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    preds = []
    preds = np.array(preds)
    targets = []
    targets = np.array(targets)
    model.eval()
    bar = tqdm(iterator) if Progress_Bar else iterator
    
    with torch.no_grad():
        
        for (x, y) in bar:
        
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_np = loss.detach().cpu().numpy()
            epoch_loss += loss_np
            preds = np.append(preds, np.argmax(y_pred.detach().cpu().numpy(), axis = 1))
            targets = np.append(targets, y.detach().cpu().numpy())
#             preds = preds.reshape(-1)
#             targets = targets.reshape(-1)
            
            if Progress_Bar:
                bar.set_description('Validation loss: %.5f' % (loss_np))
            
    
            
    return epoch_loss/len(iterator), metrics.cohen_kappa_score(targets, preds, weights='quadratic')


# # Defining Training Loop

# In[ ]:


def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):
    """ Fits a dataset to model"""
    best_valid_loss = float('inf')
    
    train_losses = []
    valid_losses = []
    valid_metric_scores = []
    
    for epoch in range(epochs):
    
        start_time = time.time()
    
        train_loss = train(model, train_iterator, optimizer, loss_criterion, device)
        valid_loss, valid_metric_score = evaluate(model, valid_iterator, loss_criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_metric_scores.append(valid_metric_score)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_name}.pt')
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val. Loss: {valid_loss:.3f} |  Val. Metric Score: {valid_metric_score:.3f}')
        
    return train_losses, valid_losses, valid_metric_scores
        
#     return pd.DataFrame({f'{model_name}_Training_Loss':train_losses, 
#                         f'{model_name}_Training_Acc':train_accs, 
#                         f'{model_name}_Validation_Loss':valid_losses, 
#                         f'{model_name}_Validation_Acc':valid_accs})


# # Training with 5-Fold CV

# In[ ]:


tr_loss=[]
val_loss=[]
val_metric=[]

for fold in range(1):
    print(f"Fitting on Fold {fold+1}")
    #Make Train and Valid DataFrame from fold
    train_df_fold = train_df[train_df['fold'] != fold]
    valid_df_fold = train_df[train_df['fold'] == fold]
    
    #Build and load Dataset
    train_data = Build_Dataset(train_df_fold, resize=(256, 256), mode='train', augmentations=train_transforms)
    valid_data = Build_Dataset(valid_df_fold, resize=(256, 256), mode='train', augmentations=test_transforms)
    train_iterator = DataLoader(train_data, shuffle=True, batch_size=16, num_workers=4)
    valid_iterator = DataLoader(valid_data, batch_size=16, num_workers=4)
    
    #Initialize model, loss and optimizer
    model = enetv2('efficientnet-b4', out_dim=6).to(device)
    loss_criterion = nn.CrossEntropyLoss().to(device)
    opt1=optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999))
    
    temp_tr_loss, temp_val_loss, temp_val_metric = fit_model(model, 'efficientnet-b4', train_iterator, valid_iterator, opt1, loss_criterion, device, epochs=3)
    
    tr_loss+=temp_tr_loss
    val_loss+=temp_val_loss
    val_metric+=temp_val_metric
    


# In[ ]:


opt2 = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
temp_tr_loss, temp_val_loss, temp_val_metric = fit_model(model, 'efficientnet-b4', train_iterator, valid_iterator, opt2, loss_criterion, device, epochs=4)

tr_loss+=temp_tr_loss
val_loss+=temp_val_loss
val_metric+=temp_val_metric


# In[ ]:


opt3 = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
temp_tr_loss, temp_val_loss, temp_val_metric = fit_model(model, 'efficientnet-b4', train_iterator, valid_iterator, opt3, loss_criterion, device, epochs=3)

tr_loss+=temp_tr_loss
val_loss+=temp_val_loss
val_metric+=temp_val_metric


# In[ ]:


opt4 = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.99))
temp_tr_loss, temp_val_loss, temp_val_metric = fit_model(model, 'efficientnet-b4', train_iterator, valid_iterator, opt4, loss_criterion, device, epochs=4)

tr_loss+=temp_tr_loss
val_loss+=temp_val_loss
val_metric+=temp_val_metric


# # **Plotting the Losses and the Metric**

# In[ ]:


len(tr_loss)


# In[ ]:


plt.rcParams.update({'font.size':18})
sns.set_style('darkgrid')
plt.rcParams.update({'font.family':'Humor-Sans'})

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
ax[0].plot(tr_loss)
ax[0].set_title('Training and Validation Loss')
ax[0].plot(val_loss)
ax[0].set_ylim((1,2))
ax[0].set_xlabel('Epoch')

ax[1].plot(val_metric)
ax[1].set_title('Val Cohen Score')
ax[1].set_xlabel('Epoch')


ax[0].legend();
ax[1].legend();


# # Making submission to leaderboard

# In[ ]:


def get_predictions(model, iterator, device):
    
    preds = []
    model.eval()
    bar = tqdm(iterator) if Progress_Bar else iterator
    
    with torch.no_grad():
        
        for (x, y) in bar:
        
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            y_pred = model(x)
            preds.append(np.argmax(y_pred.detach().cpu().numpy(), axis = 1))
            
    preds = np.array(preds)
    preds = preds.reshape(-1)
            
    return preds


# In[ ]:


test_df = pd.read_csv(data_dir+'test.csv')
sample = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')
test_df.drop(columns=['data_provider'], inplace=True)
test_img_dir = '../input/prostate-cancer-grade-assessment/test_images'
    
# #Build and load Test Data
# test_data = Build_Dataset(test_df, resize=(256, 256), mode='test', augmentations=test_transforms)
# test_iterator = DataLoader(test_data, batch_size=2, num_workers=4)
    
# #Get predictions
# y_pred = get_predictions(model, test_iterator, device)
    
# #Submit Predictions
# test_df['isup_grade'] = y_pred
# test_df.to_csv('submission.csv', index=False)


# In[ ]:


def submit(sample):
    if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
        test_data = Build_Dataset(test_df, resize=(256, 256), mode='test', augmentations=test_transforms)
        test_iterator = DataLoader(test_data, batch_size=2, num_workers=4)
        preds = get_predictions(model, test_iterator, device)
        sample['isup_grade'] = preds
    return sample


# In[ ]:


submission = submit(sample)
submission['isup_grade'] = submission['isup_grade'].astype(int)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




