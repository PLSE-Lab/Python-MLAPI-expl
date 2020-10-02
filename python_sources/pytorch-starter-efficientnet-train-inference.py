#!/usr/bin/env python
# coding: utf-8

# # Motivation

# Although I'm a beginner in computer vision, I'd like to share a starter code by pytorch. I tried to follow the manner introduced in the pytorch [tutorials](https://pytorch.org/tutorials/) since I always suffered from understanding fully customized or wrapped codes shared in the kernels. I also customized a little since the example condes in these tutorials are too simple that they don't tell us how to implement early stopping, learning rate update with validation etc... Anyway, if there is better way of coding, please leave me a comment.
# 
# I've refered the following datasets and notebooks:
# Dataset
# - [Iterative-Stratification](https://www.kaggle.com/sheriytm/iterativestratification)
# - [EfficientNet PyTorch](https://www.kaggle.com/hmendonca/efficientnet-pytorch)
# - [EfficietNet3-Pytorch-Training-Inference](https://www.kaggle.com/gopidurgaprasad/efficietnet3-pytorch-training-inference)
# 
# Notebooks
# - [iterative stratification](https://www.kaggle.com/yiheng/iterative-stratification)
# - [mixup/cutmix is all you need](https://www.kaggle.com/c/bengaliai-cv19/discussion/126504)

# # Install Necessary Libraries
# 
# Since we cannot use the Internet access in this competition, the installer files of the necessary libraries need to be added as datasets

# In[ ]:


get_ipython().system('pip install ../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master/  > /dev/null')
#!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null
get_ipython().system('pip install iterative-stratification > /dev/null')


# # Import Libraries

# In[ ]:


#+---- Basic Libraries ----+#
import sys, os, time, gc, random
from pathlib import Path
import pandas as pd
import numpy as np
import copy

#+---- Utilities Libraries ----+#
from efficientnet_pytorch import EfficientNet
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from tqdm.notebook import tqdm
import sklearn

#+---- Pytorch Libraries ----+#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import model_zoo
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

#+---- List the input data ----+#
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Initial Settings

# ## Directories

# In[ ]:


DATADIR = Path('/kaggle/input/bengaliai-cv19')
FEATHERDIR = Path('/kaggle/input/bengaliaicv19feather')
OUTDIR = Path('.')
MDL_DIR = '/models'
LOG_DIR = '/logs'
if not os.path.exists(f'.{MDL_DIR}'):
    os.mkdir(f'.{MDL_DIR}')
if not os.path.exists(f'.{LOG_DIR}'):
    os.mkdir(f'.{LOG_DIR}')


# ## Learning Parameters

# In[ ]:


DEBUG =True # if we train model in small part of dataset to save time to debug
SUBMISSION = True
WORKER = 4
SEED = 6666

BATCH_SIZE =16 ## batch size is changed from 64 to 16 because of the RAM limitation
NUM_EPOCH = 1
IMAGE_SIZE=128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME ='efficientnet-b4'
VER = 'fold_1_mixup_cutmix'
N_Fold = 10 # 10-fold cross validation
TRAIN_RATIO = 0.9 # if holdout for validation, the data size for validation dataset
CV = True # choose validation methed: if you train model by cross validation, then True, otherwise False for hold out
Fold = 1 # clarify the fold to be trained
PATIAENCE = 4 # early stopping patiance parameter


# In[ ]:


n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant


# # Data Preparation

# ## Transform class for data preprocessing and augmentations

# In[ ]:


def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images


# In[ ]:


def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


# In[ ]:


data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(0.5,0.5,0.5,0.5),
        transforms.RandomAffine(degrees=0.6),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# ## Dataset Class

# In[ ]:


# we need to prepare the dataset class by customizing the __len__ and __getitem__ funcitions
# suit for the prediction task 
class BengaliAIDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, indices=None):
        self.transform = transform
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)
      
    def __getitem__(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformation
        x = (255 - x).astype(np.float32) #/ 255.
        x = crop_char_image(x)
        x = Image.fromarray(x).convert("RGB")
        x = self.transform(x)
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x


# ## Data Separation

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Split data set used for cross validation\ntrain = pd.read_csv(DATADIR/'train.csv')\ntrain['id'] = train['image_id'].apply(lambda x: int(x.split('_')[1]))\nX, y = train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]\\\n.values[:,0], train.values[:,1:]\ntrain['fold'] = np.nan\nmskf = MultilabelStratifiedKFold(n_splits=N_Fold)\nfor i, (_, index) in enumerate(mskf.split(X, y)):\n    #print('Fold '+str(i+1))\n    train.iloc[index, -1] = i\ntrain['fold'] = train['fold'].astype('int')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values\nindices = [0] if DEBUG else [0, 1, 2, 3]\ntrain_images = prepare_image(\n    DATADIR, FEATHERDIR, data_type='train', submission=False, indices=indices)")


# In[ ]:


n_dataset = len(train_images)

if not CV:
    train_data_size = 200 if DEBUG else int(n_dataset * TRAIN_RATIO)
    valid_data_size = 100 if DEBUG else int(n_dataset - train_data_size)
    perm = np.random.RandomState(777).permutation(n_dataset)
    print('perm', perm)

    train_dataset = BengaliAIDataset(
        train_images, train_labels, transform=data_transforms['train'],
        indices=perm[:train_data_size])

    valid_dataset = BengaliAIDataset(
        train_images, train_labels, transform=data_transforms['val'],
        indices=perm[train_data_size:train_data_size+valid_data_size])
else:
    valid_idx = np.array(train[train['fold']==Fold].index)
    trn_idx = np.array(train[train['fold']!=Fold].index)
    trn_idx = trn_idx[:200] if DEBUG else trn_idx
    valid_idx = valid_idx[:100] if DEBUG else valid_idx
    
    train_dataset = BengaliAIDataset(
        train_images, train_labels, transform=data_transforms['train'],
        indices=trn_idx)
    valid_dataset = BengaliAIDataset(
        train_images, train_labels, transform=data_transforms['val'],
        indices=valid_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER)

dataloaders = {'train':train_loader, 'val': valid_loader}
dataset_sizes = {'train':len(train_dataset), 'val': len(valid_dataset)}


# In[ ]:


image, label = train_dataset[1]
print('image', image.shape, 'label', label)


# # Functions for evaluation and augmentations

# In[ ]:


def macro_recall(pred_labels, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    recall_grapheme = sklearn.metrics.recall_score(y[0], pred_labels[0],  average='macro')
    recall_vowel = sklearn.metrics.recall_score(y[1],pred_labels[1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(y[2],pred_labels[2],  average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
           f'total {final_score}')
    return final_score


# In[ ]:


def get_pred(preds_list, label_list):
    #preds_list is torch tensor to device
    #label_list is torch tensor to device
    _, pred0 = torch.max(preds_list[0], 1)
    _, pred1 = torch.max(preds_list[1], 1)
    _, pred2 = torch.max(preds_list[2], 1)
    p0 = pred0.cpu().numpy()
    p1 = pred1.cpu().numpy()
    p2 = pred2.cpu().numpy()
    pred_labels = [p0, p1, p2]
    #print(pred_labels)
    a0 = label_list[0].cpu().numpy()
    a1 = label_list[1].cpu().numpy()
    a2 = label_list[2].cpu().numpy() 
    y = [a0, a1, a2]
    #print(y)
    return pred_labels, y


# In[ ]:


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3,                shuffled_targets3, lam]
    return data, targets

def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3,               shuffled_targets3, lam]

    return data, targets


def cutmix_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1],    targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) +lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) +lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

def mixup_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0],     targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)+ lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) +lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)


# # Functions for model training

# There are train phase and validation phase in each epoch. During the train phase, the 5% and 5% of the mini-batch of the data are augumented by mixup or cutmix. The hyper-parameter alpha is set to 0.1 since too large alpha may cause [underfitting](https://arxiv.org/abs/1710.09412).
# ```
# ratio = 0.1
# randomlist = random.sample( range(length), int(ratio*length))
# mixuplist = randomlist[:int(ratio*length/2)]
# cutmixlist = randomlist[int(ratio*length/2):]
# ```
# In the following train_model function, the model is trained based on the loss function, but early stopping and learning rate scheduler work based the recall score of the validation dataset.

# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, scheduler,start_epoch,
                num_epochs, device, patiance):
    since = time.time()
    
    trn_loss_list =[]
    trn_acc_list = []
    val_loss_list =[]
    val_acc_list = []
    epoch_list = []
    recall_list = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10
    best_recall = 0.0
    torch.backends.cudnn.benchmark = True
    early_stopping_counter = 0
    
    for epoch in range(num_epochs)[start_epoch:]:
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 30)  
        
        if early_stopping_counter == patiance:
            print(f'Early Stopped since loss have not decreased for {patiance} epoch.')
            break
        epoch_list.append(epoch+1)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_loss = 0.0
            epoch_corrects = 0
            dataset_sizes = len(dataloaders[phase].dataset)
            length = int(np.floor(dataset_sizes/BATCH_SIZE))
            ratio = 0.1
            randomlist = random.sample( range(length), int(ratio*length))
            mixuplist = randomlist[:int(ratio*length/2)]
            cutmixlist = randomlist[int(ratio*length/2):]

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                #print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.transpose(1,0).to(device) #use when single label for one image

                grapheme_root = labels[0]
                vowel_diacritic = labels[1]
                consonant_diacritic = labels[2]
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):                                 
                    if phase == 'train':
                        if idx in mixuplist:
                            
                            inputs_mixed, labels_mixed = mixup(inputs, grapheme_root,                                                                vowel_diacritic, consonant_diacritic, 0.1)

                            outputs = model(inputs_mixed) 
                            grapheme_root_prd = outputs[0]
                            vowel_diacritic_prd = outputs[1]
                            consonant_diacritic_prd = outputs[2]
                            loss = mixup_criterion(grapheme_root_prd,vowel_diacritic_prd,                                               consonant_diacritic_prd, labels_mixed)
                            
                        elif idx in cutmixlist:
                            
                            inputs_cut, labels_cut = cutmix(inputs, grapheme_root,                                                                vowel_diacritic, consonant_diacritic, 0.1)

                            outputs = model(inputs_cut) 
                            grapheme_root_prd = outputs[0]
                            vowel_diacritic_prd = outputs[1]
                            consonant_diacritic_prd = outputs[2]
                            loss = cutmix_criterion(grapheme_root_prd,vowel_diacritic_prd,                                               consonant_diacritic_prd, labels_cut)
                        
                        else:
                            outputs = model(inputs)
                            grapheme_root_prd = outputs[0]
                            vowel_diacritic_prd = outputs[1]
                            consonant_diacritic_prd = outputs[2]
                            loss = (1/3)*(criterion(grapheme_root_prd, grapheme_root)+                                  criterion(vowel_diacritic_prd, vowel_diacritic) +                                     criterion(consonant_diacritic_prd, consonant_diacritic))
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        outputs = model(inputs)
                        grapheme_root_prd = outputs[0]
                        vowel_diacritic_prd = outputs[1]
                        consonant_diacritic_prd = outputs[2]
                        loss = (1/3)*(criterion(grapheme_root_prd, grapheme_root)+                              criterion(vowel_diacritic_prd, vowel_diacritic) +                                 criterion(consonant_diacritic_prd, consonant_diacritic))

                        
                # statistics: inputs.size(0) is batch size
                epoch_loss += loss.item() * inputs.size(0) # total loss for this batch
                epoch_corrects += torch.sum(torch.max(outputs[0], 1)[1] == labels[0])+                    torch.sum(torch.max(outputs[1], 1)[1] == labels[1])+                    torch.sum(torch.max(outputs[2], 1)[1] == labels[2])
                
            epoch_loss = epoch_loss / dataset_sizes
            epoch_acc = epoch_corrects.double() / (dataset_sizes*3)
            pred, lbls = get_pred(outputs, labels)
            recall = macro_recall(pred, lbls,                                       n_grapheme=168, n_vowel=11, n_consonant=7)
            
            if phase == 'train':
                trn_loss_list.append(epoch_loss)
                trn_acc_list.append(epoch_acc.cpu().numpy())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and recall > best_recall:
                best_model_wts = copy.deepcopy(model.state_dict())
                if not os.path.exists(f'.{MDL_DIR}/{MODEL_NAME}_{VER}'):
                    os.mkdir(f'.{MDL_DIR}/{MODEL_NAME}_{VER}')
                save_path = f'.{MDL_DIR}/{MODEL_NAME}_{VER}/{MODEL_NAME}_'+str(epoch+1)+'.pth'
                torch.save(model_ft.state_dict(),save_path)
                best_epoch = epoch
            
            if phase == 'val':
                if epoch == 0 or epoch == start_epoch:
                    best_recall = recall
                    
                else:
                    if recall > best_recall:
                        print(recall)
                        best_recall = recall
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        print(f'Early stopping counter: {early_stopping_counter}')

                scheduler.step(epoch_loss) 
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc.cpu().numpy())
              
                
                print('valid recall score is {:.3f}'.format(recall))
                recall_list.append(recall)

        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Recall: {:4f}'.format(best_recall))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if not os.path.exists(f'.{LOG_DIR}/log_{MODEL_NAME}_{VER}.csv'):
        log = pd.DataFrame()
        log['Epoch'] = epoch_list
        log['Train Loss'] = trn_loss_list
        log['Train Acc'] = trn_acc_list
        log['Valid Loss'] = val_loss_list
        log['Valid Acc'] = val_acc_list
        log['Recall'] = recall_list
        log.to_csv(f'.{LOG_DIR}/log_{MODEL_NAME}_{VER}.csv',index=False)
    else:
        log = pd.DataFrame()
        log['Epoch'] = epoch_list
        log['Train Loss'] = trn_loss_list
        log['Train Acc'] = trn_acc_list
        log['Valid Loss'] = val_loss_list
        log['Valid Acc'] = val_acc_list
        log['Recall'] = recall_list
        log_old = pd.read_csv(f'.{LOG_DIR}/log_{MODEL_NAME}_{VER}.csv')
        LOG = pd.concat([log_old, log], axis=0)
        LOG.reset_index(drop=True, inplace=True)
        LOG.to_csv(f'.{LOG_DIR}/log_{MODEL_NAME}_{VER}.csv',index=False)
    return model, best_epoch+1


# Since the last layer of the original efficient net is for singla class prediction, we need to customize a litte. The modifined model will give the 3 ouptuts for grapheme, consonant, and vowel.

# In[ ]:


class bengali_model(nn.Module):
    def __init__(self, num_classes1, num_classes2, num_classes3):
        super(bengali_model, self).__init__()
        #pretrain models
        #self.model = pretrainedmodels.__dict__[MODEL_NAME](pretrained=None)
        #num_ftrs = self.model.last_linear.in_features
        #self.model.last_linear = nn.Identity()
        
        # EfficientNet
        self.model = EfficientNet.from_name(MODEL_NAME)
        
        # if internet is allowed, we can use pretrained weight
        #self.model = EfficientNet.from_pretrained(MODEL_NAME)
        
        num_ftrs = 1792
        
        self.fc1 = nn.Linear(num_ftrs, num_classes1)
        self.fc2 = nn.Linear(num_ftrs, num_classes2)
        self.fc3 = nn.Linear(num_ftrs, num_classes3)

    def forward(self, x):
        #x = self.model(x) #pretrain models
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        return out1, out2, out3


# In[ ]:


# --- Model --- Stage 1
model_ft = bengali_model(n_grapheme, n_vowel, n_consonant)
model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-10, verbose=True)


# In[ ]:


torch.cuda.empty_cache()
gc.collect()


# In[ ]:


START_EPOCH = 0
model_ft, best_epoch = train_model(model_ft, dataloaders, criterion, optimizer, scheduler, START_EPOCH,
                                   NUM_EPOCH, DEVICE, PATIAENCE)


# # Evaluation

# In[ ]:


def predict(model, dataloaders, phase, device):
    model.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        if phase == 'test':
            for i, inputs in enumerate(tqdm(dataloaders)):
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, pred0 = torch.max(outputs[0], 1)
                _, pred1 = torch.max(outputs[1], 1)
                _, pred2 = torch.max(outputs[2], 1)
                preds = (pred0, pred1, pred2)
                output_list.append(preds)
            return output_list
        elif phase == 'val':
            for i, (inputs, labels) in enumerate(tqdm(dataloaders)):
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, pred0 = torch.max(outputs[0], 1)
                _, pred1 = torch.max(outputs[1], 1)
                _, pred2 = torch.max(outputs[2], 1)
                preds = (pred0, pred1, pred2)
                output_list.append(preds)
                label_list.append(labels.transpose(1,0))
            return output_list, label_list


# In[ ]:


save_path = f'.{MDL_DIR}/{MODEL_NAME}_{VER}/{MODEL_NAME}_'+str(best_epoch)+'.pth'
load_weights = torch.load(save_path)
model_ft.load_state_dict(load_weights)


# In[ ]:


# --- Prediction ---
data_type = 'val'
valid_preds_list = []
print('valid_dataset', len(valid_dataset))
valid_preds_list, valid_label_list = predict(model_ft, valid_loader, data_type, DEVICE)
gc.collect()


# In[ ]:


# Each test_preds indicates the prediction outputs of different batch
p0 = np.concatenate([valid_preds[0].cpu().numpy() for valid_preds in valid_preds_list], axis=0)
p1 = np.concatenate([valid_preds[1].cpu().numpy() for valid_preds in valid_preds_list], axis=0)
p2 = np.concatenate([valid_preds[2].cpu().numpy() for valid_preds in valid_preds_list], axis=0)
print('p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)

a0 = np.concatenate([valid_label[0].cpu().numpy() for valid_label in valid_label_list], axis=0)
a1 = np.concatenate([valid_label[1].cpu().numpy() for valid_label in valid_label_list], axis=0)
a2 = np.concatenate([valid_label[2].cpu().numpy() for valid_label in valid_label_list], axis=0)
print('a0', a0.shape, 'a1', a1.shape, 'a2', a2.shape)

pred_labels = [p0, p1, p2]
y = [a0, a1, a2]
macro_recall(pred_labels, y, n_grapheme=168, n_vowel=11, n_consonant=7)


# # Inference

# In[ ]:


def prepare_image_test(datadir, featherdir, data_type='train',
                  submission=True, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
        
        
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    #del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images, image_df_list


# In[ ]:


# --- Prediction ---

sts = time.time()
data_type = 'test'
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder

for i in tqdm(range(4)):
    indices = [i]
    test_images, df_test_img = prepare_image_test(
        DATADIR, FEATHERDIR, data_type = data_type, submission=SUBMISSION, indices=indices)
    n_dataset = len(test_images)
    print(f'i={i}, n_dataset={n_dataset}')
    test_dataset = BengaliAIDataset(
    test_images, None,
    transform=data_transforms[data_type])
    print('test_dataset', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER)
    
    test_preds = predict(model_ft, test_loader, data_type,DEVICE)
    p0 = np.concatenate([valid_preds[0].cpu().numpy() for valid_preds in valid_preds_list], axis=0) #grapheme
    p1 = np.concatenate([valid_preds[1].cpu().numpy() for valid_preds in valid_preds_list], axis=0) #vowel
    p2 = np.concatenate([valid_preds[2].cpu().numpy() for valid_preds in valid_preds_list], axis=0) #consonant
    tgt = [p2, p0, p1]
    for idx, id in enumerate(df_test_img[0].image_id.values): # df_test_img.index.values has the test image_ids
        #print(idx)
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(tgt[i].flatten()[idx]) # our model is a random integer generator between 0 and n_cls
    del test_images, df_test_img
    gc.collect()
    if DEBUG:
        break
ed = time.time()
print('Predicted in {:.3f} min'.format((ed-sts)/60))
del p0, p1, p2


# In[ ]:


# create a dataframe with the solutions 
sub_df = pd.DataFrame(
    {'row_id': row_id,
    'target':target
    },
    columns =['row_id','target'] 
)
sub_df.head()


# In[ ]:


sub_df.shape


# In[ ]:


sub = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
if len(sub) == len(sub_df):
    sub_df.to_csv('submission.csv', index=False)
    print('Prediction file saved successfully.')
else:
    sub.to_csv('submission.csv', index=False)

