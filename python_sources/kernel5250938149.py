#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dev_TPU=False
kaggle=True


# In[ ]:


if dev_TPU==True:
    get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
    get_ipython().system('python pytorch-xla-env-setup.py --version 20200515 --apt-packages libomp5 libopenblas-dev')


# In[ ]:


if dev_TPU==True:
    get_ipython().system('export XLA_USE_BF16=1')


# In[ ]:


if dev_TPU==True:
    get_ipython().system('pip install torch_xla')


# In[ ]:


if dev_TPU==True:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp


# In[ ]:


get_ipython().system(' pip install iterative-stratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn import model_selection 
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm.notebook import tqdm

from sklearn.utils import multiclass
from sklearn.preprocessing import MultiLabelBinarizer
from torch.cuda import amp

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


torch.manual_seed(10)


# In[ ]:


if kaggle==True:
    ROOT_DIR = '../input/jovian-pytorch-z2g'
    DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'
else:
    ROOT_DIR = '/content'
    DATA_DIR = '/content'
    

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = ROOT_DIR+'/submission.csv'   # Contains dummy labels for test image


# In[ ]:


df = pd.read_csv(TRAIN_CSV)
vals = df.Label.values
y=[]
for x in vals:
    y.append([int(i) for i in x.split()])

y = MultiLabelBinarizer().fit_transform(y)
multiclass.type_of_target(y)


# In[ ]:


df = pd.read_csv(TRAIN_CSV)
df['kfold'] = -1
df = df.sample(frac=1).reset_index(drop=True)
"""
kf = model_selection.KFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df)):
    df.loc[v_, 'kfold'] = f
"""
mskf = MultilabelStratifiedKFold(n_splits=5)
vals = df.Label.values
y=[]
for x in vals:
    y.append([int(i) for i in x.split()])
y = MultiLabelBinarizer().fit_transform(y)
#multiclass.type_of_target(y) - > multilabel-indicator
for f, (t_, v_) in enumerate(mskf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv("train_folds.csv", index=False)


# In[ ]:


labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}


# In[ ]:


public_threshold = 0.3
local_threshold = 0.5
def encode_label(label):
    #print(label)
    target = torch.zeros(10)
    #for l in str(label).split(' '):
    for l in label:
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=public_threshold):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
    


# In[ ]:


class HumanProteinDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['Image'], row['Label']
        #print(img_label)
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, encode_label(img_label)


# In[ ]:


mean = [0.0792, 0.0529, 0.0544]
std = [0.1288, 0.0884, 0.1374]
train_transform = transforms.Compose([
            #transforms.RandomCrop(512, padding=8, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)])
test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)])


# In[ ]:


batch_size = 64


# In[ ]:


def F_score(output, label, threshold=local_threshold, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    #return F2.mean(0)
    return F2.mean()

def F_loss(output, label, threshold=local_threshold, beta=1):
    #prob = output > threshold
    prob = output
    label = label > threshold

    TP = torch.sum(prob * label,axis=0).float()
    #TN = torch.sum((torch.ones_like(prob)-prob) * (~label),axis=0).float()
    FP = torch.sum(prob * (~label),axis=0).float()
    FN = torch.sum((torch.ones_like(prob)-prob) * label,axis=0).float()
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))

    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    #return (1-F2.mean(0))
    return (1-F2.mean())


# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()


# In[ ]:


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        #loss = F.binary_cross_entropy(out, targets) 
        loss = F_loss(out, targets)     
        #loss = FocalLoss()(out, targets)
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        #loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        loss = F_loss(out, targets)     
        #loss = FocalLoss()(out, targets)
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


# In[ ]:


class ProteinCnnModel2(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
        #return self.network(xb)
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


if dev_TPU==True:
    #TPU
    device = xm.xla_device()
else:
    #CUDA
    device = get_default_device()


# In[ ]:


device, device==torch.device('cpu')


# In[ ]:


class Oversampling:
    #def __init__(self,path):
        #self.train_labels = pd.read_csv(path)
    def __init__(self,df):
        self.train_labels = df
        self.train_labels['Label'] = [[int(i) for i in s.split()] 
                                       for s in self.train_labels['Label']]  
        #set the minimum number of duplicates for each class
        self.multi = [2,2,2,2,1,3,1,2,3,2]  #{0: 1.94, 1: 2.12, 2: 1.75, 3: 2.0, 4: 1.0, 5: 2.58, 6: 1.0, 7: 1.71, 8: 2.64, 9: 2.44}

    def get(self,image_id):
        labels = self.train_labels.loc[image_id,'Label'] if image_id           in self.train_labels.index else []
        #print(labels)
        m = 1
        for l in labels:
            if m < self.multi[l]: m = self.multi[l]
        return m


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    tk1 = tqdm(val_loader, total=len(val_loader), desc="Validating")
    #outputs = [model.validation_step(batch) for batch in tk1]
    outputs = []
    for batch in tk1:
        vstep = model.validation_step(batch)
        outputs.append(vstep)
        if dev_TPU==False:
            tk1.set_postfix(vloss=vstep['val_loss'].item())
    return model.validation_epoch_end(outputs)

#def fit(fold, epochs, lr, model, opt_func=torch.optim.SGD):
def fit(fold, epochs, lr, model, opt_func, scaler):
    torch.cuda.empty_cache()
    history = []
    best_score = 0
    df = pd.read_csv("train_folds.csv")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    val_df = df[df.kfold == fold].reset_index(drop=True)
    
    #Oversampling to increase threshold of rare classes
    train_df_orig=train_df.copy()
    s = Oversampling(train_df)
    tr_n = [idx for idx in train_df['Image'].values for _ in range(s.get(idx))]
    #print(len(tr_n),flush=True)
    l = list(set(tr_n))
    finallist = [i for i in tr_n if not i in l or l.remove(i)]
    #print(len(finallist))
    
    #train_df_orig=train_df.copy()
    for i in finallist:
        #indicies = train_df_orig.loc[train_df_orig['Image'] == i].index
        #print(indicies)
        #print(train_df_orig.loc[indicies])
        #train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
        #train_df = train_df_orig.loc[indicies]#, ignore_index=True)
        dfindex = train_df_orig[train_df_orig['Image'] == i]
        train_df.append([dfindex],ignore_index=True)
    val_df['Label'] = [[int(i) for i in s.split()] for s in val_df['Label']]
    train_ds = HumanProteinDataset(train_df, TRAIN_DIR, transform=train_transform)
    val_ds = HumanProteinDataset(val_df, TRAIN_DIR, transform=test_transform)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
    train_loader = DeviceDataLoader(train_dl, device)
    val_loader = DeviceDataLoader(val_dl, device)
    to_device(model, device);
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
        threshold=0.001,
        verbose = True,
        mode="max"
    )
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        tk0 = tqdm(train_loader, total=len(train_loader), desc="Training")
        for batch in tk0: #tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            #loss.backward()
            scaler.scale(loss).backward()
            if dev_TPU==False:
                #optimizer.step()
                scaler.step(optimizer)
            else:
                xm.optimizer_step(optimizer, barrier=True)  # Note: Cloud TPU-specific code!
            scaler.update()
            optimizer.zero_grad()
            if dev_TPU==False:
                tk0.set_postfix(loss=loss.item())
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        if result['val_score'] > best_score:
            print("Saving model for fold {} : Score {} --> {}".format(fold, best_score, result['val_score']))
            best_score = result['val_score']
            if dev_TPU==True:
                xm.save(model.state_dict(), f"model_tpufold{fold}.bin")
            else:
                torch.save(model.state_dict(), "model_mskf{}.bin".format(fold))
        model.epoch_end(epoch, result)
        scheduler.step(result['val_score'])
        history.append(result)
    return history


# In[ ]:


train=True


# In[ ]:


if kaggle==True:
    modelpath = '../input/zerotogan-folds/'
else:
    modelpath = '/content/'
opt_func = torch.optim.Adam
lr = 0.00025 #1e-4  # pudae 3rd place 0.00025 -> 0.000125 -> 0.0000625
if train == True:
    num_epochs_freezed = 2
    num_epochs_unfreezed = 5
else:
    num_epochs = 2


# In[ ]:


scaler = amp.GradScaler()
if train==True:
  #Train for all the folds
  for fold in range(0,5):
    model = ProteinCnnModel2()
    #model.load_state_dict(torch.load(modelpath+"model_fold{}.bin".format(fold)))
    model.load_state_dict(torch.load(modelpath+"model_mskf{}.bin".format(fold)))
    if fold == 4:
        model.load_state_dict(torch.load("model_mskf{}.bin".format(fold)))
    model.freeze()
    history = fit(fold, num_epochs_freezed, lr, model, opt_func, scaler)
    model.unfreeze()
    history = fit(fold, num_epochs_unfreezed, lr, model, opt_func, scaler)
else:
    #For Experimentation
    model = ProteinCnnModel2()
    model.freeze()
    if dev_TPU==False and device!=torch.device('cpu'):
        model.load_state_dict(torch.load(modelpath+"model_mskf4.bin"))#, map_location=torch.device('cpu')))
    history = fit(4, num_epochs, lr, model, opt_func,scaler)
    model.unfreeze()
    history = fit(4, num_epochs, lr, model, opt_func,scaler)


# In[ ]:


fold=4
scaler = amp.GradScaler()
model = ProteinCnnModel2()
#model.load_state_dict(torch.load(modelpath+"model_fold{}.bin".format(fold)))
model.load_state_dict(torch.load(modelpath+"model_mskf{}.bin".format(fold)))
model.freeze()
history = fit(fold, num_epochs_freezed, lr, model, opt_func, scaler)
model.unfreeze()
history = fit(fold, num_epochs_unfreezed, lr, model, opt_func, scaler)


# In[ ]:


df_test = pd.read_csv(TEST_CSV)
df_test['Label'] = [[i] for i in df_test['Label']]
test_dataset = HumanProteinDataset(df_test, TEST_DIR, transform=test_transform)
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True), device)


# In[ ]:


df_test.head()


# In[ ]:


@torch.no_grad()
def predict_dl(dl, model):
    if dev_TPU==False:
        torch.cuda.empty_cache()
    batch_probs = []
    model.to(device)
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs


# In[ ]:


if train==True:
    #Predict with all the fold models
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load(modelpath+"model_fold0.bin"))
    test_preds0 = predict_dl(test_dl, model).clone().detach()
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load(modelpath+"model_fold1.bin"))
    test_preds1 = predict_dl(test_dl, model).clone().detach()
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load(modelpath+"model_fold2.bin"))
    test_preds2 = predict_dl(test_dl, model).clone().detach()
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load(modelpath+"model_fold3.bin"))
    test_preds3 = predict_dl(test_dl, model).clone().detach()
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load(modelpath+"model_fold4.bin"))
    test_preds4 = predict_dl(test_dl, model).clone().detach()
else:
    #Experimentation
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load("model_mskf4.bin"))
    test_pred = predict_dl(test_dl, model).clone().detach()
    test_pred.shape


# In[ ]:


if train==True:
    test_pred = test_preds0 + test_preds1 + test_preds2 + test_preds3 + test_preds4
    test_pred /= 5
    test_pred.shape


# In[ ]:


test_predictions = [decode_target(x) for x in test_pred]


# In[ ]:


submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = test_predictions
submission_df.head()


# In[ ]:


sub_fname = 'resnet34_submission.csv'
submission_df.to_csv(sub_fname, index=False)


# In[ ]:


#!pip install jovian --upgrade
#import jovian
#jovian.commit(project='zerogans-protein-trainfolds')


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))

def predict_single(image):
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    model = ProteinCnnModel2()
    model.load_state_dict(torch.load("model_mskf4.bin"))
    model.to(device)
    preds = model(xb)
    prediction = preds[0]
    print("Prediction: ", prediction)
    show_sample(image, prediction)


# df = pd.read_csv("train_folds.csv")
# valdf = df[df.kfold == 0].reset_index(drop=True)
# traindf = df[df.kfold != 0].reset_index(drop=True)
# valdf['Label'] = [[int(i) for i in s.split()] for s in valdf['Label']]
# traindf['Label'] = [[int(i) for i in s.split()] for s in traindf['Label']]
# 
# valds = HumanProteinDataset(valdf, TRAIN_DIR, transform=test_transform)
# trainds = HumanProteinDataset(traindf, TRAIN_DIR, transform=test_transform)
# 

# In[ ]:


valdf[340:380], len(traindf)


# In[ ]:


#predict_single(valds[376][0]),predict_single(valds[368][0]),predict_single(valds[374][0]),predict_single(valds[375][0]),predict_single(valds[372][0])
#predict_single(valds[377][0]),predict_single(valds[371][0]),predict_single(valds[373][0]),predict_single(valds[340][0]),predict_single(valds[344][0])

torch.mean(torch.stack(predict_single(valds[344][0]),predict_single(valds[367][0])))


# len([name for name in os.listdir(TRAIN_DIR)])
# import fnmatch
# len(fnmatch.filter(os.listdir(TRAIN_DIR), '*.png'))

# class Oversampling:
#     def __init__(self,path):
#         self.train_labels = pd.read_csv(path)
#         self.train_labels['Label'] = [[int(i) for i in s.split()] 
#                                        for s in self.train_labels['Label']]  
#         #set the minimum number of duplicates for each class
#         self.multi = [2,2,2,2,1,3,1,2,3,2]  #{0: 1.94, 1: 2.12, 2: 1.75, 3: 2.0, 4: 1.0, 5: 2.58, 6: 1.0, 7: 1.71, 8: 2.64, 9: 2.44}
# 
#     def get(self,image_id):
#         labels = self.train_labels.loc[image_id,'Label'] if image_id \
#           in self.train_labels.index else []
#         #print(labels)
#         m = 1
#         for l in labels:
#             if m < self.multi[l]: m = self.multi[l]
#         return m
#     
# s = Oversampling(TRAIN_CSV)
# #print(s.train_labels.index)
# #tr_n = [idx for idx in tr_n for _ in range(s.get(idx))]
# tr_n = [idx for idx in traindf['Image'].values for _ in range(s.get(idx))]
# print(len(tr_n),flush=True)

# #from collections import Counter
# s = list(set(tr_n))
# finallist = [i for i in tr_n if not i in s or s.remove(i)]
# print(len(finallist))
# #counts = Counter(tr_n)
# #dupids = [id for id in tr_n if counts[id] > 1]
# #print(dupids)
# #len(dupids)

# s = Oversampling(TRAIN_CSV)
# tr_n = [idx for idx in traindf['Image'].values for _ in range(s.get(idx))]
# print(len(tr_n),flush=True)
# l = list(set(tr_n))
# finallist = [i for i in tr_n if not i in l or l.remove(i)]
# 
# traindf_orig=traindf.copy()    
# for i in finallist:
#     indicies = traindf_orig.loc[traindf_orig['Image'] == i].index
#     traindf = pd.concat([traindf,traindf_orig.loc[indicies]], ignore_index=True)
# 

# traindf_orig=traindf.copy()    
# for i in finallist:
#     indicies = traindf_orig.loc[traindf_orig['Image'] == i].index
#     traindf = pd.concat([traindf,traindf_orig.loc[indicies]], ignore_index=True)
# 

# #mu in "create_class_weight" is a dampening parameter that could be tuned
# 
# import numpy as np
# import math
# 
# def create_class_weight(labels_dict, mu=0.5):
#     total = np.sum([labels_dict[i] for i in labels_dict.keys()])
#     print(total)
#     keys = labels_dict.keys()
#     class_weight = dict()
#     class_weight_log = dict()
# 
#     for key in keys:
#         score = total / float(labels_dict[key])
#         score_log = math.log(mu * total / float(labels_dict[key]))
#         class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
#         class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)
# 
#     return class_weight, class_weight_log
# 
# # Class abundance for protein dataset
# labels_dict = {  #[2088, 1752, 2542, 1977, 9066, 1109, 5711, 2629, 1037, 1278]
#     0: 2088,
#     1: 1752,
#     2: 2542,
#     3: 1977,
#     4: 9066,
#     5: 1109,
#     6: 5711,
#     7: 2629,
#     8: 1037,
#     9: 1278
# }
# 
# print('\nTrue class weights:')
# print(create_class_weight(labels_dict)[0])
# print('\nLog-dampened class weights:')
# print(create_class_weight(labels_dict)[1])
# #inferred this [4,4,3,4,1,7,1,3,7,6]
