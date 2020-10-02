#!/usr/bin/env python
# coding: utf-8

# # EfficientNet Regression Model
# 
# In this Kernel I am taking the approach of an EfficientNet Architecture.
# There are some concepts included from iafoss' tiles. However, they are then aggregated to a single picture and fed through the network.
# 
# Afterwards I use an optimized threshold to turn the regression into a classification again.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import os
#from sklearn.model_selection import KFold
from radam import *
from csvlogger import *
from mish_activation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score,confusion_matrix
import warnings
import scipy as sp
import skimage.io
import cv2

warnings.filterwarnings("ignore")

# remove this cell if run locally
get_ipython().system("mkdir 'cache'")
get_ipython().system("mkdir 'cache/torch'")
get_ipython().system("mkdir 'cache/torch/checkpoints'")
torch.hub.DEFAULT_CACHE_DIR = 'cache'

# EfficientNet imports
import sys
package_path = '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)
from efficientnet_pytorch import EfficientNet

from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2


# In[ ]:


bs = 4
n_epochs = 16
tile_sz = 132
nfolds = 5
N = 16

LABELS = '../input/prostate-cancer-grade-assessment/train.csv'


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 42
seed_everything(SEED)


# # Data Preparation and Cleaning
# Data cleaning is based on this Kernel: https://www.kaggle.com/tanulsingh077/prostate-cancer-in-depth-understanding-eda-model
# Thank you a lot for your EDA!

# In[ ]:


df = pd.read_csv(LABELS).set_index('image_id')

# Wrongly labeled data
wrong_label = df[(df['isup_grade'] == 2) & (df['gleason_score'] == '4+3')]
display(wrong_label)
df.drop([wrong_label.index[0]],inplace=True)
df = df.reset_index()

# incosistency with "0" and "negative"
df['gleason_score'] = df['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df,df.isup_grade))
folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds):
    if i == nfolds-1:
        folds_splits[splits[i][1]] = 0
    else:    
        folds_splits[splits[i][1]] = 1
    
df['split'] = folds_splits
df.head(10)


# In[ ]:


df['isup_grade'].hist()


# ## Datasets

# In[ ]:


def tile(img, sz=128, N=16):
    """ Subdivide large image in tiles and return most significant squares
    
    Params:
    img: large input image
    sz: size of tiles
    N: number of most important tiles
    
    Returns: list of N most significant tiles
    """
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
        
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img


# In[ ]:


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'../input/prostate-cancer-grade-assessment/train_images/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]
        image = tile(image, sz=tile_sz, N=N)
        image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                             cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                             cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                             cv2.vconcat([image[12], image[13], image[14], image[15]])])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx]).float()
        
        return image, label


# In[ ]:


def get_transforms(*, data):
    """
    Get image transformation of data
    """
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# In[ ]:


train_dataset = TrainDataset(df, df["isup_grade"], transform=get_transforms(data='train'))
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)

for image, label in train_loader:
    print(image[0].shape)
    plt.imshow(image[0].permute(1,2,0))
    plt.show()  
    break


# I use an image size of 528x528, as it is the optimal resolution for EffcientNetB6 and its compound factor. Therefore, my tile size is also 132 instead of 128

# # Model

# In[ ]:


class Model(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        
        # Load model backbone
        model = EfficientNet.from_name('efficientnet-b6')
        
        # Get preloaded model
        if pre:
            model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b6-c76e70fd.pth'))
        
        # Encoder, runs through the pretrained efficientnet
        self.enc = model
        
        # Neural network head. After running through the neural network, this is the transfer
        nc = list(model.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,512), 
                                  Mish(),
                                  nn.BatchNorm1d(512), 
                                  nn.Dropout(0.5),
                                  nn.Linear(512,1))
        
        
    def forward(self, x):
        # Extract features
        x = self.enc.extract_features(x)
    
        # Regression
        x = self.head(x)
        
        return x


# ### Regression to Classification conversion using the OptimizedRounder

# In[ ]:


# inspired by https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
class KappaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        # define score function:
        self.func = self.quad_kappa
    
    
    def predict(self, preds):
        return self._predict(self.coef, preds)

    
    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == 'Tensor':
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.FloatTensor(preds).view(-1)

        for i,pred in enumerate(y_hat):
            if   pred < coef[0]: y_hat[i] = 0
            elif pred < coef[1]: y_hat[i] = 1
            elif pred < coef[2]: y_hat[i] = 2
            elif pred < coef[3]: y_hat[i] = 3
            elif pred < coef[4]: y_hat[i] = 4
            else:                y_hat[i] = 5
        return y_hat.int()
    
    
    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    
    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds)
        
        if type(preds).__name__ == 'Tensor':
            return cohen_kappa_score(y.cpu(), y_hat.cpu(), weights='quadratic')
        else:
            return cohen_kappa_score(y, y_hat, weights='quadratic')

    
    def fit(self, preds, y):
        ''' maximize quad_kappa '''
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter':150, 'fatol':1e-10, 'xatol':1e-10})
        self.coef = opt_res.x

        
    def forward(self, preds, y):
        ''' the pytorch loss function '''
        return torch.tensor(self.quad_kappa(preds, y))

kappa_opt = KappaOptimizer()


# # Training

# In[ ]:


# Prepare databunch
fold =  0
train_idx = df[df['split'] != fold].index
val_idx = df[df['split'] == fold].index

train_dataset = TrainDataset(df.loc[train_idx].reset_index(drop=True), 
                             df.loc[train_idx].reset_index(drop=True)["isup_grade"], 
                             transform=get_transforms(data='train'))
valid_dataset = TrainDataset(df.loc[val_idx].reset_index(drop=True), 
                             df.loc[val_idx].reset_index(drop=True)["isup_grade"], 
                             transform=get_transforms(data='valid'))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)


data = DataBunch(train_dl = train_loader, valid_dl = valid_loader)


# In[ ]:


fname = 'EFFNETB0_REGRESSION'
model = Model()

learn = Learner(data, 
                model, 
                loss_func=MSELossFlat(),
                opt_func=Over9000, 
                metrics=[kappa_opt])

logger = CSVLogger(learn, f'log_{fname}_{fold}')

learn.clip_grad = 1.0
learn.split([model.head])
learn.unfreeze()

# Fit for n_epochs cycles
learn.fit_one_cycle(n_epochs, 
                    max_lr=1e-3, 
                    div_factor=100, 
                    pct_start=0.0, 
                    callbacks = [SaveModelCallback(learn,
                                                   name=f'model',
                                                   mode='min',
                                                   monitor='valid_loss')])

# Save model
torch.save(learn.model.state_dict(), f'{fname}_{fold}.pth')


# In[ ]:


learn.recorder.plot_losses()


# ## Evaluation

# In[ ]:


train_pred,train_target, pred, target = [],[],[],[]
learn.model.eval()
with torch.no_grad():
    for step, (x, y) in progress_bar(enumerate(data.dl(DatasetType.Train)),total=len(data.dl(DatasetType.Train))):
        p = learn.model(x)
        p = p.float().cpu()
        train_pred.append(p)
        train_target.append(y.cpu())
        
    for step, (x, y) in progress_bar(enumerate(data.dl(DatasetType.Valid)),total=len(data.dl(DatasetType.Valid))):
        p = learn.model(x)
        p = p.float().cpu()
        pred.append(p)
        target.append(y.cpu())


# In[ ]:


p = torch.cat(pred)
t = torch.cat(target)
p = kappa_opt.predict(p)
print(cohen_kappa_score(t,p,weights='quadratic'), "\n")
print(confusion_matrix(t,p), "\n")
print(kappa_opt.coef)


# In[ ]:


p_train = torch.cat(train_pred)
t_train = torch.cat(train_target)
kappa_opt.fit(p_train,t_train)

p = kappa_opt.predict(p)
print(cohen_kappa_score(t,p,weights='quadratic'), "\n")
print(confusion_matrix(t,p), "\n")
print(kappa_opt.coef)


# In[ ]:


get_ipython().system("rm -r 'cache'")

