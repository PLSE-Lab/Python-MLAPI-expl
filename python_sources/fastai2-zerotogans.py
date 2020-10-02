#!/usr/bin/env python
# coding: utf-8

# # Zero to GANs course competition using FASTAI2
# ### 8th place in public leaderboard and 18th in the private
# 

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# **Ths is the code used to download and use the Google Collab GPUs**

# **Fastai2 has to be installed in kaggle/collab, so internet has to be on
# The scikit multilearn to split the data. I dont remember were arff was needed.**

# In[ ]:


get_ipython().system('pip install fastai2 -q')
get_ipython().system('pip install scikit-multilearn -q')
get_ipython().system('pip install arff -q')


# In[ ]:


import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import recall_score
# from skmultilearn.model_selection import iterative_train_test_split
from fastai2.vision import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from fastai2.basics import *
from sklearn.metrics import f1_score


# In[ ]:


# DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'
# TRAIN_DIR = DATA_DIR + '/train'                           
# TEST_DIR = DATA_DIR + '/test'                             
# TRAIN_CSV = DATA_DIR + '/train.csv'                       
# TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv' 


# **Its a good idea to save everything (model and data and code) in the Gdrive, save me a lot of time**

# In[ ]:


#Google Collab
DATA_DIR = '/content/Human protein atlas'
TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images
TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = '/content/submission.csv'                      # Contains dummy labels for test image
model = '/content/drive/My Drive/model/proteinfast_b0_12x001_001'


# **This is the code I used to get the mean and std of the figures, good practive when transfer learning is not in use**

# In[ ]:



# fnames = get_image_files(TRAIN_DIR)
# ds = Datasets(fnames, tfms=Pipeline([PILImage.create, Resize(256), ToTensor]))
# dl = TfmdDL(ds, bs=32,after_batch=[IntToFloatTensor],drop_last=True)

# mean, std = 0., 0.
# for b in progress_bar(dl):
#   mean += b[0].mean((0,2,3))
#   std += b[0].std((0,2,3))

# print((mean/len(fnames)))
# print(std/len(fnames))


# In[ ]:


#this code is used to check the GPU in google collab, so we can adjust the batch size based on the GPU memory
get_ipython().system('nvidia-smi -l 1')


# **I use cross validation of 5 folds. Divide the data set in 5 different 80% training and 20% validation
# It is important to keep all the same for everytime we run the code, thats why this code below is important**

# In[ ]:


SEED = 13
nfolds = 5
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)


# **I have made my own Focal loss which has been described to works better with multiclassification of imbalanced data.
# Although I spent a lot of time reading fastai and pytorch source code to make my own loss, it didnt helped much in the end, so I finish not using it**

# In[ ]:


#Helper Functions

class FocalLoss(BaseLoss):       
    def __init__(self, *args, axis=-1, floatify=True, thresh=0.5, **kwargs):
        super().__init__(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.thresh = thresh
        self.alpha = alpha = 1
        self.gamma = gamma = 2
        self.logits = logits =True
        self.reduce = reduce
        
    def decodes(self, x):    return x>self.thresh
    def activation(self, x): return torch.sigmoid(x)

    def forward(self, inputs, targets):
#         inputs = nn.LogSigmoid(imputs)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def encode_label(label):
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)

def encode_array(label):
    target = np.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1
    return target
def encode_test(label):
    target = [0,0,0,0,0,0,0,0,0,0]
    return target

def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)

#from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.0,d=25.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = np.zeros(len(labels))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*p), axis=None)
    p, success = opt.leastsq(error, params)
    return p


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



df_test = pd.read_csv(TEST_CSV)
df_test['imgPath'] = df_test.apply(lambda x : os.path.join(TEST_DIR,str(x['Image'])+'.png'),axis=1)


# In[ ]:





# ## **Split data for cross validation**
# **This function split the dataset and create a list of datasets**

# In[ ]:


indexes = {v:k for k,v in labels.items()}


# In[ ]:


def create_split(nfolds=5):
    df = pd.read_csv(TRAIN_CSV).sort_values("Image").reset_index(drop=True)
    df['imgPath'] = df.apply(lambda x : os.path.join(TRAIN_DIR,str(x['Image'])+'.png'),axis=1)
    submission = pd.read_csv(TEST_CSV)
    submission['imgPath'] = submission.apply(lambda x : os.path.join(TEST_DIR,str(x['Image'])+'.png'),axis=1)
    split_df = pd.get_dummies(df.Label.str.split(' ').explode())
    split_df = split_df.groupby(split_df.index).sum()
    X, y = split_df.index.values, split_df.values
    k_fold = IterativeStratification(n_splits=nfolds, order=3)
    splits = list(k_fold.split(X, y))
    fold_splits = np.zeros(df.shape[0]).astype(np.int)
    for i in range(nfolds):
        fold_splits[splits[i][1]]=i
    df['Split'] = fold_splits
    df_folds = []
    for fold in range(nfolds):
        df_fold = df.copy()
        df_fold['is_valid'] = False
        df_fold.loc[df_fold.Split==fold,'is_valid'] = True
        df_folds.append(df_fold)
    return df_folds
#Create a list of Dfs
dfs = create_split()


# ## **Data block**
# Datablock is how fastai deal with the data to form the dataloader to the leraner.
# I started trainning with smaller images (128, 256 and lastly 512)
# I used the get_x and get_y functions from the https://www.kaggle.com/bismillahkani/protein-classification-using-fastai2
# Not many transformations were applied in my dataset (I starting testing without many transformations and after training a fold I didnt want to re-train, so I keept the same) - Probably a mistake, given that was overfitted 

# In[ ]:


# imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 512
def get_x(r): return r['imgPath']
def get_y(r): return r['Label'].split(' ')

def splitter(df):
    train = df.index[~df['is_valid']].to_list()
    valid = df.index[df['is_valid']].to_list()
    return train, valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                    get_x = get_x, 
                    get_y = get_y,
                    item_tfms=Resize(size),
                    splitter=splitter, 
                    batch_tfms=[Normalize.from_stats([0.0025, 0.0017, 0.0017], [0.0040, 0.0030, 0.0047]),
                              *aug_transforms(mult=1.0, 
                                      do_flip=True, 
                                      flip_vert=True,
                                      max_rotate=45.0, 
                                      max_zoom= 1.2, 
                                      max_lighting=0.1, 
                                      max_warp=0.1, 
                                      p_affine=0.1, 
                                      p_lighting=0.1, 
                                      xtra_tfms=None, 
                                      size=size, 
                                      mode='bilinear', 
                                      pad_mode='reflection', 
                                      align_corners=True, 
                                      min_scale=1.0)])


# ## Trainning loop
# 
# * Define model
# * Run loop
# * Save Results

# I have tested this models using 4 epochs using the same leraning rate to choose the best candidate xresnet50

# In[ ]:


# #######################################epoch 	train_loss 	valid_loss 	accuracy_multi 	F_score 	f1_score 	time
# arch1 = xresnet50(n_out=10)           #  3 	     0.246077 	 0.242087 	  0.905596 	    0.521767 	0.470186 	03:18
# arch2 = xse_resnet50(n_out=10)        #  3  	 0.272805 	 0.264964 	  0.895393  	0.439709 	0.374226 	03:29
# arch3 = xse_resnext50_deep(n_out=10)  #  3  	 0.257749 	 0.254541 	  0.902160  	0.478620 	0.427855 	03:15
# arch4 = xse_resnext50_deeper(n_out=10)#  3  	 0.274632 	 0.266044 	  0.895211  	0.432996 	0.376255 	03:12
# arch5 = xse_resnext34_deeper(n_out=10)#  3  	 0.239569 	 0.234023 	  0.909422  	0.549067 	0.508727 	03:05


# **I started using a loop to run the 5 folds, but was taking forever, so I moved to Collab and run 2 folds in parallel using 2 G accounts.
# So I create the model (usually load the already trainned model and keep trainning using the new Image size and batch size**

# In[ ]:


#T
arch = xresnet50(n_out=10, pretrained=False )
df = dfs[0]# next already
dls = dblock.dataloaders(df, bs=8)
learn = Learner(dls, arch, metrics=[partial(accuracy_multi, thresh=0.5),partial(F_score, threshold=0.5),F1ScoreMulti(), RecallMulti()]) #No focalloss
learn.load('model')
learn.dls = dblock.dataloaders(df, bs=8)


# **Fastai function to find the learning rate of the last layer (freezing all others)**

# In[ ]:


learn.lr_find()


# **I have tried many learning rates**
# *  Started using 1e-3 (8 epochs) Image 128
# *  Trained using 1e-3 (12 epochs) Image 256
# *  Trained for 12 epochs (1e-7)
# *  Trained for another 12 epochs (1e-8)

# In[ ]:


learn.clip = 1
lr = 0.001
learn.fit_one_cycle(12, lr=slice(lr/1000, lr/100, lr))


# **Saved the model for each trainning - very important.
# 
# I lost a lot of time because I was training all at once and my internet / computer / kaggle / collab / something always got wrong and I had to start all from beggining**

# In[ ]:


learn.save(model'+_12ep')


# # Get predictions

# To get the predictions I had to use reorder = False, as I was getting my data Shuffled all the time.
# 

# In[ ]:


dl = learn.dls.test_dl(df_test)
pred = learn.get_preds(dl=dl, reorder = False)


# In[ ]:


predictions = [decode_target(x, threshold=0.35) for x in pred[0]] # Get predicitons using the treshold


# **I saved all predictions not decoded in CSV so I could average them to make the submission**

# In[ ]:


predict=pred[0].numpy()


# In[ ]:


stats= pd.DataFrame(predict)


# In[ ]:


stats.to_csv('foldx.csv', index=False)


# In[ ]:


submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = predictions
sub_fname = 'fastai2_foldx.csv'
submission_df.to_csv(sub_fname, index=False)


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


project_name='protein2-fastai-focalloss'


# In[ ]:


jovian.commit(project=project_name, environment=None)


# In[ ]:




