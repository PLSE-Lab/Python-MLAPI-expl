#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# * This kernel is for kagglers like me who believe in trying a lot of things and don't fear of failing as failing is the part of learning. 
# 
# * Starting today, i am going to log my progress in this kernel, this kernel will be work in progress and will contain hint about a lot of things which i tried or will try in upcoming days.
# 
# * This is an inference kernel where CV and LB score of various models and various image related tricks will be tracked, so that it will be easier to keep track of things which worked and which didn't work for this competition. By default i will not share any training code and keep models private, if you want the training kernel or models to be public feel free to mention in the comments section.
# 
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**

# ## Versions History
# 
# * [version3](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=32756285): 
#     * `model_tried`: se_resnext50_32x4d
#     * `folds`: 3 folds ensemble
#     * `image`: full image resized (224x224)
#     * `LB`: 0.66
# 
# 
# * [version4](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36118508):
#     * `model_tried`: se_resnext50_32x4d
#     * `folds`: submitted on fold 0
#     * `image`: tiles of size (16x256x256)
#     * `CV`: 0.7177
#     * `LB`: 0.67
# 
# 
# * [version5](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36122740):
#     * `model_tried`: se_resnext50_32x4d
#     * `folds`: 5 folds ensemble
#     * `image`: tiles of size (16x256x256)
#     * `CV`: [ fold0: 0.717, fold1: 0.674, fold2: 0.692, fold3: 0.704, fold4: 0.695  ]
#     * `LB`: 0.70
# 
# * [version6](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36131890): 
#     * `model_tried`: efficientnet-B3
#     * `folds`: 2 folds ensemble (fold 0 and fold 1)
#     * `image`: tiles of size (16x256x256)
#     * `CV`: [ fold0: 0.719, fold1: 0.726 ]
#     * `LB`: 0.74   
# 
# * [version7](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36150737):
#     * `model_tried`: efficientnet-B3
#     * `folds`: 5 folds ensemble
#     * `image`: tiles of size (16x256x256)
#     * `CV`: [ fold0: 0.719, fold1: 0.726, fold2: 0.729, fold3: 0.719, fold4: 0.732 ]
#     * `LB`: 0.73
# 
# 
# * [version8](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36230754): 
#     * `model_tried`: efficientnet-B3
#     * `folds`: fold 0
#     * `image`: tiles of size (16x256x256) [image resized to 512x512]
#     * `CV`: [ fold0: 0.749]
#     * `LB`: 0.81
# 
# * [version9](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36242789): 
#     * `model_tried`: efficientnet-B3
#     * `folds`: ensemble of fold 0 and fold 1
#     * `image`: tiles of size (16x256x256) [image resized to 512x512]
#     * `CV`: [ fold0: 0.749, fold1: 0.7509]
#     * `LB`: 0.81
# 
# * [version10](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36282650): 
#     * `model_tried`: efficientnet-B3
#     * `folds`: ensemble of all 5 folds
#     * `image`: tiles of size (16x256x256) [image resized to 512x512]
#     * `CV`: [ fold0: 0.749, fold1: 0.7509, fold2: 0.7592, fold3: 0.7362, fold4: 0.7676]
#     * `LB`: 0.84
# 
# * [version11](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=36741869): 
#     * `model_tried`: efficientnet-B3
#     * `folds`: ensemble of all folds [0,1,2,4]
#     * `image`: tiles of size (16x256x256) [image resized to 512x512]
#     * `CV`: [ fold0: 0.749, fold1: 0.7509, fold2: 0.7592,fold4: 0.7676]
#     * `LB`: 0.85
#     
# * [version12](https://www.kaggle.com/rohitsingh9990/panda-inference-ensemble-trying-various-models?scriptVersionId=37080102): 
#     * `model_tried`: efficientnet-B0
#     * `folds`: fold1
#     * `image`: tiles of size (9x256x256) [image resized to 512x512]
#     * `CV`: [fold1: 0.7751]
#     * `LB`: 0.69
# 
# * version13: 
#     * `model_tried`: efficientnet-B3
#     * `folds`: fold1
#     * `image`: tiles of size (16x256x256) [image resized to 768x768]
#     * `CV`: [fold1: 0.7549]
#     * `LB`: ??
# 
# ## Best LB score (before current version)
# 
# * Best LB score is from version11 which is 0.85
# 

# In[ ]:


import os
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn.functional as F
import os

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.utils.model_zoo as model_zoo
import cv2
from sklearn.metrics import cohen_kappa_score

import openslide
import skimage.io
import random
from sklearn.metrics import cohen_kappa_score
import albumentations
# General packages

from PIL import Image


# * For EDA and visualizations, Please visit https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization/comments
# 
# * For Simple inference using Resnext50 please visit https://www.kaggle.com/rohitsingh9990/panda-resnext-inference

# In[ ]:


#https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/158726


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'apt-get install -y -q xarchiver || true\n# SAVED_MODEL_PATH="../input/eff-b0-16-768/efficientnetB0_0.pth"\n\nSAVED_MODEL_PATH="../input/eff-b0-16-768"\nNEW_MODEL_PATH="/kaggle/working"\nfor model_file in $(ls $SAVED_MODEL_PATH/*.pth)\ndo\n    just_filename=$(basename "${model_file%.*}")\n    if [[ ! -e "$NEW_MODEL_PATH/$just_filename.pth" ]]; then\n        echo "Copying $model_file to $NEW_MODEL_PATH"\n        cp $model_file /tmp\n        cd /tmp/\n        mkdir -p $just_filename\n        echo 3 > $just_filename/version\n        echo ""\n        zip -u  $just_filename.pth $just_filename/version\n        echo "Moving model file $just_filename.pth to $NEW_MODEL_PATH"\n        mv $just_filename.pth $NEW_MODEL_PATH/$just_filename.pth\n    fi\n    echo "(After) Contents of \'$just_filename/version\' in the model file \'$model_file\'"\n    unzip -p $NEW_MODEL_PATH/$(basename $model_file) $just_filename/version\ndone')


# ## Config

# In[ ]:


class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    TEST_BATCH_SIZE = 16
    CLASSES = 6
    # In order to check weather your submission will work or not on test data simply set DEBUG = True
    DEBUG = False


# ## Loading Data

# In[ ]:


BASE_PATH = '../input/prostate-cancer-grade-assessment'

data_dir = f'{BASE_PATH}/test_images'
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

if config.DEBUG:
    data_dir = f'{BASE_PATH}/train_images'
    test = pd.read_csv(f'{BASE_PATH}/train.csv').head(200)


# In[ ]:


test.head()


# In[ ]:


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)


# ## EfficientNet-B3 Model

# In[ ]:


from efficientnet_pytorch import EfficientNet 


# In[ ]:


class EfficientNetB3(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB3, self).__init__()
        if pretrained == True:
            self.model = EfficientNet.from_name('efficientnet-b3')
            self.model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth'))
        else:
            self.model = EfficientNet.from_pretrained(None)            

        in_features = self.model._fc.in_features
        self.l0 = nn.Linear(in_features, config.CLASSES)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0
    
class EfficientNetB0(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB0, self).__init__()
        if pretrained == True:            
            self.model = EfficientNet.from_name('efficientnet-b0')
            self.model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))
        else:
            self.model = EfficientNet.from_pretrained(None)            

        in_features = self.model._fc.in_features
        self.l0 = nn.Linear(in_features, config.CLASSES)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0


# ## Dataset

# In[ ]:


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles


# In[ ]:


def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  


# In[ ]:


tile_size = 256
image_size = 256
n_tiles = 16
batch_size = 2


# In[ ]:


class PANDADataset(Dataset):
    def __init__(self,
            df,
            image_size,
            n_tiles=n_tiles,
            tile_mode=0,
            rand=False,
        ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        # we are in validation part
        self.aug = albumentations.Compose([
            albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True)
        ])

    def __len__(self):
        return self.df.shape[0]
    

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiff_file = os.path.join(data_dir, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, OK = get_tiles(image, self.tile_mode)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1+image_size] = this_img

                
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        
        img = images
        img = np.transpose(img, (1, 2, 0)) # orig image has shape(3,1024, 1024), converting to (1024, 1024, 3)
        img = 1 - img
        img = cv2.resize(img, (768, 768))
        write_image(f'{img_id}.png', img)
        
        # loading image
        
        img = skimage.io.MultiImage(f'{img_id}.png')[-1]
#         img = cv2.resize(img[-1], (512, 512))

        img = Image.fromarray(img).convert("RGB")
        img = self.aug(image=np.array(img))["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        

        return { 'image': torch.tensor(img, dtype=torch.float) }


# In[ ]:


ENSEMBLES = [
    {
        'model_name': 'efficientnet-b0',
        'model_weight': 'efficientnetB0_0.pth',
        'ensemble_weight': 1 
    }
]


# In[ ]:


device = config.device
models = []
for ensemble in ENSEMBLES:
    model = EfficientNetB3(pretrained=True)
    model.load_state_dict(torch.load(ensemble['model_weight'], map_location=device))
    model.to(device)
    models.append(model)


# In[ ]:


def check_for_images_dir():
    if config.DEBUG:
        return os.path.exists('../input/prostate-cancer-grade-assessment/train_images')
    else:
        return os.path.exists('../input/prostate-cancer-grade-assessment/test_images')
        


# ## Inference

# In[ ]:


model.eval()
predictions = []

if check_for_images_dir():
  
    test_dataset = PANDADataset(
        df=test,
        image_size=image_size,
        n_tiles=n_tiles,
        tile_mode=0
    )


    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
    )
    
    for model in models:
        preds = []
        for idx, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            inputs = d["image"]
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            preds.append(outputs.to('cpu').numpy())
                    
        predictions.append(np.concatenate(preds))
    predictions = np.mean(predictions, axis=0)
    predictions = predictions.argmax(1)


# ## Save results

# In[ ]:


if config.DEBUG:
    def quadratic_weighted_kappa(y_hat, y):
        return cohen_kappa_score(y_hat, y, weights='quadratic')

    count = 0
    for index, val in enumerate(test.isup_grade.values):
        if predictions[index] == val:
            count += 1

    print(f"Accuracy Train is {(count / test.shape[0])* 100}")
    print(f'Kappa Train is {quadratic_weighted_kappa(predictions, test.isup_grade.values)}')
else:
    if len(predictions) > 0:
        submission.isup_grade = predictions
    submission.isup_grade = submission['isup_grade'].astype(int)
    submission.to_csv('submission.csv',index=False)
    print(submission.head())     


# ## References:
# 
# * https://www.kaggle.com/iafoss/panda-16x128x128-tiles

# # END NOTES
# I will keep on updating this kernel with my new findings and learning in order to help everyone who has just started in this competition.
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**
