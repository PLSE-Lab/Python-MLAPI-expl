#!/usr/bin/env python
# coding: utf-8

# **Simple example of transfer learning from pretrained model using PyTorch.**
# * Metrics: f1_score

# In[ ]:


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob

import cv2
import torch
from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms
from skimage import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def kaggle_commit_logger(str_to_log, need_print = True):
    if need_print:
        print(str_to_log)
    os.system('echo ' + str_to_log)


# In[ ]:


import json
with open(r'/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:
    train_data = json.load(json_file)


# In[ ]:


df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                                'category_id': [item['category_id'] for item in train_data['annotations']],
                                'image_id': [item['image_id'] for item in train_data['annotations']],
                                'file_name': [item['file_name'] for item in train_data['images']]})

df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_image = pd.DataFrame.from_records(train_data['images'])

indices = []
for _id in df_image[df_image['location'] == 537]['id'].values:
    indices.append( df_train[ df_train['image_id'] == _id ].index )

for the_index in indices:
    df_train = df_train.drop(df_train.index[the_index])


# In[ ]:


df_train.shape


# In[ ]:


# Image.open('../input/iwildcam-2020-fgvc7/train/8792549a-21bc-11ea-a13a-137349068a90.jpg')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nindices = []\nfor i in df_train['file_name']:\n    try:\n        Image.open('/kaggle/input/iwildcam-2020-fgvc7/train/' + i)\n    except:        \n        print(i)\n        df_train.drop(df_train.loc[df_train['file_name']==i].index, inplace=True)")


# In[ ]:


df_train.shape


# In[ ]:


with open(r'/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as f:
    test_data = json.load(f)


# In[ ]:


df_test = pd.DataFrame.from_records(test_data['images'])
df_test.head()


# In[ ]:


batch_size = 64
IMG_SIZE = 64

N_EPOCHS = 1

ID_COLNAME = 'file_name'
ANSWER_COLNAME = 'category_id'
TRAIN_IMGS_DIR = r'../input/iwildcam-2020-fgvc7/train/'
TEST_IMGS_DIR = r'../input/iwildcam-2020-fgvc7/test/'


# In[ ]:


train_df, test_df = train_test_split(df_train[[ID_COLNAME, ANSWER_COLNAME]],
                                     test_size = 0.15,                                     
                                     shuffle = True
                                    )


# In[ ]:


train_df.head(5)


# In[ ]:


CLASSES_TO_USE = df_train['category_id'].unique()
CLASSES_TO_USE


# In[ ]:


NUM_CLASSES = len(CLASSES_TO_USE)
NUM_CLASSES


# In[ ]:


CLASSMAP = dict(
    [(i, j) for i, j
     in zip(CLASSES_TO_USE, range(NUM_CLASSES))
    ]
)
CLASSMAP


# In[ ]:


REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])
REVERSE_CLASSMAP


# In[ ]:


model = models.densenet121(pretrained='imagenet')


# In[ ]:


new_head = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.classifier = new_head


# In[ ]:


model.load_state_dict(torch.load('../input/iwild2020-torch/model'))


# In[ ]:


model.cuda();


# In[ ]:


normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    normalizer,
])

val_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    normalizer,
])


# In[ ]:


class IMetDataset(Dataset):
    
    def __init__(self,
                 df,
                 images_dir,
                 n_classes = NUM_CLASSES,
                 id_colname = ID_COLNAME,
                 answer_colname = ANSWER_COLNAME,
                 label_dict = CLASSMAP,
                 transforms = None
                ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):        
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_name = img_id # + self.img_ext
        img_path = os.path.join(self.images_dir, img_name)
          
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.answer_colname is not None:              
            label = torch.zeros((self.n_classes,), dtype=torch.float32)
            label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

            return img, label

        else:
            return img, img_id  


# In[ ]:


train_dataset = IMetDataset(train_df, TRAIN_IMGS_DIR, transforms = train_augmentation)
test_dataset = IMetDataset(test_df, TRAIN_IMGS_DIR, transforms = val_augmentation)


# In[ ]:


BS = 24

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)


# In[ ]:


def cuda(x):
    return x.cuda(non_blocking=True)


# In[ ]:


def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


# In[ ]:


def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging = 250):
    model.train();
    
    total_loss = 0.0
    
    train_tqdm = tqdm_notebook(train_loader)
    
    
    for step, (features, targets) in enumerate(train_tqdm):
        try:        
            features, targets = cuda(features), cuda(targets)

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % steps_upd_logging == 0:
                logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
                train_tqdm.set_description(logstr)
                kaggle_commit_logger(logstr, need_print=False)
                
        except:
            pass
        
    return total_loss / (step + 1)


# In[ ]:


def validate(model, valid_loader, criterion, need_tqdm = False):
    model.eval();
    
    test_loss = 0.0
    TH_TO_ACC = 0.5
    
    true_ans_list = []
    preds_cat = []
    
    with torch.no_grad():
        
        if need_tqdm:
            valid_iterator = tqdm_notebook(valid_loader)
        else:
            valid_iterator = valid_loader
        
        for step, (features, targets) in enumerate(valid_iterator):
            features, targets = cuda(features), cuda(targets)

            logits = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(torch.sigmoid(logits))

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)
                
        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    kaggle_commit_logger(logstr)
    return test_loss / (step + 1), f1_eval


# In[ ]:


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nTRAIN_LOGGING_EACH = 500\n\ntrain_losses = []\nvalid_losses = []\nvalid_f1s = []\nbest_model_f1 = 0.0\nbest_model = None\nbest_model_ep = 0\n\nfor epoch in range(1, N_EPOCHS + 1):\n    ep_logstr = f"Starting {epoch} epoch..."\n    kaggle_commit_logger(ep_logstr)\n    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)\n    train_losses.append(tr_loss)\n    tr_loss_logstr = f\'Mean train loss: {round(tr_loss,5)}\'\n    kaggle_commit_logger(tr_loss_logstr)\n\n    valid_loss, valid_f1 = validate(model, test_loader, criterion)  \n    valid_losses.append(valid_loss)    \n    valid_f1s.append(valid_f1)       \n    val_loss_logstr = f\'Mean valid loss: {round(valid_loss,5)}\'\n    kaggle_commit_logger(val_loss_logstr)\n    sheduler.step(valid_loss)\n\n    if valid_f1 >= best_model_f1:    \n        best_model = model        \n        best_model_f1 = valid_f1        \n        best_model_ep = epoch        ')


# In[ ]:


torch.save(best_model.state_dict(), 'model')


# In[ ]:


bestmodel_logstr = f'Best f1 is {round(best_model_f1, 5)} on epoch {best_model_ep}'
kaggle_commit_logger(bestmodel_logstr)


# In[ ]:


xs = list(range(1, len(train_losses) + 1))

plt.plot(xs, train_losses, label = 'Train loss');
# plt.plot(xs, valid_losses, label = 'Val loss');
plt.plot(xs, valid_f1s, label = 'Val f1');
plt.legend();
plt.xticks(xs);
plt.xlabel('Epochs');


# In[ ]:


SAMPLE_SUBMISSION_DF = pd.read_csv(r'../input/iwildcam-2020-fgvc7/sample_submission.csv')
SAMPLE_SUBMISSION_DF.head()


# In[ ]:


SAMPLE_SUBMISSION_DF.rename(columns={'Id':'file_name','Category':'category_id'}, inplace=True)
SAMPLE_SUBMISSION_DF['file_name'] = SAMPLE_SUBMISSION_DF['file_name'] + '.jpg'
SAMPLE_SUBMISSION_DF.head()


# In[ ]:


subm_dataset = IMetDataset(SAMPLE_SUBMISSION_DF,
                           TEST_IMGS_DIR,
                           transforms = val_augmentation,
                           answer_colname=None
                          )


# In[ ]:


SUMB_BS = 48

subm_dataloader = DataLoader(subm_dataset,
                             batch_size=SUMB_BS,
                             shuffle=False,
                             pin_memory=True)


# In[ ]:


def get_subm_answers(model, subm_dataloader, need_tqdm = False):
    model.eval();
    preds_cat = []
    ids = []
    
    with torch.no_grad():
        
        if need_tqdm:
            subm_iterator = tqdm_notebook(subm_dataloader)
        else:
            subm_iterator = subm_dataloader
        
        for step, (features, subm_ids) in enumerate(subm_iterator):
            features = cuda(features)

            logits = model(features)
            preds_cat.append(torch.sigmoid(logits))
            ids += subm_ids

        all_preds = torch.cat(preds_cat)
        all_preds = torch.argmax(all_preds, dim=1).int().cpu().numpy()
    return all_preds, ids


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbest_model.cuda();\n\nsubm_preds, submids = get_subm_answers(best_model, subm_dataloader, True)')


# In[ ]:


len(subm_preds)


# In[ ]:


ans_dict = dict(zip(submids, subm_preds.astype(str)))


# In[ ]:


df_to_process = (
    pd.DataFrame
    .from_dict(ans_dict, orient='index', columns=['Category'])
    .reset_index()
    .rename({'index':'Id'}, axis=1)    
)
df_to_process['Id'] = df_to_process['Id'].map(lambda x: str(x)[:-4])
df_to_process.head()


# In[ ]:


def process_one_id(id_classes_str):
    if id_classes_str:
        return REVERSE_CLASSMAP[int(id_classes_str)]
    else:
        return id_classes_str


# In[ ]:


df_to_process['Category'] = df_to_process['Category'].apply(process_one_id)
df_to_process.head()


# In[ ]:


df_to_process.to_csv('submission.csv', index=False)


# In[ ]:




