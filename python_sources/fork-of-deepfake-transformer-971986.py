#!/usr/bin/env python
# coding: utf-8

# # Transformer Based Model
# 
# I seen several notebooks working with LSTM. This is my inference notebook using Transformer.
# 
# It uses the TRANSFORMER model without position embedding.
# Unfortuantely I came up with this just a couple days ago and wasn't able to properly train it.
# Now I ran out of time and credits on AWS. :D
# 
# I'm also using faces of 128x128 because it needs so much memory for training.
# 
# This was based on the work from Limerobot
# https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891
# 
# It was a lot of fun to work on this competition. Thanks to Kaggle, Google, AWS and many others.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls /kaggle/input/deppfake-montez-2020


# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install facenet-pytorch\n!pip install /kaggle/input/deppfake-montez-2020/facenet_pytorch-2.2.9-py3-none-any.whl\n!pip install /kaggle/input/deppfake-montez-2020/timm-0.1.18-py3-none-any.whl\n!pip install /kaggle/input/deppfake-montez-2020/pytorch_transformers-1.2.0-py3-none-any.whl')


# In[ ]:


import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pytorch_transformers.modeling_bert import BertConfig, BertEncoder
import torch.nn as nn
from facenet_pytorch import MTCNN


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


# Thanks to limerobot: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891

class TransfomerModel(nn.Module):
    def __init__(self, cfg, cnn_model):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        self.n_cnn_features = cfg.n_cnn_features
        self.cnn_model = nn.Sequential(
            cnn_model,
            torch.nn.AdaptiveAvgPool2d(1),
            nn.Dropout(cfg.dropout)
        )

        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config) 
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1), #, cfg.target_size
            nn.Sigmoid()
        )
        self.reg_layer = get_reg()
       
    def forward(self, x, mask):
        batch_size, n_frames, n_channels, height, width = x.size() # e.g. 16 x 10 x 3 x 224 x 224
        assert n_channels == 3, f"Expecting 3 channels but got {n_channels}"
        x = x.reshape(batch_size*n_frames, n_channels, height, width) # (16*10) x 3 x 224 x 224 x = self.cnn_model(x) # (32*64) x 2048
        x = self.cnn_model(x)
        x = torch.squeeze(x)
        #print(x.shape)
        x = x.view(batch_size, n_frames, self.n_cnn_features)
        
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.encoder(x, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
 
        pred_y = self.reg_layer(sequence_output)
        return pred_y


# In[ ]:


class CFG:
    learning_rate=1.0e-4
    batch_size=4
    seq_len=15
    frame_size = 128
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=10
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01
    dropout=0.2
    emb_size=100
    hidden_size=1536
    nlayers=2
    nheads=8
    device=device
    seed=7
    n_cnn_features=1536


# In[ ]:


model = torch.load('/kaggle/input/deppfake-montez-2020/transformer11_train30_2.pkl', map_location=torch.device(device))
model.eval()
print('Model Loaded')


# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)


# In[ ]:


mtcnn = MTCNN(keep_all=True, margin=14, device=device) # margin actually has no effect


# In[ ]:


# get faces from video frames list
def get_faces(vframes):
    cropped_faces = []
    
    with torch.no_grad():
        faces, probs = mtcnn.detect(vframes)

        count = 0
        for j, faces_per_frame in enumerate(faces):
            if faces_per_frame is None:
                continue

            for i, face in enumerate(faces_per_frame):
                x1, x2, y1, y2 = round(face[0]), round(face[2]), round(face[1]), round(face[3])
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if width < 50 or height < 50 or probs[j][i] < 0.90:
                    continue

                face_box = np.zeros(4, dtype=int)
                # width
                face_box[0] = max(x1, 0)
                face_box[2] = min(face_box[0] + width, vframes[j].shape[1])

                # height
                face_box[1] = max(y1, 0)
                face_box[3] = min(face_box[1] + height, vframes[j].shape[0])

                crop_img = vframes[j][face_box[1]:face_box[3], face_box[0]:face_box[2]]
                cropped_faces.append(crop_img)

    return cropped_faces


# In[ ]:


# get faces from filename
def capture_face(name, n_frames1, n_frames2, min_num_faces):
    faces = []
    filename = os.path.join(test_dir, name)
    #print("Analyzing file: " + filename)

    try:
        vframes1 = []
        vframes2 = []
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick evenly spaced frames to sample
        # use 2 sample intervales, preferably prime numbers e.g. 17 and 11
        sample1 = np.linspace(0, v_len - 1, n_frames1).round().astype(int)
        sample2 = np.linspace(0, v_len - 1, n_frames2).round().astype(int)
        for j in range(v_len):
            ret = v_cap.grab()
            if j in sample1:
                ret, vframe = v_cap.retrieve()
                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                vframes1.append(vframe)
            else:
                if j in sample2:
                    ret, vframe = v_cap.retrieve()
                    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                    vframes2.append(vframe)
        v_cap.release()
    
        faces = get_faces(vframes1)
        del vframes1
        
        if (len(faces) < min_num_faces):
            #print('Number of faces smaller than min: {0} < {1}'.format(len(faces), min_num_faces))
            faces2 = get_faces(vframes2)
            del vframes2
            
            faces += faces2
            #print('New number of faces: {0}'.format(len(faces)))
        
    except:
        v_cap.release()

    return faces        


# In[ ]:


from fastai.vision import *
from torch.utils import data

class Dataset_Transform_Test(data.Dataset):
    def __init__(self, files, seq_len, size):
        self.files = files
        self.size = size
        self.seq_len = seq_len

    def get_faces(self, file):
        X = torch.tensor([])
        list_tensors = []
        try:
            faces = capture_face(file, 17, 11, seq_len)
            for face in faces:
                img = pil2tensor(face.astype(np.float32), np.float32) / 255
                img = vision.Image(img)
                img.resize(self.size)
                list_tensors.append(img.data)
                if len(list_tensors) == self.seq_len:
                    break
            if len(list_tensors) > 0:
                X = torch.stack(list_tensors, dim=0)
        except:
            print('failed to get faces for {0}'.format(file))
        return X
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]

        X = torch.FloatTensor(self.seq_len, 3, self.size, self.size).zero_()
        X_temp = self.get_faces(file)
        seq_len = min(self.seq_len, len(X_temp))
        mask = torch.ByteTensor(self.seq_len).zero_()
        if (seq_len > 0):
            X[:seq_len] = X_temp[:seq_len]
            mask[:seq_len] = 1
        else:
            print('mask is empty')
        
        return X.to(device), mask.to(device) #, y.cuda()


# In[ ]:


seq_len = 15
size = 128
batch_size=4
test_db = Dataset_Transform_Test(test_videos, seq_len, size)
#test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False, pin_memory=False)


# In[ ]:


def clip_value(value):
    #print(value)
    return max(min(value, 0.9), 0.1)


# In[ ]:


preds = []
xs = []
masks = []
submission = []
batch_file = []
batch_num = 0
count = 0
for i, (x, mask) in enumerate(test_db):

    if mask[5] == 0:
        submission.append([test_videos[count], 0.5])
        count += 1
        continue
    batch_file.append(test_videos[i])
    xs.append(x)
    masks.append(mask)
    batch_num += 1
    
    # continue until batch or end
    if batch_num != batch_size and i < len(test_db):
        continue
    
    x = torch.stack(xs, dim=0)
    mask = torch.stack(masks, dim=0)
    x, mask = x.to(device), mask.to(device)        

    with torch.no_grad():        
        pred = model(x, mask)
        for j, proc_file in enumerate(batch_file):
            submission.append([test_videos[count], pred[j][0].cpu().item()]) #clip_value(pred[j][0].cpu().item())]) #
            count += 1
    batch_num = 0
    batch_file = []
    xs = []
    masks = []


# In[ ]:


submission = pd.DataFrame(submission, columns=['filename', 'label'])
submission.sort_values('filename').to_csv('submission.csv', index=False)


# In[ ]:


plt.hist(submission.label, 20)
plt.show()
submission

