#!/usr/bin/env python
# coding: utf-8

# # Embeddings: grouping similar faces automagically
# This notebook demonstrates how embeddings can be used to group similar faces automagically. The idea is to develop this code into a validation strategy that ensures all actors (similar faces) are all in either train or validation sets.

# In[ ]:


get_ipython().system('pip install facenet_pytorch')
get_ipython().system('pip install pretrainedmodels')


# In[ ]:


import os
import pretrainedmodels
import pretrainedmodels.utils as utils
from shutil import copyfile
os.environ['TORCH_HOME'] = '/kaggle/working/pretrained-model-weights-pytorch'

def copy_weights(model_name):
    found = False
    for dirname, _, filenames in os.walk('/kaggle/input/'):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            if filename.startswith(model_name):
                found = True
                break
        if found:
            break
            
    base_dir = "/kaggle/working/pretrained-model-weights-pytorch/checkpoints"
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.basename(full_path)
    copyfile(full_path, os.path.join(base_dir, filename))
    
copy_weights('xception')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm.auto import tqdm
from time import time
import shutil


# # 1. Extract first face from all sample videos

# In[ ]:


list_files = [str(x) for x in Path('/kaggle/input/deepfake-detection-challenge/test_videos').glob('*.mp4')] +              [str(x) for x in Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos').glob('*.mp4')]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, select_largest=False, device=device, min_face_size = 60)


# In[ ]:


def save_frame(file, folder):
    reader = cv2.VideoCapture(file)
    _, image = reader.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(image)
    boxes, probs = mtcnn.detect(pilimg)
    if boxes is None:
        return
    if len(boxes) > 0:
        try:
            best_index = probs.argmax()
            box = [int(x) for x in boxes[best_index].tolist()]
            face_image = image[box[1]:box[3], box[0]:box[2]]
            pilface = Image.fromarray(face_image)
            imgfile = f'{Path(file).stem}.jpg'
            pilface.save(Path(folder)/imgfile)
        except:
            return


# In[ ]:


folder = '/kaggle/working/faces'
Path(folder).mkdir(parents=True, exist_ok=True)
for file in tqdm(list_files):
    save_frame(file, folder)

face_files = [str(x) for x in Path(folder).glob('*')]


# # 2. Calculate embedding vectors from all images
# Here I'm using one of my pre-trained models, as the point of this notebook is not training/generating fake/real predictions, but group similar faces using embeddings.

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')\nmodel.eval();\nnum_ftrs = model.last_linear.in_features\nmodel.last_linear = nn.Linear(num_ftrs, 2)\nmodel = model.to(device)\n\ns = torch.load('/kaggle/input/deepfakemodels/london.pt', map_location=device)\nmodel.load_state_dict(s)\nmodel.eval();\n\ntf_img = utils.TransformImage(model)")


# ### The magic happens here
# This cell below calculates a forward pass through almost all NN. It stops in the 2nd last fully connected layer. The output for each image is a vector with 2048 dimensions. Later, the algorithm will find similar faces by grouping these vectors, finding which are closest to each other.

# In[ ]:


def embeddings(model, input):
    f = model.features(input)
    x = nn.ReLU(inplace=True)(f)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    return x


# In[ ]:


list_embs = []
for face in tqdm(face_files):
    t = tf_img(Image.open(face)).to(device)
    e = embeddings(model, t.unsqueeze(0)).squeeze().cpu().detach().numpy().tolist()
    list_embs.append(e)


# In[ ]:


df = pd.DataFrame({'faces': face_files, 'embeddings': list_embs})
df['videos'] = df['faces'].apply(lambda x: f'{Path(x).stem}.mp4')
df = df[['videos', 'faces', 'embeddings']]
df.head()


# # 4. Get similar faces using Spotify's Annoy

# In[ ]:


from annoy import AnnoyIndex

f = len(df['embeddings'][0])
t = AnnoyIndex(f, metric='euclidean')
ntree = 50

for i, vector in enumerate(df['embeddings']):
    t.add_item(i, vector)
_  = t.build(ntree)


# In[ ]:


def get_similar_images_annoy(img_index):
    t0 = time()
    v, f  = df.iloc[img_index, [0, 1]]
    similar_img_ids = t.get_nns_by_item(img_index, 8)
    return v, f, df.iloc[similar_img_ids]


# In[ ]:


sample_idx = np.random.choice(len(df))  # 166, # 302
v, f, s = get_similar_images_annoy(sample_idx)


# In[ ]:


fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 6)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[0, 5])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[1, 3])
ax8 = fig.add_subplot(gs[1, 4])
ax9 = fig.add_subplot(gs[1, 5])
axx = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
list_plot = [face_files[sample_idx]] + s['faces'].values.tolist()
for i, ax in enumerate(axx):
    ax.imshow(plt.imread(list_plot[i]))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


# # All these frames are from different videos!
# # From a given reference (image on the right), the algorithm finds the 8 most similar faces in all samples (8 small images on the left)!

# To be continued...

# In[ ]:


shutil.rmtree(Path('/kaggle/working/faces'))


# # References:
# - https://blog.usejournal.com/fastai-image-similarity-search-pytorch-hooks-spotifys-annoy-9161bf517aaf
# - https://towardsdatascience.com/similar-images-recommendations-using-fastai-and-annoy-16d6ceb3b809
# - https://towardsdatascience.com/finding-similar-images-using-deep-learning-and-locality-sensitive-hashing-9528afee02f5

# In[ ]:




