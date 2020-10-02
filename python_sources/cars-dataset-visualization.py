#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install console_progressbar')


# In[ ]:


# This preprocessing portion of the code is provided by foamliu on his github repo
# https://github.com/foamliu/Car-Recognition/blob/master/pre-process.py

import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
import shutil
import random
from console_progressbar import ProgressBar
import seaborn as sns
import time


# In[ ]:


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def save_train_data(fnames, labels, bboxes):
    src_folder ='../input/stanford-cars-dataset/cars_train/cars_train/'
    num_samples = len(fnames)

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        if i in train_indexes:
            dst_folder = '/kaggle/working/data/train/'
        else:
            dst_folder = '/kaggle/working/data/valid/'

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


# In[ ]:


def save_test_data(fnames, bboxes):
    src_folder = '../input/stanford-cars-dataset/cars_test/cars_test/'
    dst_folder = '/kaggle/working/data/test/'
    num_samples = len(fnames)

    pb = ProgressBar(total=100, prefix='Save test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


# In[ ]:


def process_train_data():
    print("Processing train data...")
    cars_annos = scipy.io.loadmat('../input/cars-devkit/cars_train_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    class_ids = []
    bboxes = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        labels.append('%04d' % (class_id,))
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        class_ids.append(class_id)
        fnames.append(fname)

    labels_count = np.unique(class_ids).shape[0]
    print(np.unique(class_ids))
    print('The number of different cars is %d' % labels_count)

    save_train_data(fnames, labels, bboxes)


# In[ ]:


def process_test_data():
    print("Processing test data...")
    cars_annos = scipy.io.loadmat('../input/cars-devkit/cars_test_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = annotation[0][4][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    save_test_data(fnames, bboxes)


# In[ ]:


img_width, img_height = 224, 224

cars_meta = scipy.io.loadmat('../input/cars-devkit/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)
print('class_names.shape: ' + str(class_names.shape))
print('Sample class_name: [{}]'.format(class_names[8][0][0]))

ensure_folder('/kaggle/working/data/train')
ensure_folder('/kaggle/working/data/valid')
ensure_folder('/kaggle/working/data/test')

process_train_data()
process_test_data()


# In[ ]:


import torchvision
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


# tot_images_viz = no. of images that you want to visualize from each class
# num_class = no. of classes that you want to visualize

tot_images_viz,num_classes,img_size = 10, 16, 224

train_im_dir = './data/train/'
img_paths = [train_im_dir + str(i).zfill(4) + '/' + fn for i in range(1,5)                  for fn in os.listdir(train_im_dir + str(i).zfill(4))[:tot_images_viz] ]    
    
img_dict = {}
for i in range(1,num_classes+1):
    img_dict[i] = list(map(lambda x: train_im_dir + str(i).zfill(4) + '/' + x,
                      os.listdir(train_im_dir + str(i).zfill(4))[:tot_images_viz]))

        


# In[ ]:


def im2tensor(file_names,bs=num_classes):
    all_im = torch.zeros((bs,3,img_size,img_size))
    custom_transform = transforms.Compose([transforms.Resize((img_size, img_size)),                                           
                                           transforms.ToTensor()])
    for i,fn in enumerate(file_names):
        all_im[i,:,:,:] = (custom_transform(Image.open(fn)))
        
    return all_im


# In[ ]:


fig = plt.figure(figsize=(8,8))
imgs = []
class_name_list = scipy.io.loadmat('../input/stanford-cars-dataset/cars_annos.mat')['class_names'].flatten()
class_name_list = list(map(lambda x: x[0],class_name_list))

for i in range(0,tot_images_viz):
    fn = [img_dict[j][i] for j in range(1,num_classes+1)]
    all_im = im2tensor(fn)
    
    grid = torchvision.utils.make_grid(all_im,4)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im = plt.imshow(im, animated=True) 
    imgs.append([im])

ani = animation.ArtistAnimation(fig, imgs, interval=1000, blit=True,
                                repeat_delay=1000)
plt.tight_layout()
plt.title('Visualizing different classes')

ani.save('car_images.gif')


# In[ ]:


all_img_dict = {}
num_class_visualize,class_count = 196,[]
for i in range(1,num_class_visualize+1):
    class_count.append(len(os.listdir(train_im_dir + str(i).zfill(4))))

class_count = np.array(class_count)
class_count_df = pd.DataFrame(class_count,columns=['class_count'])
print(class_count_df.describe())
print('Class with min count',class_count.argmin())
print('Class with max count',class_count.argmax())


# In[ ]:


sns.countplot(class_count_df.class_count)
plt.title('Distibution of class counts')


# In[ ]:


all_img_dict = {}
num_class_visualize = 10
for i in range(1,num_class_visualize+1):
    all_img_dict[i] = list(map(lambda x: train_im_dir + str(i).zfill(4) + '/' + x,
                      os.listdir(train_im_dir + str(i).zfill(4))))


# In[ ]:


images = []
labels = []
resized_img_size = 100

for class_name,file_names in all_img_dict.items():
        for fn in file_names:
            image = np.array(Image.open(fn)).flatten()
            image = cv2.resize(image,(resized_img_size,
                                      resized_img_size))
    
            images.append(image.flatten())
            labels.append(class_name)

        
images = np.array(images)
print(images.shape)
labels = np.array(labels)


# In[ ]:


feat_cols = [ 'pixel'+str(i) for i in range(resized_img_size**2) ]

df = pd.DataFrame(images,columns=feat_cols)
df['y'] = labels
df['label'] = df['y'].apply(lambda i: class_name_list[i])

X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)

rndperm = np.random.permutation(df.shape[0])


# In[ ]:


pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[ ]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", num_class_visualize),
    data=df,
#     legend="full",
    alpha=1
)
plt.legend(class_name_list)


# In[ ]:


ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"], 
    ys=df.loc[rndperm,:]["pca-two"], 
    zs=df.loc[rndperm,:]["pca-three"], 
    c=df.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


# In[ ]:


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=10000)
tsne_results = tsne.fit_transform(df[feat_cols].values)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", num_class_visualize),
    data=df,
    alpha=1
)
plt.legend(class_name_list)


# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('rm -rf data/')

