#!/usr/bin/env python
# coding: utf-8

# # T3h d0g mak3rz GAN 
# 
# ##### Authors: Shay Guterman, Yogev Heskia and Eviatar Levy
# 
# ________________________________
# This is not our submission, but rather the model we used to evaluate our GAN and analyse it, best parameters and approaches were chosen and submitted. Also, our best score was calculated after the 187 epochs.
# 
# ### What worked for us?
# 
# **Making the network larger**  Got us from 60 to ~55. The Architectures that works for us has ~18 million parameters for the generator and  ~17 million for the discriminator.
# <brr>**Adding FID calculations to the training**  We noticed  large fluctuations in FID between following epochs, probably due to the unstable nature of GANS and the use of cyclic learning rate so knowing when to stop your GAN is very important. In addition, we are confined to 3 submission a day and it is critical to be able to calculate the FID through your training to get better understanding of the network behavior .
# <brr>**Voodo** We don't know why, but using model.train() before the learning module reduced some of the fluctuations in the training process and improved our score. 
# <br> Implementation of those 2 steps took us from ~55 to ~50.
# <brr>**Weight initialization** At first we implemented weight initialization with nn.init.normal_(m.weight.data, 0.0, 0.02) , the results were quite similar to our results without it. When we changed it to nn.init.normal_(m.weight.data, 0.0, 0.15-1) we improve our score to ~46.
# <brr> **Other parameters and observations** Try different seeds, some work better than others. Higher beta gets you smaller fluctuations in FID score, however gets slightly worse results.   
# <brr>
# **Other things we implemented** ColorJitter augmentation, small dropout.
# 
# 
# ### What didn't work for us?
# 
# **Post selection** We've tried to use the discriminator to improve our submission. Meaning, to generate 10,000 images that the discriminator labels them close to real images of dogs. I think this approach is valid, however I think we didn't understand the loss function good enough, and didnt have enough time to explore the idea
# <brr>
# Adding convolution layers, random search over many parameters,Larger network, CGANS (altough we really didn't try very hard).
# 
# 
# 
# ___________________
# Of course Nirjhar Roy deserves A lot of Credit: https://www.kaggle.com/phoenix9032/gan-dogs-starter-24-jul-custom-layers
# If you upvote this kernel and haven't upvoted his, please do. 
# 
# 
# 

# ## content Table and kernel flow
# 
# 1. ** Imports ** <br>
# 2. ** Conf**<br>
# 3. ** Pytorch initializations**<br>
# 4. **Data generator**<br>
# 5. **Defining Neural Nets**<br>
# > **5.1 Generator**<br>
# > **5.2 Discriminator**<br>
# > **5.3 Weight initialization**<br>
# 6. **Functions **<br>
# > **6.1 Image show functions**<br>
# > **6.2 Truncate function**<br>
# 7. **Calc FID**
# 8. **Train Module**<br>
# 9. **Define and train the GAN**<br>
# 10. **Plot FID**<br>
# 

# ## 1. Imports 
# 

# In[ ]:


import shutil
import os
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm_notebook as tqdm
from scipy.stats import truncnorm




# ## 2. Conf

# In[ ]:


# Define RGB images
CHANNEL = {'RGB': 3, 'BW': 1}
IMG_CHANNEL = CHANNEL['RGB']

# Defining path to dog images 
# Yogev, maybe add the annotions also the same way?
DATA_LOCATION = '../input/generative-dog-images/all-dogs/all-dogs/'

# Define batch size
BATCH_SIZE = 32

# Each image in the generator starts from LATENT_DIM points generated from a some noise distribution 
LATENT_DIM = 128

# The amount of parameters in the networks scales with those number
CONV_DEPTHS_G = 100
CONV_DEPTHS_D = 160


# Learning rate functions
# The learning scheduale 
LEARNING_RATE_G = 0.00025 # was 0.0003 
LEARNING_RATE_D = 0.00015 # was 0.0001
BETA_1 = 0.503

T0_interval = 100
ETA_MIN = 0.00003

# Amount of epochs, EPOCHS/T0_interval has to be integer

EPOCHS = 300

# Start submitting from epoch
SUBMIT_START = 45
# Submit every # epochs
# SUBMIT_INTERVAL = T0_interval/2
SUBMIT_INTERVAL = 10

# Define limit for training time
TIME_FOR_TRAIN = 31000 # 8.3 hours (in seconds)
# TIME_FOR_TRAIN = 7 * 60 * 60 # hours
# TIME_FOR_TRAIN = 5 * 60


real_label = 0.6
fake_label = 0.0


# ### LR scheduale - cosine annealing 
# 
# Relevant paper: https://arxiv.org/pdf/1608.03983.pdf <br>
# Also explained in: https://sidravi1.github.io/blog/2018/04/25/playing-with-sgdr
# <brr>
# <br>
# \begin{align}
# \eta_t = \eta_{min}^i + \frac{1}{2}(\eta_{max}^i - \eta_{min}^i)(1 + \cos(\frac{T_{cur}}{T_i} \pi))
# \end{align}
# 
# ![](https://sidravi1.github.io/assets/2014_04_25_sgdr_schedule.png)

# ## 3. Pytorch initializations

# In[ ]:


def seed_everything(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## 4. Data generator
# 
# Alongside data augmentation

# In[ ]:


class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples
        self.samples = self._load_subfolders_images(directory)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}".format(directory))

    def _load_subfolders_images(self, root):
        IMG_EXTENSIONS = (
            '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        def is_valid_file(x):
            return torchvision.datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)

        
        
        required_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.CenterCrop(64),
        ])
        imgs = []
        paths = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            if is_valid_file(path):
                # Load image
                img = torchvision.datasets.folder.default_loader(path)

                # Get bounding boxes
                annotation_basename = os.path.splitext(os.path.basename(path))[0]
                annotation_dirname = next(
                    dirname for dirname in os.listdir('../input/generative-dog-images/annotation/Annotation/') if
                    dirname.startswith(annotation_basename.split('_')[0]))
                annotation_filename = os.path.join('../input/generative-dog-images/annotation/Annotation/',
                                                   annotation_dirname, annotation_basename)
                tree = ET.parse(annotation_filename)
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    w = np.min((xmax - xmin, ymax - ymin))
                    bbox = (xmin, ymin, xmin + w, ymin + w)
                    object_img = required_transforms(img.crop(bbox))
                    # object_img = object_img.resize((64,64), Image.ANTIALIAS)
                    imgs.append(object_img)
        return imgs

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample)

    def __len__(self):
        return len(self.samples)
    
# torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.2),
                                            torchvision.transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=False, fillcolor=0),    
                                            torchvision.transforms.ColorJitter(brightness=0.04, contrast=0.03, saturation=0, hue=0),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_gen = DataGenerator(DATA_LOCATION, transform=transform, n_samples=85000)
train_loader = torch.utils.data.DataLoader(data_gen, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)


# ## 5. Defining Neural Nets

# ### 5.1 Generator

# In[ ]:


class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
    
    
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, nfeats, nchannels):  #was nz
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(torch.nn.ConvTranspose2d(latent_dim, nfeats * 8, 4, 1, 0, bias=False)) #was nz

        self.conv2 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))

        self.conv3 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))

        self.conv4 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))

        self.conv5 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))

        self.conv6 = spectral_norm(torch.nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        self.pixnorm = PixelwiseNorm()

    def forward(self, x):

        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.Dropout(0.03)(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.Dropout(0.03)(x)
        x = self.pixnorm(x)        
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.nn.Dropout(0.03)(x)
        x = self.pixnorm(x)
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))

        return x


# ### 5.2 Discriminator

# In[ ]:


class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y


# In[ ]:


class Discriminator(torch.nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = torch.nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32

        self.conv2 = spectral_norm(torch.nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = torch.nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16

        self.conv3 = spectral_norm(torch.nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = torch.nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8

        self.conv4 = spectral_norm(torch.nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = torch.nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()

        self.conv5 = spectral_norm(torch.nn.Conv2d(nfeats * 8 + 1, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.1)
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        # x= self.conv5(x)
        return x.view(-1, 1)


# ### 5.3 Weight initialization
# 

# In[ ]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)        
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ## 6. Functions 

# ### 6.1 Image show functions

# In[ ]:


def show_generated_img_all():
    gen_z = torch.randn(32, LATENT_DIM, 1, 1, device=device)
    gen_images = netG(gen_z).to("cpu").clone().detach()
    gen_images = gen_images.numpy().transpose(0, 2, 3, 1)
    gen_images = (gen_images + 1.0) / 2.0
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_images):
        ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
    # plt.savefig(filename)


def show_generated_img():
    row_num = 1
    col_num = 10 
    gen_z = torch.randn(row_num * col_num , LATENT_DIM, 1, 1, device=device)
    gen_images = netG(gen_z).to("cpu").clone().detach()
    gen_images = gen_images.numpy().transpose(0, 2, 3, 1)
    gen_images = (gen_images + 1.0) / 2.0
    fig = plt.figure(figsize=(20, 4))
    for ii, img in enumerate(gen_images):
        ax = fig.add_subplot(row_num, col_num, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
    plt.show()
    # plt.savefig(filename)
    


# ### 6.2 Truncate function

# In[ ]:


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


# ## 7. Calc FID
# 
# Calc FID functions

# In[ ]:


from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import warnings
from tqdm import tqdm
from PIL import Image

class KernelEvalException(Exception):
    pass

model_params = {
    'Inception': {
        'name': 'Inception', 
        'imsize': 64,
        'output_layer': 'Pretrained_Net/pool_3:0', 
        'input_layer': 'Pretrained_Net/ExpandDims:0',
        'output_shape': 2048,
        'cosine_distance_eps': 0.1
        }
}

def create_model_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')

def _get_model_layer(sess, model_name):
    # layername = 'Pretrained_Net/final_layer/Mean:0'
    layername = model_params[model_name]['output_layer']
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return layer

def get_activations(images, sess, model_name, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_model_layer(sess, model_name)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size + 1
    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
                    
        batch = images[start:end]
        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})
        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])
    if verbose:
        print(" done")
    return pred_arr


# def calculate_memorization_distance(features1, features2):
#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
#     neigh.fit(features2) 
#     d, _ = neigh.kneighbors(features1, return_distance=True)
#     print('d.shape=',d.shape)
#     return np.mean(d)

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def cosine_distance(features1, features2):
    # print('rows of zeros in features1 = ',sum(np.sum(features1, axis=1) == 0))
    # print('rows of zeros in features2 = ',sum(np.sum(features2, axis=1) == 0))
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    print('d.shape=',d.shape)
    print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)
    mean_min_d = np.mean(np.min(d, axis=1))
    print('distance=',mean_min_d)
    return mean_min_d


def distance_thresholding(d, eps):
    if d < eps:
        return d
    else:
        return 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))

    print('covmean.shape=',covmean.shape)
    # tr_covmean = tf.linalg.trace(covmean)

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, model_name, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act
    
def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    imsize = model_params[model_name]['imsize']

    # In production we don't resize input images. This is just for demo purpose. 
    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])
    m, s, features = calculate_activation_statistics(x, sess, model_name)
    del x #clean up memory
    return m, s, features

# check for image size
def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 64, is_check_png = False):
    im = Image.open(str(filename))
    if is_checksize and im.size != (check_imsize,check_imsize):
        raise KernelEvalException('The images are not of size '+str(check_imsize))
    
    if is_check_png and im.format != 'PNG':
        raise KernelEvalException('Only PNG images should be submitted.')

    if resize_to is None:
        return im
    else:
        return im.resize((resize_to,resize_to),Image.ANTIALIAS)

def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):
    ''' Calculates the KID of two paths. '''
    tf.reset_default_graph()
    create_model_graph(str(model_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1, features1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)
        if len(mm) != 0:
            m2 = mm
            s2 = ss
            features2 = ff
        elif feature_path is None:
            m2, s2, features2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)
        else:
            with np.load(feature_path) as f:
                m2, s2, features2 = f['m'], f['s'], f['features']

        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))
        print('starting calculating FID')
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('done with FID, starting distance calculation')
        distance = cosine_distance(features1, features2)        
        return fid_value, distance, m2, s2, features2
    

    
    
import zipfile
ComputeLB = False


# ## 8. Train Module

# In[ ]:


def train_module(epochs):
    FID_list = []
    epoch_list = []
    step = 0
    start = time()
    for epoch in range(epochs):
        for ii, (real_images) in enumerate(train_loader):
            end = time()
            if (end - start) > TIME_FOR_TRAIN:
                break
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, 1), real_label, device=device) + np.random.uniform(-0.1, 0.1)

            output = netD(real_images)
            errD_real = criterion(output, labels)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(fake_label) + np.random.uniform(0, 0.2)
            output = netD(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labels.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if step % 500 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch + 1, EPOCHS, ii, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                valid_image = netG(fixed_noise)
            step += 1
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)
            
        if epoch % 5 == 0:
            show_generated_img()
            print(end - start)
            
        if epoch>=SUBMIT_START and epoch%SUBMIT_INTERVAL==0:
            netG.eval()
            if not os.path.exists('../output_images'):
                os.mkdir('../output_images')
            im_batch_size = 50
            n_images = 10000
            for i_batch in range(0, n_images, im_batch_size):
                z = truncated_normal((im_batch_size, LATENT_DIM, 1, 1), threshold=1)
                gen_z = torch.from_numpy(z).float().to(device)
                # gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
                gen_images = netG(gen_z)
                images = gen_images.to("cpu").clone().detach()
                images = images.numpy().transpose(0, 2, 3, 1)
                for i_image in range(gen_images.size(0)):
                    torchvision.utils.save_image((gen_images[i_image, :, :, :] + 1.0) / 2.0,
                                                 os.path.join('../output_images', f'image_{i_batch + i_image:05d}.png'))
            shutil.make_archive('images', 'zip', '../output_images')
            show_generated_img_all()
            # UNCOMPRESS OUR IMGAES
            with zipfile.ZipFile("../working/images.zip","r") as z:
                z.extractall("../tmp/images2/")

            # COMPUTE LB SCORE
            m2 = []; s2 =[]; f2 = []
            user_images_unzipped_path = '../tmp/images2/'
            images_path = [user_images_unzipped_path,'../input/generative-dog-images/all-dogs/all-dogs/']
            public_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'
            fid_epsilon = 10e-15
            fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)
            distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])
            print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",
                    fid_value_public /(distance_public + fid_epsilon))
            FID_list.append(fid_value_public)
            epoch_list.append(epoch)

            netG.train()

        if (end - start) > TIME_FOR_TRAIN:
            break
    return epoch_list,FID_list


# ## 9. Define and train the GAN

# In[ ]:


netG = Generator(LATENT_DIM, CONV_DEPTHS_G, IMG_CHANNEL).to(device)
netD = Discriminator(IMG_CHANNEL, CONV_DEPTHS_D).to(device)   # was 3 here

netG.train()
netD.train()


weights_init(netG)
weights_init(netD)

print("Generator parameters:    ", sum(p.numel() for p in netG.parameters() if p.requires_grad))
print("Discriminator parameters:", sum(p.numel() for p in netD.parameters() if p.requires_grad))

criterion = torch.nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, 0.999))
lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
                                                                     T_0= T0_interval, eta_min=ETA_MIN)
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,

                                                                     T_0= T0_interval, eta_min=ETA_MIN)
fixed_noise = torch.randn(25, LATENT_DIM, 1, 1, device=device)
batch_size = train_loader.batch_size
# criterion = nn.MSELoss()
epoch_list,FID_list =  train_module(EPOCHS)


# ## 10. Plot FID again
# 
# Just implemented this, Hopefully it will work

# In[ ]:


import matplotlib.pyplot as plt
epoch_num = np.array(epoch_list)
fid_score = np.array(FID_list)
plt.plot(epoch_num,fid_score,linewidth= 4.0)
plt.title('Results',fontsize=18)
plt.ylabel('FID score',fontsize=14)
plt.xlabel('# Epchs',fontsize=14)
plt.grid()

print(f" Best fid = {best_fid}")
best_fid = np.min(fid_score)

plt.show()

