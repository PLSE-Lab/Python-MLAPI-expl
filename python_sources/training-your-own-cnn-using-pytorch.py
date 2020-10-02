#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this kernel we will go stepwise, through each process, understanding intuition behind each step.
# 
# ## Index:
# + [Imports](#imports)
# + [Some Useful Functions](#someUsefulFunctions)
# + [Data Preprocessing](#dataPrep)
#     - [Data Normalisation](#dataNorm)
#     - [Data Augmentation](#dataAugmentation)
# + [Visualizing Classes](#visClasses) : using t-SNE to visualize dataset in 3D.
# + [Making Data Loader](#makingDataLoader)
# + [Making CNN](#makingCNN)
# + [Training Network](#trainNet)
#     - [LR Annealing](#lrAnnealing)
#     - [Defining Fit Method](#defFitMethod)
#     - [Fitting to model](#fitting)
# + [Final Checking](#finalCheck)
# + [Convolution in action](#convInAction) : Taking one image through a few kernels from each layer of Convolution Network and look at output.
# + [Getting predictions and making submission](#gettingPreds)
# 

# # Imports <a id="imports"></a>
# ---

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torch
from torchvision import transforms


# # Some Useful Functions <a id="someUsefulFunctions"></a>
# ---

# Some of the functions are taken from Fast.ai library ([here](https://github.com/fastai/fastai)).

# In[ ]:


def to_np(v):
    """
    Converts Variable or Tensor to numpy array.
    -------------------------------------------------------
    Parameters:
        v: A pyTorch Variable or Tensor
    Output:
        Return a numpy array
    """
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if torch.cuda.is_available():
        if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

def score(model, x, y=None, ret=0):
    """
    Get either predicted probabilities or predictions or total right predictions 
    for r = 1, 2, 0 respectively.
    ---------------------------------------------------------------------------------
    Parameters:
        model: A Neural Network Model
        x: A Variable to be sent to model() function to get predictions
        y: (default=None) Actual Labels. If y not given ret can only be 1 or 2.
        ret: A parameter to tell what to return. '1' for predicted probanilities,
            '2' for predictioins and '0' for total right predicions and predicted
            probabilities.
    """
    y_pred = model(x)
    if ret == 1:
        return to_np(y_pred)  # Numpy array of probabilities (size : batch_size x 10)
    elif ret == 2:
        return to_np(y_pred).argmax(1) # Numpy array of predictions
    else:
        ypred_argmax = to_np(y_pred).argmax(axis=1)
        return np.sum(ypred_argmax == to_np(y)), y_pred # Total correct predctions and pytorch Variable of probabilities
    
def accuracy(preds, targs):
    """
    To calculate accuracy of predicted values.
    -----------------------------------------------
    Parameters:
        preds: A Variable with predicted probabilities
        targs: Variable of Actual predictions
    Output:
        Return accuracy of model.
    """
    preds = torch.max(preds, dim=1)[1]
    return (preds==targs).float().mean()

def show(img, title=None):
    """
    Function to plot a single image.
    -------------------------------------------------
    Parameters:
        img: A numpy array of image
        title: Title for plot
    Output:
        Plots image using matplotlib.
    """
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)

def plots(ims, figsize=(12, 6), rows=2, titles=None):
    """
    Plot multiple images in diff. subplots, configured by parameter "rows".
    ----------------------------------------------------------------------------
    Parameters:
        ims: An numpy array of arrays of diff images
        figsize: parameter to be passed to matplotlib "plt" function
        rows: number of rows in plot for subplots
        titles: Array of titles for all images
    Output:
        Plot a matplotlib plot with (r*c) subplots
    """
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


# # Data Preprocessing <a id="dataPrep"></a>
# ---

# In[ ]:


PATH = "../input/"


# In[ ]:


train = pd.read_csv(f'{PATH}train.csv')
test = pd.read_csv(f'{PATH}test.csv')


# In[ ]:


test.shape


# Shuffling data and splitting into training and validation set:

# In[ ]:


idx = np.random.permutation(train.shape[0])
valid_idx, train_idx = idx[:10000], idx[10000:]
xtrain, xvalid = train.iloc[train_idx,:].values, train.iloc[valid_idx,:].values

ytrain, yvalid = xtrain[:, 0], xvalid[:,0]
xtrain, xvalid = xtrain[:, 1:], xvalid[:, 1:]


# In[ ]:


plt.hist(train.iloc[:,0])


# We have nearly same quantity of all categories. Thats good.
# 
# But overall data is still less. We will use **Data Augmentation** to get more images from the existing ones.

# ### Data Normalization: <a id="dataNorm"></a>

# Data normalization in case of CNN and images helps because it makes convergence faster. 
# 
# Also we will use BatchNormalization inside our network. As training progresses and we update parameters to different extents, we lose this normalization, which slows down training and these changes amplifies as the network becomes deeper. We will see BatchNorm later. 
# 
# 

# In[ ]:


mean = xtrain.mean()
std = xtrain.std()


# In[ ]:


# Normalization
xtrain = (xtrain-mean)/std
xvalid = (xvalid-mean)/std


# ### Reshape images:

# In[ ]:


# Reshape to image dimensions.

xtrain_imgs = np.reshape(xtrain, (-1, 1, 28, 28))
xtrain_imgs.shape


# ## Data Augmentation: <a id="dataAugmentation"></a>

# Data Augmentation is a very important step in our training if we don't have enough data. It helps us create more data from existing data. It helps in **reducing over-fitting**.

# In[ ]:


import cv2
# cv2.setUseOptimized(True); cv2.useOptimized()

def rotate_cv(im_arr):
    """
    Randomly rotate array of images using opencv.
    ---------------------------------------------------
    Parameters:
        im_arr: array of numpy array of images.
    Output:
        Return array of numpy arrays of rotated images.
    """
    output = []
    degs = np.random.randint(-20, 20, len(im_arr) )
    for i in range(len(im_arr)):
        img = np.reshape(im_arr[i], (28, 28) )
        r,c,*_ = img.shape
        M = cv2.getRotationMatrix2D((c//2,r//2),degs[i],1)  # Rotating max of 20 deg in clockwise or anti-clockwise
        out = cv2.warpAffine(img,M,(c,r))
        output.append(out)
    return np.array(output)

def translate_cv(im_arr):
    """
    Randomly translate array of images using opencv.
    ---------------------------------------------------
    Parameters:
        im_arr: array of numpy array of images.
    Output:
        Return array of numpy arrays of translated images.
    """
    output = []
    pos = np.random.randint(-7, 7, len(im_arr))
    for i in range(len(im_arr)):
        img = np.reshape(im_arr[i], (28, 28) )
        r, c, *_ = img.shape
        M = np.float32([[1, 0, pos[i]],[0, 1, 0]])        # Moving max 7 pixels left or right 
        dst = cv2.warpAffine(img, M, (c,r) )
        output.append(dst)
    return np.array(output)

def gaussBlur_cv(im_arr):
    """
    Add Gaussian Blur randomly to array of images using opencv.
    ---------------------------------------------------
    Parameters:
        im_arr: array of numpy array of images.
    Output:
        Return array of numpy arrays of Blurred images.
    """
    output = []
    for i in range(len(im_arr)):
        img = np.reshape(im_arr[i], (28, 28) )
        y = int(np.random.randint(3)*0.5)
        if y: dst = cv2.GaussianBlur(img, (0, 0), 2)
        else: dst = img
        output.append(dst)
    return np.array(output)

def sobel_cv(im_arr):
    """
    Filter with Sobel randomly to array of images using opencv.
    ---------------------------------------------------
    Parameters:
        im_arr: array of numpy array of images.
    Output:
        Return array of numpy arrays of Filtered images.
    """
    output = []
    for i in range(len(im_arr)):
        img = np.reshape(im_arr[i], (28, 28) )
        y = int(np.random.randint(3)*0.5)
        if y: dst = cv2.Sobel(img, cv2.CV_8U, 1, 0)
        else: dst = img
        output.append(dst)
    return np.array(output)

def dilate_cv(im_arr):
    """
    Morphological dilation randomly to array of images using opencv.
    ---------------------------------------------------
    Parameters:
        im_arr: array of numpy array of images.
    Output:
        Return array of numpy arrays of Morphed images.
    """
    output = []
    for i in range(len(im_arr)):
        img = np.reshape(im_arr[i], (28, 28) )
        y = int(np.random.randint(3)*0.5)
        kernel = np.ones((3,3), np.uint8)
        if y: dst = cv2.dilate(img, kernel, iterations=1)
        else: dst = img
        output.append(dst)
    return np.array(output)


# In[ ]:


get_ipython().run_line_magic('time', 'xtrain_rotated = rotate_cv(xtrain_imgs[25:35]); plots(xtrain_rotated)')


# In[ ]:


get_ipython().run_line_magic('time', 'xtrain_trans = translate_cv(xtrain_imgs[0:10]); plots(xtrain_trans)')


# In[ ]:


#%time xtrain_blur = gaussBlur_cv(xtrain_imgs[0:10]); plots(xtrain_blur) # Not using it


# In[ ]:


#%time xtrain_sobel = sobel_cv(xtrain_imgs[0:10]); plots(xtrain_sobel) # Not using it


# In[ ]:


get_ipython().run_line_magic('time', 'xtrain_dilate = dilate_cv(xtrain_imgs[0:10]); plots(xtrain_dilate)')


# # Visualizing Classes (t-SNE): <a id="visClasses"></a>
# 
# t-SNE is a very popular technique for dimentionality reduction. It is mostly used for visualization of data in 2D or 3D.  
# ( [L.J.P. van der Maaten and G.E. Hinton; 9(Nov):2579--2605, 2008](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) )
# 
# We will use sklearn's implementation of t-SNE.
# For that we will take 5000 images and labels and reduce their dimentionality from 28x28 to just 3 (!). We will see if there is some difference between the classes in higher dimension, i.e. are points of same category clustered together or not and are they away from other categories or not. That is what t-SNE does. It effectively tries to project a higher dimensinal data to lower dimensions.
# 
# And to plot the points we will use plotly to plot all points in an interactive 3D space.

# In[ ]:


from sklearn.manifold import TSNE
from time import time

X = xtrain[0:5000]
Y = ytrain[0:5000]

import plotly
plotly.offline.init_notebook_mode(connected=True)

import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go

def plot_tSNE(pt, Y, title):
    """
    To plot a 3D plot using plotly. (For MNIST data only)
    ------------------------------------------------------------------------------
    Parameters:
        pt: points in 3 dimensions with corresponding labels/names in "Y"
        Y: labels/names for each point in "pt"
        title: For title of plot
    Output:
        plots a 3D interactive graph of points in "pt" having labels in "Y"
    """
    data = []
    for i in range(10):
        index = (Y == i)
        trace = go.Scatter3d(x=pt[index, 0], y=pt[index, 1], z=pt[index, 2], mode='markers',
                          marker=dict(size=6, line=dict(color=plt.cm.Set1(i / 10.)), opacity=0.97),
                          text = [f'{i}'], name=f'{i}', hoverinfo='name')
        data.append(trace)
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0), title=title)
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=title)


# In[ ]:


t0 = time()
get_ipython().run_line_magic('time', "pt = TSNE(n_components=3, n_iter=5000, init='pca', random_state=0).fit_transform(X)")
plot_tSNE(pt, Y,"t-SNE plot(time %.2fs)" %(time() - t0))


# We can clearly see the distinction between the classes even in 3 dimensions. As you can see most of the points belonging to same class are nearby.

# # Making a DataLoader: <a id="makingDataLoader"></a>
# ___

# When it comes to DataLoaders, one important thing to check is the batch size being used. A trade off between batch size and accuracy can be seen because as we increase batch size we will get more standardized error and we might not be able to learn as effectively. 
# 
# For example we might not learn some features which is present in many but still that number is quite less than sample size. Through a larger batch size we might not be able to learn those features. 
# 
# 
# ![Imgur](https://i.imgur.com/VL4PbHQ.png)
# Source: [https://www.researchgate.net/figure/Performance-comparison-CNN-for-different-batch-size-in-CI-FAR10-Testing-accuracy-for_fig24_312593963](https://www.researchgate.net/figure/Performance-comparison-CNN-for-different-batch-size-in-CI-FAR10-Testing-accuracy-for_fig24_312593963)
# 
# 
# As we decrese batch size we learn more and more features and faster. And we get a better accuracy in the end.
# 

# But there is one more trade off we have to look at. Actually two:
# 1.  If we decrease batch size too much (say, to 1) we will have to calculate loss for every single data point and we will have to backpropagate too for every one of them. That will be too time consuming. So, there is a time vs batch size trade off.
# 2. Secondly, if we decrease batch size too much (say, to 1) [SGD] we don't get a smooth learning curve. i.e. the network tries to learn every data point and moves in its direction and moves in next data points direction, then next and then next ... (And we might end at a sub optimal ans). So, mini-Batch Gradient Descent fares well most of the times and gives better result than both, as we learn those features in this case too and that too smoothly moving towards the global minimum. With mini batch gradient descent we can also get more speed because then we will be able to send data in matrices to GPU.
# 
# SGD on every data point:
# ![Imgur](https://i.imgur.com/MiiMhFq.png)
# Source: [http://ruder.io/optimizing-gradient-descent/](http://ruder.io/optimizing-gradient-descent/)

# In[ ]:


class DataLoader():
    def __init__(self, xt, yt, batch_size=32):
        """
        Data Loader for training set. Outputs Variable of batch_size images and 
        Variable of batch_size categories on each next() call on generator for
        len(xt)/batch_size calls.
        -------------------------------------------------------------------------------
        Parameters:
            xt: numpy array of images in training set
            yt: numpy array of categories in training set
            batch_size: (default=32) number of images and categories to output per call
        Output:
            Return Variable of batch_size images and a Variable of batch_size categories per call
            for len(xt)/batch_size calls.
        """
        self.xt, self.yt = xt, yt
        # Apply select augmentation to about 1/3rd of images
        #self.xt = dilate_cv(self.xt)
        self.bs = batch_size
    def __iter__(self):
        xt_trans = translate_cv(dilate_cv(self.xt))
        xt_rot = rotate_cv(xt_trans)
        xout, yout = [], []
        for i in range(len(self.xt)):
            xout.append(xt_rot[i])
            yout.append(self.yt[i])
            if len(xout) == self.bs:
                yield ( Variable(torch.cuda.DoubleTensor(np.array(xout, dtype="float32"))),
                                    Variable(torch.cuda.LongTensor(np.array(yout, dtype="float32"))) )
                xout, yout = [], []
        if len(xout) > 0: yield ( Variable(torch.cuda.DoubleTensor(np.array(xout, dtype="float32"))),
                                    Variable(torch.cuda.LongTensor(np.array(yout, dtype="float32"))) )
    def __len__(self):
        if len(self.xt)%self.bs==0 : return len(self.xt)//self.bs
        else : return len(self.xt)//self.bs+1


# In[ ]:


get_ipython().run_line_magic('time', 'dl = DataLoader(xtrain_imgs, ytrain)')


# In[ ]:


class valGenerator():
    def __init__(self, xv, yv, batch_size=500):
        """
        Randomly output Variable of batch_size images and Variable of batch_size categories per 
        next() call on generator.
        ----------------------------------------------------------------------------------------
        Parameters:
            xv: Array containing numpy arrays of diff. images
            yv: Array containing categories of images in xv
            batch_size: (default=500) parameter to control number of images and categories to output
        Output:
            Return Variable of batch_size images and Variable of batch_size categories per 
            next() call on generator.
        """
        self.data = np.zeros((len(xv), 28*28+1))
        self.bs = batch_size
        self.data[:, 0] = yv
        self.data[:, 1:785] = xv
    def __iter__(self):
        while True:
            idxs = np.random.permutation(len(self.data))[0:self.bs]
            yield (Variable(torch.cuda.DoubleTensor(self.data[idxs, 1:785].reshape((-1, 1, 28, 28)))), 
                 Variable(torch.cuda.LongTensor(self.data[idxs, 0])) )


# In[ ]:


vgen = valGenerator(xvalid, yvalid)


# # Making Convolutional Neural Network: <a id="makingCNN"></a>
# ---

# In a convolution neural network we try to learn some matricies, which are called kernels, convolutional matrix or mask. These kernels are features present in the images. It can be some border, some shape, or even some complex parts like nose, eyes etc.
# ![Imgur](https://i.imgur.com/KRiTBaX.png)
# Source: [Isma Hadji, Richard P. Wildes arXiv:1803.08834](https://arxiv.org/pdf/1803.08834.pdf)
# 

# These kernels are matched with image on every loaction to check if that feature is present there or not. Only when that feature is present, matrix multiplication with that kernel and matrix of some location in image gives a high number as ouput, which is saved in a location corresponding to location of matrix in input image in output of that convolutional layer (a matrix).  We wil see some kernels in action in the end.
# ![Imgur](https://i.imgur.com/McTaE78.jpg)
# Source: [http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html](http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html)

# **Fully Connected Layers:**
# We want to categorize our inputs and thats why we have to move to fully connected layers, to eventually have only 10 output nodes, which will tell us the probability of every category by the end of training. 
# One might think how is our network able to tell probability? Basically, it is because of the presence of softmax layer in the end which does two things:
# 1.  As it uses exponential function inside, it makes bigger numberes more bigger. By which we get contrasted values in output. Good for categorization.
# 1.  We divide by sum of all outputs, which gives a feel of probability. But, as the machine becomes more accurate, it goes closer and closer to actual probabilities.

# **ReLU : **
# ReLU is a key part in contribution of immense power of a deep neural networks. Without it or without some other similar activation functions, our NN will just be a Linear Regression problem with lots of parameters. These activation functions are the parts Deep Neural Networks which makes them fly and also gives them non linearity, and a sufficiently deep neural network can fit any function, any function at all.
# 
# Something to read: [http://neuralnetworksanddeeplearning.com/chap4.html](http://neuralnetworksanddeeplearning.com/chap4.html)

# **Dropout : ** This is basically to prevent overfitting. When we have less data and we don't want to overfit, dropout can come in handy.
# It is a technique where randomly selected neurons are dropped during training. This means that their contribution to the activation of downstream neurons is temporally set to zero on the forward propagation and any weight updates are not applied to the neuron on the back propagation.
# 
# Source: [https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

# **MaxPool : ** This and some other similar layers actually makes our matrix smaller and smaller and less complex, and saving only the important features with its location.

# In[ ]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(0.20),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=0.5),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        #print(x.size())       # Checking the size of output given out, for updating the input size of first layer in fc.
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        return self.fc(x)


# ### Why  Batch Normalization:
# **Issues With Training Deep Neural Networks : **
# 1. Internal Covariate shift:
# 
#     Covariate shift refers to the change in the input distribution to a learning system. In the case of deep networks, the input to each layer is affected by parameters in all the input layers. So even small changes to the network get amplified down the network. This leads to change in the input distribution to internal layers of the deep network and is known as internal covariate shift.
# It is well established that networks converge faster if the inputs have been whitened (ie zero mean, unit variances) and are uncorrelated and internal covariate shift leads to just the opposite.
# 
# 1. Vanishing Gradient:
# 
#       Saturating nonlinearities (like tanh or sigmoid) can not be used for deep networks as they tend to get stuck in the saturation region as the network grows deeper. Some ways around this are to use:
# 
#     * Nonlinearities like ReLU which do not saturate
#     * Smaller learning rates
#     * Careful initializations
#     
# Source:  [https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0), [Sergey Ioffe, Christian Szegedy arXiv:1502.03167](https://arxiv.org/pdf/1502.03167.pdf)
# 
# Some graphs on performance of Networks with and without Batch Normalization: 
# ![Imgur](https://i.imgur.com/D9QMn1x.png)

# ![Imgur](https://i.imgur.com/NdPuSu0.png)

# Though BatchNormalization incurs some time penalty, still it fares well than Non-normalized data:
# 
# 
# ![Imgur](https://i.imgur.com/aCXovq6.png)
# Source: [https://towardsdatascience.com/how-to-use-batch-normalization-with-tensorflow-and-tf-keras-to-train-deep-neural-networks-faster-60ba4d054b73](https://towardsdatascience.com/how-to-use-batch-normalization-with-tensorflow-and-tf-keras-to-train-deep-neural-networks-faster-60ba4d054b73)

# # Training Network: <a id="trainNet"></a>
# ---

# For training a network three things are important to set,
# 1.  **Loss Function** : which calculates the difference between current output and actual output. Here, CrossEntropyLoss is used, which is commonly used for Multi-class Classification problems.
# 1.  **Metrics** : A fuction to calculate viability of network (How robust is our network). This function can be Accuracy, F1 score etc.
# 1.  **Optimizer** : It is a function which updates our parameters, different optimizers use different techniques to update parameters in a way, so that our network learns faster and gives optimal result.

# In[ ]:


#loss = nn.CrossEntropyLoss()
metrics = [accuracy]
#opt = optim.Adam(net.parameters(), weight_decay=1e-4)
#opt = optim.SGD(net.parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)


# Configurations I used:
# 1.  Adam : with default lr( = 1 ) and different weight decays **: :** 
# Seems to coverge at suboptimal solution (about 98%), although learning was fast. Actually Adam automatically reduces learning rate for the features which are most occuring.
# 1.  SGD : with lr(= 1) and momentum and Learning Rate Annealing **: :** 
# Really bad choice; was not learning much. Actually SGD with learning rate annealing fares well in many cases, than all other optimizers, if a decent learning rate is chosen.
# 1. SGD : with  lr(=1e-5) and momentum and Learning Rate Annealing **: :** 
# Really slow on learning.
# Learning rate of 1e-3 and 1e-4 seemed to give good results.
# 1. SGD : with lr(=1e-3) and momentum and Learnig Rate Annealing and nesterov = True **: :** 
# What nesterov = True means is that it is using NAG algorithm, which doen't just move into the slope gaining speed (this is what momentum does), it looks forward and corrects its position too. 
# 
# Something to read : [http://ruder.io/optimizing-gradient-descent/](http://ruder.io/optimizing-gradient-descent)
# 
# SGD with momentum, NAG and learning rate annealing seemed to give the best results for this CNN. (atleast from the optimizers and parameters I tried.)

# In[ ]:


def set_lrs(opt, lr):
    """
    Set learning rate in each layer equal to lr.
    ----------------------------------------------------
    Parameters:
        opt: Optimizer for which lr is to be set
        lr: new learning rate for each layer
    """
    for pg in opt.param_groups: pg['lr'] = lr


# <a id="lrAnnealing"></a>
# There can be many types of annealing functions, here I have used learning rate annealing with restarts :
# 
# **Learning Rate Annealing : ** Learning rate annealing basically reduces learning rate of model when it has stopped learning. Here I have implemented a simple one. It reduces learning rate as time passes.
# 
# How is this helpful? As the dimentionality increases, our Network will have exponentially more local minimas, and they are actually nearly at same level and some worse than the others. 
# 
# Why? Because some can be steep and others a plateau, still at same level. So, if we change our weights even by small amounts in a steep minima, we might get really bad results. But if we are in a plateau at same level, a more general solution, we will get nearlly the same answer even if we get a little bit here or there. So, in LR Annealing with restarts, learning rate is increased from time to time so that we get out of steep minimas and hopefully converge to more general ans.
# 
# Source: [I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with restarts.
# arXiv preprint arXiv:1608.03983, 2016](https://arxiv.org/pdf/1608.03983)
# 

# In[ ]:


class lrAnnealing():
    def __init__(self, ini_lr, epochs, itr_per_epoch, mult_dec=False):
        """
        Class to Anneal learning rate with warm restarts with time. It decreases 
        learning rate as multiple cosine waves with dec. amplitudes.1e-5 is taken 
        as zero. (The lower point for cosine)
        ---------------------------------------------------------------------------
        Parameters:
            ini_lr: Initial learning rate
            epochs: Number of epochs
            itr_per_epoch: iterations per epoch
            mult_dec: T/F, If to use Annealing with warm restarts or hard
        """
        self.epochs = epochs
        self.ipe = itr_per_epoch
        self.m_dec = mult_dec
        self.ppw = (self.ipe * self.epochs) // 4    # Points per wave of cosine (For 4 waves per fit method)
        self.count = 0
        self.lr = ini_lr
        self.values = np.cos(np.linspace(np.arccos(self.lr), np.arccos(1e-5), self.ppw))
        self.mult = 1
    def __call__(self, opt):
        self.count += 1
        set_lrs(opt, self.values[self.count-1]*self.mult)
        if self.count == len(self.values):
            self.count = 0
            if self.m_dec: self.mult /= 2


# ### LRs trend with number of iterations will look something like this:

# In[ ]:


y = np.concatenate([np.cos(np.arange(0,1,.01)),
                   np.cos(np.arange(.3, 1, .01)),
                   np.cos(np.arange(.5, 1, .01))])


# In[ ]:


x = np.array(np.arange(0, 220, 1))


# In[ ]:


plt.plot(x, y)


# ### Defining Fit method: <a id="defFitMethod"></a>

# Here these are the steps essential for learning:
# 1.  **Predict** (Put input values throgh model, and get output);
# 1. ** Calculate Loss** (by loss function, gives out current loss);
# 1.  **Calculate gradients of loss with respect to model parameters** (loss.backward);
# 1.  **Update Parameters** (optimizer.step());

# In[ ]:


def fit(model, lr,  train_dl, valGen, n_epochs, crit, opt, metrics, annln=True, mult_dec=False):
    """
    Function to fit the model to training set and print F1 scores for both training set
    and validation set.
    -------------------------------------------------------------------------------------
    Parameters:
        model: Model (Neural Network) to which Training set will fit
        lr: Learning rate (initil learning rate if annln=True)
        train_dl: Train DataLoader which loads training data in batches (should give Tensors as output)
        valGen: Validation DataLoader which loads validation data in batches (")
        n_epochs: number of epochs
        loss: Loss function to calculate and backpropagate loss (eg: CrossEntropy)
        opt: Optimizer, to update weights (eg: RMSprop)
        metrics: Function to calculate score of model (eg: accuracy, F1 score)
        annln: (default=True) If to use LRAnnealing or not
        mult_dec: (default=True) If to dec. max Learning rate on every cosine cycle
    """
    if annln: annl = lrAnnealing(lr, 40, 500, mult_dec=mult_dec)  # itr_per_epoch = len(xtrain) // batch_size
    for epoch in range(n_epochs):
        losses = []
        vl = iter(valGen)
        dl = iter(train_dl)
        length = len(train_dl)
        for t in range(length):
            # Annealing
            if annln: annl(opt)
            
            xt, yt = next(dl)
            xt = xt.view(-1, 1, 28, 28)
            xt, yt = xt, yt

            # Forward pass: compute predicted y and loss by passing x to model.
            y_pred = model(xt)
            l = crit(y_pred, yt)
            losses.append(l)

            # Before backward pass, use the optimizer object to zero all of the gradients for the variable it will update
            # (which are Learnable weights of the model)
            opt.zero_grad()

            # Backward pass: compute gradient of the loss with respect to the model params
            l.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            opt.step()
        
        xv, yv = next(vl)
        
        yright =[]
        valLosses = []
        mult = len(xv)//100
        for i in range(0, len(xv), 100):
            if i == mult*100:
                a, b = score(model, xv[i:len(xv)], yv[i:len(xv)])
                yright.append(a)
                valLosses.append(crit(b, yv[i:len(xv)]).data.cpu().numpy())
            else:
                a, b = score(model, xv[i:i+100], yv[i:i+100])
                yright.append(a)
                valLosses.append(crit(b, yv[i:i+100]).data.cpu().numpy())
        val_scores = np.sum(yright)/len(xv)
        
        meanValLoss = 0
        for vliter in range(len(valLosses)):
            meanValLoss += valLosses[vliter]
        meanValLoss/=len(valLosses)
        
        mean_loss = 0
        for liter in range(len(losses)):
            mean_loss += losses[liter].data.cpu().numpy()
        mean_loss/=len(losses)
        #print(val_scores, mean_loss, meanValLoss)
        if epoch == 69:
            print("Epoch " + str(epoch) + "::"
                + "  loss: " + str(mean_loss)
                + ", valLoss: " + str(meanValLoss)
                +", valAcc: " + str(val_scores*100))


# ### Fitting to model: <a id="fitting"></a>

# Using Ensemble of 15 CNNs: (method used [here](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist))

# In[ ]:


# Before Ensemble Highest Score was 0.994, and now _____.
nets = [0]*10

for i in range(10):
    nets[i] = ConvNet().cuda().double()
    loss = nn.CrossEntropyLoss()
    
    #opts = [
    #        optim.Adam(nets[i].parameters(), weight_decay=1e-4),
    #        optim.SGD(nets[i].parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True),
    #        optim.RMSprop(nets[i].parameters(), weight_decay=1e-4, momentum=0.9)
    #    ]
    opt =  optim.SGD(nets[i].parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)  
    # opt = opts[np.random.randint(3)]
    
    print(f'{i}th ConvNet:')
    get_ipython().run_line_magic('time', 'fit(nets[i], 1e-2, dl, vgen, 70, loss, opt, metrics, mult_dec=True)')


# # Final Checking <a id="finalCheck"></a>
# ---

# In[ ]:


ximgs, _ = next(iter(vgen))
ximgs = ximgs[0:8]
y_pred = to_np(nets[0](ximgs))
y_pred[0:2].argmax(1)

plots(to_np(ximgs).reshape(-1, 28, 28), titles=score(nets[0], ximgs, ret=2))


# # Lets see convolution in action: <a id="convInAction"></a>

# 4 kernels from every convolutional layer:

# In[ ]:


# Lets print 4 kernels from every convolutional layer
k = []
for m in nets[0].modules():
    if isinstance(m, nn.Conv2d):
        if m.weight.data[0].size(0) > 3:
            k.append(m.weight.data[0][0:5].contiguous().cpu().numpy().reshape(-1, 3, 3))
            plots(m.weight.data[0][0:4].contiguous().cpu().numpy().reshape(-1, 3, 3), rows=1)
        else:
            k.append(m.weight.data[0:5].contiguous().cpu().numpy().reshape(-1, 3, 3))
            plots(m.weight.data[0:4].contiguous().cpu().numpy().reshape(-1, 3, 3), rows=1)


# Lets take one input images and multiply it with some of our kernels and see what we get:

# In[ ]:


# Lets take one input images and multiply it with some of our kernels and see what we get:
final = []
img = np.zeros((5, 28, 28))
for i in range(5): img[i] = to_np(ximgs[0]).reshape((28, 28))
for m in img: final.append(m)

kernel = k[0]
conv = np.zeros((5, 26, 26))

for m in range(5):
    output_img = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):
            output_img[i, j] = max(sum(sum(img[m, i:i+3, j:j+3] * kernel[m])), 0)  # with ReLU 
    final.append(output_img)

plots(final)


# Now lets take them through one of the kernels in second convolutional layer:

# In[ ]:


# Now lets take them through one of the kernels in second convolutional layer:
img = final
final = []
for i in range(5, 10): final.append(img[i])
img = np.zeros((5, 26, 26))
for i in range(5): img[i] = final[i]

kernel = k[1]
conv = np.zeros((5, 24, 24))

for m in range(5):
    output_img = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            output_img[i, j] = max(sum(sum(img[m, i:i+3, j:j+3] * kernel[m])), 0)
    final.append(output_img)

plots(final)


# Now lets take them through one of the kernels in first maxpooling layer:

# In[ ]:


# Now lets take them through one of the kernels in first maxpooling layer:
img = final
final = []
for i in range(5, 10): final.append(img[i])
img = np.zeros((5, 24, 24))
for i in range(5): img[i] = final[i]

conv = np.zeros((5, 23, 23))

for m in range(5):
    output_img = np.zeros((23, 23))
    for i in range(23):
        for j in range(23):
            output_img[i, j] = np.max(img[m, i:i+2, j:j+2])
    final.append(output_img)

plots(final)


# Now lets take them through one of the kernels in next convolutional layer:

# In[ ]:


# Now lets take them through one of the kernels in next convolutional layer
img = final
final = []
for i in range(5, 10): final.append(img[i])
img = np.zeros((5, 23, 23))
for i in range(5): img[i] = final[i]

kernel = k[2]
conv = np.zeros((5, 21, 21))

for m in range(5):
    output_img = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            output_img[i, j] = max(sum(sum(img[m, i:i+3, j:j+3] * kernel[m])), 0) # with ReLU
    final.append(output_img)

plots(final)


# Now lets take them through one of the kernels in next convolutional layer:

# In[ ]:


# Now lets take them through one of the kernels in next convolutional layer
img = final
final = []
for i in range(5, 10): final.append(img[i])
img = np.zeros((5, 21, 21))
for i in range(5): img[i] = final[i]

kernel = k[3]
conv = np.zeros((5, 19, 19))

for m in range(5):
    output_img = np.zeros((19, 19))
    for i in range(19):
        for j in range(19):
            output_img[i, j] = max(sum(sum(img[m, i:i+3, j:j+3] * kernel[m])), 0) # with ReLU
    final.append(output_img)

plots(final)


# Now lets take them through one of the kernels in first maxpooling layer:

# In[ ]:


# Now lets take them through one of the kernels in first maxpooling layer
img = final
final = []
for i in range(5, 10): final.append(img[i])
img = np.zeros((5, 19, 19))
for i in range(5): img[i] = final[i]

conv = np.zeros((5, 18, 18))

for m in range(5):
    output_img = np.zeros((18, 18))
    for i in range(18):
        for j in range(18):
            output_img[i, j] = np.max(img[m, i:i+2, j:j+2])
    final.append(output_img)

plots(final)


# As we can see some matices have retained some information and in others information retained is decreasing. i.e. those particular kernels are not able to find find what they were looking for.
# 
# Here we are using a few kernels only, from each layer. Every layer has many many kerels.
# And as we go deeper and deeper, this becomes more abstract, which thankfully our network will understand and give us most accurate predictions.

# # Getting Predictions <a id="gettingPreds"></a>
# ---

# In[ ]:


test = (test-mean)/std
test = np.resize(test, (28000, 1, 28, 28) )
test = Variable(torch.DoubleTensor(test)).cuda()
yp = np.zeros((28000 ,2), dtype='int'); yp.shape


# Because we only have about 12GB (?) GPU here, we will get output predictions sequentially: (Was getting cuda mem. error)
for i in range(0, 28000, 100):
    prob_sum = np.zeros((100, 10), dtype='float64')
    for j in range(10):
        prob_sum += to_np(nets[j](test[i:i+100]))
    yp[i:i+100] = np.array(np.vstack([np.arange(i+1, i+101), prob_sum.argmax(1)]).T)
    
res = pd.DataFrame(yp, index = np.arange(len(yp)), columns=["ImageId", "Label"])
res.to_csv("submission.csv", index=False)
res.head()

