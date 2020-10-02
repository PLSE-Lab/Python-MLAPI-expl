#!/usr/bin/env python
# coding: utf-8

# # Removing unwanted correlations in training data
# 
# The training data come from different sources, with different image sizes, and characteristics like brightness and hue. As noted in https://www.kaggle.com/taindow/be-careful-what-you-train-on, some of this image metadata - like the height and width - are correlated with the label. The most extreme case is that all 351 train images with dimensions 2048x1536 are class 0. Those correlations don't transfer to the test data (or the real world), so they should be removed so the final model doesn't learn them.
# 
# How to remove the spurious correlations? One could do heavy random augmentation, especially cropping, and hope to "cover" the correlations under the randomness of augmentation. Or one could do more targeted, deterministic preprocessing to remove the correlations, which is what we'll do here. 
# 
# In either case, one would like to test whether one has successfully removed them. To test whether a preprocessing method removes spurious correlations, we'll take the preprocessed images and apply with a transformation that obscures the ground truth - such as turning all foreground pixels white and all background pixels black, or doing a Gaussian blur with large radius. 
# 
# Then, we'll train a simple model to predict which class the image is. If that model performs much better than random, there are still some superficial correlations and we need to do more preprocessing. If we don't manage to train a good model, it's good. We can never have a guarantee that we've removed all correlations which are there on train and are not there on test, but weakening unwanted correlations is already better than nothing.
# 
# To cut down runtime, work with data that's rescaled to 1/4 of the height and 1/4 of the width (keeping the aspect ratio).
# 
# ---
# 
# This is my first public kernel, feedback would be much appreciated! If there is interest, I can publish a version of the train images with this transformation applied. Editing this notebook takes a lot of RAM on my local machine, if anybody know why please let me know...

# # Load and utils

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


train_path='../input/aptos-quarter/aptos_quarter/aptos_quarter/'
test_path = '../input/aptos2019-blindness-detection/test_images/'


# Get train ids+labels as well as test ids

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv').values
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv').values


# In[ ]:


def show_me_some_imgs(n_rows=2, n_columns=2, train=True, preprocess = lambda x:x, idx_subset = None, figsize = (20, 10)):
    """
    show some random images, preprocessed by the parameter preprocess. 
    train: whether to show train or test images.
    idx_subset: if passed, choose at random from subset of train / test set. Else, choose random images from all of train/test.
    """
    if idx_subset is None:
        idx_subset = range(len(train_df)) if train else range(len(test_df))
    fig, axs = plt.subplots(n_rows, n_columns, figsize = figsize)
    if n_rows == 1:
        axs = [axs]
    if n_columns == 1:
        axs = [ [ax] for ax in axs ]
    for row in range(n_rows):
        for column in range(n_columns):
            if train:
                idx = np.random.choice(idx_subset)
                img = preprocess(Image.open(train_path + train_df[idx][0] + ".png"))
                axs[row][column].imshow(img)
                axs[row][column].set_title(f'train img #{idx}. class {train_df[idx][1]}', fontsize=8)
            else:
                idx = np.random.choice(idx_subset)
                img = preprocess(Image.open(test_path + test_df[idx][0] + ".png"))
                axs[row][column].imshow(img)
                axs[row][column].set_title(f'test img #{idx}', fontsize=8)
                
#Test:
show_me_some_imgs()


# # Training Dataset
# 
# Let's figure out and visualize the different sources that the training dataset comes from.

# In[ ]:


train_dim_per_sample = len(train_df) * [None] #list of len(train_df) that has a string denoting the image's size for every image
for i in range(len(train_df)):
    img = Image.open(train_path + train_df[i][0] + '.png')
    train_dim_per_sample[i] = img.size
train_unique_dims = sorted(list(set(train_dim_per_sample)))
#two-step to get sorting right
train_dim_per_sample = [f'{d[0]}x{d[1]}' for d in train_dim_per_sample]
train_unique_dims = [f'{d[0]}x{d[1]}' for d in train_unique_dims]
print(f"List of sizes that occur in training set: \n {train_unique_dims} \nThat's {len(train_unique_dims)} different sizes")


# In[ ]:


train_histogram_per_dim = {}
for train_unique_dim in train_unique_dims:
    train_histogram_per_dim[train_unique_dim] = [0,0,0,0,0]
    for i in range(len(train_df)):
        if train_dim_per_sample[i] == train_unique_dim:
            train_histogram_per_dim[train_unique_dim][train_df[i][1]] += 1
print('The counts of each class, per image size, is:')
print(train_histogram_per_dim)


# We see that some of the per-size class histograms are highly imbalanced. In the real world of course, there is no correlation between image size and degree of retinopathy.
# 
# For later, we'll need a dictionary that maps each size to a list of the images with that size:

# In[ ]:


train_samples_per_dim = {}
for train_unique_dim in train_unique_dims:
    train_samples_per_dim[train_unique_dim] = []
    for i in range(len(train_df)):
        if train_dim_per_sample[i] == train_unique_dim:
            train_samples_per_dim[train_unique_dim].append(i)
print(f"For example, train_samples_per_dim[train_unique_dims[0]] = {train_samples_per_dim[train_unique_dims[0]]}")


# Let's plot some examples of every image size:

# In[ ]:


for i, train_unique_dim in enumerate(train_unique_dims):
    print(f'Size #{i}. dim: {train_unique_dim}. histogram {train_histogram_per_dim[train_unique_dim]}. ' +
         f'{ 100 * sum(train_histogram_per_dim[train_unique_dim])/ len(train_df):.3f}% of the training data')
    show_me_some_imgs(idx_subset = train_samples_per_dim[train_unique_dim], 
                      n_rows = 1, n_columns = 10, figsize = (20, 5))
    plt.show()


# Size #2 seems to come from two different sources, since there are images with two different crops. The rest of the sizes seems more homogenous in characteristics like how tight the crop is, whether there is a rectangle or triangle marker on the right side, the hue, whether there is glare, etc.
# 
# For easier comparison, plot one image per size, with relative histograms, and how much of the data is of that size:

# In[ ]:


fig, axs = plt.subplots(4, 5, figsize = (20, 15))

for i, train_unique_dim in enumerate(train_unique_dims):
    id_code = train_df[np.random.choice(train_samples_per_dim[train_unique_dim])][0]
    relative_histogram = [f"{n / len(train_samples_per_dim[train_unique_dim]):.2f}" for n in train_histogram_per_dim[train_unique_dim]]
    img = Image.open(train_path + id_code + '.png')
    axs[i//5][i%5].imshow(img)
    axs[i//5][i%5].set_title(f'#{i}. {train_unique_dim}. {relative_histogram}. ' +
         f'{ 100 * sum(train_histogram_per_dim[train_unique_dim])/ len(train_df):.2f}%.', fontsize=8)


# keep in mind that size #2 comes from two sources, here we just see one. The most common train size is #3. 
# 
# Note that almost all images that show a full circle (some of size #2, size #3, size #8) are class 0 on train. If we don't remove that information, the final network could learn to always label images that show a full circle as 0, we don't want that.
# 
# Almost all train images show the full width of the circle, with the exception of size #1. However, size #1 is by far the most common on test...

# # Test data
# Let's do the same steps for test, and visualize what sizes and crops there are. Of course, we can't look at class histograms.

# In[ ]:


test_dim_per_sample = len(test_df) * [None]
for i in range(len(test_df)):
    img = Image.open(test_path + test_df[i][0] + '.png')
    test_dim_per_sample[i] = img.size
test_unique_dims = sorted(list(set(test_dim_per_sample)))
#two-step to get sorting right
test_dim_per_sample = [f'{d[0]}x{d[1]}' for d in test_dim_per_sample]
test_unique_dims = [f'{d[0]}x{d[1]}' for d in test_unique_dims]
print(f"List of sizes that occur in test set: \n {test_unique_dims} \nThat's {len(test_unique_dims)} different sizes")


# In[ ]:


test_samples_per_dim = {}
for test_unique_dim in test_unique_dims:
    test_samples_per_dim[test_unique_dim] = []
    for i in range(len(test_df)):
        if test_dim_per_sample[i] == test_unique_dim:
            test_samples_per_dim[test_unique_dim].append(i)
print(f"For example, test_samples_per_dim[test_unique_dims[1]] = {test_samples_per_dim[test_unique_dims[1]]}")


# In[ ]:


for i, unique_dim in enumerate(test_unique_dims):
    print(f'Size #{i}. dim: {unique_dim}. ' +
         f'{ 100 * len(test_samples_per_dim[unique_dim]) / len(test_df):.3f}% of the test data')
    show_me_some_imgs(idx_subset = test_samples_per_dim[unique_dim], train=False,
                      n_rows = 1, n_columns = 10, figsize = (20, 5))
    plt.show()


# By far the most common size is size #0, corresponding to training size #1 (recall that train images are rescaled to 1/4). It is the same crop as training size #1 as well.
# 
# Here is the side by side comparison:

# In[ ]:


fig, axs = plt.subplots(3, 5, figsize = (20, 15))

for i, test_unique_dim in enumerate(test_unique_dims):
    id_code = test_df[np.random.choice(test_samples_per_dim[test_unique_dim])][0]
    img = Image.open(test_path + id_code + '.png')
    axs[i//5][i%5].imshow(img)
    axs[i//5][i%5].set_title(f'#{i}. {test_unique_dim}. ' +
         f'{ 100 * len(test_samples_per_dim[test_unique_dim])/ len(test_df):.2f}%')


# # Visualize transformations

# In[ ]:


def visualize_transform(transform, compare=False, train=True):
    """
    Visualizes a transformation, with one example per original size.
    Parameters:
    - transform: the transformation. Should take a pytorch tensor to a pytorch tensor.
    - compare: if true, show the unmodified and transformed images next to each other
    - train:
    """
    if train:
        unique_dims = train_unique_dims
        samples_per_dim = train_samples_per_dim
        df = train_df
        path = train_path
    else:
        unique_dims = test_unique_dims
        samples_per_dim = test_samples_per_dim
        df = test_df
        path = test_path  
    if compare:
        fig, axs = plt.subplots(8, 5, figsize = (20, 30))
    else:
        fig, axs = plt.subplots(4, 5, figsize = (20, 15))

    for i, unique_dim in enumerate(unique_dims):
        row = 2*(i//5) if compare else i//5
        img_idx = np.random.choice(samples_per_dim[unique_dim])
        id_code = df[img_idx][0]
        img = transforms.ToTensor()(Image.open(path + id_code + '.png'))
        axs[row][i%5].imshow(transforms.ToPILImage()(transform(img)))
        if train:
            axs[row][i%5].set_title(f'#{i}. {unique_dim}. {train_histogram_per_dim[unique_dim]}. ' +
                 f'{ 100 * sum(train_histogram_per_dim[unique_dim])/ len(train_df):.2f}%', fontsize=8)
        else:
            axs[row][i%5].set_title(f'#{i}. {unique_dim}. ' +
                 f'{ 100 * len(samples_per_dim[unique_dim])/ len(df):.2f}%', fontsize=8)
        if compare:
            axs[row+1][i%5].imshow(transforms.ToPILImage()(img))


# # Define Network
# 
# As mentioned above, we'll define a simple neural network (from a pretrained resnet18) that we'll use to test whether we have successfully removed correlations we don't want in the training set.
# 
# We'll define things in two variants: one with target='class', where the task of the NN is to predict the class of each training example. One with target='size', where the task of the NN is to predict the original size.
# 
# The reason for using target='size' is the following: The most common class on train is zero, hence if we use target='class' the network is going to guess zero if in doubt. This is going to lead to high accuracy for example for the images of original size #8, and we won't know if that's because there are still unwanted correlations or if it's just a guess. Using target='size' is more finegrained.

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from tqdm import tqdm_notebook
import cv2
device = torch.device('cuda')

class Aptos_train_ds(Dataset):
    def __init__(self, 
                 train_path, 
                 transform = lambda x:x,
                 target = 'size'):
        self.train_path = train_path
        self.train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv').values
        self.to_tensor = transforms.ToTensor() 
        self.transform=transform
        self.target = target
    def __len__(self):
        return len(self.train_df)
    def __getitem__(self, idx):
        id_code = self.train_df[idx][0]
        img = self.to_tensor(Image.open(train_path + id_code + '.png'))
        x = self.transform(img)
        if self.target == 'size':
            y = train_unique_dims.index(train_dim_per_sample[idx])
        else:
            y = self.train_df[idx][1]
        return x, y

def get_model(target = 'size'):
    model = models.resnet18(pretrained=True)
    if target == 'size':
        print('predicting size')
        model.fc = torch.nn.Linear(512, 17)
    else:
        print('predicting class')
        model.fc = torch.nn.Linear(512, 5)
    model.train()
    return model.to(device)

def get_trainvalloader(train_ds):
    idcs = torch.randperm(len(train_ds))
    train_idcs = idcs[:int(.8*len(train_ds))]
    val_idcs = idcs[int(.8*len(train_ds)):]
    
    train_loader = DataLoader(train_ds, batch_size = 32, num_workers=9, 
                                   sampler = SubsetRandomSampler(train_idcs), pin_memory=True)
    val_loader = DataLoader(train_ds, batch_size = 32, num_workers=9, 
                                   sampler = SubsetRandomSampler(val_idcs), pin_memory=True)
    return train_loader, val_loader, val_idcs.tolist()

def acc_from_data_loader(model, data_loader):
    """
    given a model and a data loader, return the accuracy of the model on the samples in the dataloader
    """
    training = model.training
    model.eval()
    val_preds = []
    val_gts = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            val_preds.append(pred)
            val_gts.append(y)
        val_preds, val_gts = torch.cat(val_preds), torch.cat(val_gts)
        acc = (val_preds == val_gts).to(torch.float).mean()
    model.train(training)
    return acc

def acc_from_subset(model, train_ds, subset):
    """
    Given a model, a dataset and a list of indices, return the accuracy of the model on the elements of the dataset which are contained 
    in the list of indices
    """
    if len(subset)==0:
        return -1
    data_loader = DataLoader(train_ds, batch_size = 32, num_workers=5, 
                                   sampler = SubsetRandomSampler(subset))
    return acc_from_data_loader(model, data_loader)

def train(model, train_loader, val_loader, num_epochs = 1):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        for (x, y) in tqdm_notebook(train_loader):
        #for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'val acc: {acc_from_data_loader(model, val_loader)}')
    
def get_acc_per_dim(transform, num_epochs = 1, target='size'):
    """
    This is the function we'll use. Train a resnet18 to predict the target.
    Then show the accuracy on val, separated by the original size.
    """
    torch.manual_seed(2019)
    torch.backends.cudnn.deterministic=True
    np.random.seed(2019)
    train_ds = Aptos_train_ds(train_path, transform = transform, target=target)
    plt.figure(figsize = (2,2))
    plt.imshow(transforms.ToPILImage()(train_ds[np.random.randint(len(train_ds))][0]))
    plt.show()
    train_dl, val_dl, val_idcs = get_trainvalloader(train_ds)
    model = get_model(target=target)
    train(model, train_dl, val_dl, num_epochs = num_epochs)
    print('Accuracies per dimension on val:')
    for i, dim in enumerate(train_unique_dims):
        num = sum(train_histogram_per_dim[dim])
        rel_hist = [ f"{h/num:.3f}" for h in train_histogram_per_dim[dim] ]
        val_samples_per_dim = list(set(train_samples_per_dim[dim]) & set(val_idcs)) 
            #intersection between val indices and indices of images with current size
        print(f'#{i: <2}. size: {dim: <8}, relative histogram: {rel_hist}, ' + 
              f'fraction of data { 100* num / len(train_df):.1f}%, ' +
              f'acc: {acc_from_subset(model, train_ds, val_samples_per_dim):.3f}')
        
to_128_torch = lambda img: torch.nn.functional.interpolate(img[None, ...], size = (128, 128), mode='bilinear', align_corners = False)[0, ...]


# As a test, see if our model can figure out the original size of resized train images: Acc of -1 means that there are no images of that original size in val. This can happen if there are very few train images of a given size, but those rare sizes we don't care about so much anyway.

# In[ ]:


transform = to_128_torch
get_acc_per_dim(transform, target='size', num_epochs=2)


# It does pretty well, i.e. resizing does not hide the original size of the image. Since the original size is in some cases highly correlated with the label, there is an unwanted correlation here.
# # Removing shape information
# One might guess that the network figures out the original size primarily by looking at what kind of crop we have, i.e., the shape of the foreground. To test that, let's come up with a transformation that makes all of the foreground white and all of the background black.

# In[ ]:


def shapify_torch(img_data):
    height, width = img_data.shape[1:]
    black = img_data[:, :int(height/20), :int(width/20)].mean(dim=(1, 2)) # get average r, g, b of top-left corner as an estimate for black value
    mask = ((img_data - black[:, None, None]).max(dim = 0)[0] > .02).to(img_data.dtype) #note torch's max with argument dim returns (max, argmax)
    return torch.stack((mask, mask, mask))
def shapify_pil(img):
    img_data = transforms.ToTensor()(img) #rgb, height, width. range [0, 1]
    img_data = shapify_torch(img_data)
    return transforms.ToPILImage()(img_data)


# In[ ]:


visualize_transform(shapify_torch, train=True, compare=True)


# Alright, now we can check whether the shape is enough to say which source the image comes from:

# In[ ]:


get_acc_per_dim(transforms.Compose([to_128_torch, shapify_torch]), target='size', num_epochs=2)


# Yes, the shape of the foreground is enough: 94% of validation data are assigned the correct original size after 1 epoch. Just for completeness, let's see if we can predict the class from the shape:

# In[ ]:


get_acc_per_dim(transforms.Compose([to_128_torch, shapify_torch]), target='class', num_epochs=2)


# Yes, to an accuracy that's much better than chance. It doesn't always guess zero either, for size #9 it guesses mostly 2. That's bad. Let's set out to remove the correlation between shape and class from the data.
# 
# First remove black space everywhere:

# In[ ]:


def crop_out_black(img):
    height, width = img.shape[1:]
    black = img[:, :int(height/20), :int(width/20)].mean(dim=(1, 2))
    rowmeans = img.mean(dim=1)
    linemeans = img.mean(dim=2)
    nonblack_rows = ((rowmeans - black[:, None]).max(dim=0)[0] > .02).nonzero()
    nonblack_lines = ((linemeans - black[:, None]).max(dim=0)[0] > .02).nonzero()
    try:
        left, right = nonblack_rows[0].item(), nonblack_rows[-1].item()
        upper, lower = nonblack_lines[0].item(), nonblack_lines[-1].item()
        img = img[:, upper:lower, left:right]
    except:
        print('crop out black didnt work')
    return img


#visualize_transform(crop_out_black, compare=True, train=False)
visualize_transform(crop_out_black, compare=True)


# That seems to work alright. Let's see if that removes the information on the original size from the shapified image:

# In[ ]:


get_acc_per_dim(transforms.Compose([crop_out_black, to_128_torch, shapify_torch]), target='size', num_epochs=2)


# It does not, the network is still very good at figuring out the original size of the train image. We'll also need to crop. 
# 
# It turns out that after removing the black, not all images are quite centered, so to make life easier let's put them back in the center.

# In[ ]:


def center(img):
    # crop such that the center of mass of non-black pixels is roughly in the center
    _, height, width = img.shape
    shapified = shapify_torch(img)[0, ...] #just take one of 3 channels
    nonzero = (shapified).nonzero().to(torch.float)
    center = nonzero.mean(dim = 0).to(torch.int)
    if center[0] > height/2: #center too low, crop from top
        new_height = 2 * (height - center[0])
        img = img[:, -new_height:, :]
    else: #center too high, crop from bottom
        new_height = 2 * center[0]
        img = img[:, :new_height, :]
    if center[1] > width/2: #center too far right, crop from left
        new_width = 2*(width- center[1])
        img = img[:, :, -new_width:]
    else: #center too far left, crop from right
        new_width = 2*center[1]
        img = img[:, :, :new_width]
    return img

transform = transforms.Compose([crop_out_black, center, to_128_torch])
visualize_transform(transform, compare=True)


# Let's think about how to crop. We should crop away some from the top and bottom of the image, since as we saw on train most images that show the full height of the circle are class 0, which is an unwanted correlation.
# 
# Should we crop from the right or left? Most of the test data just shows the center part of the retina, so it makes sense to focus on that and crop train images the same. That's what we'll do here. We're not cropping out too much area, 6% on either side.
# 
# (To get a little better with the part of the test data that shows the full width, it may be worth it to have some train images that do show the full width, for example crop train images to the center part with prob 3/4 and show the full width with prob 1/4.)
# 
# Here is a function that crops a centered image to look like train size #1 / test size #0:

# In[ ]:



def tight_crop(img): #assumes black cropped out and centered
    shapified = shapify_torch(img)[0, ...]
    if shapified.to(torch.float).mean() > .95: #already tight crop
        #print('already tight crop, passing')
        return img
    width = img.shape[2]
    width_margin = int(.06 * width)
    img = img[:, :, width_margin:-width_margin]
    shapified = shapified[ :, width_margin:-width_margin]
    num_white_per_line = shapified.sum(dim=1) / shapified.shape[1]
    white_above_threshold = (num_white_per_line > .9).nonzero()
    upper, lower = white_above_threshold[0], white_above_threshold[-1]
    img = img[:, upper:lower, :]
    return img


transform = transforms.Compose([crop_out_black, center, tight_crop, to_128_torch])
visualize_transform(transform, compare=True)


# That seems to work alright, let's see if our test network can still figure out the size after we "shapify":

# In[ ]:


transform = transforms.Compose([crop_out_black, center, tight_crop, to_128_torch, shapify_torch])
visualize_transform(transform)
get_acc_per_dim(transform, target='size', num_epochs=2)


# Surprisingly, it still can in some cases. My guess is that it's using the shape of the black areas in the corners, such as whether there is a triangle or a square marker on the right, and maybe artefacts from the resizing. To make extra sure, let's turn the four corners of each image black.

# In[ ]:



def remove_corners(img): #blacken a triangle of 1/6 at each corner. assumes square input
    corner_size = img.shape[1]//6
    mask = torch.ones( (corner_size, corner_size )).triu()
    img[:, :corner_size, :corner_size] *= mask.flip(dims=(0,))[None, :, :]
    img[:, :corner_size, -corner_size:] *= mask.flip(dims=(0,1))[None, :, :]
    img[:, -corner_size:, :corner_size] *= mask[None, :, :]
    img[:, -corner_size:, -corner_size:] *= mask.flip(dims=(1,))[None, :, :]
    return(img)

transform = transforms.Compose([crop_out_black, center, tight_crop, to_128_torch, remove_corners])
visualize_transform(transform)


# Since the foreground has exactly the same shape for every train image now, we can be sure that any unwanted correlation between the shape of the foreground and the label is gone. Standardizing the foreground shape is also done in https://www.kaggle.com/taindow/pre-processing-train-and-test-images . There the active area is cropped to a circle, here by just taking of the corners we can keep a bit more of the retina, and the result is closer to the most common crop on the test set.

# # Removing color correlations
# So, we removed the spurious shape correlation. There's a possibility that unwanted correlations are still hiding in the colors though... Let us remove the true label information by a Gaussian blur.

# In[ ]:


def gaussian_blur(img, radius=None, rel_size = None):
    if radius is None:
        radius = int(rel_size * img.shape[1])
    if radius % 2 == 0:
        radius = radius + 1
    img_numpy = img.permute(1, 2, 0).numpy()
    img_numpy = cv2.GaussianBlur(img_numpy,(radius,radius),0)
    img = torch.Tensor(img_numpy).permute(2, 0, 1)
    return img
transform = transforms.Compose([crop_out_black, center, tight_crop, to_128_torch, remove_corners, lambda img: gaussian_blur(img, rel_size=.5)])
visualize_transform(transform)


# Let's see if our model can still tell the original size / class from the blurred images:

# In[ ]:


get_acc_per_dim(transform, target='size')


# In[ ]:


get_acc_per_dim(transform, target='class')


# It can, with accuracies above chance. For example, for original size #3 it mostly guessed 0, and for original size 6 it mostly guessed 3, it's not supposed to be able to do that. Let's remove some background coloring by subtracting the Gaussian blur, as in https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping (originally Ben Graham)

# In[ ]:


def subtract_gaussian_blur(img, rel_size = .2, color_scale = 1):
    img_blurred = gaussian_blur(img, rel_size = rel_size)
    img = (4*color_scale*(img - img_blurred)).sigmoid() #sigmoid to squish to [0, 1]. Factor 4 because the slope of sigmoid at 0 is 4.
    return img

visualize_transform(subtract_gaussian_blur)


# The original color is still visible as the color of the rim. Let's combine it with the crop - by subtracting the blur first and then cropping we can get rid of the colorful rim.

# In[ ]:


transform = transforms.Compose([
    crop_out_black, 
    center, 
    tight_crop, 
    to_128_torch, 
    lambda img: subtract_gaussian_blur(img, rel_size=.2, color_scale=1), 
    remove_corners])
visualize_transform(transform)


# Those start looking really similar to each other. Let's test again whether the model can predict the original size from the blurred version:

# In[ ]:


transform = transforms.Compose([
    crop_out_black, 
    center, 
    tight_crop, 
    to_128_torch, 
    lambda img: subtract_gaussian_blur(img, rel_size=.2, color_scale=1), 
    remove_corners,
    lambda img: gaussian_blur(img, rel_size = .3)])
visualize_transform(transform)


# In[ ]:


# get_acc_per_dim(transform, target='size', num_epochs=2) #overfits
get_acc_per_dim(transform, target='size', num_epochs=1)


# It still somewhat can, in particular it's learned to keep apart original sizes 3 and 11 (the two most frequent ones) by their blurred versions. My guess is that the information is in the hue of the bright spot. But the situation is better than when we started out, let's stop here.

# # Making sure the transformation works on the original train images and on test
# 
# I've tried to keep things 'relative', but let's check:

# In[ ]:


size = 512
train_path = '../input/aptos2019-blindness-detection/train_images/'

transform = transforms.Compose([
    crop_out_black, 
    center, 
    tight_crop, 
    lambda img: torch.nn.functional.interpolate(img[None, ...], size = (size, size), mode='bilinear', align_corners = False)[0, ...] ,
    lambda img: subtract_gaussian_blur(img, rel_size=.2, color_scale=2), 
    remove_corners])

visualize_transform(transform, compare=True)


# Seems to work. Make sure it works on test:

# In[ ]:


test_path = '../input/aptos2019-blindness-detection/test_images/'
visualize_transform(transform, compare=True, train=False)


# # Apply.

# In[ ]:


#!mkdir 2_preprocessed_512
#target_dir = '2_preprocessed_512/'
#import os
#train_image_files = os.listdir('../input/aptos2019-blindness-detection/train_images')

#for i,train_image_file in enumerate( train_image_files):
#    img = transforms.ToTensor()(Image.open('../input/aptos2019-blindness-detection/train_images/' + train_image_file))
#    img = transforms.ToPILImage()(transform(img))
#    #torch.save(img, target_dir + train_image_file + '.pth')
#    img.save(target_dir + train_image_file)
#    print(i)

#import zipfile
#zf = zipfile.ZipFile('2_preprocessed_512.zip', mode='w')

#for i,train_image_file in enumerate( train_image_files):
#    zf.write(target_dir + train_image_file)
#    #zf.write('../input/train_images/' + train_image_file)
#zf.close()

#!rm -r 2_preprocessed_512

