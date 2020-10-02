#!/usr/bin/env python
# coding: utf-8

# ### This kernel is fork of [this](https://www.kaggle.com/bonhart/brain-mri-data-visualization-unet-fpn#DataGenerator-and-Data-Augmentation) kernel.
# 
# ### Steps:
# + Data Preparation
# + Visualization data
# + Datataset and DataGenerator
# + UNet
# + Train model
# + Test predictions

# # Data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport numpy as np\n\nimages = np.load("/kaggle/input/brain-tumor/brain_tumor_dataset/images.npy", allow_pickle=True)\nmasks = np.load("/kaggle/input/brain-tumor/brain_tumor_dataset/masks.npy", allow_pickle=True)\nlabels = np.load("/kaggle/input/brain-tumor/brain_tumor_dataset/labels.npy")\ninteger_to_class = {1: \'meningioma\', 2: \'glioma\', 3: \'pituitary tumor\'}\n\nprint(f"images:{images.shape}, \\\nmasks:{masks.shape}, \\\nlabels:{labels.shape}")')


# Stacking rows as a data frame.

# In[ ]:


data = np.column_stack((images, masks, labels))
data.shape


# Split data on train val test

# In[ ]:


from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(data, test_size=0.08, random_state=42)
train_data, test_data = train_test_split(train_data, test_size=0.12, random_state=42)

print("Train:", train_data.shape,
      "\nVal:", val_data.shape, 
      "\nTest:", test_data.shape,)


# # What does the data look like?

# ### Class distribution

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use("dark_background")

# https://www.kaggle.com/awsaf49/brain-tumor-visualization/data

labels, counts = np.unique(data[:,2], return_counts=True)

plt.figure(figsize=(10,6))
plt.bar(labels, counts, color=["aqua", "violet", "greenyellow"],
        tick_label=['Meningioma(1)', 'Glioma(2)', 'Pituitary Tumor(3)'])


# Annotate
for row, value in zip(labels,counts):
    plt.annotate(int(value), xy=(row, value-150), 
                rotation=0, color="black", 
                ha="center", verticalalignment='bottom', 
                fontsize=15, fontweight="bold")


# ### Samples of images of each class

# In[ ]:


import cv2

def data_to_viz(data, label, n=5):
    
    # logical slice for receiving data with the expected label
    expected_index = np.where(data[:,2] == label)
    expected_data = data[expected_index]
    
    # n random samples
    index = np.random.choice(expected_data.shape[0], n, replace=False)
    data_to_viz = expected_data[index]
    
    imgs = []
    masks = []
    labels = []
    for data_i in data_to_viz:
        
        # img
        imgs.append(cv2.resize(data_i[0], (512, 512)))

        # mask
        masks.append(cv2.resize(data_i[1].astype("uint8"), 
                                (512, 512)))

        # label
        labels.append(data_i[2])

    return np.hstack(imgs), np.hstack(masks), labels


# Data

# In[ ]:


meningiomas_imgs, meningiomas_masks, meningiomas_labels = data_to_viz(data, label=1, n=5)
glioma_imgs, glioma_masks, glioma_labels  = data_to_viz(data, label=2, n=5)
tumor_imgs, tumor_masks, tumor_labels = data_to_viz(data, label=3, n=5)

print("Meningiomas:",
      meningiomas_imgs.shape, meningiomas_masks.shape, meningiomas_labels)
print("Glioma:",
      glioma_imgs.shape, glioma_masks.shape, glioma_labels)
print("Pituitary Tumor:",
      tumor_imgs.shape, tumor_masks.shape, tumor_labels)


# Plot

# In[ ]:


# Data to visualization
from mpl_toolkits.axes_grid1 import ImageGrid

# Plot
fig = plt.figure(figsize=(25., 25.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),  # creates 1x4 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


grid[0].imshow(meningiomas_imgs, cmap="bone")
grid[0].imshow(np.ma.masked_where(meningiomas_masks == False, 
                                  meningiomas_masks), cmap='rainbow', alpha=0.3)

grid[0].set_title("Meningiomas", fontsize=20)
grid[0].axis("off")

grid[1].imshow(glioma_imgs, cmap="bone")
grid[1].imshow(np.ma.masked_where(glioma_masks == False,
                                  glioma_masks), cmap='rainbow', alpha=0.3)
grid[1].set_title("Glioma", fontsize=20)
grid[1].axis("off")

grid[2].imshow(tumor_imgs, cmap="bone")
grid[2].imshow(np.ma.masked_where(tumor_masks == False,
                                  tumor_masks), cmap='rainbow', alpha=0.3)

grid[2].set_title("Pituitary Tumor", fontsize=20)
grid[2].axis("off")


# annotations
plt.suptitle("Brain MRI Images for Brain Tumor Detection\nBrainTumorRetrieval Dataset",
             y=.80, fontsize=30, weight="bold")

# save and show
plt.savefig("dataset.png", pad_inches=0.2, transparent=True)
plt.show()


# # Datataset and DataGenerator

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


class BrainMriDataset(Dataset):
    def __init__(self, data, transforms, n_classes=3):
        
        self.data = data
        self.transforms = transforms
        self.n_classes = n_classes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        image = self.data[idx][0].astype("float32")

        # global standardization of pixels
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        
        # convert to rgb
        image_rgb = np.stack([image]*3).transpose(1,2,0)
        
        # create target masks
        label = self.data[idx][2] -1
        mask = np.expand_dims(self.data[idx][1], -1)
        
        target_mask = np.zeros((mask.shape[0], mask.shape[1], 
                                self.n_classes))
        target_mask[:,:, label : label + 1] = mask.astype("uint8")
        
        #  binary mask
        target_mask = np.clip(target_mask, 0, 1).astype("float32")
        
        # augmentations
        augmented = self.transforms(image=image_rgb, 
                                    mask=target_mask)
        image = augmented['image']
        mask = augmented['mask']
        
        return image, mask


# ### Data Transformation

# In[ ]:


transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, 
                       border_mode=0),
                        
    A.GridDistortion(p=0.5),
    A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
    A.Resize(156, 156, p=1.),
    A.RandomCrop(128, 128, p=1.)
    ])


# ### Data Generators

# In[ ]:


# train
train_dataset = BrainMriDataset(data=train_data, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, 
                              shuffle=True)

# validation
val_dataset = BrainMriDataset(data=val_data, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4, 
                            shuffle=True)

# test
test_dataset = BrainMriDataset(data=test_data, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4, 
                             shuffle=True)


# In[ ]:


def show_aug(inputs, nrows=3, ncols=5, image=True):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    i_ = 0
    
    if len(inputs) > 15:
        inputs = inputs[:15]
        
    for idx in range(len(inputs)):
    
        # normalization
        if image is True:           
            img = inputs[idx].numpy()#.transpose(1,2,0)
            #mean = [0.485, 0.456, 0.406]
            #std = [0.229, 0.224, 0.225] 
            #img = (img*std+mean).astype(np.float32)
            #img = np.clip(img, 0,1)
        else:
            img = inputs[idx].numpy().astype(np.float32)
            img = img[0,:,:]
        
        #plot
        #print(img.max(), len(np.unique(img)), img.mean())
        plt.subplot(nrows, ncols, i_+1)
        plt.imshow(img); 
        plt.axis('off')
 
        i_ += 1
        
    return plt.show()

    
images, masks = next(iter(train_dataloader))
print(images.shape, masks.shape)

show_aug(images)
show_aug(masks)


# # UNet

# In[ ]:


from torchvision.models import resnext50_32x4d

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
        
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        
        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x


# In[ ]:


class ResNeXtUNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [4*64, 4*128, 4*256, 4*512]
        
        # Down
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        # Up
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                       
        
    def forward(self, x):
        # Down
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Up + sc
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        #print(d1.shape)

        # final classifier
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)
        
        return out


# # Metric and Loss

# In[ ]:


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def dice_coef_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


# In[ ]:


model = ResNeXtUNet(n_classes=3).to(device)
adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.1)


# # Train Model

# In[ ]:


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, epoch):
    print("Start Train ...")
    model.train()

    losses = []
    accur = []

    for data, target in data_loader:

        data = data.permute(0,3,1,2).to(device)
        targets = target.permute(0,3,1,2).to(device)

        outputs = model(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = dice_coef_metric(out_cut, targets.data.cpu().numpy())

        loss = bce_dice_loss(outputs, targets)

        losses.append(loss.item())
        accur.append(train_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if lr_scheduler is not None:
        lr_scheduler.step()

    print("Epoch [%d]" % (epoch))
    print("Mean loss on train:", np.array(losses).mean(), "Mean DICE on train:", np.array(accur).mean())

    return np.array(losses).mean(), np.array(accur).mean()


# In[ ]:


def val_epoch(model, data_loader_valid, epoch, threshold=0.3):
    if epoch is None:
        print("Test Start...")
    else:
        print("Start Validation ...")

    model.eval()
    val_acc = []

    with torch.no_grad():
        for data, targets in data_loader_valid:

            data = data.permute(0,3,1,2).to(device)
            targets = targets.permute(0,3,1,2).to(device)

            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            val_dice = dice_coef_metric(out_cut, targets.data.cpu().numpy())
            val_acc.append(val_dice)

        print("Epoch:  " + str(epoch) + "  Threshold:  " + str(threshold)              + " Mean Validation DICE Score:", np.array(val_acc).mean())
        
        return np.array(val_acc).mean()


# In[ ]:


from tqdm import trange
import os
import glob

weights_dir = "weights"
if os.path.exists(weights_dir) == False:
    os.mkdir(weights_dir)

num_epochs = 30
loss_history = []
train_dice_history = []
val_dice_history = []

for epoch in trange(num_epochs):
    loss, train_dice = train_one_epoch(model, adam, scheduler, 
                                       train_dataloader, epoch)
    
    val_dice = valscore = val_epoch(model, val_dataloader, epoch)

    # train history
    loss_history.append(loss)
    train_dice_history.append(train_dice)
    val_dice_history.append(val_dice)

    # save best weights
    best_dice = max(val_dice_history)
    if val_dice >= best_dice:
        torch.save({'state_dict': model.state_dict()},
                   os.path.join(weights_dir, f"{val_dice:0.5f}_.pth"))


# Since the net did not reach a plateau, batch norm layers did not accumulate stable statistics; therefore, saved model weights in the early steps of the train loop - shows worse results, how to fix it? reach a plateau or go forward several epochs with ```torch.no_grad_():``` (dirty tricks) before saving weights or before switching the model to eval mode ```model.eval()``` for the weights that are.

# In[ ]:


# Dirty tricks
""" with torch.no_grad():
       for data, targets in data_loader_valid:
           data = data.permute(0,3,1,2).to(device)
           outputs = model(data)


model.eval()
for m in model.modules():
   if isinstance(m, nn.BatchNorm2d):
    m.track_runing_stats=False""";


# In[ ]:


# Load the best weights
best_weights =  sorted(glob.glob(weights_dir + "/*"),
                       key= lambda x: x[8:-5])[-1]
checkpoint = torch.load(best_weights)
model.load_state_dict(checkpoint['state_dict'])

print(f'Loaded model: {best_weights.split("/")[1]}')


# ### Train history

# In[ ]:


def plot_model_history(train_history,
                       val_history,
                       loss_history ,
                       num_epochs):
    
    x = np.arange(num_epochs)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train_history, label='train dice', lw=3, c="springgreen")
    plt.plot(x, val_history, label='validation dice', lw=3, c="deeppink")
    plt.plot(x, loss_history, label='dice + bce', lw=3)

    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)
    plt.legend()

    return plt.show()


# In[ ]:


plot_model_history(train_dice_history, val_dice_history, loss_history, num_epochs)


# # Test prediction

# In[ ]:


test_iou = val_epoch(model, test_dataloader, epoch=None, threshold=0.5)
print(f"""Mean IoU of the test images - {np.around(test_iou, 2)*100}%""")


# ### Global IoU with different thresholds

# In[ ]:


dices = []
thresholds = [0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.88]
for i in thresholds:
    test_dice = val_epoch(model, test_dataloader,threshold=i, epoch=None)
    dices.append(test_dice)


# In[ ]:


import random
import matplotlib.colors as mcolors

colors = random.choices(list(mcolors.CSS4_COLORS.keys()),k=len(thresholds))

plt.figure(figsize=(10,6))
plt.bar(thresholds, dices, width=0.05, color=colors)
plt.ylabel("Dice", fontsize=15)
plt.xlabel("Threshold values", fontsize=15)
plt.title("Global IoU with different thresholds", fontsize=15)


# Annotate
for row, value in zip(thresholds, dices):
    plt.annotate(f"{value*100:0.2f}%", xy=(row, value), 
                 rotation=0, color="white", 
                 ha="center", verticalalignment='bottom', 
                 fontsize=10, fontweight="bold")


# ### IoU for each class
# 

# In[ ]:


test_predictions = []
test_ground_truths = []
for data, target in test_dataloader:
    with torch.no_grad():
        data = data.permute(0,3,1,2).to(device)
        target = target.permute(0,3,1,2)
        prediction = model(data)
        test_predictions.append(prediction.detach().cpu())
        test_ground_truths.append(target)


# In[ ]:


test_predictions = torch.cat(test_predictions)
test_ground_truths = torch.cat(test_ground_truths)
#test_predictions = test_predictions.reshape(test_predictions.shape[0], -1)
#test_ground_truths = test_ground_truths.reshape(test_ground_truths.shape[0], -1)

print(test_predictions.shape, test_ground_truths.shape)


# In[ ]:


# data
dice1 = dice_coef_metric(test_predictions[:,0,:,:], test_ground_truths[:,0,:,:])
dice2 = dice_coef_metric(test_predictions[:,1,:,:], test_ground_truths[:,1,:,:])
dice3 = dice_coef_metric(test_predictions[:,2,:,:], test_ground_truths[:,2,:,:])
dices = [dice1, dice2, dice3]

# x, y
x = np.arange(3)
dices = [dice1, dice2, dice3]

# plot
plt.figure(figsize=(10, 6))
plt.bar(x, dices, 
        color=["aqua", "violet", "greenyellow"], width=0.5)

                                        
plt.xticks(x, ['Meningioma(1)', 'Glioma(2)', 'Pituitary Tumor(3)'], fontsize=15)
plt.ylabel("Dice", fontsize=15)
plt.title("Dice for each class", fontsize=15)


# Annotate
for row, value in zip(x, dices):
    plt.annotate(f"{value*100:0.3f}%", xy=(row, value), 
                 rotation=0, color="white", 
                 ha="center", verticalalignment='bottom', 
                 fontsize=10, fontweight="bold")
    
plt.show()


# ### Random test sample

# In[ ]:


index = np.random.choice(test_data.shape[0], 1, replace=False)

# image
image = test_data[index][0][0]

# global standardization of pixels
mean, std = image.mean(), image.std()
image = (image - mean) / std  
image = cv2.resize(image, (128, 128))
# convert to rgb
image = np.stack([image]*3).transpose(1,2,0)

# mask
mask = test_data[index][0][1]

# label
label = test_data[index][0][2]

print(image.shape, mask.shape, label)


# In[ ]:


#----------- Data -------------#

# predictions
preds = torch.tensor(image.astype(np.float32)).unsqueeze(0).permute(0,3,1,2)
preds = model(preds.to(device))
preds = preds.detach().cpu().numpy()

# threshold
preds[np.nonzero(preds < 0.4)] = 0.0
preds[np.nonzero(preds >= 0.4)] = 255.#1.0
preds = preds.astype("uint8")

pred_1 = preds[:,0,:,:]
pred_2 = preds[:,1,:,:]
pred_3 = preds[:,2,:,:]


#------------ Plot ------------#

# data plot
fig, ax = plt.subplots(nrows=1,  ncols=2, figsize=(10, 10))

ax[0].imshow(image)
ax[0].set_title("Image")
ax[1].imshow(mask)
ax[1].set_title(f'Ground Truth with label "{integer_to_class[label].capitalize()}"')
#ax[1].imshow(preds[0,:,:,:])
#ax[0].set_title("Preiction")
plt.suptitle("Random Test Sample",
             y=.75, fontsize=20, weight="bold")

# prediction plot
fig, ax = plt.subplots(nrows=1,  ncols=3, figsize=(10, 10))

ax[0].imshow(pred_1[0,:,:])
ax[0].set_title(f'{integer_to_class[1].capitalize()}')
ax[1].imshow(pred_2[0,:,:])
ax[1].set_title(f'{integer_to_class[2].capitalize()}')
ax[2].imshow(pred_3[0,:,:])
ax[2].set_title(f'{integer_to_class[3].capitalize()}')

