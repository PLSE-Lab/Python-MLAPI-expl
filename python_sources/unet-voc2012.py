#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F

import scipy.ndimage.morphology as morph


# In[ ]:


torch.cuda.is_available()
torch.zeros(5).cuda()


# In[ ]:


path = "../input/pascal-voc-2012/VOC2012/ImageSets/Segmentation/train.txt"
 
#print(os.listdir("../input/pascal-voc-2012/VOC2012/JPEGImages"))
f = open(path, "r").read().split('\n')
f = f[:1464]
folder_data = "../input/pascal-voc-2012/VOC2012/JPEGImages"
folder_mask = "../input/pascal-voc-2012/VOC2012/SegmentationClass"


# In[ ]:


tfs = transforms.Compose([transforms.CenterCrop((256, 256)),
                   transforms.ToTensor()])
#tfs = transforms.ToTensor()

img = Image.open(folder_data + "/" + f[20] + ".jpg").convert('RGB')
seg = Image.open(folder_mask + "/" + f[20] + ".png").convert('RGB')
resize = transforms.CenterCrop((256, 256))
seg = resize(seg)
img = resize(img)
s = np.asarray(seg).tolist()
#s[200][200]  == [0, 0, 0]
#seg
s[200][200]
seg


# In[ ]:


img


# In[ ]:


trans = transforms.ToPILImage()


# In[ ]:


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
            [224, 224, 192]
        ]
    )


# In[ ]:


d =    [[[0, 0, 0],      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 0, 0],    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 128, 0],    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 128, 0],  [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 0, 128],    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 0, 128],  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 128, 128],  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 128, 128],[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 0, 0],     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 0, 0],    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 128, 0],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 128, 0],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 0, 128],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 0, 128],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 128, 128], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 128, 128],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]],
        [[0, 64, 0],     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]],
        [[128, 64, 0],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]],
        [[0, 192, 0],    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
        [[128, 192, 0],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]],
        [[0, 64, 128],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        [[224, 224, 192],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]]

l = []

for itm in range(256):
    for item in range(256):
        #print(mask[itm][item])
        for i, j in enumerate(d):
            #print(mask[itm][item], j[0])
            if s[itm][item] == j[0]:
                l.append(j[1])
l = np.resize(l,(256, 256, 22))
l = torch.from_numpy(l).permute(2, 0, 1)
l.shape


# In[ ]:


#for i, label in enumerate(get_pascal_labels()):
#        #print(ii, label)
#        if i == 0:
#            x = torch.tensor(np.all(mask == label, axis=-1), dtype=torch.float).unsqueeze(0)
#        else:
#            x = torch.cat((x, torch.tensor(np.all(mask == label, axis=-1), dtype=torch.float).unsqueeze(0)), 0)
#
#print(x.shape)
#for i in range(21):
#    print(x[i][120])
#trans(x[1])


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.img_paths = os.listdir(folder_data)
        self.seg_paths = os.listdir(folder_mask)
        self.transform = transforms.Compose([
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor()])
        self.resize = transforms.CenterCrop((256, 256))
        self.data = len(f)
        
    def __getitem__(self, idx):
        img = Image.open(folder_data + "/" + f[idx] + ".jpg").convert('RGB')
        img = self.transform(img)
        
        seg = Image.open(folder_mask + "/" + f[idx] + ".png").convert('RGB')
        seg = self.resize(seg)
        l = []

        for itm in range(256):
            for item in range(256):
                #print(mask[itm][item])
                for i, j in enumerate(d):
                    #print(mask[itm][item], j[0])
                    if s[itm][item] == j[0]:
                        l.append(j[1])
        l = torch.from_numpy(np.resize(l,(256, 256, 22))).permute(2, 0, 1)
        return img, l
    
    def __len__(self):
        return len(f)
    
class MyDataset1(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.img_paths = os.listdir(folder_data)
        self.seg_paths = os.listdir(folder_mask)
        self.transform = transforms.Compose([
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    ])#transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.resize = transforms.CenterCrop((256, 256))
        self.data = len(f)
        
    def __getitem__(self, idx):
        img = Image.open(folder_data + "/" + f[idx] + ".jpg").convert('RGB')
        img = self.transform(img)
        
        seg = Image.open(folder_mask + "/" + f[idx] + ".png").convert('RGB')
        seg = self.resize(seg)
        mask = np.asarray(seg).astype(int)
        #zeros = torch.zeros(22, 256, 256)
        for ii, label in enumerate(get_pascal_labels()):
            if ii == 0:
                s = torch.tensor(np.all(mask == label, axis=-1), dtype=torch.float).unsqueeze(0)
                depth = morph.distance_transform_edt(s.numpy())
            else:
                s = torch.cat((s, torch.tensor(np.all(mask == label, axis=-1), dtype=torch.float).unsqueeze(0)), 0)
                depth += morph.distance_transform_edt(s[-1].numpy())
        
        return img, s, torch.from_numpy(depth).squeeze(0)
    
    def __len__(self):
        return len(f)


# In[ ]:


dataset = MyDataset1()
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          num_workers=4,
                          shuffle=True)


# In[ ]:


#dataset[1]
class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim))

    def forward(self, x):
        out = self.block(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.ConvTranspose2d(mid_dim, out_dim, kernel_size=2, stride=2))

    def forward(self, x):
        out = self.block(x)
        return out

class UNet1(nn.Module):
    def __init__(self, num_classes, in_dim=3, conv_dim=64):
        super(UNet1, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.conv_dim = conv_dim
        self.build_unet()

    def build_unet(self):
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim))
        self.enc2 = DownBlock(self.conv_dim, self.conv_dim*2)
        self.enc3 = DownBlock(self.conv_dim*2, self.conv_dim*4)
        self.enc4 = DownBlock(self.conv_dim*4, self.conv_dim*8)

        self.dec1 = UpBlock(self.conv_dim*8, self.conv_dim*16, self.conv_dim*8)
        self.dec2 = UpBlock(self.conv_dim*16, self.conv_dim*8, self.conv_dim*4)
        self.dec3 = UpBlock(self.conv_dim*8, self.conv_dim*4, self.conv_dim*2)
        self.dec4 = UpBlock(self.conv_dim*4, self.conv_dim*2, self.conv_dim)

        self.last = nn.Sequential(
            nn.Conv2d(self.conv_dim*2, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_dim),
            nn.Conv2d(self.conv_dim, self.num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        enc1 = self.enc1(x) # 16
        enc2 = self.enc2(enc1) # 8
        enc3 = self.enc3(enc2) # 4
        enc4 = self.enc4(enc3) # 2

        center = nn.MaxPool2d(kernel_size=2, stride=2)(enc4)

        dec1 = self.dec1(center) # 4
        dec2 = self.dec2(torch.cat([enc4, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec2], dim=1))
        dec4 = self.dec4(torch.cat([enc2, dec3], dim=1))

        last = self.last(torch.cat([enc1, dec4], dim=1))
        assert x.size(-1) == last.size(-1), 'input size(W)-{} mismatches with output size(W)-{}'                                             .format(x.size(-1), output.size(-1))
        assert x.size(-2) == last.size(-2), 'input size(H)-{} mismatches with output size(H)-{}'                                             .format(x.size(-1), output.size(-1))
        return last


# In[ ]:





# In[ ]:


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),##########
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)##########
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        #x.shape
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = nn.Softmax(dim=1)(out)
        #print(out.shape)
        return out


# In[ ]:


model = UNet1(22).train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#model(img)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#l = torch.ones(4, 22,1,1).cuda()
#l[:, 0] = 0.005
#def lossy(x, y, d):
#    return (((x - y)**2).sum(dim=1)*d).sum()/(256**2)

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def avg(object):
    return sum(object)/len(object)

criterion = torch.nn.BCEWithLogitsLoss()
criterion1 = torch.nn.MSELoss()
criterion2 = nn.BCELoss().cuda()


# In[ ]:


# avg_loss = 5
# for epoch in range(30):
#     #if epoch%25 == 0:
#     print('epoch: ', epoch)
#     #l=[]
#     for i, data in enumerate(train_loader):
#         input, label, depth = data
#         input = input.to(device)
#         label = label.to(device)
# #         depth = depth.to(device)
        
        
        
#         output = model(input)
# #         print(output.shape)
#         #print(type(label))
#         loss = calc_loss(output, label)
#         #print(loss)
#         #l.append(loss.item())
#         #print(loss.item())
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     #if epoch%25 == 0:
#         if i % 30== 0:
#             print('iter: ', i)

#             print('loss: ', loss)


# In[ ]:


avg_loss = 5
for epoch in range(1):
    #if epoch%25 == 0:
    print('epoch: ', epoch)
    #l=[]
    for i, data in enumerate(train_loader):
        input, label, depth = data
        input = input.to(device)
        label = label.to(device)
#         depth = depth.to(device)
        
        
        
        output = model(input)
#         print(output.shape)
        #print(type(label))
        loss = calc_loss(output, label)
        #print(loss)
        #l.append(loss.item())
        #print(loss.item())
        loss.backward()
        avg_loss*=0.99
        avg_loss += 0.01*loss.cpu().item()
        if i%20==0:
            optimizer.step()
            optimizer.zero_grad()
    #if epoch%25 == 0:
        if i % 10 == 0:
            print('iter: ', i)
            print('average loss: ', avg_loss)
            print('loss: ', loss)


# In[ ]:


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'void']
for i, name in enumerate(VOC_CLASSES):
    print(i, name)


batch = 0
trans(input[batch].cpu())


# In[ ]:


from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
for i in range(22):
    print(i)
    print(VOC_CLASSES[i])
    imshow(output[batch][i].detach().cpu())
    plt.show()


# In[ ]:


for i in range(22):
    print(i, VOC_CLASSES[i])
    imshow(trans((output[batch][4].cpu()>0).float()))
    #imshow(trans(label[batch][i].cpu()))
    plt.show()


# In[ ]:


f, axarr = plt.subplots(1,2)
axarr[0,0].imshow(output[batch][0].detach().cpu())
axarr[0,1].imshow(trans(label[batch][0].cpu()))


# In[ ]:


imshow(trans((output[batch][3].cpu()>0).float()))
print(VOC_CLASSES[3], output[batch][8][175])


imshow(trans((output[batch][15].cpu()>0).float()))
print(VOC_CLASSES[6], output[batch][6][175])


# In[ ]:


for i in range(22):
    imshow(trans((output[batch][i].cpu()>0).float()))
    print(i, VOC_CLASSES[i], output[batch][i][175])


# In[ ]:


loader = MyDataset1()
trans(loader[0][0])


# In[ ]:


x, y, z = loader[1]
trans(x)


# In[ ]:


out = model(x.unsqueeze(0).cuda())


# In[ ]:


trans(out[0][2].cpu())


# In[ ]:


imshow(out[0][0].detach().cpu())
plt.show()


# In[ ]:




