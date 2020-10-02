#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import time
from random import randint


# In[ ]:


def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 1
    img = img.reshape(cols,rows)
    img = img.T
    return img


# In[ ]:


DF = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')
DF.head()


# In[ ]:


def showSegmentation(image,rle):
    mask = rleToMask(rle,image.shape[0],image.shape[1])
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.subplot(2,1,2)
    plt.imshow(np.expand_dims(mask,axis=2)*image,cmap='gray')
    


# In[ ]:


for _ in range(5):
    img_name = DF[~DF.EncodedPixels.isna()].sample().ImageId_ClassId.values[0]
    rlePixels = (str(DF[DF.ImageId_ClassId.str.contains(img_name)].EncodedPixels.values[0]))
    img = plt.imread('/kaggle/input/severstal-steel-defect-detection/train_images/'+img_name[:-2])
    print(img.shape)
    showSegmentation(img,rlePixels)
    plt.title('Segmentation for Class: '+img_name[-1])
    plt.show()


# In[ ]:


def LabelsToNumpyMasks(Labels,height,width):
    Tensor = np.zeros((height,width),dtype=np.uint8)
    for name,rle in Labels.values:
        if str(rle) != 'nan':
#             print(rle)
            Tensor+=int(name[-1])*rleToMask(rle,height,width)
        else:
            pass
    return Tensor


# In[ ]:



class Masks(Dataset):

 def __init__(self, csv_file, transform=None):
     super(Masks,self).__init__()
     self.csv = pd.read_csv(csv_file)
     self.filenames = np.unique(self.csv.ImageId_ClassId.apply(lambda x: x[:-2]).values)
     self.transform=transform
     
 def __len__(self):
     
     return len(self.filenames)

 def __getitem__(self, idx):
     
     Labels = self.csv[self.csv.ImageId_ClassId.str.contains(self.filenames[idx])]
     masks = LabelsToNumpyMasks(Labels,256,1600)
     sample = {'name':self.filenames[idx], 'masks': masks}

     if self.transform:
         sample = self.transform(sample)

     return sample

theMasks = Masks('/kaggle/input/severstal-steel-defect-detection/train.csv')
print(len(theMasks))


# In[ ]:


sample = theMasks[randint(0,len(theMasks))]

plt.figure(figsize=(20,10))
plt.imshow(sample['masks'],cmap='winter')
plt.title(sample['name'])
plt.show()


# In[ ]:


get_ipython().system(' mkdir ../masks')


# In[ ]:


get_ipython().system('ls ..')


# In[ ]:


def noCollate(batch):
    return batch[0]


# In[ ]:


maskLoader = DataLoader(theMasks,num_workers=28,shuffle=False,batch_size=1,collate_fn=noCollate)

for j,batch in enumerate(maskLoader):
    np.save(os.path.join('/kaggle/masks/',batch["name"]),batch["masks"])
    print("Completion: {}/{}".format(j+1,len(maskLoader)),end='\r')


# In[ ]:


class SteelDefectSegmentationDataset(Dataset):

    def __init__(self, csv_file,
                 root_dir,mask_dir,
                 transform=None):
        super(SteelDefectSegmentationDataset,self).__init__()
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        _, _, files = next(os.walk(self.root_dir))
        self.filenames = files

    def __len__(self):
        
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.filenames[idx])
        image = plt.imread(img_name)
        masks_name = os.path.join(self.mask_dir,
                                self.filenames[idx]+'.npy')
        masks = np.load(masks_name)
        sample = {'name':img_name,'image': image, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:


steel_dataset = SteelDefectSegmentationDataset(csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',
                                               root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',
                                               mask_dir='/kaggle/masks')


# In[ ]:


test = steel_dataset[randint(0,len(steel_dataset))]
plt.imshow(test["image"])
plt.show()
plt.imshow(test["masks"])
plt.show()


# In[ ]:


from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet101
theNet = deeplabv3_resnet101(num_classes=5).cuda()
backNet = deeplabv3_resnet101(pretrained=True).cuda()
for param in backNet.parameters():
    param.requires_grad = False


# In[ ]:


theNet.backbone = backNet.backbone


# In[ ]:


theNet.classifier.state_dict()


# In[ ]:


pretend = theNet(torch.randn(5,3,256,256).cuda())["out"]
pretend.shape


# In[ ]:


torch.save(deeplabv3_resnet101(num_classes=5).cuda(),'/kaggle/working/ResDeepLabv3')


# In[ ]:


from torchvision.transforms import functional as F
from torchvision import transforms as T
from random import uniform


# In[ ]:


class ToPIL(object):
    def __call__(self, sample):
        sample["image"] = T.ToPILImage()(sample["image"])
        sample["masks"] = T.ToPILImage()(sample["masks"])
        return sample
    
class Resize(object):
    def __init__(self,size):
        self.size=size
    def __call__(self, sample):
        sample["image"] = T.Resize(self.size,interpolation=2)(sample["image"])
        sample["masks"] = T.Resize(self.size,interpolation=0)(sample["masks"])
        return sample
class RandomColorJitter(object):
    def __call__(self, sample):
        sample["image"] = T.ColorJitter(0.5,0.5,0.5,0)(sample["image"])
        return sample

class Affine(object):
    def __call__(self, sample):
        rndAngle = uniform(-30,30)
        rndTranslate = (uniform(0,sample["image"].size[0]/3),
                        uniform(0,sample["image"].size[1]/3))
        rndShear = [uniform(0,30) for _ in range(2)]
        sample["image"] = F.affine(sample["image"],
                                   angle=rndAngle,
                                   translate=rndTranslate,
                                   scale=1,
                                   shear=rndShear)
        sample["masks"] = F.affine(sample["masks"],
                                   angle=rndAngle,
                                   translate=rndTranslate,
                                   scale=1,
                                   shear=rndShear)
        return sample

class HFlip(object):
    def __call__(self, sample):
        sample["image"] = F.hflip(sample["image"])
        sample["masks"] = F.hflip(sample["masks"])
        return sample
class VFlip(object):
    def __call__(self, sample):
        sample["image"] = F.vflip(sample["image"])
        sample["masks"] = F.vflip(sample["masks"])
        return sample
class Perspective(object):
    def __call__(self,sample):
        perp = T.RandomPerspective()
        params = perp.get_params(sample["image"].size[0],sample["image"].size[1],0.5)
        sample['image']=F.perspective(sample["image"],*params,3)
        sample['masks']=F.perspective(sample["masks"],params[0],params[1],0)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample["image"] = T.ToTensor()(sample["image"])
        sample["masks"] = torch.LongTensor(np.array(sample["masks"]))
        return sample
    
class Normalize(object):
    def __call__(self, sample):
        sample["image"] = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(sample["image"])
        return sample


# In[ ]:


steel_dataset = SteelDefectSegmentationDataset(csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',
                                               root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',
                                               mask_dir='/kaggle/masks')


# In[ ]:


n = 6
print(n)
test = steel_dataset[n]
plt.figure(figsize=(20,6))
plt.subplot(2,1,1)
plt.imshow(test["image"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.subplot(2,1,2)
plt.imshow(test["masks"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.show()


# In[ ]:


sample = ToPIL()(steel_dataset[n])
# def test_randomperspective(self):
#     for _ in range(10):
#         height = random.randint(24, 32) * 2
#         width = random.randint(24, 32) * 2
#         img = torch.ones(3, height, width)
#         to_pil_image = transforms.ToPILImage()
#         img = to_pil_image(img)
#         perp = transforms.RandomPerspective()
#         startpoints, endpoints = perp.get_params(width, height, 0.5)
#         tr_img = F.perspective(img, startpoints, endpoints)
#         tr_img2 = F.to_tensor(F.perspective(tr_img, endpoints, startpoints))
#         tr_img = F.to_tensor(tr_img)
#         assert img.size[0] == width and img.size[1] == height
#         assert torch.nn.functional.mse_loss(tr_img, F.to_tensor(img)) + 0.3 > \
#             torch.nn.functional.mse_loss(tr_img2, F.to_tensor(img))
perp=T.RandomPerspective()
params = perp.get_params(sample['image'].size[0],sample['image'].size[1],0.5)
plt.imshow(sample['image'])
plt.show()
plt.imshow(F.perspective(sample["image"],params[0],params[1],3))
plt.show()


# In[ ]:


plt.figure(figsize=(27,3))
for i in range(0,3):
    test = Affine()(ToPIL()(steel_dataset[n]))
    plt.subplot(2,3,i+1)
    plt.imshow(test["image"])
    plt.xticks(ticks=[]);plt.yticks(ticks=[])
    plt.subplot(2,3,i+4)
    plt.imshow(test["masks"])
    plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.show()


# In[ ]:


plt.figure(figsize=(27,3))
for i in range(0,3):
    test = Perspective()(ToPIL()(steel_dataset[n]))
    plt.subplot(2,3,i+1)
    plt.imshow(test["image"])
    plt.xticks(ticks=[]);plt.yticks(ticks=[])
    plt.subplot(2,3,i+4)
    plt.imshow(test["masks"])
    plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.show()


# In[ ]:


plt.figure(figsize=(14,3))
test = HFlip()(ToPIL()(steel_dataset[n]))
plt.subplot(2,2,1)
plt.imshow(test["image"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.subplot(2,2,3)
plt.imshow(test["masks"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])

test = VFlip()(ToPIL()(steel_dataset[n]))
plt.subplot(2,2,2)
plt.imshow(test["image"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])
plt.subplot(2,2,4)
plt.imshow(test["masks"])
plt.xticks(ticks=[]);plt.yticks(ticks=[])

plt.show()


# In[ ]:


steel_dataset = SteelDefectSegmentationDataset(
    csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',
    root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',
    mask_dir='/kaggle/masks/',
    transform=T.Compose([ToPIL(),Resize((128,800)),
                         T.RandomApply([T.RandomApply([RandomColorJitter(),
                                                      Affine(),HFlip(),VFlip(),Perspective()],
                                                     p=0.7)],
                                       p=0.7)
#                          ToTensor()
                        ]))
train_loader = torch.utils.data.DataLoader(steel_dataset, batch_size=1, 
                                           shuffle=True,num_workers=4,collate_fn=lambda x: x[0])

for idx,batch in enumerate(train_loader):
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.imshow(batch["image"])
    plt.subplot(2,1,2)
    plt.imshow(batch["masks"])
    plt.show()
    if idx>3:
        break


# In[ ]:


from torch.utils.data import SubsetRandomSampler


# In[ ]:


batch_size = 4

TrainSet = SteelDefectSegmentationDataset(
    csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',
    root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',
    mask_dir='/kaggle/masks/',
    transform=T.Compose([ToPIL(),Resize((128,800)),
                         T.RandomApply([T.RandomApply([RandomColorJitter(),
                                                      Affine(),HFlip(),VFlip(),Perspective()],
                                                     p=0.7)],
                                       p=0.7),
                         ToTensor(),Normalize()
                        ]))

ValSet = SteelDefectSegmentationDataset(
    csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',
    root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',
    mask_dir='/kaggle/masks/',
    transform=T.Compose([ToPIL(),Resize((128,800)),
                         ToTensor(),Normalize()
                        ]))
seed=42
trainSplit = 0.9
indices = list(range(len(TrainSet)))
np.random.seed(seed)
np.random.shuffle(indices)
trainIndices,valIndices = indices[:int(trainSplit*len(TrainSet))],indices[int(trainSplit*len(TrainSet)):]
train_loader = torch.utils.data.DataLoader(TrainSet, batch_size=batch_size, 
                                           shuffle=False,
                                           num_workers=8,
                                           pin_memory=1,
                                           sampler=SubsetRandomSampler(trainIndices))
val_loader = torch.utils.data.DataLoader(ValSet, batch_size=batch_size, 
                                           shuffle=False,
                                         num_workers=8,
                                         pin_memory=1,
                                         sampler=SubsetRandomSampler(valIndices))


# In[ ]:


LR = 1e-4
torch.cuda.empty_cache()
lossFunc = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(filter(lambda param: param.requires_grad,theNet.parameters()),lr=LR)
epochs = 30
TrainingLosses = torch.zeros(epochs)
ValidationLosses = torch.zeros(epochs)
count = 1
for i in range(epochs):
    print(' ')
    sumLosses=torch.cuda.FloatTensor([0])
    theNet = theNet.train()
    for idx,batch in enumerate(train_loader):
        batch['image'] = batch['image'].cuda()
        batch['masks'] = batch['masks'].cuda()
        pred = (theNet(batch['image']))["out"]
        loss = lossFunc(pred,batch['masks'])
        loss.backward()
        optim.step()
        optim.zero_grad()
        sumLosses+=loss.detach()
        count+=1
        print(' Training Completion: {}%'.
              format(round((count)/(epochs*len(train_loader))*100,3)),end='\r')
        del pred, loss
    print(' Mean Training Loss: {}, Epoch: {}'.format(np.round(sumLosses.detach().cpu().numpy()/len(train_loader),5),i+1))

    print(' ')
    TrainingLosses[i]=(sumLosses.cpu()/len(train_loader))
    sumLosses=torch.cuda.FloatTensor([0])
    theNet = theNet.eval()
    with torch.no_grad():
        for idx,batch in enumerate(val_loader):
            batch['image'] = batch['image'].cuda()
            batch['masks'] = batch['masks'].cuda()
            pred = theNet(batch['image'])["out"]
            loss = lossFunc(pred,batch['masks'])
            sumLosses+=loss.detach()
            print(' Validation Completion: {}/{}'.
                  format(idx+1,len(val_loader)),end='\r')
            del pred, loss

    ValidationLosses[i]=(sumLosses.cpu()/len(val_loader))
    print(' Mean Validation Loss: {}'.format(np.round(sumLosses.detach().cpu().numpy()/len(val_loader),5)))
    print(' ')
    torch.save(theNet.classifier.state_dict(),'/kaggle/working/ResDeepLabv3Headstate'+str(i))


# In[ ]:


print("Learning rate was: "+str(LR))
print("Lowest Training Loss: {}".format(np.array(TrainingLosses).min()))
print("Lowest Validation Loss: {}, at epoch: {}".format(np.array(ValidationLosses).min(),np.array(ValidationLosses).argmin()))


# In[ ]:


plt.figure()
plt.plot(TrainingLosses,'-b')
plt.plot(ValidationLosses,'-r')
plt.show() 


# In[ ]:


get_ipython().system('ls')


# In[ ]:




