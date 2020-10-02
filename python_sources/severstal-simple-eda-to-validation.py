#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python


# ### End-To-End Severstal Steel Defect Detection (EDA, Validation)

# In[ ]:


# it ensures any edits to libraries you make are reloaded here automatically
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# any charts or images displayed are shown in this notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# avoid priniting warnings
import warnings
warnings.filterwarnings("ignore")


#display is equivalent to print but in case of data frame it makes prity print
from IPython.display import display as disp 


# ### Importing all required packages and libraries 

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('pwd # pwd (present working directory) our default current working directroy is /kaggle/working')


# ### Utility Functions

# In[ ]:


def list_files(dir_name='/kaggle/input'):
    for dirname, _, filenames in os.walk(dir_name):
        for filename in filenames:
            print(os.path.join(dirname, filename))
        


# In[ ]:


# list_files("/kaggle/input/")


# > ### Constants

# In[ ]:


train_csv_path = "../input/severstal-steel-defect-detection/train.csv"
sample_submission_csv_path = "../input/severstal-steel-defect-detection/sample_submission.csv"
train_image_path = "../input/severstal-steel-defect-detection/train_images/"
test_image_path = "../input/severstal-steel-defect-detection/test_images/"


# ### Exploring CSV file

# In[ ]:


def explore_csv(path):
    df = pd.read_csv(path)    
    disp(df)
    print("Path:", path)
    print("File Name:", path.split("/")[-1])
    print()
    print("Shape:", df.shape)
    print()
    print(df.info())
    return df


# In[ ]:


explore_csv(train_csv_path)


# ### Exploring Training Data (CSV File)

# In[ ]:


# Exploring train.csv file

train_df = pd.read_csv(train_csv_path)
disp(train_df.head())
print("\nTotal numberof Rows, Columns:",train_df.shape)
# Total numberof Rows in Training CSV
print("\nTotal numberof Rows in Training CSV:",train_df.shape[0])
print()

# Number of images with defects
defect_count = train_df.EncodedPixels.count()
print("Number of images with defects (as per CSV):",defect_count)
# Number of images with NO defects
no_defect_count = train_df.EncodedPixels.isna().sum()
print("Number of images with no defects(as per CSV):",no_defect_count)
print("Note: For each image there ar 4 rows in CSV\n")

# Number of jpegs in  training and test sets
train_fns = os.listdir('/kaggle/input/severstal-steel-defect-detection/train_images')
print("The number of Training Images:", len(train_fns))
test_fns = os.listdir('/kaggle/input/severstal-steel-defect-detection/test_images')
print("The number of Test Images:", len(test_fns))

# Number of images has at least one defect (max of 4 defects)
df = train_df.dropna()
df['ImageId'] = [x.split('_')[0] for x in df['ImageId_ClassId']]
print("\nNumber of images has AT LEAST ONE DEFECT:",len(df.ImageId.unique()))
print("Number of NO Defect images in training set:", len(train_fns) - len(df.ImageId.unique()))
print()
note = """Note: There is possible to have maximum of 4 defects 
    but in this case there are only 2 images has 3 Defects
    """
print(note)



# In[ ]:


# ploting a pie chart to show the percentage of "Defect vs No Defect"
labels = 'Defect', 'No Defect'
sizes = [defect_count, no_defect_count]
explode=(0.1,0)
plt.pie(sizes, labels=labels, autopct='%1.2f%%', explode=explode, colors=['r','g'], shadow=True, startangle=0)
plt.title('Defect vs No Defect')
plt.show()


# In[ ]:


# Number of images has at least one defect (max of 4 defects)
df = train_df.dropna()
df['ClassId'] = [x.split('_')[1] for x in df['ImageId_ClassId']]
df['ImageId'] = [x.split('_')[0] for x in df['ImageId_ClassId']]

images_per_class = df.groupby('ClassId')['ImageId'].count()
disp(images_per_class)


# In[ ]:


# plot number of images per defect class (4 classes)
fig, ax = plt.subplots() 
plt.barh(
    y=[1,2,3,4],
    width=images_per_class, 
    color=['magenta', 'red', 'green', 'cyan'], 
    tick_label=["Defect Class1","Defect Class2","Defect Class3","Defect Class4"]    
)
for y, v in enumerate(images_per_class, 1):
    ax.text(1,y,v)
plt.show()


# In[ ]:


# ploting a pie chart to show the percentage of "Defect vs No Defect"
labels = ["Defect Class1","Defect Class2","Defect Class3","Defect Class4"]
sizes = images_per_class
plt.pie(sizes, labels=labels, autopct='%1.2f%%',  colors=['r','g','m','y'], shadow=True, startangle=0)
plt.title('Defect Classes % wise')
plt.show()


# In[ ]:


defect_classes_per_image = pd.DataFrame({ 'NoOfDefects': df.groupby('ImageId')['ClassId'].count()})
one_defect_imgs = defect_classes_per_image.groupby("NoOfDefects").get_group(1).index.values.tolist()
two_defects_imgs = defect_classes_per_image.groupby("NoOfDefects").get_group(2).index.values.tolist()
three_defects_imgs = defect_classes_per_image.groupby("NoOfDefects").get_group(3).index.values.tolist()
no_of_defects = defect_classes_per_image.groupby("NoOfDefects").size()
print(f"Image with One defect: {no_of_defects[1]} \nImage with Two defects: {no_of_defects[2]}\nImage with Three defects: {no_of_defects[3]}")


# In[ ]:


# plot number of defects in each images
fig, ax = plt.subplots() 
plt.barh(
    y=[1,2,3],
    width=no_of_defects, 
    color=['magenta', 'red', 'green'],
    tick_label=["1 Defect","2 Defects","3 Defects"]    
)
for y, v in enumerate(no_of_defects, 1):
    ax.text(1,y,v)
plt.show()


# In[ ]:


# Note: it can display only 10 images in 2 cols
def display_images_in_two_cols(image_file_names):
    nrows=(len(image_file_names)//2)
    fig, ax = plt.subplots( nrows=nrows, ncols=2, figsize=(22, 2*nrows))
    ax = ax.flatten()
    for i in range(len(image_file_names)):
        img = plt.imread('/kaggle/input/severstal-steel-defect-detection/train_images/'+image_file_names[i])
        plt.tight_layout()
        ax[i].axis("off")
        ax[i].imshow(img)

# Note: it can display given images in 1 column        
def display_images(filenames):
    for i in range(len(filenames)):
        img = plt.imread('/kaggle/input/severstal-steel-defect-detection/train_images/'+filenames[i])
        plt.figure(figsize=[19,65])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)        


# In[ ]:



filenames = os.listdir('/kaggle/input/severstal-steel-defect-detection/train_images')
display_images_in_two_cols(filenames[0:12])


# In[ ]:


filenames = os.listdir('/kaggle/input/severstal-steel-defect-detection/train_images')
display_images(filenames[0:12])


# ### Reshaping-1 Training Data Set
# ImageId_ClassId, EncodedPixels --> ImageId,D1,D2,D3,D4,Count

# In[ ]:


path = '../input/severstal-steel-defect-detection/'
train = pd.read_csv(path + 'train.csv')

import numpy as np

# Unstack the EncodedPixels and store into new Dataframe
train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('_')[0])
train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['D1'] = train['EncodedPixels'][0::4].values
train2['D2'] = train['EncodedPixels'][1::4].values
train2['D3'] = train['EncodedPixels'][2::4].values
train2['D4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna('',inplace=True); 
train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1)
disp(train2.head())
print()
print("Shape:",train2.shape)
print("\nNumber of Defects for each Defect Class:")
print(np.sum(train2.iloc[:,1:]!='',axis=0))
ZeroDefectsImages = train2[train2['count']==0].ImageId.values
disp("Zero Defect Images:")
disp(ZeroDefectsImages.shape)
disp(ZeroDefectsImages)


# ### Reshaping-2 Training Data Set

# In[ ]:


import numpy as np 
# reading in the training set
data = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

# isolating the file name and class
data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_').str
data['ClassId'] = data['ClassId'].astype(np.uint8)

# keep only the images with EncodedPixels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows')
disp( squashed)
print(squashed.shape)

# squash multiple rows per image into a list
squashed = squashed[['ImageId', 'EncodedPixels', 'ClassId']].groupby('ImageId', as_index=False).agg(list) 

# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))

# select images has more than 2 defects
disp(squashed[squashed['Distinct Defect Types']> 2])

# display squashed sample
disp(squashed.sample(10))
print(f"Number of images with at least One defect: {squashed.shape[0]}")
print(squashed.shape)

# Discovering Zero Defect image file names
tempdf = data[['ImageId', 'EncodedPixels', 'ClassId']].fillna("RAM")
tempdf = tempdf[['ImageId', 'EncodedPixels', 'ClassId']].groupby('ImageId', as_index=False).agg(list)
tempdf['all_nan'] = tempdf.EncodedPixels.apply(lambda x: x.count("RAM") )
ZeroDefectsImages=tempdf[tempdf["all_nan"]==4].ImageId.values
print(f"Number of images with Zero defect: {ZeroDefectsImages.shape[0]}")
disp(ZeroDefectsImages.shape)
disp(ZeroDefectsImages)


# ## Model Training
# 

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from albumentations import Compose
from albumentations.pytorch import ToTensor
import cv2


# In[ ]:


path = '../input/severstal-steel-defect-detection/train_images/'
#path = '../input/severstal-steel-defect-detection/test_images/'
sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images/"
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


# ### Dataframe (ImageId, ClassId)
# #### creating Dataframe based on defects types (minimum one defect type)

# In[ ]:


np.random.seed(123)
udf = pd.DataFrame()
udf['ImageId'] = squashed[squashed['Distinct Defect Types'] == 1].ImageId
udf['ClassId'] = squashed[squashed['Distinct Defect Types'] == 1].ClassId
udf.ClassId = udf.ClassId.apply(lambda x: x[0])

# Converting ClassId (1,2,3,4) into Labels (0,1,2,3)
udf1 = udf[udf.ClassId == 1]
udf1["ClassId"] = 0
udf2 = udf[udf.ClassId == 2]
udf2["ClassId"] = 1
udf3 = udf[udf.ClassId == 3]
udf3["ClassId"] = 2
udf4 = udf[udf.ClassId == 4]
udf4["ClassId"] = 3
print("Single Defect Count:")
print("Class1:{}, Class2:{}, Class3:{}, Class4:{}"
      .format( len(udf1), len(udf2), len(udf3), len(udf4)))


# (Solving Class Imbalance problem)
# Since Defect 2 type images are lesser than other defects, we are reusing the same images 
# udf2 = pd.concat([udf2,udf2,udf2])  

sample_size = len(udf2)
udf1 = udf1[0:sample_size]
udf2 = udf2[0:sample_size]
udf3 = udf3[0:sample_size]
udf4 = udf4[0:sample_size]

udf = pd.concat([udf1, udf2, udf3,  udf4 ])
udf.reset_index(drop=True,inplace=True)

print()
print("Label Distribution")
print("Total Rec:{}, Class1:{}, Class2:{}, Class3:{}, Class4:{}" 
      .format( len(udf),
               len(udf[udf["ClassId"] == 0]),
               len(udf[udf["ClassId"] == 1]),
               len(udf[udf["ClassId"] == 2]), 
               len(udf[udf["ClassId"] == 3])))


train_df, val_df = train_test_split(udf, test_size=0.2, stratify=udf["ClassId"], random_state=39)
print()
print("Train DF Shape:",train_df.shape)
print("Val DF Shape:  ",val_df.shape)


# ### Utility Functions 
# #### (mask2rle, rle2mask, draw_mask, draw_contour, draw_plot, draw_OMG)

# In[ ]:


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, imgshape=(256,1600)):
    width = imgshape[0]
    height= imgshape[1]
    mask= np.zeros( width*height ).astype(np.uint8)    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1 
    mask = mask.reshape(height,width)
    return mask.T


def draw_mask(imageid, rle, classid):
    masks = np.zeros((256,1600,4), dtype=np.uint8)
    for r, c in zip(rle,classid):
        index = c - 1
        masks[:,:,index] = rle2mask(r) 
    return masks


def draw_contour(image, masks):
    for i in range(4):
        contours, hierarchy = cv2.findContours(masks[:,:,i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, palet[i], 2)        
    return image

def draw_plot(img, imageid, classid):
    plt.figure(figsize=(20,5))
    plt.axis("off")
    plt.title(imageid+ " "+str(classid))
    plt.imshow(img)


def draw_OMG(imageids):
    """ Drawing Original Image, Masked Image , Guided Image (Contour Image)
    """
    for imageid in imageids:
        try:
            rle = squashed[squashed.ImageId == imageid].EncodedPixels.tolist()[0]
            classid = squashed[squashed.ImageId == imageid].ClassId.tolist()[0]

            img = cv2.imread(path+imageid)
            draw_plot(img, imageid, classid)

            masks=draw_mask(imageid,rle,classid)
            img = draw_contour(img,masks)
            draw_plot(img, imageid, classid)

            
            for i in range(4):
                for j in range(3):
                    img[masks[:,:,i]==1,j] = palet[i][j]                    
            draw_plot(img, imageid, classid)
            
        except IndexError:
            try:
                img = plt.imread(path+imageid)
                plt.imshow(img)
            except FileNotFoundError:
                print("File Not Found:"+imageid)


# In[ ]:


# Visualization Original, Mask, Guided(Contour) 

conditions = [
    squashed['ClassId'].astype(str)=='[1]',
    squashed['ClassId'].astype(str)=='[2]',
    squashed['ClassId'].astype(str)=='[3]',
    squashed['ClassId'].astype(str)=='[4]',
    squashed['Distinct Defect Types']==2,
    squashed['Distinct Defect Types']==3
]
sample_size = 2
for condition in conditions:
    sample = squashed[condition].sample(sample_size) 
    draw_OMG(sample.ImageId.values)

draw_OMG(["000f6bf48.jpg", "db4867ee8.jpg", "0025bde0c.jpg"]) # 4, 1 2 3,  3 4
draw_OMG(["a0906d0b3.jpg"])


# ### Dataset & Data Loader 

# In[ ]:


mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
phase="train"
_tasks = Compose([
    ToTensor(),
    #Normalize(mean=mean, std=std),
    ])
class TrainDataset(Dataset):
    
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms
        self.fnames = self.df["ImageId"]
        
    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].ImageId 
        label = self.df.iloc[idx].ClassId
        img = plt.imread(path+image_id)
        img = self.transforms(image=img)               
        return img,label,image_id
    
    def __len__(self):
        return len(self.fnames)

    
train_dataset = TrainDataset(train_df, _tasks)
val_dataset = TrainDataset(val_df, _tasks)


tr_sampler = SubsetRandomSampler(range(len(train_df)))
val_sampler = SubsetRandomSampler(range(len(val_df)))


batch_size = 16 
num_workers = 4

train_loader = DataLoader(train_dataset,batch_size=batch_size,
                          num_workers=num_workers,pin_memory=True,
                          shuffle=False,sampler=tr_sampler)
val_loader = DataLoader(val_dataset,batch_size=batch_size,
                        num_workers=num_workers,pin_memory=True,
                        shuffle=False,sampler=val_sampler) 

print("Done", np.random.randint(1000))


# ### Visualization - From data loader (batch by batch)

# In[ ]:


def draw_loader_raw(k):
    batch_size = len(k[1])
    filenames = k[2]
    print(filenames)
    print("Image Shape:",k[0]["image"].shape)
    print("Label Shape:",k[1].shape)
    print("Batch Size:",batch_size)
    counter = np.zeros(4)
    for i in range(batch_size):        
        plt.figure(figsize=(20,5))
        plt.title("Class Id: "+str(k[1][i].item()+1)+ "  "+k[2][i])
        plt.imshow(k[0]["image"][i].permute(1,2,0))        
        counter[k[1][i].item()] += 1
        draw_OMG([k[2][i]])
    print("Defect Class distribution of {} images of type [1, 2, 3, 4] respectively:".format(batch_size), counter)    



k = next(iter(train_loader))    
draw_loader_raw(k)


# ### Custom Model (extended from nn.Module)

# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        ## define the layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)       
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(256*1600, 512)
        self.linear2 = nn.Linear(512, 24)
        self.linear3 = nn.Linear(24, 4)
        
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = Model()
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# ### Training

# In[ ]:


import torch.optim as optim
loss_function = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-6, momentum = 0.9, nesterov = True)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-6, betas=(0.9,0.999), eps=1e-08,amsgrad=False)
## run for 30 Epochs 
for epoch in range(50):
    train_loss, val_loss = [], []
    
    ## training part 
    model.train()
    for data, target, imageid in tqdm(train_loader,desc="Epoch {}: ".format(epoch+1)):
        data, target = data["image"].to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)  
        loss.backward()
        optimizer.step()        
        train_loss.append(loss.item()) 
        
    # evaluation part 
    model.eval()
    for data, target, imageid in tqdm(val_loader, desc="Epoch {}: ".format(epoch+1)):
        data, target =  data["image"].to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)
        val_loss.append(loss.item())
    
    print("Train Loss:(max, min):",max(train_loss), min(train_loss))
    print("Val   Loss:(max, min):",max(val_loss), min(val_loss))
    
print("DONE")


# ### Prediction on Validation Set

# In[ ]:


## dataloader for validation dataset 
ac = [0,0,0,0] # Actual Count
pc = [0,0,0,0] # Predicted Count
count = 0
lc = 0
for data, labels, imageid in val_loader:
    lc += 1
    data, labels = data["image"].to(device), labels.to(device)
    output = model(data)    
    _, preds_tensor = torch.max(output, 1)
    actual = np.squeeze(labels.cpu().numpy()) + 1
    preds = np.squeeze(preds_tensor.cpu().numpy()) + 1
    for i, l in enumerate(actual):
        ac[l-1] += 1
        if l == preds[i]:
            count = count+1
            pc[l-1] += 1

print("Actual Count")
print("     vs")
print("Predicted Count")
print(ac)
print(pc)
print()
print("****** PERCENTAGE ********")
print((np.array(pc)/np.array(ac))*100 //1)
print()

print("******* OVER ALL *********")
print(count/len(val_df))

