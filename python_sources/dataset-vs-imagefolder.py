#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image

from tqdm import tqdm_notebook as tnote
torch.manual_seed(42)


# In[ ]:


from zipfile import ZipFile
with ZipFile('../input/images.zip', 'r') as zipObj: # Unzipping images
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


# # Dataset Method

# In[ ]:


print(os.listdir("../input"))


# Lets take a quick look at how the data is structured.

# In[ ]:


train_path = "../input/train.csv.zip"
data = pd.read_csv(train_path)
data.head()


# The only two columns we are interested are id and species columns.

# In[ ]:


data.loc[:,['id','species']].head()


# 1. To load data into Pytorch we are going to define a class which inherits from the `data.Dataset` class.
# 2. By creating this class we will be able to use the `DataLoader` which simplifies the training/validatation process. 
# 3. But first we need to implement `__len__` and `__getitem__` methods for the `LeafLoader` object

# In[ ]:


class LeafLoader(Dataset):
    """Loads the Leaf Classification dataset."""

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # First 2 columns contains the id for the image and the class of the image
        self.dict = self.data.iloc[:,:2].to_dict()
        # When we index we want to get the id
        self.ids = self.dict["id"]
        

        self.classes = self.data["species"].unique() # List of unique class names
        self.class_to_idx = {j: i for i, j in enumerate(self.classes)} 
        # Assigns number to every class in the order which it appears in the data
        self.species = self.dict["species"]
        # Use this go back to class name from index of the class name
        self.path_leaf = "images" # Where the images are stored

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()
            assert isinstance(idx, int)

        num = self.ids[idx] # Id of the indexed item
        loc = f"/{num}.jpg"
        label = self.dict["species"][idx] # Find the label/class of the image at given index
        label = self.class_to_idx[label] # Convert it to int
        image = Image.open(self.path_leaf + loc)
        if self.transform:
            image = self.transform(image)

        return (image, label)


# We can then apply our standard transformations and load data into `DataLoader`

# In[ ]:


image_size = (28,28)
normalize = ((0.5), (0.5))

transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(), transforms.Normalize(*normalize)])
dataset = LeafLoader(train_path,transform)

train_size = int(0.8 * len(dataset)) # 80% of the data to be used for training
test_size = len(dataset) - train_size # The remainder for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Function above takes dataset, and lengths of train,test as input that's what we a supplying here

batch_size = 16
trainloader_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Now we can plot a batch of images to check if it's working properly

# In[ ]:


def subplot_random():
    im, lab = next(iter(trainloader))
    fig=plt.figure(figsize=(15, 15))

    for idx,(i,j) in enumerate(zip(im,lab)):
        idx +=1
        ax = fig.add_subplot(4,4,idx)
        ax.imshow(i.squeeze().numpy())
        ax.set_title(dataset.idx_to_class[j.item()])
    plt.show()

subplot_random() # We plot a batch using this helper function


# # Image Folder Method

# Image Folder Method is expecially useful when your data is structed in the following way.
# - root/Acer_Capillipes/1196.jpg
# - root/Acer_Capillipes/227.jpg
# - root/Acer_Capillipes/990.jpg
# - .
# - .
# - .
# - root/Zelkova_Serrata/1410.jpg
# - root/Zelkova_Serrata/718.jpg
# - root/Zelkova_Serrata/1136.jpg
# 
# 
# However, in this example the data is not structured in that way so we will get it in that form.

# In[ ]:


import shutil # To copy files from one directory to another


# In[ ]:


# Create a list of species to iterate on
labels = data.species.values.tolist() # Labels are the species of the leafs
def make_folders(verbose=False):
    folder_count = 0
    root = 'Data/'
    print('Total labels = ',len(set(labels)))
    for i in set(labels):
        os.makedirs(f'{root}{i}') # Make directories similar to Data/class_name
        folder_count += 1
    print("Total folders = ", folder_count )
    print(f"Root is {root}")
make_folders()


# In[ ]:


# Since we know that we have 10 images for each class we can define a function that splits  
# the list once it reaches a length of 10
def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i:i + n]


# In[ ]:


species_list = data.sort_values('species').species.unique().tolist() # Unique species
id_list = list(create_chunks(data.sort_values('species').id.values.tolist(),10)) # list of lists with sublist length of 10
dict_train = dict(zip(species_list,id_list))


# In[ ]:


# Checks if the data is correct
for key,val in dict_train.items():
    assert sorted(data[data.species == key].id.tolist()) == sorted(val)


# In[ ]:


for item,key in dict_train.items():
    for i in range(10):
        path1 = f'images/{str(dict_train.get(item)[i])}.jpg'
        path2 = f'Data/{item}'
        shutil.copyfile(path1,path2+'/'+str(dict_train.get(item)[i])+'.jpg')


# In[ ]:


root = 'Data/'
transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize(*normalize)
                               ])
dataset_fold = ImageFolder(root, transform= transform)
train,valid = random_split(dataset,[train_size,test_size])

# To load our data in batches
train_loader_folder = DataLoader(train, batch_size=16, shuffle=True)
valid_loader_folder = DataLoader(valid, batch_size=16, shuffle=False)


# In[ ]:


assert len(trainloader_dataset) == len(train_loader_folder)
assert len(testloader_dataset) == len(valid_loader_folder)


# In[ ]:


shutil.rmtree("Data")
shutil.rmtree("images")


# # References:
# `create_chunks(list_name, n):` from [DataCamp](https://www.datacamp.com/community/tutorials/lists-n-sized-chunks)
# 
