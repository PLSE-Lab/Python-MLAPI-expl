'''

# ------------------------------------  to unzip the data --------------------------------------------
import os,shutil
from PIL import Image
path='../output/test/'
for q in list(os.listdir(path)):
    #a=1
    for w in list(os.listdir(path+q)):
        
        #dst =path+q +"/"+ str(a) + ".png"
        src =path+q+"/"+ w 
        #dst =q+ dst 
        #os.rename(src, dst) 
        #a += 1 
        
        try:
            im=Image.open(src)
            # do stuff
        except:
            os.unlink(src)
            
        

        

#print(os.listdir("../output/train"))
#shutil.rmtree("../output/train/")

# Any results you write to the current directory are saved as output.
#import os
import sys
import tarfile

def maybe_extract(filename, dataset,force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall('../output')
    tar.close()
    os.rename('../output/'+root[9:],"../output/"+dataset)
    print("Extraction completed.")
    return "../output/"+dataset
  
train_folders = maybe_extract('../input/notMNIST_large.tar.gz',dataset='train')
test_folders = maybe_extract('../input/notMNIST_small.tar.gz',dataset='test')
#display(Image(filename='../output/test/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png'))


# ----------------------------- to check the size of the image ---------------------------------------
import matplotlib.pyplot as plt
import numpy as np

img=plt.imread('../output/test/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png')
img.shape # --> 28*28


# ---------------------------- create a neural network model ------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

# define a model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# normalize the data
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

# create datasets                              
train_data_dir='../output/train'
test_data_dir='../output/test'

traindataset = datasets.ImageFolder(train_data_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=64, shuffle=True)

testdataset = datasets.ImageFolder(test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=64, shuffle=True)

class=('A','B','C','D','E','F','G','H','I','J')
dataiter = iter(trainloader)
ima, lab = dataiter.next()
print(type(ima))
print(ima.shape)
print(lab.shape)
plt.imshow(ima[1].numpy().squeeze(), cmap='Greys_r')
print(class[lab[1]])
print(lab[1])

print(len(trainloader))


# train the model
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        #print("y")
        images = images.view(images.shape[0], -1)
        
        # to view an image
        # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        
# test the model
with torch.no_grad():
            model.eval()
            total=0
            matched=0
            l=0
            for images, labels in testloader:
                total+=1
                plt.imshow(images[11].numpy().squeeze(), cmap='Greys_r')
                print(labels[10])
                print(os.listdir('../output/test/'))
                try:
                    ima = images[l].view(1, 784)
                    log_ps = model(ima)
                    ps = torch.exp(logps)
                    a=ps.numpy()
                    m=int(np.where(a==a.max())[1])
                    if int(labels[l])==m:
                        matched+=1
                    print("image with label",int(labels[l]), "matches",m,"with probability", round(a.max(),2))
                    l+=1
                    
                except:
                    l+=1
                    #continue
                

print(total,matched)

'''