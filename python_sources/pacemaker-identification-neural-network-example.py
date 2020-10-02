#!/usr/bin/env python
# coding: utf-8

# # Pacemaker identification with neural networks
# This is a sample notebook to go with the pacemaker dataset.
# This is the dataset used by the paper ["Cardiac Rhythm Device Identification Using Neural Networks"](http://electrophysiology.onlinejacc.org/content/5/5/576).

# # Load the necessary libraries
# 
# The ignite library doesn't come installed by default on kaggle, so let's install it from the pytorch channel.

# In[ ]:


get_ipython().system("conda install ignite -c pytorch --yes > /dev/null  # Send output to /dev/null so we don't have to read all the installation stuff")

import time
import torch
import datetime
import torchvision
import torch.nn as nn

from collections import deque
from torchvision import models
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision


# # Settings
# ##  Settings for our dataset
# We use the mean and standard deviation of the ImageNet dataset so our images are of similar distribution to the pre-trained models we'll load.

# In[ ]:


TRAIN_DIR = "/kaggle/input/pacemakers/Train"
TEST_DIR = "/kaggle/input/pacemakers/Test"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ## Settings for training

# In[ ]:


EPOCHS = 20  # Go through the entire dataset 20 times during training
BATCH_SIZE = 32  # How many images to show the network at once; lower this if you don't have enough GPU RAM
DEVICE = "cuda"  # We'll use a GPU to speed up training if we can; remember to turn the accelerator to "GPU" on the right. If you don't have a GPU, change "cuda" to "cpu"
VERBOSE = True  # Print progress of each training loop


# # Set up our data pipeline
# 
# For training examples we'll use data augmentation to distort the images and make them look a bit different, so the neural network sees more examples, effectively.
# For the testing set, we won't adulterate them (so we can judge a more 'real world' performance)

# In[ ]:


transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomAffine(degrees=5,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05),
                            shear=5),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

transforms_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_data = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transforms_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_data = torchvision.datasets.ImageFolder(TEST_DIR, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

n_classes = len(train_data.classes)


# # Preview our data

# In[ ]:


plt.figure(figsize=(16,10))  # Larger plot size

img = train_data[0][0].numpy().transpose((1, 2, 0))
img = STD * img + MEAN

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Training set example")

img = test_data[0][0].numpy().transpose((1, 2, 0))
img = STD * img + MEAN

plt.subplot(2, 2, 2)
plt.imshow(img)
plt.axis('off')
plt.title("Testing set example")


# # Create our network
# We'll use DenseNet121 (Xception used in the original paper isn't in the Pytorch model zoo, sadly, and DenseNet is still very nice).
# Because this network will have been trained on ImageNet which has 1000 classes, and we are training on our pacemakers which have 45 classes, we need to replace the final layer of the network with a layer with 45 outputs.

# In[ ]:


model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, n_classes)
model = model.to(DEVICE)


# # Set up training scheme
# Here we tell it we want it to calculate its performance using CrossEntropyLoss (because it's a categorical problem).
# We're going to use the Ignite framework here just to make our training loops a little easier.

# In[ ]:


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad))

trainer = create_supervised_trainer(model,
                                    optimizer,
                                    loss,
                                    device=DEVICE)

evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': Accuracy(),
                                                 'loss': Loss(loss),
                                                 'precision': Precision()},
                                        device=DEVICE)


# # Set up some functions to print out our progress
# These are called 'callbacks', or 'hooks'. Ignite will run these funcstions when certain things happen, e.g. end of an epoch (every cycle, i.e. time the network's been trained a full copy of the dataset), or every iteration (every 'batch' of pictures).

# In[ ]:


@trainer.on(Events.STARTED)
def initialise_custom_engine_vars(engine):
    engine.iteration_timings = deque(maxlen=100)
    engine.iteration_loss = deque(maxlen=100)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    engine.iteration_timings.append(time.time())
    engine.iteration_loss.append(engine.state.output)
    seconds_per_iteration = np.mean(np.gradient(engine.iteration_timings)) if len(engine.iteration_timings) > 1 else 0
    eta = seconds_per_iteration * (len(train_loader)-(engine.state.iteration % len(train_loader)))
    if VERBOSE:
        print(f"\rEPOCH: {engine.state.epoch:03d} | "
              f"BATCH: {engine.state.iteration % len(train_loader):03d} of {len(train_loader):03d} | "
              f"LOSS: {engine.state.output:.3f} ({np.mean(engine.iteration_loss):.3f}) | "
              f"({seconds_per_iteration:.2f} s/it; ETA {str(datetime.timedelta(seconds=int(eta)))})", end='')
            
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    acc, loss, precision = metrics['accuracy'], metrics['loss'], metrics['precision'].cpu()
    print(f"\nEnd of epoch {engine.state.epoch:03d}")
    print(f"TRAINING Accuracy: {acc:.3f} | Loss: {loss:.3f}")
    
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    acc, loss, precision = metrics['accuracy'], metrics['loss'], metrics['precision'].cpu()
    print(f"TESTING  Accuracy: {acc:.3f} | Loss: {loss:.3f}\n")


# # Now train!

# In[ ]:


trainer.run(train_loader, max_epochs=EPOCHS)


# We seem to have achieved a respectable accuracy of over 90% on the testing set at some stages, which is good considering there are 45 classes.
# 
# You may well see accuracies above those reported in our paper when you train this network - one reason for this is because in our paper we did not continuously measure performance against the testing set during training (we used a proportion of the training set to do this), but only once at the end. This the 'correct' practice, because it prevents "lucky" runs being reported as the true accuracy.

# # Make some predictions to test our network
# Here we will take an example of each of the classes in the testing dataset and run it through the network.

# In[ ]:


model.eval()

plt.figure(figsize=(20,50))  # Larger plot size

for i_class in range(n_classes):
    
    i_img = i_class * 5  # 5 examples per class
    img_tensor, _ = test_data[i_img]
    img_numpy = img_tensor.numpy().transpose((1, 2, 0))
    img_numpy = STD * img_numpy + MEAN
    
    with torch.no_grad():
        predictions = model(torch.unsqueeze(img_tensor, 0).to(DEVICE))
        predicted_class = torch.argmax(predictions).cpu().numpy()
    
    true_class = test_data.classes[i_class][:20]
    pred_class = test_data.classes[predicted_class][:20]
    correct = "CORRECT" if true_class == pred_class else "INCORRECT"
    
    plt.subplot(9, 5, i_class+1)
    plt.imshow(img_numpy)
    plt.axis('off')
    plt.title(f"{correct}\nTrue class: {true_class}\nPredicted class: {pred_class}")
    
plt.subplots_adjust(wspace=0, hspace=1)

