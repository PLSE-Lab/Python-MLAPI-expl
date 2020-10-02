#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates just how easy it is to create a machine learning model using the fast.ai library.
# 
# To train the model you must make sure kaggle uses it's GPU rather than CPU which can be specified when creating a notebook, or selecting the GPU tab under Setting.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import the fast.ai library
from fastai.vision import *


# The first step, once we have our data, is to create a path to where the data is stored. 
# 
# With this dataset of X-ray images I'm only using the test folder as there are less files compared to the train folder. I'm not going to spend much of my time training this model and fast.ai is capable of still achieving remarkable results.

# In[ ]:


path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/')
path.mkdir(parents=True, exist_ok=True)


# The two files in the specified path are NORMAL and PNEUMONIA. These are our classes and is what our machine learning algorithm will try to differentiate.

# In[ ]:


path.ls()


# We now create our data bunch. The train parameter usually looks for a folder called train but because we want to use a small part of the data, I've added valid_pct=0.2 to make 20% of the images in our test folder, the test set and the rest will be used to train. Hopefully that's not too confusing.

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# Lets have a look at some of the images.

# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# Below we have listed the names of our classes, the number of classes, and the number of images in our training and validation sets.

# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# The model we're going to use is resnet34 and to download from Kaggle we need to make sure the Internet switch under settings is swtiched on.
# 
# We'll then create our convolutional neural net learner. 
# 
# When saving your model the cnn_learner will default to saving it in the current path directory. We can do this here because the dataset it read-only, therefor we specify the path using the model_dir parameter.

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=Path('/kaggle/working'))


# We will now train a part of our model for 4 epochs.

# In[ ]:


learn.fit_one_cycle(4)


# We can see our error rate is 13%. Not bad but can definately be improved. We'll now save our model and unfreeze, which means it will now train using all our data.

# In[ ]:


learn.save('stage-1')
learn.unfreeze()
learn.lr_find()


# We've plotted the learning rate finder below. A general rule is to find the steepest negative line just prior to it trending upwards.

# In[ ]:


learn.recorder.plot()


# I initally chose a slice of 1e5 to 2e-4 and trained for 3 epochs. This gave me a great error rate of 2.4% but the training loss was huge compared to the validation loss which showed it was underfitting. I then changed the learning rate to 3e-5 to 6e-4 and found the training and validation losses more similar, and an error rate of 3.2%. This is great improvement on the previous 13% error rate.

# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(3e-5,6e-4))


# Again we save the model so if we come back we don't have to retrain everything.

# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.load('stage-2');


# We plot a confusion matrix to see how well our model has predicted our validation set.
# 
# Sensitivity	0.8793	TPR = TP / (TP + FN)
# Specificity	1.0000	SPC = TN / (FP + TN)
# Precision	1.0000	PPV = TP / (TP + FP)
# 
# So it has predicted every normal X-ray correctly as normal, and 7 of the x-rays which showed pneumonia the model didn't spot and classified incorretly as normal.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# We could continue to train our model, even selecting a new data bunch and adding it to the cnn learner, known as transfer learning. Perhaps from the train folder this time.
# 
# But for now we will export our model. This will create a pickle file in the working directory called export.pkl.
# 
# If your are confused as where the working directory is on kaggle, your models will only appear once you have commited your notebook.

# In[ ]:


learn.path = Path('/kaggle/working')
learn.export()


# Now I want to test our model on some of the images inside the training folder.
# 
# First we'll change the path.

# In[ ]:


path = Path('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train')


# Next I have chosen 10 images, 5 from the folder labelled Pneumonia, and 5 from the folder labelled Normal and created two lists.

# In[ ]:


PNEUMONIA = ['person63_bacteria_306.jpeg',
'person26_bacteria_122.jpeg',
'person890_bacteria_2814.jpeg',
'person519_virus_1038.jpeg',
'person968_virus_1642.jpeg']

NORMAL = ['IM-0757-0001.jpeg',
'IM-0540-0001.jpeg',
'IM-0683-0001.jpeg',
'NORMAL2-IM-1288-0001.jpeg',
'NORMAL2-IM-0482-0001.jpeg',]


# In[ ]:


learn = load_learner(Path('/kaggle/working'))


# In[ ]:


print('These 5 X-rays have Pneumonia')
for l,i in enumerate(PNEUMONIA):
    img = open_image(path/'PNEUMONIA'/i)
    pred_class,pred_idx,outputs = learn.predict(img)
    if str(pred_class) == 'PNEUMONIA':
        print(f'Prediction for image {l+1} is Correct')


# In[ ]:


print('These 5 X-rays are normal and do not show any signs of Pneumonia')
for l, i in enumerate(NORMAL):
    img = open_image(path/'NORMAL'/i)
    pred_class,pred_idx,outputs = learn.predict(img)
    if str(pred_class) == 'NORMAL':
        print(f'Prediction for {l+6} is Correct')


# As shown above, the model correctly identified all images.
