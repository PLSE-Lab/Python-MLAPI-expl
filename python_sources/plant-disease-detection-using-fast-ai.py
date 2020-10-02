#!/usr/bin/env python
# coding: utf-8

# This is a comprehensive demonstration of fastai on an image dataset. [Fast.ai](https://github.com/fastai/fastai) is a deep learning library which uses pytorch as its backend. So, in many ways, it is more modern and robust than tensorflow based libraries like keras.

# For this analysis, a plant disease dataset is being used to detect and diagose plant based diseases. Plant diseases are a major threat to food security and it is important to diagnose these diseases early and accurately. This can lead to a better future for food self-sufficiency, efficient waste management etc. The domain of plant disease diagnosis has experienced a huge progress as technology has evolved -- from visual depiction of symptoms to a molecular level detection. In this notebook, we are looking to detect these diseases from images of the leaves of healthy and diseased plants.

# # Importing Libraries and Looking at the Data

# In[ ]:


from fastai import *
from fastai.vision import *
import os
from os import listdir
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
path = "../input/plantvillage/PlantVillage/"
os.listdir(path)


# So, in the dataset, we will be largely looking at the healthy and diseased variants of the leaves of tomato, potato and pepper. This is a classification problem where we have 15 labels or ground truths.

# In[ ]:


path = Path(path); path


# In[ ]:


directory_root = '../input/plantvillage/'
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(image_directory)
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# After getting the data, we can use the `get_transforms` function to add variation, say, flipping the image vertically.

# In[ ]:


tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)


# It is very essential that one figures out how to get the labels, i.e. the ground truths from which the model can learn. Different kinds of datasets will have different ways to get the labels.

# Let's take a sample file and get its label

# In[ ]:


file_path = '../input/plantvillage/PlantVillage/Potato___Early_blight/0faca7fe-7254-4dfa-8388-bbc776338c9c___RS_Early.B 7929.JPG'


# In[ ]:


dir_name = os.path.dirname(file_path)


# One can see that the label is present in the file path itself (`Potato___Early_blight`) and to get it we can split the string with respect to `/` and get all the words separated by `/` into a list. From that list we can easily access the label.

# In[ ]:


dir_length = len(dir_name.split("/"))
dir_name.split("/")


# In[ ]:


dir_name.split("/")[dir_length - 1]


# We got the label!
# Now, to generalise for all image files, we can create a function `get_labels` that performs the above steps.

# In[ ]:


def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_length = len(split_dir_name)
    label  = split_dir_name[dir_length - 1]
    return(label)


# Now that we are ready with the images and their corresponding labels, we can generate the data. We will also normalise so that all the pixel values have the same mean and standard deviation. This will help the model to learn more easily and faster.

# In[ ]:


data = ImageDataBunch.from_name_func(path, image_list, label_func=get_labels,  size=224, 
                                     bs=64,num_workers=2,ds_tfms=tfms)
data = data.normalize()


# Let's take a look at a random sample of the data.

# In[ ]:


data.show_batch(rows=3, figsize=(15,11))


# # Training the Model

# We have successfully preprocessed the data by getting the labels and generating the images. Now, we are ready to train the model. To train, the transfer learning method is being used, where one uses a pre-trained model for handling the task at hand. For this problem, the CNN architecture Resnet34 is being used. It has 34 layers in its neural network architecture and is trained on the ImageNet dataset.

# The Resnet or Residual Network essentially uses shortcut networks on top of feed forward networks. In plain feed forward networks, as the depth of the network increases the accuracy tends to get saturated. The residual networks do away with accuracy saturation by adding the shortcut networks.

# ![](https://neurohive.io/wp-content/uploads/2019/01/resnet-architecture-3.png)

# [This article](https://neurohive.io/en/popular-networks/resnet/) helps in understanding Resnets in a detailed manner.

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/models/')


# Now we make this model learn for 10 epochs. Fastai randomly selects training and validation sets so that the user does not have to. Thus, it makes the coding much easier and smaller.

# In[ ]:


learn.fit_one_cycle(10)


# In just 10 epochs, the model achieves a substantially high accuracy! Now let's see what were the most confident but wrong predictions so that we can further fine tune.

# In[ ]:


interpretation = ClassificationInterpretation.from_learner(learn)
losses, indices = interpretation.top_losses()
interpretation.plot_top_losses(4, figsize=(15,11))


# We get a basic idea about what two labels the model is most confused about, i.e. how confident the model is at predicting the wrong value. e.g. the model predicts a plant to be a potato suffering from late blight when in fact it is actually a tomato suffering from early blight. In order to see how much the model is confused, we take a look at the confusion matrix.

# In[ ]:


interpretation.plot_confusion_matrix(figsize=(12,12), dpi=60)


# Most of the time, the model predicts the output correctly (the darkest blocks), but sometimes it gets confused and predicts the wrong value with a high confidence. We can look at the labels that the model is most confused with.

# In[ ]:


interpretation.most_confused(min_val=2)


# # Fine Tuning

# In[ ]:


learn.save('classification-1')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1)


# As soon as we let the model un-learn, its error rate becomes significantly higher. So, one should find an optimum learning rate at which the model should learn.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# From the plot, it is clear that the loss increases very fast as the learning rate increases beyond 0.001. So, let's keep the learning rate within a range of 0.000001 to 0.001

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-3))


# On fine tuning, we achieve a much higher accuracy and a much lower error rate. Let's again look at the confidently wrong predictions.

# In[ ]:


interpretation = ClassificationInterpretation.from_learner(learn)
losses, indices = interpretation.top_losses()
interpretation.plot_top_losses(4, figsize=(15,11))


# In[ ]:


interpretation.most_confused(min_val=2)


# In[ ]:


learn.save('resnet34-classifier.pkl')


# Clearly, the model is now much less confused as compared to the one before fine tuning because it has much less instances of confident but wrong predictions than before. In other words, there are much less predictions of confidently wrong predictions, so, the model is becoming more confidently right!

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


os.chdir("/tmp/models/")

