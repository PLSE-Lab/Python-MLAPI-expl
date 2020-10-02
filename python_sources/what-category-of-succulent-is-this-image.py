#!/usr/bin/env python
# coding: utf-8

# # Classify images of succulents into Aloevera, Jade and Burro using image classification
# 
# ![image.png](attachment:image.png)

# ## This kernel uses the fastai vision libraries
# ### 1. Read input URLs for the three varieties of succulents as input and download the images
# ### 2. Split the input images into train, validation and test sets
# ### 3. Transform and normalize the images
# ### 4. Train a model using resnet architecture
# ### 5. Interpret results by creating a confusion matrix 
# ### 6. Predict using the model
# ### 7. Export the learner object to deploy the model on another machine
# 
# ### My learning from this project:
# At first when i opted for classes Aloevera, Jade and Plush, there was not much distinction between the classes Jade and Plush. This and the image data having un-pruned images gave a metric error of around 20 - 30%
# 
# After using a snippet bookmark to choose google images and also changing the classes to Aloevera, Jade and Burro the metric error reduced to 8%
# 
# High number of epochs causes overfitting.
# Choosing Learing Rate where the slope is the steepest gives low error.

# In[ ]:


from fastai.vision import *
from fastai import *


# In[ ]:


folder = 'aloevera'
file = 'urls_aloevera.txt'
path = Path('data/succulents')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
classes = ['aloevera','jade','burro']
download_images("../input/urls_aloevera.txt", dest, max_pics=200)


# In[ ]:


folder = 'jade'
file = 'urls_jade.txt'
path = Path('data/succulents')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
download_images("../input/urls_jade.txt", dest, max_pics=200)


# In[ ]:


folder = 'burro'
file = 'urls_burro.txt'
path = Path('data/succulents')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
download_images("../input/urls_burro.txt", dest, max_pics=200)


# In[ ]:


os.listdir("../input/")


# In[ ]:


path.ls()


# In[ ]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# ### ImageDataBunch creates the train, validation and test sets from input. 
# ### get_transforms - transforms all images to size 224x224, it also centers, crops and zooms the images
# ### Images are normalized to have pixels with mean 0 and std 1

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# In[ ]:


# If you already cleaned your data, run this cell instead of the one before
#np.random.seed(42)
#data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels='cleaned.csv',        
#                               ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# ### A look at our input images

# In[ ]:


data.show_batch(rows=2, figsize=(5,5))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ### Training a model using resnet architecture

# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# ### Train the model. Training means creating a set of weights that fit our data well. This trains only the last few layers of our model. Here we run 5 times through our entire set of training images. Error rate is around 9%, which means accuracy achieved is around 91%.

# In[ ]:


learn.fit_one_cycle(5)


# ### Save model weights so that you can reload them later

# In[ ]:


learn.save('stage-1')


# ### Analyzing results: Create confusion matrix and plot the misclassified images.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize=(7,8))


# ### We can make the model better by unfreezing and training the whole model

# In[ ]:


learn.unfreeze()


# ### If we try to train it now from the scratch, the error rate is higher.

# In[ ]:


learn.fit_one_cycle(1)


# ### Load the stage-1 weights saved earlier, unfreeze and then find the learing rate.

# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# ### Error rate now marginally reduces

# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# In[ ]:


learn.save('stage-2')


# ### Analyzing results: Create confusion matrix and plot the misclassified images.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# ### Lets find out how well our model is able to predict the succulent classes

# In[ ]:


img = open_image(path/'aloevera'/'00000050.jpg')
img.show(figsize=(4, 4))


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# ### The model has correctly predicted the image as belonging to succulent class aloevera!
# ### Now lets see how well it recognises the classes burro and jade.

# In[ ]:


img = open_image(path/'burro'/'00000015.jpg')
img.show(figsize=(4, 4))


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:


img = open_image(path/'jade'/'00000001.jpg')
img.show(figsize=(4, 4))


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# ### Export all the model information and weights learnt by the model. This export.pkl file can be used to deploy the model on a different machine.

# In[ ]:


#learn.model = learn.model.cpu()
learn.export()


# In[ ]:


path.ls()


# ### Load the learner object from export.pkl and predict.

# In[ ]:


learn = load_learner(path)


# In[ ]:


defaults.device = torch.device('cpu')


# In[ ]:


img = open_image(path/'jade'/'00000002.jpg')
img.show(figsize=(4, 4))


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# ### Thank you for viewing this kernel. Please upvote if you enjoyed it!

# Delete all output files

# In[ ]:


import shutil
shutil.rmtree("./data/succulents")

