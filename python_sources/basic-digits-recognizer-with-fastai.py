#!/usr/bin/env python
# coding: utf-8

# # Updates
# * We are going to use ResNet50 instead of Resnet34. On the last version the training loss was a little bit higher than validation loss. It could be a sign of underfitting so let's see if increasing the capacity of the network can solve this. **< Top27% Using ResNet50**
# * This time we are going to check if data augmentation using the transformations from Fast.ai can increase our ranking. We only used max_rotate, flips and no warp. **< flip_lr() and pad(reflection) reduced accuracy to 97% on validation set. Let's remove flip_lr()**. In fact, we will not flip or rotate the image at all because this might change the meaning of it.
# 
# * I was applying transformations to the validation and test sets. This can lead to bad predictions since test data should be as close to reality as possible.
# 
# * Changed transformations to include a degree of rotation

# In[ ]:


# the following three lines are suggested by the fast.ai course
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision import *
from PIL import Image


# # Import data

# In[ ]:


# Train set
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# # Exploratory data analysis

# Let's plot some of the numbers. They are 28x28 images, so we are going to get some and reshape the array.

# In[ ]:


img = [np.reshape(train.iloc[idx,1:].values,(28,28)) for idx in range(5)]


# In[ ]:


len(img)


# In[ ]:


for f in img:   
    plt.imshow(f,cmap='gray')
    plt.show()


# Let's create a dataframe with image labels and path to a image. For that we are going to save the arrays as images

# In[ ]:





# In[ ]:


#TRAIN = Path("../train")
#TEST = Path("../test")
PATH = Path('../')
TRAIN = Path('../train')
TRAIN.mkdir(parents=True, exist_ok=True)
TRAIN


TEST = Path('../test')
TEST.mkdir(parents=True, exist_ok=True)
TEST


# In[ ]:


PATH.ls()


# ### Save images

# In[ ]:


os.listdir('/kaggle/working/')


# In[ ]:


train_img = train.iloc[:,1:785]
test_img = test.iloc[:,:]


# In[ ]:


#Source: [https://www.kaggle.com/christianwallenwein/beginners-guide-to-mnist-with-fast-ai]
def save_img(data,fpath,isTest=False):
    if isTest == False:
        for index, row in data.iterrows():
    
            label,digit = row[0], row[1:]
    
            filepath = fpath
            filename = "train_{}.jpg".format(index)
            digit = digit.values
            digit = digit.reshape(28,28)
            digit = digit.astype(np.uint8)
    
            img = Image.fromarray(digit)
            img.save(filepath/filename)
            
    else:
        for index, row in data.iterrows():
    
            digit = row[:]
    
            filepath = fpath
            filename = "test_{}.jpg".format(index)
            digit = digit.values
            digit = digit.reshape(28,28)
            digit = digit.astype(np.uint8)
    
            img = Image.fromarray(digit)
            img.save(filepath/filename)


# In[ ]:


save_img(train,TRAIN,False)


# In[ ]:


sorted(os.listdir(TRAIN))[0]


# In[ ]:


save_img(test,TEST,True)


# In[ ]:


sorted(os.listdir(TEST))[0]


# In[ ]:


train['filename'] = ['train/train_{}.jpg'.format(x) for x,_ in train.iterrows()]


# In[ ]:


train.head()


# In[ ]:


test['filename'] = ['test/test_{}.jpg'.format(x) for x,_ in test.iterrows()]


# In[ ]:


test.head()


# In[ ]:


# Sanity check
img = open_image(TRAIN/'train_0.jpg')
img


# # Preprocessing the digit images and create databunch

# ### Transformations

# In[ ]:


tfms = ([*rand_pad(padding=3,size=28,mode='reflection'),zoom(scale=1.005),],[])


# In[ ]:


src = ImageList.from_df(train,PATH,cols='filename').split_by_rand_pct(0.2).label_from_df(cols='label').add_test_folder(PATH/'test')


# In[ ]:


src


# In[ ]:


data =  src.transform(tfms, size=28).databunch().normalize(imagenet_stats)


# In[ ]:


data.show_batch(2,2)


# In[ ]:


data.classes,data.c


# In[ ]:


data.train_ds[0][0]


# # Train model

# In[ ]:


learn = cnn_learner(data,models.resnet50,metrics=accuracy,model_dir='/kaggle/working/')


# In[ ]:


learn.lr_find(num_it=600)


# In[ ]:


learn.recorder.plot(skip_end=25)


# In[ ]:


lr=4e-3


# In[ ]:


learn.fit_one_cycle(10,slice(lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage1-resnet50-mnist')


# In[ ]:


learn.load('stage1-resnet50-mnist')


# ### Fine tuning

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(num_it=600)


# In[ ]:



learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,slice(1e-5,1e-4/5))


# In[ ]:


learn.save('stage2-resnet50-mnist')


# In[ ]:


learn.export()


# In[ ]:


PATH.ls()


# # Evaluation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize=(7, 7))
# Maybe feed the network with top losses to improve score?


# # Make predictions

# In[ ]:


test = ImageList.from_folder(TEST)
test


# In[ ]:


learn = load_learner(PATH, test=test)


# In[ ]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


probabilities = preds[0].tolist()
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]


# In[ ]:


class_score = np.argmax(preds, axis=1)


# In[ ]:


class_score[0].item()


# In[ ]:


sample_submission =  pd.read_csv("../input/digit-recognizer/sample_submission.csv")
sample_submission.head()


# In[ ]:


# remove file extension from filename
ImageId = [os.path.splitext(p)[0].split('test_')[1] for p in os.listdir(TEST)]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]


# In[ ]:


sorted(ImageId)[-1]


# In[ ]:


submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})
# submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("nona-submission.csv", index=False)


# In[ ]:




