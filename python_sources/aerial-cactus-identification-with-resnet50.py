#!/usr/bin/env python
# coding: utf-8

# # Aerial Cactus Identification with Resnet50#
# 
# Tell Jupyter to auto-reload functions that may have changed in the meantime, as well as plot in the notebook

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Import the relevant libraries. Only the first 2 lines are necessary to train the model, the latter two are used to "open the black box" and peek into the working of the neural net

# In[ ]:


from fastai import *
from fastai.vision import *


# Indicate where the train sets can be found. They are stored in a folder called "train", and the file names and labels are contained in a CSV file. In creating the data set, the fast.ai library automatically labels these images correctly.
# The size of the images is 32x32

# In[ ]:


PATH = "../input/"
sz=32
bs=512


# I apply transformations to my dataset. This helps in augmenting the size of the dataset. It rotates, flips vertically AND horizontally the images, and plays around with zoom and lighting. Below I show examples transformations for 1 image
# I also use on of the out-of-the-box functions to create the full dataset that can be loaded into the model. 
# I normalize the images as I am using a pre-trained ResNet50 model. I also indicate num_workers=0 as I'm using Windows, and any other number doesn't sit well with my GPU

# In[ ]:


tfms = get_transforms(flip_vert=True, max_rotate=90.)
data = ImageDataBunch.from_csv(PATH, ds_tfms=tfms,
        folder="train/train", csv_labels='train.csv', test="test/test",
        valid_pct=0.1, fn_col=0, label_col=1).normalize(imagenet_stats)


# Checking to see which artists we have in our sample

# In[ ]:


print(f'We have {len(data.classes)} different classes\n')
print(f'Classes: \n {data.classes}')


# In[ ]:


print (f'We have {len(data.train_ds)+len(data.valid_ds)+len(data.test_ds)} images in the total dataset')


# Show several rows of examples of the images in our dataset

# In[ ]:


data.show_batch(8, figsize=(20,15))


# Show how the the data augmentation transformation are applied to an example picture

# In[ ]:


def get_ex(): return open_image('../input/train/train/000c8a36845c0208e833c79c1bffedd1.jpg')

def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]


# In[ ]:


plots_f(4, 4, 8, 8, size=sz)


# This is where the magic of fast.ai happens! With just one line, I'm creating a ResNet50 model, based on the dataset we just created, testing it for classification accuracy (does it identify the right artists), storing outputs it in our "model" directory, and introducing a callback so I can see the output graphs of the learning cycles

# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=accuracy, path='../kaggle/working', model_dir='../kaggle/working/model',callback_fns=ShowGraph)


# I am using the off-the-shelf method to test the loss response to different learning rates. You don't want too high, or too low learning rates. You'd want to pick the point where the gradient of this curve is most downward sloping (at around 1e-2)

# In[ ]:


lrf=learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=5e-3


# Again, some more fast.ai magic! I'm fitting the model, i.e. training the weights of the last layer of the ResNet50 architecture to classify the cactuses (cacti?). The fit_one_cycle() method varies the learning rate to more rapidly converge to lower losses (details here: https://docs.fast.ai/callbacks.one_cycle.html) 

# In[ ]:


learn.fit_one_cycle(1,lr)


# It took the model about 20 seconds to train the last layer, and already we're at an accuracy of 99+%! And we have some room to go as the validation set loss is still significantly below the training loss 
# 
# Let's save what we have

# In[ ]:


learn.save('cactus-stage-1')


# Let's now unfreeze the other layers in the model to further tune the weights to our problem. The ResNet50 architecture model I'm using is trained on the ImageNet database, that contains many different classes that you might see in real life. Our dataset is slightly different, so we expect we can further tune intermediate layers to help differentiate characteristics of the different artists

# In[ ]:


learn.unfreeze()


# In[ ]:


lrf=learn.lr_find()
learn.recorder.plot()


# After finding an optimal learning rate again, I'm training the model again using the fit_one_cycle() method, but am telling it to differentiate the learning rates for earlier layers vs. later layers. The earlier layers are more basic (recognize corners, patterns), and will still be very applicable to our problem, whereas the latter layers discover more complex features (e.g. eyeballs, wheels, etc.) and will be less relevant

# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))


# In[ ]:


learn.save('cactus-stage-2')


# We have slightly overfitted our model (training loss is below our validation loss). Let's get the predictions on the test set and submit the results

# In[ ]:


preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


sub=pd.read_csv(f'{PATH}/sample_submission.csv').set_index('id')


# In[ ]:


clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0]+".jpg")
fname_cleaned=clean_fname(data.test_ds.items)
fname_cleaned=fname_cleaned.astype(str)
fname_cleaned


# In[ ]:


sub.loc[fname_cleaned,'has_cactus']=to_np(preds_test[:,1])
sub.to_csv(f'submission.csv')
sub.loc[fname_cleaned,'has_cactus']=to_np(preds_test_tta[:,1])
sub.to_csv(f'submission_tta.csv')


# In[ ]:


classes = preds_test.argmax(1)
classes
sub.loc[fname_cleaned,'has_cactus']=to_np(classes)
sub.to_csv(f'submission_1_0.csv')

