#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install git+https://download.radtorch.com/ -q
get_ipython().system('git clone -b nightly https://github.com/radtorch/radtorch/ -q')
get_ipython().system('pip3 install radtorch/. -q')


# In[ ]:


from radtorch import pipeline, core, utils


# In[ ]:


get_ipython().system(' mkdir /train_data/')
get_ipython().system(' mkdir /test_data/')
get_ipython().system(' unzip -q /kaggle/input/dogs-vs-cats/train.zip  -d /train_data/ ')
get_ipython().system(' unzip -q  /kaggle/input/dogs-vs-cats/test1.zip -d /test_data/ ')


# In[ ]:


train_dir = '/train_data/train/' 
test_dir = '/test_data/test1/' 


# In[ ]:


table = utils.datatable_from_filepath(train_dir, classes=['dog','cat'])

table.head()


# In[ ]:


clf = pipeline.Image_Classification(
data_directory=train_dir,
    is_dicom=False,
    table=table,
    type='nn_classifier',
    model_arch='vgg16',
    epochs=10,
    batch_size=100,
    sampling=0.15,
)


# In[ ]:


clf.data_processor.dataset_info(plot=False)


# In[ ]:


clf.run()


# In[ ]:


clf.classifier.confusion_matrix()


# In[ ]:


clf.classifier.summary()


# In[ ]:


target_image = '/test_data/test1/10041.jpg'
target_layer = clf.classifier.trained_model.features[30]

clf.cam(target_image_path=target_image, target_layer=target_layer, cmap='plasma', type='scorecam')


# In[ ]:




