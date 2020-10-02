#!/usr/bin/env python
# coding: utf-8

# This is a demonstration on how to use your GitHub repository to add custom files into your kaggle kernels. I felt this need while trying to load my own pre-trained model and show accuracy results on the test-set. I was using a deep-learning model and the training needed hours on kaggle setup. I was hoping to train the model offline and then load it on a kaggle kernel and just run on the test-set to show that it wroked.
# 
# However, the benifits are not just limited to pre-trained model files, you could just load any file this way. Say, you would like to normalize the images in your dataset but computation of mean and std images take too much time on kaggle. You could just pre-compute these images and load them through your GitHub repository.
# 
# Disclaimer: I have not tested if this method works for large files as well. My testing was limited to files with size < 1MB.
# 
# Disclaimer: It may also be possible to achieve the same by creating a new dataset with your custom files. I haven't tried that yet.
# 
# Disclaimer: If you fork the kernel then the fork will not automatically install your GitHub repo module. You would have to install it 

# In[20]:


import matplotlib.pyplot as plt


# Setting up the GitHub repository:
# 1. Create a Public Git repository (it is important that it is created Public).
# 2. Rename your custom file with  .py extension. For example, myImage.jpg should be renamed to myImage.py
# 3. Make sure you have a directory structure that can be installed using pip.  See [this stackoverflow page](http://https://stackoverflow.com/questions/8247605/configuring-so-that-pip-install-can-work-from-github) for details.
# 4. Add a setup.py file with necessary content.
# 
# Note: Make sure that you repository has no underscores in it. For example, choose name `mytest` instead of `my_test`
# 
# For reference, you can see [my GitHub repository](http://https://github.com/govindnh4cl/mytest).

# Setting up the Kaggle kernel:
# 
# ## Step1
# Kaggle provides the option to install a package from GitHub directly in the Settings tab. 'Screenshot below:

# In[9]:


# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/add_custom_package.jpg.py')
fig = plt.figure(figsize = (70,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)


# ## Step 2
# Enter your username/reponame. Screenshot below:

# In[11]:


# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/add_repo_details.jpg.py')
plt.figure(figsize = (10,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)


# ## Step 3
# Press Enter. This will install package. 'Sometimes this step thorws weird errors. Just attempt again and it should install. Screenshot below:

# In[12]:


# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/installed_package.jpg.py')
plt.figure(figsize = (10,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)


# ## Step 4
# Restart the kernel to bring the change into effect.

# ## Step 5
# You could confirm that the your covert .py files are indeed available in the kaggle filesystem by opening up the console and printing the content of directory:
#         > /opt/conda/lib/python3.6/site-packages/mytest/
# (Here mytest is my module name. Your's would be what you set in your GitHub repository.)        

# In[14]:


# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/console.jpg.py')
plt.figure(figsize = (30,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)


# That's all. Your files are available to you now on your kaggle kernel.
# 
# ## Uninstalling a package
# Say you installed your package and now have access to you files. Now, if you updated the GitHub repository, then to see the changes reflect in kaggle kernel you would need to reinstall the package. I don't yet know an explicit way to do it, I reverted to fresh Docker image and installed my package once again to achieve this. See image below:

# In[15]:


# Display screenshot
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/uninstall_package.jpg.py')
plt.figure(figsize = (20,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img)


# # Examples

# ## Load a Pre-Trained Model

# In[ ]:


from keras.models import load_model
# Read model from the model file
model = load_model('/opt/conda/lib/python3.6/site-packages/mytest/covert_keras_model.py')
# Verify that it works
model.summary()  # Print model summary


# ## Load a Custom Image 
# Have the image a covert .py extension in your python package

# In[19]:


import matplotlib.pyplot as plt
#Read image from disk
img = plt.imread('/opt/conda/lib/python3.6/site-packages/mytest/covert_legend_kid_jpg.py')

# Display image
plt.figure(figsize = (30,10))
plt.xticks([]); plt.yticks([]); 
_ = plt.imshow(img, interpolation='nearest')


# ## Load a CSV File
# Loading a CSV (taken from leaf classification dataset on kaggle) the same way

# In[ ]:


import pandas as pd
df = pd.read_csv('/opt/conda/lib/python3.6/site-packages/mytest/covert_leaf_classification_csv.py')
df.head()


# In[ ]:




