#!/usr/bin/env python
# coding: utf-8

# I like Google colab more than Kaggle Kernels because it allows me to use GPUs for more time than Kaggle Kernels and everything just works!
# 
# So here in this notebook I will show you how to use Google colab instead Kaggle Kernels for this competition.

# First of all, you need your Kaggle API key to download the dataset.
# 
# To obtain the key, follow these steps:
# 
# 1. Visit [Kaggle.com](https://kaggle.com)
# 2. Click on your profile picture on the top left corner and select "My Account".
# ![Homepage](https://i.ibb.co/RbvzbJ7/Screenshot-25.png)
# 3. Scroll down and in under API Section, Click in "Create New API Token" button.
# ![API](https://i.ibb.co/xs05NZr/Screenshot-27.png)
# 4. You will be prompted with "kaggle.json" download window, save the file, and open it with notepad or any text editor.
# 5. Select everything and copy it.
# 6. Now go to [colab](https://colab.research.google.com), Click on "New Notebook"
# ![NEW](https://i.ibb.co/2dFcscZ/Screenshot-28.png)
# 7. Click on "Runtime" > Select "Change runtime type" > Change the Hardware accelerator from "None" to "GPU"
# 8. Copy paste the below cell into your newly created colab notebook and replace the `credentials` dictionary with the dictionary you copied from `kaggle.json` file, and run the cell(it will take few minutes to run).
# 9. Happy Coding!

# In[ ]:


get_ipython().system('pip install git+https://github.com/Kaggle/kaggle-api.git --upgrade')
import os
credentials = {"username":"vijayabhaskar96","key":"123456a45847983a4537dbae3f23d612f"}
os.environ['KAGGLE_USERNAME']=credentials["username"]
os.environ['KAGGLE_KEY']=credentials["key"]
get_ipython().system('kaggle competitions download -c jovian-pytorch-z2g')
get_ipython().system('unzip jovian-pytorch-z2g.zip')


# If you have been working already on Kaggle kernel and need to move your notebook to Colab, instead of creating a new notebook on Colab, you can download(File > Download) your notebook from kaggle and upload(File > Upload notebook) it to Colab.
# 
# Remember to change the paths respective to colab environments, if you're using the starter kernel provided by Aakash then just copy and replace the below cell in your notebook.

# In[ ]:


DATA_DIR = '/content/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = '/content/submission.csv'   # Contains dummy labels for test image


# To submit, download the csv file and make the submission on Kaggle.
# To download:
# ![click](https://i.ibb.co/HBbqBR9/Screenshot-29.png)
# ![download](https://i.ibb.co/dgs4P3h/Screenshot-30.png)

# # Note: You have to do all these steps only for the first time. From the next time you can just visit colab and directly start working on your project.
# 
# # CAUTION: NEVER SHARE YOUR KAGGLE API KEY PUBLICLY.
