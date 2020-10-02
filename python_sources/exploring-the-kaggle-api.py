#!/usr/bin/env python
# coding: utf-8

# **Exploring the Kaggle API**
# * [Submit to Competitions](#section1)
# * [Create Datasets](#section2)
# * [Publish Kernels](#section3)
# 
# 
# 
# This notebook is meant to supplement the official documentation for the Kaggle API ([Link #1,](https://github.com/Kaggle/kaggle-api) [Link #2](https://www.kaggle.com/docs/api)).

# In[ ]:


import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from PIL import Image
from IPython.core import display as ICD
get_ipython().run_line_magic('matplotlib', 'inline')
def convert_to_df(csv): # needs to be repaired
    df = pd.DataFrame(data=csv)[0].str.split(',',expand=True)  
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    return df


# Before you begin you will need to:
# * Expand the right-side menu within the kernel editor
# * Enable Internet
# * Install the custom package "kaggle"
# * Replace the USER_ID and USER_SECRET with your own username and API token 
# * * Your API token can be found in the "Account" tab on your Kaggle profile.
# 

# In[ ]:


# If you are using a local machine you will need to create a JSON metadata file containing your USER_ID and USER_SECRET (  ~/.kaggle/kaggle.json) 
# and likewise if you are creating a dataset or a kernel each dataset or kernel will need to be paired with a custom JSON metadata file.  
# More information on creating JSON metadata files can be found in the official documentation ([Link #1](https://github.com/Kaggle/kaggle-api#api-credentials), [Link #2](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata), [Link #3](https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata)) 
# but for the sake of portability we will create these JSON metadata files and we will perform all other computations within the Kaggle Kernel itself.  
# Note that the user_secret is tied to a specific user_id and can be accessed from within the "Account" tab on your Kaggle profile by clicking on the "Create New API Key" button. 

# begin secret_api_token code so that nobody steals my API token.  Remove this section after forking this kernel.
from shutil import copyfile
copyfile(src = "../input/private_kaggle_api_token.py", dst = "../working/private_kaggle_api_token.py")
from private_kaggle_api_token import *
private_kaggle_api_token = private_kaggle_api_token()
# end secret_api_token code that hopefully prevents people from stealing my API token.  Remove this section after forking this kernel.

# Create a JSON file containing user-specific metadata. 
# This step is required if you want to access the Kaggle API.  
# For more info see: https://github.com/Kaggle/kaggle-api#api-credentials
USER_ID = 'paultimothymooney' # REPLACE WITH YOUR OWN USER NAME
USER_SECRET = private_kaggle_api_token # REPLACE WITH YOUR OWN PRIVATE API TOKEN
import os, json, nbformat, pandas as pd
KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
os.makedirs(KAGGLE_CONFIG_DIR, exist_ok = True)
with open(os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json'), 'w') as f:
    json.dump({'username': USER_ID, 'key': USER_SECRET}, f)
get_ipython().system('chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json')


# <a id='section1'></a>
# **Explore Kaggle Competitions: View Listings and Make Submissions**

# View listing of competitions on Kaggle

# In[ ]:


competitions_list_csv = get_ipython().getoutput('kaggle competitions list --csv')
competitions_list_df = convert_to_df(competitions_list_csv)
print('20 Competitions from Kaggle.com/competitions:')
ICD.display(competitions_list_df.head(10))


# Search for a specific competition on Kaggle

# In[ ]:


digit_recognizer_search_in_competitions_csv = get_ipython().getoutput('kaggle competitions list -s digit-recognizer --csv')
digit_recognizer_search_in_competitions_df = convert_to_df(digit_recognizer_search_in_competitions_csv)
print('Search Results for Digit-Recognizer Competition:')
ICD.display(digit_recognizer_search_in_competitions_df.head(10))


# Display leaderboard results for a specific competition on Kaggle

# In[ ]:


leaderboard_list_csv = get_ipython().getoutput('kaggle competitions leaderboard digit-recognizer -s --csv')
leaderboard_list_df = convert_to_df(leaderboard_list_csv)
print('Leaderboard for MNIST Competition:')
ICD.display(leaderboard_list_df.head(10))


# Submit to a competition on Kaggle

# In[ ]:


# Create a dummy submission file for the digit-recognizer competition
list_1=[]
for i in range(28000):
    i=i+1
    list_1.append(i)
list_2 = [7]*28000
kerasmnist = os.path.join('.', 'working/kerasmnist')
os.makedirs(kerasmnist, exist_ok = True)
df = pd.DataFrame(data={"ImageId": list_1, "Label": list_2})
df = df.to_csv("./working/kerasmnist/mnist_dummy_submission.csv", sep=',',index=False)
get_ipython().system('kaggle competitions submit digit-recognizer -f "./working/kerasmnist/mnist_dummy_submission.csv" -m "MNIST Submission from API"')


# <a id='section2'></a>
# **Explore Kaggle Datasets: View, Download, and Create Datasets **

# View datasets listings on Kaggle

# In[ ]:


datasets_list_csv = get_ipython().getoutput('kaggle datasets list --csv')
datasets_list_df = convert_to_df(datasets_list_csv)
print('20 Datasets from Kaggle.com/datasets:')
datasets_list_df_sorted = datasets_list_df.sort_values(by='lastUpdated', ascending=0)
ICD.display(datasets_list_df_sorted.head(10))


# Search for a specific dataset on Kaggle

# In[ ]:


fashion_minst_search_in_datasets_csv = get_ipython().getoutput('kaggle datasets list -s fashion-mnist --csv')
fashion_minst_search_in_datasets_df = convert_to_df(fashion_minst_search_in_datasets_csv)
print('Search Results for Fashion-MNIST Dataset:')
ICD.display(fashion_minst_search_in_datasets_df.head(10))


# Download datasets from Kaggle

# In[ ]:


# download fashion mnist dataset from Kaggle datasets platform
fashionmnist = os.path.join('.', 'working/fashionmnist')
os.makedirs(fashionmnist, exist_ok = True)
get_ipython().system('kaggle datasets download -d zalando-research/fashionmnist -p working/fashionmnist')

# download digit recognizer dataset from Kaggle competitions platform
digitrecognizer = os.path.join('.', 'working/digitrecognizer')
os.makedirs(digitrecognizer, exist_ok = True)
get_ipython().system('kaggle competitions download digit-recognizer -p working/digitrecognizer')


#  Access and visualize the data you just downloaded

# In[ ]:


data = pd.read_csv('working/digitrecognizer/train.csv')
testingData = pd.read_csv('working/digitrecognizer/test.csv')
X = data.drop("label",axis=1).values
y = data.label.values

def describeDataset(features,labels):
    print("\n'X' shape: %s."%(features.shape,))
    print("\n'y' shape: %s."%(labels.shape,))
    print("\nUnique elements in y: %s"%(np.unique(y)))
describeDataset(X,y)

def displayMNIST(flatData,labels):
    """ Displays 10 handwritten digis and 10 classification labels """
    figure,image = plt.subplots(1,10, figsize=(10,10))
    for i in range(10):
        image[i].imshow(flatData[i].reshape((28,28)))
        image[i].axis('off')
        image[i].set_title(labels[i])
displayMNIST(X,y)

def displayMNIST2(flatData,labels):
    """Display MNIST data"""
    flatData2 = data.drop("label",axis=1).values
    X2 = np.insert(flatData2,0,1,axis=1)
    figure,image = plt.subplots(1,10, figsize=(10,10))
    for i in range(10):
        tenImages = np.random.choice(X2.shape[0], 10)
        image[i].imshow(X2[tenImages,1:].reshape(-1,28))
        image[i].axis('off')
displayMNIST2(X,y)


# Create a new dataset using the data you just downloaded

# In[ ]:


# Tidy up the data before uploading it
get_ipython().system('zip -r fashionmnist.zip working/fashionmnist/')
get_ipython().system('zip -r digitrecognizer.zip working/digitrecognizer/')

# Create dataset-specific JSON metadata file
# https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata
dataset_meta_template = lambda user_id, title, file_id, nb_path: {"title": f"{title}", 
  "subtitle": "My awesomer subtitle",
  "description": "My awesomest description",
  "id": f"{user_id}/{file_id}",
  "licenses": [{"name": "CC0-1.0"}],
  "resources": 
    [{"path": "digitrecognizer.zip",
      "description": "kaggle.com/c/digit-recognizer",},
    {"path": "fashionmnist.zip",
      "description": "kaggle.com/zalando-research/fashionmnist"}],}

name_of_new_dataset='Kaggle-Dataset-Demo-From-API'
path_of_current_data = 'working'
with open('dataset-metadata.json', 'w') as f:
    meta_dict = dataset_meta_template(USER_ID,name_of_new_dataset,name_of_new_dataset,path_of_current_data)
    json.dump(meta_dict, f)
get_ipython().system('kaggle datasets create -p .')


# Add a new file to the dataset and create a new dataset version

# In[ ]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
kerasmnist = os.path.join('.', 'working/kerasmnist')
os.makedirs(kerasmnist, exist_ok = True)
np.savez("working/kerasmnist/MNIST_X_train", x_train)
np.savez("working/kerasmnist/MNIST_Y_train", y_train)
np.savez("working/kerasmnist/MNIST_X_test", x_test)
np.savez("working/kerasmnist/MNIST_Y_test", y_test)
get_ipython().system('kaggle datasets version -p . -m "Added more MNIST data"')


# <a id='section3'></a>
# **Explore Kaggle Kernels: View, Download, and Publish Kernels**

# View and search for kernels on Kaggle

# In[ ]:


kernel_listings_csv = get_ipython().getoutput('kaggle kernels list -s kaggle-kernel-demo --csv')
kernel_listings_df = convert_to_df(kernel_listings_csv)
print('Listing of Kernels on Kaggle for Search Term "Demo":')
ICD.display(kernel_listings_df.head(10))


# Download a kernel using the Kaggle API

# In[ ]:


# Create a JSON file containing kernel-specific metadata
# https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata
notebook_meta_template = lambda user_id, title, file_id, nb_path: {'id': f'{user_id}/{file_id}',
 'title': f'{title}',
 'code_file': nb_path,
 'language': 'python',
 'kernel_type': 'notebook',
 'is_private': True,
 'enable_gpu': True,
 'enable_internet': False,
 'keywords': [],
 'dataset_sources': ['keras/resnet50', 'paultimothymooney/sample-images-for-kaggle-demos'],
 'kernel_sources': [],
 'competition_sources': []}

name_of_new_kernel='Kaggle-Kernel-Demo-From-API'
path_of_current_kernel = 'working/demokernel/kaggle-kernel-demo-pre-trained-image-classifier.ipynb'
with open('kernel-metadata.json', 'w') as f:
    meta_dict = notebook_meta_template(USER_ID,name_of_new_kernel,name_of_new_kernel,path_of_current_kernel)
    json.dump(meta_dict, f)
    
# download kaggle kernel demo
demokernel = os.path.join('.', 'working/demokernel')
os.makedirs(demokernel, exist_ok = True)
get_ipython().system('kaggle kernels pull paultimothymooney/kaggle-kernel-demo-pre-trained-image-classifier -p working/demokernel')


# Publish a kernel using the Kaggle API

# In[ ]:


# publish a new version of the kaggle kernel demo
get_ipython().system('kaggle kernels push -p .')


# Tidy up the notebook output

# In[ ]:


get_ipython().system('zip -r kerasmnist.zip working/kerasmnist/')
get_ipython().system('zip -r demokernel.zip working/demokernel/')
get_ipython().system('rm -rf working/*')
get_ipython().system('rm -r working')
get_ipython().system('rm -r kernel-metadata.json')
get_ipython().system('rm -r dataset-metadata.json')
get_ipython().system('rm -r private_kaggle_api_token.py')
get_ipython().system('rm -r __pycache__/private_kaggle_api_token.cpython-36.pyc # Please dont steal my token')

