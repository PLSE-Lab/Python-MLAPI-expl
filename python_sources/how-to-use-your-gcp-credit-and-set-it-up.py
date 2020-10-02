#!/usr/bin/env python
# coding: utf-8

# Many competitions offer google cloud platform (GCP) credits. Here I put together the helper instructions how to actually use them and set everything up. 
# 
# ## Prerequisites:
# 
# You'll need 
# 
# a) a google account 
# 
# b) a valid credit or debit card
# 
# ## Starting
# 
# Login to your google account, then go to google cloud console. 
# 
# Add your card details in billing section. 
# 
# Follow the link in your email to get your credit (it should direct you to your console automatically).
# You can also avail of 300$ credit when you first try GCP. 

# ## 0.0 Create your first project
# 
# press 'My first Project' at the top of the console, and then New Project, choose a name and press Create

# ## 0.1 Create an instance 
# ![image.png](attachment:image.png)

# ## 0.2 Choose instance set up
# 
# When choosing the region see what GPUs are avaiable for this particular region, they differ. The list of GPUs available in different regions is here: 
# https://cloud.google.com/compute/docs/gpus/
# ![image.png](attachment:image.png)
# 
# Pre-emtible instances are cheaper, but they can stop suddenly and need to restart every 24 hours. I normally create 4 CPU instance with Ubunta 16.04 and 100G hard drive for the start. Do not add GPU at the start as it would not allow you. We'll make it later. 

# ## 0.3 Request quota increase
# To be able to add GPU to your instance you need to request a quota increase first. Go to Qoutas and add 1 GPU. It may take a day to get approval and you should receive an email.
# ![image.png](attachment:image.png)
# 
# Once you got an approval, get back to your instance, press Edit, add a GPU, then press start. The instance should be running.  
# 

# ## 0.4 Get static IP
# The last bit you want is to set a static IP adress for your virtual machine. On your GCP go to VPC Network -- External IP Address -- select your new instance and change IP type to static.

# ## 0.5 Connect to your instance
# If you use Putti or MobaXTerm to connect you may want to generate public-private key pair (they have keygens). Go to your instance, press edit and add copy **public** key to metadata. 
# Then go to MobaXTerm or Putty and set up and new ssh session, add your instance static IP, go to advanced settings and select "use private key" and load your **private** key there

# ![image.png](attachment:image.png)

# ## 1.0 Setting up
# Once you connected to your instance you may want to install python, conda, pytorch, cuda... on it. You may use kaggle-docker for it, but it could be too exhaustive for your project.
# 
# For GPU: 
# 
# You can install CUDA like here: https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070
# 
# Or here: https://medium.com/datadriveninvestor/complete-step-by-step-guide-of-keras-transfer-learning-with-gpu-on-google-cloud-platform-ed21e33e0b1d
# 
# Also, when you select system for your VM instance (you go to boot list -- press change and you select the system), some will have CUDA pre-installed, I think Debian will have (please correct meif I am wrong), To use containers you can also select there a container-friendly system with docker, but selecting a system with CUDA is the easiest option.  
# Once you have a GPU turned on type nvidia-smi; if CUDA is installed, you should see your GPU in the printed table.
# 
# Now you are ready to start. 
# 
# Lets take this competition as an example: https://www.kaggle.com/c/severstal-steel-defect-detection

# ## 2.0 Download dataset
# The worst and the least efficient way to do so is to go on competition dataset page and press download... wait... and then use WinSCP to upload it on your instance... 
# This is the ultimate dump stupid way, because the downloading may fail, then you may restart it, uploading fails and takes long-long-long time, leaving you in frustration and ageing before you get the data.
# 
# How do I know that? That's exactly how I was doing it for the first several months :facepalm: :lol:

# ## 2.1 Proper way to download the dataset
# 
# Install kaggle API (pip install kaggle). 
# 
# Get kaggle.json from you profile. Go to Edit profile and press "Create New API Token"
# ![image.png](attachment:image.png)

# then create a directory `mkdir .kaggle` and copy kaggle.json to it  
# 
# run `pip install kaggle`
# 
# make a bash script to download and unzip the data, you may give it a dummy name download_data.sh and then simply run:
# 
# bash download_data.sh

# In[ ]:


"""
#!/usr/bin/env bash

CUR_DIR=$pwd
DATA_DIR_LOC=dataset

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    pip install kaggle --upgrade
    kaggle competitions download -c severstal-steel-defect-detection
    mkdir train
    mkdir test
    unzip train_images.zip -d train
    unzip test_images.zip -d test
fi

cd $CUR_DIR
echo $(pwd)
"""


# WARNING: Do not forget to stop your instance!!! `sudo shutdown now` 
# 
# Also, do not forget to monitor your billing, if you go beyond your credit it will start charging your card

# In[ ]:


print("Thank you")

