#!/usr/bin/env python
# coding: utf-8

# This competition offers both AWS and GCP credits. Although AWS have deep-learning-ready system, it is sometimes buggy. And GCP just have plain ubuntu system.
# So, I made this simple guide to setup an instance. 
# 
# This is a simple guide that you can actually copy and paste to your instance command line. 

# # Part 1

# In[ ]:


sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-418


# # Reboot

# In[ ]:


sudo reboot


# # Check Sucessful

# In[ ]:


nvidia-smi


# # Part 2

# In[ ]:


sudo apt-get install --no-install-recommends     cuda-10-0     libcudnn7=7.6.4.38-1+cuda10.1      libcudnn7-dev=7.6.4.38-1+cuda10.1
sudo apt-get install -y --no-install-recommends libnvinfer5=6.0.1-1+cuda10.1     libnvinfer-dev=6.0.1-1+cuda10.1


# # Install tensorflow,keras,pytorch

# In[ ]:


sudo apt-get install python3-pip
pip3 install tensorflow
pip3 install keras
pip3 install opencv


# Thanks for looking. If it helped you, please give an upvote. Thanks!
