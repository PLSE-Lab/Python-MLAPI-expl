#!/usr/bin/env python
# coding: utf-8

# # ROS Installation In Kaggle Kernel
# In this kernel you can find the steps to install Robotics Operating System (ROS) from within a Kaggle kernel.

# In[ ]:


get_ipython().system('apt -o APT::Sandbox::User=root update')


# In[ ]:


get_ipython().system('apt -y install lsb-core')


# In[ ]:


get_ipython().system('sh -c \'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list\'')


# In[ ]:


get_ipython().system('apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654')


# In[ ]:


get_ipython().system('apt -o APT::Sandbox::User=root update')


# In[ ]:


get_ipython().system('apt install -y ros-melodic-desktop-full')


# In[ ]:


get_ipython().system('rosdep init')


# In[ ]:


get_ipython().system('rosdep update')


# In[ ]:


get_ipython().system('echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc')


# In[ ]:


get_ipython().system("bash -c 'source ~/.bashrc'")


# In[ ]:


get_ipython().system('apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential')


# # Listing ROS Info of Each of the ROS Bag Data Files

# ## ROS Bag Info of the first file:

# In[ ]:


get_ipython().system("bash -c 'source ~/.bashrc && rosbag info /kaggle/input/kinect-v2-multi-objects-with-3d-positions/depth-estimation-dataset_3.bag'")


# ## ROS Bag Info of the second file:

# In[ ]:


get_ipython().system("bash -c 'source ~/.bashrc && rosbag info /kaggle/input/kinect-v2-multi-objects-with-3d-positions/depth-estimation-dataset_4.bag'")


# ## ROS Bag Info of the third file:

# In[ ]:


get_ipython().system("bash -c 'source ~/.bashrc && rosbag info /kaggle/input/kinect-v2-multi-objects-with-3d-positions/depth-estimation-dataset_5.bag'")


# As we can see from the info of the three data files, we can get many types of information about the rosbag from this command (i.e. file size, ros topics loaded in the file, the type of message of each topic, etc...).
