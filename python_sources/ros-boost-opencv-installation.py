#!/usr/bin/env python
# coding: utf-8

# # ROS Installation In Kaggle Kernel
# In this part you can find the steps to install Robotics Operating System (ROS) from within a Kaggle kernel.

# In[ ]:


get_ipython().system('apt -o APT::Sandbox::User=root update')


# In[ ]:


get_ipython().system('apt -y install git curl wget cmake build-essential lsb-core unzip')


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


# # Boost Installation in Kaggle Kernel
# In this part you can find the steps to install Boost from within a Kaggle kernel.

# In[ ]:


get_ipython().system('wget https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.gz')


# In[ ]:


get_ipython().system('tar -xzvf boost_1_62_0.tar.gz')


# In[ ]:


cd boost_1_62_0


# In[ ]:


get_ipython().system('mkdir build')


# In[ ]:


get_ipython().system('./bootstrap.sh --with-libraries=filesystem --with-libraries=program_options')


# In[ ]:


get_ipython().system('./b2 install -j$(nproc)')


# In[ ]:


cd ../


# In[ ]:


get_ipython().system('rm -rf boost_1_62_0 && rm -rf boost_1_62_0.tar.gz')


# # OpenCV C++ Installation in Kaggle Kernel
# In this part you can find the steps to install OpenCV from within a Kaggle kernel.

# In[ ]:


get_ipython().system('wget https://github.com/opencv/opencv/archive/3.4.8.zip')


# In[ ]:


get_ipython().system('unzip 3.4.8.zip')


# In[ ]:


cd opencv-3.4.8


# In[ ]:


get_ipython().system('mkdir build')


# In[ ]:


cd build


# In[ ]:


get_ipython().system("export LD_LIBRARY_PATH='' && cmake -DWITH_TIFF=OFF -DCMAKE_BUILD_TYPE=RELEASE ..")


# In[ ]:


get_ipython().system("export LD_LIBRARY_PATH='' && make -j2")


# In[ ]:


get_ipython().system('make install')


# In[ ]:


cd ../..


# In[ ]:


get_ipython().system('rm -rf opencv-3.4.8 && rm -rf 3.4.8.zip')

