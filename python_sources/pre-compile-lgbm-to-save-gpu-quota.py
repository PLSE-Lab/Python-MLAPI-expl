#!/usr/bin/env python
# coding: utf-8

# # Pre-compile LightGBM GPU version to save GPU quota
# why create this kernel?
# when you want use LigthGMB GPU version, as this thread(https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu/comments#LightGBM-GPU-Installation) posted. you have to download and compile LightGBM every time when you started the kernel. comiling is time-wasting operation, so we can compile it well in one kernel and use it directly in other kernels without compiling it again.
# how to use it:
# * add this kernel's output to your data source;
# * do the following codes in your kernel:
# 
# > !rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
# > 
# > !apt-get install -y -qq libboost-all-dev
# 
# > %%bash
# > 
# > cd /tmp
# > 
# > tar -xvf /kaggle/input/compile-lgbm/LightGBM.tar > /dev/null
# 
# > !cd /tmp/LightGBM/python-package/;python3 setup.py install --precompile
# > 
# > !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# > 
# > !pip install -q 'pandas==0.25' --force-reinstall
# 
# there'a another time-wasting step: install libboost-all-dev (in your kernel.). but it seems no way to save time in this step, coz installation alone cost about 40s. if anyone have some idea about this please let me know.
# 
# Upvote if you like it, thank you!

# # install libboost, compiling depends on it.

# In[ ]:


get_ipython().run_cell_magic('time', '', '!apt-get install -y -qq libboost-all-dev')


# # compile LightGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%bash\n\ncd /tmp\nif [ -d "LightGBM" ];then rm -r LightGBM; fi;\ngit clone --recursive https://github.com/Microsoft/LightGBM\n\ncd ./LightGBM\nif [ -d "./build" ];then rm -r build; fi;\nmkdir build\n\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# # install LightGBM (optional here) to test if compiling is OK.

# In[ ]:


get_ipython().system('cd /tmp/LightGBM/python-package/;python3 setup.py install --precompile')

get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
# !rm -r LightGBM

# Latest Pandas version
get_ipython().system("pip install -q 'pandas==0.25' --force-reinstall")


# # tar the LightGBM folder as kernel output.

# In[ ]:


import tarfile

with tarfile.open('./LightGBM.tar', 'w') as tar:
    tar.add('/tmp/LightGBM', arcname='LightGBM')

get_ipython().system('du -sh ./*')

