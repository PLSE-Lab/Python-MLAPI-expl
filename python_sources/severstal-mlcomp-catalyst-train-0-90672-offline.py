#!/usr/bin/env python
# coding: utf-8

# ![MLComp](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/MLcomp.png)
# ![Catalyst](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)

# ### Plan:
# 
# 1. Basic setup and short discussion about MLComp and Catalyst libraries
# 
# 2. MLComp and Catalyst short tutorial
# 
# 3. Severstal DAG execution. Due to Kaggle kernels limit, we will not train final models here. You should train them yourself locally. (the code is the same). All you need to do is to remove extra params from ```mlcomp dag``` command
# 
# Or you can download them https://www.kaggle.com/lightforever/severstalmodels
# 
# 4. You can find inference here https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672
# 
# **An ensemble of the segmentation models with a postprocessing gives 0.90672 on LB. If you add the classifier from Heng CherKeng's [thread](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/106462#latest-634450)**, you should get 0.9117 on LB**

# ## Basic setup
# 
# [MLComp](https://github.com/catalyst-team/mlcomp) is a distributed DAG (Directed acyclic graph) framework for machine learning with UI. It helps to train, manipulate, and visualize.
# 
# Every machine learning pipeline is a Directed acyclic graph. [MLComp](https://github.com/catalyst-team/mlcomp) helps to execute it in a parallel manner, see the results via Web UI and manipulate the process easily.
# 
# https://github.com/catalyst-team/mlcomp
# 
# [Catalyst](https://github.com/catalyst-team/catalyst) helps to train neural networks. It helps to get good positions in the competitions.
# 
# For an example, [yu4u and the others have finished at the 4th place using Catalyst](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110337#latest-635296)
# 
# https://github.com/catalyst-team/catalyst

# Install MLComp library. It already has Catalyst in the dependencies

# In[ ]:


get_ipython().system(' pip install mlcomp')


# Start MLComp's server

# In[ ]:


get_ipython().run_cell_magic('script', 'bash --bg --out script_out', '\n! mlcomp-server start')


# The sever has just started. If you run this code locally, you can open web browser and see the control panel: http://localhost:4201

# Dags
# ![Dags](https://github.com/catalyst-team/mlcomp/raw/master/docs/imgs/dags.png)
# 
# Computers
# ![Computers](https://github.com/catalyst-team/mlcomp/raw/master/docs/imgs/computers.png)
# 
# Reports
# ![Reports](https://github.com/catalyst-team/mlcomp/raw/master/docs/imgs/reports.png)
# 
# We have no such an opportunity in a Kaggle kernel. But we can use MLComp's describe module. See below

# ## Catalyst/MLComp short tutorial

# Catalyst has 2 API: 
# 
# 1. python API, you are importing regular classes
# 2. config API, you are declaring an execution process in a special configuration file and run catalyst-dl run --config=PATH_TO_CONFIG

# We use the second scenario. Let's have a look at our Catalyst config file

# In[ ]:


ls ../input/severstal/severstal/configs_kaggle


# In[ ]:


cat ../input/severstal/severstal/configs_kaggle/catalyst_kaggle.yml


# SegmentationModelPytorch here is a wrapper of this library: https://github.com/qubvel/segmentation_models.pytorch
# 
# We are declaring a model, stages, callbacks, criterion, number of epochs, etc. in a special configuration file.
# 
# Then, we could run it via catalyst-dl run --config=../input/severstal/severstal/configs_kaggle/catalyst_kaggle.yml

# But instead of if, we are declaring one additional config for [MLComp](https://github.com/catalyst-team/mlcomp).
# 
# [MLComp](https://github.com/catalyst-team/mlcomp) helps to declare DAG and execute it in a parallel.

# In[ ]:


cat ../input/severstal/severstal/configs_kaggle/kaggle.yml


# You can see here:
# 
# 1. basic info about DAG in info section
# 
# 2. declaring DAG's structure in executors section.

# DAG's element is known as Executor. They are here: preprocess, masks, train
# 
# Each executor must be declared somewhere in your project's folder where you are running ```mlcomp dag``` command

# In[ ]:


ls ../input/severstal/severstal


# That is our project folder. Executors folder contains our executors.

# Preprocess is a standard group K-Fold stratification

# In[ ]:


cat ../input/severstal/severstal/executors/preprocess.py


# masks converts masks from csv file to regular png masks

# In[ ]:


cat ../input/severstal/severstal/executors/masks.py


# train executor is a standard executor declared in MLComp library. That is a wrapper of Catalyst.

# ## Severstal DAG execution

# Link Kaggle-specific folders

# In[ ]:


get_ipython().system(' mkdir -p ~/mlcomp/data/severstal')

get_ipython().system(' ln -s /kaggle/input/severstal-steel-defect-detection/ ~/mlcomp/data/severstal/input')
get_ipython().system(' ln -s /kaggle/working/ ~/mlcomp/db')


# Start DAG

# In[ ]:


get_ipython().system(' sleep 5')
get_ipython().system(' mlcomp dag ../input/severstal/severstal/configs_kaggle/kaggle.yml --params=executors/train/params/data_params/max_count:50 --params=executors/train/params/num_epochs:3')


# ### IMPORTANT 
# 
# If you are running this code locally, run only ```mlcomp dag ../input/severstal/severstal/configs_kaggle/kaggle.yml```
# 
# This kernel provides a demo run only. With a limited number of samples and epochs.

# Describe the DAG execution status

# In[ ]:


from mlcomp.utils.describe import describe, describe_task_names
describe_task_names(dag=1)


# If you run this kernel, you will see an auto-refreshing describe panel below. (otherwise, you see only the last plot)

# In[ ]:


describe(dag=1, metrics=['loss', 'dice'], wait=True, task_with_metric_count=3, fig_size=(10 ,15))


# #### Copy result models
# 
# Catalyst has a special mechanism named tracing. You can combine a model and that weights in as a single file.
# 
# We have done it, actually. ( we declared ```trace``` configuration for train executor).
# 
# All we need to do now, is to copy the result files.

# In[ ]:


get_ipython().system(' cp ~/mlcomp/tasks/3/trace.pth unet_resnet34.pth')
get_ipython().system(' cp ~/mlcomp/tasks/4/trace.pth unet_se_resnext50_32x4d.pth')
get_ipython().system(' cp ~/mlcomp/tasks/5/trace.pth unet_mobilenet2.pth')


# ### Conclusion
# 
# We have seen a short demonstration of MLComp/Catalyst execution process.
# 
# You can find inference here https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672 
