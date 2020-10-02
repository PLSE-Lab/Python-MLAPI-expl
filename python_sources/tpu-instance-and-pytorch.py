#!/usr/bin/env python
# coding: utf-8

# # How start work with TPU node and PyTorch?

# You can find many information about TPU and how it fast. Most of it: [Google here](https://cloud.google.com/tpu) or [Google another](https://cloud.google.com/tpu/docs/)
# 
# The next lines is about a quick way to get started with TPU and [PyTorch/XLA](https://github.com/pytorch/xla). More detail you can find in github.

# ## 1. Create an instance

# Similar with this [notebook](https://www.kaggle.com/blondinka/how-to-use-your-gcp-credit-and-set-it-up). Select PyTorch version OS with TPU.
# <img src="https://i.imgur.com/E9bekeC.png">

# ## 2. Create TPU node 
# <img src="https://i.imgur.com/sJ2IvEO.png">
# <img src="https://i.imgur.com/RrlUDv4.png">

# 1. TPU node must be in the same zone as an instance
# 2. Same network as an instance.
# 3. Select IP address.
# 
# ```
#    The IP address range must be from within the internal IP address ranges:
# 
#      10.0.0.0        -   10.255.255.255  (10/8 prefix)
#      172.16.0.0      -   172.31.255.255  (172.16/12 prefix)
#      192.168.0.0     -   192.168.255.255 (192.168/16 prefix)
#      
#     You need this IP address for an instance. 
# ```
# <img src="https://i.imgur.com/UtKh8Un.png">
# 

# ## 3. Run your instance 

# SSH into VM and activate the conda environment you wish to use:
# ```
# (vm)$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
# (vm)$ conda env list
# # conda environments:
# #
# base                  *  /anaconda3
# torch-xla-0.1              /anaconda3/envs/torch-xla-0.1
# torch-xla-0.5              /anaconda3/envs/torch-xla-0.5
# torch-xla-nightly          /anaconda3/envs/torch-xla-nightly
# 
# (vm)$ conda activate torch-xla-0.5
# (torch-xla-0.5)$ cd /usr/share/torch-xla-0.5/pytorch/xla
# (torch-xla-0.5)$ python test/test_train_mnist.py
# ```
# 
# $TPU_IP_ADDRESS - Its your IPs TPU node
# 
# If You want use bfloat16:
# `(vm)$ export XLA_USE_BF16=1`

# ## 4. Work with TPU like multiGPU:
# Examples : https://github.com/pytorch/xla/tree/master/contrib/colab
# 
# ```
# import torch_xla
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# 
# ......
# 
# def train_loop_fn(model, loader, device, context):
#     #do train loop
#     retun loss_mean
#     
# # wrap your model:
# devices = (
#     xm.get_xla_supported_devices(max_devices=num_cores) if num_cores != 0 else [])
#     
# model_parallel = dp.DataParallel(model, device_ids=devices)
# 
# # train
# for epoch in range(1, num_epochs + 1):
#     train_loss = model_parallel(train_loop_fn, loader)
# ```
