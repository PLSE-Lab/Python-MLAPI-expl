#!/usr/bin/env python
# coding: utf-8

# This notebook is my personal reference on debugging machine learning model GPU utilization. This is necessary when you discover that a model is using too little GPU (potentially none), or too much (potentially resulting in OOM due to memory leaks?).
# 
# ## How the GPU and CUDA works
# * NVIDIA owns the machine learning on a GPU vertical at this time. That means that if you're running a PyTorch model or whatever on a GPU, you're doing so on an NVIDIA card and using the NVIDIA toolchain.
# 
#   The interface to running general purpose non-graphics programs on a video card is an SDK (card firmware support, toolchain, and application progamming language) known as CUDA. CUDA is proprietary and baked into NVIDIA GPUs. There are some competing frameworks, including a couple of open source graphics engines from the OSS community, but none have real traction thanks to NVIDIA's market control.
# * The CUDA interface runs in the background on your machine. On a regular machine this is a daemon that runs at startup that is part of the drivers installed alongside the device; in Docker you need to use the `nvidia-docker` image (which has its own section later in this notebook).
# * CUDA is versioned. Major version increments are incompatible with one another; because CUDA involves on-device firmware, generally major versions and major version increments are associated with specific card architectures. For example, CUDA 1.x is for the Tesla series of cards. See [the version table on Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) for the version correspondances.
# 
# 
# ## How PyTorch works
# * PyTorch can be installed with CUDA or without CUDA support. Installation on Windows and on Linux automatically have CUDA supports, but installations on macOS do not have CUDA support unless you build PyTorch yourself. To check if your install of PyTorch has CUDA support, check if the `torch.cuda.is_available()` is `True`.
# * PyTorch maintains compatibility with a sliding range of CUDA versions. Code that utilize new CUDA features may rely on backports etecetera when PyTorch detects you are running on older CUDA version. Generally you do not have to worry about this too much, except that you might get a warning or two here or there.
# * As you would expect, vendor environments, e.g. AWS AMIs, have a longer lag behind the current version of PyTorch.
# 
# 
# ## Device visibility
# * The CUDA SDK assigns every CUDA device (GPU or maybe TPU) a sequential ID number counting up from 0.
# 
#   A global environment variable, `CUDA_VISIBLE_DEVICES`, controls the global visibility of these devices. The first step to debugging a model is verifying that this variable is set to what you expect it to be set to, or to set it yourself:
#   
#   ```bash
#   export CUDA_VISIBLE_DEVICES=0,1
#   ```
#  
#   -1 is a special flag saying no devices are visible (e.g. this tells models to use CPU). You can also theoretically use empty string (`""`) but this is sometimes buggy.
#   
#   ```bash
#   export CUDA_VISIBLE_DEVICES=-1
#   ```
#   
#   This setting is respected at the library level by PyTorch.
# 
# 
# * `torch` includes functions for checking up on the devices visible to it. From the [StackOverflow Q&A](https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu):
# 
#     ```python
#     In [1]: import torch
# 
#     In [2]: torch.cuda.current_device()
#     Out[2]: 0
# 
#     In [3]: torch.cuda.device(0)
#     Out[3]: <torch.cuda.device at 0x7efce0b03be0>
# 
#     In [4]: torch.cuda.device_count()
#     Out[4]: 1
# 
#     In [5]: torch.cuda.get_device_name(0)
#     Out[5]: 'GeForce GTX 950M'
# 
#     In [6]: torch.cuda.is_available()
#     Out[6]: True
#     ```
# 
# * Best practice for PyTorch models is allowing the specification of the devices to be used by ID from the CLI, and perhaps defaulting to all visible GPUs (if parallelization is supported by the model) or to the first visible GPU (if it is not). Because device visibility is enforced at the library and firmware level, it is not possible to specify that a model use a device that is not visible.
# * Within the model itself, the usual usage pattern is to initialize objects (model and model tensors) on CPU and then move them to GPU by running `.cuda()` operations in the necessary places. Parallelization across multiple GPUs on the same machine is done via the `nn.DataParallel` module. Parallelization across multiple GPUs running on different machines is not a built-in but is available via services like `horovod`. For more details refer to the notebook ["Notes on parallel and distributed training in PyTorch"](https://www.kaggle.com/residentmario/notes-on-parallel-distributed-training-in-pytorch).
# * Assuming that a GPU is visible and that the logic for interfacing with that GPU in the model is sound, checking visibility settings should be sufficient for getting the model running.
# 
# 
# ## Device usage
# * The best tool for checking device usage interactively is `nvidia-smi`. `nvidia-smi` is a command-line device monitoring toolkit installed as part of the CUDA SDK. Example output:
# 
#     ```
#     Sun Jan 26 16:46:05 2020       
#     +-----------------------------------------------------------------------------+
#     | NVIDIA-SMI 410.78       Driver Version: 410.78       CUDA Version: 10.0     |
#     |-------------------------------+----------------------+----------------------+
#     | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#     | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#     |===============================+======================+======================|
#     |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
#     | N/A   39C    P0    26W / 250W |     74MiB / 16280MiB |      0%      Default |
#     +-------------------------------+----------------------+----------------------+
# 
#     +-----------------------------------------------------------------------------+
#     | Processes:                                                       GPU Memory |
#     |  GPU       PID   Type   Process name                             Usage      |
#     |=============================================================================|
#     +-----------------------------------------------------------------------------+
#     ```
# 
# * `nvidia-smi` will have a separate output for every visible card on the machine. The list of visible devices should be the same as that visible to PyTorch via the `torch.cuda` library commands (if it's not you have a serious problem in your `torch` install!).
# 
#   The best solution is to use `watch -n 2 nvidia-smi` or similar to track GPU usage with periodic updates. However this command annoyingly fails when run in a Jupyter notebook; it displays only the first line of output (I don't know why). So you will have to run this command from a terminal instance.
# * PyTorch dynamically allocates memory as it is needed. TensorFlow is more annoying: it allocates the entire GPU to itself when it is initialized. Rather obviously, that makes the output of `nvidia-smi` much less useful (I'm not clear on how you determine TensorFlow's true memory utilization, but I'm sure there's a library feature somewhere for that).
# 
# 
# ## Memory leaks
# * The easiest way to create a memory leak during a run is to accumulate gradients on a tensor that are not used (e.g. missing a `torch.no_grad` or `requires_grad=False` call). The accumulated gradient data stick around until it is either consumed (via `backwards`) or the entire object is garbage collected.
# * The easiest way to create a memory leak between runs is to accumulate gradients, fail to consume them, and also fail to garbage collect the tensor.
# * If you run into an unexpected OOM error (e.g. one not caused by the model parameter space being too large for the current device) one of these two causes is likely the cause.
