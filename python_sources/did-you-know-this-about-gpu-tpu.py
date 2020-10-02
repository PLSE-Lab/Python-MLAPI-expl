#!/usr/bin/env python
# coding: utf-8

# ![](https://cdn.igromania.ru/mnt/news/2/1/c/a/c/d/89417/3a515960a4229c6a_1200xH.jpg)

# *It is said that if CPU is brain of a Computer then GPU is its soul. <br>
# GPU has been one of the greatest invention in the human history with extensive features most of them <br>
# are in high performance in gaming, high intensity graphics software, deep learning, rendering 3D 
# animations & video and much more.*

# ## GPU Architecture

# Graphics card connect to the computer through the motherboard. The motherboard supplies power to the 
# card and lets it communicate with the CPU. Newer graphics cards often require more power than the motherboard
# can provide, so they may have a direct connection to the computer's power supply.
# Most of computers with low prices have integrated GPUs where the graphics processing unit (GPU) is built onto
# the same die as the CPU. A dedicated graphics card is a piece of hardware used to manage the graphics performance
# of a computer with upto thousands of cores. It has its own dedicated RAM, fan for cooling purpose and cache memory although less compared to CPU as it only helps in same repetitve tasks.

# ## What makes GPU special ??

# Looking at the architecture, CPU is composed of few core with heavy amount of cache memory that can handle few  
# software threads simultaneously, While GPU concerns technology of parallel computing.
# Each GPUs core is simpler and runs at a lower frequency. Hence, their single - thread performance is much lower
# than that of CPUs but when they work on the large scale and parallely to perform a task, tasks are successivly 
# executed faster than CPU as they have multiple core and all the instructions are executing at a time the GPU architecture is suitable to finish the specific job much faster. Meanwhile in CPU each core is strong and considering high frequency its processing power is significant so a CPU core can execute a big instruction set but not too many times and also cannot perform execution of single task parallely. Hence these methods in GPU makes itself precise in its very own applications. As more sophisticated the GPU, the higher the resolution and the faster and smoother the motion in games and applications.

# ## Why not GPU instead of CPU ??

# In simple language GPU is very good at data-parallel computing while CPU is good at parallel processing.
# It focuses on distributing the data across different nodes, which operate on the data in parallel.
# Parallel computing is a type of computation in which many calculations or the execution of processes are 
# carried out simultaneously. Means GPU can execute one task at a time parallely with different cores, meanwhile 
# CPU can execute task but keeping other processess parallel like keeping drivers, system softwares, output devices etc up to date. Hence, GPUs are not suited to handle tasks that do not significantly benefit from or cannot be parallelized, including many common consumer applications such as word processors or spreadsheets etc.<br>
#    As OS needs to look at 100s of different types of data and make various decisions which all depends on each other<br>
# and as GPU cannot perform this tasks single handedly we need CPU.

# ![](https://analyticsindiamag.com/wp-content/uploads/2019/01/gpu_2.jpg)

# ##                                            GPU vs TPU

# GPU is a processor designed to accelerate the rendering of graphics and for parallel computing while TPU is 
# designed to perform deep & machine learning's complex task by using tensorflow framework. GPU is general purpose processor while TPU is matrix processor. GPU can be used in 3D applications, photo editing, high level experience in gaming etc, while TPU is actually developed to help researches and developers in the era of cutting edge technology in AI & ML.

# ## TPU Architecture

# Each TPU version defines the specific hardware characteristics of a TPU device.<br> 
# TPU v2 has upto 512 total TPU cores and 4 TiB of total memory & <br>
# TPU v3 has 2048 total TPU cores and 32 TiB of total memory. <br>
# Each TPU core has scalar, vector, and matrix units (MXU). <br>
# The MXU provides the bulk of the compute power in a TPU chip. <br>
# Each MXU is capable of performing 16K multiply-accumulate operations in each cycle.<br>

# ## What TPUs do

# TPU resources accelerate the performance of linear algebra computation by orgainizing multiplications and additions 
# by doing matrix multiplication which is used heavily in deep learning as well machine learning applications.
# TPUs minimize the time-to-accuracy when you train large, complex neural network models.
# Models that previously took weeks to train on other hardware platforms can converge in hours on TPUs.
# TPUs can't run word processors or execute bank transactions, but they can handle the massive multiplications
# and additions for neural networks, at blazingly fast speeds while consuming much less power.

# ## References
# https://cloud.google.com/tpu/docs/system-architecture <br>
# https://www.quora.com/What-is-the-difference-between-CPU-and-a-GPU-for-parallel-computing <br>
