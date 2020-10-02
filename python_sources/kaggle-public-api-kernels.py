#!/usr/bin/env python
# coding: utf-8

# # What is Kaggle Public API?
# kaggle API is for controlling your kaggle completitions(submit submission, show leaderboard...), kernels(push, pull...), datasets (download, create...).
# 
# Check this [kaggle API doc page](https://www.kaggle.com/docs/api), and [this github page](https://github.com/Kaggle/kaggle-api#kernels).
# ## What is Kaggle Kernels API?
# The part about kernels in the Kaggle public API.

# # Set up kaggle command line tool
# 
# For Linux or MacOS environments, you can use the following commands in your computer. For detail, check this: [detail reference to set up kaggle command line tools](https://www.kaggle.com/docs/api#interacting-with-kernels)
# 
# For the token for Kaggle's public API in the third command, reference [this](https://www.kaggle.com/docs/api#authentication).
# 
# ```sh
# pip install kaggle 
# mkdir $HOME/.kaggle 
# echo '{"username":"k1gaggle","key":"b1946ac92492d2347c6235b4d2611184"}' > $HOME/.kaggle/kaggle.json  # the token here is a fake one, you should put yours here
# chmod 600 $HOME/.kaggle/kaggle.json
# ```
# 
# # command line examples
# For complete APIs, check [official github page API](https://github.com/Kaggle/kaggle-api#kernels).
# 
# ```sh
# # show my kernel for the 'siim-acr-pneumothorax-segmentation' competition
# $ kaggle kernels list -m --competition siim-acr-pneumothorax-segmentation  
# ref                               title                    author  lastRunTime          totalVotes  
# --------------------------------  -----------------------  ------  -------------------  ----------  
# k1gaggle/learn-pytorch-mask-rcnn  learn pytorch mask-rcnn  V Zhou  2019-07-04 05:20:10           0  
# 
# # pull kernel code
# $ mkdir kaggle_api_test && kaggle kernels pull k1gaggle/learn-pytorch-mask-rcnn -p kaggle_api_test
# Source code downloaded to kaggle_api_test/learn-pytorch-mask-rcnn.ipynb
# 
# # retrive kernel's output
# $ kaggle kernels output k1gaggle/learn-pytorch-mask-rcnn -p kaggle_api_test
# Output file downloaded to kaggle_api_test/kernel.py
# Output file downloaded to kaggle_api_test/submission.csv
# Output file downloaded to kaggle_api_test/run_state_KernelRunningState.TRAINING_DONE.pkl
# ...
# Kernel log downloaded to kaggle_api_test/learn-pytorch-mask-rcnn.log 
# 
# # Get the status of the latest kernel run
# $ kaggle kernels status k1gaggle/learn-pytorch-mask-rcnn 
# k1gaggle/learn-pytorch-mask-rcnn has status "complete"
# ```
# 
# # Using API to pull code and submit new code
# ```sh
# # download your online kernel, so you can edit the kernel metadata based on this online existed kernel.
# # besides, to download the kernel, there should be a commit verion for this kernel (k1gaggle/learn-pytorch-mask-rcnn in this example)
# $ kaggle kernels pull k1gaggle/learn-pytorch-mask-rcnn -m -p test  # -m for also downloading metadata alongside kernel code
# $ ls test
# kernel-metadata.json          learn-pytorch-mask-rcnn.ipynb
# 
# # change your metadata of your kernel, reference https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata
# $ vi test/kernel-metadata.json  
# $ kaggle kernels push -p test
# Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
# Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/k1gaggle/learn-api
# 
# $ kaggle kernels status k1gaggle/learn-api
# k1gaggle/learn-api has status "running"
# 
# # pull again, just to overwrite the 'id_no' in metadata, just in case
# $ kaggle kernels pull k1gaggle/learn-api -m -p test
# 
# # edit your ipynb code
# $ kaggle kernels push -p test
# Kernel version 2 successfully pushed.  Please check progress at https://www.kaggle.com/k1gaggle/learn-api
# 
# $ kaggle kernels status k1gaggle/learn-api
# k1gaggle/learn-api has status "running"
# 
# # besides, the pushed kernels are 'committed' versions, which makes it easier to see previous versions in webpages too.
# ```
# 
# So we can edit the code in our offline env, push it to kaggle online kernel, and then let it run and wait for its output.
# 
# # My view about kaggle kernels API
# 
# Using this API, you need to push and pull, somewhat inefficient. But your code history can be maintained locally using CVS, which is easier to review.
# 
# For notebook type, you should edit .ipynb file then push that to the server(there are commands to convert .py to .ipynb, you can also use jupyter notebook/lab to edit .ipynb file). For script type kernel, it is python script.
# 
# If your scripts contains multiple files (different modules or something), it is not easy to do. As you can only add your
# "utility" scripts as separate kernels, and refer to that kernel if you want to use. An workaround way is that you upload your code to github, and in your kernel code, you download the github codes and run them.
# 
# ## Pros
# - If you need to do submission **automation**, the API will help a lot. 
# - If your network is unstable, this API can help too, so you won't suffer from disconnection from the web kernel.
# 
# ## Cons
# If you need to do interactive things (like debugging), you need to use the webpage.

# In[ ]:




