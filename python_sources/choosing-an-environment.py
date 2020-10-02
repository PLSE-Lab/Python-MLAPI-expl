#!/usr/bin/env python
# coding: utf-8

# # Choosing an Environment
# Jake Lee, TA for COMS4701
# 
# Because installing machine learning packages and downloading data sets can get cumbersome, there are several existing solutions that you can use. Here, we'll go through some setups and walk through their pros and cons.
# 
# ## Before we start... Jupyter Notebook
# 
# I **strongly** recommend that you use Jupyter Notebook, whether it be on the cloud or on your local machine. Each cell can be run individually, and variables are shared between cells. This means that you can import/load your data once (which is often the most time-consuming) and train multiple models multiple times. Take a look at the tutorials to see how it's used.
# 
# *A word of caution: be wary of in-place operations when using Jupyter notebook (such as `a = a + 1`). Every time you run that specific cell, `a` will go up by 1! This is especially an issue with data preprocessing (such as `data = data / 255`); either don't do in-place operations or put your preprocessing in a separate cell and never run it twice!*

# # Kaggle Kernel
# 
# A Kaggle kernel is a simple way to get started. It's a Jupyter Notebook environment built-in to Kaggle, supported by Google. It's especially useful because it provides a K80/P100/T4 GPU for free. Let's walk through step by step on how to set one up.
# 
# ### 1. Create a New Notebook
# Go to the "Notebooks" tab, and click "New Notebook".
# ![kernel1.png](attachment:kernel1.png)
# 
# ### 2. Set up Settings
# Set the following after expanding "advanced settings"
# 
# - Select Language: **Python**
# - Select Type: **Notebook**
#   - Script is mostly used for when submitting a kernel is part of the competition, so you're importing existing work. You can use it if you want, but it doesn't offer the flexibility of a notebook.
# - Select GPU preference:
#   - If you're just doing **Scikit** models, you do not need a GPU.
#   - If you're doing **deep learning** models, you need a GPU. However, availability may not be reliable (Kaggle may run out of GPUs and not let you start your kernel to work).
#   - Note that you're not required to do deep learning - the benchmark was set with a scikit model.
# - Enable Google Cloud Services: **Off**
# 
# and click "create".
# 
# ![kernel2.png](attachment:kernel2.png)
# 
# ### 3. Check sidebar
# After the notebook has finished loading, take a look at the sidebar.
# 
# - **Sessions**: This shows your hardware resource quota. Notice that you are allowed 10 CPUs and 1 GPU. Each notebook uses 1 CPU. The GPU in this case is likely an Nvidia K80.
#   - If you expand the dropdown, there are options to shut down other notebooks you are running.
# - **Data**:
#   - Under the **Input** section, you should see the data we provided for this competition. You can import it directly from `/kaggle/input`.
#   - Under the **Output** section, you can see any files you wrote to the current directory (e.g. saving to `submission.csv`). You can then download the file for submission.
#   - If you're running a script instead of a notebook, Kaggle should offer to submit the file after the entire script finishes.
# - **Versions**
# - **Settings**
#   - **Sharing**: Make sure this is on Private, or everyone else can see your code! If you click on this, you can select collaborators.
#   - **Internet**: On if you want to download data from OpenML or Keras.
#   - **GPU**: If you decide you want GPUs, you can turn it on here.
# 
# 
# ### 4. Code!
# You should now be able to code. Most packages should already be installed, including numpy, matplotlib, opencv, scikitlearn, tensorflow-gpu, keras, etc.

# # Google Colab
# Google Colab is essentially a Kaggle Notebook in Google Drive. It also provides a K80/P100 GPU for free. It's unclear if the specs are particularly better than Kaggle, although some users report that Kaggle Notebooks have more lag.
# 
# ### 1. Open Colab
# Go to http://colab.research.google.com. Select "NEW PYTHON 3 NOTEBOOK" from the welcome panel. A new Google Drive folder will be created for you at the root.
# ![colab1.png](attachment:colab1.png)
# 
# ### 2. Enable GPU (if needed)
# At the top toolbar, click "Edit" > "Notebook settings". 
# ![colab2.png](attachment:colab2.png)
# 
# ### 3. Upload/Download Data
# Click on the extremely small tab at the top left of the notebook. You can manage data uploading/downloading from the "Files" tab.
# 
# There is a partial MNIST dataset included in the `sample_data` directory - don't use this, it's not the full dataset!
# 
# ![colab3.png](attachment:colab3.png)
# ![colab4.png](attachment:colab4.png)
# 
# ### 4. Code!
# You should now be able to code. As with Kaggle Notebooks, most packages should already be installed.
# 
# **tip:** Tools > Settings... > Miscellaneous has some very useful settings that will bring your model to the next level.

# # Local Environment
# Do you not trust the cloud (TM)? Do you have a desktop with a powerful GPU? Do you want to code without needing internet access? No problem! You can run Jupyter Notebook Locally. Of course, you could also just write straight-up Python code, but I'll focus on Jupyter Notebook.
# 
# ### 1. Install Anaconda
# If you don't already have Anaconda installed, do so from here: https://www.anaconda.com/distribution/#download-section. It is a very popular python virtual environment manager.
# 
# ### 2. Install Packages
# You will need to install most python packages with `conda install keras`, etc. Conda does come with numpy and other scientific packages, however.
# 
# ### 3. Run Jupyter Notebook
# Open terminal/console and run `jupyter notebook`. Open the link in a browser as instructed (if one doesn't open for you).
# 
# ### 4. Write code!
# No special tricks for uploading/downloading data here, everything's running on your machine!

# # Google Cloud or AWS EC2
# 
# You shouldn't need dedicated VMs for this competition, unless you want more powerful GPUs or multiple GPUs. However, these are not free, and we discourage it.
# 
# For Google Cloud, I recommend the Deep Learning VM Click-to-Deploy, here: https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning
# 
# For Amazon AWS EC2, I recommend the Deep Learning AMI, here: https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/
# 
# These services do have first-time free credits for students, but it may take some time to get hardware quota allocations, etc.

# # Conclusion
# 
# There are a lot of different services out there to make machine learning as easy as possible. We covered:
# 
# - Kaggle Kernels: The easiest way to start, although it may be a little underpowered
# - Google Colab: Needs more data/file setup, but more reliable performance (allegedly)
# - Local Installation: If you need to develop without internet
# - Google Cloud or AWS EC2: If you're going way overkill
# 
# As always, if you enjoyed the writeup, click below to upvote!
