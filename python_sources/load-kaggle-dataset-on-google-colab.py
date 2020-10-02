#!/usr/bin/env python
# coding: utf-8

# # Easy way to load Kaggle dataset on Google Colab
# 
# I'm new at kaggle and I used to work on google colab for some time. As a result, I feel more comfortable working on colab than kaggle notebook.
# 
# Moreover, when the weekly GPU sessions finishes, it's convinent to use colab to make computations faster. So, I found this easy way to load kaggle dataset on google colab.

# At first you have to generate an api key from your profile. Then on top of the colab notebook run this:
# ```python
# import os
# 
# os.environ['KAGGLE_USERNAME'] = "username" # Your username
# os.environ['KAGGLE_KEY'] = "api_key"       # Api key provided by kaggle
# ```

# Now to download any dataset do this. I've showed this for the Dogs vs Cats dataset.
# ```sh
# !kaggle competitions download -c dogs-vs-cats -p /path/to/any/folder
# ```

# If you want the same directory style as kaggle, run this:
# 
# ```python
# try:
#   os.mkdir("/kaggle")
#   os.mkdir("/kaggle/input")
#   os.mkdir("/kaggle/input/dataset_name")
#   os.mkdir("/kaggle/working")
# except os.error:
#   pass
# ```
# 
# And to download the dataset to the correct folder, do this. Again I've showed this for the Dogs vs Cats dataset.
# 
# ```sh
# !kaggle competitions download -c dogs-vs-cats -p /kaggle/input/dogs-vs-cats
# ```
