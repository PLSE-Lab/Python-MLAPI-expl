#!/usr/bin/env python
# coding: utf-8

# Something I see folks doing a lot (and that I have definintely done myself!) is using the *absolute* file path rather than the *relative* file path when reading in files. Let's talk about what the difference is and why you should always try to use the relative file path.
# 
# ___
# 
# ### File paths
# 
# First, a quick refresher on file paths/structures. (You can skip this bit if you already know it.) In most compyting environments, folders are arranged hierarchically, like a tree. The top-level file, the one that has all of the other files nested inside of it, is called the "root", and is written "\". 
# 
# > A **directory** is just another name for a folder. Generally, they're called folders if you're interacting with them using a graphical interface, and directories if you're interacting with them programmatically. 
# 
# You're almost never going to be working in the root directory. Instead, you'll probably be further down in the file system. For this notebook, let's assume that our code and data are in seperate folders within the same partent folder, that is itself inside the root folder.
#  
# ![](https://i.imgur.com/oJr7T20.png)
# 
# 

# ###Absolute file path
# ___
# 
# The *absolute path* is the patch from the root directory to the file you want. 
# 
# ![](https://i.imgur.com/QBI1YGE.png)
# 
# You can tell  a file path is an absolute file path if it starts with "\" or "/". (Which one will depend on the operating system hosting the files structure.)
# 
# > **Why shouldn't you use the absolute path?** Because the absolute path starts at the root and moves down through each file, it will break if any of the folder names or locations are different. Since almost all computers will have some differences in thier directory structure, if you refer to an absolute file path your code is very unlikely to run in a new computational environment.

# In[ ]:


# library we'll need
import pandas as pd

# read in data using absolute path
data = pd.read_csv("/kaggle/input/WorldCupMatches.csv")


# ###Relative file path
# ___
# 
# The *relative path* is the patch from the root directory to the file you want. Only relative paths will include "..", which means "go up to the parent directory of the current directory".
# 
# ![](https://i.imgur.com/pdhKE3Q.png)
# 
# > **Why should you use the relative path?** Because the relative path only relies on part of the directory structure being the same, if you can share your code along with the relevent file strucure your cdoe will still work.

# In[ ]:


# read in data using relative path
data = pd.read_csv("../input/WorldCupMatches.csv")

