#!/usr/bin/env python
# coding: utf-8

# This Kernel shows you how to import custom functions you've written in a script and use them in a notebook kernel. To see how to write & save functions in a script, go here: [https://www.kaggle.com/rtatman/sample-function-script/](https://www.kaggle.com/rtatman/sample-function-script/).
# 
# To use this technique in a new kernel, you'll need to:
# 
# * make sure you've run & commit your script in order to generate the output file
# * add your script to your notebook kernel as a datasource (check out [the documentation](https://www.kaggle.com/docs/kernels) if you're a little fuzzy to do this :) 
# 
# Then you're good to go!

# In[ ]:


# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/my_functions.py", dst = "../working/my_functions.py")

# import all our functions
from my_functions import *


# In[ ]:


# we can now use this function!
times_two_plus_three(4)


# In[ ]:


# and this one too!
print_cat()


# And that's all there is to it. Good luck & happy coding! :)
