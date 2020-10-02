#!/usr/bin/env python
# coding: utf-8

# # Exercise Introduction
# 
# To build and test your intuition for convolutions, you will design a vertical line detector.  We'll apply it to each part of an image to create a new tensor showing where there are vertical lines.
# 
# ![Imgur](https://i.imgur.com/op9Maqr.png)
# 
# 
# 
# 
# ---
# 
# # Import Utility Functions
# We'll use some small utilty functions to load raw image data, visualize the results and give hints on your answert, etc.  .
# 

# In[ ]:


from learntools.deep_learning.exercise_1 import load_my_image, apply_conv_to_image, show, print_hints


# # Example Convolution: Horizontal Line Detector
# Here is the convolution you saw in the video.  It's provided here as an example, and you shouldn't need to change it.

# In[ ]:


# Detects bright pixels over dark pixels. 
horizontal_line_conv = [[1, 1], 
                        [-1, -1]]


# # Your Turn: Vertical Line Detector
# 
# **Replace the question marks with numbers to make a vertical line detector and uncomment both lines of code in the cell below.**

# In[ ]:


vertical_line_conv = [[1, -1], 
                     [1, -1]]
#vertical line (convulation) detector


# **Once you have created vertical_line_conv in the cell above, add it as an additional item to `conv_list` in the next cell. Then run that cell.**

# In[ ]:


#code identifies both horizontal andvertical lines
conv_list = [horizontal_line_conv , vertical_line_conv]

original_image = load_my_image()
print("Original image")
show(original_image)
for conv in conv_list:
    filtered_image = apply_conv_to_image(conv, original_image)
    print("Output: ")
    show(filtered_image)
#applies both horizontal and vertiacl line (convulation) detection


# 
# **Above, you'll see the output of the horizontal line filter as well as the filter you added. If you got it right, the output of your filter will looks like this.**
# ![Imgur](https://i.imgur.com/uR2ngvK.png)
# 
# ---
# # Keep Going
# **Now you are ready to [combine convolutions into powerful models](https://www.kaggle.com/dansbecker/building-models-from-convolutions). These models are fun to work with, so keep going.**
# 
# ---
# 
# Have questions, comments or feedback?  Bring them to [the Learn forum](https://www.kaggle.com/learn-forum)
# 
# **[Deep Learning Track Home](https://www.kaggle.com/education/deep-learning)**
