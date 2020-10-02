#!/usr/bin/env python
# coding: utf-8

# # CNN Analysis - Answer These Questions:

# ## What Are CNNs?

# Convolutional nets are a form of neural networks that are used for image classification.

# ## How Do CNNs Work?

# * Images are processed as "tensors"

# ## What are the Basic Components of CNNs?

# * **Tensor** - A scaler, matrix or array
# * **Scalar** - A single number, such as 7. Geometrically, this is a point.
# * **Vector** - A list of numbers, such as [4,7,3] Geometrically, this is a line.
# * **Matrix** - A rectangular grid of numbers, such as a spreadsheet. Geometrically, this is a two-dimensional plane. 
# * **Array** - This could be a stack of matrices, which would make a three-dimensional cube.
# 
# Source: https://skymind.ai/wiki/convolutional-network

# "A 2 x 3 x 2 tensor presented flatly"

# In[ ]:


from IPython.display import Image
Image('../input/cnn-explained/Matrix.png')

#Source: https://skymind.ai/wiki/convolutional-network


# The above flat matrix, presented via code:
#     
# [[[2,3],[3,5],[4,7]],[[3,4],[4,6],[5,8]]]
# 
# Source: https://skymind.ai/wiki/convolutional-network

# "A 2 x 3 x 2 tensor presented as a cube"

# In[ ]:


from IPython.display import Image
Image('../input/cnn-explained/Cube.png')

#Source: https://skymind.ai/wiki/convolutional-network


# Many tensors are 4 dimensions (pictured below). When an array has more than three dimensions, then it can be thought of as nested arrays, with a scalar being replaced with an array. This can happen indefinitely.

# In[1]:


from IPython.display import Image
Image('../input/cnn-explained/4d array.png')

#Source: https://skymind.ai/wiki/convolutional-network


# ## What are Important Considerations When Creating CNNs?

# ## Create a Basic CNN Here:

# In[ ]:





# ## Possible Sources (Feel Free to Research More - Just Google CNN Neural Network + "Keyword")

# * https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/
# * https://medium.com/intro-to-artificial-intelligence/simple-image-classification-using-deep-learning-deep-learning-series-2-5e5b89e97926
# * https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks
# * http://cs231n.github.io/convolutional-networks/
# * http://deeplearning.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
# * https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148
# * https://www.tensorflow.org/tutorials/estimators/cnn
# * https://www.tensorflow.org/tutorials/images/deep_cnn
# * https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
