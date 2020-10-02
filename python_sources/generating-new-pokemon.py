#!/usr/bin/env python
# coding: utf-8

# # DCGAN with Differentiable Augmentation in Pytorch
# 
# Differentiable Augmentation (*DiffAugment*) is a technique to train GAN that was published in June 2020.
# Simply put, *DiffAugment* is a Data Augmentation technique design specifically for GAN. 
# It addresses the problem when the Discriminator memorizes the training set due to a limited amount of data.
# Traditionally, we simply add more data to the training set. 
# But most of the time collecting more data is expensive and time-consuming.
# 
# 
# ## Training Progression
# 
# DCGAN without *DiffAugment* starts to generate random noise which happens from epoch 226. 
# The loss graph below could be a hint of what happened during training. 
# 
# ![loss.jpg](attachment:loss.jpg)
# 
# Even though it managed to escape this mode around epoch 875, the image quality is unsatisfactory. 
# Furthermore, on the last epoch, this model starts showing the tendency to generate random noise again as shown in the figure below.
# 
# |DCGAN             |DCGAN + *DiffAugment* 0.3         |DCGAN + *DiffAugment* 0.5     |DCGAN + *DiffAugment* 1
# |---               |---                          |---                           |---   
# |![1000_dcgan.jpg](attachment:1000_dcgan.jpg)|![1000_dcgan_03.jpg](attachment:1000_dcgan_03.jpg)|![1000_dcgan_05.jpg](attachment:1000_dcgan_05.jpg)|![1000_dcgan_1.jpg](attachment:1000_dcgan_1.jpg)|
# 
# From the figure above, we can also compare the effect of *DiffAugment* with a different probability.
# DCGAN + *DiffAugment* with a probability of 0.3 tends to generate a somewhat similar-looking Pokemon. 
# This was alleviated in DCGAN + *DiffAugment* with a probability of 0.5. 
# DCGAN that use *DiffAugment* in every iteration produces the most varying Pokemon in term of color and shape.
# It is safe to conclude that *DiffAugment* helps Generator learn to generate more diverse samples.
# 
# 
# ## FID Score across epoch
# Finally, let's look at the FID Score in the graph below.
# 
# ![fid.jpg](attachment:fid.jpg)
# 
# Training without *DiffAugment* pushes the network into Mode Collapse where G just generates a random noise with a very high FID score.
# On the other hand, *DiffAugment* helps stabilized the FID score, and the higher the probability the less fluctuation in FID score observed.
# 
# 
# ## Some Pokemon candidate from this experiment:
# 
# ![993.jpeg](attachment:993.jpeg)![939.jpeg](attachment:939.jpeg)![777.jpeg](attachment:777.jpeg)![641.jpeg](attachment:641.jpeg)![590.jpeg](attachment:590.jpeg)![449.jpeg](attachment:449.jpeg)![279.jpeg](attachment:279.jpeg)![278.jpeg](attachment:278.jpeg)![225.jpeg](attachment:225.jpeg)![59.jpeg](attachment:59.jpeg)![58.jpeg](attachment:58.jpeg)![57.jpeg](attachment:57.jpeg)![9.jpeg](attachment:9.jpeg)
# 
# 
# Link to Github Project [link](https://github.com/kvpratama/gan/tree/master/pokemon_dcgan).
# 
# 
