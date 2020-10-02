#!/usr/bin/env python
# coding: utf-8

# # 4. Exploring public models
# ### Airbus Ship Detection Challenge - A quick overview for computer vision noobs
# 
# &nbsp;
# 
# 
# Hi, and welcome! This is the fourth kernel of the series `Airbus Ship Detection Challenge - A quick overview for computer vision noobs.` This short kernel has two goals: first, to do a quick research of the currently trending public models and, second, to analyze the main approaches those models take.
# 
# 
# The full series consist of the following notebooks:
# 1. [Loading and visualizing the images](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images)
# 2. [Understanding and plotting rle bounding boxes](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes) 
# 3. [Basic exploratory analysis](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis)
# 4. *[Exploring public models](https://www.kaggle.com/julian3833/4-exploring-models-shared-by-the-community)*
# 5. [1.0 submission: submitting the test file](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file)
# 
# This is an ongoing project, so expect more notebooks to be added to the series soon. Actually, we are currently working on the following ones:
# * Understanding and exploiting the data leak
# * A quick overview of image segmentation domain
# * Jumping into Pytorch
# * Understanding U-net
# * Proposing a simple improvement to U-net model

# ## 1. Searching for public models

# We can access the most relevant models sorting by best score in the [Kernel's tab](https://www.kaggle.com/c/airbus-ship-detection/kernels?sortBy=scoreDescending&group=everyone&pageSize=20&competitionId=9988) of the competition.
# 
# <img src="https://i.imgur.com/LyaGV0A.jpg" width="70%" height="70%" />
# 
# Note that some kernels with models don't have a submission associated with them and do not appear in the top of the list. For example, the one below reports a 0.89 on it's title, higher than the top kernel in the above list. Be aware of this, you may find that the hottests models don't have a submission associated. Typically, the title will suggest there is a model in that kernel.
# 
# <img src="https://i.imgur.com/ua49m0G.jpg" width="40%" height="40%" />
# 
# On the other hand, the models are not independent: most of them are forks or improvements of previously shared models. 
# 
# In the rest of this notebook we will review the approaches, models and technologies proposed in some of the currently trending kernels, as well as their evolution trees.

# ## 2. Currently trending models

# While reading  these kernels, as we expected, we found a lot of keywords and concepts which we don't know.  We took notes of the most relevant keywords for further research on the relevant topics of the Airbus Challenge. We plan to go deeper in the following topics in future notebooks: `Image segmentation`, `Pytorch`, `U-net`. 

# ### 2.1. **0.89** - ResNet34 + U-net model by [iafoss](https://www.kaggle.com/iafoss)
# 
# &nbsp;
# 
# The most relevant public model at stake right now is the ResNet34 + U-net ensemble proposed by the user [iafoss](https://www.kaggle.com/iafoss) in his series of three kernels. All this work is based on the `fastai` package which, in turn, is fully based on `pytorch`.  
# 
# 
# Give a quick look to the following table with a summary of the notebooks; we will go into some extra details right below.
# 
# 
# | Kernel        |   Creation date | Description  | Keywords |
# | ----- |:----|: ---|: ---|
# | [Fine-tuning ResNet34 on ship detection](https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection/notebook)      |   August 26th |   `Creates a ship detection (classification) model using transfer learning and ResNet34.` | `ResNet34, object detection, learning rate annealing, class imbalance, different input sizes for same network`|
# | [U-net34 (dice 0.87+)](https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook)      |  September 1st    |  `Builds a U-net for image segmentation inspired by a fast.ai lesson using the ship detection ResNet34 from previous notebook for the encoder part.  Dice loss and Focal loss are presented and used too.` | `SSD, rotating bounding boxes, U-net, image segmentation, Carvana Challenge, dice loss, focal loss` |
# |[U-net34 submission (0.89 public LB)](https://www.kaggle.com/iafoss/unet34-submission-0-89-public-lb)    |  September 4th    |   `Ensembles the ship/no-ship ResNet34 classifier from the first notebook with the U-net image segmentation from the second one.` | `Ensemble`
# 
# .
# 
# 
# In the [first kernel, `Fine-tuning ResNet34 on ship detection`](https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection/notebook), the author trains a model for object detection retraining just some layers of a pretrained ResNet34.  This is a ship detection model (this is, given a picture, it determines whether there is a ship in it or not). The reported precision for the detection task is of ~98%. On the other hand, iafoss does a very focused training on some specific layers using learning rate annealing over images of different resolutions using the same network. Note that this model doesn't fully address the Challenge's problem, which is `image segmentation`.
# 
# <!-- TODO: add link to notebook when written --> 
# 
# The [second kernel, `U-net34 (dice 0.87+)`](https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook), addresses the problem of `image segmentation` itself. Image segmentation is the most evolved of a series of related deep learning / computer vision tasks as the picture below shows and we will cover in a *future notebook*.  
# 
# <img src='http://i.imgur.com/2JauPyI.png' />
# 
# The author mentions 2 possible approaches - `U-net` and `SSD` - and explains pros and cons of each of them. After that, he proposes a customized `U-net` model, importing the `ResNet34` of the previous kernel as the encoder part.  This model is trained very selectively only over images with at least one ship, dropping almost %80 of the dataset, based on the class imbalance observation which is discussed in [this previous notebook](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis) of the series. 
# 
# `U-net`, as we will see in *another future notebook*, is a fully convolutional network with a symmetric encoding-decoding architecture which works pretty well for `image segmentation`:
# 
# <img src='http://deeplearning.net/tutorial/_images/unet.jpg' width="50%" height="50%"/>
# 
# Image taken from [deelearning.net](http://deeplearning.net/tutorial/unet.html)
# 
# Finally, the [third kernel, `U-net34 submission (0.89 public LB)`](https://www.kaggle.com/iafoss/unet34-submission-0-89-public-lb) stacks these two models achieving a 0.889 score. So, first, the classifier determines whether a picture has ships or not and, if the picture has ships, the U-net determines where. 
# 
# There are few other kernels which are forks of some kernel of these series. In fact, the current top-ranked kernel is just a fork of this work without any addition.
#   
#  ### 2.2. **0.847**  - Baseline trivial empty submission
#  
#  &nbsp;
#  
#  [Early](https://www.kaggle.com/c/airbus-ship-detection/discussion/62376) on the competition it was publicly known that an empty submission reported an 84.7 score, because of the class imbalance which was discussed on [this kernel](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis). This model can be considered a baseline although 25% of the competitors scored lower than the one it achieves. The empty submission  can be constructed with literally one line of code:
# 
# The kernels [One line Base Model LB:0.847](https://www.kaggle.com/paulorzp/one-line-base-model-lb-0-847) and [Naive Model](https://www.kaggle.com/npatta01/naive-model) contain this solution, this code is taken from the first one:
# 
# 

# In[ ]:


#https://www.kaggle.com/paulorzp/one-line-base-model-lb-0-847
import os; 
import pandas as pd

pd.read_csv('../input/sample_submission.csv', 
            converters = {'EncodedPixels': lambda p: None}).to_csv('submission_paulorzp.csv', index=False)


# ### 2.3. **1.0** - trivial submission: submitting the test file
# 
# &nbsp;
# 
# As we will discuss in a *future notebook* of the series, the competition had a [data leakage](https://www.kaggle.com/c/airbus-ship-detection/discussion/64355) (reported on August 28th), trivializing the full competition. The organizers [took action](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388) about this and will change the evaluation set. Meanwhile, they shared the segmentations for the test set. We wrote a [7-lines notebook](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file) which creates a submission with this data, reaching a 1.0 score on the leaderboard (until the new test set is released and the scores are resetted). 
# 
# 
# 

# ### 2.4. More public models
# &nbsp;
# 
# The most relevant of the rest of the kernels are strongly based on [kmader](https://www.kaggle.com/kmader/)'s early work. As iafoss, he has built a ship detector using transfer learning and a U-net for image segmentation. kmader didn't publish an ensemble of these two models as iafoss did, but [hmendonca](https://www.kaggle.com/hmendonca) did, obtaining a 85.3 score.  
# 
# Returning to kmader's work, the ship detector can be found in the `Transfer Learning for Boat or No-Boat` kernel. It's a `DenseNet169` pretrained net with few extra trained layers around it. It's implemented with `Keras`, and the kernel uses `skimage` and `keras' data augumentation mechanisms`.  On the other hand, the U-net image segmentation model presented in  [Baseline U-net model](https://www.kaggle.com/kmader/baseline-u-net-model-part-1/notebook) is a hand-crafted implementation of a basic `U-net` model, also in `Keras`.  This model is splitted in two kernels in fact and, as far as I could understand, it predicts no-ship for every pixels (generating the 0.847 score, as the trivial baseline). 
# 
# 
# | Kernel  |   Description  |   Keywords  |
# | ----- |:----- |: ---|
# | [Baseline U-net model - Part 1](https://www.kaggle.com/kmader/baseline-u-net-model-part-1/notebook)    and [Part 2](https://www.kaggle.com/kmader/from-trained-u-net-to-submission-part-2) |  `U-net model implemented with Keras, splitted in 2 notebooks. As far as I understand, the model outputs the empty submission.` | `U-net`, `Keras`
# | [Transfer Learning for Boat or No-Boat](https://www.kaggle.com/kmader/transfer-learning-for-boat-or-no-boat)   |  `Transfer learning for ship/no-ship classification`| `DenseNet169`, `ImageDataGenerator`, `Keras' callbacks`
# | [Classification and Segmentation (-FP)](https://www.kaggle.com/hmendonca/classification-and-segmentation-fp)  |  `An ensemble of the above transfer leaning ship/no-ship classifier and the tweaked U-net model` |
# 
# &nbsp;
# 
# <!-- | [U-Net Model with submission](https://www.kaggle.com/hmendonca/u-net-model-with-submission)    |  A fork with some tweaks of the previous model ||[U-Net Model with Abstract layer submission](https://www.kaggle.com/ashishpatel26/u-net-model-with-abstract-layer-submission)|  A fork of the previous U-net model with no visible addition| -->
#     

# ## 3. Conclusion and next steps

# ### Conclusion
# 
# As we can see, the general approach of a `ship detection + image segmentation ensemble` is the most prominent in the public models right now, one using `pytorch` and one using `keras`.  This is a common way for approaching the strong imbalance with respect to the pixel labels. The image segmentation model is trained dropping all the images without ships by one author and dropping most of those images by the other. Also, in both cases, the ship detection task is addressed with transfer learning while the image segmentation task is addressed with a `U-net` model.  Currently, iafoss' is working on an `SSD`, a more complex but also more promising model, at least from his point of view.
# 
# 
# ### References
# #### Kernels
# *  [Fine-tuning ResNet34 on ship detection](https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection/notebook)
# * [U-net34 (dice 0.87+)](https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook) 
# * [U-net34 submission (0.89 public LB)](https://www.kaggle.com/iafoss/unet34-submission-0-89-public-lb)
# * [Baseline U-net model - Part 1](https://www.kaggle.com/kmader/baseline-u-net-model-part-1/notebook)    and [Part 2](https://www.kaggle.com/kmader/from-trained-u-net-to-submission-part-2) 
# * [Transfer Learning for Boat or No-Boat](https://www.kaggle.com/kmader/transfer-learning-for-boat-or-no-boat)
# * [Classification and Segmentation (-FP)](https://www.kaggle.com/hmendonca/classification-and-segmentation-fp) 
# * [One line Base Model LB:0.847](https://www.kaggle.com/paulorzp/one-line-base-model-lb-0-847) 
# * [Naive Model](https://www.kaggle.com/npatta01/naive-model)
# 
# 
# #### Other resources
# * [Deeplearning.net's U-net explanation](http://deeplearning.net/tutorial/unet.html)
# * [Deeplearning.net's FCN for 2D segmentation explanation](http://deeplearning.net/tutorial/fcn_2D_segm.html)
# * [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)
# * [How to do Semantic Segmentation using Deep learning](https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef)
# * [Deep Learning book](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html)
# 
# ### What's next?
# You can check the [next kernel](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file) of the series, where we create a trivial 1.0 scoring submission using the test segmentation csv shared by the organizers.
# 
