#!/usr/bin/env python
# coding: utf-8

# # Object detection model architecture
# 
# Object detection and image segmentation are two related visual tasks that extend past simple classification and are more complex to architect for than simple classification. In **object detection** tasks, we are to predict the bounding box for objects in an image. For **image segmentation**, we are to divide the image into different aspects of the scene, on a pixel-by-pixel basis.
# 
# These are both intrinsically very hard tasks that require interesting model architectures to address.
# 
# ## One implementation of object detection
# For my notes on object detection I refer principally to the paper ["Deep Neural Networks for Object Detection"](https://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf), a 2013-or-so era NIPS paper from Google.
# 
# The core model architecture used in this paper is relatively simple: a seven-layer convolutional neural network. Five (!) separate models are trained, each with a 255x255 input size and some number of output categorical values, each one corresponding with a $24 \times 24$ chunk of the image. Each model predicts whether a given quadrant contains the object in question or doesn't. There is one one-shot model that predicts all of the quadrants containing the object, and four supporting models that each predict one side of the bounding box: top, bottom, left, right.
# 
# Images larger than 255x255 in size are broken up into subsector scans, with 20% overlap between neighboring scan areas.
# 
# If a scene contains multiple objects, or multiple instances of an object, then at least two of the predicted object boundaries with disagree. For example, if there are two lemons in the image, located right next to one another on a table, the left edge detector might fire two lines of masks, one for each of the lemons, whilst the one-shot predictor would fire over both lemons. Post-processing on the gridded predictions is used to find the number of bounding boxes and their corresponding masks which most satisfy the network outputs.
# 
# Masks are refined using an exhaustive search over a search space composed of the ten most common bounding box aspect ratios (as discovered using k-means clustering) over ten different image percentile dimensions ($[0.1, \ldots, 0.9]$). Each of these 100 total options is scored according to how well it agrees with the neural network masks, using a closed-form sum scoring experession.
# 
# These new images go through a *second* stage of network analysis. There's more steps to the pipeline that I am omitting.
# 
# ## Review of object detection architecture
# The paper that the above paragraph summarizes is a circa-2013 effort. The following section contains my notes on a [summary paper](https://arxiv.org/pdf/1807.05511.pdf) published in 2018 which contains a more broad and up-to-date view on solutions to this task.
# 
# Broadly speaking, there are two architectures for performing object detection:
# 
# * Bounding boxes are proposed by one system and crunched and assigned classes or rejected outright by another system.
# * Bounding box proposals and class assignments are integrated into a single system.
# 
# I look at an example architecture for each.
# 
# Detecting the objects in an image largely requires solving two major overall problems:
# 
# * Scale&madash;images may contain objects at any scale.
# * Diversification&mdash;the attributes which defines an object vary from object to object, and the obviousness of individual features may vary from scene to scene.
# 
# ### Multi-stage learning with R-CNN
# R-CNN (2014) is foundational for pipelines using the first approach. All of the pipeline designs that have succeeded it are modifications on its example, so its a good representative example of the former category of learners.
# 
# First, candidate bounding boxes are generated using [selective search (paper link)](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf). A large number of candidate regions are created by generating a large number of semi-randomized starting points, using an algorithm. The starting region is then "accumulated" by examining candidate neighboring regions and selecting those that perform well in any one of a handful of different linking metrics. The metrics used are based on size, color, texture, and other properties, all picked out of the toolbox of traditional image analysis techniques. All in all eighty different metrics are used! The large number of bounding boxes generated is then passed through an SVM trained on the data, which is used to score whether the given region is an object. Algorithms that rely on this algorithm can then select as input all candidates that score above a certain threshold according to the SVM.
# 
# Each candidate zone is warped and cropped to CNN input size, then processed using a CNN that extracts 4096-dimension feature vector for the image. The feature vectors are processed using a hierarchical treee of category specific linear support vector machines.
# 
# Candidate bounding boxes that are assigned a non-negative class are further refined using a technique known as **bounding box regression**, wherein category-specific regression models are run on the input bounding boxes (and normalized image pixel information?) to massage the bounding box shapes into a predefined set of bounding box aspect ratios. This additional step significantly improves model performance in practice! [Here's a reference on this technique](https://arxiv.org/pdf/1904.06805.pdf).
# 
# Finally, a technique known as **non-maximal suppression** is used to combine bounding boxes with a very high degree of overlap into a single, smoothed-out boundary. [Here's a reference on this technique](https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_01126.pdf).
# 
# For datasets not in the millions of images, fine-tuning is used, based on a pretrained model built using the Stanford image corpus.
# 
# ### Single-shot learning with YOLO
# YOLO is the reference model architecture and pipeline I will summarize for the integrated approach.
# 
# YOLO divides the input image into an $S \times S$ grid, with each grid cell is responsible for predicting the object centered in that grid cell. Each grid cell predicts B bounding boxes and their corresponding confidence scores. Thus $B \times S^2$ total boxes are created.
# 
# Grid-derived bounding boxes which have no confidence that the selected pixels are actually an object do not contribute to the (very complicated) loss function. Confidence in non-matches are penalized strongly in an IOU fashion. Confidence in IOU matches is penalized based on bounding box error.
# 
# YOLO has 22 convolutional layers and 2 fully connected layers.
# 
# YOLOv2 adopts some of the more classical featurization algorithms featured in R-CNN to the pipeline.
# 
# ## Review of image segmentation with DeepLab
# 
# We will use DeepLab as our reference model for semantic image segmentation.
# 
# TODO: write up [the DeepLab paper](https://arxiv.org/pdf/1606.00915.pdf).
