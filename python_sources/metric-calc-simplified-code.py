#!/usr/bin/env python
# coding: utf-8

# Showing here a way to calculate the competition metric in a straightforward manner without going thru separate threshold calculation code.  I have used this metric code in my DL model and every submission result has come very close to what I came up for my validation data using this code.  This metric code is based on my understanding of the metric from the discussion here:
# 
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62717
# 
# and the notebook here:
# 
# https://www.kaggle.com/pestipeti/explanation-of-scoring-metric
# 
# Note: The script assumes that your ground truth and predicted masks have values 0 to 1. Not 0 to 255.

# A few examples:
# for IOU of 0.6464, you should get an average precision value of 0.3 (as shown in the https://www.kaggle.com/pestipeti/explanation-of-scoring-metric)
# 
# (0.6464-0.451)x2x10 = 3.908,  floor(3.908) = 3,  3/10 = 0.3 - average precision value
# 
# IOU of 0.5:
# (0.5-0.451)x2x10 = 0.98,  floor(0.98) = 0,  0/10 = 0 - average precision value
# 
# IOU of 0.51:
# (0.51-0.451)x2x10 = 1.18,  floor(1.18) = 1,  1/10 = 0.1 - average precision value
# 
# IOU of 0.94:
# (0.94-0.451)x2x10 = 9.78,  floor(9.78) = 9,  9/10 = 0.9 - average precision value
# 
# IOU of 0.98:
# (0.98-0.451)x2x10 = 10.58,  floor(10.58) = 10,  10/10 = 1.0 - average precision value
# 

# In[ ]:


def calculate_metrics(self, outputs, targets, **kwargs):
    outputs = (outputs>0).float()
    targets = targets.view(-1,128,128)
    # This is different from the Dice logic
    # calculating intersection and union for a batch
    intersect = (outputs*targets).sum(2).sum(1)
    union = (outputs+targets).sum(2).sum(1)
    # Calculates the IOU, 0.001 makes sure the iou is 1 in case intersect
    # and union are both zero (where mask is zero and predicted mask is zero) -
    # this is a case of perfect match as well.
    iou = (intersect+0.001)/(union-intersect+0.001)
    # This simple logic here will give the correct result for precision
    # without going thru each threshold
    classification_precision = ((iou-0.451)*2*10).floor()/10
    # makes any ious less than 0.451 zero as well
    classification_precision[classification_precision<0] = 0
    # If you don't want the mean for the batch, you can return a list
    # of the classification_precision as well.  
    classification_precision = classification_precision.mean()
    return classification_precision


# Let me know if you find any error in my logic by posting comments.  Do upvote if you find it useful.
