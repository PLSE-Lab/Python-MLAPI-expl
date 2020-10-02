#!/usr/bin/env python
# coding: utf-8

# # Simple chart explanation of what train, validate, and test set mean in machine learning. 
# 
# Most of the confusion lies in the difference between **validate** and **test**. 
# 
# **Bottom line**: **Validate** set is the data to evaluate the performance of different models trained using the train set. You may choose the best model out of those performance evaluated on the validate set. **That gives you your final model.** Whereas the **test** set is just to evaluate how generalized your model is. Your model won't be improved through test set. You never fine-tune your model based on the performance on the test set. 

# 
# <a href="https://imgur.com/i48905e"><img src="https://i.imgur.com/i48905e.jpg?1" title="source: imgur.com" /></a>

# 
