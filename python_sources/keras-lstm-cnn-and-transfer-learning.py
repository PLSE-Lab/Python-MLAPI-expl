#!/usr/bin/env python
# coding: utf-8

# # <b>Small data? Not a big deal.</b>

# Hi, there!
# 
# This kernel is focused on neural network approach with the help of Keras, and aims to show that it can produce great results even on small data. 
# 
# We realized that the simplest neural network approach will lead to overfitting even if we perform some parameter tuning. 
# 
# What should we do in this case? One could advice to find more data.
# 

# ![](https://cdn.someecards.com/someecards/usercards/the-data-says-we-need-more-data-d39d3.png)

# However, we decided to experiment with transfer learning instead, and check if it will work here. 
# 
# This notebook describes next steps:
# 1. Data preprocessing
# 2. Experiment with various NN architectures (LSTM, CNN)
# 3. Transfer learning 
# 4. Model tuning
# 
