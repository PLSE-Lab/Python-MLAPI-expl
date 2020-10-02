#!/usr/bin/env python
# coding: utf-8

# Thanks for Dataset To https://www.kaggle.com/vbmokin
# https://www.kaggle.com/vbmokin/datasets-for-river-water-quality-prediction
# 
# 
# This is a collection of some thematically related datasets that are suitable for different types of regression analysis. 
# 
# Datasets has data of river water quality from 8 consecutive stations of the state water monitoring system. It's should predict the value in the eighth station by the first seven stations. The numbering of stations in the dataset is done from the target station upstream, ie closest to it - first, upstream - second, etc.
# 
# Data are average monthly. 
# 
# The number of observations on stations is different (from 4 to about 20 years).
# 
# 
# ## Tasks:
# 
# Questions to be answered by regression analysis and forecasting have been possed:
# 
# * Analysis of data dependences, including EDA.
# 
# * Prediction the target data (water quaity in the target station) with the highest accuracy.
# 
# * Analysis of impact on the prediction accuracy of the first two stations (1-2) and the next five (3-7) stations separately.
# 
# 
# ## One indicator of river water monitoring data:
# 
# * [Prediction BOD in river water](https://www.kaggle.com/vbmokin/prediction-bod-in-river-water) - dataset has data of the amount of biochemical oxygen demand, which is determined in 5 days ("BOD5" or "BOD"), in river water (the maximum permissible value in Ukraine is 3 mgO/cub. dm)
# 
# * [Suspended substances prediction in river water](https://www.kaggle.com/vbmokin/suspended-substances-prediction-in-river-water) - dataset has data of the amount of suspended substances concentration ("Suspended") in river water (the maximum permissible value in Ukraine is 15 mg/cub. dm)
# 
# * [Ammonium prediction in river water](https://www.kaggle.com/vbmokin/ammonium-prediction-in-river-water) - dataset has data of the Ammonium ions concentration in river water (the maximum permissible value in Ukraine is 0.5 mg/cub. dm)
# 
# * [Phosphate prediction in river water](https://www.kaggle.com/vbmokin/phosphate-prediction-in-river-water) - dataset has data of the amount of concentration of phosphate ions (polyphosphates) in river water, mg/cub. dm
# 
# 
# ## Many indicators of river water monitoring data:
# 
#   [Dissolved oxygen prediction in river water](https://www.kaggle.com/vbmokin/dissolved-oxygen-prediction-in-river-water) - dataset has data of the 5 indicators of river water quality:
# 
# 
# - Dissolved oxygen (O2) is measured in mgO2/cub. dm (ie milligrams of oxygen (O2) in the cubic decimeter);
# 
# - Ammonium ions (NH4) concentration is measured in mg/cub. dm (ie milligrams in the cubic decimeter);
# 
# - Nitrite ions (NO2) concentration is measured in mg/cub. dm (ie milligrams in the cubic decimeter);
# 
# - Nitrate ions (NO3) concentration is measured in mg/cub. dm (ie milligrams in the cubic decimeter);
# 
# - Biochemical oxygen demand, which is determined in 5 days ("BOD5" or "BOD"). BOD5 is measured in mgO/cub. dm (ie milligrams of oxygen in the cubic decimeter).
# 
# 
# The minimum permissible value of O2 in Ukraine is 4 mgO2/cub. dm.
# 
# 
# ## InClass Prediction Competitions for these dataset:
# 
# A Competitions will soon be based on these dataset, where you can check the accuracy of your predictions.
# 
# Currently, only 
# 
# **[InClass Prediction Competition "Prediction BOD in river water"](https://www.kaggle.com/c/bod-in-river-water)**
# 
# is available for the same observation dates and monitoring stations. 
# I invite you to participate!
# 
# 
# ## Acknowledgements:
# 
# I thank the
# 
# - **[State Water Resources Agency of Ukraine](https://www.davr.gov.ua/)**
# 
# - **[Ukrainian Open Data Portal](https://data.gov.ua/)**
# 
# 
# for providing data of real water monitoring data which used for creation these datasets.
