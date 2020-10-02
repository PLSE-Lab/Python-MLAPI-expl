#!/usr/bin/env python
# coding: utf-8

# **How to create an auto-updating dataset on Kaggle**

# **Step 1: Identify a URL for a dataset that occasionally gets updated**
# 
# For example, I found [this dataset](https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-police-pedestrian-stops-and-vehicle-stops) that is updated Monday through Friday and is shared under a  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.  It details Police Pedestrian Stops and Vehicle Stops in Denver and consists of a CSV file, an xml file, and some shp/dwg/gdb files. 

# **Step 2: Create a new dataset on Kaggle by using the "create dataset from remote URL" feature**
# 
# When creating a new dataset at kaggle.com/datasets you should find that the data source selector has a list of options along the left-side menu.  Upload your dataset using the ["upload from remote URL](https://www.kaggle.com/product-feedback/75341#449911)" option and provide the URL to your file or files (each URL should point to a single file).  In the case of the Denver Police Pedestrian Stops and Vehicle Stops dataset one URL will point to the CSV file and a different URL will point to the XML file and so on.
# 
# ![](https://i.imgur.com/RYEz7F2.png)

# **Step 3: In the settings menu for your dataset turn on automatic updates.**
# 
# The default setting is "Never" but you can also change it to "Weekly" or "Monthly".
# 
# ![](https://i.imgur.com/eZQkE6m.png)
# 
# 
# 

# **Summary**:
# 
# Step 1: Identify a URL for a dataset that occasionally gets updated
# 
# Step 2: Create a new dataset on Kaggle by using the "[create dataset from remote URL](https://www.kaggle.com/product-feedback/75341#449911)" feature
# 
# Step 3: In the [settings menu](https://i.imgur.com/eZQkE6m.png) turn on automatic updates for your dataset
# 
# 

# Here I described how to make a dataset that updates automatically.  To see how to create a kernel that updates automatically, see the following link: https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1
