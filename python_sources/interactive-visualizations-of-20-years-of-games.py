#!/usr/bin/env python
# coding: utf-8

# #Interacting with Data
# An interactive visualizations of 20 years of Games using **Angular.js(Ionic Framework)** , **Python Flask** , **plot.ly** & **C3.js**.
# ## CODE
# You can find the code for both the Ionic & Python at my github repository [yaswanthsvist](https://github.com/yaswanthsvist/ign-kaggle)
# 
# ##Setup
# You need to [create](https://ionicframework.com/getting-started/) a ionic app and replace the www dirctory with ign-kaggle/UI/www directory.
# 
# 
# I am skipping the Python setup.
# 
# run pd_sample.py for apis.
# 
# ##Visualizing Data
# I have used C3.js for visualizing the pie and bar charts.
# Plot.ly for visualizing the Heatmap.
# I will try NVD3.js for other kaggle datasets. 
# 
# ##Interacting with Platforms
# Below is a sample example of interacting with platforms pie chart and showing insights of **PC** & **Play station2**  platforms.
# 
# ![superset-explore-slice](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/platformOptimized.gif)
# 
# From above we can say **PC** Platform has the highest share among all players in the market.
# **Strategy** games are the dominating genre in PC Platform.
# 
# ##Birth of Platforms:
# I have made a  platforms vs Birth year("the year in which the first game released in the platform") chart using **Plot.ly** Heatmap.
# Using this heatmap one can zoom perticular area of heat map.
# below show's a heatmap with 59 platforms vs birth year.
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/birth.gif)
# 
# ##End of Platforms:
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/endOfPlatforms.gif)
# 
# End(the year  in which the first game released in the platform) of platforms.
# 
# ##Releses/Year(1970 - Present):
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/releasesPerYear.gif)
# 
# ##Releases/Month(Jan-Dec):
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/releasesPerMonth.gif)
# 
# ##Releases/Day(1-13):
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/releasesPerDay.gif)
# 
# ##Releases/Date(dd-mm-yyyy):
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/releasesPerDate.gif)
# 
# ##Genres Pie Chart:
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/genersPie.PNG)
# 
# ##Genres Bar Chart:
# 
# ![Birth of Games](https://raw.githubusercontent.com/yaswanthsvist/ign-kaggle/master/kernel/gifs/GenersBar1.gif)
# 
# 
