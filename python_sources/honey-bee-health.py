#!/usr/bin/env python
# coding: utf-8

# # Annoted Honey Bee
# 
# ![](https://media3.giphy.com/media/kc8PUzwL0rSqk/giphy.gif)
# 
# # Data descriptions
# 
# <table style="border-collapse: collapse; width: 322.5pt; margin-left: auto; border: none; margin-right: auto;">
# <tbody>
# <tr style="height: 14.3000pt;">
# <td style="width: 322.5000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; background: #fbd5b5; border: 1.5000pt solid #f79646;" colspan="2" width="430">
# <p style="text-align: center; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Column Descriptions</span></strong></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: 1.5000pt solid #f79646; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">File</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">File name in bee_imgs folder</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: 1.5000pt solid #f79646; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Date</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Date of video capture</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: 1.5000pt solid #f79646; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Time</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Time of day of video capture (military time)</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: 1.5000pt solid #f79646; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Location</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Location (city, state, country)</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: 1.5000pt solid #f79646; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Zip Code</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Zip Code to numerically describe loaction</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Subspecies</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Subspecies of Apis mellifera species</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Health</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Health description of bee</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Pollen_Carrying</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Presence of pollen on the bee's legs</span></p>
# </td>
# </tr>
# <tr style="height: 14.3000pt;">
# <td style="width: 82.1000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #fbd5b5;" width="109">
# <p style="text-align: left; vertical-align: middle;"><strong><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Caste</span></strong></p>
# </td>
# <td style="width: 240.4000pt; padding: 0.7500pt 0.7500pt 0.7500pt 0.7500pt; border-left: none; border-right: 1.5000pt solid #f79646; border-top: none; border-bottom: 1.5000pt solid #f79646; background: #ffffff;" width="320">
# <p style="text-align: left; vertical-align: middle;"><span style="font-family: Arial; color: #000000; font-size: 10.5000pt;">Worker, Drone, or Queen bee</span></p>
# </td>
# </tr>
# </tbody>
# </table>
# <p style="text-align: center;">&nbsp;</p>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from IPython.display import IFrame

# Any results you write to the current directory are saved as output.


# # Citywise Health Classification

# In[ ]:


IFrame("https://public.tableau.com/views/CitywiseHealthClassification/CitywiseHealthClassification?:embed=y&:showVizHome=no", width=1100, height=800)
# Iframe("https://public.tableau.com/views/CitywiseHealthClassification/CitywiseHealthClassification?:embed=y&:=yes&publish=yes")


# # City wise health Records

# In[ ]:


IFrame("https://public.tableau.com/shared/6Z8S8X5P4?:embed=y&:showVizHome=no", width=1100, height=800)
# Iframe("https://public.tableau.com/views/CitywiseHealthClassification/CitywiseHealthClassification?:embed=y&:=yes&publish=yes")


# ### Stay Tuned!!!

# In[ ]:




