#!/usr/bin/env python
# coding: utf-8

# # Workflow for creating a data science portfolio
# ### By: Jeff Hale

# If you've been looking to create a data science portfolio that you can showcase your skills and use to help others learn the first question you might have asked yourself is where should your portfolio live. You might consider the following locations:
# 
# * Kaggle
# * GitHub
# * Your own Wordpress Blog
# * Your own custom Django Blog
# * Kyso - a newer website for data science exploration and sharing
# 
# When making your decision you might consider the following:
# 
# * Where is it fastest to produce new content?
# * Where is there a community of data scientists I can learn from? 
# * On the flip side where can others learn from me?
# * How can I easily share my portfolio on social media?
# 
# Here's a workflow to use so that you can have the best of lots of worlds:
# 
# 1. Produce a Jupyter Notebook in a Kaggle kernel. Share it on Kaggle. I love Kaggle because it's easy to use out of the box, with lots of convenient python packages pre-installed and easy access to data sources. It also has a large, active community of data scientists and a great format for sharing your work.
# 2. Next copy the Jupyter Notebook into GitHub by downloading it, creating a GitHub repo, and uploading it. GitHub has more mindshare than any other developer site and you will be working with developers. Many companies use GitHub so an active account can helps to build some credibility.
# 3. Import the Jupyter Notebook from GitHub into [Kyso](https://docs.kyso.io/docs/github-to-kyso). Kyso can host your portfolio. It provides iframes for embedding your code on another blog - with options to show or hide code and output from the Jupyter Notebook. 
# 4. Spread the word on Twitter, Facebook, LinkedIn, Reddit, or Hacker News from Kyso with their share buttons.
# 
# This might seem like overkill, but it's a pretty straightforward way to get your work onto a bunch of different platforms.
# 
# I'm looking for a quick way to have the content flow into a Medium post, too. Iframe embeds and url links within Medium aren't working nicely. You can use Gists, but I don't really want the steps of converting a GitHub file to a Gist every time. If you have a good suggestion for getting notebooks into Medium, please share in the comments :)
# 
# Disclaimer: I've tested out this workflow and just started using it. It might prove too cumbersome, but right now I like it. 
# Kyso seems to require a new project to be created if you update the notebook and then want to bring the updated notebook from GitHub to Kyso. You can delete the old notebook
# 
# Hope you find this helpful.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

