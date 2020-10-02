#!/usr/bin/env python
# coding: utf-8

# This notebook has been created to demonstrate an issue as mentioned here: 
# https://kaggle.com/product-feedback/154230

# I have displayed the raw text above each link for convenience.
# 
# `https://www.kaggle.com/discussion`   
# 
# https://www.kaggle.com/discussion
# 
# `<a href="https://www.kaggle.com/discussion" target="_blank">https://www.kaggle.com/discussion</a>`   
# 
# <a href="https://www.kaggle.com/discussion" target="_blank">https://www.kaggle.com/discussion</a>
# 
# 
# While editing the notebook, clicking on either link opens a new tab. After committing and viewing the notebook in the viewer, clicking on either link opens the page in the same tab. After inspecting the source code, I find that both links appear as follows:
# 
# `<a href="https://www.kaggle.com/discussion" target="_top">https://www.kaggle.com/discussion</a>`   
# 
# <a href="https://www.kaggle.com/discussion" target="_top">https://www.kaggle.com/discussion</a>

# ## Update (13 June 2020) 
# I noticed that this issue only affects URLs on the Kaggle website. For example, if we try the same "experiment" with a different URL (say, google.com), the target attribute works as intended:

# `https://www.google.com`   
# 
# https://www.google.com
# 
# `<a href="https://www.google.com" target="_blank">https://www.google.com</a>`   
# 
# <a href="https://www.google.com" target="_blank">https://www.google.com</a>
# 
# 
# `<a href="https://www.google.com" target="_top">https://www.google.com</a>`   
# 
# <a href="https://www.google.com" target="_top">https://www.google.com</a>

# ## Update (16 June 2020) 
# As per the response by Jim Plotts, dropping the `www` from the URL will fix the problem:

# `https://kaggle.com/discussion`   
# 
# https://kaggle.com/discussion
# 
# `<a href="https://kaggle.com/discussion" target="_blank">https://kaggle.com/discussion</a>`   
# 
# <a href="https://kaggle.com/discussion" target="_blank">https://kaggle.com/discussion</a>
# 
# `<a href="https://kaggle.com/discussion" target="_top">https://kaggle.com/discussion</a>`   
# 
# <a href="https://kaggle.com/discussion" target="_top">https://kaggle.com/discussion</a>
