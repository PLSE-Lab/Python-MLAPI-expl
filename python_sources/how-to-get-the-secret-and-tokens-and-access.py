#!/usr/bin/env python
# coding: utf-8

# # **The flow to get the secret and token working is clear as mud
# 
# Always bad when the directions are incomplete - after an afternoon of clicking I think we have nailed down all the steps to get access.  To avoid sharing my personal information there will be some error looking cells right now in the kernel - they should all go away when you fork and step thru the script.
# 
# DO NOT COMMIT this kernel to get things working - step thru it - the intent is to teach you how to get your token and generate the secret.
# 
# I have taken the discussion post by Eric to generate this kernel following the steps described by Paul.
# 
# [http://https://www.kaggle.com/c/ds4g-environmental-insights-explorer/discussion/129997]
# 
# When it works and you have your refresh- token than you can commit to save the kernel for reference later.
# 

# Update:  Added section at the end to shown the steps to get Earth Engine running on your local Ubuntu machine.

# Step Zero - sign up for access to Google Earth Engine - COPY the link below into a new browser tab.  In top right there should be a spot to sign up.
# 
# Use the same gmail login for this as you will use below.  You should quickly get a email "Welcome to Google Earth Engine
# 
# https://earthengine.google.com/

# Step 1: Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rasterio as rio
import folium

import ee
from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials


# Step 2: Get a token
# 
# The code step below will generate a message with a link.
# 
# COPY the link and paste it into a new browser tab (YES - leave this kernel and start a new tab) and enter to run the link.
# 
# Once the link is pasted into the new page it should walk you thru a couple of dialog pages gathering your google user name and getting your permissions (ALLOW).
# 
# At the end of those steps a "token" is generated.  Copy that token.   Come back to this kernel.
# 
# PASTE the token in the box that was generated when you ran the next cell. 
# 
# You should get a message of success.

# In[ ]:


ee.Authenticate() 


# The code generated in the above steps is not the final token - run the next cell.  This will generate the value for the secret.

# In[ ]:


get_ipython().system('cat ~/.config/earthengine/credentials')


# NOW THE SECRET CODE STUFF NEEDS TO TAKE PLACE
# 
# Click on Add-ons
# 
# Click on Secrets
# 
# Add a new secret.  Pick anything you want for the label - this label will be visible in all the kernels you use so don't get cheeky.
# 
# In the VALUE box paste your refresh_token from above.  (only the value within the quotation marks)
# 
# > Copy the code snippet into the empty cell below.  Than run this next cell.
# 
#  

# FOR KERNELS ONCE YOU HAVE COMPLETED THIS FULL SCRIPT WITH SUCCESS - ANY FUTURE
# KERNELS ONLY NEED THE FOLLOWING STEPS AND SHOULD NOT NEED TO REPEAT ANY OF THE ABOVE CELLS.

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("jimmy")


# In[ ]:


from google.oauth2.credentials import Credentials
import ee
import folium


# In[ ]:


user_secret = "jimmy" # Your user secret, defined in the add-on menu of the notebook editor
refresh_token = UserSecretsClient().get_secret(user_secret)
refresh_token
credentials = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=ee.oauth.TOKEN_URI,
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=ee.oauth.SCOPES)



# In[ ]:


ee.Initialize(credentials=credentials)


# Access to Google Earth Engine on a Ubuntu 19.10 local PC.
# 
# Steps are very similiar but some changes.
# 1.  You still need to be signed up for access per step 0 above.
# 

# 2.  Pip to install the need libaries used in the example code in the github
# 
# 

# In[ ]:


# pip install rasterio
# pip install folium
# pip install google-auth >=1.11.0
# pip install earthengine-api


# On a one time basis you need to authenticate the local machine - much like the above it will step you through a couple of dialogs to get your gmail login info and for you to allow it access.
# 
# From terminal - run

# In[ ]:


earthengine authenticate


# 

# In[ ]:


# Import, authenticate and initialize the Earth Engine library.
import ee
# ee.Authenticate()   # only need this the first time its run on a PC
ee.Initialize()

