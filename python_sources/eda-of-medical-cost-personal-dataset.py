#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from IPython.display import IFrame


# ## Read Data 

# In[ ]:


df=pd.read_csv("../input/insurance.csv")


# In[ ]:


df.shape


# ## Lets check missing Value
# 
# * seems good there is no missing value

# In[ ]:


df.isna().sum()


# ## Research Problem is to predict chrages that means Regression task
# 
# ### so this notebook figure out which feature/column is import for prediction based on exploration
# 
# #### What is BMI?
# 
# * Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

# ### BMI matters on Gender?
# 
# * BMI and Gender both are indepdent

# In[ ]:



IFrame("https://public.tableau.com/views/IsBMIdependongender/IsBMIDependonGender?:embed=y&:showVizHome=no", width=1200, height=500)


# ### Age and BMI are dependent ?
# 
# * Both are independent to each other so we can consider both feature for model

# In[ ]:


IFrame("https://public.tableau.com/views/IsAgeHaveImpactonBMI/IsAgeHaveImpactonBMI?:embed=y&:showVizHome=no", width=1200, height=500)
#https://public.tableau.com/views/IsAgeHaveImpactonBMI/IsAgeHaveImpactonBMI?:embed=y&:display_count=yes&publish=yes


# ### Age matter claimed charges ?
# 
# * yes with increasing age there is increase in claimed insurance charge

# In[ ]:


IFrame("https://public.tableau.com/views/Ageismajorfactorofcharges/Ageismajorfactorofcharges?:embed=y&:showVizHome=no", width=1200, height=500)
#https://public.tableau.com/views/Ageismajorfactorofcharges/Ageismajorfactorofcharges?:embed=y&:display_count=yes&publish=yes


# ### Smoke is Dangerous
# 
#  *  non smoker claimed charges are less compare to smoker claimed charges so we can conclude that smoking is key feature,
# and thoes who are smoker they claimed more insurance charges due to bed health

# In[ ]:


IFrame("https://public.tableau.com/views/Smokeisdangerous/Smokeisdangerous?:embed=y&:showVizHome=no", width=1200, height=700)
#https://public.tableau.com/views/Smokeisdangerous/Smokeisdangerous?:embed=y&:display_count=yes&publish=yes


# ### BMI is key parameter
# 
# * thoes who have greater than 25 BMI claimed more and hence more charges the normal range of BMI is 18.5 to 24.9

# In[ ]:


IFrame("https://public.tableau.com/views/BMIabove24notgood/BMIabove24notgood?:embed=y&:showVizHome=no", width=1200, height=700)
#https://public.tableau.com/views/BMIabove24notgood/BMIabove24notgood?:embed=y&:display_count=yes&publish=yes


# ### Major Feature Based On EDA for regression task
# 
# 1. BMI
# 2. Age
# 3. Smoke
# 
# above listed feature have direct impact on claimed charges
# 
# 

# ### Stay Connected For MOre EDA ...

# In[ ]:




