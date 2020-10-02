#!/usr/bin/env python
# coding: utf-8

# ![](https://healthitanalytics.com/images/site/article_headers/_normal/ThinkstockPhotos-495951912.jpg)
# # Introduction 
# 
# ## Importance of this dataset
# Breast cancer is one of the most prevalent cancer present in females. Although numerous imaging techniques have been put in place for the diagnostic and treatment of breast cancer, the advancement in genetic programming and machine learning over the past decades have improved the accuracy of breast cancer detection tremendously, more so than imaging techniques. To facilitate interpretation and analysis, enormous amount of mammography films have been processed by the system to help improve the visibility of peripheral areas and intensity distribution, and several methods have been reported to assist in this process. Feature extraction is one of the most important step in breast cancer detection due to its ability to differentiate between benign and malignant tumors. After extraction, image properties such as smoothness, coarseness, depth, and regularity are extracted by segmentation.
# 
# This dataset is ideal for the testing of machine learning model but since I have just started out (this is my second Kaggle), I will be merely performing an exploration data analysis. 
# 
# ## Objective
# - Find out the correlation between cancer radius, texture,parimeter, compactness, etc and whether the cancer is malignant or benign. 
# - Upon finding out their correlation, find the relationship between these variables and whether the cancer is malignant or benign.
# 

# First step is to import the required packages. Mainly Numpy, Pandas, Matplotlib and Seaborn. 

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns  


# Using pandas, we will import the dataset into our notebook.

# In[ ]:


bc_dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')


# We will then have a quick view of the dataset. 

# In[ ]:


bc_dataset.head()


# In[ ]:


bc_dataset.describe()


# For each column, what we can understand is:
# - Diagnostic is represented by M(malignant-bad) and B(Benign).
# - Types of data include 
#     * Radius, measured from center to point of parimeter (mean, standard error, worst[highest mean])
#     * Texture, measured by pixels making up the segmented area, is represented by grey-scale values (0 for black and 255 for white).
#     * Perimeter, which is the size of the core tumour 
#     * Area, area of the tumour 
#     * Smoothness, measured by variation in radius length
#     * Compactness, measured by the mean of [(square of parimeter)/area -1]
#     * Concavity, whether it concave upwards or concave downwards 
#     * Concave point (number of concave portions)
#     * Symmetry 
#     * Fractal dimension, which measures the complexity comparing how a detail in a pattern changes with the scale in which it is measured.
#     
# ### Alright, now let's see if there are any NULL, NaN or unknown values!

# In[ ]:


bc_dataset.isnull().sum()


# Great to see no null values here! Alright! Now, let's separate the data according to mean, standard error and worst.

# In[ ]:


# Separating the mean and looking into the data
list_mean=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',
      'smoothness_mean','compactness_mean','concavity_mean',
      'concave points_mean','symmetry_mean','fractal_dimension_mean']
mean_data=bc_dataset[list_mean]
mean_data.head()


# Here, we will use the PairPlot tool from Seaborn to see the distribution and relationships among variables.

# In[ ]:


# Implementing pairplot 
g = sns.pairplot(mean_data, hue = 'diagnosis')
g.map_diag(sns.distplot)
g.map_offdiag(plt.scatter)
g.add_legend()
g.fig.suptitle('FacetGrid plot (Mean)', fontsize = 20)
g.fig.subplots_adjust(top= 0.9);


# ## What do we see here?
# - Well, we can clearly see that malignant tumour seem to have longer range among all variables, except for fractal dimension.
# - Also, the peak for malignant tumour seems to appear more on the right as compared to benign tumours. 
#     * This means that malignant tumour tend to have larger radius, rougher texture, more compactness,concavity and concave points. 
#     
# ### *We will be using this dataset for majority of our analysis.*

# Here, we will do the same for the data comprising the standard of error

# In[ ]:


# Separating the mean and looking into the data
list_se=['diagnosis','radius_se','texture_se','perimeter_se','area_se',
      'smoothness_se','compactness_se','concavity_se',
      'concave points_se','symmetry_se','fractal_dimension_se']
se_data=bc_dataset[list_se]
se_data.head()


# In[ ]:


# Implementing pairplot 
gg = sns.pairplot(se_data, hue = 'diagnosis')
gg.map_diag(sns.distplot)
gg.map_offdiag(plt.scatter)
gg.add_legend()
gg.fig.suptitle('FacetGrid plot (Standard of error)', fontsize = 20)
gg.fig.subplots_adjust(top= 0.9);


# ## What do we see here?
# - Ideally, we are expecting similar standard of error when comparing datasets with malignant and benign tumours. 
#     * Yes, we do see both datasets having relatively similar values/ratios.
#     

# # Correlation
# - Here, we will want to find out the correlations among each variable.
#     * To do so we will need to plot a correlation matrix
#     * This method is normally used for feature selection. Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. When two features have high correlation, one is dropped. 
#         * This step is essential in reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.
# 
# 
# 

# In[ ]:


ax = sns.heatmap(mean_data.corr())


# ## Let's find the correlation between each variable and whether the tumour is malignant or benign based on that!

# 1. Correlation between mean of radius and malignant tumour

# In[ ]:


radius_diagnosis = ['diagnosis','radius_mean']
radius_corr =bc_dataset[radius_diagnosis]
radius_corr.radius_mean = radius_corr.radius_mean.round()
radius_m_corr = radius_corr[radius_corr['diagnosis'] == 'M'].groupby(['radius_mean']).size().reset_index(name = 'count')
radius_m_corr.corr()


# - Plotting the regression plot

# In[ ]:


sns.regplot(x = 'radius_mean', y = 'count', data = radius_m_corr).set_title("Mean radius vs Malignant count")


# ## Observation
# - For correlation coefficient, a positive 1 means a perfect positive correlation, which means both variables move in the same direction. If one goes up, the other will go up. In contrary, a negative 1 means the relationship that exists between two variables is negative 100% of the time.
#     * We do observe a relatively strong negative correlation between mean radius and malignant tumour counts. From the diagram, we do see huge counts from radius ranging between 15 to 20. Although we observed a negative correlation, we need to pay attention to the peak of the diagram. This diagram has a greater range for radius than the diagram showing benign tumour. 

# 2. Correlation between mean of radius and benign tumour

# In[ ]:


radius_b_corr = radius_corr[radius_corr['diagnosis'] == 'B'].groupby(['radius_mean']).size().reset_index(name = 'count')
radius_b_corr.corr()


# In[ ]:


sns.regplot(x = 'radius_mean', y = 'count', data = radius_b_corr).set_title("Mean radius vs Benign count")


# ## Observation
# - Significantly low correlation between radius and benign count.
# - A narrower range for radius as compared to the diagram showing malignant tumour.

# 3. Correlation between mean of texture and malignant tumour counts 

# In[ ]:


texture_diagnosis = ['diagnosis','texture_mean']
texture_corr =bc_dataset[texture_diagnosis]
texture_corr.texture_mean = texture_corr.texture_mean.round()
texture_m_corr = texture_corr[radius_corr['diagnosis'] == 'M'].groupby(['texture_mean']).size().reset_index(name = 'count')
texture_m_corr.corr()


# In[ ]:


sns.regplot(x = 'texture_mean', y = 'count', data = texture_m_corr).set_title("Mean texture vs Malignant count")


# ## Observation
# - We seem to see observation similar to mean radius. We clearly do not see a linear relationship in malignant count and mean texture. We see the peak in between 20-25.  

# 4. Correlation between texture and belign tumour 

# In[ ]:


texture_b_corr = texture_corr[texture_corr['diagnosis'] == 'B'].groupby(['texture_mean']).size().reset_index(name = 'count')
texture_b_corr.corr()


# In[ ]:


sns.regplot(x = 'texture_mean', y = 'count', data = texture_b_corr).set_title("Mean texture vs Benign count")


# ## Observation
# - We do see a narrower range of 15-20 as compared to 20-25 in the previous diagram. Similar pattern as mean radius. 

# 5. Correlation between parimeter and malignant tumour count

# In[ ]:


perimeter_diagnosis = ['diagnosis','perimeter_mean']
perimeter_corr =bc_dataset[perimeter_diagnosis]
perimeter_corr.perimeter_mean = perimeter_corr.perimeter_mean.round()
perimeter_m_corr = perimeter_corr[perimeter_corr['diagnosis'] == 'M'].groupby(['perimeter_mean']).size().reset_index(name = 'count')
perimeter_m_corr.corr()


# In[ ]:


sns.regplot(x = 'perimeter_mean', y = 'count', data = perimeter_m_corr).set_title("Mean perimeter vs Malignant count")


# In[ ]:


perimeter_b_corr = perimeter_corr[perimeter_corr['diagnosis'] == 'B'].groupby(['perimeter_mean']).size().reset_index(name = 'count')
perimeter_b_corr.corr()


# In[ ]:


sns.regplot(x = 'perimeter_mean', y = 'count', data = perimeter_b_corr).set_title("Mean perimeter vs Benign count")


# ## Observation
# - As usual, we see the peak ranging around 100-140 for malignant tumours and 70-90 for benign tumours. 
# 
# ## So why do we see similar patterns in mean area, mean texture and mean perimeter?
# - Well, if you look in the heatmap you will see two main clusters that are highly correlated. The first cluster involves area, texture, perimeter and radius, hence, we see that similar pattern.
# 
# ### Let's look into the second cluster comprising of compactness, concavity and concave points.

# In[ ]:


compactness_diagnosis = ['diagnosis','compactness_mean']
compactness_corr =bc_dataset[compactness_diagnosis]
compactness_corr.compactness_mean = compactness_corr.compactness_mean.round(2) # Round off to 2 decimal places
compactness_m_corr = compactness_corr[compactness_corr['diagnosis'] == 'M'].groupby(['compactness_mean']).size().reset_index(name = 'count')
compactness_m_corr.corr()


# In[ ]:


sns.regplot(x = 'compactness_mean', y = 'count', data = compactness_m_corr).set_title("Mean compactness vs Malignant count")


# In[ ]:


compactness_b_corr = compactness_corr[compactness_corr['diagnosis'] == 'B'].groupby(['compactness_mean']).size().reset_index(name = 'count')
compactness_b_corr.corr()


# In[ ]:


sns.regplot(x = 'compactness_mean', y = 'count', data = compactness_b_corr).set_title("Mean compactness vs Benign count")


# In[ ]:


concavity_diagnosis = ['diagnosis','concavity_mean']
concavity_corr =bc_dataset[concavity_diagnosis]
concavity_corr.concavity_mean = concavity_corr.concavity_mean.round(2) # Round off to 2 decimal places
concavity_m_corr = concavity_corr[concavity_corr['diagnosis'] == 'M'].groupby(['concavity_mean']).size().reset_index(name = 'count')
concavity_m_corr.corr()


# In[ ]:


sns.regplot(x = 'concavity_mean', y = 'count', data = concavity_m_corr).set_title("Mean concavity vs Malignant count")


# In[ ]:


concavity_b_corr = concavity_corr[concavity_corr['diagnosis'] == 'B'].groupby(['concavity_mean']).size().reset_index(name = 'count')
concavity_b_corr.corr()


# In[ ]:


sns.regplot(x = 'concavity_mean', y = 'count', data = concavity_b_corr).set_title("Mean concavity vs Benign count")


# ### We did see exact pattern as the first cluster.

# # Summary
# - Although we do not see concrete correlation between the variables and the type of tumour counts, we are able to see similar patterns between the clusters that are highly correlated on the correlation matrix. 
# - What we see is malignant tumours tend to have higher radius, texture, area, perimeter, concavity, compactness and concave points. 
