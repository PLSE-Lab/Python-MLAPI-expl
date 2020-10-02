#!/usr/bin/env python
# coding: utf-8

# This notebook contains a summary of the different plots from [Kaggle's Data Visualization Course](https://www.kaggle.com/learn/data-visualization). 
# 
# First of all, we load the different necessary modules.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# Next, we load the dataset. We have added the Breast Cancer Wisconsin dataset from UCI.

# In[ ]:


path_cancer_data = "../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv"
cancer_data = pd.read_csv(path_cancer_data)


# We can print the different features of each observation.

# In[ ]:


print(cancer_data.keys())


# # Line Charts
# 

# In[ ]:


plt.figure(figsize=(6,4))
sns.lineplot(x=cancer_data["radius_mean"],y=cancer_data["perimeter_mean"], hue=cancer_data["diagnosis"])


# We can see a correlation between the radius and the diasnosis. The perimeter is a function of the radius, and therefore it will be correlated to the diagnosis. When the perimeter exceeds a threshold, the diagnosis changes from bening to malignant.

# # Bar Charts and Heat Maps

# In[ ]:


sns.barplot(x=cancer_data['diagnosis'], y=cancer_data['area_mean'])


# In[ ]:


sns.barplot(x=cancer_data[:10].index, y=cancer_data['area_mean'][:10], hue=cancer_data['diagnosis'][:10])


# We can also plot a heatmap (some specific features were choosen for the purpose of the visualization)

# In[ ]:


labels = ['smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
heat_cancer_data = cancer_data[labels]
sns.heatmap(data=heat_cancer_data[:10], annot=True)


# # Scatter Plot

# In[ ]:


plt.figure(figsize=(6,4))
sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_mean"], hue=cancer_data["diagnosis"])


# The scatterplot shows the mean area vs the smoothness. We can see whether the area and the smoothness are correlated with the diagnosis or not by performing a linear regression.

# In[ ]:


sns.regplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_mean"]) 


# In[ ]:


sns.lmplot(x='area_mean', y='smoothness_mean', hue='diagnosis' , data=cancer_data)


# When the area is large enough, we would expect to find a malignant tumor. Apparently, the smoothness does not provide enough information. We can check it by doing a swarm plot.

# In[ ]:


sns.swarmplot(x=cancer_data['diagnosis'], y=cancer_data['smoothness_mean'])


# The malignant tumors have a slight higher smoothness compared to the benign tumors.

# # Distributions
# We can plot the histogram for the *perimeter_mean*.

# In[ ]:


sns.distplot(a=cancer_data['perimeter_mean'], kde=False)


# We can use a KDE to smooth the histogram.

# In[ ]:


sns.kdeplot(data=cancer_data['perimeter_mean'], shade=True)


# Also, we can use a joint plot to observe multiple distributions.

# In[ ]:


sns.jointplot(x=cancer_data['perimeter_mean'], y=cancer_data['smoothness_mean'], kind="kde")


# Finally, we can also change the different styles.

# In[ ]:


sns.set_style("dark")


# # References
# 1. [Kaggle's Data Visualization Course](https://www.kaggle.com/learn/data-visualization)
# 2. [Seaborn: Statistical Data Visualization](https://seaborn.pydata.org)
