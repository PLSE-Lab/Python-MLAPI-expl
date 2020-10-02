#!/usr/bin/env python
# coding: utf-8

# 
# 
#                                            Breast Cancer- Data Visualization 
#  

# In[ ]:





# In[ ]:


#importing the packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap
import seaborn as sns
print("Setup Complete")


# In[ ]:


#loading the dataset and adding the column header
cancer_data = pd.read_csv('../input/breast_cancer_prognostic.csv', names =["id","diagnosis","time","radius_mean","texture_mean",
                                                    "perimeter_mean","area_mean","smoothness_mean","compactness_mean",
                                                    "concavity_mean","concavityPoints_mean","symmetry_mean","fractal_mean",
                                                    "radius_standard_error","texture_standard_error","perimeter_standard_error",
                                                    "area_standard_error","smoothness_standard_error","compactness_standard_error",
                                                    "concavity_standard_error","concavityPoints_standard_error","symmetry_standard_error",
                                                    "fractal_standard_error","radius_worst","texture_worst","perimeter_worst","area_worst",
                                                    "smoothness_worst","compactness_worst","concavity_worst","concavityPoints_worst","symmetry_worst",
                                                    "fractal_worst","tumor_size","lymph_node"])


# In[ ]:


cancer_data.head()


# In[ ]:


print(cancer_data.info())


# a) Pairplot for selected variable from breast cancer prognostic dataset

# In[ ]:


#breast cancer diagnosis size
print(cancer_data.groupby('diagnosis').size())


# In[ ]:


g = sns.pairplot(cancer_data, vars=["time","radius_mean","texture_mean",
                                                    "perimeter_mean","area_mean","smoothness_mean","compactness_mean",
                                                    "concavity_mean","concavityPoints_mean","symmetry_mean","fractal_mean",
                                            "tumor_size"], hue ="diagnosis",palette=sns.color_palette(['#DC143C','#DAA520']))


# b) Boxplot for all the variables and  mean-value of real value features.

# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['time'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['radius_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['texture_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['perimeter_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['area_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['smoothness_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['compactness_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['concavity_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['concavityPoints_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['symmetry_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['fractal_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))


# In[ ]:


sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['tumor_size'],palette=sns.color_palette(['#DC143C','#DAA520']))


# c) Scatter plot for selected pair of variables with the:
# 
# * Strongly correlated:
# radius_mean     perimeter_mean    0.995933
# area_mean       radius_mean       0.992855
# 
# * Moderately correlated:
# fractal_dimension_worst  smoothness_worst       0.617624
# area_se                  concavity_mean         0.617427
# 
# * Weakly correlated:
# area_mean               smoothness_se             0.166777
# radius_se               smoothness_se             0.164514
# 

# In[ ]:


sns.scatterplot(x=cancer_data["radius_mean"],y=cancer_data["perimeter_mean"], palette=sns.color_palette(['#DC143C','#FFD700']),hue=cancer_data["diagnosis"])


# In[ ]:


sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["radius_mean"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])


# In[ ]:


sns.scatterplot(x=cancer_data["fractal_worst"],y=cancer_data["smoothness_worst"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])


# In[ ]:


sns.scatterplot(x=cancer_data["area_standard_error"],y=cancer_data["concavity_mean"], palette=sns.color_palette(['#DC143C','#FFD700']),hue=cancer_data["diagnosis"])


# In[ ]:


sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_standard_error"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])


# In[ ]:


sns.scatterplot(x=cancer_data["radius_standard_error"],y=cancer_data["smoothness_standard_error"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])


# d) Histograms for all the variables and standard error of real-valued features

# In[ ]:


sns.distplot(a=cancer_data['time'],color='DarkRed',vertical=False)
#variable time is slightly skewed and removing few outliers will help achiever the normal distribution
            


# In[ ]:


sns.distplot(a=cancer_data['radius_standard_error'],color='DarkRed',vertical=False)

#Left skewness observerd for radius standard error to left


# In[ ]:


sns.distplot(a=cancer_data['texture_standard_error'],color='DarkRed',vertical=False)
#texture standard error is slightly skewed to the left


# In[ ]:


sns.distplot(a=cancer_data['perimeter_standard_error'],color='DarkRed',vertical=False)
#perimeter standard error is slightly skewed to the left


# In[ ]:


sns.distplot(a=cancer_data['area_standard_error'],color='DarkRed',vertical=False)
#area standard error is slightly skewed  to the left


# In[ ]:


sns.distplot(a=cancer_data['smoothness_standard_error'],color='DarkRed',vertical=False)
#smoothness standard error is slightly skewed 


# In[ ]:


sns.distplot(a=cancer_data['compactness_standard_error'],color='DarkRed',vertical=False)
#compactness standard error is slightly skewed 


# In[ ]:


sns.distplot(a=cancer_data['concavity_standard_error'],color='DarkRed',vertical=False)
#concavity standard error is slightly skewed 


# In[ ]:


sns.distplot(a=cancer_data['concavityPoints_standard_error'],color='DarkRed',vertical=False)
#concavity standard error is normal distributed 


# In[ ]:


sns.distplot(a=cancer_data['symmetry_standard_error'],color='DarkRed',vertical=False)
#symmetry standard error is slightly skewed 


# In[ ]:


sns.distplot(a=cancer_data['fractal_standard_error'],color='DarkRed',vertical=False)
#fractal standard error is slightly skewed 


# Observation:
# For the purpose of visualization I decided not to focus much on feature importance. Instead I limited my analysis to focus on the each visualization speaks about the variable. Here are my thoughts:
# * Pairplot: 
# For pairplot visualization all the variables are selected. The observation shows few variables like smoothness worst, compactness worst, and concavity standard error doesn't have any impact of whether the tumor will be reoccuring or not. All the data points forms a a horizontal line that means there is no relationship between diagnosis and all three of these variables and can be removed.
# 
# * Boxplot:
# The boxplot is created for mean value of each -valued featutre and size of tumor. The reoccurance is higher for increased value of area mean, perimeter mean, concavity mean, radius mean , convaity point mean. Above mentioned variables can proved to be significant indicators when considering to build a prediction model.
# 
# * Scatterplot:
# Scatterplots presented above are based on two strong , two moderately and two weekly correlated features. The stronger the varibles are the linear is the relationshio. This also depicts the fact, the higher the value of correlated features, more are the chances of reoccurance of breast cancer tumor. 
# 
# * Histograms: 
# For histrograms standard error of each valued-feature is selected. I observed all the values are skewed to the left. In this dataset more than half of the data points are non-recurring. The left skewness is result of non-recurring tumor data points clustering around smaller degrees of each variable. For example, if the area standard error is smaller the chances of reoccurance of the tumor is fewer. Adding more datapoints can add value to this observation and confirm the positive linearity of the relationship.

# Reference Links:
#     
# >https://www.engineeringbigdata.com/breast-cancer-dataset-analysis-visualization-and-machine-learning-in-python/
# >https://www.cs.virginia.edu/~cs1112/term/181/resources/modules/pillow/colors/
# >https://towardsdatascience.com/machine-learning-for-breast-cancer-classification-and-bio-marker-identification-d92bdc1e0d1e
# 
# Further model building reference
# >https://machinelearningmastery.com/feature-selection-with-categorical-data/

# In[ ]:




