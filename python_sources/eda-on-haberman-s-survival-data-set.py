#!/usr/bin/env python
# coding: utf-8

# ### Haberman's Survival Data Set
# #### Survival of patients who had undergone surgery for breast cancer
# 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# #### Attribute Information:
# 
# **Feature-1** : Age of patient at time of operation 
# 
# **Feature-2** : Patient's year of operation (year - 1900)
# 
# **Feature-3** : Number of positive axillary nodes detected 
# 
# **Feature-4** : Survival status
# 
#                 (class attribute) 1 = the patient survived 5 years or longer 
#                 
#                 (class attribute) 2 = the patient died within 5 year
#                 
# As Feature-4 is a class variable , it is the result of all other three features , 
# it can also be thought of as a Dependent feature and Feature-1,2,3 are independent features 

# **OBJECTIVE** : If a new observation is to be tested then to classify whether it should be classified as class 1 or class 2
#                 based on it's features

# In[ ]:


#import the libraries needed for visualization and numeric computations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#To display the plots on to the jupyter notebook without calling the plt.show() function
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#import the dataset and understand what are the features present in it
habermans_dataset = pd.read_csv("../input/haberman.csv",names= ['age','year_of_operation','axillary_nodes','survival_class'] )


# In[ ]:


habermans_dataset.head()


# In[ ]:


#To know the number of observations and number of columns present in the dataset we use shape function
habermans_dataset.shape


# This dataset has 306 observations and 4features (1 among them is class feature )

# In[ ]:


#let's examine the survival class to examine whether the data is balanced or imbalanced 
habermans_dataset['survival_class'].value_counts()


# By above output we can understand that it is not a balanced dataset

# 
# Initially examing the bivariate analysis and from that understanding which feature is significant among those multiple features then performing univariate anlysis on it

# ### Pair Plots
# 
# To analyze the plots and find out which features are best fit to consider for future predictions

# In[ ]:



sns.set_style("whitegrid") #to display darkgrid as a background (can also be whitegrid , white , grid)


# We will be getting the output of the pairplots including class label if we run below cell. To exclude the class label in pairplot it should be a string. So,changing the categorical data into strings with meaningful names

# In[ ]:


#sns.pairplot(habermans_dataset,hue='survival_class',kind='scatter',size=3) 


# In[ ]:


habermans_dataset['survival_class'] = habermans_dataset['survival_class'].apply(lambda x: 'survived >= 5yrs' if x == 1 else 'survived < 5yrs')


# In[ ]:


habermans_dataset['survival_class'].value_counts()


# In[ ]:


sns.pairplot(habermans_dataset,hue='survival_class',kind='scatter',size=3)


# #### Observation

# From the above scatter plots it is evident that 
# 
# --> We can't classify whether the given observation is of class-1 or class-2 based on any two features from the dataset
# 
#    because in every scatter plot(out of three plots) the datapoints are not seperated in any manner. 

# -----------------------------------------------------------------------------------------------------------------------------

# ### Univariate Analysis 
#                          
#  Performing analysis on each individual feature to identiy which feature might the suitable for our objective
# 
# 
# 

# In[ ]:


#dividing the survival_class features into each individual dataframes to perform CDF,PDF


# In[ ]:


habermans_dataset_class_1 = habermans_dataset[habermans_dataset['survival_class'] == 'survived >= 5yrs']


# In[ ]:


habermans_dataset_class_1.head()


# In[ ]:


habermans_dataset_class_2 = habermans_dataset[habermans_dataset['survival_class'] == 'survived < 5yrs']


# In[ ]:


habermans_dataset_class_2.head()


# #### Plotting Histograms with PDF 
# to know the data distribution
# 

# We will plot all the features and thier Histograms and from the graphs we will select which feature can be useful among three and then perform analysis on it
# 
# **Univariate Analysis using PDF (Probability Density Function)**

# In[ ]:


#initially plotting the first feature (age)

sns.FacetGrid(habermans_dataset,hue='survival_class',size=5).map(sns.distplot,'age').add_legend()
plt.title("Histogram with PDF for 'age' feature")


# In[ ]:


sns.FacetGrid(habermans_dataset,hue='survival_class',size=5).map(sns.distplot,'year_of_operation').add_legend()
plt.title("Histogram with PDF for 'year_of_operation' feature ")


# In[ ]:


sns.FacetGrid(habermans_dataset,hue='survival_class',size=5).map(sns.distplot,'axillary_nodes').add_legend()
plt.title("Histogram with PDF for 'axillary_nodes' feature")


# **Observation**:
#     
# 1.In the histogram plots pf features 'year_of_operation' and 'age' , PDF of two plots show that , two classes ('survived>=5yrs' and 'survived<5yrs') overlapped (almost they are top on each other). So, we cannot consider both features as useful feature for our objective.
# 
# 
# 2.Whereas in the histogram plot for feature 'axillary_nodes' . Even though there is an slight overlap between two PDF of classes (survived>=5yrs and survived<5yrs) but there is possibility of classifying results based on this feature when compared to other two features. 
# 
# For indetailed analysis , we will plot CDF,boxplots and violin plots for feature 'axillary_nodes'

# ### Univariate Analysis on 'axillary_nodes' feature

# #### Plotting CDF(Cumulative Distribution Function) for this feature . 
# 
# It is the cumsum of the PDFs'. It helps in determing the percentage of class is below or above at a given point.
# 
# 

# In[ ]:


count,bin_edges = np.histogram(habermans_dataset_class_1['axillary_nodes'],density=True,bins=10)
PDF = count/sum(count)
print(PDF)
print(bin_edges)


# In[ ]:


#plt.plot(bin_edges[1:],PDF)

#computing CDF
CDF = np.cumsum(PDF)
print(CDF)


# In[ ]:


plt.plot(bin_edges[1:],PDF,label = "PDF",)
plt.plot(bin_edges[1:],CDF,label = "CDF")
plt.legend()
plt.xlabel("axillary_nodes")
plt.ylabel("probability")
plt.title("PDF and CDF of 'axillary_nodes' feature for survival_class = 'survived>=5yrs' ")


# **Observation:**
#     
#   1. This is the plot of class-1(survived>=5yrs)
#     
#   2. From the given data we know that among 306 people,225 people have 'survived>=5yrs'.
#     
#   3. From the CDF in the graph we can say that approximately "90%" of the people who "survived greater than 5yrs" has "count of axillary nodes <=10"  
#     
#     i.e., approximately 202 people who has survived>=5yrs have 'count of axillary nodes less than or equal to 10'.
#     
# 

# In[ ]:


#### plot for two survival_classes

#plot for 'survived>=5yrs'
count,bin_edges = np.histogram(habermans_dataset_class_1['axillary_nodes'],bins=20,density=True)
PDF = count/sum(count)
#calculating CDF
CDF = np.cumsum(PDF)
plt.plot(bin_edges[1:],PDF,label="PDF of survived>=5yrs")
plt.plot(bin_edges[1:],CDF,label="CDF of survived>=5yrs")

#plot for 'survived<5yrs'
count,bin_edges = np.histogram(habermans_dataset_class_2['axillary_nodes'],bins=20,density=True)
PDF = count/sum(count)
#calculating CDF
CDF = np.cumsum(PDF)
plt.plot(bin_edges[1:],PDF,label="PDF of survived<5yrs")
plt.plot(bin_edges[1:],CDF,label="CDF of survived<5yrs")
plt.legend()
plt.xlabel("axillary_nodes")
plt.ylabel("probability")
plt.title("PDF and CDF of both survival_classes")


# **Observation:**
#     
# 1.**Our observation in above case has been ruled out ** as the CDF of "survived<5yrs" class show 70% of it's observations are for the patients who has (count of axillary_nodes less than equal to 10) . As we have stated our observation only by depending upon one class (survived>=5yrs)among two classes in the above observation.
#     
# 2.In this plot, we can observe that there is an overlap between two PDFs'. 
#     
# 3.Even though there is overlapping between both. when we fix our threshold value to '5'(count of no.of axillary nodes to 5), then if we need to classify the new patient according to this feature then we might predict him to class who can 'survive 5yrs or more than 5yrs' because from the CDF of 'survived<=5yrs' show that 80% of it's observation have 'count of axillary nodes' <= 5 whereas 'survived>5yrs' class has only 45% of observations at that value.
# 
# 

# ### Box Plot and Whiskers
# Let's analyze this feature by using box-plot

# In[ ]:


sns.boxplot(data = habermans_dataset,x="survival_class",y="axillary_nodes")


# **Observation:**
#    
# 1.From this plot, it is evident that the class 'survived>=5yrs' has 75percentile of count of axillary nodes value less than 5. probably less than 4.
# 
# 2.The class 'survived<5yrs' has it's 50th percentile value more than the value of 75th percentile of 'survived>=5yrs' class.From the box plot of this class , it can be seen that most of it's values(count of axillary nodes) are '>6 and <12' (between 50th percentile and 75th percentile) of this class 

# ### Violin Plot

# In[ ]:


sns.violinplot(data = habermans_dataset,x='survival_class',y='axillary_nodes')


# ## Conclusion:
# 

# 1.Out of three features ('age','no.of.operations' and 'count of axillary_nodes') . 'axillary_nodes' is the best feature among three to classify the survival_class

# ----------------------------------------------------------------------------------------------------------------------------
# class-1 : 'survived>=5yrs'
# 
# class-2 : 'survived<5yrs'
# 
# ---------------------------------------------------------------------------------------------------------------------------

# 2.Even though 'axillary_nodes' feature has some overlappings among survival_class. but based on above observations we can classify that if the 'count of axillary nodes' is less than 5 , then it can be classified as 'survived>=5yrs' else it is more likely to fall under 'survived<5yrs' class.

# In[ ]:




