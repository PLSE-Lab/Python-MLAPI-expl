#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# ## Introduction
# - What is EDA?
#     - It is about getting familiar with data. Before applying any AI or ML technique on the data set, it is extremely important to observe the basic properties of the data and therefore EDA is performed.
#     - In simple way: Before solving a question, we have to read it and observe the things/facts given in the question. Once it is observed, we can go ahead to solve it. EDA is performed for similar purpose.
#     
# - What do we do in EDA?
#     - We apply maths, statistics, plotting techniques and various scientific methods and observe the meaning of the results produced by those various techniques.
# 

# 
# ## Dataset: Haberman Survival
# - Study was conducted between 1958-1970 at the University of Chicago's Billings Hospital on the survival of patients, undergone surgery for breast cancer.
# 
#    1. Number of Instances - 306
#    2. Number of attributes - 4 (including class attribute)
#    3. Attribute Information:    
#       (i) '30' - Patient's age at the time of operation. Numerical value.
#       
#       (ii) '64' - Year of operations (19xx). Numerical value.
#       
#       (iii) '1' - Number of positive axillary nodes detected. Numerical value.
#       
#       (iv) '1.1' - Survival status: 1-Survived 5 years or more. 2-Survived less than 5 years. 
#       
# ## Objective
#    - Explore the data and analyze the basic facts.
#    - Gather higher level information of data.
# 

# # Loading the dataset

# In[ ]:


# Importing the libraries.
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[ ]:


# Loading data
hm=pd.read_csv("../input/haberman.csv")

# Identifying shape : Number of data points(rows), features(columns)
display(hm.shape)


# In[ ]:


# Identify the features name
print(hm.columns)


# In[ ]:


# Renaming the columns as provided infomation
hm = hm.rename(columns = {'30':'age', '64':'operation_year','1':'positive_axillary_nodes','1.1':'survival_status'})


# In[ ]:


# Displaying first 10 indexex
display(hm.head(10))


# In[ ]:


# Getting the number of datapoints(class label) in each class. Class is explicitly informed.
# Here the class is survival_status and class labels are: '1', '2'
display(hm['survival_status'].value_counts())


#  -  224  represent the number of survivals of 5 years or more.
#  - 81 represent the number of survivals of less than 5 years.
#  - It is an imbalanced dataset. Imbalanced dataset is one in which number of class labels differ with significant margin.

# In[ ]:


# Getting some more information about data
hm.info()


# - It could be observed that there are no missing values and all the values are type of integer.

# ## Data Statistics

# In[ ]:


# Getting some statistics about the dataset
# It is very useful some times
display(hm.describe())


# ## Observation
# - Based on above scenario, some of the high level informations about the dataset are as follows.
#         - There are 224 patients who survived 5 years or more and 81 survived less than 5 years.
#         - There are no missing values in dataset.
#         - All the data points of each feature are in the form of integer.
#         - General statistics about each feature (represented as real values):
#             - count: Number of data points of corresponding feature.
#             - mean: Average of corresponding feature.
#             - std: Standard deviation.
#             - min: Min value.
#             - 25%, 50%, 75%: Percentile.
#             - max: Max value.

# # Analyzing through plots
#    1. Univariate
#        - Only one variable/feature is taken into consideration.
#    2. Bivariate
#        - Two variables/features are taken into consideration.
#    3. Multivariate
#        - Two or more variables are taken into consideration.
#            
#            
# ## Univariate Analysis
#   - Very simple to analyze because we have to look at only one feature.
#   - Doesn't deal with relationship or causes.
#   - Examples: PDF, CDF, Box-plot, Violin-plot, Percentage Distribution, Bar Graph, Histogram, Frequency Polygon, Pie Chart etc.

#  ### Histogram & PDF
#   - PDF - Probability Density Function (using KDE)
#   - KDE: Kernel Density Estimation.
#   - Can be used to represent the density of a random variable.
#   - Represents non-negative density distribution inside area under curve(AUC). 

# In[ ]:


# Plotting PDF for age of patients
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'age').add_legend()
plt.title("Age of patients and distribution")
plt.xlabel("Age")
plt.show()


# - Solid squared bars are called 'Histogram'
# - Smoothed curved is just the continuous version of histogram and it is called PDF.
# - Since, there is much overlapping, we can't separate both the classes by looking at the age of patients in above plot.
# - In simple way: We can't say that some particular age of patients are belonging to a particular class.

# In[ ]:


# Plotting PDF for operation of year
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'operation_year').add_legend()
plt.title("Operation year and distribution")
plt.xlabel("Operation year (19xx)")
plt.show()


# - Much overlapping according to the year of operation so in this case also, we can't distinguish both the classes.

# In[ ]:


# Histogram of positive axillry nodes
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'positive_axillary_nodes').add_legend()
plt.title("Positive axillary nodes and distribution")
plt.xlabel("Positive axillay nodes")
plt.show()


# - Much overlapping here also, so both the classes can't be distinguished.
# - Blue curved is very peaked with very less spread in above plot that represents the density of patients who survived 5 years of more.
# - We can say that, most of the patients who survived 5 years or more are having positive axillary nodes around 0.

# #### Observation
#   1. Classes can't be separated without applying Machine Learning algorithms because there is much overlapping in all the univariate plots.
#   2. Still it is observed that, most of the patients who survived 5 years or more are having 0 or very less number of positive axillary nodes.

# ### CDF
#   - CDF - Cumulative Distribution Function
#   - Non-decreasing continuous function.
#   - CDF can be driven by applying the cumulative sum of PDF for all continuous points.
#   - It is useful to observe the percentile.

# In[ ]:


# Plotting CDF of age of patients
# Here, bins indicates the number of bins data will be divided in.
counts, bin_edges = np.histogram(hm['age'], bins=10)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("CDF plot for age")
plt.xlabel("Age")
plt.ylabel("Probabilities")
plt.show()


# - Here, orange curve is CDF that is representing the percentile.
# - Blue curve represents PDF.
# - PDF says, there are 20% patients between the age of 50-60.
# - Similarly, from CDF, we can observe that, 40% patients are under the age of 50 and similarly, 90% patients are under the age of 70.

# In[ ]:


# Plotting CDF of age of patients
counts, bin_edges = np.histogram(hm['positive_axillary_nodes'], bins=10)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.title("CDF plot for positive axillary nodes")
plt.xlabel("Positive axillary nodes")
plt.ylabel("Probabilities")
plt.show()


# - Here also we can observe that, more than 80% patients are having the counts of positive axillary nodes less than 10.  

# In[ ]:


# Univariate scatter plot
survived = hm.loc[hm["survival_status"] == 1]
non_survived = hm.loc[hm["survival_status"] ==2]

plt.plot(survived["age"], np.zeros_like(survived['age']), 'o', label="1")
plt.plot(non_survived["age"], np.zeros_like(non_survived["age"]), 'o', label="2")
plt.title("Age of survival and non survival")
plt.xlabel("Age")
plt.legend()

plt.show()


# #### Observation
# - It seems like more number of patients belong to the class label 2.
# - It is also observed that, between age of 40-60, very few patients belong to class label 1.
# - Anyway, there could be such nodes which might be overlapped by another and may not be visible.
# - Similar plotting can be performed for other features as well.
# 

# ###  Box-plot
#     - It also belongs to univariate analysis
#         - Two terminology:
#             (i). Quartile:
#                 (a). Lower quartile (25%)
#                 (b). Middle quartile/median (50%)
#                 (c). Upper quartile(75%)
#             (ii). Whiskers:
#                 (a). Lower whisker
#                 (b). Upper whisker
#         - Useful to observe more information about data.

# In[ ]:


# Plotting the box-cox for survival status of age
sb.boxplot(x='survival_status', y='age', data=hm)
plt.title("Box-plot of age")
plt.xlabel("Survival status")
plt.ylabel("Age")
plt.show()


# - While considering the blue picture inside the plot:
#     - lower starting point is 30 and it goes up to 43. This part is called lower whisker.
#     - Starting point of box at 43 is called lower quartile that represents 25% of the information.
#     - A line around middle in the box is called middle quartile and represents 50% of the information.
#     - Upper line of the box is called upper quartile and represents 75% of the information.
#     - the most upper line is called upper whisker.
#     
# - Same terminology is applied for orange picture.
# 
# - Both of the boxes are overlapping much on each other by looking horizontal y therefore, that means both of the classes are not separated linearly.

# In[ ]:


# Plotting box-cox for year of operation
sb.boxplot(x="survival_status", y="operation_year", data=hm)
plt.title("Box-plot of operation of year (19xx)")
plt.xlabel("Survival status")
plt.ylabel("Operation year")
plt.show()


# In[ ]:


# Plotting box-cox for positive axillary nodes
sb.boxplot(x="survival_status", y="positive_axillary_nodes", data=hm)
plt.title("Box-plot of positive axillary nodes")
plt.xlabel("Survival status")
plt.ylabel("Positive axillary nodes")
plt.show()


# #### Observation
#    - Too much overlapping of the boxes in box-cox plotting so, it's difficult to make any precise prediction.

# ### Violin plots
#   - Combination of PDF and Box-plot so it's more informative.
#   - Represents four things:
#      1. Median
#      2. Inter quantile range
#      3. 95 percentile
#      4. Density distribution

# In[ ]:


# Violin plots for age
sb.violinplot(x="survival_status", y="age", data=hm)
plt.title("Violin plots of age")
plt.xlabel("Survival status")
plt.ylabel("Age")
plt.show()


# #### Observation
# - As it stated earlier, violin plot represents PDF and Box-cox together:
#     - When we divide the figures inside in half, clearly PDF can be seen by looking horizontal sides. Thus, one full picture either blue or orange in this case, is the combination of two symmetric combination of one PDF.
#     - A thin vertical box inside each one represents the property of box-cox plot and the meaning is same as we saw earlier in box-cox plot.
#     - One white dot in middle of each box represents the middle quartile and represents the same meaning as we saw earlier in box-cox plot.
#     - In this plot also both the class labels are overlapping on each other that means they are not linearly separable.
#     - Similar violin plots can be drawn for other feature as well.

# ## Bivariate Analysis
#   - Two variables are taken into consideration together.
#   - Most often used to understand the relationship between two variables.
#   - Ex: Scatter plot, Pair-plot, Contour plot etc.

# ### 2-D Scatter Plot
#   - Points are represented by intersection of two variables.

# In[ ]:


# Scatter plot of age and operations year
hm.plot(kind="scatter", x="age", y="operation_year")
plt.title("Scatter plot of and and year of operation")
plt.xlabel("Age")
plt.ylabel("Operation year")
plt.show()


# - Above is a simple plot that represents only the points of the intersection of age and operation year.
# - It just represents that age varies between 30-85 and operation year varies between 58-70.

# In[ ]:


# The same plot with different c
sb.set_style("whitegrid")
sb.FacetGrid(hm, hue="survival_status", size=5).map(plt.scatter, "age", "operation_year").add_legend()
plt.title("Age and Operation year")
plt.xlabel("Age")
plt.ylabel("Operation year")
plt.show()


# - All the data points are random and mixed up, so, the class is not linearly separable including above two features age and operation year.
# - One thing that can be observed here that, there more data points belonging to class label 1 as we have seen it earlier in this document.

# In[ ]:


# Scatter plot for Age and Positive axillary nodes
sb.set_style("whitegrid")
sb.FacetGrid(hm, hue="survival_status", size=5).map(plt.scatter, "age", "positive_axillary_nodes").add_legend()
plt.title("Age and Positive axillary nodes")
plt.xlabel("Age")
plt.ylabel("Positive axillary nodes")
plt.show()


# - Most of the patients who belong to class label 1 are having positive axillary nodes less than 10.

# #### Observation
#    - Class labels are not linearly separable even in bivariate plots.
#    - It is showing the same result as we have seen in earlier plots.

# ### Pair-Plot
#   - Indirectly, higher dimension than 2-D or 3-D can be visualize in 2-D plot as pair wise.
#   - Not useful for the features with much higher dimensions.
#   - If number of features are $n$ then possible combinations are ${n}_{{C}_{2}}$.

# In[ ]:


sb.set_style("whitegrid")
sb.pairplot(hm, hue="survival_status", vars=["age", "operation_year", "positive_axillary_nodes"], size=4).fig.suptitle("Pair plot of age, operation year & positive axillary nodes")
plt.show()


# #### Observation
# - These pair-wise plots are extension of scatter-plot.
# - Group of three above diagonal are almost similar as the group of three below the diagonal elements except rotation so they infer the same meanings.
# - All the combination of bivariate plots can be drawn at once in pair-wise but it refers the same meaning as we have seen in scatter plot. It could be seen by looking at corresponding variables like, age and positive axillary nodes or operation year and positive axillary nodes.

# # Conclusion
#   - This is an imbalanced dataset.
#   - Class labels are not linearly separable without applying ML algorithms.
#   - It is not possible to design very simple model based on if-else condition.
#   - Still there are some observations such as, the patients who survived 5 years or more, it is found out that they had less number of positive axillary nodes.
