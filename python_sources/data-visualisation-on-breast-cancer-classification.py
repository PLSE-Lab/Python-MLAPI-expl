#!/usr/bin/env python
# coding: utf-8

# **About the Dataset** :- The Haberman's Survival Dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# <br>
# 
# *source :- https://www.kaggle.com/gilsousa/habermans-survival-data-set/data*
# 
# 
# **Attribute Information**:
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
# 

# In[121]:


# importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# In[122]:


# loading the dataset
haberman = pd.read_csv('../input/haberman.csv', header = None, names = ['age_of_patient', 'year_of_operation', 'positive_auxiliary_nodes', 'survived_more_than_5_years'])
haberman.tail()


# ### 1. High Level Statistics

# In[123]:


# summary of the basic information about the dataset
haberman.info()


# | Observations | Conclusions |
# | :- | -: |
# | 306 entries ranging from 0 to 305 | There are only 306 rows in the dataset, which mayn't be sufficient enough data to train a good model |
# | There are 4 columns, 3 describing the datapoint and 1 for the output | The dimensionality of the datapoints is small, 3-D data. Hence, analyzing pair-plots is feasible|
# | Each column has 306 non-null values | There are no missing values in this dataset. |
# | The output i.e. `survival_status_after_5_years` column can take only 2 values. | This is a binary-classification problem. |
# | `survival_status_after_5_years` has `int64` data type | It is supposed to take only boolean values. So it should be re-mapped |
# | Remaining all columns have `int64` as their datatype | The dataset is purely numerical. So a rigourous numerical analysis needs to be done |
# 

# In[124]:


# as per attribute information provided to us, 
# 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
haberman['survived_more_than_5_years'] = haberman['survived_more_than_5_years'].map({1: True, 2: False})


# In[125]:


haberman.info()


# In[126]:


colors = {True: 'green', False: 'red'}


# In[127]:


#(Q) How many datapoints of each class are present( or how balanced is the dataset)?
(haberman["survived_more_than_5_years"].value_counts() / len(haberman)).plot.bar(color = ['green', 'red'])
(haberman["survived_more_than_5_years"].value_counts() / len(haberman))


# | Observations | Conclusions |
# | :- | -: |
# | True(0.7352),   False(0.2647) | The dataset is imbalanced with 74% of datapoints belonging to (+)ve class and 26% belonging to the negative class | 

# In[128]:


haberman.describe()


# In[129]:


# separating dataframes for the two classes
haberman_pos = haberman.loc[haberman.survived_more_than_5_years == True]
haberman_neg = haberman.loc[haberman.survived_more_than_5_years == False]


# In[130]:


# printing basic info of each-column of both classes together
pd.DataFrame({('age_of_patient' + '_pos'): haberman_pos['age_of_patient'].describe(), ('age_of_patient' + '_neg') : haberman_neg['age_of_patient'].describe()})


# In[131]:


pd.DataFrame({('year_of_operation' + '_pos'): haberman_pos['year_of_operation'].describe(), ('year_of_operation' + '_neg') : haberman_neg['year_of_operation'].describe()})


# In[132]:


pd.DataFrame({('positive_auxiliary_nodes' + '_pos'): haberman_pos['positive_auxiliary_nodes'].describe(), ('positive_auxiliary_nodes' + '_neg') : haberman_neg['positive_auxiliary_nodes'].describe()})


# In[133]:


quantiles = haberman['positive_auxiliary_nodes'].quantile(np.arange(0,1.01,0.01), interpolation='higher')


# In[134]:


plt.title("Quantiles and their Values", fontsize = 16)
quantiles.plot(figsize = (12, 8))
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('Positive Auxiliary Nodes', fontsize = 16)
plt.xlabel('Value at the quantile', fontsize = 16)
plt.legend(loc='best', fontsize = 14)

# for co-ordinate labelling at quantiles with 0.25 difference
for x,y in zip(quantiles.index[::25], quantiles.values[::25]):
    plt.annotate(s="({} , {})".format(x,y), xy=(x,y) ,fontweight='bold')

plt.show()


# In[135]:


# (Q) how much %age of patients do have no positive auxiliary nodes at all? (hit-and-trial)
print(haberman["positive_auxiliary_nodes"].quantile(0.44))
print(haberman["positive_auxiliary_nodes"].quantile(0.45))
# Around 44% patients didn't had any positive auxiliary nodes at all


# In[136]:


# (Q) how much %age of patients do have less than 5 positive auxiliary nodes? (hit-and-trial)
print(haberman["positive_auxiliary_nodes"].quantile(0.77))
print(haberman["positive_auxiliary_nodes"].quantile(0.78))
# Around 77% patients had less than 5 positive auxiliary nodes


# **Observations**
# 1. The age of patients ranges from 30 to 83, with an average of 52.
# 2. The dataset is around 50yrs old, with mean year of operation being 1963.
# 3. Even though the maximum number of positive auxiliary nodes in a patient is 52, nearly 77% of the patients are observed to have less than equal to 5 positive auxiliary nodes and nearly 44% of the patients have no positive auxiliary nodes at all.

# ### 2. Objective

# The objective is to predict whether a patient survivied for more than equal to 5yrs or not based on their age, year of operation and their number of positive auxiliary nodes.

# ### 3. Univariate Analysis

# In[137]:


#pdf
for (idx, feature) in enumerate(haberman.columns[:-1]):
    sns.FacetGrid(haberman, hue="survived_more_than_5_years", size=5, palette = colors).map(sns.distplot, feature).add_legend();
    plt.show();


# In[138]:


#cdf
haberman_ = {}
haberman_["True"] = haberman.loc[haberman["survived_more_than_5_years"] == True]
haberman_["False"] = haberman.loc[haberman["survived_more_than_5_years"] == False]

fig, axarr = plt.subplots(1, 3, figsize = (30, 10))

for (idx, feature) in enumerate(haberman.columns[:-1]):
    for truth_value in ['True', 'False']:
        counts, bin_edges = np.histogram(haberman_[truth_value][feature], bins=30, 
                                 density = True)
        pdf = counts/(sum(counts))
        cdf = np.cumsum(pdf)
        axarr[idx].plot(bin_edges[1:], cdf, color = 'green' if truth_value == 'True' else 'red')
    axarr[idx].legend(['True', 'False'])
    axarr[idx].set(xlabel = feature, ylabel = 'CDF')


# We can design a simple if-else based classifier on the feature `positive_auxiliary_nodes` with value greater than around 45(approx) to be patients which didn't survive beyond 5yrs.

# In[139]:


# Finding the exact threshold value for if-else model
haberman_['True']['positive_auxiliary_nodes'].max()
# So, the cut-off can be set at `46`.


# In[140]:


#box-plot
fig, axarr = plt.subplots(1, 3, figsize = (25, 10))
for (idx, feature) in enumerate(haberman.columns[:-1]):
    sns.boxplot(
        x = 'survived_more_than_5_years',
        y = feature,
        palette = colors,
        data = haberman,
        ax = axarr[idx]
    )    
plt.show()


# In[141]:


#violin-plot
fig, axarr = plt.subplots(1, 3, figsize = (25, 10))
for (idx, feature) in enumerate(haberman.columns[:-1]):
    sns.violinplot(
        x = 'survived_more_than_5_years',
        y = feature,
        palette = colors,
        data = haberman,
        ax = axarr[idx]
    )    
plt.show()


# **Observations**
# 1. Out of the 3 features, `positive_auxiliary_nodes` has the most significant distinct-distribution among the two-classes.
# 2. For the patients who survived for more than 5yrs, the distribution for `positive_auxiliary_nodes` seems to be more densed and centred at 0 whereas for those who lived less than 5yrs, the distrubtion is more varying and has larger values as well.
# 3. The number of patients that lived more than 5yrs was dominating in the period **1958 - 1963**, whereas patients that lived less than 5 yrs dominated in the period **1963 - 1967**. Remaining time, both were rougly equal in number.
# 4. The age of the patient doesn't seem to have any say in whether he'll live beyond 5yrs or not.
# 5. The distribution of `age_of_patients` is roughly gaussian.

# ### 4. Bivariate Analysis

# In[142]:


sns.pairplot(haberman, hue='survived_more_than_5_years', diag_kind = "kde", size = 4, palette = colors, vars = ['age_of_patient', 'year_of_operation', 'positive_auxiliary_nodes'])


# **Observation**
# 1. Although there seems to be no clear linear separation between the two classes, the scatter-plot `positive_auxiliary_nodes` Vs `year_of_operation` does a comparatively better job at separating the two classes as compared to other plots.

# ### 5. 3-D Scatter Plot

# In[143]:


import mpl_toolkits.mplot3d
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(haberman['age_of_patient'], haberman['year_of_operation'], haberman['positive_auxiliary_nodes'], 
           c = haberman['survived_more_than_5_years'].map(lambda x: {True:'green', False:'red'}[x]), marker = 'o')
ax.set_xlabel('age_of_patient')
ax.set_ylabel('year_of_operation')
ax.set_zlabel('positive_auxiliary_nodes')
plt.show()


# ### 6. Co-relation Matrix

# In[144]:


corr = haberman.corr(method = 'spearman')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# As can be seen from the spearmann rank-correlation matrix, the most feature that has most correlation with the output `survived_more_than_5_years` is `positive_auxiliary_nodes`. The correlation is negative due to the fact that more number of positive auxiliary nodes refer that the patient belongs to the `False` class.

# ### Final Thoughts
# 
# 1. The dataset is imbalanced, so we need to perform upsampling or downsampling (preferably upsampling as the dataset is already small one) while building a classifier model.
# 2. The feature `positive_auxiliary_nodes` is the most informative feature with respect to the output level we are predicting here, and hence has most feature importance (or weightage) than the remaining ones.
# 3. Since the dimensionality of the data is small and also small number of training examples are there, a high-time complexity algorithm won't be that much of a problem.
# 4. There is high chance of over-fitting to the data, given the small number of datapoints.
# 5. Even in the 3-D space the lines aren't linearly separable without a significant error, so linear models won't do that much good without any feature engineering.
# 6. Naive Bayes is not a good choice as it assumes features to gaussian incase of numeric features. Instead a weighted-KNN with more weightage to the `positive_auxiliary_nodes` might be a good choice.
