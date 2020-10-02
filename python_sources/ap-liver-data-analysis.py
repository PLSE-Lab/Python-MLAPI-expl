#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Introduction

# I'm intending this to be the first notebook of two working with this data. This notebook will be focused on exploratory analysis and visualization and the second will be about making predictive models.

# ## Loading Data, Dealing with Missing Data, and Managing Column Names

# In[ ]:


data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
data.info()


# There's 4 missing values in the Albumin and Globulin Ratio. It could be a useful predictor so it might be worthwhile to try estimating its value based on the mean of the most similar entries in the dataframe. Worst case, the data can be dropped as it's a small enough number it shouldn't affect the dataset one way or another.

# In[ ]:


missing = data[data.Albumin_and_Globulin_Ratio.isna()]

for index, row in missing.iterrows():
    age, disease, gender = row["Age"], row["Dataset"], row["Gender"]
    new_table = data[(data["Age"] == age) & (data["Gender"] == gender) & (data["Dataset"] == disease)]
    print(age, disease, gender, new_table["Albumin_and_Globulin_Ratio"].mean())
    data.set_value(index, "Albumin_and_Globulin_Ratio", new_table["Albumin_and_Globulin_Ratio"].mean())


# In[ ]:


data.info()


# I'm happy with the imputing for the missing values. The people with liver disease have slightly reduced ratios which would fit some possible impairment of Albumin synthesis by the liver while the people without liver disease have AGRs above 1

# In[ ]:


data.columns


# In[ ]:


data = data.rename(columns = {"Alkaline_Phosphotase": "ALP", "Alamine_Aminotransferase": "ALT", "Aspartate_Aminotransferase": "AST", "Total_Protiens":"Protein", "Albumin_and_Globulin_Ratio": "AGR", "Dataset": "Liver Patient"})

data["Liver Patient"].replace(2, 0, inplace = True)


# Renaming extremely long column names to their common abbreviations; changing Dataset variable to Liver Patient so it's not confusing as well as all 2s to 0s.
# 
# Previously a 1 meant liver patient and a 2 meant a non-liver patient; currently 0 means a non-liver patient and 1 means a liver patient.
# 
# Also for reference, ALP, AST, and ALT together may be called Liver Function Tests or LFTs for short. This is common medical shorthand, and will be used in this notebook.

# # Exploratory Data Analysis

# ## Count Plots

# In[ ]:


f, axes = plt.subplots(1, 2, figsize = (10, 5))
f.tight_layout(pad = 5)
sns.countplot(data = data, x = "Gender", ax = axes[0])
sns.countplot(data = data, x = "Liver Patient", ax = axes[1])


# It's worth noting that this dataset is unbalanced in both Gender and Liver Patient status; there are considerably more men and liver patients than women and non-liver patients. It's not necessarily a problem in this case but extra caution will be required when building predictive models.

# ## Pivot Tables

# In[ ]:


pd.pivot_table(data, index = "Liver Patient")


# The pivot table shows mostly what's to be expected when you compare a population of liver patients with non-liver patients. Higher bilirubin levels; higher LFTs (ALT, ALP, AST); and lower Albumin and AGR levels in liver patients which is consistent with liver damage.

# In[ ]:


pd.pivot_table(data[[col for col in data.columns if col != "Liver Patient"]], index = "Gender")


# Men tend to have higher values for ALT, and AST. They also seem to have slightly lower Albumin and Protein levels as well. Bilirubin levels are much higher in men.

# In[ ]:


pd.pivot_table(data, index = ["Liver Patient", "Gender"])


# When combining Gender and Liver Patient status into a pivot table, a few things change. The higher ALP levels in women from the previous pivot table appears to come from the women who happen to be liver patients. Of those that aren't liver patients, men have higher ALP levels. In similar fashion, the higher Albumin and Protein numbers in women appears to be explained by the liver patients. Men still have higher ALT, AST, and Bilirubin but these differences are more pronounced in liver patients.

# ## Dealing with Outliers

# In[ ]:


data.describe()


# Many of the liver metrics have gigantic jumps from the 75th percentile value to the max value, and this is to be expected as you'll occasionally have patients with extremely high lab values. In this case, these values are likely to severely compress graphs when running data through matplotlib and seaborn, so I'll be working on removing the high values. There's no reason to worry about abnormally low values because 0 is the lowest anything can be.

# In[ ]:


outliers = data[["Total_Bilirubin", "Direct_Bilirubin", "ALP", "ALT", "AST", "Protein", "Albumin", "AGR"]].copy()
index = outliers[(np.abs(stats.zscore(outliers)) < 2.5).all(axis = 1)].index
index2 = outliers[(np.abs(stats.zscore(outliers)) < 2).all(axis = 1)].index

outliers_25z = data.iloc[index, ].copy()
outliers_2z = data.iloc[index2, ].copy()


# I've made two sets filtering outliers; one set is within a z-score of 2.5 and the other is within a z-score of 2.

# In[ ]:


outliers_25z.describe()


# In[ ]:


outliers_2z.describe()


# Using a z-score of 2.5 trimmed off 76 observations that had extreme values in any of the following: Bilirubin, LFTs, Protein, and/or AGR levels. Using a z-score of 2 further trimmed off 44 observations for a grand total of 120 removals. The z-score of 2 brought down the maximum values slightly except for ALT. Whether this has an impact on distribution remains to be seen.

# ## Visualization

# The data without outliers is only necessary for the distribution plots as seaborn's boxplots can be given the showfliers argument to remove outliers from plots.

# ### LFT Distribution by Gender and Chronic Liver Disease Status - Outliers > 2.5 z-score removed

# In[ ]:


sns.set_style("darkgrid")
graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "AST")
graph


# In[ ]:


graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALT")
graph


# In[ ]:


graph = sns.FacetGrid(outliers_25z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALP")
graph


# ### LFT Distribution by Gender and Chronic Liver Disease Status - Outliers > 2 z-score removed

# In[ ]:


graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "AST")
graph


# In[ ]:


graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALT")
graph


# In[ ]:


graph = sns.FacetGrid(outliers_2z, col = "Liver Patient", row = "Gender", height = 5)
graph.map(sns.distplot, "ALP")
graph


# Both outlier trimmed datasets have roughly similar distributions for LFTs with the 2 z-score set having slightly less compressed graphs due to the lower max values.
# 
# The liver patients tend to have more right skewed data than the non-liver patients which fits the data description as well as medical expectations. Liver patients are more likely to have extreme values in LFTs than non-liver patients. ALP was a bit of an exception as there was a fair deal of right skewing in non-liver patients. This is fairly reasonable as liver damage is not the only source of high ALP in the body; bone disorders can also cause elevations in ALP levels. Since the non-liver patients are still patients, it follows that they could have another ALP elevating condition.

# ### Boxplots with Entire Dataset

# In[ ]:


f, axes = plt.subplots(1, 3, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "AST", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "ALT", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "ALP", hue = "Gender", data = data, orient = 'v', ax = axes[2], showfliers = False)


# LFT boxplots shows much the same from the distplots from above. Although I think some of the increased variability in ALP levels is better communicated in the distplots above.

# In[ ]:


f, axes = plt.subplots(1, 2, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "Total_Bilirubin", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "Direct_Bilirubin", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)


# While the Bilirubin levels in non-liver patients appears very low, they're actually more or less normal. Total Bilirubin of 1.2 mg/dL or less is the roughly normal reference value and Direct Bilirubin is about 0.3 mg/dL or less. The very large boxplots for the liver patient groups is indicative of the excess Bilirubin that's present in liver damage and the large amount of variability. On a related note, the male non-liver patients actually have somewhat elevated Bilirubin as a group but again, a few other conditions (e.g. hemolytic anemia) can cause high Bilirubin without a primary hepatic cause.

# In[ ]:


f, axes = plt.subplots(1, 3, figsize = (15, 6))
f.tight_layout(pad = 5)
sns.boxplot(x = "Liver Patient", y = "Protein", hue = "Gender", data = data, orient = 'v', ax = axes[0], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "Albumin", hue = "Gender", data = data, orient = 'v', ax = axes[1], showfliers = False)
sns.boxplot(x = "Liver Patient", y = "AGR", hue = "Gender", data = data, orient = 'v', ax = axes[2], showfliers = False)


# 
# 
# While there's some increased variability in the liver patients, both liver patients and non-patients have similar distributions of protein metrics (protein, albumin, AGR). The values suggest that approximately half of all patients fall below the minimum values for normal protein and albumin. Now in a liver setting, this can often be a sign of cirrhosis or severe, chronic liver disease. Basically, low protein and albumin in liver disease may suggest that the liver is so impaired that it cannot make vital proteins at a normal rate anymore. However, there are other reasons for low values including renal disease (e.g. Nephrotic Syndrome) and fluid retention (e.g. Congestive Heart Failure). These may be more applicable to the non-liver patients but can apply to the liver patients too.

# # Impressions

# The data appears to show increased variability in Men and Liver Patients in laboratory test values with the exception of ALP and Proteins (Albumin, AGR, and Protein). As many of the included lab values strongly pertain to liver function, it's fair to see the increased variability, and higher mean values in the Liver Patient group. Epidemiological studies suggest that men are more likely to die from chronic liver disease, and so are more likely to have it over women. That could potentially explain the gender differences. 

# To move forward with this data for predictive modeling, and test values would have to be scaled. Further, given that the dataset is unbalanced on both Gender and Liver Patient status, care will have to taken to ensure that the chosen model isn't impaired by that. Potential management of the unbalanced data could be via undersampling using the imbalanced-learn package.
