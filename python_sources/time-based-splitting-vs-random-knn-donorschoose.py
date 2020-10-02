#!/usr/bin/env python
# coding: utf-8

# # KNN: DonorsChoose Dataset

# Refer my kernel of KNN-DonorChooseDataset on Kaggle -> https://www.kaggle.com/nikhilparmar9/kernels <br>
# It is worked on a sample of 2000 Datapoints with **Random Spliting** of data points into Train, CV and Test.<br><br>
# **Time Based Splitting:**
# Below is the comparsion to understand the behaviour and result of KNN when dataset is splitted into Train, CV, Test using **Time Based Splitting** on the data column of the dataset.

# # Comparison Between
# ## Random Splitting VS Time Based Splitting
# # NOTE: Only 2000 Datapoints were sampled and used from 100K points for analysis due to less computation resources.

# ### Original ~100k Dataset is IMBALANCED with Majority Class (1): 85% and Minority Class (0): 15%

# ## 1. Random Splitting
# 1. The sampled 2000 datapoints were randomly splitted into
#    -  Train: 1200
#    -  CV   : 400
#    -  Test : 400

# ### The above training data was also Imbalanced, and therefore it was Upsampled to make it Balanced with "Resampling with Replacement Technique"

# **Upsampled Training Data Points (Random): 2014<br>**
# With Class 1: 1007<br>
# With Class 0: 1007<br>

# ## 2. Time Based Splitting
# 1. The sampled 2000 datapoints were sorted in to ascending order as per the data and then splitted into
#    -  Train: 1200
#    -  CV   : 400
#    -  Test : 400

# ### The above training data was also Imbalanced, and therefore it was Upsampled to make it Balanced with "Resampling with Replacement Technique"

# **Upsampled Training Data Points (Time Based): 1992<br>**
# With Class 1: 996<br>
# With Class 0: 996<br>

# # COMPARISON 

# # 1. SET 1.

# ![set1_auc.jpg](attachment:set1_auc.jpg)

# ![set1_cfm.jpg](attachment:set1_cfm.jpg)

# # 2. SET 2.

# ![set2_auc.jpg](attachment:set2_auc.jpg)

# ![set2_cfm.jpg](attachment:set2_cfm.jpg)

# # 3. SET 3.

# ![set3_auc.jpg](attachment:set3_auc.jpg)

# ![set3_cfm.jpg](attachment:set3_cfm.jpg)

# # 4. SET 4.

# ![set4_auc.jpg](attachment:set4_auc.jpg)

# ![set4_cfm.jpg](attachment:set4_cfm.jpg)

# # 5. SelectKBest 1000 Features on Set 2.

# ![selectKBest_auc.jpg](attachment:selectKBest_auc.jpg)

# ![selectKBest_cfm.jpg](attachment:selectKBest_cfm.jpg)

# # 6. Conclusion

# ![final_conclusion.jpg](attachment:final_conclusion.jpg)
