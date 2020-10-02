#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np 
import pandas as pd 


import os
print(os.listdir("../input"))
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score



# # Import Data Set Using Pandas.

# In[ ]:


Cancer = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")


# # Check Data Type

# In[ ]:


Cancer.info()


# ^ There are 5 columns that have to be deleted, 3 target variables and 2 features. 1) STDs: Time since first diagnosis 2) STDs: Time since last diagnosis 3) Hinselmann 4) Schiller 5) Citology
# 
# Also object data type has to be converted to numeric or integer for preprocessing.

# 

# # Count Of Biopsy Positive/Negative.

# In[ ]:


Cancer['Biopsy'].value_counts()


# # Fill Missing Values with 'NaN'

# In[ ]:


Cancerna = Cancer.replace('?', np.nan) 


# # Find Column Totals of Missing Values

# In[ ]:


Cancerna.isnull().sum() 


# ^ We Can See that 2 features above have to be omitted as they are missing significant portions.

# In[ ]:


Cancer = Cancerna


# ^ Re-define Cancerna as Cancer.

# # Deletion of 5 Attributes.

# In[ ]:


Cancer.drop(Cancer.columns[[26,27,32,33,34]], axis=1, inplace=True)


# # Check How Many Columns Are Empty

# In[ ]:


Cancer.isnull().sum()


# ^ There are 24 columns with missing values and hence they have to be filled in som way.

# # Convert Objects into Numeric For Data Preprocessing

# In[ ]:


Cancer = Cancer.convert_objects(convert_numeric=True) 


# # Fill Columns with Mean/Median.

# In[ ]:


Cancer['Number of sexual partners'] = Cancer['Number of sexual partners'].fillna(Cancer['Number of sexual partners'].mean())
Cancer['First sexual intercourse'] = Cancer['First sexual intercourse'].fillna(Cancer['First sexual intercourse'].mean())
Cancer['Num of pregnancies'] = Cancer['Num of pregnancies'].fillna(Cancer['Num of pregnancies'].median())
Cancer['Smokes'] = Cancer['Smokes'].fillna(Cancer['Smokes'].median())
Cancer['Smokes (years)'] = Cancer['Smokes (years)'].fillna(Cancer['Smokes (years)'].mean())
Cancer['Smokes (packs/year)'] = Cancer['Smokes (packs/year)'].fillna(Cancer['Smokes (packs/year)'].mean())
Cancer['Hormonal Contraceptives'] = Cancer['Hormonal Contraceptives'].fillna(Cancer['Hormonal Contraceptives'].median())
Cancer['Hormonal Contraceptives (years)'] = Cancer['Hormonal Contraceptives (years)'].fillna(Cancer['Hormonal Contraceptives (years)'].mean())
Cancer['IUD'] = Cancer['IUD'].fillna(Cancer['IUD'].median()) 
Cancer['IUD (years)'] = Cancer['IUD (years)'].fillna(Cancer['IUD (years)'].mean())
Cancer['STDs'] = Cancer['STDs'].fillna(Cancer['STDs'].median())
Cancer['STDs (number)'] = Cancer['STDs (number)'].fillna(Cancer['STDs (number)'].median())
Cancer['STDs:condylomatosis'] = Cancer['STDs:condylomatosis'].fillna(Cancer['STDs:condylomatosis'].median())
Cancer['STDs:cervical condylomatosis'] = Cancer['STDs:cervical condylomatosis'].fillna(Cancer['STDs:cervical condylomatosis'].median())
Cancer['STDs:vaginal condylomatosis'] = Cancer['STDs:vaginal condylomatosis'].fillna(Cancer['STDs:vaginal condylomatosis'].median())
Cancer['STDs:vulvo-perineal condylomatosis'] = Cancer['STDs:vulvo-perineal condylomatosis'].fillna(Cancer['STDs:vulvo-perineal condylomatosis'].median())
Cancer['STDs:syphilis'] = Cancer['STDs:syphilis'].fillna(Cancer['STDs:syphilis'].median())
Cancer['STDs:pelvic inflammatory disease'] = Cancer['STDs:pelvic inflammatory disease'].fillna(Cancer['STDs:pelvic inflammatory disease'].median())
Cancer['STDs:genital herpes'] = Cancer['STDs:genital herpes'].fillna(Cancer['STDs:genital herpes'].median())
Cancer['STDs:molluscum contagiosum'] = Cancer['STDs:molluscum contagiosum'].fillna(Cancer['STDs:molluscum contagiosum'].median())
Cancer['STDs:AIDS'] = Cancer['STDs:AIDS'].fillna(Cancer['STDs:AIDS'].median())
Cancer['STDs:HIV'] = Cancer['STDs:HIV'].fillna(Cancer['STDs:HIV'].median())
Cancer['STDs:Hepatitis B'] = Cancer['STDs:Hepatitis B'].fillna(Cancer['STDs:Hepatitis B'].median())
Cancer['STDs:HPV'] = Cancer['STDs:HPV'].fillna(Cancer['STDs:HPV'].median())


# ^ Fill continuous Variables with column mean and discrete/boolean variables with column median. There are 24 columns remaining in my final data set that have missing values. there are 6 columns that are continuous, such as; measurement in years. The remaining are all filled with the column-median.

# # Check All columns have been filled/Summary statistics.

# In[ ]:


Cancer.isnull().sum()


# In[ ]:


Cancer.describe()


# # Heatmap for Attribute Correlation.

# In[ ]:


correlationMap = Cancer.corr()

plt.figure(figsize=(40,40))

sns.set(font_scale=3)
hm = sns.heatmap(correlationMap,cmap = 'Set1', cbar=True, annot=True,vmin=0,vmax =1,center=True, square=True, fmt='.2f', annot_kws={'size': 25},
             yticklabels = Cancer.columns, xticklabels = Cancer.columns)
plt.show()


# ^ It is difficult to see which attributes are correlated, hence it would be more practical to plot heatmaps for one attribute at a time. Also the above correlations can be deduced intuitively. such as; STDs(number) correlated with STDs and IUD correlated with IUD(years) and so on. It may seem obvious, but a noteworthy correaltion to mention is; Age and number of pregnancies at 0.58 correlation.

# In[ ]:


correlationMap = Cancer.corr()
k = 16
correlations = correlationMap.nlargest(k, 'Biopsy')['Biopsy'].index



M = Cancer[correlations].corr()

plt.figure(figsize=(10,10))

sns.set(font_scale=1)
H = sns.heatmap(M, cbar=True, cmap='rainbow' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 12},
                 yticklabels = correlations.values, xticklabels = correlations.values)
plt.show()


# ^ We can see that Hormonal Contraceptives and Age, IUD and Age have an effect on Biopsy

# # Comparison Of Biopsy Result For Attributes

# In[ ]:


new_col= Cancer.groupby('Biopsy').mean()
print(new_col.head().T)

cols = ['Age', 'Number of sexual partners',
        'First sexual intercourse', 'Smokes (packs/year)',
         'Hormonal Contraceptives (years)','IUD (years)', 'Smokes (years)']

sns.pairplot(Cancer,
             x_vars = cols,
             y_vars = cols,
             hue = 'Biopsy',)


# ^ Above we have a 31 by 3 matrix. First Row of 'Biopsy' is divided into its two categories, that is 0 and 1, 0 = negative and 1 = positive for onset of cervical cancer. The following rows are all the remaining 30 features. The average is calculated for each feature, this average is for all values corresponding to 0 and 1 for biopsy-method. We can see that there are 8 features completely free from a positive reading of cervcal cancer.
# 
# 1) STDs:cervical condylomatosis 2) STDs:vaginal condylomatosis 3) STDs:syphilis 4) STDs:pelvic inflammatory disease 5) STDs:molluscum contagiosum 6) STDs:HPV 7) STDs:AIDS
# 8) STDs:Hepatitis B
# 
# Human papilloma virus (HPV) is given to be the prime causer of cervical cancer [5]. In our investiagtion above, it seems not to be detected at all by biopsy screening method, suggesting it has no effect on cervical cancer or that screening method is very inaccurate.
# 
# For positive readings of cervical cancer by Biopsy method, there is an increase for its column-average in each feature. Except for the following features, there is a decrease.
# 
# 1) number of sexual partners 2) Hormonal contraceptives 3) STDs:pelvic inflammatory disease

# # ^ Multiple Pairwise Bivariate Distributions

# Observing graph of 'First sexual intercourse' by 'Age', it is visible that < 20 years for 'first sexual intercourse' and < 40 for 'Age' has an increased occurence of positive reading in Biopsy. Also 'First sexual intercourse' by 'Number of sexual partners' has positive readings concentrated around < 20 years for 'First sexual intercourse' and < 10 for 'Number of sexual partners'.

# # Random Forest Regressor
# 

# Random forest is an ensemble algorithm that takes observations and variables and then creates decision trees. It is useful as it builds multiple decision trees then takes an average for an enhanced accuracey as compared to decision trees.

# # Creation of New columns From Existing Attributes for RF.

# YRSS:Years passed since patient had first sexual intercourse NSPP:Number of sexual partners since first time as a percentage. HPA: Hormonal Contraceptives/age TPS:Total packets of cigarettes smoked NPA:Number of pregnancies/age NSA:Number of sexual partners/age NYHC:number of years patient did not take Hormonal Contraceptives APP:number of pregnancy/number of sexual partner NHCP:number of years patient took Hormonal Contraceptives after first sexual intercourse as a percentage

# In[ ]:


Cancer['YRSS'] = Cancer['Age'] - Cancer['First sexual intercourse']
Cancer['NSPP'] = Cancer['Number of sexual partners'] / Cancer['YRSS']
Cancer['HPA'] = Cancer['Hormonal Contraceptives (years)'] / Cancer['Age']
Cancer['TPS'] = Cancer['Smokes (packs/year)'] * Cancer['Smokes (years)']
Cancer['NPA'] = Cancer['Num of pregnancies'] / Cancer['Age']
Cancer['NSA'] = Cancer['Number of sexual partners'] / Cancer['Age']
Cancer['NYHC'] = Cancer['Age'] - Cancer['Hormonal Contraceptives (years)']
Cancer['APP'] = Cancer['Num of pregnancies'] / Cancer['Number of sexual partners']
Cancer['NHCP'] = Cancer['Hormonal Contraceptives (years)'] / Cancer['YRSS']


# ^ Above, I have decided to create new columns that might better explain a positive reading of cervical cancer.

# In[ ]:


X = Cancer.drop('Biopsy', axis =1)
Y = Cancer["Biopsy"]


# ^ We have defined X and Y.

# Due to the division of columns above, there will be instances of division by zero; giving 'infinity', hence the need to replace 'infinity' with 0.

# In[ ]:


x = X.replace([np.inf], 0)


# We check to see that all columns are full for RF algorithm to run.

# In[ ]:


x.isnull().sum()


# ^ Column 'NHCP' is missing 16 values, we fill with its mean.

# In[ ]:


x['NHCP'] = x['NHCP'].fillna(x['NHCP'].mean())


# In[ ]:


x.isnull().sum()


# # Now we are ready to run RF algorithm.

# In[ ]:


get_ipython().run_line_magic('timeit', '')


# In[ ]:


model = RandomForestRegressor(max_features = 7, n_estimators = 100, n_jobs = -1, oob_score = True, random_state = 42)


# In[ ]:


model.fit(x,Y)


# max_features: This is the maximum number of variables RF is allowed to test in each node. An increase in variables to be tested at each node generally improves performance, the downside is diversity of each node is reduced which is the unique selling point of RF. For our classification problem, I will use sqaure root of count of variables, which in our case is 6.24 but rounded to 7.
# 
# n_estimators: This is the number of trees that are built before average is taken, ideal to have high number of trees, downside is code runs slower.
# 
# n_jobs: This code tells engine how many processors to use, "1" for one processor and "-1" for unrestricted.
# 
# random_state: Thise code allows solution to be easily replicated.
# 
# oob_score: This is a RF cross-validation method.
# 
# We have out of bag score, the trailing underscore after "score" means R^2 is available after model has been trained. We have a R^2 = 0.00317, this is very low.

# In[ ]:


model.oob_score_


# We will calculate a C-stat. The C-statistic also called concordance statistic gives a measure of goodness of fit for binary outcomes in a logistic regression model. It will give us the probability a randomly selected patient who has experienced cervical cancer risks has a higher risk score than a patient who has not experienced the risks. It is equal to the area under the Receiver Operating Characteristic (ROC) curve.

# In[ ]:


Y_oob = model.oob_prediction_


# In[ ]:


print("C-Stat: ", roc_auc_score(Y, Y_oob))


# ^ We have a C-stat of 0.6645, This will be our benchmark and I will try to improve this C-stat value, also C-stat ranges from 0.5 to 1, usually values from and above 0.70 are considered good model fit.
# 
# Y_oob = Y out of bag, this gives the prediction for every single observation. We can see that a lot of observations have 0 predictions and the remaining have low predictions.

# In[ ]:


Y_oob


# # Improving Model

# I will introduce dummy variables for all categorical variables in my data set. This is done to capture directionality of the categorical variables, also dummy variables allow us to use one regression equation to represent many groups. K-1, where k is the number of levels for a variable determines how many dummy variables to use.

# In[ ]:


categorical_variables = ["Smokes", "Hormonal Contraceptives", "IUD", "STDs", "STDs:condylomatosis",                    
"STDs:cervical condylomatosis", "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", 
"STDs:pelvic inflammatory disease", "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:AIDS", "STDs:HIV",                              
 "STDs:Hepatitis B", "STDs:HPV", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"]
for variable in categorical_variables :
    dummies= pd.get_dummies(x[variable], prefix=variable)
    x=pd.concat([x, dummies], axis=1)
    x.drop([variable], axis=1, inplace=True)
    


# In[ ]:


model = RandomForestRegressor(max_features = 8, n_estimators = 100, n_jobs = -1, oob_score = True, random_state = 42)
model.fit(x,Y)


# In[ ]:


print("C-stat : ", roc_auc_score(Y, model.oob_prediction_))


# ^ after introducing dummy variables for all the categorical variables, we get a slight improvement of the model. This is a 3.4% improvement which is not enough to reach our mark of 0.7.

# # Plot And Sort Features Importance

# In[ ]:


feature_importances= pd.Series(model.feature_importances_, index=x.columns)
feature_importances.plot(kind="bar", figsize=(20,20));


# ^ The Most important features are by a large margin, the columns created from existing attributes, namely; (1) YRSS = Years passed since patient had first sexual intercourse (2) NSA = Number of sexual partners/age. (3) NHCP = number of years patient took Hormonal Contraceptives after first sexual intercourse as a percentage
# (4) NYHC = number of years patient did not take Hormonal Contraceptives
# 
# These Features are somewhat correlated with a positive reading on a Biopsy screening method, it is a very weak correlation.

# # Finding Optimal Number Of Trees For RF

# In[ ]:


results=[]
n_estimator_values=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200]
for trees in n_estimator_values:
    model=RandomForestRegressor(trees, oob_score=True, n_jobs=-1,random_state=42)
    model.fit(x, Y)
    print(trees, "trees")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, n_estimator_values).plot();


# ^ 140 trees gives us the highest c-stat, i.e. 0.6798.

# # Finding Optimal Max_Features For RF

# In[ ]:


results=[]
max_features_values=["auto", "sqrt", "log2", None, 0.2, 0.9]
for max_features in max_features_values:
    model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42, 
                                max_features=max_features)
    model.fit(x, Y)
    print(max_features, "option")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, max_features_values).plot(kind="barh", xlim=(0.10, 0.8));


# ^ max_features, as a recap is; the maximum number of variables RF is allowed to test in each node. The optimal for our model is log2.

# # Finding Optimal Min_Samples_Leaf For RF

# In[ ]:


results=[]
min_sample_leaf_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,100,140,200]
for min_sample in min_sample_leaf_values:
    model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42, 
                                max_features="log2", min_samples_leaf=min_sample)
    model.fit(x, Y)
    print(min_sample, "min sample")
    roc_score=roc_auc_score(Y, model.oob_prediction_)
    print("C-stat : ", roc_score)
    results.append(roc_score)
    print(" ")
pd.Series(results, min_sample_leaf_values).plot();


# min_samples_leaf: This is the minimum number of samples required to be at each node. The optimal for our model is; 1.

# # Final Optimised Model

# In[ ]:


model=RandomForestRegressor(n_estimators=140, oob_score=True, n_jobs=-1,random_state=42,
                            max_features="log2", min_samples_leaf=1)
model.fit(x, Y)
roc_score=roc_auc_score(Y, model.oob_prediction_)
print("C-stat : ", roc_score)


# ^ We have achieved the desired 0.70 mark, this means our model has just passed the threshold of a good model fit.

# 

# 

# 

# 

# 
