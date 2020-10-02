#!/usr/bin/env python
# coding: utf-8

# # The Problem  
# ## Input
# The Cleveland database consists of 13 features that a related to the condition of the heart.
# 
# > 1. **age**: age in years
# > 2. **sex**: (1 = male; 0 = female) 
# > 3. **cp**: chest pain type (4 values)
# > 4. **trestbps**: resting blood pressure
# > 5. **chol**: serum cholestoral in mg/dl
# > 6. **fbs**: fasting blood sugar > 120 mg/dl
# > 7. **restecg**: resting electrocardiographic results (values 0,1,2)
# > 8. **thalach**: maximum heart rate achieved
# > 9. **exang**: exercise induced angina
# > 10. **oldpeak**: ST depression induced by exercise relative to rest
# > 11. **slope**: the slope of the peak exercise ST segment
# > 12. **ca**: number of major vessels (0-3) colored by flourosopy
# > 13. **thal**: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# ## Output
# Given these 13 features, the goal is to predict if a heart disease is present (the 14th variable of the database). 

# # Motivation
# This year my father had a heart attack. He was lucky that a cardiologist was present at the hospital. Hence, the heart attack was diagnosed correctly pretty fast and he was operated on quickly. However, normally the cardiologist would not have been there at this time and consequently the diagnosis would have taken far longer. It was clear to me that this task has to be automated by a machine learning algorithm. So I did a quick research and found the Cleveland database on Kaggle. My goal is to predict the presence of a heart disease given the features of the database. 
# Let's go!

# # Loading the data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

#import libraries
import warnings
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning


# In[ ]:


#loading data
data = pd.read_csv('../input/heart.csv')

#swap target variable
data["target"] = data["target"]==0


# # Analyzing the data
# ## Taking a quick glance

# In[ ]:


# get shape of dataframe
data.shape


# In[ ]:


# quick glance at the data
data.head()


# In[ ]:


# get most important statistics of variables 
data.describe()


# The data consists of 303 observations with 13 features and a label. 
# 
# - Numerical features: age, trestbps, chol, thalach, oldpeak, ca
# - Ordinal categorical: cp
# - Nominal categorical: sex, fbs, restecg, exang, slope, thal
# - Binary categgorical: sex, fbs, exang
# 
# There are no missing or unknown values. 

# ## In depth analysis of variables and relations
# ### Target variable

# In[ ]:


# plot distribution of target variable
plt.bar(data["target"].value_counts().index, data["target"].value_counts())
plt.xticks(data["target"].unique())
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.show()


# In[ ]:


data["target"].mean()


# 45% of the patients suffer from a heart disease. But be careful! The population of the sample are patients in a hospital. This is only a small biased subset of the world population. So we can not infer that 45% of all citizens suffer from a heart disease. 

# ### Age

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(data["age"])
plt.xlabel("Age")
plt.ylabel("Count")
plt.subplot(122)
sns.violinplot(data["target"], data["age"])
plt.show()


# In[ ]:


data["age"].describe()


# In[ ]:


data.loc[data["target"]==1, "age"].describe()


# In[ ]:


data.loc[data["target"]==0, "age"].describe()


# The average age of the patients of this sample is 54. The youngest patient is 29 years old, while the oldest patient is 77 years old. The age is pretty normal distributed and the distribution looks symmetric. Further it is unimodal with a modus of about 60 and 55. As we can see from the mean and the violinplot, patients with a heart disease are on average 4 years older. Further the standard deviation tells us that the age varies less among patients with a heart disease. 

# ### Sex

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["sex"].value_counts().index, data["sex"].value_counts())
plt.xticks(data["sex"].unique())
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"sex"].value_counts().index, data.loc[data["target"]==0,"sex"].value_counts())
plt.xticks(data["sex"].unique())
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"sex"].value_counts().index, data.loc[data["target"]==1,"sex"].value_counts())
plt.xticks(data["sex"].unique())
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["sex"].mean()


# In[ ]:


data.loc[data["target"]==0,"sex"].mean()


# In[ ]:


data.loc[data["target"]==1,"sex"].mean()


# 68% of all patients in the sample are male. In the subset of all patients without a heart disease 56% are male, while in the subset of all patients with a heart disease 83% are male. Hence, in this sample the propability of a male patient to have a heart disease is higher than the propability of a female patient. 

# ### Chest pain type

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["cp"].value_counts().index, data["cp"].value_counts())
plt.xticks(data["cp"].unique())
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"cp"].value_counts().index, data.loc[data["target"]==0,"cp"].value_counts())
plt.xticks(data["cp"].unique())
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"cp"].value_counts().index, data.loc[data["target"]==1,"cp"].value_counts())
plt.xticks(data["cp"].unique())
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["cp"].value_counts() / len(data["cp"]) * 100


# In[ ]:


data.loc[data["target"]==0,"cp"].value_counts() / len(data.loc[data["target"]==0,"cp"]) * 100


# In[ ]:


data.loc[data["target"]==1,"cp"].value_counts() / len(data.loc[data["target"]==1,"cp"]) * 100


# Almost half of the patients do not have chest pain. If they have chest pain, it is mostly type 2. In the subset of the patients with a heart disease the proportion of patients without chest pain is even bigger. Most of the patients without a heart disease have chest pain type 2. In this subset only roughly a quarter does not have chest pain.

# ### Resting blood pressure

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(data["trestbps"])
plt.xlabel("Resting blood pressure (in mm Hg)")
plt.ylabel("Count")
plt.subplot(122)
sns.violinplot(data["target"], data["trestbps"])
plt.ylabel("Resting blood pressure (in mm Hg)")
plt.show()


# In[ ]:


data["trestbps"].describe()


# In[ ]:


data.loc[data["target"]==0, "trestbps"].describe()


# In[ ]:


data.loc[data["target"]==1, "trestbps"].describe()


# The average blood pressure of all patients is 132mmhg. The distribution is pretty normel and unimodel. However, since it is a bit right skewed, it is not perfectly symmetric. The average blood pressure of patients without a heart disease is a tiny bit lower and the distribution has less variance and is less right skewed. 

# ### Serum cholestoral

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(data["chol"])
plt.xlabel("Serum Cholestoral (in mg/dl)")
plt.ylabel("Count")
plt.subplot(122)
sns.violinplot(data["target"], data["chol"])
plt.ylabel("Serum Cholestoral (in mg/dl)")
plt.show()


# In[ ]:


data["chol"].describe()


# In[ ]:


data.loc[data["target"]==0, "chol"].describe()


# In[ ]:


data.loc[data["target"]==1, "chol"].describe()


# The mean serum cholestoral of all patients is 246 mg/dl. It is pretty normaly distributed and unimodal. But the distribution is a bit right skewed with a maximum outlier of **564 mg/dl**. The mean does not differ much between patients with and without a heart disease. However, patients without a heart disease tend to have a bit lower serum cholestoral. The violinplot shows us, that among patients without a heart disease, there are more outliers with very high serum cholestoral.

# ### Fasting blood sugar

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["fbs"].value_counts().index, data["fbs"].value_counts())
plt.xticks(data["fbs"].unique())
plt.xlabel("Fasting blood sugar > 120 mg/dl")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"fbs"].value_counts().index, data.loc[data["target"]==0,"fbs"].value_counts())
plt.xticks(data["fbs"].unique())
plt.xlabel("Fasting blood sugar > 120 mg/dl")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"fbs"].value_counts().index, data.loc[data["target"]==1,"fbs"].value_counts())
plt.xticks(data["fbs"].unique())
plt.xlabel("Fasting blood sugar > 120 mg/dl")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["fbs"].mean()


# In[ ]:


data.loc[data["target"]==0,"fbs"].mean()


# In[ ]:


data.loc[data["target"]==1,"fbs"].mean()


# 15% of all patients have a fasting blood sugar greater then 120 mg/dl. The proportion of patients with a heart disease is 16%, while the proportion of patients without a heart disease is 14%. Hence, the proportion of patients with a fasting blood sugar greater then 120 mg/dl is a bit higher among patients with a heart disease.  

# ### Resting electrocardiographic results

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["restecg"].value_counts().index, data["restecg"].value_counts())
plt.xticks(data["restecg"].unique())
plt.xlabel("Resting electrocardiographic results")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"restecg"].value_counts().index, data.loc[data["target"]==0,"restecg"].value_counts())
plt.xticks(data["restecg"].unique())
plt.xlabel("Resting electrocardiographic results")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"restecg"].value_counts().index, data.loc[data["target"]==1,"restecg"].value_counts())
plt.xticks(data["restecg"].unique())
plt.xlabel("Resting electrocardiographic results")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["restecg"].value_counts() / len(data["restecg"]) * 100


# In[ ]:


data.loc[data["target"]==0,"restecg"].value_counts() / len(data.loc[data["target"]==0,"restecg"]) * 100


# In[ ]:


data.loc[data["target"]==1,"restecg"].value_counts() / len(data.loc[data["target"]==0,"restecg"]) * 100


# Among all patients the proportion of electrocardiographic result 0 and 1 is rougly the same. Only about 1% get the result 2. In the subset of all patients with a heart disease result 0 is more likely, while among all patients without a heart disease result 1 is more likely. 
# 

# ### Maximum heart rate 

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(data["thalach"])
plt.xlabel("Maximum Heart Rate (in bpm)")
plt.ylabel("Count")
plt.subplot(122)
sns.violinplot(data["target"], data["thalach"])
plt.ylabel("Maximum Heart Rate (in bpm)")
plt.show()


# In[ ]:


data["thalach"].describe()


# In[ ]:


data.loc[data["target"]==0, "thalach"].describe()


# In[ ]:


data.loc[data["target"]==1, "thalach"].describe()


# The average maximum heart rate of all patients is 150 bpm. The variables distribution look similar to a normal distribution and is unimodal. It is a bit left skewed. The average maximum heart rate of patients without a heart disease is on average roughly 20 bpm higher than the maximum heart rate of patients with a heart disease. The variance is a bit less. 

# ### Exercise induced angina 

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["exang"].value_counts().index, data["exang"].value_counts())
plt.xticks(data["exang"].unique())
plt.xlabel("Exercise induced angina")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"exang"].value_counts().index, data.loc[data["target"]==0,"exang"].value_counts())
plt.xticks(data["exang"].unique())
plt.xlabel("Exercise induced angina")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"exang"].value_counts().index, data.loc[data["target"]==1,"exang"].value_counts())
plt.xticks(data["exang"].unique())
plt.xlabel("Exercise induced angina")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["exang"].value_counts() / len(data["exang"]) * 100


# In[ ]:


data.loc[data["target"]==0, "exang"].value_counts() / len(data.loc[data["target"]==0, "exang"]) * 100


# In[ ]:


data.loc[data["target"]==1, "exang"].value_counts() / len(data.loc[data["target"]==1, "exang"]) * 100


# For only 32% of all patients exercise did induce angina. However, for most of the patients with heart disease exercise induced angina. This is contrary to the patients without heart disease. The large majority of patients without heart disease did not have symptoms of angina while exercising. 

# ### ST depression induced by exercise relative to rest

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.hist(data["oldpeak"])
plt.xlabel("ST depression")
plt.ylabel("Count")
plt.subplot(122)
sns.violinplot(data["target"], data["oldpeak"])
plt.ylabel("ST depression")
plt.show()


# In[ ]:


data["oldpeak"].describe()


# In[ ]:


data.loc[data["target"]==0, "oldpeak"].describe()


# In[ ]:


data.loc[data["target"]==1, "oldpeak"].describe()


# The average st depression induced by exercise relative to rest is 1. The distribution is not normal and not symmetric. It is unimodal with a modus of 0. It is heavily right skewed with a maximum outlier of 6.2. The average st depression of patients without a heart disease is with 0.6 about a third of the average st depression of patients with a heart disease. Further the variance is smaller and the distribution is less right skewed. 

# ### Slope of the peak exercise ST segment

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["slope"].value_counts().index, data["slope"].value_counts())
plt.xticks(data["slope"].unique())
plt.xlabel("Slope of the peak exercise ST segment")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"slope"].value_counts().index, data.loc[data["target"]==0,"slope"].value_counts())
plt.xticks(data["slope"].unique())
plt.xlabel("Slope of the peak exercise ST segment")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"slope"].value_counts().index, data.loc[data["target"]==1,"slope"].value_counts())
plt.xticks(data["slope"].unique())
plt.xlabel("Slope of the peak exercise ST segment")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["slope"].value_counts() / len(data["slope"]) * 100


# In[ ]:


data.loc[data["target"]==0, "slope"].value_counts() / len(data.loc[data["target"]==0, "slope"]) * 100


# In[ ]:


data.loc[data["target"]==1, "slope"].value_counts() / len(data.loc[data["target"]==1, "slope"]) * 100


# Among all patients a slope of 2 and 1 are equally likely in this sample. Only 7% of all patients have a slope of 0. 
# In the subset of all patients with a heart disease a slope of 1 is way more likely than a slope of 2 while among all patients without a heart disease a slope of 2 is way more likely than a slope of 1. 

# ### Number of major vessels colored by flourosopy

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["ca"].value_counts().index, data["ca"].value_counts())
plt.xticks(data["ca"].unique())
plt.xlabel("Number of major vessels")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"ca"].value_counts().index, data.loc[data["target"]==0,"ca"].value_counts())
plt.xticks(data["ca"].unique())
plt.xlabel("Number of major vessels")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"ca"].value_counts().index, data.loc[data["target"]==1,"ca"].value_counts())
plt.xticks(data["ca"].unique())
plt.xlabel("Number of major vessels")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["ca"].value_counts() / len(data["ca"]) * 100


# In[ ]:


data.loc[data["target"]==0, "ca"].value_counts() / len(data.loc[data["target"]==0, "ca"]) * 100


# In[ ]:


data.loc[data["target"]==1, "ca"].value_counts() / len(data.loc[data["target"]==1, "ca"]) * 100


# For the majority of the patients no major vessels are colored by flourosopy. The higher the number major vessels colored by flourosopy, the less patients are in this category. Among all patients with a heart disease, the average number of vessels colored by flourosopy is much higher than the number of patients without a heart disease. Most of the patients without a heart disease do not have any vessels colored by flourosopy.

# ### Thallium Stress Tests

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
plt.bar(data["thal"].value_counts().index, data["thal"].value_counts())
plt.xticks(data["thal"].unique())
plt.xlabel("Thallium Stress Tests")
plt.ylabel("Count")
plt.title("Patients")
plt.subplot(132)
plt.bar(data.loc[data["target"]==0,"thal"].value_counts().index, data.loc[data["target"]==0,"thal"].value_counts())
plt.xticks(data["thal"].unique())
plt.xlabel("Thallium Stress Tests")
plt.ylabel("Count")
plt.title("Patients without heart disease")
plt.subplot(133)
plt.bar(data.loc[data["target"]==1,"thal"].value_counts().index, data.loc[data["target"]==1,"thal"].value_counts())
plt.xticks(data["thal"].unique())
plt.xlabel("Thallium Stress Tests")
plt.ylabel("Count")
plt.title("Patients with heart disease")
plt.show()


# In[ ]:


data["thal"].value_counts() / len(data["thal"]) * 100


# In[ ]:


data.loc[data["target"]==0, "thal"].value_counts() / len(data.loc[data["target"]==0, "thal"]) * 100


# In[ ]:


data.loc[data["target"]==1, "thal"].value_counts() / len(data.loc[data["target"]==1, "thal"]) * 100


# Most of all patients get result 2. The proportion of result 3 is pretty high as well. Result 1 and 0 is very rare. For all patients without a heart disease the proportion of result 2 is even higher. Most of the patients with a heart disease get result 3. 

# ## Analysis of correlations

# In[ ]:


f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr().apply(abs), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# We can see that, **chest pain**, **angina** and **st depression** have the highest correlation with the **target** variable. This is a hint, that subjective feelings are really important to predict the presence of a heart disease. The **fasting blood sugar** has almost no correlation with the target variable. The correlation matrix shows us, that not all features are independent from each other. For example there is a pretty high correlation between the **st depression** and the **slope of the peak**. Further there is a high correlation between **angina** and **chest pain**. Maybe we can fuse those variables together with pca. 

# # Feature selection
# Since there is only a tiny correlation between the target variable and the fasting blood sugar, I will drop this variable. 

# In[ ]:


# dropping variables
data.drop('fbs', axis=1, inplace=True)


# # Dummy encoding
# We have to turn the nominal categorical variables with multiple classes into binary variables. Therefore we have to use dummy encoding. 

# In[ ]:


# split data in features and labels
features, labels = data.drop("target", axis=1), data["target"]


# In[ ]:


# encode as dummys
features_dummy = pd.get_dummies(features, columns=['restecg', 'slope', 'thal'], drop_first=True)


# In[ ]:


features_dummy.head()


# # Feature scaling
# From the data analysis we know that st depression is pretty skewed. To deal with that we can take the log and look if the distribution then look normal. 

# In[ ]:


# compare distributions
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.hist(features_dummy['oldpeak'])
plt.title("Not Transformed")
plt.subplot(1,2,2)
plt.hist(np.log(features_dummy["oldpeak"]+1))
plt.title("Log Transformed")
plt.show()


# As we can see, log transforming the variable helps in making the distribution normal. Hence, we will apply log transform!

# In[ ]:


# taking the log
features_dummy["oldpeak_log"] = np.log(features_dummy["oldpeak"]+1)


# In[ ]:


# drop old variable
features_dummy.drop('oldpeak', axis=1, inplace=True)


# In[ ]:


features_dummy.head()


# Further we want each variable having the same effect. Therefore standardizing the data makes a lot of sense. 
# 

# In[ ]:


ss = StandardScaler()


# In[ ]:


features_ss = pd.DataFrame(ss.fit_transform(features_dummy), columns=features_dummy.columns)


# In[ ]:


features_ss.head()


# # Data compression
# The correlation matrix in the data analysis step showed us, that there are some strong relationships between the features. Therefore we might archieve some great results by compressing the data with pca. But first we have to estimate the number of components we want to use. 

# In[ ]:


#filling the list with explained variance ratio by components
exp_var = PCA(16).fit(features_ss).explained_variance_ratio_


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(np.arange(1,17,1), np.cumsum(exp_var))
plt.bar(np.arange(1,17,1), exp_var)
plt.xticks(np.arange(1,17,1))
plt.xlabel("Number Components")
plt.ylabel("Explained Variance Ratio")
plt.show()


# The first components explain the majority of the variance. After the 5th component the ratio decreases slowly. Therefore I decided to work with 4 components

# In[ ]:


#compressing the data 
pca_4 = PCA(4, random_state=42)
features_compressed = pd.DataFrame(pca_4.fit_transform(features_ss))


# In[ ]:


#plotting the correlation matrix
f = plt.figure(figsize=(10, 10))
plt.matshow(features_compressed.corr().apply(abs), fignum=f.number)
plt.xticks(range(features_compressed.shape[1]), features_compressed.columns, fontsize=14, rotation=45)
plt.yticks(range(features_compressed.shape[1]), features_compressed.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# As we can see in the correlation matrix the features are now independent. This is a assumption of many learning algorithms, so it is really important to fix this!

# ## Interpreting the components

# In[ ]:


plt.bar(np.arange(0,4,1),pca_4.explained_variance_ratio_)
plt.xticks(np.arange(0,4,1))
plt.xlabel("Component")
plt.ylabel("Explained Variance")
plt.title("Components")
plt.show()


# The plot tells us that the first component explains a lot of variance in the features. Each additional component explains a bit less variance. The ratio is decreasing pretty linear for the last three components.

# In[ ]:


plt.figure(figsize=(20, 10))
sns.heatmap(pd.DataFrame(pca_4.components_, columns = features_ss.columns).apply(abs), annot = True)
plt.xlabel("Features")
plt.ylabel("Components")
plt.title("Weights of components")
plt.show()


# The first component, which explains a lot of variance, weights features high, that are directly measured from the heart like the maximum heart rate and features regarding the ST segment. This means a lot of variance is caused by those measurement from the heart and because they relate to each other they can be fused in one component. The second component weights the sex and the results from the Thallium Stress Tests pretty hight. So they cause a lot of variance as well and they relate to each other so they becaume one component as well. The third variable emphasizes the related blood pressure and serum cholestoral. So they get compressed in one component as well.

# # Split the data 
# Now I will split the data in:
# - training set: 70%
# - test set: 30%
# 
# We do not need a cross-validation set because grid search is working with k-fold cross-validation by itself.

# In[ ]:


# split the data
features_train, features_test, labels_train, labels_test = train_test_split(features_compressed, labels, test_size=0.3, shuffle=True, random_state=42)


# # Fitting and evaluating models

# ## Logistic Regression
# We will start with logistic regression as the classifier.
# As the algorithm to use in the optimization problem I chose 'liblinear'. 
# The regularization type will be 'l2'. 
# We keep maximum iteration as the default(max_iter=100). 
# 
# We use Grid Search to estimate the following hyperparameter:
# - **C**: Inverse of regularization strength
# 
# As the evaluation metric we will use F2 score, because in healthcare we do not want to have false negatives and therefore we want to weight recall more than precision.  

# In[ ]:


# define possible parameter
grid_param = {'C':[0.0001, 0.0005, 0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100]}
scorer = make_scorer(fbeta_score, beta=2)
lr_clf = LogisticRegression(random_state=42, solver='liblinear') 
lrg_clf = GridSearchCV(lr_clf, grid_param, scoring=scorer, cv=5, iid=True)


# In[ ]:


# disable metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
# fit models
lrg_clf = lrg_clf.fit(features_train, labels_train)


# In[ ]:


# get best model
lrg_clf.best_estimator_


# In[ ]:


# get best score
lrg_clf.best_score_


# We could archieve the best F2-Score=0.8 with C=0.0001. 

# In[ ]:


lrg_preds = lrg_clf.predict(features_test)


# In[ ]:


def evaluate(preds, labels):
    f2 = fbeta_score(preds, labels, beta=2)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    print("F2-Score: {0}, Accuracy: {1}, Precision: {2}, Recall: {3}".format(f2, acc, prec, rec))


# In[ ]:


evaluate(lrg_preds, labels_test)


# We could archieve a F2-Score of 0.81 on the test set. 
# We have a accuracy of 84%, which means that 84% of the patients were correctly classified. Since we optimized for the F2-Score our recall is even better. The recall is 85%, which means, that 85% of all patients with a heart disease were classified as having a heart disease. Our precision is 80% and therefore a bit worse. 80%  of patients classified as having a heart disease do really have a heart disease. 

# ## Support Vector Machine 
# We continue with support vector machine as a classifier.
# 
# We have to estimate the **kernel** type with Grid Search. 
# Further let us estimate the penalty parameter **C** of the error term with Grid Search.

# In[ ]:


# define possible parameter
grid_param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100],
              'kernel':['linear','poly','rbf','sigmoid']}
svm_clf = SVC(random_state=42, gamma='auto') 
svmg_clf = GridSearchCV(svm_clf, grid_param, scoring=scorer, cv=5, iid=True)


# In[ ]:


# fit models
svmg_clf = svmg_clf.fit(features_train, labels_train)


# In[ ]:


# get best model
svmg_clf.best_estimator_


# In[ ]:


# get best score
svmg_clf.best_score_


# The best svm has a F2-Score of 82%, uses a sigmoid kernel and C=0.1 as the penalty parameter for the error. 

# In[ ]:


svmg_preds = svmg_clf.predict(features_test)


# In[ ]:


# get evaluation scores
evaluate(svmg_preds, labels_test)


# The SVM archieved a F2-Score of 81% on the test set. Since it has a accuracy of 0.85, 85% of the patients were classified correctly. We got a recall of 0.9 and a precision of 0.79. This means 79% of all patients that were classified as having a heart disease do really have a heart disease. Further we can conclude that 90% of all patients, that have a heart disease, were also calssified as having a heart disease. 

# ## Adaptiv boosted decision trees
# We will go on with a decision tree as a classificator. But instead of training just one decision tree, we will train an ensemble of decision trees. For that we will use a technique called adaptive boosting. We have many weak decision trees, that get trained on a random subset of the data and then they form a strong classifier by voting. Better weak classifiers have more votes. 
# 
# As the criterion for the decision tree we will use the **gini impurity** and we set the max depth to one.  
# We use Grid Search to estimate the **maximum depth**, the **number of trees** and the **learning rate**.

# In[ ]:


# define possible parameter
grid_param = {'n_estimators':[1,3,10,15,20,30,50,75,100],
              'learning_rate':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3,4,5]}
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=42) 
adag_clf = GridSearchCV(ada_clf, grid_param, scoring=scorer, cv=5, iid=True)


# In[ ]:


# fit the model
adag_clf = adag_clf.fit(features_train, labels_train)


# In[ ]:


# get best model
adag_clf.best_estimator_


# In[ ]:


# get best score
adag_clf.best_score_


# The best decision tree ensemble has a F2-Score of 76%. The learning rate is 3 and the number of trees is 75.

# In[ ]:


adag_preds = adag_clf.predict(features_test)


# In[ ]:


# get evaluation scores
evaluate(adag_preds, labels_test)


# With the ensemble of weak decision trees we archieved a F2-Score of 0.75 on the test set. Further we have a accuracy of 79%, which means that we have classified 79% of the patients correctly. The precsion is 72% and the recall is 88%. Consequently 72% of all patients, that were classified as having a heart disease do really have a heart disease and 88% of all patients, that have a heart disease, were classified as having a heart disease. 

# ## Gaussian Naive Bayes Classificator
# The last model we will train is the gaussian naive bayes classificator. 
# Since we are not healthcare experts, we can not come up with a good prior and therefore the prior has to be calculated from the data. Hence, we do not have to set any parameters, which means that we don't have to use grid search. 

# In[ ]:


gnb_clf = GaussianNB()


# In[ ]:


# fitting the classifier
gnb_clf = gnb_clf.fit(features_train, labels_train)


# In[ ]:


# predicting on test set
gnb_preds = gnb_clf.predict(features_test)


# In[ ]:


# evaluate model
evaluate(gnb_preds, labels_test)


# # Choosing best model
# In my opinion support vector machine did the best job for this problem. It has the best F2-Score and the best recall. And the recall is very important here, because we really do not want false negatives. This would mean, that we would tell a person with a heart disease that he or she is healthy. SVM does not have the best precision, but that's okay, since we better diagnose one patient more as ill, than one patient less. 

# # Conclusion 
# We analyzed the variables of the data deeply and also looked at their correlations. We found that the fasting blood sugar is not related to the presence of a heart disease and dropped the variable. Further we noticed, that there are many strong correlation between the variables, which enabled us to compress the data to only 4 variables after scaling and transforming the variables. We trained many models, but found that support vector machine did the best job with respect to the recall. The recall is very important in healthcare problems because we really want to avoid false negatives. 

# In[ ]:




