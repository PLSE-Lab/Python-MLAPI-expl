#!/usr/bin/env python
# coding: utf-8

# # Heart Disease UCI - Part 1 (Exploratory Data Analysis)

# The notebook below deals with the data analysis of the Heart Disease dataset. In order to keep the notebook concise, I have divided the entire notebook into two parts. The first part deals explicitly with the data analysis while the second part is involved with the fitting of various classification models on this simple dataset.
# 

# **Importing the libraries used in this notebook**

# In[ ]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


# **Reading and previewing the data**

# In[ ]:


#Read the dataset from the path and see a preview of it
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()


# **Checking the data types of all the columns**

# In[ ]:


#Check the datatypes of the columns
df.dtypes


# **Checking for count of missing data present in the columns**

# In[ ]:


#Checking out for count of missing values in each column
df.isnull().sum()


# **Bar plot of frequency of Target column **

# In[ ]:


#Checking the distribution of target variable
sns.countplot(df["target"]);


# **Getting the list of all columns for analysis**

# In[ ]:


df.columns


# # ***Exploratory Data Analysis ***

# In the remaining section, the analysis of the categorical random variables(RV) and continuous random variables(RV) have been done in the following way:
# 
# 1. **Categorical random variable columns**
# For the categorical columns, we will be getting 3 plots for each variable: the bar plot of the different levels of the categorical variable, how the target class is varying across each of the levels and finally the odds of having a heart disease with respect to each level of the column.
# 
# 2. **Continuous random variable columns**
# For the continuous columns, we will be checking the distribution of the random variable and will also check the mean value of the random variable across the categories of the target variable.
# In some cases, we will also try to bin the continuous random variable into several categories to transform it into a categorical random variable and carry on with the EDA mention in point 1.

# **AGE : Continuous RV**

# In[ ]:


fig, axes = plt.subplots(1,2,figsize = (15,5))

#Checking the distribution of this column
sns.distplot(df["age"], ax = axes[0]).set_title("Age distribution")

#Target wise old peak average
df.groupby(by = ["target"])["age"].mean().plot(kind = "bar", ax = axes[1], title="Target wise mean age");


# We will now try to bin the age variables to age group baskets of:
# 
# 1. **Young** : (20-40] years
# 2. **Young2Old** : (40-50] years
# 3. **Old** : (50-60] years
# 4. **Senior** : (60-70] years
# 5. **Fragile** : 70+ years
# 
# We will be using pd.cut function to perform a neat binning of the continuous variable age into a categorical variable ageGrp which signifies the age group of a person.

# In[ ]:


#Creating bins for age
lstBins = [20,40,50,60,70,90]
df["ageGrp"] = pd.cut(df["age"], bins = lstBins, labels = ["Young", "Young2Old", "Old", "Senior", "Fragile"])


# **Age Group : Categorical RV**

# In[ ]:


#Checking the distribution of age grp variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["ageGrp"], ax = axes[0]).set_title("Age grp distribution")

#Impact of sex on heart disease
dfTemp = pd.crosstab(df["ageGrp"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Age grp wise heart disease count");

#Calculating the odds of having heart disease for each age grp type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The above analysis reveals that the odds of having a heart disease increase drastically for the fragile group of people i.e. people who age is greater than 70 years.
# 
# Also surprising is that the chances of heart disease is less for people in the Old and Senior age group. 
# There could be 2 reasons for this:
# 1. The data might not be truely random.
# 2. This particular heart disease does not directly develop over time(age) and the chances of having this disease has more to do with other parameters.

# The below swarm plot also helps to see class separation boundary inside each age group.

# In[ ]:


#Check the swarm plot of the age grp variable
sns.swarmplot(x="ageGrp", y="age",hue='target', data=df).set_title("Target(0/1) separation among age groups");


# **Sex : Categorical RV**

# In[ ]:


#Checking the distribution of sex variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
df["sex"].plot(kind="hist", ax = axes[0], title="sex distribution");

#Impact of sex on heart disease
dfTemp = pd.crosstab(df["sex"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Sex wise heart disease count");

#Calculating the odds of having heart disease for each sex type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The analyis clearly shows that the odds of having a heart disease is drastically high for one sex group as compared to the other.

# **Chest Pain : Categorical RV**

# In[ ]:


#Checking the distribution of chest pain variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
df["cp"].plot(kind="hist", ax = axes[0], title="Chest pain distribution");

#Impact of chest pain on heart disease
dfTemp = pd.crosstab(df["cp"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Chest pain wise heart disease count");

#Calculating the odds of having heart disease for each chest pain type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The chest pain wise heart disease reveals that a chest pain type of 0 is the only one which ensures that the odds of heart disease is negligible while all the other chest pain types typically result in a heart disease or are a syptom of heart disease.

# **Resting Blood Pressure : Continuous RV**

# In[ ]:


fig, axes = plt.subplots(1,2,figsize = (15,5))

#Checking the distribution of this column
sns.distplot(df["trestbps"], ax = axes[0]).set_title("BP distribution")

#Target wise old peak average
df.groupby(by = ["target"])["trestbps"].mean().plot(kind = "bar", ax = axes[1], title="Target wise mean BP ");


# The above graph does not give out much information and shows that the mean blood pressure among people who have heart disease versus those who do not, is roughly same. 
# 
# To explore more into this variable, we will be binning the blood pressure into the below categories:
# 1. **Very low** : (70-100]
# 2. **Low** : (100-120]
# 3. **Normal** : (120-140]
# 4. **High** : (140-160]
# 5. **Very high** : (160-220]

# In[ ]:


#Binning the rest systolic BP
bpCatLst = [70,100,120,140,160,220]
df["bpGrp"] = pd.cut(df["trestbps"], bins = bpCatLst, labels = ["very low", "low", "normal","high","very high"])


# **Blood Pressure Group : Categorical RV**

# In[ ]:


#Checking the distribution of BP grp variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["bpGrp"], ax = axes[0]).set_title("BP grp distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["bpGrp"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="BP grp wise heart disease count");

#Calculating the odds of having heart disease for each BP grp type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The above analysis again makes a surprising revelation that the odds of having a heart disease decrease almost linearly with the increase in blood pressure. This makes us believe that increase blood pressure alone does not cause a heart disease and there are clearly other factors working in conjunction to cause a heart diease.

# **Cholestrol : Continuous RV**

# In[ ]:


fig, axes = plt.subplots(1,2,figsize = (15,5))

#Checking the distribution of this column
sns.distplot(df["chol"], ax = axes[0]).set_title("Cholestrol distribution")

#Target wise old peak average
df.groupby(by = ["target"])["chol"].mean().plot(kind = "bar", ax = axes[1], title="Target wise cholestrol mean");


# Like Blood group variable, this variable also has roughly same mean across the target class categories and we should further look to discretizing this variable and then see its distribution.
# 
# Following is the classification which I have done for this variable:
# 
# 1. Normal : (100-200]
# 2. Borderline high : (200-239]
# 3. High : (239-300]
# 4. Very high : (300-350]
# 5. Risky high : (350-700]

# In[ ]:


#Binning the cholestrol levels
cholCatLst = [100,200,239,300,350,700]
df["cholGrp"] = pd.cut(df["chol"], bins = cholCatLst, labels = ["normal", "borderline high", "high","very high","risky high"])


# In[ ]:


#Checking the distribution of cholestrol grp variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["cholGrp"], ax = axes[0]).set_title("CHOLESTROL group distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["cholGrp"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Cholestrol wise heart disease count");

#Calculating the odds of having heart disease for each  type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The above graphs clearly establish the relationship between the cholestrol groups and the odds of having a heart disease.

# **Fasting Blood Sugar : Categorical RV**

# In[ ]:


#Checking the distribution of fasting blood sugar variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["fbs"], ax = axes[0]).set_title("Blood Sugar distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["fbs"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Blood Sugar wise heart disease count");

#Calculating the odds of having heart disease for each BS grp type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# **Resting ECG : Categorical RV**

# In[ ]:


#Checking the distribution of rest ECG variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["restecg"], ax = axes[0]).set_title("ECG distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["restecg"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="ECG wise heart disease count");

#Calculating the odds of having heart disease for each type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# **Exercize induced angina : Categorical RV**

# In[ ]:


#Checking the distribution of exang variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["exang"], ax = axes[0]).set_title("Exercize induced angina distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["exang"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="EXANG wise heart disease count");

#Calculating the odds of having heart disease for each type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# **Old Peak : Continuous RV**

# In[ ]:


fig, axes = plt.subplots(1,2,figsize = (15,5))

#Checking the distribution of this column
sns.distplot(df["oldpeak"], ax = axes[0]).set_title("Old peak distribution")

#Target wise old peak average
df.groupby(by = ["target"])["oldpeak"].mean().plot(kind = "bar", ax = axes[1], title="Target wise Old peak mean");


# The old peak column's mean value for each class of target shows a considerable difference and it seems that the individual variable without binning is sufficient to explain the presence of a heart disease.

# **Slope of ST depression in ECG : Categorical RV**

# In[ ]:


#Checking the distribution of slope variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["slope"], ax = axes[0]).set_title("Slope distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["slope"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Slope wise heart disease count");

#Calculating the odds of having heart disease for each  type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# **Coloured Arteries in Flouroscopy : Cateforical RV**

# In[ ]:


#Checking the distribution of CA variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["ca"], ax = axes[0]).set_title("Coloured arteries distribution")

#Impact of BP grp on heart disease
dfTemp = pd.crosstab(df["ca"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="Coloured arteries wise heart disease count");

#Calculating the odds of having heart disease for each  type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The number of mojor vessels coloured seems to impact the odds directly with the highest and lowest levels of CA resulting a higher odds of having a heart disease.

# **THAL : Categorical RV**

# In[ ]:


#Checking the distribution of variable and its impact on heart disease
fig, axes = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x = df["thal"], ax = axes[0]).set_title("THAL distribution")

#Impact on heart disease
dfTemp = pd.crosstab(df["thal"], df["target"])
dfTemp.plot(kind="bar", ax = axes[1], title="THAL wise heart disease count");

#Calculating the odds of having heart disease for each type
dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)
dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);


# The analysis shows that the odds of having a heart disease is much higher in level 2 of THAL variable and that the odds are almost same for level 1 and level 3. This suggests that the levels 1 and 3 can be clubbed to make a single level.

# # Remarks
# 
# I hope the above analysis helped you visualize how each categorical and continuous random variable was interacting with the target variable and how odds of having a heart disease changes with each level of the categorical variable.
# 
# The above analysis is definitely not exhaustive but we will take a break from the EDA and as mentioned at the start of the notebook, we will go ahead with the modelling part of the problem in Part 2 of this series.
# 
# **Update**: The link to the Part 2 of the notebook has been added below. 
# Do check it out! https://www.kaggle.com/shrijan19/heart-disease-part-2-modelling-interpretation
# 
# Hope you were able to learn something from this notebook. In case you have any doubts or suggestions, just shoot them in the comments section.

# In[ ]:




