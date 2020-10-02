#!/usr/bin/env python
# coding: utf-8

# # Intro
# This notebook were done in conjunction with the [Warm Up: Machine Learning with a Heart](https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/page/107/)Warm Up: Machine Learning with a Heart hosted by DataDriven. The main idea of this notebook is to write a first draft analysis and model with given data.<br><br>
# **The competition supplied 13 variables of patients and try to predict whether they will get heart disease or not using these variables.** <br><br> Have a look down here about what data were supplied 

# Obtained from [medium.com](https://medium.com/@dskswu/machine-learning-with-a-heart-predicting-heart-disease-b2e9f24fee84), this is the summary of given data<br><br>"""" There are 15 columns in the dataset, where the `patient_id` column is a unique and random identifier. The target is the `heart_disease_present`, a binary feature (1-got heart disease, 0-no). The remaining 13 features are described in the section below.<br><br>
# `slope_of_peak_exercise_st_segment` (type: int): the slope of the peak exercise ST segment, an electrocardiography read out indicating quality of blood flow to the heart<br>
# `thal` (type: categorical): results of thallium stress test measuring blood flow to the heart, with possible values normal, fixed_defect, reversible_defect<br>
# `resting_blood_pressure` (type: int): resting blood pressure<br>
# `chest_pain_type` (type: int): chest pain type (4 values)<br>
# `num_major_vessels` (type: int): number of major vessels (0-3) colored by flourosopy<br>
# `fasting_blood_sugar_gt_120_mg_per_dl`(type: binary): fasting blood sugar > 120 mg/dl<br>
# `resting_ekg_results` (type: int): resting electrocardiographic results (values 0,1,2)<br>
# `serum_cholesterol_mg_per_dl` (type: int): serum cholestoral in mg/dl<br>
# `oldpeak_eq_st_depression` (type: float): oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms<br>
# `sex` (type: binary): 0: female, 1: male<br>
# `age` (type: int): age in years<br>
# `max_heart_rate_achieved` (type: int): maximum heart rate achieved (beats per minute)<br>
# `exercise_induced_angina` (type: binary): exercise-induced chest pain (0: False, 1: True) """"

# <br><br><br><br><br><br>

# # 1. Read and import data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_values = pd.read_csv("/kaggle/input/train_values.csv")
train_labels = pd.read_csv("/kaggle/input/train_labels.csv")
test_values = pd.read_csv("/kaggle/input/test_values.csv")
submission_format = pd.read_csv("/kaggle/input/submission_format.csv")


# ## <p style="text-indent :1em;" >1.1 Inspect, merge and segregate data

# In[ ]:


# Inspecting data
train_labels.head()


# In[ ]:


test_values.head()


# > > Okay so what I'm doing now is to put the patient_id of the train & test set in a seperate dataframe and drop them from the main dataframe(so test&train_values got 13 columns). Then I'll merge the the train values & labels together (so 14 columns here). And finally append test_values (13 columns) underneath the train_values with the last columns (results) remains null values. <br><br> I combined the train & test set together so that any transformation that I make later will be applied to both sets

# In[ ]:


# Save the patient_id in a separate dataframe
train_patient_id = train_labels['patient_id']
test_patient_id = test_values['patient_id']

# Delete the patient_id columns in our data sets because its not really useful for regression/machine learning
train_labels.drop(columns='patient_id',inplace=True)
train_values.drop(columns='patient_id',inplace=True)
test_values.drop(columns='patient_id',inplace=True)


# In[ ]:


# Combine the train_values and train_labels(contains results)
train = pd.concat([train_values,train_labels],axis=1)

# Append the test_values set nderneath the train sets
df = train.append(test_values,ignore_index=True)

df.info()


# > > So you can see that all columns contain 270 entries except heart_disease_prsent ( whicch is the target/result column). This is because we dont have this data for the test_values set, hence becomes null when we combine them. 

# > > We have 1 object column('thal'), so lets convert that to category so that it will be easier to analyse later

# In[ ]:


df['thal'] = df['thal'].astype('category')


# In[ ]:


df['thal'].value_counts()


# <br><br><br><br><br>

# # 2. Exploratory data analysis

# ## <p style="text-indent :1em;" >2.1 Understand data types

# In[ ]:


# Use describe just to inspect the values from each numeric columns. So I know whether column is boolean or not, whats the mean and std
df.describe()


# > > I figure that I will always need to know what columns are what (either binary, multi category or continous). So I decided to write an algorithm for this. 
# <br> What I did down here is to store the columns & its unique values in separate arrays of **binary, category & continous.**
# <br> I defined *multi category* of having unique values between 3-10 (but can easily change this if you encounter multi category with values more than this.) 
# <br> *Continous* like age, will definitely have a lot of uniue values<br><br>
# Having these arrays, later on i can easily use the array as the mask to get the desired columns (ie: df [ theArray.index ] )

# In[ ]:


binary=[]; binary_num=[]
categ=[];categ_num=[]
cont=[];cont_num=[]
for col in df.columns:
    if df[col].nunique() == 2:
        binary.append(col)
        binary_num.append(df[col].nunique())
    elif (df[col].nunique() > 2) & (df[col].nunique()<10):
        categ.append(col)
        categ_num.append(df[col].nunique())
    else:
        cont.append(col)
        cont_num.append(df[col].nunique())

listBinary = pd.Series(dict(zip(binary,binary_num)))
print('BINARY FEATURES(num unique):\n',listBinary)
listCateg = pd.Series(dict(zip(categ,categ_num)))
print('\n\nMULTI CATEGORY FEATURES(num unique):\n',listCateg) # \n is to skip 1 line
listCont = pd.Series(dict(zip(cont,cont_num)))
print('\n\nCONTINOUS FEATURES(num unique):\n',listCont)


# > > <br>-- 1. age - **Continous**
# <br>-- 2. sex - **Binary** 
# <br>-- 3. chest pain type (4 values) - **Multiple Categorical**
# <br>-- 4. resting blood pressure - **Continous**
# <br>-- 5. serum cholestoral in mg/dl - **Continous**
# <br>-- 6. fasting blood sugar > 120 mg/dl - **Binary** 
# <br>-- 7. resting electrocardiographic results (values 0,1,2) - **Multiple Categorical**
# <br>-- 8. maximum heart rate achieved - **Continous**
# <br>-- 9. exercise induced angina - **Binary** 
# <br>-- 10. oldpeak = ST depression induced by exercise relative to rest - **Continous**
# <br>-- 11. the slope of the peak exercise ST segment - **Multiple Categorical**
# <br>-- 12. number of major vessels (0-3) colored by flourosopy - **Multiple Categorical**
# <br>-- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect - **Multiple Categorical**
# <br><br> Target: heart_disease_present - Binary
# 

# ## <p style="text-indent :1em;" >2.2 Normalised the distribution of continous variables

# ### <p style="text-indent :3em;" > age<br>
# 

# > > > Lets have a look at the distribution of Age. I'm also plotting Age with hue on heart_disease_present so we can see whats the difference between old people and young people

# In[ ]:


# Plot distribution of Age
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
sns.countplot(listCont.index[0],data=df)
plt.subplot(2,1,2)
sns.countplot(listCont.index[0],data=df, hue='heart_disease_present')
plt.show()


# > > > It looks normally distributed (maybe can support with more stats tool but im satisfied enough looking at it).<br><br>
# You can see in the 2nd plot that orange is blue(no disease) is higher on the left side and orange(yes disease) on the right side. This means that older people have more risk getting heart disease (which is expected). 

# <br><br><br>

# ### <p style="text-indent :3em;" >max_heart_rate_achieved

# In[ ]:


# Plot distribution of max_heart_rate_achieved
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
plt.hist(df['max_heart_rate_achieved'],bins=20)
# sns.countplot(listCont.index[1],data=df)
plt.subplot(2,1,2)
sns.countplot(listCont.index[1],data=df, hue='heart_disease_present')
plt.show()

# # Supplementary.. just looking if theres correlation between age & max_heart_rate with hue on heart_disease
# sns.scatterplot(df['max_heart_rate_achieved'],df['age'],hue =df['heart_disease_present'])


# > > > Looks a bit skewed to the right? lets check the skewness value (if between -1 & +1, its acceptable)<br>On the 2nd plot, seems like theres more risk of heart disease when the max_heart_rate is low. 

# > > > According to wiki, skewness is the measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. Based on what I understand, skewness between -0.5 & +0.5 is good but anything between -1 and +1 is acceptable too.

# In[ ]:


# Check Skewness 
from scipy.stats import skew
print(skew(df['max_heart_rate_achieved']))

# Sweet, its about -0.52 which lies between -1 & +1. So no need to transform


# <br><br><br>

# ###  <p style="text-indent :3em;" >oldpeak_eq_st_depression (a measure of abnormality in electrocardiograms)<br>

# In[ ]:


# Plot distribution of oldpeak_eq_st_depression
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
# plt.hist(df['oldpeak_eq_st_depression'],bins=20)
sns.countplot(listCont.index[2],data=df)
plt.subplot(2,1,2)
sns.countplot(listCont.index[2],data=df, hue='heart_disease_present')
plt.show()


# > > > It looks to me that this can be converted into categorical feature by segregating them into 6 groups, <br> =0 
# <br> <1
# <br> <2 
# <br> <3
# <br> <4
# <br> <5
# <br> <10
# <br><br> Then we inspect the histogram plot again.

# In[ ]:


bins = [-1, 0, 1, 2, 3, 4, 5, 10]
names = [0, 1, 2, 3, 4, 5, 6]

df['OldPeakRange'] = pd.cut(df['oldpeak_eq_st_depression'], bins, labels=names)

# print(df.OldPeakRange.value_counts()) # Double check this by looking at the value from the first subplot above


# In[ ]:


# Plot distribution of oldpeak_eq_st_depression
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
# plt.hist(df['oldpeak_eq_st_depression'],bins=20)
sns.countplot('OldPeakRange',data=df,hue='heart_disease_present')


# <br><br><br>

# ### <p style="text-indent :3em;" >resting_blood_pressure<br>

# In[ ]:


# Plot distribution of resting_blood_pressure
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
sns.countplot(listCont.index[3],data=df)
plt.subplot(2,1,2)
sns.countplot(listCont.index[3],data=df, hue='heart_disease_present')
plt.show()


# <br><br><br>

# ### <p style="text-indent :3em;" >serum_cholestrol_mg_per_dl

# In[ ]:


# Plot distribution of serum_cholesterol_mg_per_dl
plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
sns.countplot(listCont.index[4],data=df)
plt.subplot(2,1,2)
sns.countplot(listCont.index[4],data=df,hue='heart_disease_present')
plt.show()


# > > > Since the range is huge, I thought maybe its better to separate them into groups. According to MedicalNewsToday they can be grouped into:<br><br>
# (<200) - Normal <br>
# (<239) - Borderline <br>
# (>240) - High <br><br>
# 
# Link - (https://www.medicalnewstoday.com/articles/315900.php)

# In[ ]:


bins = [0, 200, 239, 270, 1000]
names = ['normal','borderline','high','veryhigh']

df['SerumCholestrolRange'] = pd.cut(df['serum_cholesterol_mg_per_dl'], bins, labels=names)

print(df.SerumCholestrolRange.value_counts()) # Double check this by looking at the value from the first subplot above


# In[ ]:


# Plot distribution of serum_cholesterol_mg_per_dl
plt.figure(figsize=(15,4))
sns.countplot('SerumCholestrolRange',data=df,hue='heart_disease_present')


# <br><br><br>

# ### <p style="text-indent :3em;" >Summary of Continous Variables
# 

# > > > -> So two features have been transformed to categorical, that is `oldpeak_eq_st_depression` & `serum_cholesterol_mg_per_dl` that are now stored in new columns called `OldPeakRange` & `SerumCholestrolRange`. So lets delete the first 2<br><br>
# -> Other data remains as it is: `age`, `max_heart_rate_achieved` & `resting_blood_pressure`

# In[ ]:


cat_cols = ['OldPeakRange', 'SerumCholestrolRange']

for col in cat_cols:
    df[col]=df[col].astype('category')
    
df.info()


# In[ ]:


df.drop(columns = ['oldpeak_eq_st_depression', 'serum_cholesterol_mg_per_dl'],inplace=True)


# In[ ]:


df.info()


# <br><br><br>

# ## <p style="text-indent :1em;" >2.2 Converting integer 'categorical' columns to category

# > > So some of the categorical columns that I found just now are stored as integer. This is not an issue but will be better if I can convert them now because I'm planning to apply one-hot encoder on the categorical features and pd.get_dummies will automatically apply this method on all columns that are either object or category type. <br><br><br>
# Recall that we have the `listCateg` that contains the categorical columns as follows: <br>
# <br>
# `chest_pain_type`:  4<br>
# `num_major_vessels`: 4<br>
# `resting_ekg_results`: 3<br>
# `slope_of_peak_exercise_st_segment`: 3<br>
# `thal`: 3<br>
# dtype: int64 

# In[ ]:


for col in listCateg.index:
    df[col]=df[col].astype('category')
print(df.info())


# > > > Now lets just check the total unique values that I have in each category columns and sum overall to see how many columns that will be created

# In[ ]:


uniqueVal = []
for col in df.columns:
    if df[col].dtype not in ('int64','float64'):
        print(col,':',df[col].nunique())
        uniqueVal.append(df[col].nunique())

print('Total Unique values from categorical data: ',sum(uniqueVal))


# > > > What it means here is that the one-hot encoder will create about another 28 columns. (28-7=21 if I exclude the first dummy for each column)<br><br> I will drop the first column of the dummy variables created for each categorical columns and the code automatically delete the columns that we apply dummy, so in total I should get <br><br>14 original + 21 new - 7 old = 28 columns

# In[ ]:


# Uising pd.get_dummies to one-hot encode the categorical data
new_df = pd.get_dummies(df, drop_first=True)

new_df.info()


# > > >The info above tells you how the categorical columns were transformed into one-hot encoders.

# <br><br>

# ## <p style="text-indent :1em;" >2.3 Other unanswered questions?

# > > Yup there are. I have explored and answered a lot of questions till this point, but there are few that I'm still curious about. So here comes the questions:<br><br>
# 1. How does `exercise_induced_angina`,`fasting_blood_sugar_gt_120_mg_per_dl` and `sex` affect `heart_disease_present`
# <br><br>
# Recall, <br>
# -- exercise_induced_angina is exercise-induced chest pain (0: False, 1: True),<br>
# -- fasting_blood_sugar_gt_120_mg_per_dl (type: binary): fasting blood sugar > 120 mg/dl<br>
# -- sex (0-female, 1-male)<br>
# 

# In[ ]:


# Make a countplot with hue on the heart disease for each variables
plt.figure(figsize=(15,12))
plt.subplot(3,1,1)
sns.countplot('exercise_induced_angina',data=new_df,hue='heart_disease_present')
plt.subplot(3,1,2)
sns.countplot('fasting_blood_sugar_gt_120_mg_per_dl',data=new_df,hue='heart_disease_present')
plt.subplot(3,1,3)
sns.countplot('sex',data=new_df,hue='heart_disease_present')

plt.show()


# > > Here's what I can say about the plots
# 1. People who did exercise-induced chest pain tends to get more heart disease. Makes sense
# 2. Not much correlation between fasting blood sugar less than and higher than 120 mg/dl. hmm not a doctor, so cant really comment
# 3. Male got higher risk of getting heart disease ! Wow, is this just accidental or not? 

# <br><br><br><br><br>

# # 3. Preparing the data (train_test_split)

# Recall that I have combined our train & test set. Train initially got 180 rows while submission got 90 rows. So I ungroup them. <br><br>
# This section will be divided into:
# 1. Ungroup new_df into train & submission set. Get the target variable and store in dataframe y (heart_disease_present')
# 2. Drop the target variable in both submission & training sets
# 3. Now its ready to split the train set into two groups: (X_train,y_train) for training the model & (X_test,y_test) for validating the model on unseen data
# <br><br>

# <br><br>
# ## <p style="text-indent :1em;" >3.1 Separate the target(`heart_disease_present`) and store it in y, make a new copy of `new_df`

# > Remove y (heart_disease_present) and get only the frst 180 rows
# 

# In[ ]:


# Store heart_disease_present in y
y = new_df[['heart_disease_present']]
y = y.loc[:179,:]
y.shape


# > Make a copy of new_df called readySet so I can apply transformation later

# In[ ]:


# Make a new copy of new_df and remove heart_disease_present

readySet = new_df.copy()

readySet.drop(columns = 'heart_disease_present',inplace=True)

readySet.info()


# ## <p style="text-indent :1em;" > 3.2 Apply standardScaler on ['int64'] data

# > To do this i will need to find columns with 'int64' first, store them in a list, then use this list to apply the Standard Scaler on the data. <br><br>
# Bear in mind that this will convert the pandas dataframe to numpy array. So moving forward I will deal with numpy array
# 

# In[ ]:


# Make a list of int64 data columns
listIntCol=[]
for col in readySet.columns:
    if readySet[col].dtype not in ['uint8']:
        listIntCol.append(col)
     
        
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Make a Column transformer where I only want to Standard Scale int64 columns
ct = ColumnTransformer([
        ('somename', StandardScaler(), listIntCol)
    ], remainder='passthrough')

readySet_trans = ct.fit_transform(readySet)


# <br><br>
# ## <p style="text-indent :1em;" >3.3 Split the training data and data for submission

# > > This is just splitting the data that will be use for train & validation and the data that will be use for submission

# In[ ]:


trainSet = readySet_trans[:180] # Include index 180
submissionSet = readySet_trans[180:] # Does not include index 180
print(trainSet.shape)
print(submissionSet.shape)

# # If you want to understand about numpy slicing or get confused about it, play around with this example
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# h = x[:3]
# print(h)
# j = x[3:]
# print(j)


# <br><br>
# ## <p style="text-indent :1em;" >3.4 Now lets split trainSet into training & validation group! 

# > > Now splitting the training data into 2 groups: for training and for validation on unseen data. So I'll split 70% for training and 30% for validation

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainSet,y,stratify=y, test_size=0.3, random_state=123)


# In[ ]:


# Check the percentage of each distinct values in y_train & y_test if they are the same

print(pd.value_counts(y_train.values.flatten())/len(y_train))
print(pd.value_counts(y_test.values.flatten())/len(y_test))

# yup same


# <br><br><br><br><br>

# # 4. Building Basic Model

# So i'll start with XGBClassifier model and Logistic Regression model. The idea is like this:<br>
# 1. Instantiate model
# 2. Fit and predict the model
# 3. Print the score and Log Loss (which will be use for competition grading)

# ### XGBClassifier

# In[ ]:


# Import necesary packages
from xgboost import XGBClassifier 
from sklearn.metrics import recall_score, log_loss


xgb = XGBClassifier(seed=123)

xgb.fit(X_train,y_train)

preds = xgb.predict(X_test)

print('Score : {:.4f}'.format(xgb.score(X_test,y_test)))

print('Log Loss : {:.2f}'.format(log_loss(y_test,preds)))


# <br>
# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

lr = LogisticRegression(solver='liblinear')

lr.fit(X_train,y_train)

preds_lr = lr.predict(X_test)

print('Score : {:.4f}'.format(lr.score(X_test,y_test)))

print('Log Loss : {:.2f}'.format(log_loss(y_test,preds_lr)))

lr.get_params()


# <br><br><br><br>
# # 5. Tuning the model

# So to tune the model, I started with defining the parameters that I want to tune and what are the range of values to try out. <br> I use randomizedSearchCV for XGB because to iterate over all combinations is time consuming, so randomizedSearchCV will randomly pick out any combination and iterate it based on how many iteration I specify. <br>For the Logistic Regression, I use GridSearch CV because theres notmuch parameters to tune, so the computational power should have no problem handling them<br>In both cases, the idea of developing the model is similar to the previous section, the only difference is that I added more parameters to find the optimal model

# ### Randomized XGBClassifier

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss

xgb_param_grid = {
    'learning_rate': np.arange(0.01, 0.2, 0.01),
    'max_depth': np.arange(3,10,1),
    'n_estimators': np.arange(50, 200, 50),
    'colsample_bytree':np.arange(0.5,1,0.1),
    'min_child_weight': np.arange(0,2,0.5)
}

randomized_xgb = RandomizedSearchCV(estimator=xgb,
                        param_distributions = xgb_param_grid,
                        cv=10,n_iter=40, scoring="recall",verbose=1,random_state=12) # why 12? Aaron Rodgers

# Fit the estimator
randomized_xgb.fit(X_train,y_train)

preds = randomized_xgb.predict(X_test)

# Compute metrics
print('Best score: ',randomized_xgb.best_score_)
print('\nBest estimator: ',randomized_xgb.best_estimator_)
print('\n Log Loss : {:.4f}'.format(log_loss(y_test,preds)))


# #### Summary XGBClassifier vs RandomSearch(XGBClassifier):
# <br>Default Log Loss =  7.68
# <br>Random Log Loss = 6.3961

# <br><br><br>
# ### GridSearch Logistic Regression

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

# Ignore Warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

lr_param_grid = {
    'penalty': ['l1','l2'],
    'C': np.arange(0.01,1.1,0.1)
}

grid_lr = GridSearchCV(estimator=lr,
                        param_grid = lr_param_grid,
                        cv=5, scoring="recall",verbose=1) # why 12? Aaron Rodgers

# Fit the estimator
grid_lr.fit(X_train,y_train)

preds_rand_lr = grid_lr.predict(X_test)

# Compute metrics
print('Best score: ',grid_lr.best_score_)
print('\nBest estimator: ',grid_lr.best_estimator_)
print('\n Log Loss : {:.4f}'.format(log_loss(y_test,preds_rand_lr)))


# #### Summary LogisticRegression vs GridSearch(LogisticRegression)<br>
# <br>Default Log Loss : 4.48
# <br>GridSearch Log Loss : 4.4773
# <br>

# <br><br><br><br>

# # 6. Submission File

#  Here predicting the submissionSet data and compile them in a csv file to be submitted to the competition.

# In[ ]:


predictions = grid_lr.predict(submissionSet)

# recall I have submission_format file

test = pd.read_csv("/kaggle/input/test_values.csv")
PatientId = test['patient_id']

submission = pd.DataFrame({ 'patient_id': PatientId,
                            'heart_disease_present': predictions })
submission.to_csv(path_or_buf ="HeartDisease_Submission.csv", index=False)
print("Submission file is formed")


# <a href="HeartDisease_Submission.csv"> Download File </a>

# # Wohhooo Not bad! Top 30% for first trial!

# ![image.png](attachment:image.png)

# <br><br>
# # Potential improvements:<br>
# 1. Convert continous variables to categorical <br>
# 2. Check pearson correlation and drop variables that seem the same<br>
# 3. Apply engineering features on potential variables
# 
# <br><br> Let me know if you have any suggestion. 
# # Cheers! :)
