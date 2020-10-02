#!/usr/bin/env python
# coding: utf-8

# # A Beginner's approach to Diabetes Classification

# Hello Viewers!
# 
# Thanks for taking a moment to stop by and looking at my Kernel. This notebook is aimed at drafting a simple approach to solve the diabetes classification problem.
# 
# Without talking much, let's get into the solution.

# ## But, Where's the data?

# Here it is...
# 
# We have a dataset on health attributes of several patients and we need to classify whether the patient has diabetes or not.
# 
# People do talk about being independant and dependant. What are they?
# 
# Simple!
# 
# * **Independant Features** : These are the predictor variables using which you predict the outcome.
# * **Dependant Feature** : This is the target variable which we are going to predict.
# 
# Okay, what independant features do we have for now?
# * Pregnancies
# * Glucose
# * Blood Pressure
# * Skin Thickess
# * Insulin
# * BMI
# * Diabetes Pedigree Function
# * Age
# 
# Based on these attributes, we will predict the target variable.
# * Outcome

# ## Importing Libraries and Dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-deep')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# As we have imported the required libraries, let us grab the dataset. I'm storing a copy of it, as we may require it in the future.

# In[ ]:


df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
original = df.copy()


# ## Check Point - Quick Inspection of the Data

# Let's run some simple scripts to check the shape, info and summary statistics of the dataset.

# In[ ]:


print('Data has', df.shape[0], 'rows and', df.shape[1], 'columns')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# Do you see something abnormal? Just look at the minimum value of each feature.
# 
# * Pregnancies : 0
#     * This is acceptable as this dataset has the mix of genders.
# * Glucose
#     * Can Glucose level be zero in any case? Something to ponder over.
# * Blood Pressure
#     * Even patients with low blood pressure would never have BP as 0.
# * Skin Thickness
#     * Can skin thickness be zero? I do not think so, but we'll note this for further investigation.
# * Insulin
#     * Insulin could be very low in fasting conditions. But zero? Okay, hold it for now!
# * BMI
#     * Zero BMI? Impossible!

# The summary statistics is the best measure of getting a quick insight about the accuracy of the data. The above abnormalities could be either erratic or rare occurences. We'll check the dataset further during our EDA process.

# In[ ]:


import missingno as msna
msna.matrix(df)
plt.show()


# It is evident from the above figure that the dataset does not have any missing values.
# 
# *Oh! You noticed that from the df.info() method itself? Cool! You've got an eagle's eye*

# ## Exploratory Data Analysis

# ## Univariate

# ### Target Variable

# Let's check the composition of the target varible. 
# 
# Here 0 corresponds to being normal and 1 to diabetic. We'll use the normalize parameter to tell the value_counts() method that we need the percentage rather than the actual numbers.

# In[ ]:


df['Outcome'].value_counts(normalize = True)


# 65% of the people in the dataset are normal while the rest are diabetic.
# 
# Is this an imbalanced dataset? I would hesitate to say so, we still have 35% of the class of interest. So let's go ahead.

# ### PDF and ECDF - Predictor Variables

# Confused by the terms?
# 
# These are the statistical methods to check the distribution of the data.
# 
# * **Probability Distribution Function**:
# Probability of getting a value, if it is randomly chosen from the data.
# 
# * **Empirical Cumulative Distribution Function**:
# Probability of getting less than a value, if it is randomly chosen from the data.
# 
# We'll check the plots to further understand this definitions.
# 
# Shall we practise writing simple functions as part of our analysis?

# In[ ]:


plt.rcParams['figure.figsize'] = (18, 7)

def univariate_plot(x):
    plt.subplot(121)
    sns.distplot(x, color = 'seagreen')
    plt.title('Probability Distribution Function', fontsize = 15)
    plt.ylabel('Probability')
    
    n = len(x)
    a = np.sort(x)
    b = np.arange(1, 1 + n) / n
    plt.subplot(122)
    plt.plot(a, b, color = 'seagreen', marker = '.', linestyle = 'none')
    mean_x = np.mean(x)
    plt.axvline(mean_x, label = 'Mean', color = 'k')
    skew = '               Skew : ' + str(round(x.skew(), 2))
    plt.annotate(skew, xy = (mean_x, 0.5), fontsize = 16)
    plt.legend()
    plt.title('Empirical Cumulative Distribution Function', fontsize = 15)


# This is a simple function which gets a feature as an input argument and plot the PDF and ECDF of the feature.
# 
# In addition to this, we'll also use the function to calculate the skewness of the feature.
# 
# * Skewness: Distortion present in the data. If the feature is not normally distributed (Gaussian), it may skew either towards the left or right. Most models would perform better with normally distributed data, so let's check how the features are distributed.

# In[ ]:


univariate_plot(df['Age'])


# The Age feature is right skewed as you see the tail is long in the right end. How to cross verify this? The skewness factor is 1.13, which is positive. The positive skewness means that the feature is right skewed.
# 
# From the ECDF graph, it is observed the probability of getting an observation with age less than the mean age is ~60%.

# In[ ]:


univariate_plot(df['BMI'])


# In[ ]:


univariate_plot(df['Pregnancies'])


# In[ ]:


univariate_plot(df['Glucose'])


# In[ ]:


univariate_plot(df['BloodPressure'])


# In[ ]:


univariate_plot(df['SkinThickness'])


# In[ ]:


univariate_plot(df['Insulin'])


# In[ ]:


univariate_plot(df['DiabetesPedigreeFunction'])


# As we have checked the distribution of features, let's check the dataset to get answers for our questions we have framed by looking at the summary statistics.

# The maximum number of pregnancies is 17. Let's check the dataset with observations having more than 10 pregnancies. 

# In[ ]:


df.loc[df['Pregnancies'] > 10, :]


# The age of people who had more than 10 pregnancies seem to be either middle staged or old. Let's keep this feature as such.

# In[ ]:


df.loc[df['BMI'] == 0]


# When we slice the dataset with BMI values equal to zero, we also see zero values in neighboring features as well. These observations must be erratic.

# In[ ]:


df.loc[df['Glucose'] == 0]


# In[ ]:


df.loc[df['BloodPressure'] == 0]


# In[ ]:


df.loc[df['SkinThickness'] == 0]


# In[ ]:


df.loc[df['Insulin'] == 0]


# In[ ]:


drop_index = df.loc[(df['BMI'] == 0) & (df['BloodPressure'] == 0) & (df['SkinThickness'] == 0) & (df['Insulin'] == 0), :].index
df.drop(drop_index, axis = 0, inplace = True)


# We've dropped the observations which have zero values in all the four features. 

# In[ ]:


for i in df.columns.tolist():
    print(i, '-', len(df.loc[df[i] == 0, :]))


# In[ ]:


df.sample(20)


# In[ ]:


df['Outcome'] = df['Outcome'].astype('str')


# We've converted the target variable to string type as this would help us in plotting doing the bivariate analysis.

# ### Bivariate Analysis

# In[ ]:


plt.rcParams['figure.figsize'] = (17, 6)

def plot_box(x):
    plt.subplot(121)
    sns.boxplot(y = x, x = 'Outcome', data = df)
    plt.title(x, fontsize = 16)
    
    plt.subplot(122)
    sns.violinplot(y = x, x = 'Outcome', data = df)
    plt.title(x, fontsize = 16)


# In[ ]:


plot_box('Age')


# In[ ]:


plot_box('Pregnancies')


# In[ ]:


plot_box('Insulin')


# In[ ]:


plot_box('BMI')


# In[ ]:


plot_box('BloodPressure')


# In[ ]:


plot_box('SkinThickness')


# In[ ]:


plot_box('DiabetesPedigreeFunction')


# In[ ]:


plot_box('Glucose')


# As we have just completed the bivariate analysis, the common thing we have noticed that there were many outliers present in the features. We'll treat them in the subsequent sections.

# ## Feature Transformation

# Let's first substitute the zero values in the features with np.NaN so that we can go ahead and simply impute the entries as the next step.

# In[ ]:


df['BMI'] = np.where(df['BMI'] == 0, np.nan, df['BMI'])
df['Glucose'] = np.where(df['Glucose'] == 0, np.nan, df['Glucose'])
df['BloodPressure'] = np.where(df['BloodPressure'] == 0, np.nan, df['BloodPressure'])
df['SkinThickness'] = np.where(df['SkinThickness'] == 0, np.nan, df['SkinThickness'])


# As we have converted them to null entries, let's impute them with fillna() method.

# In[ ]:


df['BMI'].fillna(27, inplace = True)
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)


# Let's check the summary statistics of the dataset again to verify whether the changes are applied.

# In[ ]:


df.describe()


# We'll drop the entries which have abnormal entries. We'll have a threshold for these features and drop the observations which exceed them. Although dropping entries should not be the first option, I'm doing it as there are not many observations which we would lose by doing so.

# In[ ]:


df.loc[df['SkinThickness'] > 60]


# In[ ]:


df.drop(df.loc[df['SkinThickness'] > 60].index, axis = 0, inplace = True)


# In[ ]:


df.loc[df['BMI'] > 55]


# In[ ]:


df.drop(df.loc[df['BMI'] > 55].index, axis = 0, inplace = True)


# In[ ]:


df.describe()


# We have did some changes and they definitely would have influenced the skewness of the feature. We'll use the skew() method to check the skewness. Let's also write a function to plot the distribution of the features.

# In[ ]:


for i in df.select_dtypes(['int64', 'float64']).columns.tolist():
    print(i, ':', df[i].skew())


# In[ ]:


def skew_visual():    
    plt.rcParams['figure.figsize'] = (20, 8)

    plt.subplot(241)
    sns.distplot(df['Pregnancies'], color = 'k')
    plt.title('PDF - Pregnancies')

    plt.subplot(242)
    sns.distplot(df['Glucose'], color = 'k')
    plt.title('PDF - Glucose')

    plt.subplot(243)
    sns.distplot(df['BloodPressure'], color = 'k')
    plt.title('PDF - BloodPressure')

    plt.subplot(244)
    sns.distplot(df['Insulin'], color = 'k')
    plt.title('PDF - Insulin')

    plt.subplot(245)
    sns.distplot(df['SkinThickness'], color = 'k')
    plt.title('PDF - SkinThickness')

    plt.subplot(246)
    sns.distplot(df['Age'], color = 'k')
    plt.title('PDF - Age')

    plt.subplot(247)
    sns.distplot(df['DiabetesPedigreeFunction'], color = 'k')
    plt.title('PDF - DiabetesPedigreeFunction')

    plt.subplot(248)
    sns.distplot(df['BMI'], color = 'k')
    plt.title('PDF - BMI')
    plt.tight_layout()


# In[ ]:


skew_visual()


# From the above plot, it is obvious that there are four features which are skewed. Let's check different transformations to see whether they can be fixed.

# In[ ]:


for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:
    print(i, ':', np.sqrt(df[i]).skew())


# In[ ]:


for i in ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']:
    print(i, ':', np.log1p(df[i]).skew())


# Don't we think it is better to have a dataframe with the feature names and the skewness factor for each of them. We can compare the skewness by appyling different transformations.

# In[ ]:


pd.DataFrame({'Feature' : ['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction'],
             'Actual' : [df[i].skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],
             'Squared' : [np.sqrt(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],
             'Cubed' : [(df[i] ** (1/3)).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]],
             'Logged' : [np.log1p(df[i]).skew() for i in df[['Pregnancies', 'Insulin', 'Age', 'DiabetesPedigreeFunction']]]})


# In[ ]:


df_v1 = df.copy()


# Let's save a copy of the dataset as we are going to transform the features. Let's create new features with the transformations applied and we can drop the redundant features later. We'll plot the distribution of the features with the new features to check h

# In[ ]:


df['Pregnancies_trans'] = np.sqrt(df['Pregnancies'])
df['Insulin_trans'] = np.log1p(df['Insulin'])
df['Age_trans'] = np.log1p(df['Age'])
df['DiabetesPedigreeFunction_trans'] = df['DiabetesPedigreeFunction'] ** (1/3)


# In[ ]:


def skew_visual_trans():    
    plt.rcParams['figure.figsize'] = (20, 8)

    plt.subplot(241)
    sns.distplot(df['Pregnancies_trans'], color = 'k')
    plt.title('PDF - Pregnancies')

    plt.subplot(242)
    sns.distplot(df['Glucose'], color = 'k')
    plt.title('PDF - Glucose')

    plt.subplot(243)
    sns.distplot(df['BloodPressure'], color = 'k')
    plt.title('PDF - BloodPressure')

    plt.subplot(244)
    sns.distplot(df['Insulin_trans'], color = 'k')
    plt.title('PDF - Insulin')

    plt.subplot(245)
    sns.distplot(df['SkinThickness'], color = 'k')
    plt.title('PDF - SkinThickness')

    plt.subplot(246)
    sns.distplot(df['Age_trans'], color = 'k')
    plt.title('PDF - Age')

    plt.subplot(247)
    sns.distplot(df['DiabetesPedigreeFunction_trans'], color = 'k')
    plt.title('PDF - DiabetesPedigreeFunction')

    plt.subplot(248)
    sns.distplot(df['BMI'], color = 'k')
    plt.title('PDF - BMI')
    plt.tight_layout()


# In[ ]:


skew_visual_trans()


# The transformations did not give the expected result, as we see some distortions. We'll try binning the values in those features to get rid of it.

# In[ ]:


df['Pregnancies_bin'] = np.where(df['Pregnancies'] == 0, 0,
                                np.where((df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 1,
                                        np.where((df['Pregnancies'] > 5) & (df['Pregnancies'] <= 10), 2, 3)))


# In[ ]:


df['Insulin_bin'] = np.where(df['Insulin'] == 0, 0,
                                np.where((df['Insulin'] > 0) & (df['Insulin'] <= 50), 1,
                                        np.where((df['Insulin'] > 50) & (df['Insulin'] <= 200), 2, 3)))


# In[ ]:


bins = np.arange(0, 100, 10)
names = [1, 2, 3, 4, 5, 6, 7, 8, 9]

df['Age_bin'] = pd.cut(df['Age'], bins = bins, labels = names)


# In[ ]:


df.sample(10)


# In[ ]:


df_v2 = df.copy()


# In[ ]:


df = df[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction_trans', 'Age_bin', 'Insulin_bin', 'Pregnancies_bin', 'Outcome']]


# In[ ]:


df.sample(10)


# In[ ]:


df['Outcome'] = df['Outcome'].astype('int64')


# ### Feature Scaling

# From the summary statistics, we have observed that the range of values in every feature differ. Let's apply the feature scaling techniques to avoid this.
# 
# There are different scaling techniques.
# * Normalization - This would scale the values between a minimum and maximum value. In most cases, it would be between 0 and 1.
# * Standardization - This technique is aimed at making the standard deviation of the feature to zero.
# 
# I'm not including the mathematical expressions for the above techniques. We can do this manually but let's go with the predefined methods available in Scikit - Learn's preprocessing module.

# In[ ]:


df_scaled_mms = df.copy()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()


# In[ ]:


cols = df_scaled_mms.columns.tolist()

df_scaled_mms = pd.DataFrame(mms.fit_transform(df), columns = cols)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


df_scaled_sc = pd.DataFrame(sc.fit_transform(df), columns = cols)


# In[ ]:


df_scaled_mms.describe()


# There are some points to note here.
# 
# * The minimum value of all the features is 0.
# * The maximum is at 1.
# * The mean is in the same range across the features.

# In[ ]:


df_scaled_sc.describe()


# The standardized dataset has the unit standard deviation (1). So we have applied two techniques and stored in different dataframes.

# Let's check the correlation matrix for each of the dataframes to check the correlation between the features.

# In[ ]:


plt.rcParams['figure.figsize'] = (10, 8)

sns.heatmap(df.corr() * 100, annot = True, cmap = 'coolwarm')
plt.title('Correlation - Before Scaling', fontsize = 16)
plt.show()


# In[ ]:


sns.heatmap(df_scaled_mms.corr() * 100, annot = True, cmap = 'plasma')
plt.title('Correlation - Normalized Data', fontsize = 16)
plt.show()


# In[ ]:


sns.heatmap(df_scaled_sc.corr() * 100, annot = True, cmap = 'Set1')
plt.title('Correlation - Standardized Data', fontsize = 16)
plt.show()


# From the heatmaps, we did not observe any difference in correlation among the dataframes. So let's go ahead with the normalized dataframe and include only the features which most correlate with the target variable.

# In[ ]:


df_scaled_mms.head()


# ## Model Building

# We'll import the required libraries for model building and the evaluation of the model.

# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Let's segregate the independant and dependant features.

# In[ ]:


X = df_scaled_mms.drop(columns = ['Insulin_bin', 'Outcome'])
Y = df_scaled_mms['Outcome']


# Then we'll split the dataset into train and test datasets.

# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.25, stratify = Y, random_state = 42)


# In[ ]:


print('Train_X - ', train_x.shape)
print('Test_X - ', test_x.shape)
print('Train_Y - ', train_y.shape)
print('Test_Y - ', test_y.shape)


# We'll write a function which takes a model name as an argument to fit, predict and evaluate the model.

# In[ ]:


def model_build(x):
    model = x
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print('Accuracy :', accuracy_score(test_y, pred))
    print('F1 :', f1_score(test_y, pred))
    kfold = KFold(n_splits = 5, random_state = 56)
    cv_res = cross_val_score(x, train_x, train_y, cv = kfold, scoring = 'accuracy')
    print('CV Accuracy Score - 5 Splits :', cv_res.mean())
    print('Precision :', precision_score(test_y, pred))
    print('Recall :', recall_score(test_y, pred))


# In[ ]:


model_build(LogisticRegression())


# In[ ]:


model_build(DecisionTreeClassifier())


# In[ ]:


model_build(RandomForestClassifier())


# In[ ]:


model_build(KNeighborsClassifier())


# In[ ]:


model_build(SVC())


# In[ ]:


model_build(GaussianNB())


# In[ ]:


rf = RandomForestClassifier()
rf.fit(train_x, train_y)
pred = rf.predict(test_x)


# As we've tried with various models, let's take the model with the good recall score and check the confusion matrix for the predictions.

# In[ ]:


pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')


# What do we observe?
# 
# * 105 normal people are identified as normal (TP)
# * 39 diabetic people are classified as diabetic (TN)
# * 19 normal people are identified as diabetic (FN)
# * 27 diabetic people are classified as normal (FP)

# Now the question arises, why did we chose the recall score as the evaluation metric?
# 
# As per the problem statement, the model should classify the diabetic people rightly. The impact would be more if diabetic people are classified as normal than the vice versa.
# 
# As FP has more influence, let's check the recall score of the model. Recall is the measure of the total 1's correctly classified out of the actual 1's.
# 
# Looking at the problem statement, our model should have high recall score.

# In[ ]:


print(rf.feature_importances_)


# In[ ]:


print(train_x.columns)


# Let's see a voting classifier would help us get high recall score than 59, which the Random Forest baseline model has scored.

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


estimators = [('DT', DecisionTreeClassifier()), ('RF', RandomForestClassifier())]
vc = VotingClassifier(estimators)
vc.fit(train_x, train_y)
pred = vc.predict(test_x)


# In[ ]:


print('Accuracy :', accuracy_score(test_y, pred))
print('Precision :', precision_score(test_y, pred))
print('Recall :', recall_score(test_y, pred))


# This did not turn fruitful as the combination of the models is putting the recall score down. We can try any combination using the Voting Classifier.

# In[ ]:


pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')


# Now, let's try to find the optimal parameters to be given to the Random Forest Classifier by trying out with different combinations of values.

# In[ ]:


depth = []
estimators = []
recall = []

for i in np.arange(2, 10):
    for j in np.arange(100, 300, 10):
        rf = RandomForestClassifier(max_depth = i, n_estimators = j, n_jobs = -1)
        rf.fit(train_x, train_y)
        pred = rf.predict(test_x)
        r = recall_score(test_y, pred)
        depth.append(i)
        estimators.append(j)
        recall.append(r)


# In[ ]:


res = pd.DataFrame({'Depth' : depth,
                   'Estimators' : estimators,
                   'Recall Score' : recall})
res.sort_values(by = 'Recall Score', ascending = False).head()


# In[ ]:


rf = RandomForestClassifier(max_depth = 9, n_estimators = 120, n_jobs = -1)
rf.fit(train_x, train_y)
pred = rf.predict(test_x)


# In[ ]:


pd.DataFrame(confusion_matrix(test_y, pred), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')


# From the above dataframe, max_depth of 9 and 120 estimators are giving a recall score of 59. I do not see any improvement. Let's try some other techniques.

# Let us adjust the thresholds by using the predict_proba method. 
# 
# We'll use three thresholds - 0.4, 0.5 and 0.6. The model would classify as 1, if the user defined threshold is exceeded. We'll compare the recall scores for different thresholds and conclude on the best one.

# In[ ]:


prob = rf.predict_proba(test_x)[:, 1]

probabilities = pd.DataFrame({'Probability' : prob,
             'P(0.4)' : '',
             'P(0.5)' : '',
             'P(0.6)' : ''})


# In[ ]:


probabilities['P(0.4)'] = np.where(probabilities['Probability'] > 0.4, 1, 0)
probabilities['P(0.5)'] = np.where(probabilities['Probability'] > 0.5, 1, 0)
probabilities['P(0.6)'] = np.where(probabilities['Probability'] > 0.6, 1, 0)


# In[ ]:


probabilities.head()


# I guess selecting 0.4 as the threshold would work out. Let's plot the confusion matrix for the same and check the recall score.

# In[ ]:


pd.DataFrame(confusion_matrix(test_y, probabilities['P(0.4)']), index = ["Actual 0's", "Actual 1's"], columns = ["Predicted 0's", "Predicted 1's"]).style.background_gradient(cmap = 'Set2')


# In[ ]:


recall_score(test_y, probabilities['P(0.4)'])


# Looking at the confusion matrix, the model is now classifying the actual 1's better than before. The recall score has improved from the earlier approaches.

# We can further improve the model by doing feature engineering(which requires the subject matter expertise) and bring in various ensemble techniques which are not covered in this kernel. 
# 
# Hope you enjoyed reading this notebook. Kindly upvote and leave a comment if you like my work. Please let me know your suggestions in the comments section. Thanks!

# In[ ]:




