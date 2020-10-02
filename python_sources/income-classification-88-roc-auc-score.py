#!/usr/bin/env python
# coding: utf-8

# # Background

# This dataset was obtained from UCI Machine Learning Repository. The aim of this problem is to classify adults in two different groups based on their income where group 1 has an income less than USD 50k and group 2 has an income of more than or equal to USD 50k. The data available at hand comes from Census 1994

# The variables present in the dataset are as follows - 
# 
# - age: continuous.
# - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - fnlwgt: continuous.
# - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - education.num: continuous.
# - marital.status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - sex: Female, Male.
# - capital-gain: continuous.
# - capital-loss: continuous.
# - hours-per-week: continuous.
# - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 

# # Part 1: Data and Libraries Setup

# In[ ]:


#Importing basic libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Silencing warnings 

import warnings
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment',None) #Silencing the Setting with Copying Warning


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


filepath = '../input/adult-census-income/adult.csv'
df = pd.read_csv(filepath)


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Part 2: Data Pre-Processing

# In[ ]:


df.describe()


# **Distribution of Target Variable:**
# 
# 76% of the population earns below 50K USD which indicates an unequal distribution of the target variable. We will have to consider this while training our ML model

# In[ ]:


# df.rename(columns={'income':'salary'},inplace=True)


# In[ ]:


df['income'].value_counts()


# In[ ]:


df['income'].value_counts(normalize=True)


# **Identifying categorical and numerical variables separately**
# 
# This will help us in using the suitable type of plots to visualise each variable later during EDA

# In[ ]:


cols_df = pd.DataFrame(df.dtypes)
num_cols = list(cols_df[cols_df[0]=='int64'].index)
cat_cols = list(cols_df[cols_df[0]=='object'].index)[:-1] #excluding target column of income 
print('Numeric variables includes:','\n',num_cols)
print('\n')
print('Categorical variables includes','\n',cat_cols)


# **Treating Missing Values**

# In[ ]:


# Data description mentions that unknown values are repalced with '?'. Let us check counts for it

for i in list(df.columns):
    print(i,df[df[i]=='?'][i].count())


# In[ ]:


#Replacing '?' string with NaN values
df.replace(to_replace='?',value=np.nan,inplace=True)


# In[ ]:


df.isna().sum()


# All 3 variables with missing values are categorical in nature. As we have a larger number of available observations, let us drop these observations and use rest of the data for analysis

# In[ ]:


df.dropna(axis=0,inplace=True)


# # Part 3: Exploratory Data Analysis:
# 
# This section is split into 2 parts where visualisations are bifurcated depending on the variable type - numeric and categorical

# ## Numeric Variables

# **1. Correlation between numberic variables:** There is no strong positive or negative correlation between these variables. Thus we are unlikely to run into any multi-collinearity issues in the model

# In[ ]:


plt.figure(figsize=(5,5))
correlation = df.corr()
matrix = np.triu(correlation)
sns.heatmap(correlation,cmap='coolwarm',square=True,linecolor='black',linewidths=1,
            mask=matrix,annot=True)
plt.show()


# **2. Histograms for each variable:** 
# 
# - Younger population falls under the lower income category whereas the age distribution for above 50K income is normally distributed
# - The mean observation of education level is higher for the above 50k income group with most observations falling above education level of 9 
# - Capital gain and capital loss have largely zero values hence we will exclude it from our feature matrix
# - Also, fnlwgt by definition is a control parameter kept in the census data. We may need to exclude it later from our model depending on it's impact on performance
# - Majority of lower income group individuals work for less than 40 hours whereas the higher income group individuals have a more right skewed distribution of working hours  

# In[ ]:


for i in num_cols:
    plt.figure(figsize=(6,3))
    df[df['income']=='<=50K'][i].hist(color='mediumblue')
    df[df['income']=='>50K'][i].hist(color='firebrick')
    plt.title(i)
    plt.show()


# ## Categorical Variables

# **Count plot observations**: We are only looking at top 6 categories for each variable on different salary levels and total level (for benchmarking our observations) 
# 
# - Leading attributes of <=50K income group:
#     - No clear distinction on the workclass since both groups are majorly employed in the private sector
#     - High school or some college graduates
#     - Unmarried (could be partly due to younger age)
#     - Common occupation includes craft/repair, admin-clerical and other services (may be unorganised in nature)
#     - Equal proportion of individuals who are not in a family setting and are husbands
#     - Race distribution is similar to overall population distribution for this group 
#     - Gender and native country follow a pattern similar to the entire population for both the groups (i.e. largely males from USA itself) 
# 
# 
# - Leading attributes of >50K income group:
#     - Largely employed in the private sector
#     - More frequenct bachlor degree holders (in line with our observation on higher education level)
#     - Significant proportion is married and living with a spouse
#     - These individuals hold highly specialised positions in the workforce like executive management and prof-specialty (which partly explain the higher income)
#     - Most play an active family role as a husband or wife
#     - Whites dominate this income group, however it should be noted this is also the general population distribution

# In[ ]:


# Instead of using sns.catplot() we use the below loop to create a cross tab 
# which will also include the total column for better comparison

for i in cat_cols:
    ct = pd.crosstab(df[i],df['income'],margins=True, margins_name="Total")
    ct.drop(labels='Total',axis=0,inplace=True) #Removing subtotal row 
    ct.sort_values(by='Total',ascending=False,inplace=True) #Sorting based on total column
    #Selecting only top 6 categories for plotting
    ct.iloc[:6,:].plot(kind='bar',colormap='viridis',edgecolor='black')  
    plt.xlabel(' ')
    plt.title(str(i).capitalize())
    plt.legend(loc=1)
    plt.show()


# # Part 4: Feature Selection

# In[ ]:


df['income'].replace(to_replace='<=50K',value=0,inplace=True)
df['income'].replace(to_replace='>50K',value=1,inplace=True)


# In[ ]:


#Identifying categorical columns where more than 90% of observations belong only to one categroy

cat_drop = []
for i in cat_cols:
    if (df[i].value_counts(normalize=True)[0]) > 0.9:
        cat_drop.append(i)
        
print(cat_drop)


# In[ ]:


#Similarly for numerical columns

num_drop = []
for i in num_cols:
    if df[i].value_counts(normalize=True).iloc[0] > 0.9:
        num_drop.append(i)
        
print(num_drop)


# 
# - In native country close to 91% of observations belong to one category (i.e. United States)
# - Whereas in capital-gain and capital-loss, ~92% and ~95% of values are zeroes respectively which will not contribute in modelling
# 
# Hence, we **drop these 3 variables (native-country, capital-gain, capital-loss)** from the feature matrix

# In[ ]:


X = df.drop(labels = cat_drop + num_drop + ['income'],axis=1)
y = df['income']


# In[ ]:


X.head(2)


# In[ ]:


y.value_counts()


# Let us take a look at education-num and education as these variables are largely similar in nature

# In[ ]:


sns.boxplot('education.num','education',data=df)
plt.show()


# In[ ]:


ed_cross = pd.crosstab(df['education.num'],df['education'])


# In[ ]:


ed_cross


# Above cross tab shows that each education number corresponds to an education level and represents the same data. Hence, we **drop the education column** from our feature matrix to avoid redundant data 

# In[ ]:


X.drop('education',axis=1,inplace=True)


# We also **drop fnlwgt column** as it is only a standardisation parameter and fails to give us any direction during model interpretation

# In[ ]:


X.drop('fnlwgt',axis=1,inplace=True)


# # Part 5: Feature Engineering

# In this section, we will attempt to simplify our input parameters (features) as we have many categorical variables with higher number of unique values 

# **1. Workclass**
# 
# 74% of individuals work in the private sector. Let us consolidate all other categories under a common head of 'Non-Private'

# In[ ]:


X['workclass'].value_counts(normalize=True)*100


# In[ ]:


#Listing all options other than private
to_replace = list(X['workclass'].unique())
to_replace.remove('Private')

#Placing all other categories under one bracket
X.replace(to_replace,'Non-Private',inplace=True)
X['workclass'].value_counts(normalize=True)*100


# **2. Race**
# 
# 86% of individuals are whites hence let us club all other races under a common 'Other' bracket to reduce the number of categories

# In[ ]:


X['race'].value_counts(normalize=True)*100


# In[ ]:


#Listing all options other than white
to_replace = list(X['race'].unique())
to_replace.remove('White')

#Placing all other categories under one bracket
X.replace(to_replace,'Other',inplace=True)
X['race'].value_counts(normalize=True)*100


# **3. Marital Status**

# In[ ]:


X['marital.status'].value_counts(normalize=True)*100


# In[ ]:


#Let us consolidate all options where individuals were married at least once (i.e. all options other than never-married)
to_replace = list(X['marital.status'].unique())
to_replace.remove('Never-married')

#Placing all other categories under one bracket
X.replace(to_replace,'Married',inplace=True)

#Renaming the 'Never-married' category to 'Single'
X.replace('Never-married','Single',inplace=True)

#Checking the final output
X['marital.status'].value_counts(normalize=True)*100


# # Part 6: Encoding, Splitting and Scaling

# **1. Encoding categorical data**

# In[ ]:


X.head(2)


# In[ ]:


X.shape


# In[ ]:


#Separating the categorical variables in feature matrix that need to be encoded 

cols_X = pd.DataFrame(X.dtypes)
X_cat_cols = list(cols_X[cols_X[0]=='object'].index)
X_num_cols = list(cols_X[cols_X[0]=='int64'].index)


# In[ ]:


X_num_cols


# In[ ]:


X = pd.get_dummies(data=X,prefix=X_cat_cols,drop_first=True)


# In[ ]:


X.head(2)


# **2. Splitting data into training and validation sets**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.25,random_state=101)


# **3. Scaling numeric variables in the feature matrix**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X_train.head(2)


# In[ ]:


X_train[X_num_cols] = sc.fit_transform(X_train[X_num_cols])
X_val[X_num_cols] = sc.transform(X_val[X_num_cols])


# In[ ]:


X_train.head()


# After categorical encoding, our final feature matrix has 25 columns. We will use this for our model training in the next section 

# # Part 7: Classification Model Training 

# In[ ]:


from sklearn import __version__ 
print(__version__)
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, log_loss, brier_score_loss


# **1. Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=101)
log.fit(X_train,y_train)
log_y_pred = log.predict_proba(X_val)
log_roc = roc_auc_score(y_val,log_y_pred[:,-1])
print('ROC AUC score : ',log_roc)
print(log.get_params())


# In[ ]:


d = {'Baseline Logistic Regression' : [log_roc]}

results = pd.DataFrame(d,index=['ROC AUC Score'])
results = results.transpose()
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **2. Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=101)
dt.fit(X_train, y_train)
dt_y_pred = dt.predict_proba(X_val)
dt_roc = roc_auc_score(y_val,dt_y_pred[:,-1])
print('ROC AUC score : ',dt_roc)
print(dt.get_params())


# In[ ]:


results.loc['Baseline Decision Tree'] = dt_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **3. Random Forest Classifier** 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=101)
rf.fit(X_train,y_train)
rf_y_pred = rf.predict_proba(X_val)
rf_roc = roc_auc_score(y_val,rf_y_pred[:,-1])
print(rf_roc)
print(rf.get_params())


# In[ ]:


results.loc['Baseline Random Forest'] = rf_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **4. Support Vector Classifier - Linear Kernel** 

# In[ ]:


#Linear kernel

from sklearn.svm import SVC

svc_l = SVC(kernel='linear',random_state=101,probability=True)
svc_l.fit(X_train,y_train)
svcl_y_pred = svc_l.predict_proba(X_val)
svcl_roc = roc_auc_score(y_val,svcl_y_pred[:,-1])
print('ROC AUC score : ',svcl_roc)
print(svc_l.get_params)


# In[ ]:


results.loc['Baseline SVC Linear'] = svcl_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **5. K-Nearest Neighbours**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train,y_train)
knn_y_pred = knn.predict_proba(X_val)
knn_roc = roc_auc_score(y_val,knn_y_pred[:,-1])
print('ROC AUC score : ',knn_roc)
print(knn.get_params())


# In[ ]:


results.loc['Baseline KNN'] = knn_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **6. Naive Bayes Classifier**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_y_pred = nb.predict_proba(X_val)
nb_roc = roc_auc_score(y_val,nb_y_pred[:,-1])
print('ROC AUC score : ',nb_roc)
print(nb.get_params())


# In[ ]:


results.loc['Baseline Naive Bayes'] = nb_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **7. XGBoost Classifer**

# In[ ]:


from xgboost import XGBClassifier
xg = XGBClassifier(random_state=101)
xg.fit(X_train,y_train)
xg_y_pred = xg.predict_proba(X_val)
xg_roc = roc_auc_score(y_val,xg_y_pred[:,-1])
print('ROC AUC score : ',xg_roc)
print(xg.get_params())


# In[ ]:


results.loc['Baseline XGBoost'] = xg_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# **8. Cat Boost**

# In[ ]:


from catboost import CatBoostClassifier
cat = CatBoostClassifier(silent=True)
cat.fit(X_train,y_train)
cat_y_pred = cat.predict_proba(X_val)
cat_roc = roc_auc_score(y_val,cat_y_pred[:,-1])
print('ROC AUC score : ',cat_roc)
print()
print(cat.get_all_params())


# In[ ]:


results.loc['Baseline Cat Boost'] = cat_roc
results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)
results


# # Part 8: Conclusion

# **Top model:**
# 
# - From the above list we can see that CatBoost model performs the best
# - All other models also have a very close performance metric indicating that due to clean data, we are able to achieve decent model performance across models
# - **Hence, our final model obtained using CatBoost is able to predict the income group of the test data with an ROC AUC score of 88.4%**

# Note:
# - I have worked on this project as my first machine learning classification project hence there may certainly be room for improvement further. Please feel free to leave your comments and suggestions which may help me to improve further
# - This project can also be found on my GitHub profile (along with test data predictions) - https://github.com/ishitashah23/learningprojects/tree/master/Income-category%20Classification%20Project
# - Big shoutout to Shrey Bohara for taking up this project along with me. You can check his notebook solution here - https://www.kaggle.com/shreybohara/beginner-s-income-prediction-89-score 
