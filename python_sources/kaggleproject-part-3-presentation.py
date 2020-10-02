#!/usr/bin/env python
# coding: utf-8

# ----------
# **Kaggle Mini-Project III: Presentation**
# =====================================
# Maggie Maurer
# 
# CoderGirl, DataScience Cohort
# 
# July 2019
# 
# ----------

# ### <a id='1'>1. Please read in order.  I reference a lot of findings from part 1 and 2.  Thank you for all of your help this past 6 months!!</a> 

# - <a href='#1'>1. Introduction</a>  
#     - <a href='#1.2'>1.2. Feature Directory</a> 
# - <a href='#2'>2. Libraries and Data</a>  
#     - <a href='#2.1'>2.1. Loading libraries</a> 
#     - <a href='#2.2'>2.2. Reading data</a> 
# - <a href='#3'>3. Exploratory Data Analysis (EDA)</a> 
#     - <a href='#3.1'>3.1. Shape, Head, Describe</a> 
#     - <a href='#3.2'>3.2. Target Distribution</a> 
#     - <a href='#3.3'>3.3. Univariate analysis</a> 
#     - <a href='#3.4'>3.4 Bivariate analysis (Feature vs Target)</a> 
#     - <a href='#3.5'>3.5 Multivariate analysis</a> 
# - <a href='#4'>4. Predictive Modeling</a>
#     - <a href='#4.1'>4.1. K-Nearest Neighbors</a> 
#     - <a href='#4.2'>4.2. Random Forests</a> 
# - <a href='#5'>5. Conclusions</a>

# # <a id='1'>1. Introduction</a> 
# Of all the applications of machine-learning, diagnosing any serious medical disease using a predictive model is going to difficult. If the output of a preditive model is treatment, such as surgery or medication, or even the absence of treatment, people are going to want to know why the model predicted their particular course of action.
# 
# This dataset gives 13 features along with a target condition (the presence or absence of Heart Disease). Below, the data is explored using K-Nearest Neighbors and then investigated using Machiene Learning explainability tools and techniques.
# 
# This dataset was created by the following:
# * Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D. 
# * V.A. Medical Center, Long Beach and Cleveland Clinic Foundation, Robert Detrano, M.D., Ph.D.
# * University Hospital, Zurich, Switzerland: William Steinbrunn, M.D. 
# * University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.

# ## <a id='1.1'>1.1. Feature Directory</a> 
# It's a fairly simple dataset to understand set of data. However, due to the numeric qualifiers given to the categorical data, what the values mean is not obvious. 
# 
# 1.  age: The person's age in years
# 2.  sex: Provider-identified gender 
#     * 0 = Female
#     * 1 = Male
# 3. cpain: The type of chest pain experienced
#    * 0: Asymptomatic Pain
#    * 1: Typical Angina Pain
#    * 2: Atypical Angina Pain
#    * 3: Non-Angina Pain
# 4. resting_BP: Resting Systolic Blood Pressure (mm Hg) upon Hospital Admission
# 5. chol: Serum Cholesterol (mg/dl)
# 6. fasting_BS: Fasting Blood Sugar (mmol/L)
#    * 0: Lower than 120 mmol/L
#    * 1: Greater than 120 mmol/L
# 7. resting_EKG: Resting EKG Results
#    * 0: Normal EKG results
#    * 1: Showing probable or definite left ventricular hypertrophy by Estes' criteria
#    * 2: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 8. max_HR: Maximum Heart Rate Achieved (bpm)
# 9. 'exercise_ANG': Exercise Induced Angina (EIA)
#    * 0: No, they did not experience EIA
#    * 1: Yes, they experienced EIA
# 10. ST_depression: ST Depression (mm) Induced by Exercise Relative to Rest
# 11. ST_depressionAB: ST Depression Abnormalitiles
#     * 0: Normal-> The pts ST depression was 0
#     * 1: Abnormal-> The pts ST depression greater than 0
#     * This is a engineered feature.  If you would like to see how or why I engineered it, please examine [Part 1](https://www.kaggle.com/maurerm/kaggleproject-part-1-exploratory-data-analysis) of this Kaggle Assignment.
# 12. m_exercise_ST: The Slope of the Peak Exercise ST Segment
#     * 0: Upsloping
#     * 1: Flat
#     * 2: Downsloping
# 13. no_maj_vessels: Number of Major Vessels (0-3) Colored by Flourosopy
#     * Either 0, 1, 2, 3, or 4
# 13. thal: Thalium Stress Test Result Results
#     * 1: Fixed defect
#     * 2: Normal
#     * 3: Reversible defect
# 14. target': Absence or Presence of Heart Disease
#    * 0: no heart disease
#    * 1: heart disease present

# # <a id='1'>1. Librairies and Data</a> 
# ## <a id='#1.1'>1.1. Loding Libraries</a> 

# In[ ]:


#Data Analysis Libraries
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import statistics as st
from scipy import stats
from scipy import interp
import statistics as st
import math
import os
from datetime import datetime
import itertools

#Visualization Libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set(color_codes=True)
from IPython.display import HTML
from IPython.display import display
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import HTML
from pdpbox import pdp, info_plots
import shap
shap.initjs()
def multi_table(table_list):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )

#sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score,confusion_matrix, classification_report, confusion_matrix, jaccard_similarity_score, f1_score, fbeta_score

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Imputer,MinMaxScaler, LabelEncoder

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, validation_curve, RandomizedSearchCV, cross_val_predict, StratifiedKFold

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn import datasets

#misc
from functools import singledispatch
import eli5
from eli5.sklearn import PermutationImportance
import shap
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings


# ## <a id='1.2'>1.2. Reading data</a> 

# In[ ]:


heart = pd.read_csv("../input/heart.csv")


# Change DataFrame to match preprocessing changes made in [Part 1](https://www.kaggle.com/maurerm/kaggleproject-part-1-exploratory-data-analysis) of this Kaggle Assignment.

# In[ ]:


heart2= heart.drop(heart.index[164])
heart2.columns=['age', 'sex', 'cpain','resting_BP', 'chol', 'fasting_BS', 'resting_EKG', 
                'max_HR', 'exercise_ANG', 'ST_depression', 'm_exercise_ST', 'no_maj_vessels', 'thal', 'target']

heart2['chol']=heart2['chol'].replace([417, 564], 240)
heart2['chol']=heart2['chol'].replace([407, 409], 249)

heart2['ST_depressionAB']=heart2['ST_depression'].apply(lambda row: 1 if row > 0 else 0)
heart2A=heart2.iloc[:,0:11]
heart2B=heart2.iloc[:,11:14]
heart2C=heart2.loc[:,'ST_depressionAB']
heart2C=pd.DataFrame(heart2C)
heart2C.head()
heart2 = pd.concat([heart2A, heart2C, heart2B], axis=1, join_axes=[heart2A.index])

heart2.loc[48, 'thal']=2.0
heart2.loc[281, 'thal']=3.0

PHD=heart2.loc[heart2.loc[:,'target']==1]
AHD=heart2.loc[heart2.loc[:,'target']==0]
heart2.head()


# To make the dataset a little more intuiative, I made a second dataset, heart3, where the numeric descriptors have been changed to words.  This will improve interpretation later on.

# In[ ]:


#heart3 (descriptive)
heart3=pd.DataFrame.copy(heart2)

heart3['sex']=heart3['sex'].replace([1, 0], ['Male', 'Female'])
heart3['cpain']=heart3['cpain'].replace([0, 1, 2, 3], ['Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-Angina'])
heart3['fasting_BS']=heart3['fasting_BS'].replace([1, 0], ['BS > 120 mg/dl', 'BS < 120 mg/dl'])
heart3['resting_EKG']=heart3['resting_EKG'].replace([0, 1, 2], ['Normal', 'Left Ventricular Hypertrophy', 'ST-T Wave Abnormality'])
heart3['exercise_ANG']=heart3['exercise_ANG'].replace([0, 1], ['Absent', 'Present'])
heart3['m_exercise_ST']=heart3['m_exercise_ST'].replace([0, 1, 2], ['Upsloping', 'Flat', 'Downsloping'])
heart3['thal']=heart3['thal'].replace([1, 2, 3], ['Fixed Defect', 'Normal', 'Reversible Defect'])
heart3['target']=heart3['target'].replace([0, 1], ['Absent', 'Present'])

heart3['chol']=heart3['chol'].replace([417, 564], 240)
heart3['chol']=heart3['chol'].replace([407, 409], 249)

heart3.loc[48, 'thal']="Normal"
heart3.loc[281, 'thal']="Reversible Defect"

PHD3=heart3.loc[heart3.loc[:,'target']=="Present"]
AHD3=heart3.loc[heart3.loc[:,'target']=="Absent"]
heart3.head()


# # <a id='2'>2. Exploratory Data Analysis (EDA)</a>  
# ## <a id='2.1'>2.1. Shape, Head, Describe</a> 

# In[ ]:


numrows= heart3.shape[0]
numcolumns=heart3.shape[1]
display(heart3.head(5), heart3.describe(), print("Number of Rows:", numrows),print("Number of Columns:", numcolumns))


# ## <a id='2.2'>2.2. Target Distribution</a> 

# In[ ]:


ax1 = sns.countplot(heart3['target'], palette="BuPu")
plt.title("Distribution of HD Diagnosis", size=30)
plt.ylabel("Frequency", labelpad=40, size=20)
plt.xlabel("HD Diagnosis", labelpad=40, size=20)


# ## <a id='2.3'>2.3. Univariate analysis</a> 

# ### <a id=''>Numerical data</a> 

# In[ ]:


fig = plt.figure(figsize=(20,20))


plt.subplot(3, 2, 1)
warnings.filterwarnings('ignore')
ax1 = sns.distplot(heart2['age'], kde=False, color='blueviolet')
ax1.set_xlabel("Age (yrs)")
second_ax1 = ax1.twinx()
second_ax1.yaxis.set_label_position("left")
sns.distplot(heart2['age'], ax=second_ax1, kde=True, hist=False, color='blue')
second_ax1.set_yticks([])
plt.title("Distribution of Age", size=15)
plt.ylabel("Frequency")

plt.subplot(3, 2, 2)
warnings.filterwarnings('ignore')
ax1 = sns.distplot(heart2['max_HR'], kde=False, color='blueviolet')
ax1.set_xlabel("Maximum HR (bpm)")
second_ax1 = ax1.twinx()
second_ax1.yaxis.set_label_position("left")
sns.distplot(heart2['max_HR'], ax=second_ax1, kde=True, hist=False, color='blue')
second_ax1.set_yticks([])
plt.title("Distribution of Maximum Heart Rate", size=15)
plt.ylabel("Frequency")

plt.subplot(3, 2, 3)
warnings.filterwarnings('ignore')
ax1 = sns.distplot(heart2['resting_BP'], kde=False, color='blueviolet')
ax1.set_xlabel("Resing Systolic BP (mm Hg)")
second_ax1 = ax1.twinx()
second_ax1.yaxis.set_label_position("left")
sns.distplot(heart2['resting_BP'], ax=second_ax1, kde=True, hist=False, color='blue')
second_ax1.set_yticks([])
plt.title("Distribution of Resting Systolic Blood Pressure", size=15)
plt.ylabel("Frequency")

plt.subplot(3, 2, 4)
warnings.filterwarnings('ignore')
ax1 = sns.distplot(heart2['ST_depression'], kde=False, color='blueviolet')
ax1.set_xlabel("ST Depression")
second_ax1 = ax1.twinx()
second_ax1.yaxis.set_label_position("left")
sns.distplot(heart2['ST_depression'], ax=second_ax1, kde=True, hist=False, color='blue')
second_ax1.set_yticks([])
plt.title("Distribution of Exercise Induced ST Depression", size=15)
plt.ylabel("Frequency")
plt.xlim(0,7)

plt.subplot(3, 2, 5)
warnings.filterwarnings('ignore')
ax1 = sns.distplot(heart2['chol'], kde=False, color='blueviolet')
ax1.set_xlabel("Serum Cholesterol (mg/dl)")
second_ax1 = ax1.twinx()
second_ax1.yaxis.set_label_position("left")
sns.distplot(heart2['chol'], ax=second_ax1, kde=True, hist=False, color='blue')
second_ax1.set_yticks([])
plt.title("Distribution of Serum Cholesterol",size=15)
plt.ylabel("Frequency")

plt.show()


# ### <a id=''>Categorical data</a> 

# In[ ]:


fig = plt.figure(figsize=(20,20))
plt.subplot(3, 3, 1)
sns.countplot(heart3['sex'], palette="BuPu")
plt.title("Gender Distribution", size=15)
plt.ylabel("Frequency")
plt.xlabel("Provider-Identified Gender")

plt.subplot(3, 3, 2)
sns.countplot(heart3['cpain'], palette="BuPu")
plt.title("Distribution of Chest Pain Type", size=15)
plt.ylabel("Frequency")
plt.xlabel("Chest Pain Description")

plt.subplot(3, 3, 3)
sns.countplot(heart3['fasting_BS'], palette="BuPu")
plt.title("Fasting Blood Sugar Distribution", size=15)
plt.ylabel("Frequency")
plt.xlabel("Level of Fasting BS (mmol/L)")

plt.subplot(3, 3, 4)
sns.countplot(heart3['resting_EKG'], palette="BuPu")
plt.title("Distribution of Resting EKG Results", size=15)
plt.ylabel("Frequency")
plt.xlabel("EKG Results")

plt.subplot(3, 3, 5)
sns.countplot(heart3['exercise_ANG'], palette="BuPu")
plt.title("Distribution of Exercise Induced Angina", size=15)
plt.ylabel("Frequency")
plt.xlabel("Exercise Induced Angina")

plt.subplot(3, 3, 6)
sns.countplot(heart3['m_exercise_ST'], palette="BuPu")
plt.title("Distribution of the ST Segment Slope", size=15)
plt.ylabel("Frequency")
plt.xlabel("Slope  (Peak Exercise)")

plt.subplot(3, 3, 7)
sns.countplot(heart3['ST_depressionAB'], palette="BuPu")
plt.title("ST Depression Abnormalities", size=15)
plt.ylabel("Frequency")
plt.xlabel("ST Depression Abnormalities")

plt.subplot(3, 3, 8)
sns.countplot(heart3['no_maj_vessels'], palette="BuPu")
plt.title("No. of Major Vessels Colored by Flouroscopy", size=15)
plt.ylabel("Frequency")
plt.xlabel("Number of Major Vessels")

plt.subplot(3, 3, 9)
sns.countplot(heart3['thal'], palette="BuPu")
plt.title("Thalium Stress Test Results", size=15)
plt.ylabel("Frequency")
plt.xlabel("Results")


# ## <a id='2.4'>2.4. Bivariate analysis (Feature vs Target)</a>

# ### <a id=''>Correlation</a>

# In[ ]:


mask = np.zeros_like(heart2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True 
plt.figure(figsize=(20,20))
sns.heatmap(heart2.corr(),vmax=.8, center=0,
            square=True, linewidths=.1, mask=mask, cbar_kws={"shrink": .5},annot=True)


# In[ ]:


heart2.corr()


# Let's look more specifically at the correlations between the features and the target. 
# 
# The following dataframe is organized in descending order of the absolute value of the correlation between the feature and the target.

# In[ ]:


corre=heart2.corr()
TargetCorr=corre.loc[:'thal','target']
TargetCorr=pd.DataFrame(TargetCorr)
TargetCorr['AbsVal']=TargetCorr['target'].apply(lambda row: abs(row))
TargetCorr['Rank']=pd.DataFrame.rank(TargetCorr['AbsVal'])
TargetCorr['Feature']=TargetCorr.index
TargetCorr = TargetCorr.set_index('Rank') 
TargetCorr = TargetCorr.sort_index(ascending=0)
TargetCorr = TargetCorr.set_index('Feature') 
TargetCorr=TargetCorr.loc[:,'target']
TargetCorr=pd.DataFrame(TargetCorr)
TargetCorr.columns=["Correlation with Target"]
TargetCorr


# ### <a id=''>Paired T-Test</a>
# 
# I am only going to run the paired T-Test on the quantitative features.  As the discrete feautes have, at maximum, 5 discrete values, comparing their means would not provide meaningful information.  

# In[ ]:


PHD=heart2.loc[heart2.loc[:,"target"]==1]
AHD=heart2.loc[heart2.loc[:,"target"]==0]

from scipy.stats import ttest_ind
def rowz(ttest): 
    name=ttest_ind(PHD[ttest], AHD[ttest])
    name=list(name)
    name = pd.DataFrame(np.array(name))
    name=name.T
    col=["t-statistic", "p_value"]
    name.columns=col
    return name

AGE=rowz('age')
AGE.loc[:,"Names"]="Age"
RESTING_BP=rowz('resting_BP')
RESTING_BP.loc[:,"Names"]="Resting_BP"
CHOLESTEROL=rowz('chol')
CHOLESTEROL.loc[:,"Names"]="Cholesterol"
MAX_HR=rowz('max_HR')
MAX_HR.loc[:,"Names"]="Max_HR"
ST_DEP=rowz('ST_depression')
ST_DEP.loc[:,"Names"]="ST_Depression"

PVALS = pd.concat([AGE, RESTING_BP,CHOLESTEROL,MAX_HR, ST_DEP], axis=0)
PVALS=PVALS.set_index(PVALS["Names"])
P_VALS= PVALS.drop('Names',axis=1)

P_VALS


# As we can see, almost all of the p-values are significant (<0.05).
# * ST_Depression: 0.000000000000005815
# * Maximum Heart Rate: 0.00000000000002476
# * Age: 0.001039
# * Resting Blood Pressure: 0.010927
# 
# The only non-significant p-value is cholesterol (0.07985).
# 
# 
# This means that for ST depression, maximum heart rate, age, and resting blood pressure there is less than a 5% chance that the differences between the target sample^ means could have occured by chance  alone.
#    
#    ^ Target Sample: Absence or presence of heart disease
# 

# For more in-depth bivariate analysis, please go to [Part 1](https://www.kaggle.com/maurerm/kaggleproject-part-1-exploratory-data-analysis) of this Kaggle Assignment.

# ## <a id='2.5'>2.5. Multivariate analysis</a> 

# In[ ]:


sns.pairplot(heart2,vars = ['resting_BP', 'chol','max_HR','ST_depression', 'age'],hue='target')


# ## <a id='3'>3. Predictive Modeling</a> 

# ## <a id='3.1'>3.1. K-Nearest Neighbors</a> 

# In [Part 2](https://www.kaggle.com/maurerm/kaggleproject-part-2-modeling) of this Kaggle Assignment I tested 6 predictive models Logisitic Regression, K-Nearest Neighbors, Decision Tree, AdaBoost with a Decision Tree Base, and XGBoost).
# 
# I am first examining K-Nearest Neighbors because the model fitted by GridSearchCV had high measures of accuracy, even in the face of cross validation.  

# In[ ]:


#seperate independent (feature) and dependent (target) variables
#KNN cannot process text/ categorical data unless they are be converted to numbers
#For this reason I did not input the heart3 DataFrame created above
X=heart2.drop('target',1)
y=heart2.loc[:,'target']

#Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=.2,random_state=40)

#Call classifier and, using GridSearchCV, find the best parameters
knn = KNeighborsClassifier()
params = {'n_neighbors':[i for i in range(1,33,2)]}
modelKNN = GridSearchCV(knn,params,cv=10)
modelKNN.fit(X_train,y_train)
modelKNN.best_params_   

#Use the above model (modelKNN) to predict the y values corresponding to the X testing set
predictKNN = modelKNN.predict(X_test)

#Compare the results of the model's predictions (predictKNN) to the actual y values
accscoreKNN=accuracy_score(y_test,predictKNN)
print('Accuracy Score: ',accuracy_score(y_test,predictKNN))
print('Using k-NN we get an accuracy score of: ',
      round(accuracy_score(y_test,predictKNN),5)*100,'%')


# Permutation importance is a great tool for understanding the affects of features on the Machiene Learing model.  Specifically, after a model has been fit, it shuffels individual variables in the validation data and looks at their the effect on accuracy. 

# In[ ]:


perm = PermutationImportance(modelKNN).fit(X_test, y_test)
eli=eli5.show_weights(perm, feature_names = X.columns.tolist())
eli


# Let's take a closer look at the numerical features using a Partial Dependence Plot. 
# 
# These plots change a single variable in a single row across a range of values and calculate the effect those changes have on the outcome. It does this for several rows and plots the average effect.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'no_maj_vessels'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.ylim(-0.015,0.01)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# So as the number of major blood vessels increases, the probability of heart disease decreases. 
# 
# That makes sense, as the more major blood vessels that are colored indicates more bloodflow to the heart. 
# However, the blue confidence are very large and show that this might not be true.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'age'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
#plt.ylim(-0.025,0.01)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# Interestingly, it appears as if the likelyhood of Heart Disease goes down with age.  I wonder why. 

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'chol'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
#plt.ylim(-0.025,0.01)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# Interestingly, it appears as if the likelyhood of Heart Disease goes down with cholesterol levels.  Hmm...

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'ST_depression'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
#plt.ylim(-0.025,0.01)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# ST_depression doesn't seem to affect the liklihood of Heart Disease.  

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'max_HR'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.ylim(-0.005,0.2)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# It appears that as the maximum heart rate increases, so does the likelihood of a Heart Disease diagnosis. 

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)
X_test=pd.DataFrame(X_test)
X_test

base_features = X.columns.values.tolist()

feat_name = 'resting_BP'

pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.ylim(-0.005,0.2)
#plt.xticks(np.arange(0, 4, step=1))
plt.show()


# Resting Blood Pressure also doesn't give a lot of weight to this model, but it dose appear that the lowest likelihood of Heart Disease is from 120-140 systolic blood pressure, which is still high.  But the more extreme values seem more, albiet slighlty, associated with Heart Disease. 

# ## <a id='3.2'>3.2. Random Forest</a> 

# I am using Random Forest because my model in [Part 2](https://www.kaggle.com/maurerm/kaggleproject-part-2-modeling) of this Kaggle Assignment had decent accuracy values and because the Random Forest has the ability to rank the important.  In fact, Random Forests are often used for feature selection in a data science workflow.

# In[ ]:


X= heart2.drop('target',1)
y= heart2['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=.3,random_state=40)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# In[ ]:


feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp


# In[ ]:


# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()                 


# # <a id='4'>4. Conclusions</a> 

# * Data indicates that as the number of major blood vessels increases, the probability of heart disease decreases. However, the blue confidence intervals are quite large, so this may not be accurate. 
# * There appears that the likelyhood of a Heart Disease diagnosis goes down as age and cholesterol increase.
# * According to the Partial Dependence Plot, ST depression doesn't seem to affect the liklihood of Heart Disease.  However, it is ranked high as an important feature using the Random Forest model.
# * It appears that as the maximum heart rate increases, so does the likelihood of a Heart Disease diagnosis. 
# * Interestingly, it dose appear that the lowest likelihood of Heart Disease is from 120-140 systolic blood pressure, which is still high.  But the more extreme values seem more, albiet slighlty, associated with Heart Disease.  
