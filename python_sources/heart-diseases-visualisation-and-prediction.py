#!/usr/bin/env python
# coding: utf-8

# **DIAGNOSING HEART DISEASES**

# In[ ]:


#importing libraries
                                             
import pandas as pd                                    # for dataframe
import numpy as np                                     # for numerical operations
from fancyimpute import KNN                            # for knn imputations
from scipy.stats import chi2_contingency               # for scientific calculations
import matplotlib.pyplot as plt                        # for visualisations
import seaborn as sns                                  # for visualisatons
from random import randrange,uniform                   # to generate random number
from sklearn.model_selection import train_test_split   # for implementing stratified sampling
from sklearn import tree                               # for implementing decision tree algorithm in data
from sklearn.tree import export_graphviz               #  plot tree
from sklearn.metrics import accuracy_score             # for implementing decision tree algorithm in data
from sklearn.metrics import confusion_matrix           # for calculating error metrics of various models
from sklearn.ensemble import RandomForestClassifier    # for implementing random forest model on data
import statsmodels.api as sn                           # for applying logistic model on data set
from sklearn.neighbors import KNeighborsClassifier     # for implementing knn model
from sklearn.naive_bayes import GaussianNB             # for implementing naive bayes
from sklearn import model_selection                    # for selecting model
from sklearn.metrics import classification_report,roc_auc_score,roc_curve # for model evaluation
from sklearn.metrics import classification_report      # for model evaluation
import pickle                                          # for saving the final modelimport seaborn as sns #for plotting
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  # for calculating VIF
from statsmodels.tools.tools import add_constant
np.random.seed(123) #ensure reproducibility
pd.options.mode.chained_assignment = None  #hide any pandas warnings


# Loading the data

# In[ ]:


hdata = pd.read_csv("../input/heart.csv")


# In[ ]:


hdata.head(10)


# It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
# 
# 1. age: The person's age in years
# 2. sex: The person's sex (1 = male, 0 = female)
# 3. cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 4. trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 5. chol: The person's cholesterol measurement in mg/dl
# 6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 8. thalach: The person's maximum heart rate achieved
# 9. exang: Exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 12. ca: The number of major vessels (0-3)
# 13. thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 14. target: Heart disease (0 = no, 1 = yes)
# To avoid  Hypothesizing  I'm going to take a look at online guides on how heart disease is diagnosed, and look up some of the terms above.
# 
# Diagnosis: The diagnosis of heart disease is done on a combination of clinical signs and test results. The types of tests run will be chosen on the basis of what the physician thinks is going on 1, ranging from electrocardiograms and cardiac computerized tomography (CT) scans, to blood tests and exercise stress tests 2.
# 
# Looking at information of heart disease risk factors led me to the following: high cholesterol, high blood pressure, diabetes, weight, family history and smoking 3. According to another source 4, the major factors that can't be changed are: increasing age, male gender and heredity. Note that thalassemia, one of the variables in this dataset, is heredity. Major factors that can be modified are: Smoking, high cholesterol, high blood pressure, physical inactivity, and being overweight and having diabetes. Other factors include stress, alcohol and poor diet/nutrition.
# 
# I can see no reference to the 'number of major vessels', but given that the definition of heart disease is "...what happens when your heart's blood supply is blocked or interrupted by a build-up of fatty substances in the coronary arteries", it seems logical the more major vessels is a good thing, and therefore will reduce the probability of heart disease.
# 
# Given the above, I would hypothesis that, if the model has some predictive ability, we'll see these factors standing out as the most important.
# 
# Let's change the column names to be a bit clearer,
# 

# In[ ]:


hdata.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# Exploratory Data Analysis.
# converting some variables to proper data types to improve interpretation

# In[ ]:


hdata['sex'][hdata['sex'] == 0] = 'female'
hdata['sex'][hdata['sex'] == 1] = 'male'

hdata['chest_pain_type'][hdata['chest_pain_type'] == 1] = 'typical angina'
hdata['chest_pain_type'][hdata['chest_pain_type'] == 2] = 'atypical angina'
hdata['chest_pain_type'][hdata['chest_pain_type'] == 3] = 'non-anginal pain'
hdata['chest_pain_type'][hdata['chest_pain_type'] == 4] = 'asymptomatic'

hdata['fasting_blood_sugar'][hdata['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
hdata['fasting_blood_sugar'][hdata['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

hdata['rest_ecg'][hdata['rest_ecg'] == 0] = 'normal'
hdata['rest_ecg'][hdata['rest_ecg'] == 1] = 'ST-T wave abnormality'
hdata['rest_ecg'][hdata['rest_ecg'] == 2] = 'left ventricular hypertrophy'

hdata['exercise_induced_angina'][hdata['exercise_induced_angina'] == 0] = 'no'
hdata['exercise_induced_angina'][hdata['exercise_induced_angina'] == 1] = 'yes'

hdata['st_slope'][hdata['st_slope'] == 1] = 'upsloping'
hdata['st_slope'][hdata['st_slope'] == 2] = 'flat'
hdata['st_slope'][hdata['st_slope'] == 3] = 'downsloping'

hdata['thalassemia'][hdata['thalassemia'] == 1] = 'normal'
hdata['thalassemia'][hdata['thalassemia'] == 2] = 'fixed defect'
hdata['thalassemia'][hdata['thalassemia'] == 3] = 'reversable defect'

hdata['target'][hdata['target'] == 0] = 'no'
hdata['target'][hdata['target'] == 1] = 'yes'


# In[ ]:


#Encoding Variable
#Assigning levels to the categories
lis = []
for i in range(0, hdata.shape[1]):
    if(hdata.iloc[:,i].dtypes == 'object'):
        hdata.iloc[:,i] = pd.Categorical(hdata.iloc[:,i])
        hdata.iloc[:,i] = hdata.iloc[:,i].cat.codes 
        hdata.iloc[:,i] = hdata.iloc[:,i].astype('object')
        lis.append(hdata.columns[i])


# In[ ]:


sns.countplot(x="target", data=hdata, palette="bwr")
plt.show()


# In[ ]:


countNoDisease = len(hdata[hdata.target == 0])
countHaveDisease = len(hdata[hdata.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(hdata.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(hdata.target))*100)))


# In[ ]:


countFemale = len(hdata[hdata.sex == 0])
countMale = len(hdata[hdata.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(hdata.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(hdata.sex))*100)))


# In[ ]:


hdata.groupby('target').mean()


# In[ ]:


pd.crosstab(hdata.age,hdata.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[ ]:


pd.crosstab(hdata.sex,hdata.target).plot(kind="bar",figsize=(15,6),color=['blue','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **Scatter plot** for thalassemia and cholesterol

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='cholesterol',y='thalassemia',data=hdata,hue='target')
plt.show()


# **Scatterplot **for thalassemia vs. resting_blood_pressure

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='thalassemia',y='resting_blood_pressure',data=hdata,hue='target')
plt.show()


# In[ ]:


plt.scatter(x=hdata.age[hdata.target==1], y=hdata.thalassemia[(hdata.target==1)], c="green")
plt.scatter(x=hdata.age[hdata.target==0], y=hdata.thalassemia[(hdata.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[ ]:


pd.crosstab(hdata.fasting_blood_sugar,hdata.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[ ]:


hdata.dtypes


# In[ ]:


#### Missing Value Analysis
hdata.isnull().sum()


# In[ ]:


hdata.head(10)


# In[ ]:


# checking statistical values of dataset
hdata.describe()


# In[ ]:


# store numeric variables in cnames
cnames=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels']


# **Outlier Analysis****

# In[ ]:


# Plot boxplot to visualise outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(hdata['resting_blood_pressure'])


# In[ ]:


# Detect outliers and replace with NA

for i in cnames:
    #print(i)
    q75,q25=np.percentile(hdata.loc[:,i],[75,25])  # extract quartiles 
    iqr=q75-q25                                         # calculate IQR
    minimum=q25-(iqr*1.5)                               # calculate inner and outer frames
    maximum=q75+(iqr*1.5)
    
    #print(minimum)
    #print(maximum)
    hdata.loc[hdata.loc[:,i] < minimum, i] = np.nan
    hdata.loc[hdata.loc[:,i] > maximum, i] = np.nan

    missing_value=pd.DataFrame(hdata.isnull().sum())   # calculating missing values


# In[ ]:


hdata=pd.DataFrame(KNN(k=3).fit_transform(hdata),columns=hdata.columns)  #performing knn imputation
hdata.isnull().sum()    


# **FEATURE SELECTION**

# In[ ]:


##Correlation analysis
#Correlation plot
df_corr = hdata.loc[:,cnames]
df_corr


# In[ ]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# In[ ]:


X = add_constant(hdata)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# In this data set all variables are equally important. No need to drop any variable from data set.

# In[ ]:


# Normality Check
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(hdata['chest_pain_type'],bins='auto')


# In[ ]:


plt.hist(hdata['age'],bins='auto')


# From this we can predict that data is not uniformly distributed so we will go for normalisation

# In[ ]:


#Normalisation
for i in cnames:
    #print(i)
    hdata[i]=(hdata[i]-np.min(hdata[i]))/(np.max(hdata[i])-np.min(hdata[i]))


# In[ ]:


hdata.head(10)


# **MODEL DEVELOPMENT**

# DECISION TREE

# Fitting of decision tree model to data set

# In[ ]:


# replace target variable  with yes or no
hdata['target'] = hdata['target'].replace(0, 'No')
hdata['target'] = hdata['target'].replace(1, 'Yes')


# In[ ]:


# to handle data imbalance issue we are dividing our dataset on basis of stratified sampling
# divide data into train and test
X=hdata.values[:,0:13]
Y=hdata.values[:,13]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[ ]:


# Decision tree - we will build the model on train data and test it on test data
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
# predict new test cases
C50_Predictions = C50_model.predict(X_test) # applying decision tree model on test data set


# In[ ]:


data1=hdata.drop(['target'],axis=1)


# In[ ]:


#Create dot file to visualise tree  #http://webgraphviz.com/
dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(C50_model, out_file=dotfile,feature_names=data1.columns)


# In[ ]:


# Confusion matrix of decision tree
CM = pd.crosstab(y_test, C50_Predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]


# In[ ]:


#check accuracy of model
accuracy=((TP+TN)*100)/(TP+TN+FP+FN)
accuracy


# In[ ]:


# check false negative rate of the model
fnr=FN*100/(FN+TP)
fnr


# **CLASSIFICATION REPORT**

# In[ ]:


print(classification_report(y_test,C50_Predictions))


# **RANDOM FOREST**

# Fitting of random forest model to dataset

# In[ ]:


RF_model = RandomForestClassifier(n_estimators = 700).fit(X_train, y_train)
RF_model


# Now we evaluate the model

# In[ ]:


# Apply RF on test data to check accuracy
RF_Predictions = RF_model.predict(X_test)
# To evaluate performance of any classification model we built confusion metrics
CM =pd.crosstab(y_test, RF_Predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]


# In[ ]:


#check accuracy of model
accuracy=((TP+TN)*100)/(TP+TN+FP+FN)
accuracy


# In[ ]:


# check  of the model
fnr=FN*100/(FN+TP)
fnr


# **CLASSIFICATION REPORT******

# In[ ]:


print(classification_report(y_test,RF_Predictions))


# **KNN MODEL**

# Let fit the knn model in dataset

# In[ ]:


# knn implementation
knn_model=KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)


# In[ ]:


# predict knn_predictions 
knn_predictions=knn_model.predict(X_test)


# In[ ]:


# build confusion metrics
CM=pd.crosstab(y_test,knn_predictions)
CM


# In[ ]:


# try K=1 through K=25 and record testing accuracy
k_range = range(1, 26)

# We can create Python dictionary using [] or dict()
scores = []
from sklearn import metrics
# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)


# In[ ]:


#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]


# In[ ]:


#check accuracy of model
accuracy=((TP+TN)*100)/(TP+TN+FP+FN)
accuracy


# In[ ]:


# check false negative rate of the model
fnr=FN*100/(FN+TP)
fnr


# **CLASSIFICATION REPORT**

# In[ ]:


print(classification_report(y_test,knn_predictions))


# **NAIVE BAYES**

# In[ ]:


# Naive Bayes implementation
NB_model=GaussianNB().fit(X_train,y_train)


# In[ ]:


# predict test cases 
NB_predictions=NB_model.predict(X_test)


# In[ ]:


# build confusion metrics
CM=pd.crosstab(y_test,NB_predictions)
CM


# In[ ]:


#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]


# In[ ]:


#check accuracy of model
accuracy=((TP+TN)*100)/(TP+TN+FP+FN)
accuracy


# In[ ]:


# check false negative rate of the model
fnr=FN*100/(FN+TP)
fnr


#  **CLASSIFICATION REPORT**

# In[ ]:


print(classification_report(y_test,NB_predictions))


# **COMPARING MODELS**

# In[ ]:


methods = [ "C50_model","RF_model","knn_model","NB_model"]
accuracy = [75.4,75.4,77.0, 73.7]
colors = ["purple", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.show()


# In[ ]:




