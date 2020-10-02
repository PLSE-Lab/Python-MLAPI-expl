#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Providing a passingmark criteria which will be used to categorize the students
passmarks = 40 
#Reading the Data from a local repo in the system
df = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


# Let's Get some max,min,standard deviation for the Data Frame
df.describe()


# In[ ]:


#Also let's check for any missing values if in Data Set
df.isnull().sum()
# We find that no such missing values are there which will not be the case everytime. 
#But for Now since there are None....Let's continue with it.


# In[ ]:


# Let Us Explore Math Score at First Instance# Let U 
p = sns.countplot(x="math score" , data=df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 90)


# In[ ]:


# Let's find the number of students Passed and failed according to the passing Score:
df['MathPassingStatus'] = np.where(df['math score'] < passmarks , 'Failed!' , 'Passed!')
df.MathPassingStatus.value_counts()


# In[ ]:


#Let's Plot a Graph for Passed Students:
p = sns.countplot(x='parental level of education' , data = df , hue = 'MathPassingStatus' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(), rotation = 90)
#Here we plot the graph in context to the Parental level of Education and depending upon that, showing the Number of Students passed or failed.


# In[ ]:


#Now exploring the Writing Score:
p= sns.countplot(x = "writing score" , data = df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 90)


# In[ ]:


#Here We are Analyzing on the Attribute of Lunch:#Here We 
p = sns.countplot(x='lunch' , data =df , hue = 'MathPassingStatus' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 90)


# In[ ]:


#Similarly going for race/ethnicity:
p = sns.countplot(x='race/ethnicity' , data = df , hue = 'MathPassingStatus' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(), rotation = 90)


# In[ ]:


# Now students passing the Writing Exam:
df['WritingPassingStatus'] = np.where(df['writing score']<passmarks , 'Failed!','Passed!')
df.WritingPassingStatus.value_counts()


# In[ ]:


#Plot for the Passed or failed, and seeing the Variation w.r.t Parental Level of Education:
p = sns.countplot(x='parental level of education' , data = df, hue = 'WritingPassingStatus', palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation =90)


# In[ ]:


#Now exploring the Writing Score:
p= sns.countplot(x = "writing score" , data = df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 65)


# In[ ]:


#Here We are Analyzing on the Attribute of Lunch:
p = sns.countplot(x='lunch' , data =df , hue = 'WritingPassingStatus' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 65)


# In[ ]:


#Similarly going for race/ethnicity:#Similar 
p = sns.countplot(x='race/ethnicity' , data = df , hue = 'WritingPassingStatus' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(), rotation = 60)


# In[ ]:


# Similarly for the Reading Score:
p=sns.countplot(x="reading score" , data =df,palette = "muted")
plt.show()


# In[ ]:


# Number of Students Passed??
df['ReadingPassStatus'] = np.where(df['reading score'] < passmarks , 'Failed!' , 'Passed!')
df.ReadingPassStatus.value_counts()


# In[ ]:


#Finding % of Marks:
df['Total_Marks'] = df['math score'] + df['reading score'] + df['writing score']
df['Percent'] = df['Total_Marks']/3


# In[ ]:


#Let us Check how many Students totally passed in All Subjects:
df['OverAllPassingStatus'] = np.where(df.Total_Marks < 215 , 'Failed' , 'Passed!')
df.OverAllPassingStatus.value_counts()


# In[ ]:


p =  sns.countplot(x="Percent" , data = df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 0)


# In[ ]:


#Let us do the grading for the students now:
def GetGrade(Percent,OverAllPassingStatus):
    if(OverAllPassingStatus == 'Failed!'):
        return 'Failed'
    if(Percent >= 80):
        return 'A'
    if(Percent >= 70):
        return 'B'
    if(Percent >= 60):
        return 'C'
    if(Percent >= 50):
        return 'D'
    if(Percent >= 40):
        return 'E'
    else:
        return 'Failed!'


# In[ ]:


df['Grade'] = df.apply(lambda x: GetGrade(x['Percent'], x['OverAllPassingStatus']),axis =1)
df.Grade.value_counts()


# In[ ]:


#Plotting Grades in an Obtained Order
sns.countplot(x="Grade" , data=df,order = ['A','B','C','D','E','F'] , palette = "muted")
plt.show()


# In[ ]:


#Plotting with variation of Perental Education:
p = sns.countplot(x='parental level of education', data=df,hue='Grade',palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 30)


# In[ ]:


#Lunch Variation
p = sns.countplot(x='lunch', data=df,hue='Grade',palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 30)


# In[ ]:


#Test Prep Course Variation
p = sns.countplot(x='test preparation course', data=df,hue='Grade',palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 30)


# In[ ]:


#Race/Ethnicity Variation
p = sns.countplot(x='race/ethnicity', data=df,hue='Grade',palette = 'bright')
_ = plt.setp(p.get_xticklabels(),rotation = 30)


# **Now We will Apply Several Machine Learning Algo's based on Understanding and will see the variation of each Algorithm**
# Being a starter I am trying to bring all the attained and practiced knowledge towards the implementation for this Data Set..!!

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#Getting Data Points for All Scores availbale
ML_DataPoints = pd.read_csv(filepath_or_buffer = "../input/StudentsPerformance.csv",header = 0,
                           usecols = ['math score','reading score','writing score'])
#Getting Test Prep Course Values
ML_Labels = pd.read_csv(filepath_or_buffer = "../input/StudentsPerformance.csv",header = 0,usecols=['test preparation course'])

#Load MinMaxScaler
MNScaler = MinMaxScaler()
MNScaler.fit(ML_DataPoints) #Fitting the Scores
T_DataPoints = MNScaler.transform(ML_DataPoints) #Transform the Scores


#Load Label Encoder#Load La 
LEncoder = LabelEncoder()
LEncoder.fit(ML_Labels)
T_Labels = LEncoder.transform(ML_Labels)

#Split the DATA SET
XTrain,XTest,YTrain,YTest = train_test_split(T_DataPoints,T_Labels,random_state=10)

#Apply Random Forest Classifier:
RandomForest = RandomForestClassifier(n_estimators = 10,random_state=5)

RandomForest.fit(XTrain,YTrain)


# In[ ]:


RandomForest.fit(XTest,YTest)


# In[ ]:


RandomForest.score(XTrain,YTrain)


# In[ ]:


RandomForest.score(XTest,YTest)
#We see the Model is Underfitting..!!


# **Using Logistic Regression..!!**

# In[ ]:


model_now = LogisticRegression()
model_now.fit(XTrain,YTrain)


# In[ ]:


y_pred = model_now.predict(XTest)


# In[ ]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(YTest,y_pred)
print(ac)


# **Using SVM**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()
model.fit(XTrain,YTrain)


# In[ ]:


model.score(XTrain,YTrain)


# In[ ]:


model.score(XTest,YTest)
#We found Almost Same Accuracy..!!


# **Decsion Tree Implementation..!!**

# In[ ]:


model_tree = DecisionTreeClassifier()
model_tree.fit(XTrain,YTrain)


# In[ ]:


model_tree.score(XTrain,YTrain)


# In[ ]:


model_tree.score(XTest,YTest)


# **Here We find....the Data is Overfitting the Model..!!**

# Being a Starter....I am stucked here as which appropriate Algo will fit the Data to the best..!!
# Thanks for Having a look and suggestions and improvements are always welcomed..!!
# 

# # Ok....So i did the above exploration as a Beginner, now practicing a lot, I have some specific steps to follow where out objective is to Achieve a **Logistic Regression** model to find out the pass/fail students based on our chosen cutoff.
# 
# ## The improvements to be performed are:
# 1. Deduce metrics such as "Total Marks" and "Passed/Failed" for our dataframe, as our outcome variable is "Passed/Failed".
# 2. Next we will create dummy variables for the categorical variables, and then look for correlations, before creating dummy variables and after creating dummy variables.
# 3. Apply Logistic Regression and find the variables, if applicable use RFE to find the variables which are TRUE, as they are selected from the RFE process.
# 4. Use StatsModels to find the best variables,best R2_Score, Accuracy,Precision and Confusion Matrix.
# 5. Provide Conclusion.

# In[ ]:


df.head()


# In[ ]:


df.nunique()


# In[ ]:


# Since we need to create dummy variables and from my inference, we don't need MathPassingStatus/ReadingPassingStatus and WritingPassingStatus, so we will drop these.
# Next we will dummy code the variables -> gender/lunch/test preparation course.
# Next we will Label Encode the variables -> race/ethinicity / parental level of education / Grade
# The we will standardize the remaining Numerical Variables to bring them onto one scale for modelling.
marks_df = df.drop(['MathPassingStatus','WritingPassingStatus','ReadingPassStatus'],1)
marks_df.head()


# In[ ]:


# Ok, as mentioned we have dropped the non-required columns, now let's perform some visualization for the dataframe
sns.pairplot(marks_df);


# In[ ]:


# So we see a Linear Relationship between the variables, let's plot their correlation
sns.heatmap(marks_df.corr(),annot=True);
plt.title('Correaltion for Marks Data Frame');


# In[ ]:


marks_df = marks_df.drop('Total_Marks',axis=1)
# Dropping highly correlated variables


# In[ ]:


# we see that Total Marks has high correlation for all the individual subjects and obviously, it is because it has been derived from the sum of all subjects, so we will keep it for modelling.
# Next we will convert our categorical variables to Numerical Variables.
dummy_df = pd.get_dummies(marks_df[['gender','test preparation course','lunch']],drop_first=True)
marks_df = pd.concat([marks_df,dummy_df],axis = 1)
marks_df.head(50)


# In[ ]:


marks_df.info()
# So now, we will drop the coulmns from which we have got dummies as they are now insignificant


# In[ ]:


marks_df = marks_df.drop(['gender','lunch','test preparation course'],axis=1)
marks_df.head()


# In[ ]:


# Next we go onto Label Enocding for the Variables -> race/ethinicity , parental level of education , Grade and Overall Passing Status
marks_df['race/ethnicity'] = marks_df['race/ethnicity'].astype('category')
marks_df['parental level of education'] = marks_df['parental level of education'].astype('category')
marks_df['Grade'] = marks_df['Grade'].astype('category')
marks_df['OverAllPassingStatus'] = marks_df['OverAllPassingStatus'].astype('category')
marks_df.info()


# In[ ]:


#Group A->0,Group B->1,#Group C->2,Group D->3,Group E->4,
marks_df['race/ethnicity'] = marks_df['race/ethnicity'].cat.codes
#associate's degree -> 0 , bachelor's degree -> 1, high school -> 2, master's degree - >3 , some college ->4 , some high school ->5
marks_df['parental level of education'] = marks_df['parental level of education'].cat.codes
# A->0 , B->1,C->2,D->3,E->4,Failed ->5
marks_df['Grade'] = marks_df['Grade'].cat.codes
marks_df.head()


# In[ ]:


marks_df.OverAllPassingStatus = marks_df.OverAllPassingStatus.cat.codes
# Passed->1,Failed - >0
marks_df.head()


# In[ ]:


sns.heatmap(marks_df.corr(),annot=True)


# In[ ]:


# Dropping negatively high correlated variables!
marks_df = marks_df.drop(['Grade','Percent'],axis=1)
marks_df.head()


# In[ ]:


# Now we will Standardize the untouches variables, which are: match score/writing score/readin score/Total_Marks/Percent
# But after splitting the data!
y = marks_df.OverAllPassingStatus
X = marks_df.drop('OverAllPassingStatus',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
X_train.head() 


# In[ ]:


cols_to_standardize = ['math score','writing score','reading score']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[cols_to_standardize] = scaler.fit_transform(X_train[cols_to_standardize])


# In[ ]:


X_train.head()


# In[ ]:


X_train.corr()


# In[ ]:


plt.figure(figsize=(16,9))
sns.heatmap(X_train.corr(),annot=True);
plt.title('Correlation for the training set');


# In[ ]:


#Let's see what is our passing rate
passed = round(sum(marks_df.OverAllPassingStatus/len(marks_df.OverAllPassingStatus.index))*100,2)
passed


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logistic_reg = LogisticRegression()
rfe = RFE(logistic_reg,5)
rfe = rfe.fit(X_train,y_train)


# In[ ]:


rfe.support_


# In[ ]:


cols_we_need = X_train.columns[rfe.support_]
cols_we_need


# In[ ]:


# Let's now access the Stats Model as we have found the columns that are to be used. Also, we have standardised the columns too.
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[cols_we_need])
model = sm.GLM(y_train,X_train_sm,family = sm.families.Binomial())
res = model.fit()


# ## Ok, this is an error of Perfect Sepration, where one of our column is total biased. Looking for solution and will get back. Till then keep Kaggling!
