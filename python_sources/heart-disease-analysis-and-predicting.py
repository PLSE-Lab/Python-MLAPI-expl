#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE ANALYSIS

# # Contents
1) INTRODUCTION2) READING THE DATA3) DATA EXPLORATION4) CREATING MODEL5) EXPLANATION6) CONCLUSION
# # 1) INTRODUCTION:
Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others.

The term "heart disease" is often used interchangeably with the term "cardiovascular disease." Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease.


We have a data which classified if patients have heart disease or not according to features in it. We will try to use this data to create a model which tries predict if a patient has this disease or not. We will use logistic regression (classification) algorithm.
# # 2) Reading the data

# In[120]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

Now,we are setting our working directory in Python 3 Notebook
# In[121]:


heart=pd.read_csv('../input/heart.csv')

once our dataset is loaded we can make slicing and dicing of our data.Now,By Using the following snippet we can see the loaded data.
# In[122]:


# To see the first top five rows in our uploaded data #
heart.head()

As we can see in the output, we have following variables :


age: The person's age in years
sex: The person's sex (1 = male, 0 = female)
cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
chol: The person's cholesterol measurement in mg/dl
fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
thalach: The person's maximum heart rate achieved
exang: Exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
ca: The number of major vessels (0-3)
thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
target: Heart disease (0 = no, 1 = yes)
# In[123]:


# To see the Last five rows in our uploaded data #
heart.tail()


# In[124]:


# Now we will look at summary statistics of our data #
# summary statistics are used to summarize a set of observations, in order to communicate the largest amount of information as simply as possible #

heart.describe()

 Using this function describe() we can analysis between the numerical values contained in the data set. Using this function we can get count, mean, std, min,25%, 50%, 75%,max.
 Here we can see most of the  column values are categorical variables like (fbs,sex,exang)
# In[125]:


#   To get a concise summary of the dataframe #
heart.info()

As we can see in the output, the summary includes list of all columns with their data types and the number of non-null values in each column. 
we also have the value of rangeindex provided for the index axis.
# In[126]:


# We will list all the columns in our loaded dataset #
heart.columns


# In[127]:


# Now we will see how many rows and columns are present in our loaded dataset #
heart.shape

As we can see in the output, we have 303 rows and 14 columns in our data.
# In[128]:


# To find the how many missing values in our data #
heart.isnull().sum()

As we can see in the output,there is no missing values in our data
# # 3) Data Exploration
our data is ready for  slicing and dicing, we need to explore our data by using various plots using  matplotlib.pyplot as plt library,seaborn as sns library.

By doing this we can visually analyse each and every variables in our data.first of all we will look at our target variable in our data set (target - have disease or not (1=yes, 0=no))
# In[129]:


heart.target.value_counts()

As we can see in the output, 165 nos have disease and 138 nos not have disease
# In[130]:


sns.countplot(x="target",data=heart)

As we can see in the output, "0" refers  have disease and "1" not have disease.Now we will analyze each and every columns in our data.first we will take Age variable in our data set.
# In[131]:


heart.age.value_counts()[:15]


# In[132]:


sns.barplot(x=heart.age.value_counts()[:15].index,y=heart.age.value_counts()[:15].values)
plt.xlabel('Age')
plt.ylabel('Age Count')
plt.title('Age Analysis System')
plt.show()


# In[133]:


heart.sex.value_counts()


# In[134]:


sns.countplot(x='sex', data=heart)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

As we can see in the output, we have 96 females and 207 malesNow, we are going to analyze both  variables the sex(male and female) and the heart health situation(target variable).
# In[135]:


male_disease=heart[(heart.sex==1) & (heart.target==1)]          ## Here we have sex=1(male) and target =1(have disease)
male_NO_disease=heart[(heart.sex==1) & (heart.target==0)]       ## Here we have sex=1(male) and target =0(have no disease )
print(len(male_disease),"male_disease")
print(len(male_NO_disease),"male_NO_disease")

As we can see in the output, we have 93 males "have disease" and 114 "male_NO_disease"
# In[136]:


a=len(male_disease)
b=len(male_NO_disease)
sns.barplot(x=['male_disease ','male_NO_disease'],y=[a,b])
plt.xlabel('Male and Target')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()


# In[137]:


female_disease=heart[(heart.sex==0) & (heart.target==1)]          ## Here we have sex=0(female) and target =1(have disease)
female_NO_disease=heart[(heart.sex==0) & (heart.target==0)]       ## Here we have sex=0(female) and target =0(have no disease )
print(len(female_disease),"female_disease")
print(len(female_NO_disease),"female_NO_disease")

As we can see in the output, we have 72 females "have disease" and 24 "female_NO_disease"
# In[138]:


c=len(female_disease)
d=len(female_NO_disease)
sns.barplot(x=['female_disease ','female_NO_disease'],y=[c,d])
plt.xlabel('Female and Target')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()


# In[139]:


heart["cp"].value_counts()

As we can see in the output, For Chest Pain Type Analysis we have four categories.
# In[140]:


sns.countplot(x='cp', data=heart)
plt.xlabel(" Chest type")
plt.ylabel("Count")
plt.title("Chest type Vs count plot")
plt.show()

Now for each Chest pain type(0,1,2,3) Vs Target ( 1=have disease,0=have no disease) we will analyze.
# In[141]:


print(len(heart[(heart.cp==0)&(heart.target==0)]),"=cp_zero_target_zero")
print(len(heart[(heart.cp==0)&(heart.target==1)]),"=cp_zero_target_one")
print(len(heart[(heart.cp==1)&(heart.target==0)]),"=cp_one_target_zero")
print(len(heart[(heart.cp==1)&(heart.target==1)]),"=cp_one_target_one")


# In[142]:


target_0=len(heart[(heart.cp==0)&(heart.target==0)])
target_1=len(heart[(heart.cp==0)&(heart.target==1)])
plt.subplot(1,2,1)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("Chest_type_0 Vs count plot")


target_0=len(heart[(heart.cp==1)&(heart.target==0)])
target_1=len(heart[(heart.cp==1)&(heart.target==1)])
plt.subplot(1,2, 2)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("Chest_type_1 Vs count plot")

As we can see in the output, For the Chest type 0 number of heart Disease is 39 Nos and for the Chest type 1 number of heart Disease is 41 Nos.
# In[143]:


print(len(heart[(heart.cp==2)&(heart.target==0)]),"=cp_two_target_zero")
print(len(heart[(heart.cp==2)&(heart.target==1)]),"=cp_two_target_one")
print(len(heart[(heart.cp==3)&(heart.target==0)]),"=cp_three_target_zero")
print(len(heart[(heart.cp==3)&(heart.target==1)]),"=cp_three_target_one")


# In[144]:


target_0=len(heart[(heart.cp==2)&(heart.target==0)])
target_1=len(heart[(heart.cp==2)&(heart.target==1)])
plt.subplot(1,2,1)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("Chest_type_2 Vs count plot")


target_0=len(heart[(heart.cp==3)&(heart.target==0)])
target_1=len(heart[(heart.cp==3)&(heart.target==1)])
plt.subplot(1,2, 2)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("Chest_type_3 Vs count plot")

As we can see in the output, For the Chest type 2 number of heart Disease is 69 Nos and for the Chest type 3 number of heart Disease is 16 Nos.
# In[145]:


plot = heart[heart.target == 1].trestbps.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)
plot.set_title("Resting blood pressure", fontsize = 20)

As we can see in the output, For the trestbps(resting blood pressure) variable has the maximum 130mm Hg on admission to the hospital for the person who have heart disease.
# In[146]:


heart.chol.value_counts()[:20]


# In[147]:


sns.barplot(x=heart.chol.value_counts()[:20].index,y=heart.chol.value_counts()[:20].values)
plt.xlabel('chol')
plt.ylabel('Count')
plt.title('chol Counts')
plt.xticks(rotation=45)
plt.show()

Now we need to find for all the persons in the chol variable wheather heart disease is effected or not.

# In[148]:


age_unique=sorted(heart.age.unique())
age_chol_values=heart.groupby('age')['chol'].count().values
mean_chol=[]
for i,age in enumerate(age_unique):
    mean_chol.append(sum(heart[heart['age']==age].chol)/age_chol_values[i])
    


# In[149]:


plt.figure(figsize=(10,5))
sns.pointplot(x=age_unique,y=mean_chol,color='red',alpha=0.8)
plt.xlabel('age',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('chol',fontsize = 15,color='blue')
plt.title('age vs chol',fontsize = 15,color='blue')
plt.grid()
plt.show()


# In[150]:


print(len(heart[(heart.fbs==1)&(heart.target==0)]),"=fbs_one_target_zero")
print(len(heart[(heart.fbs==1)&(heart.target==1)]),"=fbs_one_target_one")


# In[151]:


target_0=len(heart[(heart.fbs==1)&(heart.target==0)])
target_1=len(heart[(heart.fbs==1)&(heart.target==1)])
plt.subplot(1,2,1)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("fbs_type_1 Vs count plot")

As we can see in the output, for Fbs type 1 (fasting blood sugar > 120 mg/dl) the number of heart disease is 23 Nos.
# In[152]:


print(len(heart[(heart.restecg==1)&(heart.target==0)]),"=restecg_one_target_zero")
print(len(heart[(heart.restecg==1)&(heart.target==1)]),"=restecg_one_target_one")


# In[153]:


plot = heart[heart.target == 1].thalach.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 10)
plot.set_title("thalach", fontsize = 15)

As we can see in the output, For the thalach variable has 162 maximum heart rate achieved for the person who have heart disease.
# In[154]:


heart.thal.value_counts()


# In[155]:


print(len(heart[(heart.thal==3)&(heart.target==0)]),"=thal_three_target_zero")
print(len(heart[(heart.thal==3)&(heart.target==1)]),"=thal_three_target_one")


# In[156]:


target_0=len(heart[(heart.thal==3)&(heart.target==0)])
target_1=len(heart[(heart.thal==3)&(heart.target==1)])
plt.subplot(1,2,1)
sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])
plt.ylabel("Count")
plt.title("thal_type_3 Vs count plot")


# In[157]:


print(len(heart[(heart.thal==6)&(heart.target==0)]),"=thal_7_target_zero")   # Here thal for (6 = fixed defect) has no heart disease
print(len(heart[(heart.thal==6)&(heart.target==1)]),"=thal_7_target_one")


# In[158]:


print(len(heart[(heart.thal==7)&(heart.target==0)]),"=thal_7_target_zero")  # Here thal for (7 = reversable defect) has no heart disease
print(len(heart[(heart.thal==7)&(heart.target==1)]),"=thal_7_target_one")

Creating Dummy Variables for 'cp', 'thal' and 'slope' because this columns has categorical variables we'll turn them into dummy variables
# In[159]:


cp = pd.get_dummies(heart['cp'], prefix = "cp", drop_first=True)
thal = pd.get_dummies(heart['thal'], prefix = "thal" , drop_first=True)
slope = pd.get_dummies(heart['slope'], prefix = "slope", drop_first=True)

#Removing the first level.


# In[160]:


data = pd.concat([heart, cp, thal, slope], axis=1)
data.head()

now we have to drop the variables  cp, thal, slope in the object data because we have created dummy variables
# In[161]:


data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)
data.head()

After creating the dummy variables we will drop target variable.
# In[162]:


x = data.drop(['target'], axis=1)
y = data.target


# In[163]:


print(x.shape)

now we will see the correlation between each variables in our dataset
# In[164]:


x.corr()

 Now we will Normalize our Data
# In[165]:


x = (x - x.min())/(x.max()-x.min())
x.head()

Now We will split our data 80% of our data will be train data and 20% of it will be test data.
# In[166]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

In our dataset our dependent variable is a categorical variable so we will go for a Logistic Regression model.
Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
# # 4) Creating model

# # LOGISTIC REGRESSION

# In[167]:


from sklearn.linear_model import LogisticRegression
logi = LogisticRegression()
logi.fit(x_train, y_train)
logi.score(x_test, y_test)


# In[168]:


from sklearn.model_selection import GridSearchCV
 ## Setting parameters for GridSearchCV
params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
logi_model = GridSearchCV(logi,param_grid=params,cv=10)


# In[169]:


logi_model.fit(x_train,y_train)
logi_model.best_params_


# In[170]:


logi = LogisticRegression(C=1, penalty='l2')
logi.fit(x_train, y_train)
logi.score(x_test, y_test)


# # 5) Explanation
Now we will see confusion matrix,A confusion matrix is a technique for summarizing the performance of a classification algorithm.

Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.

Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.
# In[171]:


from sklearn.metrics import confusion_matrix
cm_lg = confusion_matrix(y_test, logi.predict(x_test))
sns.heatmap(cm_lg, annot=True)
plt.plot()


# # DECISION TREE
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.
# In[172]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)                                # HERE WE ARE FITTING THE VALUES OF BOTH x_train,y_train


# In[173]:


predict=dtree.predict(x_test)                               # HERE WE ARE PREDICTING y_test values.
predict


# In[174]:


#NOW WE WILL SEE CONFUSION MATRIX FOR DECISION TREE

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))


from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test,predict )
sns.heatmap(cm_tree, annot=True)
plt.plot()


# In[175]:


from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(y_test,predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL


# # Random Forest
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# In[176]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)


# In[177]:


rfc_predict=rfc.predict(x_test)                                # HERE WE ARE PREDICTING y_test values.
rfc_predict


# In[178]:


#NOW WE WILL SEE CONFUSION MATRIX FOR RANDOM FOREST

from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test,rfc_predict )
sns.heatmap(cm_rf, annot=True)
plt.plot()


print(classification_report(y_test,rfc_predict))


# In[179]:


from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(y_test,rfc_predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL


# # K Nearest Neighbors
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:

In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
# In[180]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)


# In[181]:


from sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
Classifier.fit(x_train,y_train)


# In[182]:


y_predict=Classifier.predict(x_test)                                # HERE WE ARE PREDICTING y_test values.                
y_predict


# In[183]:


#NOW WE WILL SEE CONFUSION MATRIX FOR K NEAREST NEIGHBOR

from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test,y_predict )
sns.heatmap(cm_knn, annot=True)
plt.plot()


# In[184]:


from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(y_test,y_predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL


# # Confusion Matrixes

# In[185]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lg,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,2)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cm_tree,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,3)
plt.title("Random forest")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,3,4)
plt.title("K Nearest Neighbor Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)

I am new with data science. Please comment me your feedbacks to help me improve myself.Suggestions are welcome
I hope you find this kernel helpful and some UPVOTES would be very much appreciated.Thank you for you're time
