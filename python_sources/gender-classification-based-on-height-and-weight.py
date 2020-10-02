#!/usr/bin/env python
# coding: utf-8

# <h1>Gender classification based on height and weight</h1>
# <br>
# Here's my first data analysis on Kaggle. I am currently learning to become data analyst and data scientist. So, please correct me if I have made any mistake in my analysis.
# 
# This is a very basic classification problem where we classify whether a person is male or female given his weight and height. 
# 
# Data Source: https://www.kaggle.com/mustafaali96/weight-height

# In[ ]:


#importing all the dependencies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report


# In[ ]:


#importing data and storing into dataframe
path = "../input/weight-height.csv"
df = pd.read_csv(path)
df.head()


# <h2>Describing/Cleaning Data</h2>
# <br>
# ****Target Variable:**** Gender (categorical)
# <br>
# ****Predictor Variables**:** Height (numerical), Weight (numerical)

# In[ ]:


#Checking if there are any null values in dataframe at all
df.isnull().sum().sum()


# In the entire dataset, there are no null values. So, we do not need to worry about null values.

# In[ ]:


print(df.describe())


# <h2>Descriptive analytics:</h2>
# <br>
# (Writing in non-technical terms:)
# ***
# **1. Mean:** An average person in the sample data has 66.36 inch height and 161.44 pound weight. 
# ***
# **2. Minimum and maximum:** The lowest height of the person in the sample is 54.26 inch while the maximum height of the person in the sample is 78.99 inches.The person with the lowest weight weighs 64.7 pounds and the person with the highest weight weights 269.98 pounds in the sample.
# ***
# **3. Percentiles:**  25% of the people in the sample has height less than 63.5 inches while 25% of the people in the sample has height more than 69.17 inches. 25% of the people in the sample weights less than 135.81 pounds while 25% of people in the sample weigh more than 187.17 pounds.
# ***
# **4. Median:** 50% of the people in the sample has height more or less than 66.31 inches. 50% of the people in the sample weigh more or less than 161.21 pounds.
# ***
# **5. Standard deviation:** On average, height of the person varies by 3.84 inches from the average height of person. On average, weight of the person varies by 32.10 pounds from the average weight of person.

# In[ ]:


#Checking if data is balanced or not using pie chart for our target variable - gender
labels = df.Gender.unique()
sizes = [(df.Gender == labels[0]).sum(), (df.Gender == labels[1]).sum()]
plt.pie(sizes, labels = labels, autopct='%1.1f%%', startangle = 90)
plt.title("Checking if data is balanced or not with a pie chart")
plt.show()


# There are equal no. of male and female in the dataset. Hence, the data is balanced. So we need not worry about balancing data.

# In[ ]:


#Outliers
plt.boxplot(df.Weight)
plt.title('Box-plot for weight')
plt.show()

plt.boxplot(df.Height)
plt.title('Box-plot for height')
plt.show()


# **Using Tukey's rule for Box-plot:**
# There is one outlier for weight, and there seems to be multiple outliers for height. However, these outliers are necessary for our analysis as they can impact the gender. Hence, we keep these outliers in the data.

# In[ ]:


#finding relationship between gender and height
print(df.groupby('Gender')['Height'].describe())


# **Relationship between Gender and Height:**
# <br>
# It can be seen that the mean of height and standard deviation of height for male and female are different. Hence, there is a meaningful difference between gender and height. Thus, there seems to be a relationship between gender and height.

# In[ ]:


#finding relationship between gender and weight
print(df.groupby('Gender')['Weight'].describe())


# **Relationship between Gender and Weight:**
# <br>
# It can be seen that the mean of weight and standard deviation of weight for male and female are different. Hence, there is a meaningful difference between gender and height. Thus, there seems to be a relationship between gender and height.

# In[ ]:


#scatter plot to analyse height vs weight
weight = df['Weight']
height = df['Height']
plt.scatter(height, weight)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title("Height vs Weight")
plt.show()


# **Relationship between height and weight:**
# <br>
# There appears to be a positive relationship between weight and height too. Thus, there could be an interaction effect while predicting gender.

# In[ ]:


#histogram for height
plt.hist(height, bins = 10)
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Histogram for height')
plt.show()

#histogram for weight
plt.hist(weight, bins = 10)
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Histogram for weight')
plt.show()


# **Distribution of data:** Both height and weight seems to be normally distributed.

# <h2>Data Analysis</h2>
# <h3>Splitting data </h3>
# Now, that we have described our data and now that we know the data is clean to analyse, we create test and train data set. The general 80:20 split has been used where 80% of data is split into training set and 20% of data is split into test set.

# In[ ]:


#model to split data
train, test = train_test_split(df, test_size=0.2)
print("Test data set")
print(test.head())
print()
print("Train data set")
print(train.head())
print('')
print("No. of data in test:" +str(len(test)))
print("No. of data in train:" +str(len(train)))

print(train.groupby('Gender').count())
#creating x and y variables
feature_names = ['Height', 'Weight']
x_train = train[feature_names].values.tolist()
y_train = train['Gender']
x_test = test[feature_names].values.tolist()
y_test = test['Gender']


# The train dataset is more or less balanced too.

# <h2>Classifiers</h2>
# Since this is a classific binary classification problem, following classifiers have been chosen:
# 1. Binary Decision Tree
# 2. Support Vector Machine
# 3. k-n Neighbours Classifier
# 4. Naive Bayes Classifier

# In[ ]:


#defining classifiers
clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC(gamma='auto')
clf3 = neighbors.KNeighborsClassifier()
clf4 = GaussianNB()

#fitting data
clf1 = clf1.fit(x_train,y_train)
clf2 = clf2.fit(x_train,y_train)
clf3 = clf3.fit(x_train, y_train)
clf4 = clf4.fit(x_train, y_train)

#making predictions
prediction1 = clf1.predict(x_test)
prediction2 = clf2.predict(x_test)
prediction3 = clf3.predict(x_test)
prediction4 = clf4.predict(x_test)


# <h2>Evaluating Model:</h2>
# **1. Classification Accuracy**

# In[ ]:


#checking accuracy
r1 = accuracy_score(y_test, prediction1)
r2 = accuracy_score(y_test, prediction2)
r3 = accuracy_score(y_test, prediction3)
r4 = accuracy_score(y_test, prediction4)

print("Accuracy score of Model 1: DecisionTreeClassifier is "+str(r1))
print("Accuracy score of Model 2: SupportVectorMachine is "+str(r2))
print("Accuracy score of Model 3: KNN is "+str(r3))
print("Accuracy score of Model 4: GaussianNB is "+str(r4))


# Overall:
# 1. Decision Tree Classifier is correct 87.1% of the times.
# 2. SVM is correct 91% of the times
# 3. KNN is correct 90.1% of the times
# 4. Naive Bayes is correct 88.55% of the times
# <br>
# <br>
# Seems like SupportVectorMachine has greater accuracy than other classifiers.
# 
# <h2>2. Misclassification Rate</h2>

# In[ ]:


#finding misclassification rate
mr1 = (1-metrics.accuracy_score(y_test, prediction1))
mr2 = (1-metrics.accuracy_score(y_test, prediction2))
mr3 = (1-metrics.accuracy_score(y_test, prediction3))
mr4 = (1-metrics.accuracy_score(y_test, prediction4))
print("Misclassification Rate of Decision Tree: "+ str(mr1))
print("Misclassification Rate of SVM: "+ str(mr2))
print("Misclassification Rate of KNN: "+ str(mr3))
print("Misclassification Rate of Naive Bayes: "+ str(mr4))


# Overall:
# 1. Decision Tree Classifier is incorrect 12.9% of the times.
# 2. SVM is incorrect 8.9% of the times
# 3. KNN is incorrect 9.8% of the times
# 4. Naive Bayes is incorrect 11.45% of the times  
# 
# <br>SVM seems to be a better model.
# 
# <h2>3. Sensitivity and Specificity</h2>
# 

# In[ ]:


#confusion matrix

#decision tree
cm1 = confusion_matrix(y_test, prediction1)
TP1 = cm1[1,1]
TN1 = cm1[0,0]
FP1 = cm1[0,1]
FN1 = cm1[1,0]

#svm
cm2 = confusion_matrix(y_test, prediction2)
TP2 = cm2[1,1]
TN2 = cm2[0,0]
FP2 = cm2[0,1]
FN2 = cm2[1,0]

#knn
cm3 = confusion_matrix(y_test, prediction3)
TP3 = cm3[1,1]
TN3 = cm3[0,0]
FP3 = cm3[0,1]
FN3 = cm3[1,0]

#naive-bayes
cm4 = confusion_matrix(y_test, prediction4)
TP4 = cm4[1,1]
TN4 = cm4[0,0]
FP4 = cm4[0,1]
FN4 = cm4[1,0]

#sensitivity
sen1 = TP1 / float(FN1 + TP1)
sen2 = TP2 / float(FN2 + TP2)
sen3 = TP3 / float(FN3 + TP3)
sen4 = TP4 / float(FN4 + TP4)

#specificity
spec1 = TN1 / (TN1 + FP1)
spec2 = TN2 / (TN2 + FP2)
spec3 = TN3 / (TN3 + FP3)
spec4 = TN4 / (TN4 + FP4)

#printing
print("Sensitivity Rate of Decision Tree: "+ str(sen1))
print("Sensitivity Rate of SVM: "+ str(sen2))
print("Sensitivity Rate of KNN: "+ str(sen3))
print("Sensitivity Rate of Naive Bayes: "+ str(sen4))
print()
print("Specificity Rate of Decision Tree: "+ str(spec1))
print("Specificity Rate of SVM: "+ str(spec2))
print("Specificity Rate of KNN: "+ str(spec3))
print("Specificity Rate of Naive Bayes: "+ str(spec4))
print()
print("Classification Report: Decision Tree")
print(classification_report(y_test, prediction1))
print("Classification Report: SVM")
print(classification_report(y_test, prediction2))
print("Classification Report: KNN")
print(classification_report(y_test, prediction3))
print("Classification Report: Naive-Bayes")
print(classification_report(y_test, prediction4))


# When the actual value is male:
# 1. Decision Tree is correct 90% of the time
# 2. SVM is correct 93% of times
# 3. KNN is correct 92.42% of the times
# 4. Naive Bayes is correct 90% of the times.
# 
# When the actual value is female:
# 1. Decision Tree is correct 86% of the time
# 2. SVM is correct 91% of times
# 3. KNN is correct 89% of the times
# 4. Naive Bayes is correct 87% of the times.
# 
# Even here, SVM is a better model.

# In[ ]:


#Making a prediction for a user
height = 71
weight = 176
prediction = clf2.predict([[height, weight]])
print("The classifier predicts that you could be " +str(prediction[0]))


# Well. It predicted my gender correctly. You could give it a try. If it does not predict it, know that the classifier is wrong 8% of the times.
# 
# 
# <h2>Limitations</h2>
# * This is a very basic model that is prediciting gender based on only 2 variables. Other factors such as age, nationality/race can be important too.
