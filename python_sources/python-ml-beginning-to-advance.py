#!/usr/bin/env python
# coding: utf-8

# **Comparing machine learning with human**
# 
# In our life we are taking decision based on past experience.In same way,Machine (computer) is taking decision based on past data. Here, we can compare machine with human brain and past data with past experience.
# 
# Machine Learning algorithms are helping machine(computer) to learn from data.In same way, neurons are helping to brain to learn from past experience.Here,we can compare neurons with machine learning algorithms.********
# 
# **Why Machine Learning?**
# 
# By taking data,Human can take decision by computing physically but problem is now a days we have millions of data.Here,we need a lot of computation, resources,time which can increase cost of business.For automating these things machine learning comes into picture.Machine can give more accurate result in less time.
# 
# We can fit machine learning code in our website for dynamic result based on browser(customer) experience.
# Now a days, Companies are generating more revenue due to Machine Learning methodology.They are showing results in website based on browser (customer) experience.They can evaluate success rate of their business, product based on customer experience.
# 
# **Different types of data**
# 
# We can divide data into 2 parts i.e.Qualitative and Quantitative.
# 
# **a)Qualitative**
# Nominal data(name,address etc.),categorical(Male,Female)
# 
# **b)quantitative**
# contineous(4,4.5,8,8.1),discrete(4,5,8),ordinal(feedback rating-1,2,3,4,5)
# 
# **Difference between structured and unstructured data**
# 
# In real world,We have 2 types data i.e. structured and unstructured.In real time,80% data are present in unstructured way(This type of data we can get from social media(tweeter,facebook,gmail,websites) and 20% data are present in structured way(This type of data we can get from Database,CSV,Excel etc.
# 
# **Different steps to implement a Machine Learning Algorithm(Including Mchanie Learning Algorithm deployment in RealTime)**
# 
# Step 1:Collect structured or unstructured data from different sourses.
# 
# step 2:Try to understand defination or meaning of each column(Why we need these columns for driving business).
# 
# step 3:Data Preprocessing(Data Cleansing).Here,we are trying to check format of each column,null value treatment,creating new columns using existing column etc.
# 
# step 4:EDA(Exploratory Data Analysis).Here we are trying to summarize large dataset in tabular or visualization format.It is helpful for deriving step 5.
# 
# step 5:Feature selection/Feature reduction(selecting independent columns for predictor(X) and dependent target column(Y)) and Feature Extraction(use PCA,different normalization method,different standadization method).
# 
# Note:We can select independent columns for predictor(X) before or after using Machine Learning Algorithm.
# 
# step 6:Apply K-Fold cross validation on dataset using different machine algorithm.select algorithm based on highest average value(computing based on accuracy) which has given by K-Fold cross validation method.Here,We can say,Cross Validations are useful for Model(Algorithm) selection.
# 
# step 7:Divide data into 2 part i.e.Train(75% or 80% or 70% etc.) and Test(25% or 20% or 30%) ratio.For each Train and Test,we need to choose/divide into 2 part i.e.predictor and target.
# 
# step 8:Apply Machine Learning algorithm(Choose apropriate Machine learning algorithm based on K-Fold cross validation result) on training dataset.
# 
# step 9:Check accuracy and try to increase accuracy by removing columns(Non important columns).
# step 10:Apply algorithm on Testing dataset and check overfitting(variance) and accuracy.
# 
# step 11:Share column names and result with your client and take confirmation about columns.Because some times we are focusing on clients requirement rather than accuracy.
# 
# step 12:Convert your python code into pickel and connect with your website(eg:website has build using python and these python code will call pickel package by proving some input.Based on input Machine learning algorithm will execute and it will pass the result to function caller and based on result python code will take data from database(it will do some computation based on inside python code) then it will tell to front end for displying on wesite).This process is called deploying Machine Learning Algorithm in production in real time.
# 
# Note:What is pickle?
# 
# Pickel package is used for serializing and de-serializing python object and it can be saved in disk.Load this file in python environment and use in your code.
# **Data Preprocessing**
# 
# This step is very important.Because if we will not treat raw data properly than it will reflect to our model accuracy.
# Accuracy is depending on Data Preprocessing,Feature selection,Feature Extraction and Machine Learning Algorithm.We can say,80% accuracy is depending on Data Preprocessing,Feature selection and Feature Extraction and 20% accuracy is depending on Machine Learning Algorithm.
# 
# **1)Checking format of each column.**
# 
# **2)Remove duplicate.**
# 
# **3)Null value Treatment**
# 
# There are different way of null value treatment
# a)Calculate null value percentage of complete dataset.If null value(%)>25% than Remove rows with null value.If null value(%)<25% than replace null value with mean(recommended) or median or mode.
# 
# b)We can use machine learning algorithms(Linear regression,K nearest neighbour,cluster etc.) to fill null value(Cost Effective).
# 
# c)Back fill(replace current null value with next value of current row and column) or forward fill(replace current null value with previous value of current row and column).
# 
# **4)Outlier Treatment**
# 
# a)Outlier data can mislead ML algorithms.
# 
# b)Delete outliers row or replace with mean or median or mode.
# 
# c)Create normal distribution graph and take 99.3% area of graph(99.3% data are near to mean) for further process and 0.7% data are outlier. 
# 
# ![image.png](attachment:image.png)
# 
# d)We can resolve heteroscadicity problem in linear regression algorithm.
# 
# **How to detect Outlier in dataset?**
# 
# 1)using Quartile method(Q1,Q3,IQR,LQR(Lower Quartile Range),UQR(Upper Quartile Range)
# 2)using Z-score/Z-test
# 3)standard Deviation 
# 4)Scatter plot(Bivariate Analysis)
# 5)Dbscan (Density Based Spatial Clustering of Applications with Noise)
# **5)Feature Engineering**
# 
# As per domain knowledge or coverting ordinal data to numerical,create new features but it is difficult and expansive.
# 
# **EDA(Exploratory Data Analysis)**
# 
# a)In this process,we are trying to find out business insights based on raw data.
# 
# b)Summmarizing data using "group by" function in python.
# 
# c)Visualizing data using graph and interactive Dashboard and share with your client.
# 
# d)Based on EDA,Business buddies are taking decision(like New business planning,Success or Failure of existing plan,New business location as per population and business demand,fixing price of product by comparing with other company product(using web scrapping) etc.)
# 
# In EDA,We can do 3 types of Analysis,
# 
# 1)Univariate Analysis:
# 
# We can do this analysis by using one variable/column.Different types of analysis like mean,median,mode,maximum,minimum, count, quartile, histogram,pie chart,bar chart standard deviation,z-test etc.
# 
# 2)Bivariate or Multivariate Analysis
# 
# We can do this by using two or more variables/columns(Relationship between 2 or more variables).Different types of analysis like pearson correlation coefficient,scatter plot,bar chart,quartile,ANOVA,MANOVA,chi-sqare,covariance etc.
# 
# **Feature Selection/Feature Reduction**
# 
# a)Forward Selection(Heuristic search startegy/Wrapper Method)
# 
# b)backward selection(Heuristic search startegy/Wrapper Method)
# 
# c)stepwise selesction
# 
# d)univariate analysis(T-test,Z-test,F-test,ANOVA and chi-square test using hypothesis testing)(Filter Method)
# 
# e)Bivariate analysis(Pearson correlation coefficient)
# 
# f)Signal to noise ratio
# 
# g)Lasso or Ridge or elastic net Linear regression(It is only applicable for Linear regression data).It is coming under Emedded method.
# 
# h)Tree pruning(less branch/depth for more accuracy and removing overfitting.
# 
# i)PCA(Principal Component Analysis) for selecting variable with more variance artificial liner line.
# 
# j)Factor Analysis
# 
# **Advantage of Feature Selection/Feature Reduction**
# 
# a)Less number of features.
# 
# b)Less size of data.
# 
# c)Less computation for training a dataset.
# 
# d)More accuracy.
# 
# **Different type of Machine Learning Algorithm**
# 1)supervised ML
# 2)Unsupervised ML
# 3)Reinforcement ML
# 
# 1)Supervised ML Algorithm
# 
# Here we are doing future prediction by using Dataset.
# 
# Eg.Linear Regression,Logistic Regression,Decision Tree,Random Forest,Ensemble Algorithm(Bagging,Boosting,Stacking),KNN(K Nearest Neighbour),Naive Bayes
# 
# 2)Unsupervised ML Algorithm
# 
# Here we are grouping Dataset for targeting some specific group of people
# 
# Eg.Cluster(Hierarchical Cluster,K-mean Cluster)
# 
# 3)Reinforcement ML Algorithm
# 
# Taking a action based on environment.

# 
