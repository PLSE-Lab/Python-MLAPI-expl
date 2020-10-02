#!/usr/bin/env python
# coding: utf-8

# **Predicting Customer Churn at a Bank Practice**
# 
# 
# In This study we are going to do data mining with a bank dataset targeting customers churn problematic and i will be describing each and every step  with comments in the cell or out when needed.
# 
# Data science practice require much attention to get value of insights, it is based on unlocking hidden patterns or opportunities from a bunch of data accumulated over time, before we tackle the mining and predictions  as a data scientist i have to get a hint on 2 key points:
# 
# 1. Business understanding or domain knowledge:What cause a customer to churn by considering  relation of things and people towards the business
# 2. Analytical approach: Break it into pieces to understand what is required in terms of data,understanding and modelling results with a specific approach.

# **EDA (Exploratory Data Analysis)**

# In[ ]:


# Any results you write to the current directory are saved as output.
# import necessary libraries.note that i will be importing necessary libraries if need all along
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns#visualization
sns.set(style="ticks", color_codes=True)
import matplotlib.ticker as mtick # For specifying the axes tick format 
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization


# In[ ]:


#Import dataset into and display head of your dataset

churn = pd.read_csv('../input/Churn_Modelling.csv')
churn.head()


# As said understanding the relation of things, people and business metrics all plays a big to understand those small obvious patterns  before the core exploratory phase,thats why  some of the attributes  have almost no impact into our results such as Surname,CustomerId and RowNumber as long as we have an index(counts) it will serve as an ID.

# In[ ]:


#drop unnecessary attributes and display new dataset
churn = churn.drop(["RowNumber","CustomerId","Surname"], axis =1)


# In[ ]:


#Describe how big is out dataset helps to understand how big will be our analysis and requirements.
print("Rows : ",churn.shape[0])
print("Columns  : ",churn.shape[1])


# In[ ]:


#I check if there is any NaN values that can bring biased scenario, all column attributes should return false to verify this 
churn.isnull().any()


# In[ ]:


#count our unique values without duplication of same figure

print ("\nUnique values :  \n",churn.nunique())


# How huge are our attributes? in the meantime it is quite easier to spot that Geography,Gender,HasCrCard and IsActiveMember are categorical attributes that can corresponds to (yes/no) or 1/0 to define its state. so the rest of the attributes should be continuous attributes.

# In[ ]:


#what are our data types
churn.dtypes


# In[ ]:


#Mean=> the are a lot of average calculations in statistics so i used mean the check the average possibility of attributtes to impact the situation
churn.groupby(['Exited']).mean()


# The above Mean calculation is one of the ways to understand the average of our data, this helps us to investigate any to note into our study whenever it is high or low measures it can noted so that all along the study helps us to avoid biases. the difference between Exited and un 
# Exited are not that remarkable to give us a big picture of our study so any case can appear. But to note that:
# 
# * Customers with low creditscore tend to churn,reasonable!
# * on average older customers are the most to churn,questionable?
# * customers with high balance are churning probably they are getting attracted by other banks offer to raise the wealth and this corresponds with their estimatedsalary also
# * tenure,creditcard and being active mean are not explicitly helping in this case to hightlight anything big

# Before exploring deep our dataset  we can see attributes correlation with churn as we have seen that mean results are not so informative, so i first change dummies varibles which are Geography and Gender in our dataset so that we can get to define correlation of attributes to churn way easy.

# In[ ]:


#Let's convert all the categorical variables into dummy variables
df = pd.get_dummies(churn)
df.head()


# In[ ]:


plt.figure(figsize=(10,4))
df.corr()['Exited'].sort_values(ascending = False).plot(kind='bar')


# Let's check the exactitude correlation with figures 

# In[ ]:


plt.figure(figsize = (20,10))
sns.heatmap((df.loc[:, ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Exited','Geography_France','Geography_Germany','Geography_Spain','Gender_Female','Gender_Male']]).corr(),
            annot=True,linewidths=.5);


# The above graphs gives a hint on our journey:
# * Age,
# * Geography_germany,
# *  Balance 
# * gender_female 
# * Estimatedsalary 
# 
# all seems to have high correlation with the churn, remember this is exploratory many factors play along to change the situation in prediction outcomes.
# 
# 

# **Data Exploration**
# 
# This section is the core part of understanding the problem and channel late to right features, as said before we need to establish possible relations in our attributes and this is where to strongest part of trading off comes in to secure the best predictions.
# 
# Note: **Exited ** will always play the role of a target

# In[ ]:


# Passing labels and values
lab = churn["Exited"].value_counts().keys().tolist()
val = churn["Exited"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  0.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .2
              )
layout = go.Layout(dict(title = "Customer churn",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)


# 
# **20.4 Exited,
# 79.6% un Exited**
# This is a loss over a certain time yet there is no clear factor for churning maybe it is the nature of the business better be alert. but also it gives hope because the default test dataset is always 25% which will help to punish gradually.

# **Head to Head attributes relations**
# 
# Categotical attributes
#  and Continuous attributes
#        on impacting
#      Exited, here is not about the counts but a specifically churned and unchurned vs the service. 

# In[ ]:


#Categorical attirbutes churn rate
fig, axs = plt.subplots(2, 2, figsize=(15, 8))
sns.countplot(x= churn.Geography, hue = 'Exited' ,data=churn, ax =axs[0][0])
sns.countplot(x=churn.Gender, hue = 'Exited' ,data=churn, ax=axs[1][0])
sns.countplot(x=churn.HasCrCard, hue = 'Exited' ,data=churn, ax=axs[0][1])
sns.countplot(x=churn.IsActiveMember, hue = 'Exited' ,data=churn, ax=axs[1][1])
plt.ylabel('count')


# 
# What does Categorical attributes Highlights:
# 
# * Geographical location can determine the success of your business and can be a great tool to know how to play with your market as france show a huge number of customers with low churn.
# 
# * Apparently it is possible that customers without credit card are churning and it is obvious that the ones with it are not churning much.
# 
# * Female customers are churning than male this would be a factor of several things that can't be described without additional informations, also a great deal to consider gender so that the retention plan prepare promotions or offers based on affected gender,
# *  Not a suprise that inactive customers are churning than active ones

# * **Continous attributes churn rate**
# 
# For continuous attribute i will have to normalize its values in order to compare its churn it won't be possible to plot Balance and age in the same plot yet they have very different figures.

# In[ ]:


fig, axarr = plt.subplots(3, 2, figsize=(15, 8))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = churn , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = churn, ax=axarr[2][1])


#  As seen before the mean and correlation maps outcomes relate to these continuous results. but they are more detailed in terms of churned and unchurned distribution with estimated numbers .
# 
# Over all conclusion is that all attributes have its impact to the performance for instance Tenure and CreditScore are functions of age the more you get older the more the relation of a customer and a bank become stronger as sign of loyalty, and of course balances depends most of the case by your salary. these attributes are going to help us to engineer more case scenario by brin up new features that will help to punish negativity into predictions.
# 

# **Feature Engineering and Preparation**

# We are going to create new features from what we have and based on the relationship of attributes and prepare the existing ones to be ready to predict our next client possibly to churn and these stage is normally standardize head to head attributes as decided,
# for balance and Estmatedsalary this is quantitative relation then we will find its ratio and for tenure and creditscore over age .
# 
# Note that i am using df dataframe because i have already converted categorical attributes to dummy variables in other to process it excellently.
# 

# In[ ]:


df['BalanceEstimatedSalaryRatio'] = df.Balance/(df.EstimatedSalary)
df['TenureOverAge'] = df.Tenure/(df.Age)
df['CreditScoreOverAge'] = df.CreditScore/(df.Age)
df.head()


# In[ ]:


con_v=['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary','BalanceEstimatedSalaryRatio','TenureOverAge','CreditScoreOverAge']
minVec = df[con_v].min().copy()
maxVec = df[con_v].max().copy()
df[con_v] = (df[con_v]-minVec)/(maxVec-minVec)
df.head()


# ****Algorithms Modelling****
# 
# This study is task is a binary case study and obviosly as a supervised classification learning i would like to chose these models below, we will explore its results as we proceed.

# Our dataset have 20.4% on churned customers this means we will try to predict , thats why i selected to use default est_train_split model which deliver 25% test set and 75% traing set.because it covers the churned figure which prevent biases inside the model itself.

# In[ ]:


# Create Train & Test Data
from sklearn.model_selection import train_test_split
y = df['Exited'].values
x = df.drop(columns = ['Exited'])
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)


# 
# **Logistic Regrssion**

# In[ ]:


# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = x.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x))
x.columns = features


# In[ ]:


# Running logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
result = model.fit(x_train, y_train)
prediction_test = model.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test))# Print the prediction accuracy


# In[ ]:


# getting the weights of all the variables on regression model
weights = pd.Series(model.coef_[0],
                 index=x.columns.values)
weights.sort_values()[-13:].plot(kind = 'barh')
weights.sort_values(ascending = False)


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(x_train, y_train)
# Make predictions
prediction_test = model_rf.predict(x_test)
probs = model_rf.predict_proba(x_test)
print (metrics.accuracy_score(y_test, prediction_test))# Print the prediction accuracy


# In[ ]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=x.columns.values)
weights.sort_values()[-13:].plot(kind = 'barh')
weights.sort_values(ascending = False)


# **Support Vecor Machine (SVM)**

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
model.svm = SVC(kernel='linear') 
model.svm.fit(x_train,y_train)
preds = model.svm.predict(x_test)
metrics.accuracy_score(y_test, preds)# Print the prediction accuracy


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifiers = [
    KNeighborsClassifier(5),    
]
# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    # Create classifier, train it and test it.
    clf = item
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print (round(score,3),"\n", "- - - - - ", "\n") # Print the prediction accuracy
    


# **Model testing and validate on testing data**
# 
# Here i will provide first customers to churn as per our model predictions and we provide a risky dataframe which include a column(pro_true) for those to probably churning in the near future based on our selected model.
# 
# **After observing our models results Random forest beat the others!**

# In[ ]:


# apply on Random forest body and we will directly display five first customers as per our model
x_test["prob_true"] = prediction_test
df_risky = x_test[x_test["prob_true"] > 0.9]
display(df_risky.head()[["prob_true"]])


# In[ ]:


df_risky.shape


# In[ ]:


df_risky.head()


# **Conclusions**
# 
# In this study i have been dealing with a bank dataset with the aim to predict customers who may churn in the near future to prepare any retention plan after discovering patterns hidden in the dataset, at the beginning there was no clue of what it is happening because the patterns from mean and correlations map couldn't easily provide any clear insight thats where machine learning shows off what it can that our brains can't apparently. we have seen that churning customers in the past were 20,4% this is slightly not a trouble as long as it is low figure compared to the default sample of test data in model testing and this why i have taken 25% default set so that it can cover any biases behind.
# 
# Random forest beat other algorithms because it has the highest accuracy of 0.87 means 87%, referred to no free lunch theorem in machine learning that every machine learning is valid anyway it all depends on the situations and parameters passed to it to bring good results, in our case we can defend Random forest because it has shown its capabilities compared to others as long as we passed same data and same concept to all algorithms, on top of that random forest has a topology of multiple trees and this helps to avoid overfitting the model which is the main problem sometimes in machine learning. note that this model can perform way better if we feed it with more historical datasets this will help the model to capture many and necessary information to provide more accuracy in predictions.

# **Thank You.**

# In[ ]:




