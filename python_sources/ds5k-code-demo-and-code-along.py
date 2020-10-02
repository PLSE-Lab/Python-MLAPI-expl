#!/usr/bin/env python
# coding: utf-8

# # Demystifying Data: Intro to Data Science and the DS5k
# Bryce Peake, General Assembly (Washington, DC)

# ## Sample case study - Let's think about customer churn!
# I work at a hospitality company that has many corporate customers who book hotel rooms for their employees, and sometimes there's churn. We don't have enough account managers to assign one-on-one, but we lose a lot of money if no one knows there's a problem!  
# 
# *Current state*:  Randomly assign account managers right now, and typically after a customer churns - it costs a lot more to get them back!
# 
# *Ask*: Automate alerts for high-risk churn, and capture accounts before they leave our orbit. 
# 
# *Future State*: Automate assigning account managers based on manager skill and churn risk. 

# #### The data (don't worry, it's been masked/fabricated)
# 
# Here are the fields and their definitions:
# 
#     Name : Name of the latest contact at Company
#     Age: Customer Age
#     Total_Purchase: Total Purchased
#     Account_Manager: Binary 0=No manager, 1= Account manager assigned
#     Years: Totaly Years as a customer
#     Num_sites: Number of employees that use the service.
#     Onboard_date: Date that the name of the latest contact was onboarded
#     Location: Client HQ Address (Masked)
#     Company: Name of Client Company (Masked)
#     Churn: Binary 0=Stayed (No Churn), 1=Churned

# In[ ]:


#Import your libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import the data
df = pd.read_csv("../input/customer_churn.csv")
df.head(10)


# In[ ]:


#How many customers in this data set churned?
df.Churn.value_counts()


# In[ ]:


#Visualize some of the relationships
sns.heatmap(df.corr())


# In[ ]:


sns.jointplot(x = "Total_Purchase", y = "Years", data = df)


# In[ ]:


#Prep data for training a model
from sklearn.model_selection import train_test_split
X = df[["Age", "Total_Purchase", "Years", "Num_Sites"]]
y = df.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)


# In[ ]:


#create the model by 
from sklearn.linear_model import LogisticRegression #IMPORT
LogReg = LogisticRegression() #INSTANTIATE
LogReg.fit(X_train, y_train) #FIT
y_pred = LogReg.predict(X_test) #PREDICT


# In[ ]:


#Evaluate the model with a classifcation report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))


# In[ ]:


#Visualize the power of our model
#Explain what AUC is, and create a score and visualization
#import the metrics for the AUC
from sklearn.metrics import roc_auc_score #Area under the curve
from sklearn.metrics import roc_curve #Receiver Operator Curve

log_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, LogReg.predict_proba(X_test)[:,1])


#let's plot it!
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % log_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

#make it pretty!
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#Predict new data
new_customer = pd.read_csv("../input/new_customers.csv")
new_customer.reset_index(inplace = True)
new_customer.columns = ["Names", "Age", "Total_Purchase", "Account_Manager", "Years", "Num_sites", 
                     "Onboard_date", "Location", "Company", "Churn"]
new_customer.head(10)


# In[ ]:


new_customer["Churn"] = LogReg.predict(new_customer[["Age", "Total_Purchase", "Years", "Num_sites"]])
new_customer["Churn Risk"] =  LogReg.predict_proba(new_customer[["Age", "Total_Purchase", "Years", "Num_sites"]])[:,1]
new_customer["Safe Likelihood"] = LogReg.predict_proba(new_customer[["Age", "Total_Purchase", "Years", "Num_sites"]])[:,0]

new_customer[["Names", "Company", "Total_Purchase", "Churn Risk"]].loc[new_customer["Churn"] == 1]


# What this demonstrates is how best to move from descriptive to predictive analytics, and hinting at prescriptive would involve calculating the impact of certain features, and adjusting the ones you see. For example, if you know that Cannon-Benson is about to experience a massive budget shortfall that will drop their buying power 23%, you can see how that would increase the risk of churn (and suggest next best actions)!

# ## Your turn! HUD Census Income Data
# In this next exercise, you'll create your own basic machine learning model!
# 
# Our data is a census set that has been used by HUD, and we'll be trying to predict whether a household's income is above or below $50k.
# 
# Follow along with me!

# In[ ]:


#Our libraries are already imported above, so we can skip that step!


# In[ ]:


#let's load in our data. 
HUD_df = pd.read_csv(<replace this>)


# In[ ]:


#we want to know how many people are at a low economic level (-50000)


# In[ ]:


#let's relabel that so that machine learning can do its magic...


# In[ ]:


#let's look at all of the possible columns


# In[ ]:


#for the sake of ease, let's drop all NA - this is usually a bad idea!
HUD_df.dropna(inplace = True)


# In[ ]:


#choose your favorite, and put it in a list. A list is designated by []
favorite_columns = []
X = HUD_df[favorite_columns]
y = 


# In[ ]:


#Train, Test Split so that we can build an EVALUATABLE model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[ ]:


#Let's build a classification model!

#import
#instantiate
#fit
#predict


# In[ ]:


#Now evaluate it using your model's '.score()' method, which takes your test set


# In[ ]:


#Evaluate it again using the classification report - pass it your test and pred
print(classification_report(y_test, y_pred))


# In[ ]:




