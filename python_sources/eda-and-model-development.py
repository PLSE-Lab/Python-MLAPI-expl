#!/usr/bin/env python
# coding: utf-8

# ## About The Dataset

# Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
# 
# He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
# 
# Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
# 

# ### Features
# 1.battery_power: Total energy a battery can store in one time measured in mAh
# 
# 2.blue: Has bluetooth or not
# 
# 3.clock_speed: speed at which microprocessor executes instructions
# 
# 4.dual_sim: Has dual sim support or not
# 
# 5.fc: Front Camera mega pixels
# 
# 6.four_g: Has 4G or not
# 
# 7.int_memory: Internal Memory in Gigabytes
# 
# 8.m_dep: Mobile Depth in cm
# 
# 9.mobile_wt: Weight of mobile phone
# 
# 10.n_cores: Number of cores of processor
# 
# 11.pc: Primary Camera mega pixels
# 
# 12.px_height: Pixel Resolution Height
# 
# 13.px_width: Pixel Resolution Width
# 
# 14.ram: Random Access Memory in Mega Bytes
# 
# 15.sc_h: Screen Height of mobile in cm
# 
# 16.sc_w: Screen Width of mobile in cm
# 
# 17.talk_time: longest time that a single battery charge will last 
# 
# 18.three_g: Has 3G or not
# 
# 19.touch_screen: Has touch screen or not
# 
# 20.wifi: Has wifi or not
# 
# 21.price_range: This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).                            

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


project_data = pd.read_csv("../input/mobile-price-classification/train.csv")


# In[ ]:


x=project_data.shape
print("Number of Data points/Observations in train dataset are:-",x[0])
print("Number of Features in train dataset  are:-",x[1])


# In[ ]:


#sample of our Data
project_data.head(10)


# In[ ]:


#checking for the null values
project_data.isna().sum()


# Observation:-                     
# There are no NULL values in these features

# In[ ]:


#basic info like datatype of the features 
project_data.info()


# In[ ]:


#to find the corelation between the columns
corr=project_data.corr()
fig = plt.figure(figsize=(15,12))
r = sns.heatmap(corr, cmap='Purples')
r.set_title("Correlation ")


# Observations:-                       
# 1.Ram has a positive correlation with Price Range                  
# 2.We can also see that 4g and 3g features as  relation because most of        phones which support 4g also supports 3g
#           

# In[ ]:


#to know more about the features
project_data.describe()


# **Observations:-**                 
# 1.Max Battery power is 2000mah                          
# 2.75% of the phones have Dual sim avilability                      
# 3.50% of the mobile phones has  32gb of memory                        
# 4.most of the  mobile phones are screen touch enabled and supports 3g , 4g and are wifi enabled
# 

# 
# # EXPLORATORY DATA ANALYSIS (EDA) #

# ## Univariate Analysis

# In[ ]:


#pie chart representation
x=project_data['dual_sim'].value_counts()
labels='Supports Dualsim: '+str(x[1]),'Does not support Dualsim:- '+str(x[0])
sizes=[x[1],x[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Observations:-                       
# 1.1019 mobile phones are Dualsim Enabled                   
# 2.981  phones does not support  Dualsim feature

# In[ ]:


#pie chart representation
x=project_data['four_g'].value_counts()
labels='Supports 4 g: '+str(x[1]),'Does not support 4 g:- '+str(x[0])
sizes=[x[1],x[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Observations:-                       
# 1.1043  phones are 4g Enabled                   
# 2.957   phones does not support  4g

# In[ ]:


#pie chart representation
x=project_data['three_g'].value_counts()
labels='Supports 3 g: '+str(x[1]),'Does not support 3 g:- '+str(x[0])
sizes=[x[1],x[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Observations:-                       
# 1.1523  phones are 3g Enabled                   
# 2.447   phones does not support  3g                                        
# Most of the phones are 3g enabled

# In[ ]:


#pie chart representation
x=project_data['wifi'].value_counts()
labels='Wifi Enabled: '+str(x[1]),'Does not support Wifi:- '+str(x[0])
sizes=[x[1],x[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Observations:-                       
# 1.1014 mobile phones supports WIFI                  
# 2.986  mobile phones does not support  WIFI

# In[ ]:


#pie chart representation
x=project_data['touch_screen'].value_counts()
labels='touchscreen Enables: '+str(x[1]),'Does not support Touchscreen:- '+str(x[0])
sizes=[x[1],x[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Observations:-                       
# 1.1006  phones have screen touch feature                  
# 2.994   phones does not have screen touch feature

# In[ ]:


sns.countplot(x='fc', data=project_data)
plt.show()


# Observations:-                       
# Most of the phones have front camera megapixels =0 means many of the phones does not have camera

# In[ ]:


sns.countplot(x='n_cores', data=project_data)
plt.show()


# Observations:-                       
# There is less variation in number of cores

# In[ ]:


sns.countplot(x='ram', data=project_data)
plt.show()


# Observations:-                       
# ram as lot of Unique values

# ## Bivariate Anlysis

# In[ ]:


sns.boxplot(y='clock_speed',x='price_range',data=project_data)
plt.show()


# Observations:-                       
# clock speed does'nt have much impact on the price of a phone

# In[ ]:


sns.boxplot(x='touch_screen',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Touch screen phones are more costlier than the other phones

# In[ ]:


sns.boxplot(x='dual_sim',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Phones that support dual sim are more costlier than the other which don't

# In[ ]:


sns.boxplot(x='four_g',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Phones that 4g enabled are more costlier than the other phones

# In[ ]:


sns.boxplot(x='three_g',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Phones that 3g enabled are more costlier than the phones that does'nt support 3g 

# In[ ]:


sns.boxplot(x='wifi',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Phones that has WIFI are more costlier than the other which does not have WIFI

# In[ ]:


sns.boxplot(x='blue',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# Phones that has bluetooth are more costlier than the other which don't

# In[ ]:


sns.boxplot(x='n_cores',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# number of cores doesn't have much impact on the price range of a phone

# In[ ]:


sns.boxplot(x='fc',y='price_range',data=project_data)
plt.show()


# Observations:-                       
# front camera pixels doesn't have much impact on the price range of a phone

# In[ ]:


sns.boxplot(x='ram',y='price_range',data=project_data)


# Observation:-                           
# There are many different values for ram but we can clearly see that has ram increases price range also increases

# In[ ]:


# pairwise scatter plot: Pair-Plot.
sns.set_style("whitegrid");
sns.pairplot(project_data,hue='price_range',vars=['n_cores', 'dual_sim','four_g', 'ram','touch_screen','wifi','talk_time','three_g'])
plt.legend()
plt.show() 


# Observations:-                       
# ram and number of number of cores can easily differentiate price ranges of the mobile phones 

# ***Other features plots where removed because no proper observations were made from those plots***

# ## Making Data Model Ready

# In[ ]:


# considering only those features that has impact on price_range from our anlysis
x = project_data[['three_g','battery_power','blue','dual_sim','four_g','px_height','px_width','ram','touch_screen','wifi','fc']]
y = project_data['price_range']
print("shape of x train is" ,x.shape)
print("shape of y train is" ,y.shape)


# In[ ]:



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print("shape of x train is: ",x_train.shape)
print("shape of y test is-" ,y_test.shape)


# # Applying Models on the Data

# ### Logistic Regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
model = LogisticRegression()
model.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))


# In[ ]:


print("Train Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


print("Test Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# ### Support Vector Classifier SVC

# In[ ]:


from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))


# In[ ]:


print("Train Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


print("Test Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# ### Decision Tree Classifier classifier

# In[ ]:



from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(min_samples_split=10)#we use  min sample split value for preventing model from overfitting 
model.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))


# In[ ]:


print("Train Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


print("Test Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(min_samples_split=10)
model.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))
print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))


# In[ ]:


print("Train Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_train_pred,y_train), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


print("Test Confusion Matrix")
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test_pred,y_test), annot=True,annot_kws={"size": 16}, fmt='g')
plt.show()


# In[ ]:


project_data.columns


# # SUMMARY:

# **EDA SUMMARY:-**                                      
# *ram has direct impact on the price range of the phones*              
# *Features like                      
#  1.Dual sim                      
#  2.wifi                   
#  3.4g                              
#  4.3g                          
#  5.Touch screen                        
# Has more impact on the phones prices*
# 

# In[ ]:


#Models Summary
#http://zetcode.com/python/prettytable/
from prettytable import PrettyTable
    
x = PrettyTable()
x.field_names = ["Model","Test Accuracy"]
x.add_row(["Logistic Regression(LR)",81.83])
x.add_row(["Suppoer Vector Classifier(SVC)",86.6])
x.add_row(["Decision Tree Classsifier",82.6])
x.add_row(["Random Forest",85.3])
print(x)


# **Support Vector classifier has more accuracy than other classification models**
