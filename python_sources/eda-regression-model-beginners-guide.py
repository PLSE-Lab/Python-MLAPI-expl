#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc


# In[ ]:


sns.set()


# In[ ]:


data=pd.read_csv(r"../input/insurance/insurance.csv")


# In[ ]:


data.head(10)


# # Handling Missing Values

# In[ ]:


data.isnull().sum()#No Missing Values


# # Descriptive Statistics

# In[ ]:


data.describe().T


# In[ ]:


data['smoker'].value_counts()


# In[ ]:


plt.subplot(1,2,1)
plt.pie(data['smoker'].value_counts(),labels=data['smoker'].value_counts().index,autopct="%.1f%%")
plt.title('Smoker vs Non-Smoker Count')

plt.subplot(1,2,2)
plt.pie(data['sex'].value_counts(),labels=data['sex'].value_counts().index,autopct="%.1f%%")
plt.title('Male vs Female Count')
plt.show()


# In[ ]:


plt.figure(figsize=(4,4))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True,linewidths=0.5)
plt.show()


# In[ ]:


plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
sns.boxplot(data['age'])
plt.subplot(3,1,2)
sns.boxplot(data['bmi'])
plt.subplot(3,1,3)
sns.boxplot(data['charges'])
plt.show()


# In[ ]:


sc.skew(data['charges'])
#Charges data is highly skewed


# 1. Age is uniformly distributed
# 
# 2. Bmi has few extreme values
# 
# 3. Charges data is highly skewed

# In[ ]:


gender_smoker=pd.crosstab(data['sex'],data['smoker'])
gender_smoker


# In[ ]:


gender_smoker.plot(kind="bar")
plt.title("Smoker(Yes/No) Count by Gender")
plt.show()


#   *More male smokers than female*

# In[ ]:


sns.pairplot(data)


# In[ ]:


#LET'S GRAB THE USEFUL ONES


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.scatterplot(data['bmi'],data['charges'],hue=data.smoker)
plt.subplot(1,3,2)
sns.scatterplot(data['age'],data['charges'],hue=data.smoker)
plt.subplot(1,3,3)
sns.scatterplot(data['children'],data['charges'],hue=data.smoker)
plt.show()


# *1.Clearly, smokers are paying much extra*
# 
# *2.Relation of bmi & charges is not prominent for non-smokers.
# While for smokers, we do see a linear relationship which suggests: smokers with high bmi are definitely paying much more.*
# 
# *3. A linear relation of age & charges is prominent for non-smokers. Few datapoints are random & towards the higher charges, maybe due to bmi as additional factor*
# 

# In[ ]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.scatterplot(data[data["sex"]=="female"]['age'],data[data["sex"]=="female"]['charges'],hue=data[data["sex"]=="female"]["smoker"])
plt.title("Female CHARGES vs AGE")
plt.subplot(1,2,2)
sns.scatterplot(data[data["sex"]=="male"]['age'],data[data["sex"]=="male"]['charges'],hue=data[data["sex"]=="male"]["smoker"])
plt.title("Male CHARGES vs AGE")
plt.show()


# *Male & Female CHARGES vs AGE has similar pattern*

# In[ ]:


sns.distplot(data['age'])
plt.title("Distribution of Age")
plt.show()


# In[ ]:


young_smokers=data[(data['age']<20) & (data['smoker']=='yes')]
young_smokers.head()


# *We have got smokers of age less than 20*

# In[ ]:


sns.boxplot(x=young_smokers.charges)
plt.title("Charges range of young smokers < 20 yrs of age")
plt.show()


# In[ ]:


sns.boxplot(x=young_smokers['region'],y=young_smokers['charges'])
plt.title("Regionwise Charges Distribution for smokers of age < 20yrs ")
plt.show()


# *In SOUTH EAST region, the young smokers are paying more as compared to other regions*

# In[ ]:


young_smokers.groupby(['region','smoker']).sum()['charges']


# In[ ]:


young_smokers[young_smokers.smoker=='yes']['region'].value_counts()


# *Let's go deep into REGION-CHARGES relation*

# In[ ]:


sns.boxplot(x=data['region'],y=data['charges'],hue=data.smoker)
plt.title("Regionwise Charges Distribution")
plt.show()


# In[ ]:


sns.countplot(data.region,hue=data.smoker)


# In[ ]:


higher_charge=data[data.charges>16639.912515]


# In[ ]:


plt.figure(figsize=(10,12))
plt.subplot(3,1,1)
sns.countplot(higher_charge['region'],hue=higher_charge['smoker'])
plt.subplot(3,1,2)
sns.countplot(higher_charge['sex'],hue=higher_charge['smoker'])
plt.subplot(3,1,3)
sns.boxplot(higher_charge['age'],color="0.25")
plt.show()


# *Charges Outliers are predominantly due to MALE-SOUTHEAST smokers* 

# In[ ]:


higher_charge.head(2)


# In[ ]:


higher_charge['BMI Category']=pd.cut(higher_charge.bmi,bins=[0,18.5,25,30,53],labels=['Underweight','Healthy weight','Overweight','Obese']).values
higher_charge['Age Category']=pd.cut(higher_charge.age,bins=[18,30,42,54,64],labels=['Young Adult','Middle Adult','Senior Adult','Elder']).values
higher_charge


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(higher_charge['age'],higher_charge['charges'],hue=higher_charge['BMI Category'])
plt.subplot(1,2,2)
sns.scatterplot(higher_charge['age'],higher_charge['charges'],hue=higher_charge['smoker'])
plt.show()


# *The higher charges data is pulled CLEARLY by OBESE i.e bmi>30 and they turned out to be smoker too(about>140), which is why they are paying high insurance*

# In[ ]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(higher_charge['BMI Category'])
plt.subplot(1,2,2)
sns.countplot(higher_charge['Age Category'])
plt.show()


# *Charges(Senior Adult> Middle Adult> Young Adult> Elder)*
# 

# In[ ]:


sns.countplot(higher_charge.sex,hue=higher_charge['BMI Category'])
plt.show()


# *MALE OBESE is significantly higher than FEMALE*

# In[ ]:


plt.figure(figsize=(20,5))
sns.countplot(data.age,hue=data.smoker)
plt.title("AGE COUNT")
plt.show()


# *Maximum insurances are done in 18-19yrs of age*

# # Hypothesis Testing

# 1. Effect of gender on smoking habits SEX-SMOKER

# In[ ]:


Ho="Gender has no effect on smoking habits"
Ha="Gender has effect on smoking habits"
chi,p_value,dof,expected=sc.chi2_contingency(gender_smoker)
if (p_value>0.05):
    print(Ho)
else:
    print(Ha)


# In[ ]:


chi
#tabular_chi=3.84,thus reject Ho


# In[ ]:


gender_smoker#clearly we have more male smokers


# 2. Effect of region on Charges: REGION-CHARGES

# In[ ]:


Ho="Region has no effect on charges"
Ha="Region has effect on charges"
sample=data.sample(100)
charges=pd.cut(sample['charges'].sort_values(ascending=False),bins=15)
chi2,p,ddof,ex=sc.chi2_contingency(pd.crosstab(sample['region'],charges))


# In[ ]:


if (chi2<58.124):
    print(Ho)
else:
    print(Ha)


# # Central Limit Theorm On CHARGES

# In[ ]:


sns.distplot(data['charges'])


# In[ ]:


#Forming sample dataset randomly
import random
n=1000
ks=100
sample_dataset=[]
for i in range(0,n):
    sample_dataset.append(random.choices(data["charges"],k=ks))


# In[ ]:


#Calculating mean from means of sample sets: SAMPLE MEAN/s_mean
sample_means=[]
for i in sample_dataset:
    sample_means.append(np.mean(i))
s_mean=np.mean(sample_means)
s_mean


# In[ ]:


#ACTUAL DATA MEAN
data['charges'].mean()


# *Sample mean is almost equal to actual data mean*

# In[ ]:


s_var=np.var(sample_means)
s_std=np.std(sample_means)


# In[ ]:


s_var


# In[ ]:


data['charges'].var()/(100)#Actual variance


# In[ ]:


s_std


# In[ ]:


data['charges'].std()/np.sqrt(100)#Actual standard_deviation


# *Sample statistics is almost equal to population statistics for CHARGES*

# # Data Preprocessing

# In[ ]:


data[data['charges']>16639.912515]
#this comprises of about 25% data


# In[ ]:


ub_bmi=34.595000+1.5*(34.595-26.315)#upperboundary_bmi


# In[ ]:


data['bmi']=np.where(data['bmi']>ub_bmi,ub_bmi,data['bmi'])#replacing outliers in bmi with ub_bmi


# In[ ]:


data['bmi'].plot(kind="box")
plt.show()#No outliers


# In[ ]:


data=pd.get_dummies(data)#encoding CATEGORICAL DATA


# In[ ]:


data.head()


# In[ ]:


sns.heatmap(data.corr(),cmap="cool")#High corr with smoker


# # Linear Regression

# In[ ]:


#Splitting data into train & test
from sklearn.model_selection import train_test_split
x=data.drop(['charges'],axis=1)
y=data['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


#Training the model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


#Predicting the values
y_pred=lr.predict(x_test)
y_pred_train=lr.predict(x_train)


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score#TO TEST THE MODEL


# In[ ]:


mean_squared_error(y_test,y_pred)


# In[ ]:


np.sqrt(35746516.8773678)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


r2_score(y_train,y_pred_train)

