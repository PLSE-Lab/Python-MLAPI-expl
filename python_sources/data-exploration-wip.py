#!/usr/bin/env python
# coding: utf-8

# # TOC TOC
# 
# 1. [Importing neccesary modules](#importing-neccesary-modules)
# 2. [Loading data](#loading-data)
# 3. [Exploring data](#exploring-data)
#     1. [Application Train](#application-train)

# # Importing neccesary modules <a name="importing-neccesary-modules"></a>
# 
# In this section I wil make the import necessary for this notebook.

# In[2]:


#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os


print(os.listdir("../input"))


# # Loading data<a name="loading-data"></a>
# 
# In this section we will load the necessary data.

# In[3]:


app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')


# # Exploring data <a name="exploring-data"></a>
# ## Application Train <a name="application-train"></a>
# The first step to do is the exploring data. According to the information provide by Home Credit the main table is the application\_{train|test}.csv file. So, here we go.

# In[4]:


print (app_train.columns)
print(app_train.head(5))


# Ok, quickly I am see that we have 122 attributes (uff!) And I am already seeing a NaN value.

# In[5]:


pd.set_option('display.max_rows', 122)
nulls_data = app_train.isnull().sum().sum()
print("There are {} null data on the dataset".format(nulls_data))
print(app_train.isnull().sum())


# We can see that we have a lot of missing value between EXT\_SOURCE\_1 and EMERGENCYSTATE\_MODE. Thene between AMT\_REQ\_CREDIT\_BUREAU\_HOUR and AMT\_REQ\_CREDIT\_BUREAU\_YEAR.
# 
# What about this data? What is the meaning?

# In[6]:


print(app_train.info(verbose=True))


# The variable TARGET show me the if the loan was repayed or not, if the client have payment difficulties. Let us look what is the distribution of the TARGET variable on the dataset

# In[7]:


app_train.TARGET.value_counts().plot.bar()


# In[8]:


target_true = app_train[app_train["TARGET"] == 1].shape[0]
target_false = app_train[app_train["TARGET"] == 0].shape[0]
total = target_true + target_false
print(total)
print("target true: {}".format(target_true))
print("target false: {}".format(target_false))
print("Payment difficulties {}".format(target_true/(total)*100) )


# We can see that the only the 8.07% had payment dificulties, and 91.92% were not (that is not bad is not?)
# 
# ## Gender <a name="gender"></a>
# Continue with the gender exploration.

# In[9]:


print(app_train.CODE_GENDER.head(5))
app_train.CODE_GENDER.value_counts().plot.bar()
app_train_total = app_train.shape[0]
female_total = app_train[app_train.CODE_GENDER == 'F'].shape[0]
male_total = app_train[app_train.CODE_GENDER == 'M'].shape[0]
xna_total = app_train[app_train.CODE_GENDER == 'XNA'].shape[0]
print("Gender XNA: {}?".format(app_train[app_train.CODE_GENDER == 'XNA'].shape[0])) # What is XNA gender?
print("% of Female target: {}".format(female_total/total*100))
print("% of Male target: {}".format(male_total/total*100))


# 
# Let's go to see the distribution of the repayed loan for gender

# In[10]:


tab = pd.crosstab(app_train.TARGET, app_train.CODE_GENDER)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# We can see that the gender female and male gender have almost the same difficulties to pay the loan. On the other hand, female gender represents the highest percentage that did not have any problem to pay the loan. In any case, there is no marked difference.

# In[11]:


fig = plt.figure(figsize=[5,5])
# Female
ax = fig.add_subplot(121)
female = app_train[app_train.CODE_GENDER == 'F']
female.TARGET.value_counts().plot.bar()
ax.set_title("Female loans")
# Male
ax = fig.add_subplot(122)
male = app_train[app_train.CODE_GENDER == 'M']
male.TARGET.value_counts().plot.bar()
ax.set_title("Male loans")


# In[12]:


female_true = female[female.TARGET == 1].shape[0]
female_false = female[female.TARGET == 0].shape[0]

male_true = male[male.TARGET == 1].shape[0]
male_false = male[male.TARGET == 0].shape[0]

print("{:.2f}% female with difficulties".format(female_true/female_total*100))
print("{:.2f}% female with not difficulties".format(female_false/female_total*100))
print("{:.2f}% male with difficulties".format(male_true/male_total*100))
print("{:.2f}% male with not difficulties".format(male_false/male_total*100))
# print(female_true, female_false)
# print(male_true, male_false)


# We learn that the just the 7% of women have any difficulties to pay their loan, and just the 10.14% of men have difficulties

# ## Name Contract Type
# 
# Now, I will study the NAME\_CONTRACT\_TYPE feature

# In[13]:


print(app_train.NAME_CONTRACT_TYPE.head(5))


# In[14]:


app_train.NAME_CONTRACT_TYPE.value_counts().plot.bar()
revolving_loans = app_train[app_train.NAME_CONTRACT_TYPE == 'Revolving loans'].shape[0]
cash_loans = app_train[app_train.NAME_CONTRACT_TYPE == 'Cash loans'].shape[0]
print("Cash loans represent the {:.2f}%".format(cash_loans/app_train_total*100))
print("Revolving loans represent the {:.2f}%".format(revolving_loans/app_train_total*100))


# In[15]:


tab = pd.crosstab(app_train.CODE_GENDER, app_train.NAME_CONTRACT_TYPE)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In[16]:


tab = pd.crosstab(app_train.TARGET, app_train.NAME_CONTRACT_TYPE)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In this section we learn:
# 1. There are 2 kind of loans: Revolving loans and Cash Loans
# 2. The most popular loans is the Cash Loans (90.48%) (versus Revolving loans -> 9.52% )
# 3. Both female loans and male loans represent the same (aprox) porcentage of kind of contract 
# 4. Both the target 0 and 1 represent the same (aprox) porcentage of kind of contract 

# ## FLAG_OWN_CAR
# 
# Now we will study the FLAG_OWN_CAR variable. It seems that represent if the people that have a loan, have their own car.

# In[17]:


print(app_train.FLAG_OWN_CAR.head(5))


# In[18]:


app_train.FLAG_OWN_CAR.value_counts().plot.bar()
car_yes = app_train[app_train.FLAG_OWN_CAR == 'Y'].shape[0]
car_no = app_train[app_train.FLAG_OWN_CAR == 'N'].shape[0]
print("Own Car Yes {:.2f}%".format(car_yes/app_train_total*100))
print("Not Own Car {:.2f}%".format(car_no/app_train_total*100))


# In[19]:


tab = pd.crosstab(app_train.TARGET, app_train.FLAG_OWN_CAR)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In[20]:


tab = pd.crosstab(app_train.CODE_GENDER, app_train.FLAG_OWN_CAR)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# We learn:
# 
# 1. From the sample, the 34.01% have own car, and the 65.99% have not own car
# 2. There are not differences about if a person has or not a car and if he / she will have difficulties to pay the loan.
# 3. Thre are more male than female that have own car
# 

# ## FLAG_OWN_REALTY
# This variable tell me if the client owns a house or flat

# In[21]:


print(app_train.FLAG_OWN_REALTY.head(5))


# In[22]:


app_train.FLAG_OWN_REALTY.value_counts().plot.bar()
house = app_train[app_train.FLAG_OWN_REALTY == 'Y'].shape[0]
flat = app_train[app_train.FLAG_OWN_REALTY == 'N'].shape[0]
print("Owns house {:.2f}%".format(house/app_train_total*100))
print("Owns flat {:.2f}%".format(flat/app_train_total*100))


# In[23]:


tab = pd.crosstab(app_train.TARGET, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In[24]:


tab = pd.crosstab(app_train.CODE_GENDER, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In[25]:


tab = pd.crosstab(app_train.FLAG_OWN_CAR, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# We learn:
#     
#  1. The 69.37% of people's sample is owns house. the 30.63% is owns flat.
#  2. For people with difficulties, the majority is own of house. Same for owns flat.
#  3. The same occur for male and female

# ## CNT_CHILDREN
# Now will study this variable.

# In[26]:


print(app_train.CNT_CHILDREN.head(5))


# In[46]:


app_train.CNT_CHILDREN.value_counts().plot.bar()
print(app_train[app_train.CNT_CHILDREN == 19].shape)


# In[57]:


app_train['CNT_CHILDREN'] = app_train.apply(lambda x: 3 if x.CNT_CHILDREN > 2.0 else x.CNT_CHILDREN, axis=1)

# tdr['Team1_Score'] = tdr.apply(lambda r: r.WScore if r.Pred == 1.0 else r.LScore, axis=1)


# In[58]:


tab = pd.crosstab(app_train.TARGET, app_train.CNT_CHILDREN)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




