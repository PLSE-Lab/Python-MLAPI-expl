#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


data.head()


# In[ ]:


print(data.info())


# In[ ]:


data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
data.info()


# We can see that there are 11 vacancy data.

# In[ ]:


data.dropna(inplace = True)
data.info()


# Discard these 11 vacancy data

# In[ ]:


plt.figure(figsize = (11,5))
data.MonthlyCharges.plot.hist(color ='lightgreen')
#plt.hist(data.TotalCharges,10)
plt.title('MonthlyCharges', size = 15, color = 'purple')
plt.xlabel('Charges_groups', size = 13, color = 'b')
plt.xticks(size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.show()


# Look at the general distribution of MonthlyCharges.

# In[ ]:


charges_group = [[0,20]]
for i in range(6):
    charges_group.append([i+20 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(6):
    i_ = data.loc[(data.MonthlyCharges >= i*20) & (data.MonthlyCharges < (i+1)*20)]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(6), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(6), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(6), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Monthlycharges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer who has no partner',size = 13, color = 'c')
plt.show()


# With the increase of monthly charges,  the churn rate of customers is increasing. Therefore, we should reduce MonthlyCharges and reduce customer turnover.

# In[ ]:


plt.figure(figsize = (11,5))
data.TotalCharges.plot.hist(color ='lightgreen')
#plt.hist(data.TotalCharges,10)
plt.title('TotalCharges', size = 15, color = 'purple')
plt.xticks(size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.show()


# Look at the general distribution of TotalCharges.

# In[ ]:


charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000) & (data.Partner == 'Yes')]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and NotChurn per of customer who has partner',size = 16, color = 'c')
plt.show()


# In[ ]:


charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000) & (data.Partner == 'No')]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer who has no partner',size = 16, color = 'c')
plt.show()


# In[ ]:


charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000)]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer',size = 15, color = 'c')
plt.show()


# It can be seen that in the 7032 samples, the proportion of customers leaving  is decreasing as the TotalCharges increases. TIn connection with the above two diagrams that separate partner from customers without partner, it can be drawn that this rule is not related to whether the customer has partne. Thus, as long as we increase the TotalCharges of customer, customers will be able to retain customers if they are accustomed to our products. At the same time, when retaining customers, the TotalCharges of customers is higher, so the benefits to the company will double. Therefore, we can launch a strategy of giving customers more preferential services in the early stage service, so as to "suck" customers and let customers get used to the products of the company.

# In[ ]:


plt.figure(figsize = (8,8))
sns.set(style = 'whitegrid')
sns.countplot(data.SeniorCitizen, hue = data.Churn )
plt.show()


# The bar chart shows that the main customers of our company are new arrival customers, and the proportion of customers who leave these cities is much lower than that of customers who have settled in cities and use our products. As a result, the company can introduce relevant preferential policies for new customers coming to the city. For example, college students who come to urban schools give them special preferences so as to attract them to "get used to" our products and retain customers for a long time.

# In[ ]:


plt.figure(figsize = (10,8))

sns.set(style = 'whitegrid')
sns.boxplot(x = data.SeniorCitizen, y = data.MonthlyCharges, hue = data.Churn)

plt.title('Total Revenue by Seniors and Non-Seniors', color = 'orange', size = 15)


# Based on the above comparison:
# Although the old citizens have a small proportion of their customers, they spend more on the company. However, as we can see, their probability of disengagement is higher than that of young citizens.
# The numbers make sense: older people spend more time at home because they are retired or less stressed or relaxed, so they consume more television, which leads to more and more expensive meals.
# Therefore, we can introduce more preferential policies for the elderly residents, because their consumption level is higher, if this strategy works, it will greatly enhance the company's income.

# In[ ]:


plt.figure(figsize = (10,5))
sns.set(style = 'whitegrid')
sns.countplot(data.PaymentMethod, hue = data.Churn )


# The column chart shows that the customers who pay the "Electronic check" in the 7032 customer samples are much more likely to be divorced. This may be because the "Electronic check" payment method has some problems affecting the satisfaction of customers, for example, the process is too cumbersome. So you can try to find problems that may exist in the "Electronic check" payment method and improve them.

# In[ ]:


plt.figure(figsize = (15,5))
sns.countplot(data.InternetService, hue = data.Churn)


# There is a huge trend of customer churn in optical fiber services. Customers may be dissatisfied with the service. Companies can try to find problems that may exist in this service.

# In[ ]:


numerics = data[['tenure','MonthlyCharges', 'TotalCharges', 'Churn']]
plt.figure(figsize = (10,10))
sns.regplot(x = 'tenure', y = 'TotalCharges', data = numerics,color = 'c')
plt.title('Relationship between loyalty months and total revenue',color = 'orange',size = 15)


# Figure is a more regular triangle. The slope of one side on the triangle represents the highest consumer's monthly consumption level. These consumers have the highest consumption, but consumers with different cumulative consumption time form a side on the triangle. The slope below one edge of the X axis represents the bottom consumer's monthly consumption level. These consumers are the lowest, but consumers with different consumption time constitute a side below the triangle.
