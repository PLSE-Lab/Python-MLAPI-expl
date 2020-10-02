#!/usr/bin/env python
# coding: utf-8

# In this Exploratory Data Analysis I will go through the Beauty Data Set and visualize the connection between the interesting variables.
# In the end I will examine the factor "beauty" when it comes to wages.

# In[ ]:


import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df= pd.read_csv('../input/beauty.csv', delimiter=',')


# In[ ]:


df.head()


# In[ ]:


#checking for na
df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True,cmap="YlGnBu")


# We can see slight positive correlations between wage and expertise, wage and education, married and exper, female and service, education and service and a slightly negative correlation between wage and females, education and exper, female and exper and female and married

# In[ ]:


sns.pairplot(df[['wage', 'female', 'educ',
       'exper']]);


# In[ ]:


plt.figure(1 , figsize = (15 , 6))
for gender in [1 , 0]:
    plt.scatter(x = 'wage' , y = 'exper' , data = df[df['female'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Wage in thousands/month'), 
plt.ylabel('Years of Expertise') 
plt.title('Level of Expertise compared to Wage with the distinction gender')
plt.legend(['Female','Male'])
plt.show()


# It looks like men tend to earn more the more expertise they gain, while women earn the same

# In[ ]:


w_dev_wage = df[df['female']==1]['wage'].groupby(df['exper']).mean()


# In[ ]:


m_dev_wage = df[df['female']==0]['wage'].groupby(df['exper']).mean()


# In[ ]:


plt.figure(1 , figsize = (15 , 6))
sns.lineplot(data= w_dev_wage)
sns.lineplot(data= m_dev_wage)
plt.xlabel('Years of Expertise'), 
plt.ylabel('Mean Wage in thousand/Month') 
plt.title('Years of Expertise compared to Wage with the distinction gender')
plt.legend(['Female','Male'])
plt.show()

We can see that there is one outlier with a high wage and low exper
# In[ ]:


g = sns.PairGrid(df[['female', 'wage', 'educ',
       'exper']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=8);


# The densest areas are workers with low exper and low wages

# In[ ]:


data = df[["wage","educ"]].groupby(["educ"],as_index=False).mean().sort_values(by="educ", ascending=False)


# In[ ]:


sns.barplot(x="educ",y="wage",data=data)


# The married workers tend to earn more. Therefore we will look into that variable

# In[ ]:


df.married.value_counts()/df.married.size


# Almost 70% of the people in the dataset are married.

# In[ ]:


married_ppl = df[df.married ==1]
married_ppl.female.value_counts()/ married_ppl.female.size


# Out of the married people about 3/4 are male. 

# Therefore we need to take into consideration that the variable married is strongly correlated to male and therefore I am going to take a closer look at males only

# In[ ]:


df[["female","service"]].groupby("female").mean()


# We can see that out of the service workers far more are female, since service and wage don't have a very strong relationship on the first look, we will continue with the gender

# In[ ]:


sns.boxplot(x="female",y="wage",data=df)


# We can see that males tend to earn more and have a bigger range of wage if we don't take the one best-earning female into consideration

# In[ ]:


sns.boxplot(x="female",y="wage",data=df, showfliers=False)


# When we kick out the outliers we can see it more clearly

# In[ ]:


sns.regplot(x="educ", y="exper", data=df)


# We can see that the more education someone has, the less expertise the person has.

# In[ ]:


df["looks"].value_counts().sort_values(ascending=False)


# In[ ]:


df["looks"].value_counts().sort_values(ascending=False).plot.bar()


# Since the dataset is called "Beauty" I also want to take a look into the variable "beauty" altough it doesn't seem to have a great impact on the wage. After a look into the count of each variable we see that there are many more "average looking people"(3) in the dataset and close to zero extraordinary looking (1 or 5) people.Of course there are many more "average looking" people than extraordinary looking, but we still have to be critical whether such a dataset is meaningdful. Also we need to take in mind that beauty is extremely biased. Nevertheless I have to diagrams that show what the correlation indicates: that beauty doesn't play a role.

# In[ ]:


sns.jointplot(x="looks",y="wage", data = df, kind="hex")


# In[ ]:


sns.regplot(x="looks",y="wage", data = df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




