#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import missingno as mno
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


df.head(
)


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df['Partner'] = df["Partner"].map({'Yes':1, 'No':0})


# In[ ]:


df['Dependents'] = df["Dependents"].map({'Yes':1, 'No':0})


# In[ ]:


df['PhoneService'] = df["PhoneService"].map({'Yes':1, 'No':0})


# In[ ]:


df["InternetService"] = df["InternetService"].map({'DSL':1, 'Fiber optic':2, 'No':0})


# In[ ]:


df["MultipleLines"] = df["MultipleLines"].map({"No phone service":2, "Yes":1, "No":0})


# In[ ]:


df["OnlineSecurity"] = df["OnlineSecurity"].map({'No internet service':2, 'Yes':1, 'No':0})


# In[ ]:


df["OnlineBackup"] = df["OnlineBackup"].map({'Yes':1, 'No':0, 'No internet service':2})


# In[ ]:


df["DeviceProtection"] = df["DeviceProtection"].map({'Yes':1, 'No':0, 'No internet service':2})


# In[ ]:


df["StreamingTV"] = df["StreamingTV"].map({'Yes':1, 'No':0, 'No internet service':2})


# In[ ]:


df["StreamingMovies"] = df["StreamingMovies"].map({'Yes':1, 'No':0, 'No internet service':2})


# In[ ]:


df["TechSupport"] = df["TechSupport"].map({'Yes':1, 'No':0, 'No internet service':2})


# In[ ]:


df["Contract"] = df["Contract"].map({'Month-to-month':0, 'One year':1, 'Two year':2})


# In[ ]:


df["PaperlessBilling"] = df["PaperlessBilling"].map({'Yes':1, 'No':0})


# In[ ]:


df["gender"] = df["gender"].map({'Female':0, 'Male':1})


# In[ ]:


df["PaymentMethod"] = df["PaymentMethod"].map({'Electronic check':0, 'Mailed check':1, 'Bank transfer (automatic)':2,
       'Credit card (automatic)':3})


# In[ ]:


df["Churn"] = df["Churn"].map({'Yes':1, 'No':0})


# In[ ]:


df


# In[ ]:





# In[ ]:


flatui = ["#ff3838", "#ff6b81", "#a4b0be", "#ffa502","#7bed9f", "#2ed573"]
plt.figure(figsize=(14,15))
sns.heatmap(df.corr(), annot=True,cmap= flatui, fmt= '.2f', linewidths=0.1 )
plt.show()


# In[ ]:


df.groupby('SeniorCitizen', as_index=False)['Churn'].mean()


# In[ ]:


df.groupby('InternetService', as_index=False)['Churn'].mean()


# In[ ]:


df.groupby('PaperlessBilling', as_index=False)['Churn'].mean()


# In[ ]:


df.groupby('Churn').mean()


# In[ ]:


sns.distplot(df['Churn'],rug=True)
plt.show()


# In[ ]:


df.head()


# In[ ]:


pd.crosstab(df.tenure,df.Churn).plot(kind="bar",figsize=(15,8),color=['blue','red' ])
plt.title('Churn for tenure')
plt.xlabel('tenure')
plt.ylabel('Churn Frequency')
plt.show()


# In[ ]:


pd.crosstab(df.MonthlyCharges, df.Churn).plot(kind='line', figsize=(15,10), color=['green','red'])
plt.title('Churn for Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel("Churn Frequncy")
plt.show()


# In[ ]:


pd.crosstab(df.gender,df.Churn).plot(kind="bar",figsize=(10,5),color=['cyan','coral' ])
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Churn", "Yes Churn"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.catplot(x="Contract", y="MonthlyCharges", hue="Churn", kind="box", data=df, height=5, aspect=2 )


# In[ ]:




