#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import where as IF


# 1. Import claims_data.csv and cust_data.csv which is provided to you and combine the two datasets appropriately to create a 360-degree view of the data. Use the same for the subsequent questions.

# In[ ]:


claims = pd.read_csv("../input/eda-on-insurance-claim-dataset/claims.csv")
demo = pd.read_csv("../input/demographics/demo.csv")
demo.head(2)


# In[ ]:


claims.head(2)


# In[ ]:


comb_df = pd.merge(right = claims,
                   left = demo, 
                   right_on = "customer_id", 
                   left_on = "CUST_ID",
                   how = "outer"
                  )
comb_df.drop(columns = ["customer_id"], inplace = True)
comb_df.head(2)


# In[ ]:


comb_df.head()


# 2. Perform a data audit for the datatypes and find out if there are any mismatch within the current datatypes of the columns and their business significance.
# 3. Convert the column claim_amount to numeric. Use the appropriate modules/attributes to remove the $ sign.

# In[ ]:


comb_df["DateOfBirth"] = pd.to_datetime(comb_df.DateOfBirth, format = "%d-%b-%y")
comb_df.loc[(comb_df.DateOfBirth.dt.year > 2020),"DateOfBirth"]=comb_df[comb_df.DateOfBirth.dt.year > 2020]["DateOfBirth"].apply(lambda x: x - pd.DateOffset(years=100))
comb_df["claim_date"] = pd.to_datetime(comb_df.claim_date, format = "%m/%d/%Y")
comb_df["Contact"] = pd.to_numeric(comb_df.Contact.str.replace("-",""),downcast='float')
comb_df["claim_amount"] = pd.to_numeric(comb_df.claim_amount.str.replace("$",""),downcast='float')
comb_df.head(2)


# 4. Of all the injury claims, some of them have gone unreported with the police. Create an alert flag (1,0) for all such claims.

# In[ ]:



comb_df["flag"] = IF(comb_df.police_report == "No", 0 ,
                    IF(comb_df.police_report == "Yes", 1, np.nan))
comb_df.drop(columns = ["police_report"], inplace = True)


# 5. One customer can claim for insurance more than once and in each claim, multiple categories of claims can be involved. However, customer ID should remain unique. Retain the most recent observation and delete any duplicated records in the data based on the customer ID column.

# In[ ]:


comb_df = comb_df.groupby('CUST_ID').first().reset_index(drop = True)


# 6. Check for missing values and impute the missing values with an appropriate value. (mean for continuous and mode for categorical)

# In[ ]:


comb_df.head()


# In[ ]:


comb_df["incident_cause"].isna().sum()


# In[ ]:


cat_col = ["gender","State","Segment","incident_cause","claim_area","claim_type","fraudulent","flag"]
con_col = ["claim_amount"]


# In[ ]:


for col in cat_col:
    comb_df[col] = comb_df[col].fillna(comb_df[col].mode()[0])
comb_df[con_col] = comb_df[con_col].fillna(comb_df[con_col].mean())
comb_df.head()


# In[ ]:


# comb_df["incident_cause"].fillna(0, inplace =True)
comb_df["State"].isna().sum()


# 7. Calculate the age of customers in years. Based on the age, categorize the customers according to the below criteria
#         Children < 18
#         Youth 18-30
#         Adult 30-60
#         Senior > 60

# In[ ]:


comb_df["Age"] = round((comb_df.claim_date - comb_df.DateOfBirth).apply(lambda x: x.days)/365.25, 2)


# In[ ]:


comb_df["Age_grp"] = IF(comb_df.Age < 18, "Children",
                        IF(comb_df.Age < 30, "Youth",
                         IF(comb_df.Age < 60, "Adult",
                          IF(comb_df.Age < 100, "Senior", "NaN"
                           
                          )
                         )
                        )
                       )
comb_df["Age_grp"] = comb_df["Age_grp"].fillna(comb_df["Age_grp"].mode())
comb_df.groupby(by = "Age_grp").count()
# comb_df.head()


# What is the average amount claimed by the customers from various segments?

# In[ ]:


comb_df.groupby(by = "Segment")[["claim_amount"]].mean()


# What is the total claim amount based on incident cause for all the claims that have been done at least 20 days prior to 1st of October, 2018.

# In[ ]:


comb_df.loc[comb_df.claim_date < "2018-09-10",:].groupby("incident_cause")["claim_amount"].sum().add_prefix("total_")


# 10. How many adults from TX, DE and AK claimed insurance for driver related issues and causes?

# In[ ]:


comb_df.loc[(comb_df.incident_cause.str.lower().str.contains("driver") 
             & ((comb_df.State == "TX") | (comb_df.State == "DE") | (comb_df.State == "AK"))),:].groupby(by = "State")["State"].count()


# 11. Draw a pie chart between the aggregated value of claim amount based on gender and segment. Represent the claim amount as a percentage on the pie chart.

# In[ ]:


f1 = comb_df.groupby(by = ["gender","Segment"])["claim_amount"].sum().reset_index()
f1.head()


# In[ ]:


res = f1.pivot(index = "Segment", columns = "gender", values = "claim_amount")
res


# In[ ]:


res.T.plot(kind = "pie", subplots = True, legend = False, figsize = (15,8))
plt.show()


# 12. Among males and females, which gender had claimed the most for any type of driver related issues? E.g. This metric can be compared using a bar chart

# In[ ]:


f2 = comb_df.loc[(comb_df.incident_cause.str.lower().str.contains("driver"))].groupby(by = "gender")[["gender"]].count().add_prefix("CountOf_").reset_index()
f2


# In[ ]:


sns.barplot(x = "gender", y = "CountOf_gender", data = f2 )
plt.show()


# 13. Which age group had the maximum fraudulent policy claims? Visualize it on a bar chart.

# In[ ]:


comb_df.head()


# In[ ]:



comb_df.groupby(by = "Age_grp")[["fraudulent"]].count()


# In[ ]:


comb_df[(comb_df.Age_grp == np.nan)]


# In[ ]:


val = comb_df['Age_grp'].mode()[0]
print(val)
comb_df.loc[:,"Age_grp"] = comb_df.loc[:,'Age_grp'].fillna(value = val)


# In[ ]:


comb_df[(comb_df.Age_grp == "nan")]


# In[ ]:


comb_df['Age_grp'].mode()[0]


# In[ ]:




