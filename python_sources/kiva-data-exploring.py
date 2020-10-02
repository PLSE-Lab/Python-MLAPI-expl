#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


kiva_loans = pd.read_csv("../input/kiva_loans.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
kiva_region_location = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")


# In[ ]:


secator = kiva_loans['sector'].value_counts()


# In[ ]:


kiva_loans['date'] = pd.to_datetime(kiva_loans['date'])
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'])
kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'])
kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'])


# In[ ]:


borrower_genders = kiva_loans['borrower_genders']
borrower_genders = borrower_genders.str.split(', ')
#.apply(lambda x:x.count("female"))
borrower_genders.fillna("", inplace=True)
# print(borrower_genders.value_counts())
kiva_loans['female_borrowers'] = borrower_genders.apply(lambda x:x.count('female'))
kiva_loans['male_borrowers'] = borrower_genders.apply(lambda x:x.count('male'))
kiva_loans['borrowers'] = kiva_loans['female_borrowers'] + kiva_loans['male_borrowers']


# In[ ]:


loan_amount_by_country = kiva_loans.groupby('country').sum()['loan_amount'].sort_values(ascending=False)[:10]
#plt.figure()
#sns.barplot(x=loan_amount_by_country.index, y=loan_amount_by_country.values, color="salmon")


# ## **Distribution of loan_amount**

# In[ ]:


plt.figure()
sns.distplot(kiva_loans['loan_amount'])


# In[ ]:


country = kiva_loans['country']


# ## Countries by number of loans

# In[ ]:


country_counts = country.value_counts()
other = country_counts[10:].sum()
country_counts = country_counts[:10]
country_counts.set_value("others", other)
plt.figure()
sns.set()
sns.barplot(x=country_counts.index, y=country_counts.values, palette="Blues_d")


# In[ ]:


explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slic
plt.figure()
country_counts.plot(kind="pie",explode=explode, autopct='%1.1f%%')


# In[ ]:


kiva_loans.repayment_interval.value_counts()


# ## Countries by  number of irregular payments

# In[ ]:


irregular = kiva_loans[kiva_loans.repayment_interval == "irregular"].country.value_counts()
plot_irregular = irregular.sort_values(ascending=False).head(10)
plt.figure()
sns.barplot(x=plot_irregular.index, y=plot_irregular.values)


# In[ ]:


counting_irregular = kiva_loans.country.value_counts()[
    irregular.index
]
counting_irregular.sort_values(ascending=False).head(10)


# ## Countries having more percentage of irregular payments

# In[ ]:


irregular_payments_poor = (irregular/counting_irregular).sort_values(ascending=False).head(10) # poor payment
plt.figure()
print(irregular_payments_poor)
sns.barplot(x=irregular_payments_poor.index, y=irregular_payments_poor.values)


# ## Countries having less percentage of irregular payments

# In[ ]:


good_payment_irregular = (irregular/counting_irregular).sort_values().head(10) # good payment
plt.figure()
print(good_payment_irregular)
sns.barplot(x=good_payment_irregular.index, y=good_payment_irregular.values)


# ## Number of borrowers vs loan_amount

# In[ ]:


plt.figure()
sns.regplot(x="borrowers", y="loan_amount", data=kiva_loans, x_estimator=np.mean)


# ## Number of borrowers VS repayment intervals

# In[ ]:


plt.figure()
sns.pointplot(x="repayment_interval", y="borrowers", data=kiva_loans)


# ## Number of female borrowers VS repayment intervals

# In[ ]:


plt.figure()
sns.pointplot(x="repayment_interval", y="male_borrowers", data=kiva_loans)


# ## Number of male borrowers VS repayment intervals

# In[ ]:


plt.figure()
sns.pointplot(x="repayment_interval", y="female_borrowers", data=kiva_loans)


# In[ ]:


plt.figure()
sns.pointplot(x="repayment_interval", y="male_borrowers", data=kiva_loans[
    (kiva_loans['female_borrowers'] > 0) & (kiva_loans['male_borrowers'] > 0) ]
             )


# In[ ]:




