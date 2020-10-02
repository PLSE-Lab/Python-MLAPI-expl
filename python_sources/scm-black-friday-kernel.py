#!/usr/bin/env python
# coding: utf-8

# ## My First Kernel: Black Friday 

# #### Kernel Still in Progress...

# Description
# The dataset here is a sample of the transactions made in a retail store. The
# store wants to know better the customer purchase behavior against different
# products. Specifically, here the problem is a regression problem where we are
# trying to predict the dependent variable (the amount of purchase) with the
# help of the information contained in the other variables.
# 
# Classification problem can also be settled in this dataset since several
# variables are categorical, and some other approaches could be "Predicting the
# age of the consumer" or even "Predict the category of goods bought". This
# dataset is also particularly convenient for clustering and maybe find different
# clusters of consumers within it.

# In[ ]:


# LOAD PACKAGES
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# LOAD DATA
df = pd.read_csv('../input/BlackFriday.csv')


# This is an overview of the data

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# Layout of the type of data in the DataFrame

# In[ ]:


df.dtypes.value_counts()


# The 'NaN' values were originally blanks, therefore wherever there's a 'NaN', it will be converted to zero

# In[ ]:


df.fillna(value=0,inplace=True)


# In[ ]:


# LOOKING AT THE DATA AGAIN
df.head(10)


# 'User_ID' is regarded as an int64, though it is similar to 'Product_ID' in terms of meaning, 'User_ID' shall be converted
# to an object type data. Similarly Product_Category_2 and Product_Category_3 can be converted to integers

# In[ ]:


df['User_ID'] = df['User_ID'].apply(lambda x: str(x))
df['Product_Category_2'] = df['Product_Category_2'].astype(int)
df['Product_Category_3'] = df['Product_Category_3'].astype(int)


# In[ ]:


df.info()


# In[ ]:


# CHECK THE NUMBER OF UNIQUE VARIABLES IN EACH COLUMN 
# (NOTE: ZERO WILL ALSO BE REGARDED AS A CATEGORY UNDER THE PRODUCT_CATEGORIES)

for col in df.drop('Purchase',axis=1):
    x = df[col].nunique()
    print(f'{col} column has {x} number of unique categories')


# ### Data exploration

# Further data exploration will look at distibution of each of the features and if any insights can be gathered from them,
# first looking at the features on their own

# In[ ]:


# Remove duplicate User_ID's and the Purchase column to determine the general demographics of the data
grouped_df = df.drop_duplicates('User_ID')


# In[ ]:


grouped_df.reset_index(drop=True, inplace=True)


# In[ ]:


grouped_df.drop('Purchase', axis=1, inplace=True)


# In[ ]:


grouped_df.head()


# In[ ]:


len(grouped_df) == df['User_ID'].nunique()


# This makes sense since we removed the duplicates and all that is left are the unique User_ID

# In[ ]:


purchases_df = df.groupby('User_ID')['Purchase'].agg('sum').pipe(pd.DataFrame).reset_index()


# In[ ]:


purchases_df.head()


# In[ ]:


# To verify the numbers
purchases_df['Purchase'].sum() == df['Purchase'].sum()


# In[ ]:


# Since it's True, we can merge the two DataFrames
new_df = pd.concat([grouped_df,purchases_df],axis=1)


# In[ ]:


# Check if there are any missing (NaN) values
new_df = pd.DataFrame(new_df)
new_df.info()


# ### Gender

# In[ ]:


new_df.head()


# In[ ]:


plt.figure(figsize=(16,4))
plt.subplot(121)
new_df['Gender'].value_counts().plot.bar(rot=0)
plt.title('Gender Head Count')
plt.ylabel('count')

# frequency of each Gender's purchase at the store
plt.subplot(122)
df['Gender'].value_counts().plot.bar(rot=0)
plt.title('Number of products')
plt.ylabel('count')

plt.tight_layout()


# There were more males who made a purchase on the day than there were females which also explains the large number of products purchased by males than females. However, females might have been accompanying their male companion on the day which could also be a factor in this significant difference.

# In[ ]:


male_frequency_of_purchase = df[df['Gender']=='M'].count()[0] / new_df[new_df['Gender']=='M'].count()[0]
female_frequency_of_purchase = df[df['Gender']=='F'].count()[0] / new_df[new_df['Gender']=='F'].count()[0]


# In[ ]:


print('On average, Males purchased {:.0f} products on Black Friday\nwhereas on average Females purchased {:.0f} products'.format(male_frequency_of_purchase, female_frequency_of_purchase))


# In[ ]:


# Total purchase of each Genre
plt.style.use('ggplot')
new_df.groupby('Gender')['Purchase'].sum().plot.bar(rot=0)
plt.title('Total Expenditure');


# In[ ]:


male_spending = df[df['Gender']=='M']['Purchase'].sum()
female_spending = df[df['Gender']=='F']['Purchase'].sum()
avg_male_spending = df[df['Gender']=='M']['Purchase'].mean()
avg_female_spending = df[df['Gender']=='F']['Purchase'].mean()


# In[ ]:


print("In total Males spent {:.2f} times more money than Females on Black Friday,\nwith an average of {:.2f} for Males and {:.2f} for Females".
      format(male_spending/female_spending,avg_male_spending,avg_female_spending))


# In[ ]:


# Average spending per product
per_product_avg_female = avg_female_spending/female_frequency_of_purchase
per_product_avg_male = avg_male_spending/male_frequency_of_purchase


# In[ ]:


print('On average, Males spent {:.2f} per product and Females spent {:.2f} per product'.
     format(per_product_avg_male,per_product_avg_female))


# Summary:
# 
# To sum it all up regarding Gender, in terms of purchases, more males made more purchase on the day than females which transferred to higher total purchases, 3.31 times more for males than females. Furthermore on average males purchased more products than females which got transferred to a higher purchase average for males than females. However it can be assumed that at the store on the day, majority of products purchased by males were male products, similarly for females. Then it can be argued that a high cost per product endured by females than males is also a contributing factor in lower female statistics.    

# ### Age

# In[ ]:


age_list = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']


# In[ ]:


plt.figure(figsize=(16,10))

# Distribution by age group
plt.subplot(221)
new_df[new_df['Gender']=='M']['Age'].value_counts().plot.bar()
plt.title('Number of Male User_ID\'s by Age group')
plt.ylabel('count')
plt.xlabel('Age')

plt.subplot(222)
new_df[new_df['Gender']=='F']['Age'].value_counts().plot.bar()
plt.title('Number of Female User_ID\'s by Age groups')
plt.ylabel('count')
plt.xlabel('Age')


# frequency of each Age's purchase at the store
plt.subplot(223)
sns.countplot('Age',data=df, order=age_list)
plt.title('Number of products purchased by different Age groups')
plt.ylabel('count')

# FILTERED BY GENDER

# frequency of each Age's purchase at the store filtered by gender
plt.subplot(224)
sns.countplot('Age',hue='Gender',data=df, order=age_list)
plt.title('Number of products purchased by grouped by Age and Gender')

plt.tight_layout()


# The '26-35' age group seems to have purchased a lot of products on Black Friday with males in each age group purchasing more products than females

# In[ ]:


# Looking at Purchases
plt.figure(figsize=(16,10))

# Grouped by Age
plt.subplot(221)
df.groupby('Age')['Purchase'].sum().plot.bar()
plt.title('Purchases by Age')
plt.ylabel('Purchase')
plt.xlabel('Age')

# Grouped by Age and Gender
plt.subplot(223)
df[df['Gender']=='M'].groupby('Age')['Purchase'].sum().plot.bar()
plt.title('Purchases by Age, Males')
plt.ylabel('Purchase')
plt.xlabel('Age')

plt.subplot(224)
df[df['Gender']=='F'].groupby('Age')['Purchase'].sum().plot.bar()
plt.title('Purchases by Age, Females')
plt.ylabel('Purchase')
plt.xlabel('Age')

plt.tight_layout()


# The purchase trends are all similar to the number of users per each age group, it seems overall, a strong support is for the age group 18-45 year olds. 

# ### City Category

# In[ ]:


new_df['City_Category'].unique()


# In[ ]:


# Marital Status, 0: Single, 1:Married
df['Marital_Status'] = ['single' if x == 0 else 'married' for x in df['Marital_Status']]


# In[ ]:


plt.figure(figsize=(16,15))

plt.subplot(321)
sns.countplot('City_Category',data=df)
plt.title('Number of products purchased by City Category')

plt.subplot(322)
sns.barplot('City_Category','Purchase',data=df, estimator=np.mean)
plt.title('Mean Purchases by City Category')

plt.subplot(323)
sns.countplot('City_Category', hue='Gender',data=df)
plt.title('Number of products purchased by Gender and City Category')


plt.subplot(324)
sns.barplot('City_Category','Purchase',hue='Gender',data=df, estimator=np.mean)
plt.title('Mean Purchases and Gender by City Category')

plt.subplot(325)
sns.countplot('City_Category', hue='Marital_Status',data=df)
plt.title('Number of products purchased by Gender and City Category')

plt.subplot(326)
sns.barplot('City_Category','Purchase',hue='Marital_Status',data=df, estimator=np.mean)
plt.title('Mean Purchases and Marital_Status by City Category')

plt.tight_layout()


# More purchase done in City_Category B followed by City_Category C then by City_Category  even though on average City_Category_B Purchase is the highest. Furthermore based on Marital_Status, there are a lot more single people in each City_Category than there are married people, with Marital_Status not impacting the average purchases. A further look into the Age distribution of each city and occupational distribution for each City_Category follows:

# In[ ]:


df['Occupation'].unique()


# In[ ]:


plt.figure(figsize=(14,18))

plt.subplot(411)
sns.barplot(x='Occupation',y='Purchase',hue='City_Category',data=df,estimator=np.sum)
plt.title('Purchases by Occupation in each City Category')
plt.legend(bbox_to_anchor=(1.1,1))

plt.subplot(412)
sns.barplot(x='Occupation',y='Purchase',hue='City_Category',data=df,estimator=np.mean)
plt.title('Mean Purchases by Occupation in each City Category')
plt.legend(bbox_to_anchor=(1.1,1))

plt.subplot(413)
sns.barplot(x='Age',y='Purchase',hue='City_Category',data=df,estimator=np.sum,order=age_list)
plt.title('Purchases by Age in each City Category')
plt.legend(bbox_to_anchor=(1.1,1))

plt.subplot(414)
sns.barplot(x='Age',y='Purchase',hue='City_Category',data=df,estimator=np.mean,order=age_list)
plt.title('Mean Purchases by Age in each City Category')
plt.legend(bbox_to_anchor=(1.1,1))

plt.tight_layout()


# Overall looking at the mean Purchase, occupation and age do not have a significant bearing on Purchase 

# ### Stay in Current City

# In[ ]:


df['Stay_In_Current_City_Years'].unique()


# In[ ]:


stay_list = ['0','1','2','3','4+']


# In[ ]:


# Marital Status, 0: Single, 1:Married
new_df['Marital_Status'] = ['single' if x == 0 else 'married' for x in new_df['Marital_Status']]


# In[ ]:


plt.figure(figsize=(16,5))

plt.subplot(121)
new_df[new_df['Gender']=='M']['Stay_In_Current_City_Years'].value_counts().plot.bar()
plt.title('No. of years in a city: Males')

plt.subplot(122)
new_df[new_df['Gender']=='F']['Stay_In_Current_City_Years'].value_counts().plot.bar()
plt.title('No. of years in a city: Females')

plt.tight_layout()


# Majority of the people have stayed in the city for only a year, with males being the majority in each year category

# To check the influence of years in the city on Purchase, and whether it is affected by Gender and/or Marital Status 

# In[ ]:


plt.figure(figsize=(16,10))

plt.subplot(221)
sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',data=df,order=stay_list,estimator=np.sum)

plt.subplot(222)
sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',data=df,order=stay_list,estimator=np.mean)

plt.subplot(223)
sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',hue='Gender',data=df,order=stay_list,estimator=np.mean)

plt.subplot(224)
sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',hue='Marital_Status',data=df,order=stay_list,estimator=np.mean)

plt.tight_layout()


# As already established, number of years in a city of '1' are the highest spenders, though as the years go by which is also correlated to as people get older, the numbers decline. This might be due to the store attracts younger customers on Black Friday than older customers. 

# In[ ]:


# For further insight between Age Distribution and number of years in the city
pivot_df = pd.pivot_table(data=new_df,index='Age',columns='Stay_In_Current_City_Years',values='Purchase',aggfunc=np.count_nonzero)


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(pivot_df,cmap='coolwarm',annot=True);


# Majority of people in each age category have stayed in the city for 1 year, with the numbers declining as the number of years in the city increase as well as age increase post '26-35' years category. The trend suggests as people get older they leave the city.

# ### Analyzing the Product Categories

# In[ ]:


print('cat_1')
print(df['Product_Category_1'].unique(),df['Product_Category_1'].nunique())
print('\n')
print('cat_2')
print(df['Product_Category_2'].unique(),df['Product_Category_2'].nunique())
print('\n')
print('cat_3')
print(df['Product_Category_3'].unique(),df['Product_Category_3'].nunique())


# In[ ]:


def category_plots(cat):
    
    plt.subplot(611)
    sns.countplot(x=cat,data=df[df[cat]!=0]) # Zero had replaced 'NaN' meaning no purchase was made of the Product_Category 
    plt.title(f'Number of {cat} purchased')
    
    plt.subplot(612)
    sns.barplot(x=cat,y='Purchase',hue='Gender',data=df[df[cat]!=0],estimator=np.sum)
    plt.title(f'Total {cat} purchased according to Gender')
    
    plt.subplot(613)
    sns.barplot(x=cat,y='Purchase',hue='Marital_Status',data=df[df[cat]!=0],estimator=np.sum)
    plt.title(f'Total {cat} purchased according to Marital_Status')
    
    plt.subplot(614)
    sns.barplot(x=cat,y='Purchase',hue='Age',data=df[df[cat]!=0],estimator=np.sum)
    plt.title(f'Total {cat} purchased filtered by Age group')
    
    plt.subplot(615)
    sns.barplot(x=cat,y='Purchase',hue='City_Category',data=df[df[cat]!=0],estimator=np.sum)
    plt.title(f'Total {cat} purchased filtered by City_Category')
    
    plt.subplot(616)
    sns.barplot(x=cat,y='Purchase',hue='Stay_In_Current_City_Years',data=df[df[cat]!=0],estimator=np.sum)
    plt.title(f'Total {cat} purchased filtered by Years staying in the city')
    
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(16,30))
category_plots('Product_Category_1')


# Category 5, followed by 1, then 8 are the standout Product_Category_1 products purchased on Black Friday with males being the dominant gender in all categories. Though the according to purchase amount Category 1 seems to have the most amount spent on it, mostly by single people. Ages between 26-35 are the dominant customers according to amount spent on Product_Category_1, and for those people staying in City_Category_B. Those people who have stayed in the city for 1 year seem to be those who purchased the most from almost each product category

# In[ ]:


plt.figure(figsize=(16,30))
category_plots('Product_Category_2')


# Similar results are obtained in Product_Category_2, though the most purchased products are category 8, followed by 2, then 14, 16 and then 15.

# In[ ]:


plt.figure(figsize=(16,30))
category_plots('Product_Category_3')


# Nothing new regarding filtered data can be intepreted from Product_Category_3

# In[ ]:




