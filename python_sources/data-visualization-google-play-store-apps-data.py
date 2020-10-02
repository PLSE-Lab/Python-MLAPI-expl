#!/usr/bin/env python
# coding: utf-8

# #### Import the Libraries

# In[ ]:


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


gplay_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


gplay_df.info()


# In[ ]:


gplay_df.head()


# In[ ]:


gplay_df.shape


# # Assess the dataset
# > Two types of assessment done are:
# 
# > ###### Visually with both Excel and Pandas
# > During the gathering stage, dataset is opened in pandas with df.head() and df.info() in order to get a feel for the datasets.The dataset with around  rows was manageable in Excel and using the filters function gave a good feel of the data inside each of the three datasets. From Excel it was quickly identified the many incorrect details in the dataset for the Apps and the strange rating scores being used.
# 
# > ##### Programmatically using Pandas
# Some further sampling and investigation was done in pandas using df.info(), df.head(), df.sample(), df['column'].value_counts()

# # Cleaning dataset
# ### Quality Issues:
# > ##### 1. Dropping the Rating NaN values and Taking '19.0' Rating as incorrect entry and changing it to '1.9'
# > ##### 2. Changing the datatype of 'Installs' column to Int and Replacing '+' with ''
# > ##### 3. Changing the Datatype of  'Price' Column from Object to Float
# > ##### 4. Changing the datatype of 'Reviews' column to Int and Replacing 'M' with ''
# > ##### 5. 'Last Updated' Column changed to appropriate datatype 'datetime'
# > ##### 6. Android Ver and Current Ver stripping of 'and up', and replacing 'Varies with Device' with 'NaN'
# > ##### 7. Imputation of the NaN with Median in Numeric Columns and Mode in Categorical Columns
# 
# 
# ### Tidiness Issues:
# > ##### 1. Renaming the columns: Size and Reviews
# > ##### 2. 'Genre' column splitted to 'Primary Genre' and 'Secondary Genre'
# 

# In[ ]:


gplay_df.isna().sum()


# #### Checking for duplicate rows

# In[ ]:


duplicate_ser = gplay_df[gplay_df.duplicated()]
len(duplicate_ser)


# In[ ]:


gplay_df.drop_duplicates(inplace=True)


# In[ ]:


gplay_df.rename(columns={'Reviews':'ReviewCount','Size':'AppSize'},inplace=True)


# ### Reusable Methods

# In[ ]:


def strip_cols(col_name):
    col_name=col_name.str.replace('$','')
    col_name=col_name.str.replace('+','')
    col_name=col_name.str.replace(',','')
    col_name=col_name.str.replace('M','e+6')
    col_name=col_name.str.replace('k','e+3')
    col_name=col_name.str.replace(' and up','')
    #col_name=col_name.str.strip('.GP','')
    #col_name=col_name.str.strip('W','')
    #col_name=col_name.str.strip('-prod','')
    
    return col_name
    
    
def change_dtype(col_name):
    col_name=col_name.astype('float')
    return col_name

def change_intdtype(col_name):
    col_name=col_name.astype('int64')
    return col_name

def replace_nan(col):
    col = col.replace('Varies with device',np.nan)
    return col
    
    


# ### Cleaning App Column

# In[ ]:


gplay_df.App.value_counts()


# In[ ]:


gplay_df.App.nunique()


# ### Cleaning Rating Column

# In[ ]:


gplay_df['Rating'].value_counts()


# In[ ]:


# taking the Rating as 1.9 instead of 19
gplay_df['Rating'].replace('19.0','1.9',inplace=True)


# ### Cleaning 'Price' column

# In[ ]:


gplay_df.Price.value_counts().sort_index()


# #### Dropping the Price value 'Everyone'

# In[ ]:


gplay_df.drop(gplay_df[gplay_df['Price']=='Everyone'].index,inplace=True)


# ####  Price In Dollars:  Remove the 'Dollar' symbol

# In[ ]:



gplay_df['Price'] = strip_cols(gplay_df['Price'])
gplay_df['Price'] = change_dtype(gplay_df['Price'])


# #### Validating the Column after Cleaning

# In[ ]:


gplay_df.Price.value_counts().sort_index()


# ### Cleaning 'AppSize' column

# In[ ]:


gplay_df.AppSize.value_counts()


# #### Replacing the Varies with Device to NaN, Changing the Datatype to float and Replacing the 'M' and 'k'

# In[ ]:


gplay_df['AppSize'] = replace_nan(gplay_df['AppSize'])


# In[ ]:


gplay_df['AppSize'] = strip_cols(gplay_df['AppSize'])
gplay_df['AppSize'] = change_dtype(gplay_df['AppSize'])


# #### Validating the Columns after Cleaning

# In[ ]:


gplay_df['AppSize'].value_counts()


# ### Cleaning Install column

# In[ ]:


gplay_df['Installs'].value_counts()


# #### Stripping the '+' and changing the Datatype to 'INT'

# In[ ]:


gplay_df['Installs'] = strip_cols(gplay_df['Installs'])
gplay_df['Installs'] = change_intdtype(gplay_df['Installs'])


# #### Validate

# In[ ]:


gplay_df['Installs'].value_counts().sort_index()


# ### Cleaning 'ReviewCount' Column

# In[ ]:


gplay_df.ReviewCount.value_counts()


# In[ ]:


gplay_df['ReviewCount'] = strip_cols(gplay_df['ReviewCount'])
gplay_df['ReviewCount'] = change_intdtype(gplay_df['ReviewCount'])


# #### Validate

# In[ ]:


gplay_df.ReviewCount.value_counts().sort_index()


# ### Cleaning 'Genres' column

# In[ ]:


gplay_df['Genres'].value_counts().sort_values()


# #### Splitting the Genres with ';' into Primary and Seconday Genres

# In[ ]:


prim = gplay_df.Genres.apply(lambda x:x.split(';')[0])
gplay_df['Prim_Genre']=prim
gplay_df['Prim_Genre'].tail()


# In[ ]:


sec = gplay_df.Genres.apply(lambda x:x.split(';')[-1])
gplay_df['Sec_Genre']=sec
gplay_df['Sec_Genre'].tail()


# In[ ]:


group_gen=gplay_df.groupby(['Prim_Genre','Sec_Genre'])
group_gen.size().head(20)


# In[ ]:


gplay_df.drop(['Genres'],axis=1,inplace=True)


# ### Cleaning 'Last Updated' Column

# In[ ]:


gplay_df['Last Updated'].value_counts().sort_values()


# #### Changing the Datatype of Column 'Last updated' to Datetime

# In[ ]:


gplay_df['Last Updated'] = pd.to_datetime(gplay_df['Last Updated'])


# #### Validate

# In[ ]:


gplay_df['Last Updated'].value_counts().sort_index()


# In[ ]:


#### data is from year 2010,May to 2018,Aug
from datetime import datetime,date
gplay_df['Last_Updated_Days']=gplay_df['Last Updated'].apply(lambda x: date.today()-datetime.date(x))
gplay_df['Last_Updated_Days'].head()


# In[ ]:


gplay_df['Last_Updated_Days'] = gplay_df['Last_Updated_Days'].dt.days


# ### Cleaning 'Current Ver' Column

# In[ ]:


gplay_df['Current Ver'].value_counts().sort_values()


# #### Dropping the incorrect Value 'MONEY' and Replacing the 'Varies with Device' to 'NaN'

# In[ ]:


gplay_df.drop(gplay_df[gplay_df['Current Ver']=='MONEY'].index,inplace=True)


# In[ ]:


gplay_df['Current Ver'] = replace_nan(gplay_df['Current Ver'])


# In[ ]:


gplay_df['Current Ver'].sample(20)


# ### Cleaning 'Android Ver' column

# In[ ]:


gplay_df['Android Ver'].value_counts().sort_values()


# #### Stripping 'and up' and 'Varies with Device' to 'NAN'

# In[ ]:


gplay_df['Android Ver'] = strip_cols(gplay_df['Android Ver'])
gplay_df['Android Ver'] = replace_nan(gplay_df['Android Ver'])
gplay_df['Android Ver'].replace('4.4W','4.4',inplace=True)


# #### Validating the Column after cleaning

# In[ ]:


gplay_df['Android Ver'].value_counts().sort_values()


# In[ ]:


gplay_df['Category'].value_counts().sort_values()


# In[ ]:


gplay_df['Type'].value_counts() 


# In[ ]:


gplay_df['Content Rating'].value_counts() 


# In[ ]:


gplay_df.info()


# In[ ]:


# categorical and Numerical Values:
num_var = gplay_df.select_dtypes(include=['int64','float64']).columns
cat_var = gplay_df.select_dtypes(include=['object','datetime64','timedelta64']).columns
num_var,cat_var


# In[ ]:


gplay_df.isna().sum()


# In[ ]:


missing_perc = (gplay_df.isna().sum()*100)/len(gplay_df)
missing_df = pd.DataFrame({'columns':gplay_df.columns,'missing_percent':missing_perc})
missing_df


# #### missing values are less than 30% hence imputation can be applied- MODE(Categorical), MEDIAN(Numeric)
# #### Applying Imputation

# In[ ]:


col_cat = ['Type','Current Ver','Android Ver'] #Categorical Var.
for col in col_cat:
    gplay_df[col].fillna(gplay_df[col].mode()[0],inplace=True)
    
col_num=['Rating','AppSize'] #Numerical Var.
for col in col_num:
    gplay_df[col].fillna(gplay_df[col].median(),inplace=True)


# In[ ]:


gplay_df.isna().sum()


# In[ ]:


gplay_df.info()


# ### Storing Clean Dataset to 'Clean_GplayApps.csv'

# In[ ]:



gplay_df.to_csv('Clean_GplayApps.csv',index=False)


# # Exploratory Data Analysis

# In[ ]:


# After Cleaning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
gplay_df = pd.read_csv('Clean_GplayApps.csv')


# In[ ]:


gplay_df.head()


# In[ ]:


num_var = gplay_df.select_dtypes(include=['int64','float64'])
col_num = num_var.columns
num_var


# In[ ]:


cat_var = gplay_df.select_dtypes(include=['object'])
col_cat = cat_var.columns
cat_var


# ### Univariate Analysis
# ##### Numerical Variables
#      1.Rating 
#      2.ReviewCount 
#      3.AppSize 
#      4.Installs
#      5.Price 
#      6.Last_Updated_Days 

# In[ ]:


num_var.hist(figsize=(9,9),bins=50);


#     -Mostly Apps have been updated frequently as the graph shows the drop.
#     -Only few apps have less ratings, rest most of the apps have ratings between 4 and 5.
#     -AppSize graph shows the decreasing trend which tells that less number of apps are big in size.
#    Further Analysis will be done on the variables to get more insights.

# In[ ]:


sns.boxplot(gplay_df['Rating']);


# In[ ]:


sns.boxplot(gplay_df['ReviewCount']);


# In[ ]:


sns.boxplot(gplay_df['Installs']);


# In[ ]:


df_paid = gplay_df[gplay_df['Type']=='Paid']
df_paid


# In[ ]:


df_paid['Price'].skew()
sns.boxplot(df_paid['Price']);


# In[ ]:


sns.boxplot(gplay_df['AppSize']);


# #### Removing the Outliers

# In[ ]:


print('Skewness before Removing Outliers')
lis=list(col_num)
lis.remove('Price');
for col in lis:
    print(gplay_df[col].skew());
lis


# In[ ]:


print('Skewness before removing the outliers')
print(df_paid['Price'].skew())


# In[ ]:


per = df_paid['Price'].quantile([0.10,0.90]).values
df_paid['Price']=np.clip(df_paid['Price'],per[0],per[1])
print('Skewness after removing the outliers')
print(df_paid['Price'].skew())


# In[ ]:


sns.boxplot(df_paid['Price']);


# In[ ]:


#Skewness in betweek-1 and +1 is said normally distributed and any deviation indicates extreme values 
print('Skewness after cleaning outliers')
for col in lis:
    perc = gplay_df[col].quantile([0.10,0.90]).values
    gplay_df[col]=np.clip(gplay_df[col],perc[0],perc[1])
    print(gplay_df[col].skew())


# Graphs after removing the Outliers

# In[ ]:


sns.boxplot(gplay_df['Rating']);


# In[ ]:


sns.boxplot(gplay_df['ReviewCount']);


# In[ ]:


sns.boxplot(gplay_df['Installs']);


# In[ ]:


sns.boxplot(gplay_df['AppSize']);


# #### Categorical Variables
#     1.App
#     2.Category
#     3.Type
#     4.Content Rating
#     5.Last updated
#     6.Prim Genre
#     7.Sec Genre
#     8.Current Ver
#     9.Android Ver

# In[ ]:


sns.countplot(data=gplay_df,x='Type');


# > #### More number of Apps are free of cost.

# In[ ]:


gplay_df.Category.value_counts().plot(kind='bar');


# > ###  3 Categories of Apps with more number of Apps are: Family,Game,Tools

# In[ ]:


gplay_df['Content Rating'].value_counts().plot(kind='bar');


#     -Apps for Everyone are more in number which results in more popularity of Google play store

# In[ ]:


gplay_df.Prim_Genre.value_counts().plot(kind='bar');


# In[ ]:


gplay_df.Sec_Genre.value_counts().plot(kind='bar');


# In[ ]:


gplay_df['Last Updated'].value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(8,8));
gplay_df["Last Updated"].value_counts().sort_index().plot(kind='line');
plt.ylim(bottom = 0);
plt.xlabel('Last Updated Date');
plt.ylabel('Count of Apps');


# In[ ]:


sns.lineplot(x='Rating',y='Installs',data=gplay_df);


# > #### Curve does not represent continuous growth in number of installations as increase in Ratings, we can see sudden downward peak around 4.3 Rating and again decreasing trend for greater than 4.6.

# In[ ]:


sns.lineplot(x='AppSize',y='Installs',data=gplay_df);


# > ### We can say that Appsize is not affecting much the number of installations of the apps.Graph is almost linear with the size of app.

# In[ ]:


sns.lineplot(x='Price',y='Installs',data=df_paid);
plt.xlabel('Price (Dollars)');


# > #### More number of installations of app which costs approx 7$

# In[ ]:


sns.lineplot(y='ReviewCount',x='Installs',data=gplay_df);


# > #### Linear graph,which shows that the number of installations increased with the number of ReviewCounts

# In[ ]:


sns.lineplot(x='Rating',y='Price',data=df_paid);


# In[ ]:


sns.lineplot(x='Rating',y='AppSize',data=gplay_df);


# In[ ]:


sns.lineplot(x='Rating',y='ReviewCount',data=gplay_df);


# In[ ]:


sns.lineplot(x='AppSize',y='ReviewCount',data=gplay_df);


# > #### Almost linear graph with ReviewCount.

# In[ ]:


sns.barplot(x='Type',y='Installs',data=gplay_df);


# In[ ]:


sns.barplot(x='Type',y='ReviewCount',data=gplay_df);


# In[ ]:


plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='Price',data=df_paid);


# In[ ]:


plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='ReviewCount',data=gplay_df);


# In[ ]:


plt.figure(figsize=(8,5));
sns.barplot(x='Content Rating',y='Installs',data=gplay_df);


# In[ ]:


plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='ReviewCount',data=gplay_df);


# In[ ]:


plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='Price',data=df_paid);


# In[ ]:


plt.figure(figsize=(30,5));
sns.barplot(x='Android Ver',y='Installs',data=gplay_df);


# In[ ]:


plt.figure(figsize=(30,5));
sns.lineplot(x='Android Ver',y='Price',data=df_paid);


# > #### Android Version 7.1 is having most number of installs
# > #### HIgher Android Versions have apps which are costlier than the older version apps.

# In[ ]:




