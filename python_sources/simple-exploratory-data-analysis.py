#!/usr/bin/env python
# coding: utf-8

# Import Required Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing


# Read the Dataset

# In[ ]:


googleappdata = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# Step 1: Describe the Dataset

# In[ ]:


## Check the dimension of the dataset
print("Dimesion of the dataset:", googleappdata.shape)
print("No of rows:", googleappdata.shape[0])
print("No of cols:", googleappdata.shape[1])


# In[ ]:


## Check the features
print("Features of the dataset:\n",googleappdata.columns)


# In[ ]:


## Check the datatype and information of the data
googleappdata.info()


# In[ ]:


## Preview the dataset
googleappdata.head()


# In[ ]:


#appdata.iloc[5]


# In[ ]:


## Check if there are any missing values
googleappdata.isna().sum()


# Step 2: Clean the Dataset

# In[ ]:


## lets drop the rows containing missing values in content rating, current version, android version, type cols
googleappdata.dropna(axis=0,inplace=True,subset=['Android Ver','Current Ver','Type','Content Rating'])


# In[ ]:


## Check if there are any other missing values
googleappdata.isna().sum()


# In[ ]:


## Check rows having missing values in Rating
googleappdata[googleappdata['Rating'].isna()].head()


# In[ ]:


## Statistics of Rating cols
googleappdata.Rating.describe()
## "We can impute with mean value 4.19"


# In[ ]:


## Missing value imputation of Rating coloumn
mean_r = round(googleappdata.Rating.mean(),2)
googleappdata['Rating']=googleappdata['Rating'].fillna(mean_r)
print("Checking if there are any missing values:\n",googleappdata.Rating.isna().sum())


# In[ ]:


## Lets make a copy for all cleaning and transformation
df_appdata = googleappdata


# In[ ]:


## Univariate Analysis
## Lets look at CATEGORY column
print("No of unique categories:",df_appdata.Category.nunique())
print("List of unique categories:\n",df_appdata.Category.unique())


# In[ ]:


## Frequency distribution of Category
fq=df_appdata.Category.value_counts()
ptg=round(((df_appdata.Category.value_counts())/df_appdata.shape[0])*100,2)
fqtble = pd.DataFrame({'Frequency':fq,'Percentage':ptg})
fqtble


# In[ ]:


## Plotting the Category frequency
plt.figure(figsize=(10,8))
ax=sns.barplot(x=fqtble.index,y=fqtble.Frequency,data=fqtble)
plt.title("Frequency/Count plot for Category column")
plt.xticks(rotation=80)
plt.show()


# In[ ]:


## Alternate way of Plotting the Category frequency
plt.figure(figsize=(10,8))
ax=sns.countplot(y='Category',data=df_appdata)
plt.title("Frequency/Count plot for Category column")
plt.xticks(rotation=80)
plt.show()


# In[ ]:


## One hot encoding for category variable
df_appdata1=df_appdata
oh = pd.get_dummies(df_appdata1['Category'],prefix='CTG')
oh=oh.drop('CTG_BEAUTY',axis = 1)
df_appdata1=pd.concat([df_appdata1,oh],axis=1)
df_appdata1.head()


# In[ ]:


## lets look at Rating coloumn
df_appdata1.Rating.describe()
## we can see max and min rating, and 75% of data have values <=4.5


# In[ ]:


## Lets plot distribution plot of Rating col
plt.figure(figsize=(8,6))
sns.distplot(df_appdata1.Rating)
plt.title("Distribution plot of Rating")
plt.show()


# In[ ]:


## lets look at Review column
print("Datatype of Reviews:",df_appdata1.Reviews.dtype)
## Lets convert object to float/int 
df_appdata1['Reviews']=df_appdata1['Reviews'].astype('int64')
print("Datatype pf Reviews after conversion:",df_appdata1.Reviews.dtype)
print("Description of Review coloum:\n",df_appdata1.Reviews.describe())


# In[ ]:


### lets look at Size column now
### We need to remove 'M' and 'k'and convert them to numeric values.
def SizePreprocess(size):
    lists = []
    for item in size:
        if item[-1]=='M':
            item=float(item[:-1])
            lists.append(item)
        elif item[-1]=='k':
            item=item[:-1]
            item=float(item)/1000
            lists.append(item)
        else:
            lists.append(21.5)  ## imputing the mean of ratings
    return lists


# In[ ]:


df_appdata2=df_appdata1


# In[ ]:


float(df_appdata2.Size[1][:-1])*1000


# In[ ]:


### Preprocessing the size column
d=pd.DataFrame(df_appdata2['Size'])
df_appdata2['Newsize']=d.apply(SizePreprocess)
df_appdata2.head()


# In[ ]:


x = df_appdata2[df_appdata2['Newsize']!=0]['Newsize']
x.mean()


# In[ ]:


## Size stats
df_appdata2.Newsize.describe()
## minimum size=0 and max=100, average size of any app is 18Mb


# In[ ]:


## distribution plot of Size
plt.figure(figsize=(8,6))
sns.distplot(df_appdata2.Newsize)
plt.title("Distribution plot of Size")
plt.show()


# In[ ]:


df_appdata2['Size'].value_counts() ##Since Varies with device is 1694, we might have to impute mean size


# In[ ]:


## distribution plot of Size
plt.figure(figsize=(8,6))
sns.distplot(df_appdata2.Newsize)
plt.title("Distribution plot of Size")
plt.show()


# In[ ]:


## Lets deal with Installs col now
df_appdata3=df_appdata2
print("Number of unique values in Installs:",df_appdata3['Installs'].nunique())
print("Values in Installs:",df_appdata3['Installs'].unique())


# In[ ]:


## Bar graph to look at Installs
plt.figure(figsize=(8,6))
sns.barplot(x=df_appdata3['Installs'].unique(),y=df_appdata3['Installs'].value_counts())
plt.title("Installs Plot")
plt.xticks(rotation =60)
plt.show()


# In[ ]:


df_appdata3.columns


# In[ ]:


### Lets look at Type coloumn
print("Types of apps are:",df_appdata3['Type'].unique())
df= pd.DataFrame({'Type':df_appdata3['Type'].unique(), 'Count':df_appdata3['Type'].value_counts()})
print("Freq table for type\n",df)
plt.figure(figsize=(6,4))
sns.barplot(x=df['Type'],y=df['Count'])
plt.title("How many are Free/Paid??")
plt.show()
## Most of the apps are free


# In[ ]:


### Label encoding Type Coloumn
df_appdata4 = df_appdata3
labelencoder = preprocessing.LabelEncoder()
df_appdata4['TypeEN']= labelencoder.fit_transform(df_appdata4['Type'])
print("Label encoded types are\n:",df_appdata4['TypeEN'].unique())


# In[ ]:


## Lets look at the Price coloumn
## As most of the apps are free, the Price of the app will be 0(10032 values are 0 which matches with free type count)
df=pd.DataFrame({'Price':df_appdata3['Price'].unique(),'Count': df_appdata3['Price'].value_counts()})
df.head()


# In[ ]:


## Clean the Price column. Remove the dollar sign from the price and convert the type tp float
df_appdata5=df_appdata4
df_appdata5['NewPrice']=df_appdata4['Price'].str.replace('$','') 
print("New values of Price are\n:",df_appdata5['NewPrice'].unique())
print("Datatype of Price before conversion:",df_appdata5['NewPrice'].dtype)
df_appdata5['NewPrice'] = df_appdata5['NewPrice'].astype('float')
print("Datatype of Price after conversion:",df_appdata5['NewPrice'].dtype)


# In[ ]:


##Lets look at content rating
df_appdata6 = df_appdata5
print("Categories of Content Rating are:", df_appdata6['Content Rating'].unique())
CRdummies = pd.get_dummies(df_appdata6['Content Rating'],prefix='CR')
CRdummies= CRdummies.drop('CR_Unrated',axis=1)
df_appdata6= pd.concat([df_appdata6,CRdummies],axis=1)
df_appdata6.head()


# In[ ]:


df_appdata6.columns


# In[ ]:


### Looking at Genres coloumn now
print("How many categories are in the Genres col:",df_appdata6.Genres.nunique())
## lets look at some popular Genres
plt.figure(figsize=(25,15))
sns.barplot(x=df_appdata6['Genres'].unique(),y=df_appdata3['Genres'].value_counts())
plt.title("Genres Plot")
plt.xticks(rotation =90)
plt.show()


# In[ ]:


## Which are the Top 20 Genres 
gendf=pd.DataFrame({'Genres':df_appdata6['Genres'].unique(),'Count': df_appdata6['Genres'].value_counts()})
gendf.head()
## Alternate way of Plotting the Category frequency
plt.figure(figsize=(8,6))
ax=sns.barplot(x='Genres',y='Count',data=gendf[0:19])
plt.title("Which are Top 20 Genres ?? ")
plt.xticks(rotation=80)
plt.show()


# In[ ]:


## Last Updated data
df_appdata7 = df_appdata6
print("Type of this coloumn is :",df_appdata7['Last Updated'].dtype)
## lets convert it into date format
df_appdata7['New Last Updated'] = pd.to_datetime(df_appdata7['Last Updated'])
print("Type after conversion is:", df_appdata7['New Last Updated'].dtype)
df_appdata7['New Last Updated'].head()


# In[ ]:


## latest updated app
df_appdata7[df_appdata7['New Last Updated'] == df_appdata7['New Last Updated'].max()]


# In[ ]:


print(df_appdata7['Current Ver'].nunique())
print(df_appdata7['Current Ver'].max())


# In[ ]:


print(df_appdata7['Android Ver'].nunique())
print(df_appdata7['Android Ver'].max())


# In[ ]:


### lets look at Installs column now
### We need to remove '+' from the string
def InstallsPrep(Installs):
    lists = []
    for item in Installs:
        if item[-1]=='+':
            item=float(item[:-1].replace(',',''))
            lists.append(item)
        else:
            lists.append(item)  ## imputing the mean of ratings
    return lists


# In[ ]:


### Preprocessing the Installs column
i=pd.DataFrame(df_appdata7['Installs'])
df_appdata7['NewInstalls']=i.apply(InstallsPrep)
df_appdata7.head()


# In[ ]:


### lets check the correlation 
g = sns.pairplot(df_appdata7,vars=["Rating","Newsize","NewPrice","Reviews","NewInstalls"],hue = "Type")
plt.title("Pair Plot of Google App Data")
### Price, Size and Installs are highly right skewwed


# In[ ]:


## lets check the correlation of all the variables in the final dataset
fig,ax = plt.subplots(figsize=(6,6))
ax = sns.heatmap(df_appdata7[["Rating","Newsize","NewPrice","Reviews","NewInstalls"]].corr(),annot=True,linewidths=.5,fmt='.2f')
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
plt.show()
### Reviews and Installs are correlated
### Size and Reviews are correlated

