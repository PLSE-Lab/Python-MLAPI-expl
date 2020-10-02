#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_path = '../input/titanic/train.csv'


# In[ ]:


df = pd.read_csv(data_path)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


177/891*100


# In[ ]:


#misssing value handling:
df["Age"].mean()


# ## filling Age column missing value wrt cabin Feature

# In[ ]:


temp_df = df[df["Age"].isna()]


# In[ ]:


temp_df.Cabin.isna().sum()


# In[ ]:


temp_df.shape


# In[ ]:


temp_df[temp_df["Cabin"].notna()]["Cabin"]


# In[ ]:


177-158


# In[ ]:


indices = []
for ind, i in enumerate(temp_df[temp_df["Cabin"].notna()]["Cabin"].values):
    if i.startswith("C"):
        indices.append(ind)
indices


# In[ ]:


indic = []
for (i,ind) in list(zip(df[df["Cabin"].notna()]["Cabin"],df[df["Cabin"].notna()]["Cabin"].index)):
    if i.startswith("C"):
        indic.append(ind)


# In[ ]:


indic


# In[ ]:


B_fill = int(df[df["Cabin"].notna()].loc[indic]["Age"].mean())


# In[ ]:


B_fill


# In[ ]:


indices


# In[ ]:


rt = df[df["Age"].isna()]
ind = rt[rt["Cabin"].notna()].iloc[indices].index


# In[ ]:


ind


# In[ ]:


df.loc[ind] = df.loc[ind].fillna(B_fill)


# In[ ]:


df.loc[ind]


# * i have done this process single value "C", you can also do rest of it unique values in cabin column.
# * repeat this process until you those 19 values done.

# ## Filling Age column leftover one with Pclass

# In[ ]:


df[df["Age"].isna()]


# In[ ]:


(158/df.shape[0])*100


# In[ ]:


notna_df = df[df["Age"].notna()]


# In[ ]:


p_fill = int(notna_df[notna_df["Pclass"]==3]["Age"].mean())


# In[ ]:


p_fill


# In[ ]:


na_age_df = df[df["Age"].isna()]


# In[ ]:


ind = na_age_df[na_age_df["Pclass"] == 3].index


# In[ ]:


df.loc[ind,"Age"] = p_fill


# In[ ]:


df.loc[ind]


# In[ ]:


notna_df = df[df["Age"].notna()]


# In[ ]:


p_fill = int(notna_df[notna_df["Pclass"] == 1]["Age"].mean())


# In[ ]:


na_age_df = df[df["Age"].isna()]


# In[ ]:


ind = na_age_df[na_age_df["Pclass"] == 1].index


# In[ ]:


df.loc[ind,"Age"] = p_fill


# In[ ]:


df.loc[ind]


# In[ ]:


notna_df = df[df["Age"].notna()]


# In[ ]:


p_fill = int(notna_df[notna_df["Pclass"] == 2]["Age"].mean())


# In[ ]:


na_age_df = df[df["Age"].isna()]


# In[ ]:


ind = na_age_df[na_age_df["Pclass"] == 2].index


# In[ ]:


df.loc[ind,"Age"] = p_fill


# In[ ]:


(687/df.shape[0])*100


# **rest of it are dropped and also dropped cabin column as it has missing value more than 78%**

# In[ ]:


df.drop("Cabin", axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df.dropna(axis=0, inplace=True)


# In[ ]:


df.isna().sum()


# ### EDA v2

# ## Other way to fill missing values:

# In[ ]:


df.isna().sum()


# In[ ]:


fill_val = int(df["Age"].mean())


# In[ ]:


df["Age"].fillna(fill_val, inplace=True)


# In[ ]:


df.drop("Cabin", axis =1, inplace=True)
df.dropna(axis=0, inplace=True)


# In[ ]:


df.isna().sum()


# # Visualization

# In[ ]:


df.columns


# ## Univarient analysis:

# In[ ]:


df["Survived"].value_counts()


# In[ ]:


sns.barplot(df["Survived"].value_counts().index, df["Survived"].value_counts().values)
plt.title(" Survived column ")
plt.show()


# * these plot shows people survived and not survived

# In[ ]:


df["Pclass"].value_counts()


# In[ ]:


plt.barh(df["Pclass"].value_counts().index, df["Pclass"].value_counts().values)
plt.yticks(sorted(df["Pclass"].value_counts().index))
plt.title("Pclass")
plt.show()


# In[ ]:


df["Sex"].value_counts()


# In[ ]:


plt.bar(df["Sex"].value_counts().index, df["Sex"].value_counts().values)
plt.title("Sex")
plt.show()


# In[ ]:


df = df.astype({"Age":"int"})


# In[ ]:


df["Age"].value_counts()


# In[ ]:


plt.figure(figsize=(19,5))
plt.bar(df["Age"].value_counts().index, df["Age"].value_counts().values)


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,15,15])
ax.axis('equal')
ax.pie(df["Age"].value_counts().values, labels= df["Age"].value_counts().index, autopct='%1.1f%%',textprops={'fontsize': 85})
plt.show()


# In[ ]:


df["SibSp"].value_counts()


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1.3])
ax.axis('equal')
plt.pie(df["SibSp"].value_counts().values,labels = df["SibSp"].value_counts().index,
        autopct='%1.1f%%',textprops={'fontsize': 15})
plt.tight_layout()
plt.legend()
plt.show()


# In[ ]:


df['Parch'].value_counts()


# In[ ]:


explode = np.zeros(len(df['Parch'].value_counts()))
explode


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1.3])
ax.axis('equal')

ax.pie(df['Parch'].value_counts().values ,labels = df['Parch'].value_counts().index ,
        explode= explode,
        autopct='%1.1f%%',textprops={'fontsize': 10})

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.legend()
plt.show()


# In[ ]:


df["Fare"].value_counts()


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(df["Fare"])
plt.xticks(np.arange(0,600,15))
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
sns.violinplot(y= df["Fare"])
plt.show()


# In[ ]:


df["Embarked"].value_counts()


# In[ ]:


plt.bar(df["Embarked"].value_counts().index, df["Embarked"].value_counts().values)
plt.show()


# ## Multi-Varient Analysis:

# In[ ]:


df.columns


# In[ ]:


sns.catplot('Embarked',data=df,hue='Pclass',kind='count')


# In[ ]:


sns.catplot('Pclass',data=df,hue='Survived',kind='count')


# In[ ]:


sns.catplot('Sex',data=df,hue='Survived',kind='count')


# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot( data=df, x= "Survived", y= "Age")


# In[ ]:


df.columns


# In[ ]:


sns.catplot(x = "Pclass", col="Embarked", data = df, kind='count')


# In[ ]:


sns.catplot(x = 'Age',y ='Sex' , hue ='Survived',data = df, kind='bar')


# In[ ]:


df = df.astype({"Age":"int"})


# #### Feature Engineering and creating additional feature wrt Age column

# In[ ]:


df["age_grouping"] = df["Age"].apply(lambda x: "0-25" if x <=25 else "25-50" if x > 25 and x <= 50 else "above50")


# In[ ]:


df.head()


# In[ ]:


sns.catplot(x = "age_grouping", col="Sex", data= df, kind='count')


# * out of 890 people, 580 were male and rest are female
# * out of 890 people, 520 were age group belongs 25-50

# In[ ]:


sns.catplot(x = "age_grouping", col="Survived", data= df, kind='count')


# * in 890 people , 330 people were not survived 37% of people in the ship they were of age group 25-50 
# * in 890 people , 180 people were not survived 20% of people in the ship they were of age group 0-25
# * in 890 people , 40 people were not survived 4% of people in the ship they were of age group above50
# * in 890 people , 195 people were survived 21% of people in the ship they were age group 25-50
# * in 890 people , 125 people were survived 14% of people in the ship they were age group 0-25
# * in 890 people , 20 people were survived 2% of people in the ship they were age group above50

# #### using sub plots

# In[ ]:


fig, ax = plt.subplots(2,1)

fig.set_figheight(10)
fig.set_figwidth(9)

ax[0].bar(df["Embarked"].value_counts().index, df["Embarked"].value_counts().values)
ax[1].bar(df["Pclass"].value_counts().index, df["Pclass"].value_counts().values)


# In[ ]:


col = ['Pclass', 'Sex', 'SibSp','Parch', 'Embarked', 'age_grouping']


# In[ ]:



for i in col:
    sns.catplot(x = i, col ='Survived',data = df, kind='count')
    plt.title(i)
    plt.show()


# In[ ]:


sns.catplot(x = "age_grouping", col="Sex", data= df, kind='count')
sns.catplot(x = "age_grouping", col="Survived", data= df, kind='count')


# In[ ]:




