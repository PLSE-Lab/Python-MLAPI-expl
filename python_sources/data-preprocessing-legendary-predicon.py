#!/usr/bin/env python
# coding: utf-8

# # Pokemon Dataset
# ## Data Cleaning and preprocessing , Legendary Prediction

# ### Importing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')


# ### EDA : Exploratory data analysis

# In[ ]:


#Display the first elements of the dataset
df.head()


# Check the shape of the dataset

# In[ ]:


#check the shape of the dataframe
df.shape


# In[ ]:


#We use the columns function to see the all columns names
df.columns


# In[ ]:


#Descriptive statistics using describe
df.describe()


# In[ ]:


#Check the types of the columns
df.dtypes


# In[ ]:


#Number of null values per column
df.isnull().sum()


# In[ ]:


#Number of Legendary and non-legendary Pokemons
print(df.Legendary.value_counts())


# ## Categorical Data

# In[ ]:


#Plotting 'isLegendary' using seaborn function 'countplot' with countplot function
plt.figure(figsize=(8, 4))
sns.countplot(data = df , y = 'Legendary')


# In[ ]:


#Now we try a Pie chart
fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
Leg = df.Legendary.value_counts() 
labels = ['Normal', 'Legendary']
ax.pie(Leg, labels = labels,autopct='%1.2f%%',colors=['skyblue','red'])
plt.show()


# In[ ]:


#Print the number of types using nunique()
print('Number of Types : ',df['Type 1'].nunique())
#Print the  Types using unique()
print('Types : ',df['Type 1'].unique())


# In[ ]:


#Plotting 'Type_1' using seaborn function 'countplot' with countplot function
plt.figure(figsize=(12, 6))
sns.countplot(data = df , x = 'Type 1')


# In[ ]:


#Plotting 'Type_1' using seaborn function 'countplot' with countplot function, and with the type as a hue (splitting criteria)
plt.figure(figsize=(15, 6))
sns.countplot(data = df , x = 'Legendary', hue='Type 1')


# In[ ]:


#Plotting 'Type_1' using seaborn function 'countplot' with countplot function, with the legendary 'hue'
plt.figure(figsize=(15, 6))
sns.countplot(data = df , x = 'Type 1', hue='Legendary')


# In[ ]:


#Plotting 'isLegendary' using seaborn function 'countplot' for only 'with countplot function, with the legendary 'hue'
plt.figure(figsize=(12, 6))
sns.countplot(data = df[df.Legendary ], x = 'Type 1')
print('Number of Legendary : ' ,len(df[df.Legendary]))


# In[ ]:


#Plot 'Type_2' Variable using countplot
plt.figure(figsize=(12, 6))
sns.countplot(data = df , x = 'Type 2')


# In[ ]:


#Plot 'Type_1' with 'isLegendary' as a hue
plt.figure(figsize=(15, 6))
sns.countplot(data = df , x = 'Type 2', hue='Legendary')


# In[ ]:


#Let's see the Water pokemon per generation, and see how many legendary pokemon is introduced per generation
plt.figure(figsize=(10, 6))
sns.countplot(data = df[df['Type 1']=='Water'] , x = 'Generation', hue='Legendary',palette="Blues")


# In[ ]:


#Let's see the Fire pokemon per generation, and see how many legendary pokemon is introduced per generation
plt.figure(figsize=(10, 6))
sns.countplot(data = df[df['Type 1']=='Fire'] , x = 'Generation', hue='Legendary',palette="Reds")


# In[ ]:


#Plot the number of Legendary Pokemons introduced per Generation :
plt.figure(figsize=(15, 6))
sns.countplot(data = df[df.Legendary == True] , x = 'Generation')


# ## Let's move to Numerial Values

# In[ ]:


# Density Plot and Histogram of Defense Feature
plt.figure(figsize=(10, 6))
sns.distplot(df['Defense'], hist=True, kde=True, 
             bins=40, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3,'shade': True,})


# In[ ]:


# Density Plot and Histogram of Attack Feature
plt.figure(figsize=(10, 6))
sns.distplot(df['Attack'], hist=False, kde=True, 
             bins=20, color = 'red', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 3,'shade': True,})


# In[ ]:


# Density Plot and Histogram of HP Feature
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
sns.distplot(df['HP'], hist=False, kde=True, 
              color = 'green',
             kde_kws={'linewidth': 3,'shade': True,})


# In[ ]:


# Density Plot and Histogram of the Speed Feature
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
sns.distplot(df['Speed'], hist=False, kde=True, 
              color = '#FEC803',
             kde_kws={'linewidth': 3,'shade': True,})


# In[ ]:


# Density Plot and Histogram of Total Points Feature
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
sns.distplot(df['Total'], hist=False, kde=True,  color = '#12C803',
             kde_kws={'linewidth': 3,'shade': True,})


# In[ ]:


#Scatter plot of Attack and Defense 
plt.figure(figsize=(10, 7))
lab = ['Attack', 'Defense']
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.scatter(x = df.Attack, y = df.Defense, alpha=0.5)
plt.show()


# In[ ]:


#We will Color the Legendary Pokemons with the Red Color
colors = []
for index,row in df.iterrows():
  if row['Legendary']==True:
    colors.append('red')
  else:
    colors.append('blue')


# In[ ]:


#Now we plot using the Color List we created
plt.figure(figsize=(10, 7))
lab = ['Attack', 'Defense']
plt.xlabel('Attack',fontsize=40)
plt.ylabel('Defense',fontsize=40)
plt.scatter(x = df.Attack, y = df.Defense, alpha=0.5,c=colors)
plt.show()


# In[ ]:


#Seaborn makes our life easier
plt.figure(figsize=(10, 7))
sns.scatterplot(data = df , x="Attack", y="Defense", hue="Legendary")


# In[ ]:


#Now we scatter plot with a linear regression function, using regplot (regression plot), to see the relation between defence and weight
plt.figure(figsize=(10, 7))
plt.xlabel('Defense',fontsize=40)
plt.ylabel('Weight_kg',fontsize=40)
sns.regplot(data = df , x="Defense", y="Attack",color='brown')


# In[ ]:


#Now we scatter plot with a linear regression function, using regplot (regression plot), to see the relation between defence and HP
plt.figure(figsize=(10, 7))
plt.xlabel('Defense',fontsize=40)
plt.ylabel('HP',fontsize=40)
sns.regplot(data = df , x="Defense", y="HP",color = 'green')


# In[ ]:


# To see both Legendary and Non Legendary classes regression plot, we use lmplot function
plt.figure(figsize=(15, 10))
sns.lmplot(data = df , x="Defense", y="HP", hue="Legendary")


# In[ ]:


#Move to the Box plot, lets try 'Attack'
sns.boxplot(x=["Attack"], data=df,orient="v")


# In[ ]:


#We can use the defeined function in Pandas, but it is not so pretty
df.boxplot(figsize=(10,6))


# In[ ]:


df.columns


# In[ ]:


#Lets use slicing , to extract Numerical Values
dfNum = df[['Attack','Defense','Speed']]


# In[ ]:


dfNum.head()


# In[ ]:


#Melt function help us transform our Dataframe to a format
#where one or more columns are identifier variables, while all other columns, considered measured variables
melt = pd.melt(dfNum)


# In[ ]:


melt.head()


# In[ ]:


#Now we can use our 'melt' dataframe to diplay multiple columns
plt.figure(figsize=(15, 10))
sns.boxplot(data = melt,x="variable", y="value")


# In[ ]:


df.Legendary==1


# In[ ]:


#Now we Seperate Legendary and Normal Pokemons, so we can plot them in a seperated way, to gain more insight
#We start with Legendary Pokemons
dfLegend = dfNum[df.Legendary==1]


# In[ ]:


#Now we move to Normal Pokemons
dfNormal= dfNum[df.Legendary==0]


# In[ ]:


#We boxplot Normal pokemns Dataset
plt.figure(figsize=(15, 10))
sns.boxplot(data = pd.melt(dfNormal),x="variable", y="value")


# In[ ]:


#Now we Plot Legendary Pokemons
plt.figure(figsize=(15, 10))
sns.boxplot(data = pd.melt(dfLegend),x="variable", y="value")


# In[ ]:


df[df.Legendary == 1].groupby('Type 2').Legendary.count().sum()
#More than the half


# In[ ]:


for index,row in df.iterrows():
  try :
    np.isnan(row['Type 2']) #is nan generate an error of non convertable to float type is used
    df.at[index,'TwoTypes'] = True
  except:
    df.at[index,'TwoTypes'] = False


# In[ ]:


df.isnull().sum()


# In[ ]:


#Time to drop unecessary Features
df.drop(['Name','Type 2','Generation'],axis=1,inplace = True)


# In[ ]:


#Quick Check
df.head()


# #### Now we will encode our Categorical Data, using LabelEncoder from sklearn

# In[ ]:


from sklearn import preprocessing


# In[ ]:


le = preprocessing.LabelEncoder()


# In[ ]:


#First We fit our encoder
le.fit(df['Type 1'])


# In[ ]:


#And then we transform our Data
df['Type 1'] = le.transform(df['Type 1'])
df['Type 1']


# In[ ]:


#Another Quick Check
df.head()


# In[ ]:


#Describe the dataset
df.describe()


# ### We seperate the Features from Target, in Our case Target  'Legendary'

# In[ ]:


X = df.drop(['Legendary'],axis = 1)
y =df.Legendary


# In[ ]:


#Feature Scaling : 

from sklearn.preprocessing import StandardScaler

Sc = StandardScaler()

# We need to fit and transform the dataset
dfScale = Sc.fit_transform(df)


# ### Train_Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#We use a 70%/30% Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)


# ### To choose the ML algorithm, we can look at this map :
# ![Image](https://miro.medium.com/max/1400/1*tpOZa-wju9pD-5vFd-Es-g.png)
# 

# In[ ]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# In[ ]:


names = ["Support Vector Classifier", "K-Neighbors Classifier", "Random Forest Classifier"]


# In[ ]:


classifiers = [SVC(),
               KNeighborsClassifier(),
               RandomForestClassifier() ]


# In[ ]:


for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc = accuracy_score(preds,y_test)
    precision = precision_score(y_test,preds)
    recall = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)
    cm = confusion_matrix(y_test,preds)
    print (name, 'Accuracy  :  ', "%.2f" %(acc*100),'%', ', Precision',"%.3f" %precision, ' , Recall : ', "%.3f" %recall, ' , F1-Score : ',"%.3f" %f1)
    print('The confusion Matrix : ' )
    print(cm)
    print(' *-----------------------------------------------------------------------------------------------------*')


# In[ ]:




