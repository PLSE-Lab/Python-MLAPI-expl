#!/usr/bin/env python
# coding: utf-8

#   **POKEMON DATA EXPLORATION n VISUALIZATION. [please upvote it if u find it helpful.]**

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


df=pd.read_csv(r"../input/Pokemon.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


# type 2 has some null values. we need to fill them with type 1
df['Type 2'].fillna(df['Type 1'],inplace=True)


# In[ ]:


df.info() # null values  filled with corressponding type 1 values.


# In[ ]:


#df.head()
# can drop # as indexing is already done
del df['#']
df.head()


# In[ ]:


df.columns.unique()


# ######  now we can see the types and distribution of various categorical features.

# In[ ]:


# consider type 1
df['Type 1'].value_counts()


# In[ ]:


# a count plot to better visualize.
sns.factorplot(x='Type 1',kind='count',data=df,size=5,aspect=3)


# In[ ]:


# a pie to visulaize the relative proportions.
labels = ['Water', 'Normal', 'Grass', 'Bug', 'ychic', 'Fire', 'Electric', 'Rock', 'Other']
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1', '#FFED0D', '#16F5A7']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0.0, 0.1, 0, 0.1) 
plt.pie(x=sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0,counterclock=True)
plt.axis('scaled')
plt.title("Percentage of Different types of Type 1 Pokemons")
fig=plt.gcf()
fig.set_size_inches(9,9)
plt.show()


# In[ ]:


# consider type 2
df['Type 2'].value_counts()


# In[ ]:


# agin a countplot.
sns.factorplot(x='Type 2',kind='count',data=df,size=5,aspect=3)


# In[ ]:


# similarly a pie chart for type 2
labels = ['Poison', 'Fire', 'Flying', 'Dragon', 'Water', 'Bug', 'Normal',
       'Electric', 'Ground', 'Fairy', 'Grass', 'Fighting', 'Psychic',
       'Steel', 'Ice', 'Rock', 'Dark', 'Ghost']
sizes = [49,40,99,29,73,20,65,33,48,38,58,46,71,27,27,23,30,24]
colors = ['B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1', '#FFED0D', '#16F5A7','B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1','#ff4125', '#aa0b00']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0.0, 0.1, 0, 0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0)
plt.pie(x=sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0,counterclock=True)
plt.axis('scaled')
plt.title("Percentage of Different types of Type 2 Pokemons")
fig=plt.gcf()
fig.set_size_inches(9,9)
plt.show()


# In[ ]:


df['Legendary'].value_counts() # implies most of the pokemons were not legendary


# In[ ]:


sns.factorplot(x='Legendary',kind='count',data=df,size=5,aspect=1)


# In[ ]:


# similarly for Generation
df['Generation'].value_counts()


# In[ ]:


sns.factorplot(x='Generation',kind='count',data=df,size=5,aspect=1)


# In[ ]:


# viewing the descriptive measures of various  numeric features
df.describe()


# ######  we have now considered each feature one by one and seen its types and distribution.

# ######  now we can depict the  distribution of various other features. or more specifically numeric features.

# In[ ]:


sns.factorplot(data=df,kind='box',size=9,aspect=1.5)


# ######  TO INTERPRET THE BOXPLOT---     
# 
# 1. the bottom line shows the min value of a particular numeric feature.
# 2. the upper line tells the max value.
# 3. the middle line of the box is the median or the 50% percentile.
# 4. the side lines of the box are the 25 and 75 percentiles respectively.

# ######  the above boxplot clearly shows the variation of various numeric features. For eg- consider Total. The df.describe() shows that median  (50%) of Total is 450 and this is also clearly evident from the boxplot corressponding to Total.  

# 

# ###### we can now depict the corelation b/w the various features using a corelation map.

# In[ ]:


cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 


# ###### BREAKING IT DOWN--
# 1. firstly calling .corr() method on a pandas data frame returns a corelation data frame containing the corelation values b/w the various attributes.
# 
# 
# 2. now we obtain a numpy array from the corelation data frame using the np.array method.
# 
# 
# 3. nextly using the np.tril_indices.from() method we set the values of the lower half of the mask numpy array to False.
# this is bcoz on passing the mask to heatmap function of the seaborn it plots only those squares whose mask is False.     therefore if we don't do this then as the mask is by default True then no square will appear. 
# Hence in a nutshell we obtain a numpy array from the corelation data frame and set the lower values to False so that we can visualise the corelation. In order for a full square just use the [:] operator in mask in place of tril_ind... function.
# 
# 
# 4. in next step we get the refernce to the current figure using the gcf() function of the matplotlib library and set the figure size.
# 
# 
# 5. in last step we finally pass the necessary parameters to the heatmap function.
# 
#    DATA=the corelation data frame containing the 'CORELATION' values.
#    
#    MASK= explained earlier.
#    
#    vmin,vmax= range of values on side bar
#    
#    SQUARE= to show each individual unit as a square.
#    
#    ANNOT- whether to dispaly values on top of square or not. In order to dispaly pass it either True or the cor_mat.
#    
#    CBAR= whether to view the side bar or not.

# ######  NOW FINALLY INFERENCES FROM THE PLOT ::)
#  for simplicity I have again plotted the plot.

# In[ ]:


cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 


# ######  1. firstly note that generation doesnt have any corelation with any other feature. so we can drop it iff we want. 

# ###### 2. also note that 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed' are highly correlated to total.

# ###### 3. self realtion i.e. of a feature to itself is always 1.0 .

# ###### 4. note that the corelation of A->B is always equal to that of B->A which is quite obvious and is evident from the below heatmap.

# In[ ]:


# just to show full square. ::)))
cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[:] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 


# ###### now let us see the variation of variables with Type 1

# In[ ]:


# similarly we can do this for type 2.


# In[ ]:


df.head()


# In[ ]:


# we can make a function that take 2 arguements -- the independent variable and the dependent variable.
# the dependent variable will be the categorical variable such as the type 1 or type 2 against which we want to plot--
# the independent variable which will be the numeric variable which we want to plot against the categorical variable.

def comp_against(dep_cat,indep_num,dfd):
#     fig, axes = plt.subplots(3,1)
#     fig.set_size_inches(15, 12)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='bar',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='swarm',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='box',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='strip',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='violin',size=5,aspect=3)


# In[ ]:


# now we can call the function like this. Below I have used Type 1 like a dep variable. Similarly we can do for others.
comp_against('Type 1','Total',df)


# In[ ]:


comp_against('Type 1','HP',df)


# In[ ]:


comp_against('Type 1','Attack',df)


# In[ ]:


comp_against('Type 1','Defense',df)


# In[ ]:


comp_against('Type 1','Sp. Atk',df)


# In[ ]:


comp_against('Type 1','Sp. Def',df)


# In[ ]:


# now similarly we can change the categorical variable like Type 2 etc... 
# and plot various numeric features like Total ,HP etc etc... .


# ######  Lastly we can also compare two pokemons. The function takes the  names of two pokemons and the parameter as input and compares them with on different aspects.

# In[ ]:


def comp_pok(name1,name2,param):
    a = df[(df.Name == name1) | (df.Name ==name2)]
    sns.factorplot(x='Name',y=param,data=a,kind='bar',size=5,aspect=1,palette=['#0000ff','#FFB6C1'])
    


# In[ ]:


# calling the function with differnt paraemters for two dummy pokemons ---   Bulbasaur and Ivysaur


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','Total')


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','HP')


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','Attack')


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','Sp. Atk')


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','Sp. Def')


# In[ ]:


comp_pok('Bulbasaur','Ivysaur','Defense')  


# In[ ]:


# and similarly... we can pass the names of the pokemons and the parameter to compare for any 2 pokemons.


# In[ ]:





# # THE END.

# In[ ]:




