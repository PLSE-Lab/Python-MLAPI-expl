#!/usr/bin/env python
# coding: utf-8

# In this notebook we will explore realtionship of input variables with each other and more importantly with the output variable 'Survived'. These insights will help us in making choices while creating the model to fit the training data.
# In this notebook we will follwo below course to check the relationship betwen different variables.
# 1. Find out datatype of different columns
# 2. Create a function to plot distribution of categotical variables and try to make inferences from based on chart
# 3. Extract additional features from dataset that are hidden in noise, hence not readily available for analysis

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# Now find out number of distinct values in each column. Knowing unique value in each column will help us in deciding which combinations of variable to plot as what kind of chart. Looking at the data above, we can clearly see that columns Name, Age, Ticket and Fare do not have categorical data. Age and Fare are continuous numeric data whereas NAme  and Ticket are strings.
Two factors that greatly influence charting and understading of data are number of unique values in categorical columns and number of records with missing data.

Let's check number of unique records in each categorical column:
# In[ ]:


l_cat = ['Survived', 'Pclass','Sex','SibSp','Parch','Cabin', 'Embarked']
df_train[l_cat].nunique()


# Cabin column has 147 values. That is a lot to make sense in any king of categorical chart. Let's see what are those values and if there is any way to extract some feature from it.

# In[ ]:


df_train['Cabin'].unique()


# Looks like we can extract the first letter of Cabin number to enginner new feature. Cabin number spans from Axx to Fxx; so we will be able to reduce number of unique values from 147 to 6 by extracting first letter. Although it is not clearly mentioned in data describtion, first letter of ticket may be indicative of row of rooms or berth. In either case it is worth testing if that factor has any influence on survival rate of passengers
# 
# Let's move on to checking number of missing values in each column:

# In[ ]:


df_train.isnull().sum(axis=0)


# Cabin and Age comuns have a lot of rows with missing data. We need to be careful while interpreting the charts containing these two variables because distribution seen in chart may not be true distribution of data.

# **First steps in plotting categorical data:**
# 
# An intuitive approach to see whether a variable is  significantly affecting the outcome is to see distribution of category and outcome. If a particular category is producing certain outcome a lot more or lot less frequently compared expected outcome based purely on average, then there is more incentive for us to investigate that category.
# 
# For example if we have dataset of 200 movies produced by a Mr. X and Mr.Spielberg, 100 each. For Mr. X, dataset has 50 hit and 50 flop movies. For Mr Spielberg, dataset has 80 hit and 20 flop movies. In this case, since Mr Spielberg's count of hit movies(80) is far more that statistical expectation of 50 hit movies, we know that whether a movie is directed by Mr Spielberg is an important factor in deciding whether a movie will be a hit.
# 
# To visualize this relationship, I have written below function that represents importance of categorical factor in the form of size and color of bubble. Shade of green color is used to show high positive difference from expected average outcome and shade of red color is used to  show high negative diffrence from extected average outcome. If outcomes are around expected average outcome then bubles are painted in shade of yellow. Code for this function is as follows:

# In[ ]:


def bubblechart(ip,op,count_col,data):
    f, ax= plt.subplots(len(ip),1, figsize=(10,5*len(ip)), squeeze=False)
    
    # Find unique output values 
    y_tick_label = pd.Series(data[op[0]].unique()).sort_values().tolist()
    y_tick_value = pd.Series(np.arange(1,len(y_tick_label)+ 1,1)).tolist()
    
    for i,ip_param in enumerate(ip):
        
        # Find unique input values
        x_tick_label = pd.Series(data[ip_param].unique()).sort_values().tolist()
        x_tick_value = pd.Series(np.arange(1,len(x_tick_label)+ 1,1)).tolist()
         
        #Calculate parameters required for bubblechart
        gr_ip_op = data.groupby([ip_param]+op)[count_col].count()
        gr_ip = data.groupby([ip_param])[count_col].count()
        df1 = pd.DataFrame(gr_ip_op).reset_index()
        df2 = pd.DataFrame(gr_ip).reset_index()
        df3 = df1.merge(df2,how='inner',on=ip_param)
        df3['proportion'] = df3[count_col+'_x']/df3[count_col+'_y']*3000.0
        df3['x_tick_value'] = df3[ip_param].apply(lambda x : x_tick_value[x_tick_label.index(x)] )
        df3['y_tick_value'] = df3[op[0]].apply(lambda x : y_tick_value[y_tick_label.index(x)] )
        
        #Plot the bubblechart
        x=df3['x_tick_value']
        y=df3['y_tick_value']
        s=df3['proportion']
        c=df3['proportion']/3000.0
        ax[i,0].scatter(x=x,y=y,s=s,c=c, cmap='RdYlGn')
        ax[i,0].grid(b=True)
        
        ax[i,0].set_xlabel(ip[i])
        ax[i,0].set_xticks(x_tick_value)
        ax[i,0].set_xticklabels(x_tick_label,rotation=45)
        
        ax[i,0].set_ylabel(op[0])
        ax[i,0].set_yticks(y_tick_value)
        ax[i,0].set_yticklabels(y_tick_label)
        
        
    plt.tight_layout 


# To begin with, let's investigate relationship between sex and survival on Titanic

# In[ ]:


bubblechart(['Sex'],['Survived'],'PassengerId',df_train)


#  In this case expected average outcome for males to survive was 50% (since only two outcome are possible) but much less than 50% males survived. This is shown by small red bubble on top right corner.
# 
# We can clearly see from above chart that there is a strong relation between sex of passenger and survival rate. If you are a male you are much more likely to see Jack at the bottom of the ocean. If you are a female then you are more likely to  live to tell your tale and make a movie about it.

# Let's check the relationship with other variables all at once.

# In[ ]:


bubblechart(['Pclass','SibSp','Parch'],['Survived'],'PassengerId',df_train,)


# Based on these charts, we can notice below pattern :
# 
# * Pclass : 
#     *     If you are a Pclass=1 passenger then you are much more likely to have survived compared to passengers in Pclass = 3. 
#     *     Pclass = 2 is not affecting survival rate in either positive or negative direction. 
#     
# * SibSp :
#     *     More the number of siblings on titanic, less were the chances that passenger survived
#     *     This means that we can use this variable as a continuous variable i.e we do not have to create dummy variables for this category when we are developing predictive model.
#  
# *  Parch
#     *    Survival rate looks directly proportional to number of parent / children passenger have on Titanic.
#     *    Just like SibSp, even for Parch we can use continuous variable while modelling instead of using dummy for each category.    

# Now, as promised, we will extract some features from text and see whether any of those features show significant relationship with survival.
# 
# We will extract two features
# 1. Title of the passanger
# 2. First letter of the cabin number

# **Find relationship between title and survival rate:**
# 
# First we have to write a function that extracts title of the passenger. We will use regular expressions to do this[](http://)

# In[ ]:


import re
def findtitle(str_name):
    m = re.search(r', (?P<Title>.*)\.',str_name)
    return m.group('Title')


# In[ ]:


df_train['Title'] = df_train['Name'].apply(findtitle)


# In[ ]:


bubblechart(['Title'],['Survived'],'PassengerId',df_train)


# Above chart shows that there are some titles that have very high chance of survival whereas some other titles have very low chance of surviving. This chart shows that we can use title as variable (with one dummy variable for each category) to predict chances of survival..

# **Find realtionship between cabin letter and survival rate**
# 
# while checking the result of this variable, we will have to remember that more than 50% of the data for this column is missing.

# In[ ]:


df_train['Cabin'] = df_train['Cabin'].apply(lambda x : 'X' if pd.isnull(x) else x[:1])


# In[ ]:


bubblechart(['Cabin'],['Survived'],'PassengerId',df_train)


# If we were to use cabin as a  categorical variable in our model, we will use B, D, E T and X as dummy categorical variables. Important thing to notice here is that even if we have assigned placeholder cabin X to all the passengers with no cabin name, they are significantly more likely to die than survive. Possibly there is an underlying pattern with passengers that were missing cabin number like thier cabin numbers were not meticulously documented because they had baught cheap tickets and such passengers were given less preference in rescue effort.

# In second part I will use insights from this excercise and build a predictive model.
