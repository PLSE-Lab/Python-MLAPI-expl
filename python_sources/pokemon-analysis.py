#!/usr/bin/env python
# coding: utf-8

# ## Statistical Analysis ##
# This is my first notebook that I will be running on Kaggle. Also the inspiration to this notebook has come from Ashwin whose visualisation and specifications done on the same Data Set was exemplary. For my love of Pokemon I have taken this small initiative to this notebook. It is a beginner's work, so I hope all of you will like it. 

# In[ ]:


#importing all the modules required
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df =  pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a variable
dff = pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a variable


# So in the first two blocks of code we see that we have imported the files form the Data Base and also, we have imported the necessary libraries. Now lets move into further Data Analysis. 
# 

# In[ ]:


df.head()


# This gives us a brief overview of the Data. 

# In[ ]:


df.describe()#summary of the statistical analysis


# This gives us a summary of the entire statistical Analysis of the Data set we have at hand. 

# In[ ]:


sns.FacetGrid(df, hue="Legendary", size = 10)    .map(plt.scatter, "Attack", "Defense")    .add_legend()


# In this diagram we see that Pokemons who are Legendary in nature tend to have higher attack and Defense properties in comparison to those who are not.
# The non Legendary Pokemons seem to have average to below average Attack and Defense powers which is in line with the concept of Poekmons where the Legendary Pokemons were considered to be the strongest of the lot.

# In[ ]:


print(df.drop(["#","Type 2","Legendary","Type 1","Sp. Atk","Sp. Def"], axis=1).boxplot(by="Generation",figsize=(12, 6)))


# The code above shows the categorical distribution using the boxplots diagram to show the various distributions on the basis of the Generations of the Pokemons.

# In[ ]:


df.columns = df.columns.str.upper().str.replace('_', '')#converting headings to upper case and reducing hyfens
df = df.set_index('NAME') #change and set the index to the name attribute
df.index = df.index.str.replace(".*(?=Mega)", "")#there are many mega pokemons hence this is used to substitute them
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True)#replacing NAN values in Type 2 with corresponding Type 1
df=df.drop(['LEGENDARY','#'],axis=1) #drop the columns with axis=1;axis=0 is for rows


# In[ ]:


df2 = df[(df['TYPE 1']=='Fire')]#selecting fire type 
df2 = df2.drop('GENERATION', axis=1)#dropping generations from total Calculations
print("Fire Pokemon with Max HP:",df2['HP'].argmax())
print("Fire Pokemon with Max SPEED:", df2['SPEED'].argmax())
print("Fire Pokemon with Max Sp.Attack:", df2['SP. ATK'].argmax())
print("Fire Pokemon with Max Sp.Defense:", df2['SP. DEF'].argmax())
print("Fire Pokemon with Max Attack:", df2['ATTACK'].argmax())
print("Fire Pokemon with Max Defense:", df2['DEFENSE'].argmax())
print("Best Fire Pokemon:", df2['TOTAL'].argmax())


# So in this we find out the best Fire Pokemons in the respective fields and then finding out the overall best fire pokemon depending on the total. This can be done for other Type of Pokemons also. One just has to change Fire with the respective Pokemon Type, that is grass, fairy, poison etc.

# In[ ]:


df3 = df[(df['TYPE 1']=='Fire')]#selecting fire type 
df3 = df3.drop(['GENERATION','TOTAL'], axis=1)
sns.boxplot(data = df3)
plt.ylim(0,200)#change the scale of the plot
sns.plt.show()


# We have used BoxPlots to describe the Different attacks of the Fire Pokemon. We see over here that Special Attack is the most power attack for the fire pokemons. This can again be changed by changing the Type to whichever Pokemon you  desire. 

# In[ ]:


# We can look at an individual feature of Pokemons through a boxplot
plt.subplots(figsize = (15,5))
sns.boxplot(x="TYPE 1", y="ATTACK", data=df)
sns.plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
sns.boxplot(x="TYPE 1", y="DEFENSE", data=df)
sns.plt.show()


# From the above Box Plots we can conclude that the Dragon Pokemon has the highest attack whereas the steel Pokemon has the highest defense. 
# 

# In[ ]:


temp2 = dff.pivot_table(values='Legendary',index=['Generation'],aggfunc=np.mean)
print('\nProbility of having a Legendary Pokemon for each Generation')
print(temp2)


# Over here we use a pivot table to calculate the probability of having a Legendary Pokemon based on the Generation type.
# The highest was in case of the third Generation. 

# In[ ]:


temp3 = pd.crosstab(dff['Type 1'], dff['Legendary'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# The graph above shows the Possibility of having a Legendary Pokemon based on the Type of Pokemon. Highest is in case of Psychic.

# In[ ]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))


# In[ ]:


#Logistic Regression Test
outcome_var = 'Legendary'
model = LogisticRegression()
predictor_var = ['Generation']
classification_model(model, dff,predictor_var,outcome_var)


# We did a Logistic Regression test on the Data set to see how much Accuracy we are getting between the Generation of the Pokemon and whether or not it is Legendary and got the above results. 

# In[ ]:




