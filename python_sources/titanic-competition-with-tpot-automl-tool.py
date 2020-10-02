#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Create scaler: scaler
#scaler = StandardScaler()


# In[ ]:


ls ../input


# In[ ]:


titanic = pd.read_csv('../input/train.csv')
titanic.head(5)


# # Data Exploration

# In[ ]:


titanic.groupby('Sex').Survived.value_counts()


# In[ ]:


titanic.groupby(['Pclass','Sex']).Survived.value_counts()


# In[ ]:


id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived)
id


# # Data Munging

# The data set has 5 categorical variables which contain non-numerical values: Name, Sex, Ticket, Cabin and Embarked.

# In[ ]:


titanic.dtypes


# Let's check the number of levels that each of the five categorical variables have.

# In[ ]:


for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
    print("Number of levels in category '{0}': \b {1:2d} ".format(cat, titanic[cat].unique().size))


#  Sex and Embarked have few levels. Let's find out what they are.

# In[ ]:


for cat in ['Sex', 'Embarked']:
    print("Levels for category '{0}': {1}".format(cat, titanic[cat].unique()))


# Let's code these levels manually into numerical values. For nan i.e. the missing values, let's simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set.

# In[ ]:


titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


titanic = titanic.fillna(-999)
pd.isnull(titanic).any()


# Survived vs Embarked

# In[ ]:


pd.crosstab(titanic.Embarked, titanic.Survived)


# In[ ]:


# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# let's plot many different shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
titanic.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    

plt.subplot2grid((2,3),(0,1))
plt.scatter(titanic.Survived, titanic.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
titanic.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(titanic.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
titanic.Age[titanic.Pclass == 1].plot(kind='kde')    
titanic.Age[titanic.Pclass == 2].plot(kind='kde')
titanic.Age[titanic.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
titanic.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(titanic.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")


# Drop the unused features from the dataset.

# In[ ]:


titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)


# In[ ]:


np.isnan(titanic_new).any()


# In[ ]:


titanic_new.info()


# Finally we store the class labels, which we need to predict, in a separate variable.

# In[ ]:


titanic_class = titanic['class'].values


# # Data Analysis using TPOT

# To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error. The model selection and tuning is entirely taken care of by TPOT, so if we want to, we can skip creating this validation set.

# In[ ]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size


# 
# 
# After that, we proceed to calling the fit, score and export functions on our training dataset. To get a better idea of how these functions work, refer the TPOT documentation [here](http://epistasislab.github.io/tpot/api/).
# 
# 
# 

# In[ ]:


tpot = TPOTClassifier(generations=999,verbosity=5,population_size=9990,n_jobs=10)
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])


# In[ ]:


tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)


# In[ ]:


tpot.export('tpot_titanic_pipeline.py')


# Let's have a look at the generated code. As we can see, the random forest classifier performed the best on the given dataset out of all the other models that TPOT currently evaluates on. If we ran TPOT for more generations, then the score should improve further.

# In[ ]:


get_ipython().run_line_magic('load', 'tpot_titanic_pipeline.py')


# ## Make predictions on the submission data

# In[ ]:


# Read in the submission dataset
titanic_sub = pd.read_csv('../input/test.csv')
titanic_sub.describe()


# The most important step here is to check for new levels in the categorical variables of the submission dataset that are absent in the training set. We identify them and set them to our placeholder value of '-999', i.e., we treat them as missing values. This ensures training consistency, as otherwise the model does not know what to do with the new levels in the submission dataset.

# In[ ]:


for var in ['Cabin']: #,'Name','Ticket']:
    new = list(set(titanic_sub[var]) - set(titanic[var]))
    titanic_sub.loc[titanic_sub[var].isin(new), var] = -999


# We then carry out the data munging steps as done earlier for the training dataset.

# In[ ]:


titanic_sub['Sex'] = titanic_sub['Sex'].map({'male':0,'female':1})
titanic_sub['Embarked'] = titanic_sub['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


titanic_sub = titanic_sub.fillna(-999)
pd.isnull(titanic_sub).any()


# While calling MultiLabelBinarizer for the submission data set, we first fit on the training set again to learn the levels and then transform the submission dataset values. This further ensures that only those levels that were present in the training dataset are transformed. If new levels are still found in the submission dataset then it will return an error and we need to go back and check our earlier step of replacing new levels with the placeholder value.

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic_sub['Cabin'].values])
titanic_sub = titanic_sub.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


# Form the new submission data set
titanic_sub_new = np.hstack((titanic_sub.values,SubCabinTrans))


# In[ ]:


np.any(np.isnan(titanic_sub_new))


# In[ ]:


# Ensure equal number of features in both the final training and submission dataset
assert (titanic_new.shape[1] == titanic_sub_new.shape[1]), "Not Equal"


# In[ ]:


# Generate the predictions
submission = tpot.predict(titanic_sub_new)


# In[ ]:


# Create the submission file
final = pd.DataFrame({'PassengerId': titanic_sub['PassengerId'], 'Survived': submission})
final.to_csv('submission.csv', index = False)


# In[ ]:


final.shape


# There we go! We have successfully generated the predictions for the 418 data points in the submission dataset, and we're good to go ahead to submit these predictions on Kaggle.
