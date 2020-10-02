#!/usr/bin/env python
# coding: utf-8

# # Titanic Passenger Data Analysis
# 
# This notebook contains my analysis of the Titanic passenger data. This is my first upload here to Kaggle. I decided to upload this only after I've made it, so apologies if it is not 100% readable for anyone. Next one will be better :-D. Open to any feedback, comments or suggestions! 
# 
# ![image of Titanic](https://vignette.wikia.nocookie.net/titanic/images/6/6e/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg/revision/latest?cb=20171123230214)
# 
# The process wil consist of data asking the right questions, data wrangling, exploring and communicating. 
# 
# ## Questions
# The data contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic. The data includes: 
# - Survival 
# - Ticket class
# - Sex
# - Age in years
# - Number of siblings / spouses aboard Titanic
# - Number parents / children aboard Titanic 
# - Ticket number 
# - Passenger fare
# - Cabin number 
# - Port of embarkation
# 
# In the end, it seems interesting to see the relation between several aspects and survival. However, first some more general questions: 
# - How many of these 891 passengers are first, second, third class or crew? 
# - How many survived of these? How does this relate to the overal ratio of about 64% of people on board who perished? 
# - What is the ratio between men and women on board? How does this vary per class? 
# - What percentage of passengers was travelling without family? 
# - What percentage of passengers embarked at one of the 3 ports (Cherbourg, Queenstown, Southampton)
# 
# Then, when we have this general sense of all the data, we can start relating things to survival rate. Let's look at survivalrates for: 
# - Men vs. women 
# - Classes or crew
# - Baby's, children, adults 
# - People travelling alone versus people travelling with family 
# 
# Finally, it would be interesting to see the relation between survival and ticket fare. 
# 
# Anyway... let's start there. Seems like a good point!!
# 
# 

# ## Data wrangling
# Before we do anything, we need to load our data, check and correct anomalies, and format it in a clear an concise manner.

# In[79]:


# Right, first some modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="whitegrid")
sns.set_palette('Blues')

# Some magic
get_ipython().run_line_magic('matplotlib', 'inline')

# Now let's load some data 
rawdata_df = pd.read_csv('../input/titanic_data.csv')


# In[80]:


rawdata_df.head()


# In[81]:


rawdata_df.tail()


# In[82]:


rawdata_df.describe()


# I notice a couple of things. First, the passengerid column seems fit to be our index column. Let's fix that. 

# In[83]:


df = rawdata_df.set_index('PassengerId')
df.head()


# Better, however... it seems as though there is actually no crew in this database. Let's see if we can identify anything that would set crew apart from passengers. Lets try Pclass, ticket, and fare columns. We'll limit ourselves to the first rows. 

# In[84]:


print(set(rawdata_df['Pclass'].values))
print(set(rawdata_df['Ticket'].values[0:50]))
print(set(rawdata_df['Fare'].values[0:50]))
print(set(rawdata_df['Embarked'].values))
print(set(rawdata_df['Sex'].values))
print(rawdata_df['Age'].isnull().sum())


# Nope, no crewmembers. Finding number one. Also intersting is the fact that some people have no cabin number. From the small excerts we have seen, this seems to occur typically in third class, which would make sense. Let's check this. First we'll have to change the NaN values to 'Unknown', because column with strings and NaNs doesn't work well. 

# In[85]:


df["Cabin"] = df["Cabin"].fillna("Unknown")
df['Embarked'] = df['Embarked'].fillna('Unknown')
print ("Percentage of people with unknown room in 1st class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 1)).sum()*100 / float((df["Pclass"] == 1).sum())))
print ("Percentage of people with unknown room in 2nd class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 2)).sum()*100 / float((df["Pclass"] == 2).sum())))
print ("Percentage of people with unknown room in 3rd class: {0:.1f}%"
    .format(((df["Cabin"] == "Unknown") & (df["Pclass"] == 3)).sum()*100 / float((df["Pclass"] == 3).sum())))


# Right, it seems this column is not too reliable. Let's discard it. 
# 
# Other findings: 
# - Ticket numbers seem a bit random. Let's discard it as well. 
# - Some people have not given their age. Too bad.
# 
# However, apart from this, this dataset seems pretty clean and straightforward. Let's rename a few columns.

# In[86]:


ports = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S' : 'Southampton'}
classes = {1: 'First', 2: 'Second', 3: 'Third'}

def set_port(port):
    if port == 'Unknown':
        return 'Unknown'
    else:
        return ports[port]

def set_class(classn):
    if math.isnan(classn):
        return 'Unknown'
    else:
        return classes[classn]

df['Embarked'] = df['Embarked'].apply(set_port)
df['Pclass'] = df['Pclass'].apply(set_class)


# In[87]:


print(df.groupby('Embarked').size())


# In[88]:


df.rename(columns = {'Survived' : 'Survival rate', 'Pclass' : 'Class', 'Embarked' : 'Port of embarkment'}, inplace=True)
df.head()


# Yay!! Time for some analysis. 

# ## Exploration
# First lets look at the overall mortality rate. According to wikipedia, overall survivalrate was ~36%. 

# In[89]:


print("Survivalrate: {}%".format(round(df['Survival rate'].mean()*100)))


# Ok, that matches up rather well. Now lets group by class, to see how that is divided. 

# In[90]:


df_byclass = df.groupby(['Class','Port of embarkment'])

graph_data = pd.DataFrame(df_byclass.size(), columns={'Passengers'}).reset_index()
print(graph_data)


# It seems there are two unkown passengers, but let's neglect them. 

# In[91]:


ax = graph_data.pivot(index='Class', columns='Port of embarkment', values='Passengers').plot(kind='bar', stacked=True)
ax.set_ylabel('Passengers');


# Now, lets look at ticket prices, gender ratio and ticket prices per class.

# In[92]:


df.groupby('Class')['Fare'].describe()


# 
# Alright, pretty much what we would expect. However, there seems to be people who did not pay anything. Let's dive into that, and see if there are any anomalies there we need to correct.

# In[93]:


df.loc[df.Fare == 0]


# Okay, nothing special to see here. Guess these people just kinda got lucky, or something... doesn't give a sense of reliability on the account of the Fare column. Too bad. 
# 
# Right, where were we?? Oh yeah... gender ratio's per class. 

# In[94]:


graph_data = pd.DataFrame(df.groupby(['Class', 'Sex']).size(), columns = {'Passengers'}).reset_index()
ax3 = sns.barplot(x="Class", y="Passengers", hue="Sex", data=graph_data)


# Apparently, the lower you go in society, the more male dominated the world becomes. Mo' money, mo' bitches... apparently. 
# 
# Finally, let's look at age in these categories. 

# In[95]:


ax2 = sns.boxplot(x="Class", y="Age", 
                  hue="Sex", order = classes.values(), hue_order=['male', 'female'], 
                  data=df[np.isfinite(df['Age'])])


# Interesting. I see 2 things: 
# - Rich people are older
# - Women are younger
# 
# One thing that surprises me is that the age difference between men and women is highest in both 1st and 3rd class. My hypothesis was something like "rich men have younger girlfriends". Still true, but apparatently poor men also have young girlfriends. Hmm... don't know what to make of this yet. 
# 
# ## Survival rates
# Time to look at the most morbid part of all of this. Let's see what relation between fatality rates and other parameters we can find. Let's begin with the classes. 

# In[96]:


ax5 = sns.barplot(x="Class", y="Survival rate", order = classes.values(), data=df, ci=None)
ax5.set_ylim([0,1]);


# As expected, we see that first class passengers had a better change of surviving the tragic ordeal. But let's see what happens when we add gender to the mix. 

# In[97]:


ax4 = sns.barplot(x="Class", y="Survival rate", hue="Sex"
                  , order = list(classes.values()), hue_order=['male', 'female'], data=df, ci=None)


# Very interesting!! It seems women did go first!! 
# 
# Also, we see that first class women have a ~95% chance of surviving, whereas 3rd class men have a ~15% chance of surviving. No wonder Rose lived, and Jack did not!
# 
# Next up... children first?? 
# 
# However, first we need to drop the columns with unkown age! 

# In[98]:


def is_child(age): 
    if age < 18:
        return "Child"
    else:
        return "Adult"

df['Agegroup'] = df['Age'].apply(is_child)
df_knownage = df[np.isfinite(df['Age'])]
ax7 = sns.barplot(x='Class', y='Survival rate', 
                  order = list(classes.values()), hue='Agegroup', hue_order = ['Child', 'Adult'], data=df_knownage, ci=None)


# Interesting, let's dive a little deeper. 

# In[99]:


agegroups = ['Infant', 'Child', 'Teenager', 'Adolescent', 'Adult', 'Senior']

def agegroup(age): 
    if age < 4.:
        return agegroups[0]
    elif age < 10.: 
        return agegroups[1]
    elif age < 20.: 
        return agegroups[2]
    elif age < 30.: 
        return agegroups[3]
    elif age < 65.: 
        return agegroups[4]
    else:
        return agegroups[5]
    
df['Agegroup'] = df['Age'].apply(agegroup)
df_knownage = df[np.isfinite(df['Age'])]
ax = df_knownage.groupby(['Class','Agegroup']).size().unstack()[agegroups].plot(kind='bar', stacked=True);
ax.set_ylabel('Passengers');


# In[100]:


ax8 = sns.barplot(x='Class', y='Survival rate', 
                  hue='Agegroup', order = list(classes.values()), hue_order = agegroups, data=df_knownage, ci=None)


# So, I guess we can conclude it was woman and children first, especially if you are born with a silver spoon in hand. Interesting to see that teenagers are apparently not considered to be children. 
# 
# Another interesting thing is that there must have been one first class Baby/Toddler that did not make it out alive. Let's see who the poor soul was.  

# In[101]:


df.loc[(df['Class'] == 'First') & (df['Survival rate'] == 0) & (df['Agegroup'] == 'Infant')]


# Oh wow... there is actually quite the story behind all this. You can read it here in this [link](https://www.telegraph.co.uk/history/10581757/Lost-child-of-the-Titanic-and-the-fraud-that-haunted-her-family.html).

# ![Image of Helen](https://secure.i.telegraph.co.uk/multimedia/archive/02795/loraine_2795338b.jpg)

# ## Using Machine Learning/Random Forest to predict feature importance for survival
# Based on feedback I got from a friend, I will try to use a random forest classifier to predict feature importance in predicting survival rates. I will use the Scikit Learn toolbox for this.

# In[102]:


from sklearn.ensemble import RandomForestClassifier

# Make train data
feature_cols = ['Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Port of embarkment' ]
bool_cols = ['Class', 'Sex','Port of embarkment' ]
X = df[feature_cols]
y = df['Survival rate']

print(X.isnull().sum())
print (X.shape)
print (y.shape)


# There is quite a few NaNs in the age column, which might distort the feature importance (since we already know most if these NaNs are in third class, this is rather likely to combine the age and class features). Let's clear them out. 

# In[103]:


rows = np.isfinite(X.Age)
X = X.loc[rows]
y = y.loc[rows]

print (X.isnull().sum())
print (X.shape)
print (y.shape)


# In[104]:


print (sum(X.Fare == 0))


# So there are 7 rows with unkown fare. Let's take them out as well. 

# In[105]:


rows = X.Fare != 0
X = X.loc[rows]
y = y.loc[rows]


# Now let's encode all the rows, save the encoders (so we can use them later to code and decode stuff) and train the classifier. 

# In[106]:


# Normalize string labels 
from sklearn.preprocessing import LabelEncoder

labelencoders = {}
for column in bool_cols:
    labelencoders[column] = LabelEncoder().fit(X[column])

X[bool_cols] = X[bool_cols].apply(LabelEncoder().fit_transform)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X,y)

# Get importance 
output = pd.DataFrame(clf.feature_importances_, index=feature_cols, columns=['Feature importance'])
output = output.sort_values(by='Feature importance', ascending=False)
print (output)


# Interesting. Fare is way more important than class. One would think these two are about equal in importance, since they should be somehow correlated. Let's compare distribution of fare for those who survived with those who didn't for the three classes. 

# In[107]:


ax8 = sns.boxplot(x="Class", y="Fare", 
                  hue="Survival rate", order = classes.values(), data=df_knownage)
ax8.set_ylim([0,180]);


# So apparently, it was women, children and money first, especially in second and first class. I wonder what this says about how things went down in those two hours. Did people buy their way to safety? Or where the more expensive cabins better positioned (e.g. closer to the lifeboats, were alarmed earlier). 
# 
# Now let's look at what our classifier makes of the Rose & Jack story. So we know rose was 17 and Jack was 20. Rose travelled first class, Jack third. Both had no siblings on board, but Rose had her mother. Both embarked in Southampton. And finally, for fare we will take the mean for respectively a first and third class ticket. 

# In[108]:


Rose = dict(zip(feature_cols, ['First', 'female', 17, 0, 1, 87.961582, 'Southampton']))
Jack = dict(zip(feature_cols, ['Third', 'male', 20, 0, 0, 13.229435, 'Southampton']))

for column in bool_cols:
    Rose[column] = labelencoders[column].transform([Rose[column]])
    Jack[column] = labelencoders[column].transform([Jack[column]])

Rose_norm = []; Jack_norm = []
for column in feature_cols:
    Rose_norm.append(Rose[column])
    Jack_norm.append(Jack[column])

predict = clf.predict(np.array([Rose_norm, Jack_norm]))
predict_proba = clf.predict_proba(np.array([Rose_norm, Jack_norm]))

print ('Likelihood of Rose surviving Titanic disaster: {}%'.format(predict_proba[0,1]*100))
print ('Likelihood of Jack surviving Titanic disaster: {}%'.format(predict_proba[1,1]*100))

strings = ['perishes!!', 'survives!!']
print ('\nCLASSIFIER PREDICTIONS:')
print ('Rose ' + strings[predict[0]])
print ('Jack ' + strings[predict[1]])


# So... my predictor has correctly predicted the death of Jack, and the survival of Rose. Woohaa!!!
# 
# ![Image of rosejack](http://4.bp.blogspot.com/-mHEPVnt3j4s/UGMmfyH12vI/AAAAAAAAAVo/b6HYmB-75gQ/s1600/jack-rose-raft-600x290.jpeg)

# ## Conclusions
# Well, that pretty much concludes my research on this topic. Main conclusions: 
# - Women and childer did go first, but also did money
# - Teenagers are not children 
# - Mo' money, mo' bitches
# - Rich men have young bitches
# - Poor men have young bitches as well
# - People shouldn't make such a fuss about the whole Jack & Rose thing. Statistics pretty much dictate that she should live, and he shoud die. Period. And no... he wouldn't have fit on the bloody door. 
# 
# ## Important learnings
# Next time, check all the columns for NaNs at the beginning. Also, at the beginning, rename all the entries. The script should be: 
# - For numerical columns: check maximum, minimum, mean, stdev, and check for NaNs
# - For string columns: check all possible values using set, check number of NaNs. Rename if necessary. 
# 
