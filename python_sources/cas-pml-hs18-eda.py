#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

print('-'*25)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Evaluation
from sklearn.metrics import cohen_kappa_score,make_scorer
from sklearn.model_selection import StratifiedKFold

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from wordcloud import WordCloud
from matplotlib.colors import ListedColormap

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('darkgrid')
pylab.rcParams['figure.figsize'] = 12,8


# ## The target variable: Adoption Speed
# 
# * 0 - Pet was adopted on the same day as it was listed.
# * 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
# * 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
# * 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
# * 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

# ## Data Fields
# 
# * PetID - Unique hash ID of pet profile
# * AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# * Type - Type of animal (1 = Dog, 2 = Cat)
# * Name - Name of pet (Empty if not named)
# * Age - Age of pet when listed, in months
# * Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# * Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# * Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# * Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# * Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# * Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# * MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# * FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# * Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# * Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# * Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# * Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# * Quantity - Number of pets represented in profile
# * Fee - Adoption fee (0 = Free)
# * State - State location in Malaysia (Refer to StateLabels dictionary)
# * RescuerID - Unique hash ID of rescuer
# * VideoAmt - Total uploaded videos for this pet
# * PhotoAmt - Total uploaded photos for this pet
# * Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

# In[ ]:


breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'


# In[ ]:


train['Type'] = train['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
test['Type'] = test['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')


# There is a difference in the distribution of the pet type in both sets. We have to see if this impacts in the prediction score later on. 

# In[ ]:


color2 = ["#ffa600","#003f5c"]
color3 = ["#ffa600","#bc5090","#003f5c"]
color4 = ["#ffa600","#ef5675","#7a5195","#003f5c"]
color5 = ["#ffa600","#ff6361","#bc5090","#58508d","#003f5c"]
color6 = ["#ffa600","#ff6e54","#dd5182","#955196","#444e86","#003f5c"]
color7 = ["#ffa600","#ff764a","#ef5675","#bc5090","#7a5195","#374c80","#003f5c"]


# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Type', palette=color2);
plt.title('Number of cats and dogs in train and test data');


# In[ ]:


g = sns.countplot(x='AdoptionSpeed', hue='Type', data=train, palette=color2)
plt.title('Adoption speed classes rates');
ax=g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  


# In[ ]:


cats = train.loc[train['Type'] == 'Cat']

g = sns.countplot(x='AdoptionSpeed', data=cats, palette=color5)
plt.title('Adoption speed for cats');
ax=g.axes
for p in ax.patches:
      ax.annotate(f"{p.get_height() * 100 / cats.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  


# Looks like cats are adopted a bit faster, but if they're not adapted after 1 months, they are les likely to get adapter during month 2 and 3.

# In[ ]:


dogs = train.loc[train['Type'] == 'Dog']

g = sns.countplot(x='AdoptionSpeed', data=dogs, palette=color5)
plt.title('Adoption speed for dogs');
ax=g.axes
for p in ax.patches:
      ax.annotate(f"{p.get_height() * 100 / dogs.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  


# Dogs are adapted at a slower rate than cats. Only after one month they are more likey to be adpoted. 

# In[ ]:


# copy from: https://www.kaggle.com/artgor/exploration-of-data-step-by-step
main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()

def prepare_plot_dict(df, col, main_count):
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())
        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0
    return plot_dict

def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count):
    """
    Plotting countplot with correct annotations.
    """
    g = sns.countplot(x=x, data=df, hue=hue, palette=color5);
    plt.title(f'AdoptionSpeed {title}');
    plt.legend(["1st day", "1st week", "1st month", "2nd & 3rd month", "never"]);
    ax = g.axes
   
    plot_dict = prepare_plot_dict(df, x, main_count)

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),
             textcoords='offset points')  


# In[ ]:


make_count_plot(df=train, x='Type', title='by pet Type')


# In[ ]:


def show_adaptionspeed_barplot(df, compare_column, column_names=None, label=None):
    
    # prepare columns to show
    unique_column_names = sorted(df[compare_column].unique())
    index = list(range(0,len(unique_column_names)))
    if column_names is None:
        column_names = unique_column_names
    
    # calculate % for all AdoptionSpeeds
    df = df.groupby([compare_column, 'AdoptionSpeed']).size().reset_index().pivot(columns='AdoptionSpeed', index=compare_column, values=0)
    totals = [i+j+k+l+m for i,j,k,l,m in zip(df[0], df[1], df[2], df[3], df[4])]
    speed0 = [i / j * 100 for i,j in zip(df[0], totals)]
    speed1 = [i / j * 100 for i,j in zip(df[1], totals)]
    speed2 = [i / j * 100 for i,j in zip(df[2], totals)]
    speed3 = [i / j * 100 for i,j in zip(df[3], totals)]
    speed4 = [i / j * 100 for i,j in zip(df[4], totals)]

    # plot
    barWidth = 0.85
    plt.bar(index, speed0, color='#ffa600', edgecolor='white', width=barWidth, label="1st day")
    plt.bar(index, speed1, bottom=speed0, color='#ff6361', edgecolor='white', width=barWidth, label="1st week")
    plt.bar(index, speed2, bottom=[i+j for i,j in zip(speed0, speed1)], color='#bc5090', edgecolor='white', width=barWidth, label="1st month")
    plt.bar(index, speed3, bottom=[i+j+k for i,j,k in zip(speed0, speed1, speed2)], color='#58508d', edgecolor='white', width=barWidth, label="2nd & 3rd month")
    plt.bar(index, speed4, bottom=[i+j+k+l for i,j,k,l in zip(speed0, speed1, speed2, speed3)], color='#003f5c', edgecolor='white', width=barWidth, label="never")

    # Custom x axis     
    plt.xticks(index,column_names)
    plt.xlabel(label)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    # Show graphic
    plt.show()
     


# In[ ]:


show_adaptionspeed_barplot(train, 'Type', ['Dog','Cat'], 'AdoptionSpeed by Type')


# ## Name EDA 
# Now we look at the name of the pet. 

# In[ ]:


print("Missing names in train set: %d" % pd.isna(train['Name']).sum())
print("Missing names in test set: %d" % pd.isna(test['Name']).sum())


# In[ ]:


train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')

train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)
test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)


# In[ ]:


pd.crosstab(train['has_name'], train['AdoptionSpeed'], normalize='index')


# In[ ]:


make_count_plot(df=train, x='has_name', title='by name available')


# Having no name apears to be quite bad. They are less likely to be adopted compared to the base rate. Having a name seems to have no impact on the adaption rate.
# 
# What about the name itself? Does a common name help for the adaption? Or is a special name more helpful?

# In[ ]:


train.Name.value_counts().head(10)


# As we can see, there are several pets with a name "No Name". We should treat them as has_name = false and see if it has an impact.

# In[ ]:


train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)
test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)


# In[ ]:


make_count_plot(df=train, x='has_name', title='by name available')


# As we can see, there is quite a change on the first day, dropping from -25% to -19%. This could be due to the fact, that there are not many instances for the first day. 

# In[ ]:


show_adaptionspeed_barplot(train, 'has_name', ['No','Yes'], 'AdoptionSpeed by has_name')


# In[ ]:


# defining a function which returns a list of top names
def top_names(df, top_percent):
    df_withnames = df[df.has_name != 0]
    items = df_withnames.shape[0]
    top_names = []
    counter = 0
    for i,v in df_withnames.Name.value_counts().items():
        if (counter/items)>top_percent:
            break
        top_names.append(i)
        counter = counter + v  
    return top_names


# In[ ]:


top_names(train, 0.05)


# In[ ]:


topnames = top_names(train, 0.2)
train['has_topname'] = train['Name'].apply(lambda row: 1 if row in topnames else 0)
make_count_plot(df=train, x='has_topname', title='by topname')


# In[ ]:


show_adaptionspeed_barplot(train, 'has_topname', ['No','Yes'], 'AdoptionSpeed by has_topname')


# In the training set we can see that there is quite an effect when having a top name. 

# ## Age EDA

# In[ ]:


print("Missing Age in train set: %d" % pd.isna(train['Age']).sum())
print("Missing Age in test set: %d" % pd.isna(test['Age']).sum())


# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.title('Distribution of pets age')
train['Age'].plot('hist', label='train',colormap=ListedColormap(color2[0]))
test['Age'].plot('hist', label='test',colormap=ListedColormap(color2[1]))
plt.legend();

plt.subplot(1, 2, 2)
plt.title('Distribution of pets age (log)')
np.log1p(train['Age']).plot('hist', label='train', colormap=ListedColormap(color2[0]))
np.log1p(test['Age']).plot('hist', label='test', colormap=ListedColormap(color2[1]))
plt.legend();


# Looks like there is a difference in the train and test set regarding the ages of the pets. There are less younger pets in the test set. 
# What about the age iteself?

# In[ ]:


sns.distplot(train["Age"], kde=True, color=color2[0])


# There is a clear pattern visible, each 12 month there is a peak. We can also see, that there are some 0 values. Lets check those.

# In[ ]:


print("pets with age 0: %d" % len(train[train.Age ==0]))
print("pets from a group: %d" % len(train[train.Gender==3]))
print("pets with age 0 and in a group: %d" % len(train[(train.Gender==3) & (train.Age==0)]))


# In[ ]:


train[(train.Gender==3) & (train.Age==0)].head()


# In some cases, there are kittens with her mother with an age of 0. But sometimes its just a group of some kittens and thus correct with age 0. 
# We can try to take 'Name' and the 'Description' into account and try to make some corrections to those ages. 

# ## Breed EDA

# In[ ]:


print("Missing Breed1 in train set: %d" % (pd.isna(train['Breed1']).sum() + len(train[train.Breed1 == 0])))
print("Missing Breed1 in test set: %d" %  (pd.isna(test['Breed1']).sum() + len(test[test.Breed1 == 0])))

print("Missing Breed2 in train set: %d" %  (pd.isna(train['Breed2']).sum() + len(train[train.Breed2 == 0])))
print("Missing Breed2 in test set: %d" %  (pd.isna(test['Breed2']).sum() + len(test[test.Breed2 == 0])))


# Lets include the breed names an see an example in our data.

# In[ ]:


breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}

train['Breed1_name'] = train['Breed1'].apply(lambda x: breeds_dict[x] if x in breeds_dict else 'Unknown')
train['Breed2_name'] = train['Breed2'].apply(lambda x: breeds_dict[x] if x in breeds_dict else '')

test['Breed1_name'] = test['Breed1'].apply(lambda x: breeds_dict[x] if x in breeds_dict else 'Unknown')
test['Breed2_name'] = test['Breed2'].apply(lambda x: breeds_dict[x] if x in breeds_dict else '')

train[['Breed1_name', 'Breed2_name']].sample(10)


# To have a quit overview about the breeds, we use a wordcloud to visualize the top breeds. 

# In[ ]:


fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(train.loc[train['Type'] == 'Cat', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(train.loc[train['Type'] == 'Dog', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(train.loc[train['Type'] == 'Cat', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed2')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(train.loc[train['Type'] == 'Dog', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2')
plt.axis("off")
plt.show()


# There are a lot of "mixed breeds" pets in our data set. Sometimes the "Breed2" is the same as "Breed1" but sometimes not. A new feature "mixed_breed" is indicating, if a pet has two different breeds.

# In[ ]:


def mixed_breed(row):
    if row['Breed1'] == 307:
        return 1
    elif row['Breed2'] == 0:
        return 0 
    elif row['Breed2'] != row['Breed1']:
        return 1
    else:
        return 0

train['mixed_breed'] = train.apply(mixed_breed, axis=1)
test['mixed_breed'] = test.apply(mixed_breed, axis=1)


# In[ ]:


make_count_plot(df=train, x='mixed_breed', title='by mixed_breed')


# In[ ]:


show_adaptionspeed_barplot(train, 'mixed_breed', ['No','Yes'], 'AdoptionSpeed by mixed_breed')


# Mixed breeds tend to be adopter slower then pure breeds. 
# Lets see the values in the test set.

# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='mixed_breed', palette=color2);
plt.title('Mixed breeds in train and test data');


# Again there is a small difference in the train and test set. But this could be due to the fact, that there are more Cats in the test set, and cats are tends to be more pure breeds.

# ## Gender EDA
# 1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets

# In[ ]:


plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='Gender', title='by gender')

plt.subplot(1,2,2)
show_adaptionspeed_barplot(train, 'Gender', ['Male','Female', 'Group'], 'AdoptionSpeed by gender')


# Males tend to be adapted faster than females. Groups of pets need obiously more time to get adapted, as all of the group have to be adapted before it is reflected in the data.

# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Gender', palette=color3);
plt.title('Number of pets by gender in train and test data');


# The distribution of gender is very similar in the test and train set.

# ## Color EDA
# 
# A pet can have up to 3 colors. 

# In[ ]:


colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')


# In[ ]:


print("Missing Color1 in train set: %d" % (pd.isna(train['Color1']).sum() + len(train[train.Color1 == 0])))
print("Missing Color1 in test set: %d" %  (pd.isna(test['Color1']).sum() + len(test[test.Color1 == 0])))

print("Missing Color2 in train set: %d" %  (pd.isna(train['Color2']).sum() + len(train[train.Color2 == 0])))
print("Missing Color2 in test set: %d" %  (pd.isna(test['Color2']).sum() + len(test[test.Color2 == 0])))

print("Missing Color3 in train set: %d" %  (pd.isna(train['Color3']).sum() + len(train[train.Color3 == 0])))
print("Missing Color3 in test set: %d" %  (pd.isna(test['Color3']).sum() + len(test[test.Color3 == 0])))


# All pets have at least one color assigned. 

# In[ ]:


plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='Color1', title='by main color')

plt.subplot(1,2,2)
show_adaptionspeed_barplot(train, 'Color1', ['Black','Brown','Golden','Yellow','Cream','Gray','White'], 'AdoptionSpeed by main color')


# Most of the animals are black or brown. The adaption rate seems to differ a bit between the colors. 
# The distribution of main colors is the same on the train and test set.

# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Color1', palette=color7);
plt.title('Number of pets by gender in train and test data');


# Lets see if having more than one color affects the adaption speed.

# In[ ]:


def number_of_colors(row):
    if row['Color1'] == 0:
        return 0
    elif (row['Color2'] != 0 and row['Color3'] == 0):
        return 2
    elif (row['Color2'] != 0 and row['Color3'] != 0):
        return 3
    else:
        return 1

train['number_of_colors'] = train.apply(number_of_colors, axis=1)
test['number_of_colors'] = test.apply(number_of_colors, axis=1)


# In[ ]:


plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='number_of_colors', title='by number of colors')

plt.subplot(1,2,2)
show_adaptionspeed_barplot(train, 'number_of_colors', ['One','Two', 'Three'], 'AdoptionSpeed by number of colors')


# Pets with one color seem to have less chance to get adopted. Having three colors gives a boost on day one. 

# ## MaturitySize EDA
# 
# Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)

# In[ ]:


print("Missing MaturitySize in train set: %d" % (pd.isna(train['MaturitySize']).sum() + len(train[train.MaturitySize == 0])))
print("Missing MaturitySize in test set: %d" %  (pd.isna(test['MaturitySize']).sum() + len(test[test.MaturitySize == 0])))
print("Unique values of MaturitySize in train set: %s" %  train.MaturitySize.unique())


# Interessting, there is always a size specified, the value "Not Specified" is not available in the train nor test set. "Extra Large" is almost non existant, thus we combine it with "Large".

# In[ ]:


train.MaturitySize.replace([4],[3], inplace=True)


# In[ ]:


plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='MaturitySize', title='by MaturitySize')

plt.subplot(1,2,2)
show_adaptionspeed_barplot(train, 'MaturitySize', ['Small','Medium','Large','Extra Large'], 'AdoptionSpeed by MaturitySize')


# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='MaturitySize', palette=color4);
plt.title('Number of pets by MaturitySize in train and test data');


# ## FurLength EDA
# (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)

# In[ ]:


print("Missing FurLength in train set: %d" % (pd.isna(train['FurLength']).sum() + len(train[train.FurLength == 0])))
print("Missing FurLength in test set: %d" %  (pd.isna(test['FurLength']).sum() + len(test[test.FurLength == 0])))


# In[ ]:


plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='FurLength', title='by FurLength')

plt.subplot(1,2,2)
show_adaptionspeed_barplot(train, 'FurLength', ['Short','Medium','Long'], 'AdoptionSpeed by FurLength')


# All pets have a specified fur length. While having long fur increases the chances to get adopted quite a lot, short fur means the opposite. But there are not many pets in this category, thus the conclusion could be wrong as well. 

# In[ ]:


sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='FurLength', palette=color3);
plt.title('Number of pets by FurLength in train and test data');


# We have seen, that there are breeds containing also a fur length. Lets see if those two values match. 

# In[ ]:


wrong_hair_length = []
all_data = pd.concat([train, test])
for i, row in all_data[(all_data.Breed1_name.str.contains('Hair')) | (all_data.Breed2_name.str.contains("Hair"))].iterrows():
    if ('Short' in row['Breed1_name'] or 'Short' in row['Breed2_name']) and row['FurLength'] == 1:
        continue
    if ('Medium' in row['Breed1_name'] or 'Medium' in row['Breed2_name']) and row['FurLength'] == 2:
        continue
    if ('Long' in row['Breed1_name'] or 'Long' in row['Breed2_name']) and row['FurLength'] == 3:
        continue
    wrong_hair_length.append((row['PetID'], row['Breed1_name'], row['Breed2_name'], row['FurLength'], row['dataset_type']))

wrong_df = pd.DataFrame(wrong_hair_length)
print(f"There are {len(wrong_df[wrong_df[4] == 'train'])} pets whose breed and fur length don't match in train")
print(f"There are {len(wrong_df[wrong_df[4] == 'test'])} pets whose breed and fur length don't match in test")
wrong_df.sample(8)


# There are pets with non matching breed and fur length. We can try to take the more specialized value from both (long instead of medium) or the more general value (medium) in those cases and see how it affects the model.

# ## Health EDA
# 
# There are 4 variables which refer to the health of the pets:
# 
# * Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# * Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# * Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# * Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)

# In[ ]:


print("Missing Vaccinated in train set: %d" % pd.isna(train['Vaccinated']).sum())
print("Missing Vaccinated in test set: %d" %  pd.isna(test['Vaccinated']).sum())
      
print("Missing Dewormed in train set: %d" % pd.isna(train['Dewormed']).sum())
print("Missing Dewormed in test set: %d" %  pd.isna(test['Dewormed']).sum())
      
print("Missing Sterilized in train set: %d" % pd.isna(train['Sterilized']).sum())
print("Missing Sterilized in test set: %d" %  pd.isna(test['Sterilized']).sum())
      
print("Missing Health in train set: %d" % pd.isna(train['Health']).sum())
print("Missing Health in test set: %d" %  pd.isna(test['Health']).sum())


# In[ ]:


plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Vaccinated', title='by Vaccinated')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'Vaccinated', ['Yes','No','Not sure'], 'AdoptionSpeed by Vaccinated')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Vaccinated', palette=color3);
plt.title('Number of pets by Vaccinated in train and test data');


# In[ ]:


plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Dewormed', title='by Dewormed')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'Dewormed', ['Yes','No','Not sure'], 'AdoptionSpeed by Dewormed')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Dewormed', palette=color3);
plt.title('Number of pets by Dewormed in train and test data');


# In[ ]:


plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Sterilized', title='by Sterilized')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'Sterilized', ['Yes','No','Not sure'], 'AdoptionSpeed by Sterilized')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Sterilized', palette=color3);
plt.title('Number of pets by Sterilized in train and test data');


# In[ ]:


print("Healthy in train set: %d" % (len(train[train.Health == 1])))
print("Minor Injury in test set: %d" %  (len(test[test.Health == 2])))
print("Serious Injury in test set: %d" %  (len(test[test.Health == 3])))
print("Not Specified in test set: %d" %  (len(test[test.Health == 0])))

plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Health', title='by Health')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'Health',['Healthy','Minor Injury','Serious Injury'], 'AdoptionSpeed by Health')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Health', palette=color3);
plt.title('Number of pets by Health in train and test data');


# Interessingly, pets are adopted faster if they are not vaccinated, dewormed or sterilized.
# Most of the pets are healthy, very few with a minor injury and just a handfull with serious injuries. 

# ## Quantity EDA
# 
# As we have seen during the Age analysis, there are also group of pets out for adoption. 

# In[ ]:


train['Quantity'].value_counts()


# In[ ]:


test['Quantity'].value_counts()


# Most of the pets are just a single pet but there are groups of up to 18 pets. 

# In[ ]:


train['is_group'] = train['Quantity'].apply(lambda x: True if x > 1 else False)
test['is_group'] = test['Quantity'].apply(lambda x: True if x > 1 else False)


# In[ ]:


plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='is_group', title='by is_group')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'is_group',['No','Yes'], 'AdoptionSpeed by is_group')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='is_group', palette=color2);
plt.title('Number of pets by is_group in train and test data');


# As excpeted, groups are less likey to be adopted. This makes sense, as the speed is defined by the the adoption of the latest pet in a group. 

# ## Fee EDA
# 
# Some animals are available for a fee, whilst most of them are for free.

# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.title('Distribution of pets fee')
train['Fee'].plot('hist', label='train',colormap=ListedColormap(color2[0]))
test['Fee'].plot('hist', label='test',colormap=ListedColormap(color2[1]))
plt.legend();

plt.subplot(1, 2, 2)
plt.title('Distribution of pets fee (log)')
np.log1p(train['Fee']).plot('hist', label='train', colormap=ListedColormap(color2[0]))
np.log1p(test['Fee']).plot('hist', label='test', colormap=ListedColormap(color2[1]))
plt.legend();


# In[ ]:


train['is_free'] = train['Fee'].apply(lambda x: True if x == 0 else False)
test['is_free'] = test['Fee'].apply(lambda x: True if x == 0 else False)


# In[ ]:


plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='is_free', title='by is_free')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'is_free',label='AdoptionSpeed by is_free')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='is_free', palette=color2);
plt.title('Number of pets by is_free in train and test data');


# Free pets have a better chance to get adapted. Although it would be interessting to see what prices are paid. Maybe there is a limit what people are willing to pay. 

# ## State EDA

# In[ ]:


states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['state_name'] = train['State'].apply(lambda x: states_dict[x] if x in states_dict else 'Unknown')
test['state_name'] = test['State'].apply(lambda x: states_dict[x] if x in states_dict else 'Unknown')


# In[ ]:


fig= plt.subplots(figsize=(18,8))
ax = sns.countplot(x="state_name", data=train, order = train["state_name"].value_counts().index,palette=color7)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='grey', ha='center', va='bottom')


# Selangor and Kuala Lumpur are the states with most adoptions. 

# ## Rescuer EDA
# 
# The rescuer identifed by a hash code. 

# In[ ]:


print("Number of pets by top rescuer in train set: %d" % train['RescuerID'].value_counts().head(1))
print("Number of pets by top rescuer in test set: %d" %  test['RescuerID'].value_counts().head(1))
print("Unique rescuer in train set: %d" %  len(train['RescuerID'].unique()))
print("Unique rescuer in test set: %d" %  len(test['RescuerID'].unique()))
print("Rescuer from train also in test set: %d" %  train.RescuerID.isin(test.RescuerID).sum())
print("Rescuer from test also in train set: %d" %  test.RescuerID.isin(train.RescuerID).sum())


# There some really busy rescuers around. What is interesting, the rescuers are completely distinct between the two datasets. 
# If we want the use the rescuer feature, we need to prepare it for modeling. 

# ## VideoAmt EDA

# In[ ]:


train['VideoAmt'].value_counts()


# There are a view videos around. More than one videos exists, but those are very rare. We can check if having a video at all has some impact. 

# In[ ]:


train['has_video'] = train['VideoAmt'].apply(lambda x: True if x > 0 else False)
test['has_video'] = test['VideoAmt'].apply(lambda x: True if x > 0 else False)

plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='has_video', title='by has_video')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'has_video',label='AdoptionSpeed by has_video')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='has_video', palette=color2);
plt.title('Number of pets by has_video in train and test data');


# Having a video helps, but there are just very few cases where there are videos. Lets check the number of photos.

# ## PhotoAmt EDA

# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.title('Distribution of PhotoAmt')
train['PhotoAmt'].plot('hist', label='train',colormap=ListedColormap(color2[0]))
test['PhotoAmt'].plot('hist', label='test',colormap=ListedColormap(color2[1]))
plt.legend();


# In[ ]:


train['has_photo'] = train['PhotoAmt'].apply(lambda x: True if x > 0 else False)
test['has_photo'] = test['PhotoAmt'].apply(lambda x: True if x > 0 else False)

plt.figure(figsize=(24, 16));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='has_photo', title='by has_photo')

plt.subplot(2,2,2)
show_adaptionspeed_barplot(train, 'has_photo',label='AdoptionSpeed by has_photo')

plt.figure(figsize=(24, 16));
plt.subplot(2,2,3)
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='has_photo', palette=color2);
plt.title('Number of pets by has_photo in train and test data');


# Almost all of the pets have at least one photo. Having none is extremly bad for the pet. 

# ## Description EDA

# In[ ]:




