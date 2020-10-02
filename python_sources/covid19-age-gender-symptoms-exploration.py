#!/usr/bin/env python
# coding: utf-8

# The objective of this notebook is to explore the data by plotting word clouds, correlation matrixes and interpretable tree views through features as age, gender and symptoms of COVID-19 and the risk of death of the infected person based on records where the person lives or dies as "low_risk" and "high_risk" labels.
# 
# There is a word cloud for the patient summary without some popular words of COVID-19 context such as "Wuhan" or "China" in a way to allow a view not to the origin of this pandemic but focusing on the patient condition.
# 
# There are some word clouds for the patient symptoms with adjustment to match some words variations (see stemming).
# 
# There are some correlation matrixes before each plotted tree. The trees parameters and the data was adjusted in a way not to get the highest accuracy but to create simple, small and interpretable trees.
# 
# One-hot-encoding was applied to the distinct words in symptoms column so each word column was given the prefix "has_" and value 1 if the symptoms text contains the word, otherwise 0. Some words was adjusted to match its varitions like the column "has_musc" for "muscular" or "muscle" words occurency (see stemming).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Read datasets and lowercase data

# In[ ]:


main_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
columns = list(main_df.columns)
columns.sort()
pd.DataFrame(columns)


# In[ ]:


main_df.ID.count()


# In[ ]:


for i in columns:
    print(i)
    try:
        main_df[i] = main_df[i].str.lower()
    except:
        pass
    #print(main_df[i].unique())
    #print('#'*15)


# In[ ]:


main_df_ = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
columns = list(main_df_.columns)
columns.sort()
pd.DataFrame(columns)


# In[ ]:


main_df_.id.count()


# In[ ]:


for i in columns:
    print(i)
    try:
        main_df_[i] = main_df_[i].str.lower()
    except:
        pass
    #print(main_df[i].unique())
    #print('#'*15)


# # Clean some columns from both datasets and match them

# ## COVID19_open_line_list.csv
# ## Clear 'age' column
# 
# Remove rows without numerical age data. Replace null ages with age column mean.

# In[ ]:


main_df.age.unique()


# In[ ]:


filtered_df = main_df[main_df.age.str.contains('(-|\.|[a-z])', regex=True) != True]
filtered_df.age.unique()


# In[ ]:


filtered_df['age'].isnull().count()


# In[ ]:


filtered_df['age'] = pd.to_numeric(filtered_df['age'])
filtered_df['age'][filtered_df['age'].isnull()] = filtered_df['age'].mean()

filtered_df.age.describe()


# ## Clear 'outcome' column
# 
# Replace outcomes with 'died' and 'death' for 'high_risk', then 'discharged', 'discharge', 'stable' and 'recovered' for 'low_risk'.
# 
# Set 'unknown' where its null.

# In[ ]:


filtered_df['outcome'].value_counts()


# ## Translate death to 'high_risk' value

# In[ ]:


filtered_df = filtered_df[filtered_df['outcome'] != 'critical condition, intubated as of 14.02.2020']
filtered_df = filtered_df[filtered_df['outcome'] != '05.02.2020']

filtered_df['outcome'][filtered_df['outcome'] == 'died'] = 'high_risk'
filtered_df['outcome'][filtered_df['outcome'] == 'death'] = 'high_risk'
filtered_df['outcome'][filtered_df['outcome'] == 'discharged'] = 'low_risk'
filtered_df['outcome'][filtered_df['outcome'] == 'discharge'] = 'low_risk'

filtered_df['outcome'][filtered_df['outcome'] == 'stable'] = 'low_risk'
filtered_df['outcome'][filtered_df['outcome'] == 'recovered'] = 'low_risk'

filtered_df['outcome'][filtered_df['outcome'].isnull()] = 'unknown'

filtered_df['outcome'].value_counts()


# ## Clear 'sex' column
# 
# Set 'unknown' where data is null.

# In[ ]:


filtered_df['sex'].value_counts(), filtered_df['sex'].isnull().value_counts()


# In[ ]:



filtered_df['sex'][filtered_df['sex'].isnull()] = 'unknown'
filtered_df['sex'].value_counts()


# ## COVID19_line_list_data.csv
# ## Clear 'age' column

# In[ ]:


main_df_.age.unique()


# In[ ]:


filtered_df_ = main_df_.copy()
filtered_df_.age


# In[ ]:


filtered_df_['id'][filtered_df_['age'].isnull()].count()


# In[ ]:


filtered_df_['age'][filtered_df_['age'].isnull()] = filtered_df_['age'].mean()

filtered_df_.age.describe()


# ## Make 'outcome' column

# In[ ]:


filtered_df_['death'].value_counts()


# ## Translate death to 'high_risk' value

# In[ ]:


filtered_df_['death'][filtered_df_['death']!='0'] = 'high_risk'
filtered_df_['death'][filtered_df_['death']!='high_risk'] = 'low_risk'
filtered_df_['death'].value_counts()

filtered_df_['outcome'] = filtered_df_['death']

filtered_df_['outcome'][filtered_df_['outcome'].isnull()] = 'unknown'


# ## Clear 'gender' column rename to 'sex'

# In[ ]:


filtered_df_['gender'].value_counts(), filtered_df_['gender'].isnull().value_counts()


# In[ ]:


filtered_df_['gender'][filtered_df_['gender'].isnull()] = 'unknown'
filtered_df_['gender'].value_counts()


# In[ ]:


filtered_df_['sex'] = filtered_df_['gender']
#filtered_df_.drop('gender', axis=1, inplace=True)
filtered_df_.head(5)


# In[ ]:


filtered_df_['symptoms'] = filtered_df_['symptom']
#filtered_df_.drop('symptom', axis=1, inplace=True)
filtered_df_


# # Concatenate datasets

# In[ ]:


result = pd.concat([filtered_df,filtered_df_])
result


# In[ ]:


filtered_df = result


# # Word clouds

# # What are the frequent words in the 'summary' column about the patient condition?

# In[ ]:


summ_str = str(result['summary'].unique())
summ_str = summ_str.replace("'",'')
summ_str = summ_str.replace('nan ','')
summ_str = summ_str.replace('\n',',')
summ_str = summ_str.replace('[',',')
summ_str = summ_str.replace(']',',')
summ_str = summ_str.replace(', ',',')
summ_str = summ_str.replace(';','')
summ_str = summ_str.replace(',',' ')
summ_str = summ_str.replace('/',' ')
summ_str = summ_str.replace('-',' ')
summ_str = summ_str.replace('.',' ')
summ_str = summ_str.replace('(',' ')
summ_str = summ_str.replace(')',' ')
for i in range(10):
    summ_str = summ_str.replace(str(i),' ')
summ_str = summ_str.replace('ing ',' ')
summ_str = summ_str.replace('ness ',' ')
summ_str = summ_str.replace(' new ',' ')
summ_str = summ_str.replace(' confirmed ',' ')
summ_str = summ_str.replace(' covid ',' ')
summ_str = summ_str.replace(' onset',' ')
summ_str = summ_str.replace(' went',' ')
summ_str = summ_str.replace(' imported',' ')
summ_str = summ_str.replace(' symptom',' ')
summ_str = summ_str.replace(' singapore',' ')
summ_str = summ_str.replace(' korea',' ')
summ_str = summ_str.replace(' hong',' ')
summ_str = summ_str.replace(' kong',' ')
summ_str = summ_str.replace(' japan',' ')
summ_str = summ_str.replace(' south',' ')
summ_str = summ_str.replace(' wuhan',' ')
summ_str = summ_str.replace(' hokkaido',' ')
summ_str = summ_str.replace(' arrived',' ')
summ_str = summ_str.replace(' returned',' ')
summ_str = summ_str.replace(' visited',' ')
summ_str = summ_str.replace(' male',' ')
summ_str = summ_str.replace(' female',' ')
summ_str = summ_str.replace(' ncid',' ')
summ_str = summ_str.replace(' germany',' ')
summ_str = summ_str.replace(' france',' ')
summ_str = summ_str.replace(' taiwan',' ')
summ_str = summ_str.replace(' thailand',' ')
summ_str = summ_str.replace(' spain',' ')
summ_str = summ_str.replace(' china',' ')
summ_str = summ_str.replace(' italy',' ')
summ_str = summ_str.replace(' tianjin',' ')
summ_str = summ_str.replace(' yunnan',' ')
summ_str = summ_str.replace(' shaanxi',' ')
summ_str = summ_str.replace(' nagoya',' ')
summ_str = summ_str.replace(' malaysia',' ')
summ_str = summ_str.replace('patient patient',' ')
summ_str = summ_str.replace('s ',' ')
summ_set = summ_str.split(' ')
summ_set = summ_set[:]
summ_set = set(summ_set)

wordcloud = WordCloud(max_words=300, background_color="white").generate(summ_str)
plt.figure(figsize=(13,7))
plt.imshow(wordcloud, interpolation='spline16')
plt.axis("off")
plt.show()


# # What are the frequent words in the 'symptoms' column?

# In[ ]:


symp_str = str(result['symptoms'].unique())
symp_str = symp_str.replace("'",'')
symp_str = symp_str.replace('nan ','')
symp_str = symp_str.replace('\n',',')
symp_str = symp_str.replace('[',',')
symp_str = symp_str.replace(']',',')
symp_str = symp_str.replace(', ',',')
symp_str = symp_str.replace(';','')
symp_str = symp_str.replace(',',' ')
symp_str = symp_str.replace('ing ',' ')
symp_str = symp_str.replace('ness ',' ')
symp_str = symp_str.replace('s ',' ')
symp_str = symp_str.replace(' los ',' ')
symp_str = symp_str.replace(' feaver ',' ')
symp_str = symp_str.replace(' feve\\\\ ',' ')
symp_str = symp_str.replace(' in ',' ')
symp_str = symp_str.replace(' of ',' ')
symp_str = symp_str.replace(' with ',' ')
symp_str = symp_str.replace(' ye ',' ')
symp_set = symp_str.split(' ')
symp_set = symp_set[:]
symp_set = set(symp_set)

wordcloud = WordCloud(max_words=300, background_color="white").generate(symp_str)
plt.figure(figsize=(13,7))
plt.imshow(wordcloud, interpolation='spline16')
plt.axis("off")
plt.show()


# In[ ]:


symp_set


# ### Some cleaning and stemming

# In[ ]:


discard_list = ['',
 'a',
 'abdominal',
 'ach',
 'and',
 'body',
 'breath',
 'chest',
 'diarrhea',
 'diarrheoa',
 'diarrhoea',
 'dizzi',
 'esophageal',
 'eventually',
 'eye',
 'feel',
 'feversore',
 'flu-like',
 'heart',
 'high',
 'joint',
 'lack',
 'lesion',
 'limb',
 'mild',
 'mouth',
 'left',
 'muscle',
 'muscular',
 'nasal',
 'nose',
 'no',
 'of',
 'on',
 'other',
 'pharyngalgia',
 'pharyngeal',
 'pharyngiti',
 'pharynx',
 'rhinorrhea',
 'rhinorrhoea',
 'short',
 'showed',
 'similar',
 'sneeze',
 'tight',
 'throatfatiguevomit',
 'to',
 'ye']
for i in discard_list:
    symp_set.discard(i)
symp_set


# In[ ]:


add_list =['chill','diarrh','dizz','esophag','flu','musc','pharyn','rhinorrh']
for i in add_list:
    symp_set.add(i)
symp_set


# # What are the frequent words in the'symptoms' column based on the values of the 'outcome' column?

# In[ ]:


for i in ['low_risk','high_risk','unknown']:
    print('OUTCOME: '+i)
    temp_df = filtered_df[filtered_df['outcome']==i]
    print(temp_df.outcome.value_counts())
    symp_str = str(temp_df['symptoms'].unique())
    symp_str = symp_str.replace("'",'')
    symp_str = symp_str.replace(',',' ')
    symp_str = symp_str.replace('.',' ')
    symp_str = symp_str.replace('nan',' ')
    wordcloud = WordCloud(max_words=300, background_color="white").generate(symp_str)
    plt.figure(figsize=(13,7))
    plt.imshow(wordcloud, interpolation='spline16')
    plt.axis("off")
    plt.show()


# # Create new dataset only with age, sex and symptoms columns

# In[ ]:


featured_df = pd.DataFrame(filtered_df['age'])

featured_df['sex'] = filtered_df['sex']
featured_df = pd.concat([pd.get_dummies(featured_df['sex'], prefix='gender'), 
                        featured_df['age']
                      ], axis=1)

filtered_df['symptoms'][filtered_df['symptoms'].isnull()] = 'None'
featured_df['symptoms'] = filtered_df['symptoms']
for i in symp_set:
    #print(i)
    featured_df['has_'+i] = 0
    featured_df['has_'+i][featured_df['symptoms'].str.contains(i)] = 1
featured_df['has_asymptomatic'][featured_df['symptoms'].isnull()] = 1
featured_df.drop('symptoms',axis=1,inplace=True)

featured_df['outcome'] = filtered_df['outcome']

#featured_df = featured_df.dropna()
featured_columns = list(featured_df.columns)
featured_df


# In[ ]:


featured_df.to_csv('COVID19-age-gender-symptoms-outcome.csv')


# # Symptoms dataset with age, gender and known outcome information

# In[ ]:


df_concat = featured_df[featured_df['outcome'] != 'unknown']
df_concat


# ### List data columns

# In[ ]:


model_features = list(df_concat.columns)
model_features


# # Is the dataset balanced?

# In[ ]:


df_concat.outcome.value_counts().plot(kind="bar")


# In[ ]:


df_concat.outcome.value_counts()


# Unbalanced data spotted

# # What are the correlations between gender, age, symptoms and the outcome information?

# In[ ]:


pearsoncorr = pd.concat([df_concat, 
            pd.get_dummies(df_concat['outcome'], prefix='outcome')
          ], axis=1).corr()

#fig, ax = plt.subplots(figsize=(15*5,7*5))    

#sns.heatmap(pearsoncorr, annot=True, ax=ax)


# In[ ]:


m = pearsoncorr.min().isnull()
m


# There are a lot of unused columns as we are dealing only with data with known outcome.

# Remove unused columns

# In[ ]:


for i in zip(list(m.axes[0]),list(m)):
    if i[1]:
        print(i)
        df_concat.drop(i[0], axis=1, inplace=True)


# In[ ]:


df_concat


# # Plot correlations

# In[ ]:


pearsoncorr = pd.concat([df_concat, 
            pd.get_dummies(df_concat['outcome'], prefix='outcome')
          ], axis=1).corr()

fig, ax = plt.subplots(figsize=(15*5,7*5))    

sns.heatmap(pearsoncorr, annot=True, ax=ax)


# # How would a tree model be able to fit to the outcome through age, gender and symptoms in an interpretable way?

# In[ ]:


model_features = list(df_concat.columns)
X_inputs = df_concat[model_features[:-1]]
X_inputs


# LabelEncoder classes will be used ahead.

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df_concat[model_features[-1:]])
Y_outputs = le.transform(df_concat[model_features[-1:]])

Y_outputs = pd.get_dummies(df_concat[model_features[-1:]], prefix='outcome')
Y_outputs


# In[ ]:


Y_outputs.shape, Y_outputs.sum()#,np.array(Y_outputs)


# ## Rebalance dataset

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_inputs, Y_outputs = ros.fit_resample(X_inputs, np.array(Y_outputs))


# In[ ]:


pd.DataFrame(Y_outputs)


# In[ ]:


#Y_outputs = np.array([i.argmax() for i in Y_outputs])
#Y_outputs
#for i in Y_outputs:
#    print(i)


# # Decision Tree - age, gender, symptoms -> outcome

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
rs = 0
mins_s = 2
max_depth = 24

model = tree.DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf=mins_s, random_state=rs,
                                        criterion= 'entropy',splitter='best')

# Train
model.fit(X_inputs, Y_outputs)
model.score(X_inputs, Y_outputs)
        


# In[ ]:


model.get_depth()


# In[ ]:


model


# In[ ]:


classes = list(le.classes_)
classes


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree_age_sex_symptoms.dot', feature_names = list(X_inputs.columns),
                class_names = classes,
                rounded = True, proportion = False, precision = 2, filled = True)


# In[ ]:


get_ipython().system('dot -Tpng tree_age_sex_symptoms.dot -o tree_age_sex_symptoms.png -Gdpi=128')


# # Tree visualization trhough age, gender and symptoms columns to the outcome

# In[ ]:


from IPython.display import Image
Image(filename = 'tree_age_sex_symptoms.png')


# # How would it be if age and gender is droped out but the outcome 'unknown' is added?

# In[ ]:


df_concat = featured_df.copy()
df_concat.drop(['gender_female',
 'gender_male',
 'gender_unknown',
 'age'], axis=1, inplace=True)
df_concat


# # How would be the new correlations?

# In[ ]:


pearsoncorr = pd.concat([df_concat, 
            pd.get_dummies(df_concat['outcome'], prefix='outcome')
          ], axis=1).corr()

fig, ax = plt.subplots(figsize=(15*5,7*5))    

sns.heatmap(pearsoncorr, annot=True, ax=ax)


# In[ ]:


model_features = list(df_concat.columns)
model_features


# # How would a tree model fit the outcome through the symptoms in an interpretable way?

# In[ ]:


X_inputs = df_concat[model_features[:-1]]
X_inputs


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df_concat[model_features[-1:]])
Y_outputs = le.transform(df_concat[model_features[-1:]])

Y_outputs = pd.get_dummies(df_concat[model_features[-1:]], prefix='outcome')
Y_outputs


# In[ ]:


Y_outputs.shape, Y_outputs.sum(),np.array(Y_outputs)


# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_inputs, Y_outputs = ros.fit_resample(X_inputs, np.array(Y_outputs))


# In[ ]:


pd.DataFrame(Y_outputs)


# In[ ]:


Y_outputs = np.array([i.argmax() for i in Y_outputs])
Y_outputs


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
rs = 0
mins_s = 125
max_depth = 24
max_acc = 0

model = tree.DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf=mins_s, random_state=rs,
                                        criterion= 'entropy',splitter='best')

# Train
model.fit(X_inputs, Y_outputs)
model.score(X_inputs, Y_outputs)


# In[ ]:


model.get_depth()


# In[ ]:


classes = list(le.classes_)
classes


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree_symptoms_risk.dot', feature_names = list(X_inputs.columns),
                class_names = classes,
                rounded = True, proportion = False, precision = 2, filled = True)


# In[ ]:


get_ipython().system('dot -Tpng tree_symptoms_risk.dot -o tree_symptoms_risk.png -Gdpi=128')


# # Tree visualization through symptoms columns to the outcome

# In[ ]:


from IPython.display import Image
Image(filename = 'tree_symptoms_risk.png')


# In[ ]:



filtered_df[['age','sex','symptoms','outcome']][filtered_df['symptoms'].str.contains('cold')]


# # So what are the frequent words in symptoms when it contains 'fever', 'cough', 'sore', 'pneumonia', 'myalgia' and 'cold'  values mixed in it?

# In[ ]:


symptoms = ['fever', 'cough', 'sore', 'pneumonia', 'myalgia', 'cold']

for word in symptoms:
    print('#'*30)
    print('WORD:'+word)
    temp_df = filtered_df[filtered_df['symptoms'].str.contains(word)]
    print(temp_df.outcome.value_counts())
    try:
        print('Ages histogram')
        temp_df.age.plot.hist()
    except:
        pass
    for i in ['sex','symptoms','outcome']:
        print('#'*15)
        print('WORD:'+word)
        print('COLUMN:'+i)
        symp_str = str(temp_df[i].unique())
        symp_str = symp_str.replace("'",'')
        symp_str = symp_str.replace(',',' ')
        symp_str = symp_str.replace('.',' ')
        #print(symp_str)
        try:
            wordcloud = WordCloud(max_words=300, background_color="white").generate(symp_str)
            plt.figure(figsize=(13,7))
            plt.imshow(wordcloud, interpolation='spline16')
            plt.axis("off")
            plt.show()
        except:
            pass
    for word_ in symptoms:
        if word_ != word:
            print('#'*30)
            print('WORDS:'+word+' AND '+word_)
            temp_df_ = temp_df[temp_df['symptoms'].str.contains(word_)]
            print('Outcome counts')
            print(temp_df.outcome.value_counts())
            try:
                print('Age histogram')
                temp_df_.age.plot.hist()
            except:
                pass
            for i in ['sex','symptoms','outcome']:
                print('WORDS:'+word+' AND '+word_)
                print('COLUMN:'+i)
                print(i)
                symp_str = str(temp_df_[i].unique())
                symp_str = symp_str.replace("'",'')
                symp_str = symp_str.replace(',',' ')
                symp_str = symp_str.replace('.',' ')
                #print(symp_str)
                try:
                    wordcloud = WordCloud(max_words=300, background_color="white").generate(symp_str)
                    plt.figure(figsize=(13,7))
                    plt.imshow(wordcloud, interpolation='spline16')
                    plt.axis("off")
                    plt.show()
                except:
                    pass


# In[ ]:




