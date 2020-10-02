#!/usr/bin/env python
# coding: utf-8

# # Petfinder
# 
# If you are not familiar on what this competition is about you can read the description here:
# 
# https://www.kaggle.com/c/petfinder-adoption-prediction#description
# 
# This kernel is just for my practice and I have took some ideas from other kernel. I have tried to cite them though.
# 
# For example some of the EDA stuff is inspired by this great EDA kernel https://www.kaggle.com/artgor/exploration-of-data-step-by-step.
# 
# Feel free to comment on anything related to this kernel as I'm still really new to data science and would appreciate any comments that could help me improve my skills.
# 
# 
# **Note this is still work in progress**

# ## Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
seed = 93


# In[ ]:


# Basic
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import scipy
import os
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from pandas_summary import DataFrameSummary
from IPython.display import display

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


PATH = '../input/'


# In[ ]:


get_ipython().system('ls {PATH}')


# ## Reading in the data

# In[ ]:


train_raw = pd.read_csv(f'{PATH}/train/train.csv', low_memory=False)
test_raw = pd.read_csv(f'{PATH}/test/test.csv', low_memory=False)


# ## Data overview
# 
# Let's try to build a basic intuition about the data first.
# 
# ### Response variable (AdoptionSpeed)
# - 0 - Pet was adopted the same day as it was listed
# - 1 - Pet was adopted between days 1-7 after being listed (1st week).
# - 2 - Pet was adopted between 8 and 30 days (1st month).
# - 3 - Pet was adopted between 31 and 90 days.
# - 4 - No adoption after 100 days
# 
# 
# ### Independent variables:
# - PetID - Unique hash ID of pet profile
# - AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# - Type - Type of animal (1 = Dog, 2 = Cat)
# - Name - Name of pet (Empty if not named)
# - Age - Age of pet when listed, in months
# - Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# - Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# - Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# - Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# - Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# - Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# - MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# - FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# - Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# - Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# - Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# - Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# - Quantity - Number of pets represented in profile
# - Fee - Adoption fee (0 = Free)
# - State - State location in Malaysia (Refer to StateLabels dictionary)
# - RescuerID - Unique hash ID of rescuer
# - VideoAmt - Total uploaded videos for this pet
# - PhotoAmt - Total uploaded photos for this pet
# - Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.
# 
# So the task is simple. Predict the Adoptionspeed of an animal based on these features.
# 
# 
# Let's then look at the dataframes to build some intuition about the data

# ### Size of datasets

# In[ ]:


train_raw.shape, test_raw.shape


# We can see that the dataset is quite small. We should think about using cross validation, if there's no time series in the data.

# In[ ]:


train_raw.head(3)


# In[ ]:


test_raw.head(3)


# ### Summary statistics

# In[ ]:


train_raw.describe().T


# #### Notes:
#     
#     -  Maximum age is 255, Probably people have filled out some random number as an age, when they do not know the real age.
#     
#     - There's a lot of categorical features with some number as encoding, we should find out what each these numbers means, when we examine each feature more closely.

# In[ ]:


test_raw.describe().T


# Test set shows the same relationship with the age variable.

# ## Data cleaning
# ### Null values
# Let's check which columns contain null values.

# In[ ]:


train_raw.isnull().sum(axis=0)


# Here we can see that Name and Description fields contain null values. These null values could affect the adoptionspeeds in some way. We should explore this more in the visualization part.

# In[ ]:


train_raw.isnull().sum(axis=1).sort_values(ascending=False).head(10)


# Here we see that there are no rows that contain two NaN values.

# ### Removing constant features
# If a feature is constant among all rows, it contains no useful information for our models, so we should just remove it.
# 
# It's convenient to do all the feature engineering to both train and test set at the same time, so we create one big dataframe here. We also create a new feature to each of the dataframes to indicate whether it was from the test or train set. This makes it easier to plot stuff.

# In[ ]:


train_raw['from_dataset'] = 'train'
test_raw['from_dataset'] = 'test'
alldata = pd.concat([train_raw, test_raw], axis = 0)


# In[ ]:


feats_counts = alldata.nunique(dropna = False)


# In[ ]:


feats_counts.sort_values()[:10]


# We found no constant features, therefore we cannot remove any columns here.

# ### Removing duplicated features
# It is pretty clear from just looking at our dataframe that we do not have any duplicated features, therefore we can skip this step here. If our dataframe was larger it would be good to check, since having two features with exactly the same values just wastes computation.
# 
# Final thing to do is to fill all the NaN values with some field so our models can use them and we can find them easier.

# In[ ]:


alldata.fillna('NaN', inplace=True)


# ## EDA
# Now that we have built some intuition about the data, it's time to find some more insights using various EDA techniques such as plotting.
# 
# Let's remind ourselves what the features are.

# In[ ]:


alldata.head(5).T


# ### AdoptionSpeed (Our response variable)
# 
# Let's recap what AdoptionSpeed meant:
# - 0 - Pet was adopted the same day as it was listed
# - 1 - Pet was adopted between days 1-7 after being listed (1st week).
# - 2 - Pet was adopted between 8 and 30 days (1st month).
# - 3 - Pet was adopted between 31 and 90 days.
# - 4 - No adoption after 100 days
# 

# In[ ]:


train_raw['AdoptionSpeed'].value_counts().sort_index().plot('bar');


# Here we can see that 0 is the class that has significantly less observations. This makes sense, since 0 means that the pet was adopted on the same day it was listed and this is quite unlikely by applying commonsense.
# 
# Also we can see that the most frequent class is 4, which means that the pet was not adopted in 100 days. Finding out what features drive this class is the most important thing for the analysis, since then we  can maybe save those pets.

# ## Type
# According to the kaggle page Type means:
# 
#     - 1 --> dog
#     - 2 --> cat

# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(14,6))
ax = sns.countplot(x="from_dataset", data=alldata, hue='Type')
plt.title('Number cats and dogs in train and test sets');


# From this picture we can see that the train set contains more dogs and the test set contains more cats.
# 
# Since our dataset is fairly small this is likely due to random chance occured while sampling the data.

# ### Age
# We previously found out that age contains some unintuitive values.
# 
# Let's investigate this feature further

# In[ ]:


plt.figure(figsize=(14,6))
plt.ylabel('Age')
plt.plot(alldata['Age'], '.');


# By plotting the age feature against its index we can see that the data is shuffled correctly. We can also see many points with unintuitive age values.
# 
# After some googling I found that oldest dog that ever lived was ~30 years old [https://en.wikipedia.org/wiki/List_of_oldest_dogs] and the oldest ever cat was ~38 years old [https://en.wikipedia.org/wiki/List_of_oldest_cats], therefore there's many values that are just impossible.
# 
# Let's see what age values are the most common ones.

# In[ ]:


alldata['Age'].value_counts().head(20)


# This table reveals us something else that is strange. Age of 12, 24, 36, 48, 60, 72 are high in the listing. This does not make intuitive sense, so it could be some behaviour that could help our model. We could create a feature Age modulo 12 to capture this behaviour to make it easier for tree based models to split on this feature if it has some significance.

# In[ ]:


alldata['age_mod_12'] = alldata['Age'].apply(lambda x: True if (x%12)==0 else False)


# ## Name
# Let's see what names are the most frequent to find some patterns.

# In[ ]:


# dogs
dogs = alldata.loc[alldata['Type'] == 1]
# cats
cats = alldata.loc[alldata['Type'] == 2]


# In[ ]:


dogs['Name'].value_counts().head(20)


# In[ ]:


cats['Name'].value_counts().head(20)


# We can see that in both of the cases NaN is the most common name. This could be important. 
# 
# We also see some common names like Kitty, No Name, Kittens appear quite frequently.
# 
# We could make a new feature isNaNname to help tree based models to make a split on this phenomena.
# 
# Let's next check if there are names with just 1 or 2 characters long. These names are likely meaningless could contain some information about the adoptionspeeds.

# In[ ]:


alldata['NaN_name'] = alldata['Name'].apply(lambda x: True if str(x) == 'NaN' else False)


# In[ ]:


alldata['NaN_name'].value_counts()


# In[ ]:


alldata[alldata['Name'].apply(lambda x: len(str(x))) < 3]['Name'].unique()


# We can see that our hypothesis was true and we can see bunch on names that are nonsense.
# 
# Now let's create a new feature based on this info.

# In[ ]:


alldata['name_len_one_or_two'] = alldata['Name'].apply(lambda x: True if len(str(x)) < 3 else False)


# In[ ]:


alldata['name_len_one_or_two'].value_counts()


# Let's test this same hypothesis for names that are the length of 3.

# In[ ]:


alldata[alldata['Name'].apply(lambda x: len(str(x))) == 3]['Name'].unique()


# There's quite a few non sense names in here also, but also some real names. Let's keep this in mind, but not do any feature engineering yet.
# 
# One useful feature could be just length of the name, then the models could split on name_len < 4 for example.

# ### Fee
# The adoption fee could be interesting feature, since money always drives human behaviour.

# In[ ]:


plt.figure(figsize=(14,6))
plt.ylabel('Fee')
plt.plot(alldata['Fee'], '.');


# Here we can see that most pets have no fees. Also there some abnormally high fees e.g 2000, 3000.
# 
# **TODO**: Examine this more closely

# ## RescuerID
# RescuerID is another interesting feature. My hypothesis is that there are some recuers, who constantly rescue animals and therefore use more effort in the process --> Faster adoptionSpeed

# In[ ]:


train_raw['RescuerID'].value_counts().head(15)


# In[ ]:


test_raw['RescuerID'].value_counts().head(15)


# It is odd that the test set does not seem to contain any of the same ID:s. This tells us something about how the organizers have split the data. It is not a random split, but might be somehow related to the people who have rescued the animals.
# 
# This also means that we most likely should remove the RescuerID in the modelling part.
# 
# 
# We can still use the amount of animals a person has rescued as a feature. It would make sense that it matters in the prediction in some way.

# In[ ]:


top_20_rescuers = list(train_raw['RescuerID'].value_counts()[:20].index)
top_20_data = train_raw.loc[train_raw['RescuerID'].isin(top_20_rescuers)]


# In[ ]:


plt.figure(figsize=(10,4))
top_20_data['AdoptionSpeed'].value_counts().sort_index().plot('bar');
plt.title('AdoptionSpeed of the top20 rescuers');


# In[ ]:


plt.figure(figsize=(10,4))
train_raw['AdoptionSpeed'].value_counts().sort_index().plot('bar');
plt.title('Adoptionspeed in the whole training sample');


# It looks like that the top rescuers pets are less likely to be in category 4. This supports my hypothesis that these frequent rescuers do extra work in order for the pet to get adopted.
# 
# Also it seems that on average top rescuers animals take longer time to get adopted than in the whole sample. This could be due to the fact that people who rescue a lot of animals probably rescue animal that are more sick than people who just give their dog away because of some life situation.
# 
# 
# **TODO**: Continue EDA with more features based on the feedback from the Random forest feature importance.

# ## Feature engineering
# 
# ### Basic features
# Let's generate three basic features that came to mind:
#     1. Description length
#     2. Name length
#     3. How many rescued animals a specific rescuerID has

# In[ ]:


def desc_len_feature(df):
    descs = np.stack([item for item in df.Description])
    desc_len = [len(item) for item in descs]
    # Add the features to the dataframe
    df['desc_length'] = desc_len
    
def rescue_count_feature(df):
    rescuers_df = pd.DataFrame(df.RescuerID)
    rescuer_counts = rescuers_df.apply(pd.value_counts)
    rescuer_counts.columns = ['rescue_count']
    rescuer_counts['RescuerID'] = rescuer_counts.index
    df = df.merge(rescuer_counts, how='left', on='RescuerID')
    return df

def name_len_feature(df):
    names = np.stack([item for item in df.Name])
    name_len = [len(item) for item in names]
    df['name_length'] = name_len
    return df


# In[ ]:


# We have the convert the IDs into categories in order to create the features.
alldata['RescuerID'] = alldata.RescuerID.astype('category')
alldata['RescuerID'] = alldata.RescuerID.cat.codes
desc_len_feature(alldata)
name_len_feature(alldata)
alldata = rescue_count_feature(alldata)


# ### Natural language API stuff
# The organizers have kindly provided us with some scores on the descriptions, by running them through the Google NLP API. Let's use these scores as features as well.

# In[ ]:


# Kudos to https://www.kaggle.com/artgor/exploration-of-data-step-by-step
def parse_sentiment_files(datatype):
    sentiments = {}
    for filename in os.listdir('../input/' + datatype + '_sentiment'):
        with open('../input/' + datatype + '_sentiment/' +  filename, 'r') as f:
            sentiment = json.load(f)
            pet_id = filename.split('.')[0]
            sentiments[pet_id] = {}
            sentiments[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
            sentiments[pet_id]['score'] = sentiment['documentSentiment']['score']
            sentiments[pet_id]['language'] = sentiment['language']
            
    return sentiments

def sentiment_features(df, sentiments):
    df['lang'] = df['PetID'].apply(lambda x: sentiments[x]['language']
                                 if x in sentiments else 'no')
    df['magnitude'] = df['PetID'].apply(lambda x: sentiments[x]['magnitude']
                                       if x in sentiments else 0)
    df['score'] = df['PetID'].apply(lambda x: sentiments[x]['magnitude']
                                   if x in sentiments else 0)
    return df


# In[ ]:


train_sentiment = parse_sentiment_files('train')
test_sentiment = parse_sentiment_files('test')
sentiment_features(alldata, train_sentiment);
sentiment_features(alldata, test_sentiment);


# In[ ]:


alldata.head()


# ### Frequency encodings
# In frequency encoding we utilize the frequencies of categories. It can help tree based models deal with high cardinality categorical features more easily, if the frequency of breed has some correlation with the target variable.

# In[ ]:


cols = ['Breed1']
for col in cols:
    frequencies = dict(alldata[col].value_counts()/alldata[col].shape[0])
    alldata[col + '_frequency'] = alldata[col].apply(lambda x: frequencies[x])


# ### Bag of words features
# As we can see the dataframe also contains text descriptions of the pets. This probably has at least some relationship with the target variable. We need to turn the description into some meaningful features that our algorithms can utilize. First we use bag of words techniques. These techniques rely on counting how many times words appear in documents.
# 
# We will use a variation of this technique called term frequency inverse document frequency (**TFIDF**) to generate some features from the descriptions.
# 
# This technique relies on counting the frequncies of words, but also solves the problem of frequent but uninformative words like "the" by taking the logarithm of the frequency.
# 
# Let's look at an example to understand this.

# In[ ]:


headlines = ["President trump won the election", "The world was shocked",
              "Barcelona won the champions league"]


# Let's imagine we have these three news headlines and want to generate tfidf features out of them.
# 
# 
# First thing to do is count the frequencies of the words.

# In[ ]:


vectorizer = CountVectorizer(analyzer='word')
X = vectorizer.fit_transform(headlines)
columns = [x for x in vectorizer.get_feature_names()]
pd.DataFrame(X.todense(), columns=columns)


# Here we can see that CountVectorizer simply counts how many times words appear in a corpus. For example word "the" appears in all of the documents.
# 
# As we can imagine word "the" has no relevance in prediciting the adoption speed, so we want to scale it down. This is done by taking the inverse document freuquency transformation. 
# 
# Basically we take the logarithm of the frequency, so very common words get lower values. 

# In[ ]:


count_matrix = pd.DataFrame(X.todense(), columns=columns)
tfidf = TfidfTransformer()
inverse_frequencies = tfidf.fit_transform(count_matrix)
pd.DataFrame(inverse_frequencies.todense(), columns=columns)


# **NOTE** sklearn implementation does some more complicated stuff to prevent division by zero etc. But the basic idea stays the same. We can see word "the" getting lower values, because it appears in all of the documents.

# In[ ]:


def tfidf_features(corpus):
    tfv = TfidfVectorizer(min_df=2,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,)
    tfv.fit(list(corpus))
    X = tfv.transform(corpus)
    return X
    
def svd_features(df, freq_matrix, n_comps=1):
    svd = TruncatedSVD(n_components=n_comps) #Choose 20 most relevant ones.
    svd.fit(freq_matrix)
    freq_matrix = svd.transform(freq_matrix)
    freq_matrix = pd.DataFrame(freq_matrix, columns=['svd_{}'.format(i) for i in range(n_comps)])
    df = pd.concat((df, freq_matrix), axis=1)
    return df


# In[ ]:


X = tfidf_features(alldata['Description']); X


# Here we can see one of the problems with this approach even with moderately small dataset the word frequency matrix becomes huge. To combat this issue let's take SVD of this matrix and use only the most relevant components.
# 
# After some hyperparameter tuning it seems that only using the first component helps in the prediction.
# 
# Should investigate what could be the reason for this since it seems odd, maybe random forest cannot utilize this information very well..
# 
# We should also try word2vec approach, since it can capture relationships between words.

# In[ ]:


alldata=svd_features(alldata, X)


# In[ ]:


alldata.head()


# ## Building a baseline RF model
# 
# Randomforest is algorithm based on the concept of bagging (bootstrap aggregating). In random forest you create multiple decision trees by randomly sampling the rows and columns of the data. This way you create multiple trees that contain random errors. The key word is random, since if your errors are truly random the expected value of random errors is 0. Then when you combine these trees you find the true relationship. If the errors are not random the model won't work well.
# 

# In[ ]:


# Store PetID for later
train_pet_ids = train_raw.PetID
test_pet_ids = test_raw.PetID


# In[ ]:


alldata = alldata.drop(['Description', 'PetID', 'Name', 'lang', 'RescuerID'], axis=1)


# In[ ]:


# Split the feature engineered dataframe back into test and train sets.
train = alldata.loc[alldata['from_dataset'] == 'train']
test = alldata.loc[alldata['from_dataset'] == 'test']


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['AdoptionSpeed'])
y = pd.DataFrame(y, columns=['AdoptionSpeed'])


# In[ ]:


train = train.drop(['from_dataset', 'AdoptionSpeed'], axis=1)
test = test.drop(['from_dataset', 'AdoptionSpeed'], axis=1);


# In[ ]:


m = RandomForestClassifier(n_estimators=500, random_state=seed,
                           max_features='sqrt',
                           min_samples_leaf=25, n_jobs=-1);
m.fit(train, y);


# ### Feature importance
# It's good to look at Random Forest feature importance plot to get some idea of what features are important. This has the issue that when there's collinearity the feature importance can be skewed and the importances are split between the colinear features, but it still gives us some idea on what features might be important.

# In[ ]:


def rf_feature_importance(df, m):
    df = pd.DataFrame({'cols' : df.columns, 'imp' : m.feature_importances_}).sort_values('imp', ascending=False)
    return df


# In[ ]:


fi = rf_feature_importance(train, m)
fi[:25].plot('cols', 'imp', kind='barh', legend=False);


# ### Notes:
#     - Rescue_count seems to be very good feature, so our hypothesis was correct, that who rescued the animal matters in the adoptionspeed.
#     
#     - Age is the second most important feature. Im interested to see how the split went. Let's try tree interpreter to do that.
#     
#     - Age%12 feature had some importance, mayber there's something going on there.
#  
#     - name_len_one_or_two and NaN_name did not any impact on the model. This could be due to collinearity with the name_length feature. Or maybe this fact does not matter in the prediction
#     
#     - Description based features are very high in the feature importance. We should really investigate word2vec approach in order to generate more advanced features from the descriptions.

# ### Removing features
# Let's remove all features with importance less than half a percent as having those just wastes computation.

# In[ ]:


train.shape


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)
train = train[to_keep].copy()
test = test[to_keep].copy()
train.shape


# In[ ]:


train.head(3)


# ## Building the final model
# Let's use k-fold cross validation, so we get all information out of our limited dataset.

# In[ ]:


m = RandomForestClassifier(n_estimators=500, random_state=seed,
                           max_features='sqrt',
                           min_samples_leaf=1, n_jobs=-1)

test_preds = np.zeros(test.shape[0])
results=[]
n_folds = 4
cv = StratifiedKFold(n_splits=n_folds, random_state=seed)
for (train_idx, valid_idx) in cv.split(train,y):
    m.fit(train.iloc[train_idx], y.iloc[train_idx])
    score = metrics.cohen_kappa_score(y.loc[valid_idx], m.predict(train.iloc[valid_idx]), weights='quadratic')
    results.append(score)
    y_test = m.predict(test)
    test_preds += y_test.reshape(-1)/n_folds
    
mean = np.mean(results)
std = np.std(results)


# In[ ]:


print(f'Mean kappa score: {mean} with std: {std}')


# In[ ]:


fi = rf_feature_importance(train, m)
fi[:25].plot('cols', 'imp', kind='barh', legend=False);


# ## Submission

# In[ ]:


df_sub = pd.DataFrame({'PetID' : test_pet_ids})
df_sub['AdoptionSpeed'] = test_preds
df_sub['AdoptionSpeed'] = df_sub['AdoptionSpeed'].astype(int)
df_sub.head()


# In[ ]:


df_sub.to_csv(f'submission.csv', index=False)

