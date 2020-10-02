#!/usr/bin/env python
# coding: utf-8

# ### ** in work**
# 

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(12, 10)})

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')
y = train_df['AdoptionSpeed']
train_df['dataset_type'] = 'train'
test_df['dataset_type'] = 'test'
train_id = train_df['PetID']
test_id = test_df['PetID']
all_data0 = pd.concat([train_df, test_df])
all_data = all_data0.drop(columns = 'AdoptionSpeed')
train_df.head()


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


display(train_df.describe())
display(train_df.info())


# ### Data Fields  
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

# ### AdoptionSpeed  
# Contestants are required to predict this value. The value is determined by how quickly, if at all, a pet is adopted. The values are determined in the following way:   
# 0 - Pet was adopted on the same day as it was listed.   
# 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.   
# 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.   
# 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.  
# 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

# In[ ]:


train_df['AdoptionSpeed'].value_counts().sort_index(ascending=False).plot(kind='barh', 
                                                                          figsize=(15,6))
plt.title('Adoption Speed (Target)', fontsize=18)


# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.countplot(x='Type', data=train_df).set(xticklabels=['Dog', 'Cat'])
plt.title('Type(train)', fontsize=18)

plt.subplot(122)
sns.countplot(x='Type', data=test_df).set(xticklabels=['Dog', 'Cat'])
plt.title('Type(test)', fontsize=18)


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(x='Type', data=train_df, hue="AdoptionSpeed").set(xticklabels=['Dog', 'Cat'])
plt.title('Type/AdoptionSpeed(train)', fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(x='Gender', data=train_df, hue="AdoptionSpeed")
plt.title('Gender/AdoptionSpeed(train)', fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


train_df['Mixed_Breed'] = train_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
train_df['Num_Color'] = train_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
train_df['Description'].fillna("", inplace=True)
train_df['Description_Length'] = train_df.Description.map(len)


# In[ ]:


test_df['Mixed_Breed'] = test_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
test_df['Num_Color'] = test_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
test_df['Description'].fillna("", inplace=True)
test_df['Description_Length'] = test_df.Description.map(len)


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(x='Mixed_Breed', data=train_df, hue="AdoptionSpeed")
plt.title('Mixed_Breed/AdoptionSpeed(train)', fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(x='Num_Color', data=train_df, hue="AdoptionSpeed")
plt.title('Num_Color/AdoptionSpeed(train)', fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.distplot(train_df['Age'], fit=norm)

plt.subplot(122)
res = stats.probplot(train_df['Age'], plot=plt)

display(train_df['Age'].skew())


# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.distplot((train_df['PhotoAmt']), fit=norm)

plt.subplot(122)
res = stats.probplot((train_df['PhotoAmt']), plot=plt)

display((train_df['PhotoAmt']).skew())


# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(121)
sns.distplot((train_df['Description_Length']), fit=norm)

plt.subplot(122)
res = stats.probplot((train_df['Description_Length']), plot=plt)

display((train_df['Description_Length']).skew())


# In[ ]:


import json
#getting description sentiment analyses
    
def get_desc_anly(type, recalc):
    if recalc == 1:
        if type == "train":
            path = "../input/train_sentiment/"#../input/train_sentiment/
        elif type == "test":
            path = "../input/test_sentiment/"#../input/test_sentiment/
        print("Getting description sentiment analysis for",type+"_sentiment.csv")
        files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

        df = pd.DataFrame(columns=["PetID", "DescScore", "DescMagnitude"])
        i = 0
        for f in files:
            #print(path + f)
            with open(path+f, encoding="utf8") as json_data:
                data = json.load(json_data)
            
            df.loc[i]= [f[:-5],data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"]]
            i = i+1
        df.to_csv(type+"_sentiment.csv", index=False)
    elif recalc == 0:
        df = pd.read_csv(type+"_sentiment.csv")
    return df

    
train_snt = get_desc_anly("train", 1)
test_snt = get_desc_anly("test", 1)
    
train_df = train_df.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
test_df = test_df.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()

train_df["DescScore"].fillna(0, inplace=True)
train_df["DescMagnitude"].fillna(0, inplace=True)

train_df["Name"].fillna("none", inplace=True)
test_df["DescScore"].fillna(0, inplace=True)
test_df["DescMagnitude"].fillna(0, inplace=True)

test_df["Name"].fillna("", inplace=True)

train_df["NameLength"] = train_df["Name"].str.len()
test_df["NameLength"] = test_df["Name"].str.len()


# In[ ]:


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
train_df.loc[:, 'vertex_x'] = vertex_xs
train_df.loc[:, 'vertex_y'] = vertex_ys
train_df.loc[:, 'bounding_confidence'] = bounding_confidences
train_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
train_df.loc[:, 'dominant_blue'] = dominant_blues
train_df.loc[:, 'dominant_green'] = dominant_greens
train_df.loc[:, 'dominant_red'] = dominant_reds
train_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train_df.loc[:, 'dominant_score'] = dominant_scores
train_df.loc[:, 'label_description'] = label_descriptions
train_df.loc[:, 'label_score'] = label_scores


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
test_df.loc[:, 'vertex_x'] = vertex_xs
test_df.loc[:, 'vertex_y'] = vertex_ys
test_df.loc[:, 'bounding_confidence'] = bounding_confidences
test_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
test_df.loc[:, 'dominant_blue'] = dominant_blues
test_df.loc[:, 'dominant_green'] = dominant_greens
test_df.loc[:, 'dominant_red'] = dominant_reds
test_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
test_df.loc[:, 'dominant_score'] = dominant_scores
test_df.loc[:, 'label_description'] = label_descriptions
test_df.loc[:, 'label_score'] = label_scores


# In[ ]:


train_df_num = train_df.drop(columns = train_df.dtypes[train_df.dtypes=='object'].index)


# In[ ]:


test_df_num = test_df.drop(columns = test_df.dtypes[test_df.dtypes=='object'].index)


# In[ ]:


train_df_num.head()


# In[ ]:


train_df_num.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


# In[ ]:


x = train_df_num.drop(columns = 'AdoptionSpeed')
y = train_df_num.AdoptionSpeed


# In[ ]:


x_tst = test_df_num


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=True)


# In[ ]:


X_train.shape, X_test.shape, x_tst.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,                            ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
RANDOM_STATE = 0


# In[ ]:


# some classifiers for test
classifiers = [KNeighborsClassifier(n_jobs=-1),
               
               LogisticRegression(),
               LogisticRegression(C=0.005, penalty='l1', class_weight=None, fit_intercept=False, max_iter=100, tol=0.01),
               LogisticRegression(C=0.0001, penalty='l2', class_weight=None, fit_intercept=True, max_iter=150, tol=1e-06),
               LogisticRegression(C=96, penalty='l1', class_weight='balanced', fit_intercept=True, max_iter=50, tol=0.0001),
               LogisticRegression(C=54, penalty='l2', class_weight='balanced', fit_intercept=False, max_iter=450, tol=0.001),
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, max_depth=12, max_features=0.8, min_samples_leaf=2, subsample=0.2),
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=5, max_features=0.6, min_samples_leaf=10, subsample=0.8),
               GradientBoostingClassifier(n_estimators=200, learning_rate=0.001, max_depth=90, max_features=0.5, min_samples_leaf=20, subsample=0.2),
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=20, max_features=0.6, min_samples_leaf=24, subsample=0.7),
               GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=90, max_features=0.5, min_samples_leaf=20, subsample=0.2),
               RandomForestClassifier(n_jobs=-1),
               RandomForestClassifier(n_estimators=225, bootstrap=True, max_depth=83, max_features='auto', min_samples_leaf=5, min_samples_split=5,  n_jobs=-1),
               RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=12, min_samples_leaf=1, oob_score=True, n_jobs=-1),
               ExtraTreesClassifier(n_jobs=-1),
               ExtraTreesClassifier(n_estimators=200, bootstrap=False, max_depth=80, max_features='auto', min_samples_leaf=6, min_samples_split=5, n_jobs=-1),
               AdaBoostClassifier(),
               AdaBoostClassifier(n_estimators=225, algorithm='SAMME.R', learning_rate=0.2),
               DecisionTreeClassifier(),
               DecisionTreeClassifier(criterion='entropy', max_depth=110, max_features='auto', min_samples_leaf=6, min_samples_split=330)
               ]
classifiers_names = ['knn1',
                     'lr1', 'lr2', 'lr3', 'lr4', 'lr5',
                     'gb1', 'gb2', 'gb3', 'gb4', 'gb5', 
                     'rf1', 'rf2', 'rf3',
                     'et1', 'et2', 
                     'adb1', 'adb2',
                     'dt1','dt2',
                    ]


# In[ ]:


classifiers_predictions = pd.DataFrame()
for name, classifier in zip(classifiers_names, classifiers):
    classifier.fit(X_train, y_train)
    train_predictions = pd.Series(classifier.predict(X_train))
    test_predictions = classifier.predict(X_test)
    
    classifiers_predictions[name] = test_predictions
    print('{0}: ({1} - {2})'.format(name,
                                    cohen_kappa_score(y_train, train_predictions, weights='quadratic') ,
                                    cohen_kappa_score(y_test, test_predictions, weights='quadratic')))


# In[ ]:


classifiers_predictions = classifiers_predictions[['knn1', 
                     'lr1', 'lr2', 'lr3', 'lr4', 'lr5',
                     'gb1', 'gb2', 'gb3', 'gb4', 'gb5', 
                     'rf1', 'rf2', 'rf3',
                     'et1', 'et2', 
                     'adb1', 'adb2',
                     'dt1','dt2',
                    ]]


# In[ ]:


sns.heatmap(classifiers_predictions.corr(), linewidths=.5);
plt.yticks(rotation=0);
plt.xticks(rotation=30);
sns.set(font_scale=2)


# In[ ]:


# some classifiers for test
classifiers = [
               LogisticRegression(C=96, penalty='l1', class_weight='balanced', fit_intercept=True, max_iter=50, tol=0.0001),
               
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, max_depth=12, max_features=0.8, min_samples_leaf=2, subsample=0.2),
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=5, max_features=0.6, min_samples_leaf=10, subsample=0.8),
               GradientBoostingClassifier(n_estimators=200, learning_rate=0.001, max_depth=90, max_features=0.5, min_samples_leaf=20, subsample=0.2),
               GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=20, max_features=0.6, min_samples_leaf=24, subsample=0.7),
               RandomForestClassifier(n_estimators=225, bootstrap=True, max_depth=83, max_features='auto', min_samples_leaf=5, min_samples_split=5,  n_jobs=-1),
               RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=12, min_samples_leaf=1, oob_score=True, n_jobs=-1),
               ExtraTreesClassifier(n_estimators=200, bootstrap=False, max_depth=80, max_features='auto', min_samples_leaf=6, min_samples_split=5, n_jobs=-1),
               AdaBoostClassifier(),
               AdaBoostClassifier(n_estimators=225, algorithm='SAMME.R', learning_rate=0.2),
               DecisionTreeClassifier(criterion='entropy', max_depth=110, max_features='auto', min_samples_leaf=6, min_samples_split=330)
               ]
classifiers_names = ['lr4', 
                     'gb1','gb2', 'gb3', 'gb4', 
                     'rf2', 'rf3',
                     'et2',  
                     'adb1','adb2',
                     'dt2',
                    ]


# In[ ]:


def simple_blending(basic_algorithms, meta_algorithm, X_train, X_test, y_train, test_df, part1_ratio=0.9, random_state=None):
    tr = pd.DataFrame()
    tst = pd.DataFrame()
    y = pd.DataFrame()
    X_train_part1, X_train_part2,    y_train_part1, y_train_part2 = train_test_split(X_train, y_train, test_size=1-part1_ratio, random_state=random_state)
    
    for index, basic_algorithm in enumerate(basic_algorithms):
        #print(index)
        basic_algorithm.fit(X_train_part1, y_train_part1)

        part2_predictions = basic_algorithm.predict(X_train_part2)
        tr[index] = part2_predictions

        test_predictions = basic_algorithm.predict(X_test)
        tst[index] = test_predictions
        
        test_pred = basic_algorithm.predict(test_df)
        y[index] = test_pred
        
    meta_algorithm.fit(tr, y_train_part2)
    
    return meta_algorithm.predict(tst), meta_algorithm.predict(y)


# In[ ]:


r = pd.DataFrame()
experiments = list()
for i in range(1, 10):
    simple_blending_predictions, result = simple_blending(classifiers,
                                              LogisticRegression(C=5, random_state=RANDOM_STATE),
                                              X_train, X_test, y_train, x_tst,
                                              part1_ratio=0.9,
                                              random_state=i)
    r[i] = result
    #print(simple_blending_predictions)
    print(cohen_kappa_score(y_test, simple_blending_predictions, weights='quadratic'))
    experiments.append(cohen_kappa_score( y_test, simple_blending_predictions, weights='quadratic'))
print('mean kappa: {0}\nstd: {1}'.format(round(np.mean(experiments), 4), round(np.std(experiments), 5)))


# In[ ]:


r['avg'] = r.mean(axis=1).round(0).astype(int)
r.avg.head()


# In[ ]:


def simple_blending_features(basic_algorithms, meta_algorithm, X_train, X_test, y_train, test_df, part1_ratio=0.5, random_state=None):
    tr = pd.DataFrame()
    tst = pd.DataFrame()
    y = pd.DataFrame()
    
    X_train_part1, X_train_part2,    y_train_part1, y_train_part2 = train_test_split(X_train, y_train, test_size=1-part1_ratio, random_state=random_state)
    
    tr = tr.append(X_train_part2)
    tst = tst.append(X_test)
    y = y.append(test_df)
    
    for index, basic_algorithm in enumerate(basic_algorithms):
        #print(index)
        basic_algorithm.fit(X_train_part1, y_train_part1)

        part2_predictions = basic_algorithm.predict(X_train_part2)
        tr[index] = part2_predictions

        test_predictions = basic_algorithm.predict(X_test)
        tst[index] = test_predictions
        
        test_pred = basic_algorithm.predict(test_df)
        y[index] = test_pred
    
    meta_algorithm.fit(tr, y_train_part2)

    return meta_algorithm.predict(tst), meta_algorithm.predict(y)


# In[ ]:


r2 = pd.DataFrame()
experiments = list()
for i in range(1, 10):
    simple_blending_features_predictions, result2 = simple_blending_features(classifiers,
                                              LogisticRegression(C=5, random_state=RANDOM_STATE),
                                              X_train, X_test, y_train, x_tst,
                                              part1_ratio=0.9,
                                              random_state=i)
    r2[i] = result2
    #print(simple_blending_predictions)
    print(cohen_kappa_score(y_test, simple_blending_features_predictions, weights='quadratic'))
    experiments.append(cohen_kappa_score( y_test, simple_blending_features_predictions, weights='quadratic'))
print('mean kappa: {0}\nstd: {1}'.format(round(np.mean(experiments), 4), round(np.std(experiments), 5)))


# In[ ]:


r2['avg'] = r2.mean(axis=1).round(0).astype(int)
r2.avg.head()


# In[ ]:


from pandas.util.testing import assert_series_equal
assert_series_equal(r['avg'], r2['avg'])


# In[ ]:


submission = pd.DataFrame(data={'PetID' : test_df['PetID'], 'AdoptionSpeed' : r['avg']})
submission.to_csv('submission.csv', index=False)


# In[ ]:


# check submission
submission.head(5)


# In[ ]:


# Plot 1
plt.figure(figsize=(15,4))
plt.subplot(211)
train_df['AdoptionSpeed'].value_counts().sort_index(ascending=False).plot(kind='barh')
plt.title('Target Variable distribution in training set')

# Plot 2
plt.subplot(212)
submission['AdoptionSpeed'].value_counts().sort_index(ascending=False).plot(kind='barh')
plt.title('Target Variable distribution in predictions')

plt.subplots_adjust(top=2)


# In[ ]:




