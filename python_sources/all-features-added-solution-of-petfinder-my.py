#!/usr/bin/env python
# coding: utf-8

# This Kernel is created to explain everything in competition [PetFinder.my](https://www.kaggle.com/c/petfinder-adoption-prediction), it encourage everyone to develop algorithms to predict the adoptability of pets, guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.
# 
# Meanwhile, I will add my ideas and explanations in between, to tell the reader while I perform such action in such step.
# 
# Finally, I will mention my future working direction in the last chapter, hopefully it can give you some inspiration
# 
# The kernel is still updating. Your upvotes and folks will be my best motivation.
# 

# # Maintainance Log
# 
# | Time       | version | remark                                                                |  Score  | commit version |
# |------------|---------|-----------------------------------------------------------------------|
# | 2019-01-20 | v1.0.0    | basic models only using features from train.csv, combiner model used  | **0.343**  | - |
# | 2019-01-20 | v1.0.1   | do balancing on class 0  |  0.339 | -|
# | 2019-01-21 | v1.1.0   | add length of description as a feature to test  |  0.338 | -|
# | 2019-01-21 | v1.1.1   | add regularization for xgb |  0.343 |  v12 |
#     | 2019-01-24 | v2.0.0 | add features from descritption, refactored the code  |  0.337 | v13|
#      | 2019-01-24 | v2.0.1| cancel balancing |  0.329 | v15|
#       | 2019-01-28 | v2.1.0| use tf-idf to extract feature from description text |   | v16|
#     | TBD | v3.0.0 | add features from images  |  - | -|
# 

# # Table of Content:
# * [Introduction](#introduction)
#     * [Input](#input)
#     * [Ranking Criteria](#ranking)
#     * [Output](#output)
# * [Exploratory Data Analysis](#eda)
#     * [Data loading](#loading)
#     * [Main Data Exploration](#mainEDA)
#         * [AdoptionSpeed](#adoptionspeed)
#         * [Type](#type)
#         * [Age](#age)
#         * [Gender](#gender)
#         * [State](#state)
#         * [Photo amount and video amount distribution](#amount)
# * [Modelling](#model0)
#     * [Model Selection](#selection)
#     * [Model from original train dataset](#model1)
#     * [Model with description sentiments](#model2)
#     * [Model with images features and above](#model3)
# * [Feature Importance and Conclusion](#importance)
# * [Result and Submission](#result)

# # Some Flags may be used to control process

# In[ ]:


BALANCING = False
MODEL_USE = 3
# 0 is run all model(it takes quite long)
# 1 is Model from original train dataset,
# 2 is Model from original train dataset and description,
# 3 Model with images features and above, yet to be implemented


# # Introduction <a class="anchor" id="Introduction"></a>

# ## Input <a class="anchor" id="input"></a>
# First, Let's look into what data they have provided

# In[ ]:


import os
print(os.listdir("../input"))


# here are the information provided by the offical, I assumed that you have already read the [data introduction](https://www.kaggle.com/c/petfinder-adoption-prediction/data).  I listed them down here for short.
# ### File descriptions 
# * train.csv - Tabular/text data for the training set
# * test.csv - Tabular/text data for the test set
# * sample_submission.csv - A sample submission file in the correct format
# * breed_labels.csv - Contains Type, and BreedName for each BreedID. Type 1 is dog, 2 is cat.
# * color_labels.csv - Contains ColorName for each ColorID
# * state_labels.csv - Contains StateName for each StateID
# * Images - pets' photos
# * Image Metadata - analysis on Face Annotation, Label Annotation, Text Annotation and Image Properties. 
# * Sentiment Data - profile's description  analysis on sentiment and key entities. 
# 
# detailed analysis of the data will be done at [next chapter](#eda).

# ## Ranking Criteria <a class="anchor" id="ranking"></a>
# As Shown in [evalution tab](https://www.kaggle.com/c/petfinder-adoption-prediction#evaluation),  the result will be scored based on the quadratic weighted kappa and highest score ranking higher. The implementation is as below:

# In[ ]:


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# ## Output <a class="anchor" id="output"></a>
# Output will be submission.csv, should include these 2 columns: 
# 
# *PetID, AdoptionSpeed*
# 

# # Exploratory Data Analysis <a class="anchor" id="eda"></a>
# 
# From this chapter you can get some deeper insight about the data. From the data introduction we can know that the data is actually very easy to combine and transform. Train and Test is the main data,  breeds, colors, states files are only id to name mapping files, they don't contain extra information. The extra information is on description and images of pets, but PetFinder.my had already convert them into sentiment data, which will be easy to integrate with the main data file. Let's dive deep down to the main data.
# 
# ## Data loading<a class="anchor" id="loading"></a>
# First, Let's load Tabular data first
# 

# In[ ]:


import pandas as pd
import numpy as np

breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')
states = pd.read_csv('../input/state_labels.csv')

# train['dataset_type'] = 'train'
# test['dataset_type'] = 'test'


# ## Main Data Exploration<a class="anchor" id="mainEDA"></a>

# In[ ]:


train.head(10)


# In[ ]:


train.info()


# ### AdoptionSpeed <a class="anchor" id="adoptionspeed"></a>
# First of all, let's see target distribution, it is always the first thing to look into when you get data, the purposes are:
# * Get to know the amount of the each classes
# * Identify the skewness of the classes, do balancing on training data if necessary

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


train['AdoptionSpeed'].value_counts()


# In[ ]:


train['AdoptionSpeed'].value_counts().sort_index().plot('barh')
plt.title('Adoption speed classes comparison');


# we can see that class 0 has much fewer amount than the others. In another word, only a small percentage of pets were adopted within 7 days(That's the reason why they held this the competition). **Balancing technique need to be taken when modelling**
# 
# ### Type <a class="anchor" id="type"></a>
# 
# What's the second column you want to inspect? In my opinion, it should be 'Type'
# 

# In[ ]:


train['Type'].value_counts().sort_index().plot('barh')


# where (1 = Dog, 2 = Cat), as explained in the [Data Fields explanation](https://www.kaggle.com/c/petfinder-adoption-prediction/data). We can see that they have  similar amount
# 
# 

# ### Age <a class="anchor" id="age"></a>
# Age may be an important factor on the adoptablity of the pet, let's dive deep into it
# 

# In[ ]:


train['Age'].describe()


# 75% are under 12 months. But for better understanding, let's cap at 60 months and see their distribution

# In[ ]:


plt.hist(train['Age'],bins=list(range(0,60,1)))


# We can see that the age of the pet are mostly below 20 months.
# 
# One interesting finding is that people prefer to input YEAR rather than MONTH when filling the pet's age, so there are peaks at every 12 months
# 

# ### Gender <a class="anchor" id="gender"></a>
# 

# In[ ]:


# Gender distribution
train['Gender'].value_counts().rename({1:'Male',2:'Female', 3:'Mixed (Group of pets)'}).plot(kind='barh')
# plt.yticks(fontsize='xx-large')
plt.title('Gender distribution', fontsize='xx-large')


# ### State <a class="anchor" id="state"></a>
# Refer to the states(or city) in Malaysia. Pet's in larger city may have more chances to be adopted

# In[ ]:


states


# In[ ]:


states_to_ID = states.set_index('StateName')
state_value_counts = train['State'].value_counts(ascending=False)
state_distribution = states_to_ID['StateID'].map(state_value_counts).sort_values(ascending=False)
state_distribution


# 
# As I know, Kuala Lumpur is a city of the state of Selangor. So  we may need to convert cities into states

# In[ ]:


train['State'] = train['State'].replace(41401, 41326)# convert Kuala Lumpur to Selangor 


# There is no record for Perlis, so the value above is NaN
# 
# 

# ### Photo amount and video amount distribution <a class="anchor" id="amount"></a>

# In[ ]:


train['PhotoAmt'].describe()


# we can see maximum 30 photos are uploaded for a pet.  Let's plot all possible photo numbers

# In[ ]:


train['PhotoAmt'].plot(kind='hist', 
                          bins=30, 
                          xticks=list(range(31)))
plt.title('Photo Amount distribution')
plt.xlabel('Photos')


# In[ ]:


train['VideoAmt'].describe()


# In[ ]:


train['VideoAmt'].plot(kind='hist', 
                          bins=8, 
                          xticks=list(range(9)))
plt.title('Video Amount distribution')
plt.xlabel('Video')


# ## Description Length

# In[ ]:


train['Description'] = train['Description'].fillna('')
test['Description'] = test['Description'].fillna('')
train['desc_len'] = train['Description'].apply(lambda x: len(x))


# In[ ]:


train['desc_len'].describe()


# In[ ]:


test['desc_len'] = test['Description'].apply(lambda x: len(x))
test['desc_len'].describe()


# 

# we can see most people uploaded only 1 video of the pet

# # Data Cleaning <a class="anchor" id="clean"></a>
# we need to drop these columns
# * 'AdoptionSpeed'. It had been used as target
# * 'Name', 'RescuerID', 'PetID'. they won't be helpful from basic understanding.
# * 'Description'. it had been transformed into sentiments
# 

# In[ ]:


# Clean up DataFrames
# Will try to implement these into the model later
target_train = train['AdoptionSpeed']
cleaned_train = train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
test_pet_ID = test['PetID']
test_X = test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])


# In[ ]:


cleaned_train.head()


# next we need to check if there are null values inside the traing and testing dataframe

# In[ ]:


target_train.isnull().values.any()


# In[ ]:


test_X.isnull().values.any()


# The data is really clean! Isn't it? it saved our valuable time. Thanks PetFinder.my for cleaning for us

# # Modelling <a class="anchor" id="model0"></a>
# 
# training.csv already contain some features. the other feaures are from images and description of the pet. In order to test the importance of the features from mages and description, I will do modelling in mulitiple steps.
# 
# ## Model Selection <a class="anchor" id="selection"></a>
# From Model, here I will select Random Forest and XGBoost to ensemble. Random Forest is to **reduce variance** and XGBoost is to **reduce bias**. Together they can get better result. 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC

seed = 42


# In[ ]:


class EnsembleModel:
    
    def __init__(self,balancing=False):
        self.balance_ratio = 5 if balancing else 1
        self.rf_model = RandomForestClassifier()
        self.lgb_model = lgb.LGBMClassifier()
        self.rand_forest_params= {
            'bootstrap': [True, False],
            'max_depth': [20,30],
            'min_samples_leaf': [20, 30],
            'min_samples_split': [8,10],
            'n_estimators': [200,250],
            'random_state' : [seed]
        }
        self.lgb_params = {'objective' : ['multi:softprob'],
              'eta' : [0.01],
              'max_depth' : [6,7],
              'num_class' : [5],
              'num_leaves':[40,50],
              'lambda' : [0.75],
              'reg_alpha':[1e-5, 1e-2],
              'silent': [1]
        }
        self.svm = SVC()
        self.svm_params = {'kernel':['linear'],
                           'C':[0.5,0.75],
                           'gamma': ['auto'],
                           'decision_function_shape':['ovo','ovr'],
                           #'shrinking':[True,False]
                          }

        self.rf_best_param = None
        self.lgb_best_param = None
        self.svm_best_param = None
        self.columns = None
        
    
    def set_scorer(self,kappa):
        self.kappa = kappa
        self.scorer = make_scorer(kappa)
        
    def set_param(self,rf_param,lgb_param,svm_param):
        self.rf_best_param = rf_param
        self.lgb_best_param = lgb_param
        self.svm_best_param = svm_param
    
    def tune_best_param(self,x_train,y_train):
        weights_train = [self.balance_ratio if i==0 else 1 for i in y_train.tolist()]
        
        svm_gridsearch = GridSearchCV(self.svm, self.svm_params,
                                      cv=3,
                                      scoring=self.scorer,verbose=1, 
                                      refit=True
                                     )
        svm_gridsearch.fit(x_train, y_train, sample_weight = weights_train)
        self.svm = svm_gridsearch.best_estimator_
        self.svm_best_param = svm_gridsearch.best_params_
        print('tuning for svm finished')
        
        rf_gridsearch = GridSearchCV(estimator = self.rf_model, 
                                      param_grid = self.rand_forest_params, 
                                      cv = 5, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=self.scorer)
        rf_gridsearch.fit(x_train, y_train, sample_weight = weights_train)
        print('tuning for rf finished')
        self.rf_model = rf_gridsearch.best_estimator_
        self.rf_best_param = rf_gridsearch.best_params_
        
        lgb_gridsearch = GridSearchCV(self.lgb_model, self.lgb_params, n_jobs=-1, 
                   cv=5, 
                   scoring=self.scorer,
                   verbose=1, refit=True)
        lgb_gridsearch.fit(x_train, y_train, sample_weight = weights_train)
        print('tuning for lgb finished')
        self.lgb_model = lgb_gridsearch.best_estimator_
        self.lgb_best_param = lgb_gridsearch.best_params_
        
        
        
        print('best param for rf is:')
        print(self.rf_best_param)
        print('best param for lgb is:')
        print(self.lgb_best_param)
        print('best param for svm is:')
        print(self.svm_best_param)
    
    # let's try combining the 3 models together by averging
    def _avg(self,y_1,y_2,y_3):
        return np.rint((y_1 + y_2)/2.0).astype(int)

    def re_fit_with_best_param(self,X,y):
        if self.rf_best_param == None or self.lgb_best_param == None or self.svm_best_param == None: 
            print('use tune_best_param() method to get best param first')
            return
        weights_train = [self.balance_ratio if i==0 else 1 for i in y.tolist()]
        self.rf_model = RandomForestClassifier()
        self.lgb_model =  lgb.LGBMClassifier()
        self.svm = SVC()
        self.rf_model.set_params(**self.rf_best_param)
        self.lgb_model.set_params(**self.lgb_best_param)
        self.svm.set_params(**self.svm_best_param)
        self.rf_model.fit(X,y,sample_weight=weights_train)
        self.lgb_model.fit(X,y,sample_weight=weights_train)
        self.svm.fit(X,y,sample_weight=weights_train)
        print('refit finished')
    
    def validate(self,x_valid, y_valid):
        rf_score = self.kappa(self.rf_model.predict(x_valid), y_valid)
        print('{} score: {}'.format('rf', round(rf_score, 4)))
        lgb_score = self.kappa(self.lgb_model.predict(x_valid), y_valid)
        print('{} score: {}'.format('lgb', round(lgb_score, 4)))
        svm_score = self.kappa(self.svm.predict(x_valid), y_valid)
        print('{} score: {}'.format('svm', round(svm_score, 4)))
        score = kappa(self._avg(self.lgb_model.predict(x_valid), self.rf_model.predict(x_valid), self.svm.predict(x_valid)) , y_valid)
        print('{} score on validation set: {}'.format('combiner', round(score, 4)))
        self.columns = x_valid.columns

    def predict(self,test_X):
        rf_result = self.rf_model.predict(test_X)
        lgb_result = self.lgb_model.predict(test_X)
        svm_result = self.svm.predict(test_X)
        final_result = self._avg(rf_result,lgb_result,svm_result)
        return final_result


    def get_feature_importance(self):
        rf_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.rf_model.feature_importances_.tolist()})
        lgb_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.lgb_model.feature_importances_.tolist()})
        svm_feature_importances = pd.DataFrame({'Feature':self.columns.tolist(),'importance':self.svm.coef_.tolist()})
        overall_feature_importance = pd.merge(rf_feature_importances, lgb_feature_importances, svm_feature_importances, on='Feature', how='outer')
        overall_feature_importance['avg_importance'] = (overall_feature_importance['importance_x'] + overall_feature_importance['importance_y']+ overall_feature_importance['importance_z'])/3
        overall_feature_importance = overall_feature_importance.sort_values(by=['avg_importance'], ascending=False)
        return overall_feature_importance


# 
# 
# ## Model from original train dataset <a class="anchor" id="model1"></a>
# 
# ### Data Cleaning <a class="anchor" id="clean"></a>
# we need to drop these columns
# * 'AdoptionSpeed'. It had been used as target
# * 'Name', 'RescuerID', 'PetID'. they won't be helpful from basic understanding.
# * 'Description'. it had been transformed into sentiments
# 

# In[ ]:


# Clean up DataFrames
# Will try to implement these into the model later
target_train = train['AdoptionSpeed']
cleaned_train = train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
test_pet_ID = test['PetID']
test_X = test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 
                                                      target_train, 
                                                      test_size=0.2, 
                                                      random_state=seed)


# ### Model Tuning
# 
# From Version 2.0, we make the model ensembling as a class so that it will be easier to replicate.

# In[ ]:


if MODEL_USE == 1 or MODEL_USE==0:
    first_model = EnsembleModel(balancing=True)
    first_model.set_scorer(kappa)
    first_model.tune_best_param(x_train, y_train)
    first_model.validate(x_valid, y_valid)


# ## Model with description sentiments <a class="anchor" id="model2"></a>
# Let's examinate how a description sentiment file looks like:

# In[ ]:


import json
filename = os.listdir("../input/train_sentiment")[1]
filename = "../input/train_sentiment/"+filename
with open(filename, 'r') as f:
    sentiment = json.load(f)
sentiment  


# Entities in the json are too complex to use, but we can see that *documentSentiment': {'magnitude': 0.8, 'score': 0.4}* is easy to use. File name format is PetID.json, Let's use these 2 variables

# In[ ]:


def load_desc_sentiment(path):
    all_desc_sentiment_files = os.listdir(path)
    count_file = len(all_desc_sentiment_files)
    desc_sentiment_df = pd.DataFrame(columns=['PetID','desc_senti_magnitude','desc_senti_score'])
    current_file_index = 1
    for filename in all_desc_sentiment_files:
        with open(path+filename, 'r') as f:
            sentiment_json = json.load(f)
            petID = filename.split('.')[0]
            magnitude = sentiment_json['documentSentiment']['magnitude']
            score = sentiment_json['documentSentiment']['score']
            desc_sentiment_df = desc_sentiment_df.append({'PetID': petID, 'desc_senti_magnitude':magnitude,'desc_senti_score':score},                                                          ignore_index=True)
            if current_file_index % 1000 == 0 or current_file_index == count_file :
                print('current progress: %d file of %d loaded' %(current_file_index,count_file))
            current_file_index += 1
    return desc_sentiment_df


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
tfv = TfidfVectorizer(min_df=2,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        )
tfv.fit(train['Description'])
desc_X_train =  tfv.transform(train['Description'])
desc_X_test = tfv.transform(test['Description'])
print(desc_X_train.shape)
print(desc_X_test.shape)


# In[ ]:


svd = TruncatedSVD(n_components=5)
svd.fit(desc_X_train)
# print(svd.explained_variance_ratio_.sum())
# print(svd.explained_variance_ratio_)
desc_X_train = svd.transform(desc_X_train)
desc_X_test = svd.transform(desc_X_test)
print("desc_X_train (svd):", desc_X_train.shape)
print("desc_X_test (svd):", desc_X_test.shape)


# In[ ]:


train_desc_sentiment_df = load_desc_sentiment("../input/train_sentiment/")
test_desc_sentiment_df = load_desc_sentiment("../input/test_sentiment/")


# In[ ]:


# train_desc_sentiment_df['score_times_mag'] = train_desc_sentiment_df['desc_senti_magnitude'] * train_desc_sentiment_df['desc_senti_score']
# test_desc_sentiment_df['score_times_mag'] = test_desc_sentiment_df['desc_senti_magnitude'] * test_desc_sentiment_df['desc_senti_score']


# In[ ]:


train_desc_sentiment_df.head(5)


# In[ ]:


desc_X_train = pd.DataFrame(desc_X_train, columns=['desc_{}'.format(i) for i in range(svd.n_components)])
desc_X_test = pd.DataFrame(desc_X_test, columns=['desc_{}'.format(i) for i in range(svd.n_components)])
train_with_desc = pd.concat([train,desc_X_train],axis=1)
test_with_desc = pd.concat([test,desc_X_test],axis=1)


# In[ ]:


target_train = train_with_desc['AdoptionSpeed']
joint_train = train_with_desc.merge(train_desc_sentiment_df, how='left',left_on=['PetID'],right_on=['PetID'])
cleaned_train = joint_train.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])
cleaned_train.fillna(0.0,inplace=True)

test_pet_ID = test_with_desc['PetID']
joint_test = test_with_desc.merge(test_desc_sentiment_df, how='left',left_on=['PetID'],right_on=['PetID'])
test_X = joint_test.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])
test_X.fillna(0.0, inplace=True)

x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 
                                                      target_train, 
                                                      test_size=0.2, 
                                                      random_state=seed)


# In[ ]:


if MODEL_USE == 2 or MODEL_USE==0:
    second_model = EnsembleModel(balancing=True)
    second_model.set_scorer(kappa)
    second_model.tune_best_param(x_train, y_train)
    second_model.validate(x_valid,y_valid)


# ## Model with images features and above<a class="anchor" id="model3"></a>
# To be implemented in version 3.0
# 

# In[ ]:


def add_meta_feature(path,df):
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
    pet_id = df['PetID']
    for pet in pet_id:
        try:
            with open(path + pet + '-1.json', 'r') as f:
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
    df.loc[:, 'vertex_x'] = vertex_xs
    df.loc[:, 'vertex_y'] = vertex_ys
    df.loc[:, 'bounding_confidence'] = bounding_confidences
    df.loc[:, 'bounding_importance'] = bounding_importance_fracs
    df.loc[:, 'dominant_blue'] = dominant_blues
    df.loc[:, 'dominant_green'] = dominant_greens
    df.loc[:, 'dominant_red'] = dominant_reds
    df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    df.loc[:, 'dominant_score'] = dominant_scores
    df.loc[:, 'label_description'] = label_descriptions
    df.loc[:, 'label_score'] = label_scores
#     df = df.drop(['label_description'])
    return df



if MODEL_USE == 3 or MODEL_USE==0:
    train_with_meta = add_meta_feature('../input/train_metadata/', train_with_desc)
    target_train = train_with_meta['AdoptionSpeed']
    cleaned_train = train_with_meta.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed', 'label_description'])
    cleaned_train.fillna(0.0,inplace=True)
    
    test_with_meta = add_meta_feature('../input/test_metadata/', test_with_desc)
    test_pet_ID = test_with_desc['PetID']
    test_X = test_with_meta.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'label_description'])
    test_X.fillna(0.0, inplace=True)

    x_train, x_valid, y_train, y_valid = train_test_split(cleaned_train, 
                                                          target_train, 
                                                          test_size=0.2, 
                                                          random_state=seed)


# In[ ]:


x_train.shape


# In[ ]:


# Metadata:
# train_df_ids = train[['PetID']]
# train_df_metadata = pd.DataFrame(train_metadata_files)
# train_df_metadata.columns = ['metadata_filename']
# train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
# train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)
# print(len(train_metadata_pets.unique()))

# pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))
# print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / train_df_ids.shape[0]))


# In[ ]:


if MODEL_USE == 3 or MODEL_USE==0:
    third_model = EnsembleModel(balancing=True)
    third_model.set_scorer(kappa)
    third_model.tune_best_param(x_train, y_train)
    third_model.validate(x_valid,y_valid)


# Marvelous! Combiner model do give us better result! on validation set
# Acutually you can play with more models and tune the best params to score the top leaderboard

# # Feature Importance and Conclusion <a class="anchor" id="importance"></a>
# 
# Let's analyze the feature importance 

# In[ ]:


model = None
if MODEL_USE == 1:
    model = first_model
if MODEL_USE == 2: 
    model = second_model
if MODEL_USE == 0 or MODEL_USE == 3: # if all 3 model is enabled, we just use the 3rd model
    model = third_model


# In[ ]:





# In[ ]:


# overall_feature_importance = model.get_feature_importance()
# overall_feature_importance.head(5)
# overall_feature_importance.drop(['importance_x','importance_y'],axis=1).set_index('Feature').plot(kind='bar')


#  The model are trained on 80% of the training set (20% was left out for validation. now we will use the best param obtained in previous step to train on the full dataset

# In[ ]:


model.re_fit_with_best_param(cleaned_train,target_train)


# # Result and Submission <a class="anchor" id="result"></a>

# In[ ]:


final_result = model.predict(test_X)


# In[ ]:


submission_df = pd.DataFrame(data={'PetID' : test_pet_ID.tolist(), 
                                   'AdoptionSpeed' : final_result})
submission_df.head(5)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:


len(model.svm.coef_.ravel())


# In[ ]:


model.lgb_model.feature_importances_


# In[ ]:




