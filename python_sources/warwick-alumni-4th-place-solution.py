#!/usr/bin/env python
# coding: utf-8

# # 4th Place Solution Using TF-IDF and Logistic Regression
# Warwick Alumni would like to congratulate the winning teams and all teams who made it to the end of the competition! Although the discussion board was not particularly focused on developing better and more novel solutions, it was nice to see teams helping each other out with respect to the data leak and standing up for one another when it came to the final ranking.
#   
# # Table of Contents
#   
# 1. [Overview of Approach](#Overview)
# 2. [Import Modules and Define Custom Functions](#importModules)  
#     * [Mean Average Precision @ 2 (MAP@2)](#map2)  
#     * [Logistic Regression with Two Predictions](#lr_map2)
#     * [Label Transformer](#labelTransform)
#     * [Generating Predictions](#recommendTwo)
#     * [Preparing the Submission](#submitKaggle)
# 3. [User Parameters](#userParams)
# 4. [Basic Data Cleaning](#dataCleaning)
# 5. [Model Training](#modelTraining)
#     * [Parameter Optimisation](#paramOpt)
#     * [Generating the Submission](#genSub)
# 6. [Conclusion](#conclusion)
# 

# # Overview of Approach <div id=Overview></div>
# In this kernel, we present a walkthrough of our solution for the National Data Science Challenge 2019 (Advanced Category) task of predicting product attributes from text and images. We will use the Camera feature from the Mobile dataset simply because it is the smallest and would require less time to train.
# 
# Our approach involved the following steps:  
#   
# 1. **Basic Data Cleaning:** Translation of keywords in **training and test set titles**. We defined **keywords** as the words in class labels for the target features.
# 2. **Feature Extraction:** Generation of binary term frequency (BTF) using `TfidfVectorizer` for (1) keywords and (2) titles.
# 3. **Model Training:** Training of a Logistic Regression model using the OneVsRest scheme.  
#   
# We used the same approach for **all target features**, with minimal tweaks to the `TfidfVectorizer` and `LogisticRegression` parameters.  

# # Import Modules and Define Custom Functions <div id=importModules></div>
# First, we import the required modules and define several custom functions that we used.

# In[ ]:


# Import required modules
import json
import numpy as np
import pandas as pd
import time
import warnings

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ### Mean Average Precision @ 2 (MAP@2) <div id=map2></div>
# First, we defined a function to compute the MAP@2 metric. Note that optimising for accuracy and optimising for MAP@2 are not quite the same thing. Accuracy only scores the class with the highest predicted probability, while MAP@2 scores the **classes with the two highest predicted probabilities**. Hence, we used this function in cross validation for parameter optimisation.

# In[ ]:


# MAP2
def MAP2(y_true, y_pred):
    
    # Ensure number of rows are the same
    if len(y_true) != y_pred.shape[0]:
        
        # Throw an error
        raise Exception("Length of ground truth vector and predictions differ.")
    
    # Compute Average Precision (AP)
    ap = ( (y_pred[:, 0] == y_true).astype(int) +
           (y_pred[:, 1] == y_true).astype(int) / 2 )
    
    # Compute mean of AP across all observations
    output = np.mean(ap)
    
    # Return
    return output


# ### Logistic Regression with Two Predictions <div id=lr_map2></div>
# Next, we defined** a function for our `Pipeline` that outputs a dataframe / Numpy array with 2 columns that represent the 1st and 2nd predictions, respectively. We needed this because the scorer that we fed into `GridSearchCV` (function for performing a grid search for optimal parameters using cross validation) requires exactly that: a dataframe / Numpy array with 2 columns.

# In[ ]:


# Logistic regression that outputs 2 recommendations
class lr_map2(BaseEstimator, ClassifierMixin):
    
    def __init__(self, multi_class='ovr', solver='saga', max_iter=100, C=1.0, random_state=123, n_jobs=4, class_weight=None):
        
        self.multi_class=multi_class
        self.solver=solver
        self.C=C
        self.class_weight = class_weight
        self.max_iter=max_iter
        self.random_state=random_state
        self.n_jobs=n_jobs
    
    def fit(self, X, y=None):
        
        self.model = LogisticRegression(
            multi_class=self.multi_class,
            solver=self.solver,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X, y=None):
        
        # Predict probability of all classes
        pred_probs = self.model.predict_proba(X)
        
        # Extract top two classes
        pred = self.model.classes_[np.apply_along_axis(lambda x: x.argsort()[-2:][::-1], axis=1, arr=pred_probs)]
        
        return pred


# ### Label Transformer <div id=labelTransform></div>
# This function was used to process titles in our `Pipeline`. It takes several `TfidfVectorizer` parameters, and generates binary term frequencies (BTFs) for the input data. Another input is `labels_tgt`, which essentially represents **keywords** (words in the class labels of the target feature). The function returns a `csr_matrix` containing only binary features.  
#   
# *Note: We also had a variant of `LabelTransform` called `TitleTransform` that we used to input stopwords. Implementing it is easy, so we won't convolute the post by providing the code for that function.*

# In[ ]:


# Transformer to fit TF-IDF vectorizer with target labels and transform titles
class LabelTransform(TransformerMixin):
    
    def __init__(self, labels_tgt, ngram_range, max_df, min_df):
        
        self.labels_tgt = labels_tgt
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
    
    def set_params(self, labels_tgt=None, ngram_range=None, max_df= None, min_df=None):
        
        if labels_tgt:
            self.labels_tgt = labels_tgt
        
        if ngram_range:
            self.ngram_range = ngram_range
        
        if max_df:
            self.max_df = max_df
        
        if min_df:
            self.min_df = min_df
    
    def fit(self, X, y=None):
        
        # Initialise TF-IDF vectorizer for target labels
        self.vect_labels = TfidfVectorizer(
            
            # USE COUNTS
            use_idf=False, norm=False, binary=True,
            
            # ALLOW SINGLE ALPHANUMERICS
            token_pattern='(?u)\\b\\w+\\b',
            
            # TUNE THESE
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df
        )
        
        # Fit to target labels
        self.vect_labels.fit(self.labels_tgt)
        
        return self

    def transform(self, X, y=None):
        
        # Transform
        output = self.vect_labels.transform(X)
        
        # Return
        return output


# ### Generating Predictions <div id=recommendTwo></div>
# Next, we defined a function for providing the required 2 predictions. It takes as its input a `Pipeline` object (with `LogisticRegression`, not `lr_map2` as the estimator) and the test set titles, and returns a dataframe containing the 2 predictions.

# In[ ]:


# Function for extracting top 2 recommendations
def recommend_two(model, X_val):
    
    # Obtain predictions
    pred_probs = model.predict_proba(X_val)
    
    # Extract top two classes
    pred = pd.DataFrame(model.classes_[np.apply_along_axis(lambda x: x.argsort()[-2:][::-1], axis=1, arr=pred_probs)])
    
    return pred


# ### Preparing the Submission <div id=submitKaggle></div>
# Finally, we defined a simple function for formatting the predictions from `recommend_two` for submission on Kaggle.

# In[ ]:


# Function for preparing submissions
def submit_kaggle(itemid, var, pred):
    
    # Establish output
    output = pred.copy()
    
    # Set itemid
    output = pd.DataFrame(itemid.astype(str) + '_' + str(var))
    output.rename(columns = {'itemid': 'id'}, inplace=True)
    
    # Set tagging
    output['tagging'] = pred.iloc[:, 0].astype(int).astype(str) + ' ' + pred.iloc[:, 1].astype(int).astype(str)
    
    # Return
    return output


# # 0. Setting User Parameters <div id=userParams></div>
# We used one cell to input user parameters:  
#   
# * `VAR`: Which target feature to train model for.
# * `LABEL`: Which target feature to use for keywords. For some target features like Mobile Operating System, it made no sense to use the corresponding keywords, because sellers typically do not write a phone's OS in a sales listing. In this case, we used Phone Model instead. The key idea is to choose a set of keyword that has a strong link to the target feature **and** appears in the titles.  
# * `main`: The training set.
# * `val`: The test set.
# * `main_cb`: Dictionary of mappings from encoded labels to keywords.  
# * `stop_words`: User-defined stopwords.
#   
# This enabled us to use a single Jupyter notebook as a template, and modify these parameters as required to test models and generate predictions from target features from any dataset.  
#   
# As you can see, we used a small set of stopwords for our final solution. This is because removing stopwords actually *decreased* performance on the test set. We suspect that there were insufficient words used in the models - increasing the bank of vocabulary should have improved model performance.

# In[ ]:


# INPUT ATTRIBUTE AND LABEL HERE:
VAR = 'Camera'
LABEL = VAR
DATASET = 'mobile'

# Read data
main = pd.read_csv('../input/mobile_data_info_train_competition.csv')
val = pd.read_csv('../input/mobile_data_info_val_competition.csv')

# Import codebook
with open('../input/mobile_profile_train.json', 'r') as f:
    main_cb = json.load(f)

# Configure stopwords
stop_words = set([
    'promo','diskon','baik','terbaik', 'murah',
    'termurah', 'harga', 'price', 'best', 'seller',
    'bestseller', 'ready', 'stock', 'stok', 'limited',
    'bagus', 'kualitas', 'berkualitas', 'hari', 'ini',
    'jadi', 'gratis'
])


# # 1. Basic Data Cleaning <div id=dataCleaning></div>
#   
# ## Translations
# The only thing we did to clean the titles was to perform translations. The set of keywords below were words that were (1) keywords and (2) frequently used. Note also that we cleaned the test set titles because we used the test set titles to train the model. As we mentioned earlier, the models did not seem to have sufficient vocabulary. Hence, we used the test set titles to augment the vocabulary provided in the training set.  
#   
# Translations on the test set were also made for consistency. Training our classifier on features extracted from translated titles and predicting on **un-translated test titles** caused our score to stagnate toward the ending phase of the competition. Only when we rectified this did we jump from 9th to 4th place on the final day.  
#   
# Note that we used different sets of translations for different datasets. For example, in the Beauty dataset, the Colour Group feature contained more colours in Bahasa Indonesian than in the Mobile Colour Family feature.   
#   
# ## Images
# We did not bother to use the images. They were not standardised and would have contributed noise to our models.  
#   
# ## Mistake in Sequence of Data Processing
# To use the test set titles, we accidentally extracted them along with the training set titles **before** translations were made. Had we done the extraction after the translations, our score could have been better.

# In[ ]:


# Set output csv name
filename = 'mobile_' + VAR.lower().replace(' ', '_') + '.csv'

# Get all titles - BIG MISTAKE
all_titles = main['title'].append(val['title'])

# Drop image
main.drop('image_path', axis = 1, inplace=True)

# Delete missing values
main = main[~main[VAR].isnull()]

# Translate words
main['title'] = main['title'].str.replace('tahun', 'year')
main['title'] = main['title'].str.replace('bulan', 'month')
main['title'] = main['title'].str.replace('hitam', 'black')
main['title'] = main['title'].str.replace('putih', 'white')
main['title'] = main['title'].str.replace('hijau', 'green')
main['title'] = main['title'].str.replace('merah', 'red')
main['title'] = main['title'].str.replace('ungu', 'purple')
main['title'] = main['title'].str.replace(' abu', 'gray')
main['title'] = main['title'].str.replace('perak', 'silver')
main['title'] = main['title'].str.replace('kuning', 'yellow')
main['title'] = main['title'].str.replace('coklat', 'brown')
main['title'] = main['title'].str.replace('emas', 'gold')
main['title'] = main['title'].str.replace('biru', 'blue')
main['title'] = main['title'].str.replace('tahan air', 'waterproof')
main['title'] = main['title'].str.replace('layar', 'touchscreen')

# Translate words
val['title'] = val['title'].str.replace('tahun', 'year')
val['title'] = val['title'].str.replace('bulan', 'month')
val['title'] = val['title'].str.replace('hitam', 'black')
val['title'] = val['title'].str.replace('putih', 'white')
val['title'] = val['title'].str.replace('hijau', 'green')
val['title'] = val['title'].str.replace('merah', 'red')
val['title'] = val['title'].str.replace('ungu', 'purple')
val['title'] = val['title'].str.replace(' abu', 'gray')
val['title'] = val['title'].str.replace('perak', 'silver')
val['title'] = val['title'].str.replace('kuning', 'yellow')
val['title'] = val['title'].str.replace('coklat', 'brown')
val['title'] = val['title'].str.replace('emas', 'gold')
val['title'] = val['title'].str.replace('biru', 'blue')
val['title'] = val['title'].str.replace('tahan air', 'waterproof')
val['title'] = val['title'].str.replace('layar', 'touchscreen')

# Configure target labels
labels_tgt = pd.Series(list(main_cb[LABEL].keys()))

# Rename data
X_data = main['title']
y_data = main[VAR]


# # 2. Model Training <div id=modelTraining></div>
# In this section, we present to sets of code. The first set of code is for parameter optimisation, and the second is for generating the submission.  
#   
# ## Parameter Optimisation <div id=paramOpt></div>
# First, we generate a `FeatureUnion` object for the two sets of extracted features. The first set of extracted features involved a `TfidfVectorizer` trained on **keywords**, and used to transform the combined titles. Intuitively, this represented the **direct matches** between a title and the target feature's classes. The second set of extracted features involved another `TfidfVectorizer` trained on **titles**, and used to transform the combined titles. This represented more **complex relationships** between a title and the target feature's classes.

# In[ ]:


# Set up feature union
opt_feats = FeatureUnion(
    [
        ('labels', LabelTransform(
            labels_tgt=labels_tgt,
            ngram_range=(1,1),
            max_df=0.2,
            min_df=1
        )),
        
        # Using LabelTransform to speed up code
        ('titles', LabelTransform(
            labels_tgt=all_titles,
            ngram_range=(1,3),
            max_df=0.3,
            min_df=1
        ))
    ]
)


# To demonstrate what this looked like:

# In[ ]:


# Create temporary Pipeline
temp_pipe = Pipeline([('test_feats', opt_feats)])

# Fit to data
temp_pipe.fit(X_data, y_data)

# Transform
temp_pipe_output = temp_pipe.transform(X_data)

# Select rows
pd.DataFrame(temp_pipe_output[:20, :10].toarray())


# Next, we set up the parameter grid (`params_test`) and the testing pipeline (`test_pipe`) with the `FeatureUnion` object above as the only `Transformer` and our custom Logistic Regression function with 2 predictions (`lr_map2`) as the `estimator`.

# In[ ]:


# Params
params_test = {
    'lr__C': [1]
}

# Test pipe
test_pipe = Pipeline(
    [
        ('tfidf', opt_feats),
        ('lr', lr_map2(multi_class='ovr', solver='saga', C=1))
    ]
)


# Next, we set up `GridSearchCV` to search the parameter grid for an optimal setting using MAP@2 as the evaluation metric. We put the results into a dataframe for easy viewing.  
#   
# **Note:** This took a very long time for target features with many categories like Phone Model. Hence, we eventually gave up and used default settings for the Mobile features with more samples and all Beauty and Fashion features.

# In[ ]:


# Initialise GridSearchCV
opt_test = GridSearchCV(
    estimator = test_pipe,
    param_grid = params_test,
    cv=5,
    scoring=make_scorer(MAP2, greater_is_better=True),
    iid=False,
    verbose=20,
    n_jobs=4,
)

# Start timer
start_time = time.time()

# Fit
opt_test.fit(X_data, y_data)

# Stop timer
end_time = time.time()
print('Time taken: %s mins.' % ('{:.2f}'.format((end_time-start_time)/60)))

# Extract parameters
cv_results = pd.DataFrame(opt_test.cv_results_['params'])

# Extract mean test score
cv_results['mean_test_score'] = pd.Series(opt_test.cv_results_['mean_test_score'])
cv_results['std_test_score'] = pd.Series(opt_test.cv_results_['std_test_score'])

# Display
cv_results.sort_values('mean_test_score', ascending=False)


# ## Generating the Submission <div id=genSub></div>
# The code below was used to set up a `Pipeline` with the optimal settings and generate a submission.

# In[ ]:


# Set up feature union
opt_feats = FeatureUnion(
    [
        ('labels', LabelTransform(
            labels_tgt=labels_tgt,
            ngram_range=(1,1),
            max_df=0.2,
            min_df=1
        )),
        
        ('titles', LabelTransform(
            labels_tgt=all_titles,
            ngram_range=(1,3),
            max_df=0.3,
            min_df=1
        ))
    ]
)

# Set up Logistic Regression
opt_clf = LogisticRegression(
    multi_class='ovr',
    solver='saga',
    C=1,
    random_state=123,
    n_jobs=4
)

# Set up full pipeline
opt_pipe = Pipeline(
    [
        ('tfidf', opt_feats),
        ('lr', opt_clf)
    ]
)

# Train on dataset
opt_pipe.fit(X_data, y_data)

# Obtain predictions
pred = recommend_two(opt_pipe, val['title'])

# Prepare dataset
submission = submit_kaggle(val.itemid, VAR, pred)

# Output
submission.to_csv(filename, index=False)


# # Conclusion <div id=conclusion></div>
# We used extremely simple techniques to achieve our private leaderboard score of 0.46673, which earned us the 4th-place spot. We wished to have tested more complex techniques (e.g. deep learning), but we were pleasantly surprised by how far this simple approach brought us. Although we may not have achieved the highest leaderboard score, we recommend Shopee to consider the tradeoff between model training time and performance. Our approach offers a good balance of both.  
#   
# For the data scientists out there, our key learning points are:
#   
# 1. No model is too simple to be tested.
# 2. Exhaust the possibilities of (1) what features you extract, (2) how you process them, and (3) which ML algorithms you use them with.  
#   
# By sharing this solution, we hope to encourage aspiring data scientists, folks who are uninitiated but want to know more, and leaders who wish to push data science in their organisation with the simple message that **data science does not need to be as complex as people say**. Sometimes, good intuition and a simple approach is all you need.
