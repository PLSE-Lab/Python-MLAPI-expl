#!/usr/bin/env python
# coding: utf-8

# ### Intro
# Week1:
# - Intro
# - Feature Preprocess and Extraction
# 
# Week2:
# - EDA
# - Validation
# - Data leaks
# 
# Week3:
# - Metrics
# - Mean-encodings
# 
# Week4:
# - Advanced features
# - Hyperparameter optimization
# - Ensembles
# 
# Week5:
# - Final Projects
# - Winning Solutions
# 
# Kaggle, DrivenData, CrowdAnalityX, CodaLab, DataScienceChallenge.net, Datascience.net, KDD, VizDoom
# 
# ---
# ---
# ---

# ### Recap
# ML Algorithms
# - Linear(Ligistig Regression, SVM)
# - Tree-based (Decision Tree, Random Forest, GDBT, XGBoost, LightGBM)
# - kNN
# - Neural Networks(Image, Sound, Text, Sequence - Tensorflow, Keras, dmlc MXNet, Pytorch, Lasagne)
# 
# ##### Links
# + https://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively
# - http://manishbarnwal.com/blog/2017/02/08/the_curse_of_bias_and_variance/
# - http://manishbarnwal.com/blog/2017/03/28/h2o_with_r/
# - http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html
# 
# scikit-learn, Vowpal Wabbit(for large datasets)
# 
# - No Free Lunch Theorem
# - No Silver Buller Algorithm
# - Linear Models split space into 2 subspaces
# - Tree-based methods splits space into boxes
# - kNN measure closeness
# - FFNNs produce smooth non-linear decision boundary
# - GBDT and NNs are the most powertful methods (sometimes other may be better)
# 
# ---
# ---
# ---

# Feature Preprocession is necessary
# Feature Generation is powerful
# Preprocessing and Generation depend on a model type

# ### Numeric Features Preprocessing
# 
# 
# * Tree-based models doesn't depend on scaling
# 
# Clipping for outliers
# - Rank transformation: can be a better option than MinMaxScaler if we have outliers
# - Min-Max Scaler: (X-min(X))/(max(X)-min(X))
# - Standard Scaler: (X-mean(X))/std(X) 
# - Log Transformation: np.log(1+x)
# - Raising to the power <1: np.sqrt(x+2/3)
# 
# ### Numeric Features Generation
# - creativity and data understanding
# - multiply, divide
# - a2 + b2 = c2
# - fractional part
# 
# 
# ---
# ---
# ---

# ### Categorical Features Preprocessing
# - ordinal features(sorted in some meaningful order)
#     - Ticket class:1,2,3
#     - Driver's license: A,B,C,D
#     - Education: undergrad,grad,doctoral
# - Always use Encoding. 
#     - sklearn.preprocessing.LabelEncoder encodes in alphabetical order(often used of tree-based models)
#     - pandas.factorize encodes by order of appearance
#     - frequency encoding: Encode via mapping values to their frequencies(often used of tree-based models)
#     - one-hot (tree-based models may slow)(often used of non-tree-based models)
#     - interactions of categorical features can help linear models
#     
# ---
# ---
# ---

#  ### Datetime Features Preprocessing
#  - Periodicity
#      - time moments in a period or time passed since particular event.
#      - Day number in week, month, season, year, second, minute, hour
#  - Time Since
#      - Row independent: Time since 1 January 1970
#      - Row dependent: Number of days left until next holiday
#  - Difference: 
#      - datetime_feature1 - datetime_feature2
#      
# ### Coordinate Features Preprocessing
#    - Interesting Places: Distance to the nearest school(best), hospital, shop 
#    - Center of clusters: Distance to most expensive flat
#    - aggregate statistics: Mean price of the place. For objects surrounding area
# 
# ---
# ---
# ---

# ### Handling Missing Values
# - Missing values may hidden by replacing other value which is not a number
# - fill with other value: -999,-1, etc
# - fill with mean, median
# - fill with reconstructed value(use nearby values...)
# - new binary feature "isnull" can be beneficial
# - avoid filling nans before feature generation
# - XGBoost can handle NaN

# ### Feature Extraction from Text and Images
# 
# - Bag of Words:
#     - Each unique column for each word and fill with count of word in document (sklearn.feature_extraction.text.CountVectorizer)
#     - Huge vectors
# - Tf-Idf:
#     - Boost more important features while decrease the useless ones: TF-IDF: sklearn.feature_extraction.text.TfidfVectorizer
#         - tf = 1/x.sum(axis=1)[:,None]
#         - x = x*tf
#         - idf = np.log(x.shape[0] / (x>0).sum(0))
#         - x = x*idf
#  - N-grams(for local context):
#      - this is a bike
#      - n = 1: this,is,a,bike
#      - n = 2: this is, is a, a bike
#      - n = 3: this is a, is a bike
# - Word2Vec: Vector representations of words and texts
#     - king + woman - man = queen
#     - For words: Word2Vec, Glove, FastText, etc
#     - For sentences: Doc2Vec, etc
#     - Use pretrained models
#     - Relatively small vectors
# 
# - CNN
#     - process of pretrained(VGG, ResNet) model tuning is called fine tuning
#     - Image Augmentation (rotate,noise)
#     - Features can be etracted from different layets
# ### Text Preprocessing
# - Lowercase
# - Lemmatization, Stemming : I had a car -> I have car, We have cars -> We have car
#     - Stemming: Democr <- democracy, demcratic, democratization
#     - Lemmatization: Democracy <- democracy, demcratic, democratization
# - Stopwords (NLTK, max_df in CountVectorizer can handle stop words)
# 
# 

# In[ ]:




