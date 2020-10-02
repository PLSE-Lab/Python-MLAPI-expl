#!/usr/bin/env python
# coding: utf-8

# # Cat Challenge - A Workflow Guide: PCA + Logistic Regression + Wide and Deep Neural Networks
# ### How can we encode different types of categorical features?
# 
# #### Guilherme Vieira Dantas
# 
# ![GentleCat](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTSFF8K1-XmiGDkB7i_eIqxD_6ImLH4BXMsLMhZNAbiDeqR9oO0)
# 
# The aim of this notebook is to show the classical steps in a Machine Learning project that depends on a dataset characterized by many different types of categorical features. We will follow the conventional data science approach:
# 
# 1. Data reading + Exploration / Transformation
# 2. Dimentionality Reduction
# 3. Model Selection + Validation
# 4. Stacking Individual Models
# 5. Conclusions
# 
# I will be direct and I will also try to show different techniques that can be applied here, without losing time or lines of code. Let's begin!

# # 0. Importing the Libraries
# 
# Here we can take a look at all the libraries and resources that will be necessary to our project. It's interesting to aways start by this step to create a code with more quality and maintenability:

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from keras.layers import Dense, Dropout, Concatenate, Input, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

from category_encoders import TargetEncoder
from plotnine import *

import os
for dirname, _, filenames in os.walk('..'):
    for filename in filenames:
       print(os.path.join(dirname, filename))


# # 1. Reading and Exploring / Transforming the Data
# 
# Finally, let's start by taking a look at what we have:

# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 

input_dir = '/kaggle/input/cat-in-the-dat-ii/'


# In[ ]:


df_train_raw = pd.read_csv(input_dir + 'train.csv')
df_test_raw = pd.read_csv(input_dir + 'test.csv')

df_train_raw.head()


# Transforming some binary features to zeros and ones:

# In[ ]:


df_train_raw_transformed = df_train_raw.    assign(bin_3 = df_train_raw['bin_3'].map(lambda X: 0 if (X == 'F') else                                             (1 if (X == 'T') else np.nan))).    assign(bin_4 = df_train_raw['bin_4'].map(lambda X: 0 if (X == 'N') else                                             (1 if (X == 'Y') else np.nan)))

df_train_raw_transformed.head()


# In[ ]:


df_test_raw_transformed = df_test_raw.    assign(bin_3 = df_test_raw['bin_3'].map(lambda X: 0 if (X == 'F') else                                             (1 if (X == 'T') else np.nan))).    assign(bin_4 = df_test_raw['bin_4'].map(lambda X: 0 if (X == 'N') else                                             (1 if (X == 'Y') else np.nan)))


# In[ ]:


print("ORD_0:")
print(df_train_raw_transformed['ord_0'].unique())
print("---\nORD1:")
print(df_train_raw_transformed['ord_1'].unique())
print("---\nORD2:")
print(df_train_raw_transformed['ord_2'].unique())
print("---\nORD3:")
print(df_train_raw_transformed['ord_3'].unique())
print("---\nORD4:")
print(df_train_raw_transformed['ord_4'].unique())
print("---\nORD5:")
print(df_train_raw_transformed['ord_5'].unique())


# OK. We have not many columns in our problem. But we do have some columns with many different values, which can be a problem. Why? Well, when we transform string features to binary features we will create a number of dimentions equal to the number of different features less $1$ (we can subtract one because the case where all the columns are equal to zero corresponds to one of the strings). 
# 
# This kind of encoding is called "Binary Encoding" and when we get many dimentions thanks to the great number of different string values we get the problem known as the "curse of dimentionality" and the number of features with a given dummy column equal to one decreases exponentially:
# 
# ![CurseOfDimentionality](https://www.elasticfeed.com/wp-content/uploads/3e5fca2cb938bbc9f5a1cae43bac0944.jpg)
# 
# That's why it's interesting to find ordinal features in our dataset: we can just number them following some "sorting criteria". This is specially good when we are working with trees but to get assured that this choice is suitable to a model such like a Logistic Regression, we must have values that follow a linear relationship with the probability of getting a one in the output. It makes sense to say that since we apply a logistic function in a linear combination of the inputs to get the output of the model:
# 
# ![LogisticRegression](https://d2bwk5eec7cz2z.cloudfront.net/2018/09/9.png)
# 
# Anyway, we will use, for now, the simplest form of encoding, we will:
# * Keep the binary features just like they are
# * Label with $1$, $2$, $3$,... all the features that seem to follow an ordering rule
# 
# We will transform the categorical features in the next section.

# In[ ]:


def apply_dict(X, dict_in):
    try:
        out = dict_in[X]
    except:
        out = X
    return out

dict_ord_1 = dict(Novice=1, Contributor=2, Expert=3, 
                  Master=4, Grandmaster=5)

dict_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 
              'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6}

dict_ord_3 = dict_ord_4 = dict()
for i in range(ord('a'), (ord('z') + 1)):
    dict_ord_3[chr(i)] = dict_ord_4[chr(i + ord('A') - ord('a'))] = i - ord('a') + 1
    
df_train = df_train_raw_transformed.    assign(ord_1 = df_train_raw_transformed['ord_1'].map(lambda X: apply_dict(X, dict_ord_1))).    assign(ord_2 = df_train_raw_transformed['ord_2'].map(lambda X: apply_dict(X, dict_ord_2))).    assign(ord_3 = df_train_raw_transformed['ord_3'].map(lambda X: apply_dict(X, dict_ord_3))).    assign(ord_4 = df_train_raw_transformed['ord_4'].map(lambda X: apply_dict(X, dict_ord_4)))


# In[ ]:


df_test = df_test_raw_transformed.    assign(ord_1 = df_test_raw_transformed['ord_1'].map(lambda X: apply_dict(X, dict_ord_1))).    assign(ord_2 = df_test_raw_transformed['ord_2'].map(lambda X: apply_dict(X, dict_ord_2))).    assign(ord_3 = df_test_raw_transformed['ord_3'].map(lambda X: apply_dict(X, dict_ord_3))).    assign(ord_4 = df_test_raw_transformed['ord_4'].map(lambda X: apply_dict(X, dict_ord_4)))


# Checking the results:

# In[ ]:


df_train.head()


# How many samples we have to train?

# In[ ]:


len(list(df_train_raw_transformed.index))


# Yes, that's a huge number. We will probably face some difficulties to finetune our models...let's check graphically how many different features we have per column:

# In[ ]:


df_count_per_col = pd.DataFrame(df_train.nunique())
df_count_per_col.columns = ['Values']
df_count_per_col.index.name = 'Feature'
df_count_per_col.reset_index(inplace=True)

ggplot(df_count_per_col[df_count_per_col['Feature'] != 'id'], 
       aes(x = 'Feature', y = 'Values', fill = 'Feature')) + geom_bar(stat = 'identity', color = 'black') +\
    theme(axis_text_x = element_text(angle = 90, hjust = 1), legend_position = 'none') +\
    ggtitle('Different Features per Column') + ylab('Count')


# Another important information to explore is the missing data rate: how many "NAN's" do we have per column?

# In[ ]:


df_missing_col = pd.DataFrame(dict(PercMissing = df_train.isnull().sum() / 
                                   len(df_train.index))).reset_index()
df_missing_col.loc[df_missing_col['PercMissing'] > 0, :]

ggplot(df_missing_col[df_missing_col['Feature'] != 'id'], 
       aes(x = 'Feature', y = 'PercMissing', fill = 'Feature')) + geom_bar(stat = 'identity', color = 'black') +\
    theme(axis_text_x = element_text(angle = 90, hjust = 1), legend_position = 'none') +\
    ggtitle('Different Features per Column') + ylab('Count')


# It would be nice to follow a similar approach with the rows and check if we have a row with many missing values. If so, we can just drop them out without the risk of losing many informations.
# 
# It's not a good idea to try to do that with 600.000 different index values. So, let's just check the maximum missing data rate along all the rows:

# In[ ]:


df_missing_row = pd.DataFrame(dict(PercMissing = df_train.isnull().sum(axis=1) / 
                                   len(df_train.index))).reset_index()

df_missing_row.columns = ['Index', 'PercMissing']
print(str(100 * max(df_missing_row['PercMissing'])) + ' %')


# It's really small, so, we will not drop out any row and we are ready to start transforming our dataset. Also, since we have no high missing NA values per columns or rows, we don't need to spend a lot of time in the missing data imputation step.
# 
# Let's start out data transformation step by separing the variables in different classifications:

# In[ ]:


binary_features = ['bin_' + str(i) for i in range(0, 5)]
nominal_features_low_count = ['nom_' + str(i) for i in range(0, 5)]
nominal_features_high_count = ['nom_' + str(i) for i in range(5, 10)]
ordinal_features_low_count = ['ord_' + str(i) for i in range(0, 5)]
ordinal_features_high_count = ['ord_5']
date_features = ['day', 'month']


# And we will use the "SimpleImputer" class to fill the missing data. But I will just "correct" a problem in its source code: [Stack Overflow Question](https://datascience.stackexchange.com/questions/66034/sklearn-simpleimputer-too-slow-for-categorical-data-represented-as-string-values). Imputing the median of the values is slow and using the "constant" imputation method for each column with the most frequent value as constant can speed up the code considerably!
# 
# Thanks to the techniques of object orientation, we can inherit the methods of the original function and create a new "most_frequent" imputation method where we use, in each column, the Stack Overflow's purposed solution:

# In[ ]:


class SimpleImputerCorrected(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy='most_frequent', verbose=False):
        
        self.strategy = strategy
        self.preprocessor = None
        self.verbose = verbose

    def fit(self, X, y=None):
        
        if self.strategy == 'most_frequent':
            
            col_list = list(X.columns)
            col_transformers = list()
            col_mode = X.mode(axis=0).iloc[0]
        
            for curr_col in col_list:
                curr_transformer_name = 'T_' + curr_col
                curr_imputer = SimpleImputer(strategy='constant', 
                                             fill_value=col_mode[curr_col])
                col_transformers.append((curr_transformer_name, curr_imputer, 
                                         [curr_col]))
            
            self.preprocessor = ColumnTransformer(transformers=col_transformers, verbose=self.verbose)
            
        else:
            self.preprocessor = SimpleImputer(strategy=self.strategy)
            
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)


# All the steps described until now will be inserted in a Scikit's Pipeline! It's a functionality that allow us to define a sequence of transformation and modeling steps as a class that can be used in different datasets when, for example, our model is running in production. Let's see some Pipeline examples in the next figure:
# 
# ![Pipelines](https://www.researchgate.net/publication/334565019/figure/fig1/AS:782364141690881@1563541555043/The-Auto-Sklearn-pipeline-12-contains-three-main-building-blocks-a-Data.png)
# 
# Our first pipelines will be not so complicated and the code is self-explainable:

# What can we do with the ordinal and categorical features with many different values? I will apply the same threatment: the **Target Encoding**. It consists in transforming each category into the averages of the targets obtained for this category. 
# 
# I am considering that our ordinal feature with a high number of different values (ord_5) is categorical: we have $2$ random letters together, some of them in uppercase, some of them in lowercase characters and I am not really sure about how could I possibly transform it to an ordinal number that makes sense (should we sum the ASCII values? Should we sum the ASCII values after transforming the characters to their lowercase versions?)

# In[ ]:


pass_features = binary_features + ordinal_features_low_count
one_hot_features = nominal_features_low_count
avg_features = nominal_features_high_count + ordinal_features_high_count

pass_pipeline = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('scalling', StandardScaler())
], verbose = True)

one_hot_pipeline  = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('encoding', OneHotEncoder(sparse = False))
], verbose = True)

avg_pipeline = Pipeline(steps = [
    ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
    ('encoding', TargetEncoder()),
    ('scalling', StandardScaler())
], verbose = True)

encoder = ColumnTransformer(
    transformers=[
        ('pass_pipeline', pass_pipeline, pass_features),
        ('one_hot_pipeline', one_hot_pipeline, one_hot_features),
        ('avg_pipeline', avg_pipeline, avg_features)
], verbose = True, sparse_threshold=0)

print(one_hot_features)


# In[ ]:


df_train.iloc[1:5, 1:-2]


# In[ ]:


array_train = encoder.fit_transform(df_train.iloc[:, 1:-2], df_train['target'])
print(array_train)


# Now, we will use in the next section the Principal Component Analysis technique to remove redundant dimentions of our variables.
# 
# # 2. Dimentionality Reduction With Principal Component Analysis
# ## 2.1. Linear Transformations
# 
# The principal component analysis is, itself, a Scikit-Learn model. It's basically a simple linear transformation that is applied to out dataset. When I say that I am applying a linear transformation over a vector, I'm saying that I'm multiplying this vector by a matrix:
# 
# $$V_{Transformed} = A.V_{Before}$$
# 
# Where $V_{Before}$ is a single vector, with a number of dimentions equal to the number of columns of our original dataframe and we have as many vectors as rows in our dataset.
# 
# Initially, we apply a $N \times N$ matrix over our vectors, where $N$ is the number of columns. It means that we are using all the output dimentions of the vector.
# 
# ![LinearTransformation](http://duriansoftware.com/joe/media/gl3-linear-transformation-matrices-01.png)
# 
# If we apply the same linear transformation over different vectors, as we can see in the figure above, the angle between the two vectors will change and some components will be "stretched" ou "shrinked" as we can see in the figure above.
# 
# In the Principal Component Analysis we don't apply any linear transformation over our dataset. We will change the angle of each pair of vectors to $90$ degrees and obtain independent components (i.e vectors with no mutual projections over each other):
# 
# ![PCA](https://i2.wp.com/www.sportscidata.com/wp-content/uploads/2019/08/Principal_Component_Analysis_print.png?fit=1024%2C683&ssl=1)
# 
# ** This transformation is applied over the centered vectors, as we will explain in the next section **

# ## 2.2. Variances and Features Importance
# 
# Suppose we have two non orthogonal vectors in a bidimentional space and we want to calculate the sum of the variance along the $2$ dimensions. Then we have:
# 
# $$Var(X_{12}) = Var(X_1) + Var(X_2) + 2.Cov(X_1, X_2)$$
# 
# After applying a PCA over this pair, we obtain $2$ orthogonal vectors and the total variance can be calculated without the covariance term:
# 
# $$Var(X^*_{12}) = Var(X^*_1) + Var(X^*_2)$$
# 
# Where:
# $$ X^*_1 = \Lambda . (X_1 - \mu_{1,2}) $$
# $$ X^*_2 = \Lambda . (X_2 - \mu_{1,2}) $$
# $$ X^*_1 \perp X^*_2 \rightarrow X^*_1 \bullet X^*_2 = 0 $$
# $$ \mu_{1,2} = \frac{X_1 + X_2}{2} $$
# 
# And $\Lambda$ is our linear transformation PCA matrix. When we lose the covariance term and subtract the mean of each vector, something magic happens:
# 
# $$ Var({X^*_k}) = \mathbb{E} [{X^*_k}^2 - \mathbb{E} [{X^*_k}]^2] $$
# 
# Since we centered our data:
# 
# $ \mathbb{E} [{X^*_k}]^2 = 0 $
# 
# Then:
# 
# $\frac{Var({X^*_k})}{Var_{Total}} = \frac{\mathbb{E} [{X^*_k}^2]}{Var_{Total}} = \frac{|| {X^*_k}^2 ||_2}{Var_{Total}}$
# 
# And the participation of each feature over the total variance is proportional to the squared norm of the principal component! It also applies to more dimentions (we would do exactly the same steps).
# 
# Components with neglectable importances may be ignored without any risk to spoil the final model and with the benefit of reducting the computational weight of the model.
# 
# Applying our Scikit's PCA:

# In[ ]:


data_pca = PCA().fit_transform(array_train) 


# This is the transformed dataset:

# In[ ]:


data_pca


# In[ ]:


data_pca.shape


# I usually like to plot the feature importances from the most important variable to the less important one in a Paretto's diagram:

# In[ ]:


var_pca = data_pca.var(axis = 0)
imp_pca = var_pca / sum(var_pca)
cum_imp_pca = np.cumsum(imp_pca)

list_vars = var_pca.tolist() + imp_pca.tolist() + cum_imp_pca.tolist()
named_vars = ['Variance'] * len(var_pca) +             ['Importance'] * len(imp_pca) +             ['Cumulative_Importance'] * len(cum_imp_pca)
        
df_pca = pd.DataFrame(dict(
    IndexStr=[str(i) for i in 3 * list(range(0, len(named_vars) // 3))],
    Index=3 * list(range(0, len(named_vars) // 3)),
    Variable=named_vars,
    Value=list_vars
))

df_pca.head()


# After processing the Paretto's plot data in a dataframe, let's check the diagram:

# In[ ]:


ggplot(aes(x = 'Index', y = 'Value')) +    geom_bar(aes(fill = 'IndexStr'), color = 'black', stat = 'identity', data = df_pca[df_pca['Variable'] == 'Importance']) +    theme(legend_position = 'none') + ggtitle('Cumulative Variables Importance') +    geom_line(data = df_pca[df_pca['Variable'] == 'Cumulative_Importance'], color = 'blue') +    geom_point(data = df_pca[df_pca['Variable'] == 'Cumulative_Importance'], color = 'blue') +    geom_point(data = df_pca[(df_pca['Variable'] == 'Cumulative_Importance') &                             (df_pca['Value'] > 0.9999999999)], color = 'red', shape = 'x', size = 5)


# The red "X" is marked over the points where the cumulative importance is really, really near of $100 \%$. It seems that the feature importances of the last $4$ terms are numerically equal to zero:

# In[ ]:


imp_pca[35:-1]


# It's true! And we are ready to create our final pipeline, including the dimentionality reduction. It will be created with the aid of a parametric function:

# In[ ]:


def get_preprocessor(pass_features=binary_features + ordinal_features_low_count, 
                     one_hot_features=nominal_features_low_count, 
                     avg_features=nominal_features_high_count + ordinal_features_high_count, 
                     te_smoothing=1,
                     pca_threshold=0.9999):

    pass_pipeline = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('scalling', StandardScaler())])

    one_hot_pipeline  = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('encoding', OneHotEncoder(sparse = False))])

    avg_pipeline = Pipeline(steps = [
        ('imputation', SimpleImputerCorrected(strategy='most_frequent')),
        ('encoding', TargetEncoder(smoothing = te_smoothing)),
        ('scalling', StandardScaler())])

    encoder = ColumnTransformer(
    transformers=[
        ('pass_pipeline', pass_pipeline, pass_features),
        ('one_hot_pipeline', one_hot_pipeline, one_hot_features),
        ('avg_pipeline', avg_pipeline, avg_features)
    ], sparse_threshold=0)
    
    if (pca_threshold > 0):
        preprocessor = Pipeline(steps = [('encoder', encoder), ('pca', PCA(n_components=pca_threshold))])
    else:
        preprocessor = Pipeline(steps = [('encoder', encoder)])
    return preprocessor


# # 3. Logistic Regression - Regularization + Validation

# The kind of transformations that we will apply to our features will be a hyperparameter of a bigger model. We will aways use the "pass_features" (aka binary features) without any kind of transformation, since it's not strictly necessary. Also, all the features with many different values will be target encoded since we have to avoid the curse of dimentionality and our notebook memory limitations.
# 
# Then, we can make three choices:
# * Encode nominal features with few different values and ordinal features with few different values with one hot encoding (we call this "minimum_te")
# * Encode just nominal features with low count with one hot encoding and use target encoding in the ordinal features with few different values ("medium_te") or
# * Encode all nominal and ordinal features with target encoding ("maximum_te" configuration)
# 
# We will test the three possibilities.

# In[ ]:


pass_list = [pass_features, pass_features, pass_features]

one_hot_list = [nominal_features_low_count + ordinal_features_low_count,
                nominal_features_low_count, []]

avg_list = [avg_features,
            avg_features + ordinal_features_low_count,
            avg_features + ordinal_features_low_count + nominal_features_low_count]

transformers_name = ['minimum_te', 'medium_te', 'maximum_te']
transformers_list = zip(transformers_name, pass_list, one_hot_list, avg_list)

hyper_pipe_dict = {
    'lor_model__solver': ['saga'],
    'lor_model__penalty': ['elasticnet'],
    'lor_model__C': [0.1, 1, 10],
    'lor_model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'preprocessor__transformers': transformers_list,
    'preprocessor__target_smoothing': [0.1, 1, 10],
    'preprocessor__pca_threshold': [0.9999]
}


# Preparing a list with all hyperparameters possible combinations:

# In[ ]:


y_train = df_train['target']
k_fold_n_splits, k_fold_use = 20, 3

iter_list = []
C_list = []
l1_list = []
t_list = []
smoothing_list = []
solver_list = []
penalty_list = []
train_score_list = []
pca_list = []
test_score_list = []

for t_name, t_pass, t_one_hot, t_avg in hyper_pipe_dict['preprocessor__transformers']:
    for C in hyper_pipe_dict['lor_model__C']:
        for l1_ratio in hyper_pipe_dict['lor_model__l1_ratio']:
            for target_smoothing in hyper_pipe_dict['preprocessor__target_smoothing']:
                for solver in hyper_pipe_dict['lor_model__solver']:
                    for penalty in hyper_pipe_dict['lor_model__penalty']:
                        for pca_threshold in hyper_pipe_dict['preprocessor__pca_threshold']:
                            iter_list.append([C, l1_ratio, t_name, t_pass, t_one_hot, t_avg, 
                                              target_smoothing, solver, penalty, pca_threshold])


# The cross validation takes time. So, I saved the results in a .csv that will be loaded if the flag "cross_validate" is equal to "False". But you can easily reproduce my gridsearch by setting the flag to "True" :)

# In[ ]:


cross_validate = False


# Here we have the cross validation loop:

# In[ ]:


if cross_validate:
    
    for i in range(0, len(iter_list)):
        
        print('Progress:')
        print('{0:.2%}'.format(i / len(iter_list)))
        print('-----')
        
        C, l1_ratio, t_name, t_pass, t_one_hot, t_avg, target_smoothing, solver, penalty, pca_threshold = iter_list[i]
        preprocessing_pipeline = get_preprocessor(t_pass, t_one_hot, t_avg, target_smoothing, pca_threshold=pca_threshold)
        x_train = preprocessing_pipeline.fit_transform(df_train.iloc[:, 1:-2], y_train)
                        
        k_fold_obj = StratifiedKFold(n_splits=k_fold_n_splits, shuffle=True)
        k_fold_count = 0
        for train_index, test_index in k_fold_obj.split(x_train, y_train):
            if k_fold_count >= k_fold_use:
                break
            else:
                k_fold_count += 1
                            
                x_cv_train, y_cv_train = x_train[train_index], y_train[train_index]
                x_cv_test, y_cv_test = x_train[test_index], y_train[test_index]
                            
                curr_lor = LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, penalty=penalty, n_jobs=-1)
                curr_lor.fit(x_cv_train, y_cv_train)
                            
                y_cv_train_pred = curr_lor.predict_proba(x_cv_train)[:, 1]
                y_cv_test_pred = curr_lor.predict_proba(x_cv_test)[:, 1]
                            
                roc_auc_train = roc_auc_score(y_cv_train, y_cv_train_pred)
                roc_auc_test = roc_auc_score(y_cv_test, y_cv_test_pred)
                            
                C_list.append(C)
                l1_list.append(l1_ratio)
                t_list.append(t_name)
                smoothing_list.append(target_smoothing)
                solver_list.append(solver)
                penalty_list.append(penalty)
                pca_list.append(pca_threshold)
                train_score_list.append(roc_auc_train)
                test_score_list.append(roc_auc_test)
            
    print('Progress:')
    print('100%')
    print('-----')
    
    df_cv = pd.DataFrame(
        dict(
            C=C_list,
            L1=l1_list,
            T=t_list,
            Smooth=smoothing_list,
            Solver=solver_list,
            Penalty=penalty_list,
            PCA_Threshold=pca_list,
            Train_AUC=train_score_list,
            Test_AUC=test_score_list
        )
    )
    df_cv.to_csv('/kaggle/working/df_cv.csv', index=False)
            
else:
    df_cv = pd.read_csv('../input/cv-results-cat-challenge/df_cv.csv')


# The $10$ best hyperparameter combinations are:

# In[ ]:


df_cv.sort_values(by='Test_AUC', ascending=False).iloc[0:10, :]


# Grouping the cross-validation folds and taking the average:

# In[ ]:


df_cv_grouped = df_cv.groupby(['C', 'L1', 'T', 'Smooth', 'Solver', 'Penalty']).mean().sort_values(by='Test_AUC', ascending=False)
df_cv_grouped.iloc[0:10, :]


# We can check the distribution of the scores before grouping the AUC values per folds and after taking the mean. Before taking the average:

# In[ ]:


ggplot(df_cv, aes(x='Test_AUC')) + geom_histogram(bins = 50, fill = 'lightblue') +    ggtitle('CV Test AUC Distribution (Before Grouping)') + xlab('ROC-AUC') + ylab('Count - 50 Bins')


# After taking the mean:

# In[ ]:


ggplot(df_cv_grouped, aes(x='Test_AUC')) + geom_histogram(bins = 50, fill = 'lightgreen') +    ggtitle('CV Test AUC Distribution (After Grouping)') + xlab('ROC-AUC') + ylab('Count - 50 Bins')


# So, let's take the best $2$ models:

# In[ ]:


# C 	L1 	    T 	        Smooth 	Solver 	Penalty
# 0.1 	0.25 	maximum_te 	0.1 	saga 	elasticnet
# 1.0 	0.50 	minimum_te 	10.0 	saga 	elasticnet

opt_pipe_A = get_preprocessor(pass_features, [], avg_features + ordinal_features_low_count + nominal_features_low_count, 0.1)
opt_pipe_B = get_preprocessor(pass_features, ordinal_features_low_count + nominal_features_low_count, avg_features, 10.)

opt_pipe_A.fit(df_train.iloc[:, 1:-2], y_train)
opt_pipe_B.fit(df_train.iloc[:, 1:-2], y_train)

opt_model_A = LogisticRegression(C=0.1, l1_ratio=0.25, solver='saga', penalty='elasticnet')
opt_model_B = LogisticRegression(C=1.0, l1_ratio=0.50, solver='saga', penalty='elasticnet')

x_pipe_A = opt_pipe_A.transform(df_train.iloc[:, 1:-2])
x_pipe_B = opt_pipe_B.transform(df_train.iloc[:, 1:-2])

opt_model_A.fit(x_pipe_A, y_train)
opt_model_B.fit(x_pipe_B, y_train)


# In[ ]:


target_A = opt_model_A.predict_proba(opt_pipe_A.transform(df_test.iloc[:, 1:-1]))[:, 1]
target_B = opt_model_B.predict_proba(opt_pipe_B.transform(df_test.iloc[:,1:-1]))[:, 1]


# It's interesting to notice that the model $B$ (the second best model) let us get better results in the submission ($76.601 \%$)

# In[ ]:


pd.DataFrame({
    'id': df_test['id'],
    'target': target_A
}).to_csv('/kaggle/working/df_out_logistic_model_A.csv', index=False)

pd.DataFrame({
    'id': df_test['id'],
    'target': target_B
}).to_csv('/kaggle/working/df_out_logistic_model_B.csv', index=False)


# In out next step, we will try to make a Neural Network and combine its results with the predictions of the Logistic Regression.
# 
# # 4. Neural Network - Finding a Good Architecture

# In our Neural Network, we will use One Hot Encoding in every possible feature (except the binary ones, which are already encoded and the ordinal / nominal features that have many different values).
# 
# We can create a model with the necessary complexity to "decode" all the dummified features.

# In[ ]:


default_pipeline = get_preprocessor(pass_features=binary_features, 
                                    one_hot_features=nominal_features_low_count + ordinal_features_low_count, 
                                    avg_features=nominal_features_high_count + ordinal_features_high_count)

x_train = default_pipeline.fit_transform(df_train.iloc[:, 1:-2], y_train)
x_test = default_pipeline.transform(df_test.iloc[:, 1:-1])
x_train


# In[ ]:


x_train.shape


# Using the function Keras API we will create a "Wide and Deep" neural network to solve our problem.
# 
# ![Wide And Deep NN](https://www.researchgate.net/profile/Kaveh_Bastani/publication/328161216/figure/fig3/AS:679665219928064@1539056224036/Illustration-of-the-wide-and-deep-model-which-is-an-integration-of-wide-component-and.ppm)
# 
# We will not explicitly use embedding layers in our model but the principle is the same. Wide and Deep Neural Networks were originally proposed by Google in $2016$, to use in the APP Store recommendations system:

# In[ ]:


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

def get_wide_and_deep(use_montecarlo=False):

    inputs = Input(shape=(81,))

    deep = Dense(81, activation='elu')(inputs)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(40, activation='elu')(deep)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(20, activation='elu')(deep)
    deep = Dropout(0.5)(deep) if not use_montecarlo else Dropout(0.5)(deep, training=True)
    deep = Dense(10, activation='elu')(deep)

    deep_and_wide = Concatenate()([deep, inputs])
    deep_and_wide = BatchNormalization()(deep_and_wide)
    deep_and_wide = Dense(1, activation='sigmoid')(deep_and_wide)

    model_nn = Model(inputs=inputs, outputs=deep_and_wide)
    model_nn.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy', auc])
    
    return model_nn


# Printing the layers of our model:

# In[ ]:


model_nn = get_wide_and_deep()
model_nn.summary()


# We can even draw the layers of our Wide and Deep NN:

# In[ ]:


plot_model(model_nn, to_file='model.png', show_shapes=True, show_layer_names=True)


# The Dropout layers represent a regularization mechanism that trains our neural network after removing $X \%$ random weights of the training algorithm. The $X \%$ value in our model is $50 \%$. It's a smart mechanism that is able to create a robust neural network, that is capable to take reasonable predictions evn without some of its synaptic weights! 
# 
# The BatchNormalization layer standardizes the values by learning the variance and the means of previous batch values. It can help us to don't have explosive (positive or negative) values in out Neural Network.
# 
# Finally, the EarlyStopping callback will train our Neural Network while the validation score increases. We give $30$ chances to our model to get a higher score, otherwise, we just stop training. This parameter is called the Early Stopping "patience":

# In[ ]:


df_train['target'].mean()


# In[ ]:


train_idx.shape


# In[ ]:


es = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,verbose=1, mode='max', baseline=None, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_auc', factor=0.5,patience=3, min_lr=1e-6, mode='max', verbose=1)

n_folds = 10
sfk = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
pred_list, hist_list = [], []

fold_index = 1
for train_idx, val_idx in sfk.split(x_train, y_train):
    
    print('\n---\nFold Index\n---\n: ' + str(fold_index))
    fold_index += 1
    
    model_nn = get_wide_and_deep()
    history_nn = model_nn.fit(x_train[train_idx], y_train[train_idx],
                              validation_data = (x_train[val_idx], y_train[val_idx]),
                              callbacks=[es, rlr], 
                              epochs=100, batch_size=1024, 
                              class_weight={0: 0.2, 1: 0.8},
                              verbose=0)
    
    pred_list.append(model_nn.predict(x_test))
    hist_list.append(history_nn)


# Also, notice that the validation set is composed of $10 \%$ of the training dataset. Finally, predicting the Neural Network final result:

# In[ ]:


pred_list = [list(X) for X in pred_list]
pred_list_formatted = [np.array([Y[0] for Y in X]) for X in pred_list]
target_nn = list(np.mean(pred_list_formatted, axis=0))


# In[ ]:


out_nn = model_nn.predict(x_test)


# Let's plot the training and validations curves of our fitting process:

# In[ ]:


history_nn.history.keys()


# In[ ]:


history_nn.history.keys()


# In[ ]:


list_val = history_nn.history['val_loss'] +           history_nn.history['val_accuracy'] +           history_nn.history['val_auc'] +           history_nn.history['loss'] +           history_nn.history['accuracy'] +           history_nn.history['auc']

n_epochs = len(history_nn.history['val_loss'])

list_steps = 6 * (list(range(1, n_epochs + 1)))

list_metrics = (n_epochs * ['Loss']) + (n_epochs * ['Accuracy']) +               (n_epochs * ['AUC']) + (n_epochs * ['Loss']) +               (n_epochs * ['Accuracy']) + (n_epochs * ['AUC'])

list_kind = (n_epochs * ['Validation']) + (n_epochs * ['Validation']) +            (n_epochs * ['Validation']) + (n_epochs * ['Training']) +            (n_epochs * ['Training']) + (n_epochs * ['Training'])

df_nn_history = pd.DataFrame(dict(Step=list_steps, Value=list_val, Metric=list_metrics, Kind=list_kind))
df_nn_history.head()


# In[ ]:


ggplot(df_nn_history[df_nn_history['Step'] > 5], aes(x='Step', y='Value', colour='Kind')) +    geom_line(aes(group='Kind')) +    facet_grid('Metric ~ .', scales='free') +    ggtitle('Training / Validation Metrics') + geom_point(aes(shape = 'Kind'))


# In[ ]:


target_nn = [X[0] for X in out_nn.tolist()]
target_nn[1:5]


# In[ ]:


pd.DataFrame({
    'id': df_test['id'],
    'target': target_nn
}).to_csv('/kaggle/working/df_out_wide_and_deep.csv', index=False)


# We can also blend the models, and consider the average of the neural network output with the best logistic regression we found (model B):

# In[ ]:


pd.DataFrame({
    'id': df_test['id'],
    'target': (np.array(target_nn) + np.array(target_B)) / 2
}).to_csv('/kaggle/working/df_out_models_avg.csv', index=False)


# The best model is the last one: when we take the average of the best models and they have different predictions, then we create a better estimator, thanks to the central limit theorem:
# 
# ![Central Limit Theorem](https://i.stack.imgur.com/wPGzI.png)
# 
# We can imagine that each model is a "coin" with a good probability to take a $1$ (imagine that we we have a $0$ in one side and a $1$ on the other side). So, if the coins generate reasonably independent results and we take the average value that we get after flipping them, we will tend to get a normal distribution, with a low standard deviation and centered around a value that is near to one (closer than the expected value of the Bernouilli distribution of each individual coin!)
# 
# The theory justifies the better results :)
# 
# So, maybe, we can apply a **MonteCarlo Dropout** instead of removing the Dropout effect during the testing phase (Keras does it by default). With this approach, we convert our deterministic neural network into a probabilistic one and then we can create many different estimators and get the average predictions. Many different configurations with few weights will be considered:
# 
# ![Dropout Figure](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSz-CH9Jv9p3gJjn8pSF3EHebiRHUof84YOzExRI3QuCzqubwkn)

# Creating a model with the same weights with the MonteCarlo parameter equal to "True":

# In[ ]:


model_nn_mc = get_wide_and_deep(use_montecarlo=True)
model_nn_mc.set_weights(model_nn.get_weights())


# Running $100$ simulations:

# In[ ]:


predictions_list = []
n_montecarlo_sims = 100
for i in range(n_montecarlo_sims):
    if (i % 10) == 0:
        print('Monte Carlo Simulation - Iteration: ' + str(i + 1) + '/' + str(n_montecarlo_sims))
    predictions_list.append([X[0] for X in model_nn_mc.predict(x_test)])


# Taking the average:

# In[ ]:


predictions_list = [np.array(X) for X in predictions_list]
target_nn_mc = sum(predictions_list) / n_montecarlo_sims


# Saving the obtained model and the average of the MonteCarlo NN with the logistic regression:

# In[ ]:


pd.DataFrame({
    'id': df_test['id'],
    'target': target_nn_mc
}).to_csv('/kaggle/working/df_out_models_mc.csv', index=False)


# In[ ]:


pd.DataFrame({
    'id': df_test['id'],
    'target': (target_nn_mc + np.array(target_B)) / 2
}).to_csv('/kaggle/working/df_out_models_avg_mc.csv', index=False)


# We also have a small improvement after applying the Monte Carlo Dropout model and blending it with the Logistic Regression.

# # 5. Final Conclusions
# 
# So, here I can list what I learned with this challenge:
# 
# 1. We can use different encoding schemes to tranform our categorical features in numbers that can be processed by machine learning models. The best encoding scheme depends not only on the number of different features of the model, but also on the kind of estimator that we will use.
# 2. Sometimes, simple models as Logistic Regressions may already give us great results and additional efforts are, sometimes, useless. I mean: would it by really important to improve the score in $0.5 \%$ after applying a huge effort in a complex neural network? Well, it depends on the kind of business or application.
# 3. Wide and Deep neural networks may be seen as an optimal linear combination between a simple logistic regression and a deep neural network. So, it's a mix between a complex model and an optimal one. It was interesting to use it here.
# 4. When working with neural networks, we should consider alternative regularization mechanisms such as Dropout layers or BatchNormalization layers to improve the consistency and prediction power of the estimator.
# 5. The EarlyStopping is an elegant way to avoid overfitting. But I got better results when I increased the patience from $10$ to $30$. Why? If the convergence of the weights is slow, we can wait more in order to obtain a better estimator.
# 6. Blending independent models may help us to reach better results thanks to the Central Limit Theorem :)
# 
# If you read until this point, thanks a lot! Take this ASCII lucky cat for you!
# 
#          ,_         _,
#          |\\.-"""-.//|
#          \`         `/
#         /    _   _    \
#         |    a _ a    |
#         '.=    Y    =.'
#           >._  ^  _.<
#          /   `````   \
#          )           (
#         ,(           ),
#        / )   /   \   ( \
#        ) (   )   (   ) (
#        ( )   (   )   ( )
#        )_(   )   (   )_(-.._
#       (  )_  (._.)  _(  )_, `\
#        ``(   )   (   )`` .' .'
#     jgs   ```     ```   ( (`
#                          '-'
# 
