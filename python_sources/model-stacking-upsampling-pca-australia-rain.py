#!/usr/bin/env python
# coding: utf-8

# # Predicting If it Rains Tomorrow in Australia
# 
# ### Rich Caputo
# 
# In this project, we will be analyzing rainfall weather in Australia and building a binary classification algorithm to predict whether or not it will rain tomorrow. This dataset is relatively clean but will require some feature engineering to extract the full essense of the data. Our hope is to build a robust and easy to use model using sklearn pipelines. All data transformations will be fit to the training set to avoid data snooping.

# In[ ]:


# Get essential packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Sci-kit Learn Essentials
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Silence warnings
import warnings
warnings.filterwarnings('ignore')


# ### Loading and Exploring Data

# In[ ]:


# Need to drop RISK_MM due to data leakage
# Set date to index and sort
df = pd.read_csv('../input/weatherAUS.csv', parse_dates=['Date'], index_col=0).drop('RISK_MM', axis=1).sort_index()
df.head()


# In[ ]:


df.info()


# As we can see, there is a fair bit of null entries in our data. We will need to impute these. Let's examine how long our data set spans.

# In[ ]:


df.index.min(), df.index.max()


# In[ ]:


df.index.max() - df.index.min()


# That's strange: it seems that although we have 3524 days of data, we have 142193 total entries in our data. Let's examine this.

# In[ ]:


df.index.value_counts()


# In[ ]:


df.loc['2013-12-22']


# Aha: Although we have entries with the same date occasionally, this is due to data representing different locations in Australia. Let's examine temperatures in different regions!

# In[ ]:


# Plot Min/Max Temps by Location
df.groupby('Location')['MinTemp'].plot(figsize=(17, 8))
plt.title('Min Temperature in Australia Colored by Region')
plt.xlabel('Date')
plt.ylabel('Temperature (Celsius)')
plt.show()


# Although we don't have space to label the legend, this plot gives us an overall sense of how temperature fluctuates over time in Australia. Perhaps a good way to capture this seasonality would be to add a Month column to the data (we will do this during the modeling phase).
# 
# Let's now examine how our features correlate.
# 

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.corr(),
            linewidths=0.1,
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
plt.show()


# There are some obvious correlations here, such as max/min temp vs. temp9am and temp3pm. Also notice how cloud and sunshine variables are negatively correlated. No suprise there. Interestingly, it seems temp and humidity are inversely correlated. So is sunshine and humidity. Also, temperature and pressure are slightly negatively correlated. Since the models we are using are non-linear, we don't have to worry to much about removing co-linear features as our models will capture these relationships.

# An interesting thing to do would be to visualize how this data varies by location in more ways than just temperature. To do this, we will perform PCA to project the data into 2-dimmensions and visualize data based on location. For fitting purposes, we will impute NaN values using a custom DataFrameImputer, which will fill string NaN's to the most frequent value while gitting numeric NaN's to the current mean. This is easy for now since we are using the whole dataset, but will require some modifications for model fitting with a train and test set.

# In[ ]:


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        """
    
    def fit(self, X, y=None):
        """ Imputes categoricals with 'O' and numericals with the mean. """
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        """ Executes transformation from fit. """
        return X.fillna(self.fill)

    
df_nl = df.drop('Location', axis=1)
cat_cols = list(df_nl.select_dtypes(include=object, exclude=None).columns)

pipe = make_pipeline(
    DataFrameImputer(),
    StandardScaler(),
    PCA(n_components=3)   
)

X_red = pipe.fit_transform(pd.get_dummies(df_nl))


# In[ ]:


X_red


# In[ ]:


# Plot 2D PCA projection
cmap = plt.get_cmap('jet', 20)
cmap.set_under('gray')
fig, ax = plt.subplots(figsize=(20, 20))
cax = ax.scatter(X_red[:, 0], X_red[:, 1], 
                 c=df.Location.astype('category').cat.codes, 
                 s=10, 
                 cmap=cmap, 
                 alpha=0.7)
fig.colorbar(cax, extend='min')
plt.title('2D PCA Projection of Data Colored by Location')
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

# Plot 3D PCA Projection
cmap = plt.get_cmap('jet', 20)
cmap.set_under('gray')
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(X_red[:, 0], X_red[:, 1], X_red[:, 2], 
                 c=df.Location.astype('category').cat.codes, 
                 s=10, 
                 cmap=cmap, 
                 alpha=0.7)
fig.colorbar(cax, extend='min')
plt.title('3D PCA Projection of Data Colored by Location')
plt.show()


# ### Data Preprocessing

# Although these two plots above look like a random confetti party, they reveal some interesting information about the data. It seems that different locations have different projections as we can see from the coloring - certain colors seem to cluster together! This tells us that location will likely be an important feature when it comes to building our model.
# 
# Recall that our goal is to predict the RainTomorrow variable. For this task, we will split our data into training and test sets, fit preprocessing steps to the training data, and finally fit a few ML algorithms to the training data. Since in practice our model would be used in the future, we need to similulate looking ahead in time. In order to do this, we will not shuffle our training and test sets. This will ensure that our test set encompasses most recent data.

# In[ ]:


# Get X, y
X, y = df.drop('RainTomorrow', axis=1), df.RainTomorrow

# Recast y as int.  1: 'Yes', 0: 'No'
y = (y == 'Yes').astype('int64')


# Let's check the distribution of our target variable.

# In[ ]:


y.hist()
plt.show()


# In[ ]:


y.value_counts()/len(y)


# Our target variable is imbalanced. By predicting that it will not rain every day, we will have a model that is already 77.5% accurate. A popular package for this problem is imbalanced learn. We will use over sampling to balance the distribution of our target during our second part of the modelling phase, i.e. stacking. Note that this doesn't always lead to favorable results. For now, we will do some feature engineering in hopes of improving our model. This will include adding deltas, i.e. how much a variable changes over the course of a day, and seasonality, i.e. what month it is.

# In[ ]:


# Add seasonality
X['Month'] = X.index.month_name()

# Get initial columns
cat_cols = list(X.select_dtypes(include=object).columns)
num_cols = [c for c in X.columns if c not in cat_cols]

# Get intraday columns
am_cols = sorted([c for c in num_cols if c.endswith('9am')])
pm_cols = sorted([c for c in num_cols if c.endswith('3pm')])

# Add intraday delta
for am_col, pm_col in zip(am_cols, pm_cols):
    X[am_col[:-3] + '_delta'] = X[pm_col] - X[am_col]
    
# Add max temp - min temp delta
X['MinMaxTemp_delta'] = X.MinTemp - X.MaxTemp
    
# Get train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()


# Next step is to fit preprocessing steps to our data. We fit separate pipelines to categorical data and numerical data as we need to perform different preprocessing steps based on the data type. In the case of numeric data, we will be imputing missing values with the median for each column and then fitting a standard scaler. For categorical data, we will be imputing missing values with the most frequent value and then one-hot-encoding. Finally, we will union these two pipelines.

# In[ ]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection. """
    
    def __init__(self, columns=[]):
        """ Get selected columns. """
        self.columns = columns
        
    def transform(self, X):
        """ Returns df with selected columns. """
        return X[self.columns].copy()
    
    def fit(self, X, y=None):
        """ Do nothing operation. """
        return self


# -- Get Pipelines --

# Get categoricals and numericals
cat_cols = list(X.select_dtypes(include=object).columns)
num_cols = [c for c in X.columns if c not in cat_cols]

# Fit numerical pipeline
num_pipeline = make_pipeline(
    DataFrameSelector(num_cols),
    SimpleImputer(strategy='median'),
    StandardScaler()
)

# Fit categorical pipeline
cat_pipeline = make_pipeline(
    DataFrameSelector(cat_cols),
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse=False)
)

# Union pipelines
full_preproc = FeatureUnion(transformer_list=[
    ("cat_pipeline", cat_pipeline),
    ("num_pipeline", num_pipeline)
])

# (Optional) Reduce dimension to contain nearly all explained variance
# This will lower memory consumption and speed up training time
# at a small performance cost
red_preproc = make_pipeline(
    full_preproc,
    # Keep 95% of explained variance
    PCA(0.95)
)


# In[ ]:


X_train = full_preproc.fit_transform(X_train)
X_train.shape


# In[ ]:


X_test = full_preproc.transform(X_test)
X_test.shape


# ## Model Building
# 
# Our data is now fully preprocessed and we are ready to fit a model. To gain optimal performance, will be using model stacking. This means fitting a bunch of classifiers and using their predictions as input to a strong classifier for final predictions. The reason we do this is because it is often the case that different machine learning models find different features more important when making classification decisions. When we combine each of these learners, however, we will have a model that recognizes each feature with stronger importance.
# 
# The models we will be using as 'weak' learners are LogisticRegression, GaussianNB, RandomForestClassifier, AdaBoostClassifier, SGDClassifier, and GradientBoostingClassifier. These are robust algorithms each with different strengths, so our hope is that they will perform even better when aggregated. We will then use the (out-of-fold) predictions from these models to fit a strong XGBoostClassifier. We will only use grid search to optimize the performance of our final classifier for the sake of time. We will also use modified code from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python.

# In[ ]:


from pprint import pprint
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    # Use log loss so we can get probability predictions
    'SGDClassifier': SGDClassifier(loss='log'),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'GaussianNB': GaussianNB(),
}

pprint(models)


# In total, we have 6 weak learners. We will use their default hyperparameter settings and only worry about optimizing our final model. Now, we need to fit these learners to the training data and create a new level 2 training set where each column represents the predictions Y_A of model A. We will need to genereate these predictions "out-of-fold," meaning that we will divide our data into K-folds and for each fold fit the model on the other K-1 folds and predict on the hold-out fold. This will avoid generating leakage for in final model. For more information, see this article on model stacking: https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html.
# 
# We will use a function to generate out-of-fold predicitions.

# In[ ]:


# Some useful parameters which will come in handy later on
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)

def get_oof(clf, x_train=X_train, y_train=y_train, x_test=X_test):
    """ Get's out of fold predictions for a classifier. 
        Credit: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
        Slightly modified to handle sklearn 0.20.3
    """
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)
        # Use predict_proba and get confidence of model predicting 1 = 'Yes'
        oof_train[test_index] = clf.predict_proba(x_te)[:, 1]
        # For test set, we use each of the 5 folds
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]
    # and average predictions
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


# Next, we need to wrap a new function to generate a dataset of out-of-fold predictions for each model and concatenate them. This is going to take a while since we are cross validating 6 models 5 times each on a relatively large dataset. Note that instead of predicting the outcome variable, we will be using the probability that the model predicts 'Yes' to rain tomorrow. This will allow our stacked model to learn from the confidence of it's input models.

# In[ ]:


def build_pred_data(clf_dict=models):
    """ Get's oof predictions for every classifier in a dictionary of models. """
    X_train, X_test = pd.DataFrame(), pd.DataFrame()
    
    # Columns will be model predictions 
    # labeled according to model
    for key, clf in clf_dict.items():
        X_train[key], X_test[key] = get_oof(clf)
        
    # Lastly, MinMax Scale to avoid bias
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train, X_test

X_train_f, X_test_f = build_pred_data() 


# In[ ]:


# Examine train data 
X_train_f.head()


# In[ ]:


# Examine test data
X_test_f.head()


# Great! Now we are ready to train our final model. We will be using XGBoost and performing a grid search for the best hyperparameters. Before we do that, let's examine how our base learner model predictions correlate. We would prefer that models have lower correlation, but it is unlikely we will see very low correlation since our base learners are already quite robust.

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Model Probabilities', y=1.05, size=15)
sns.heatmap(X_train_f.corr(), 
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
plt.show()


# As we can see, our models have high correlation overall, but certain models, such as GaussianNB and RandomForestClassifier, are slightly less correlated than the rest. GaussianNB fits based on an independence assumption, and it is clear that many of our features are dependent. Our hope is that having slightly less correlated models will improve our final model. Let's visualize what our training data looks like in 2D and overlay y_train values.

# In[ ]:


def plot_2d_space(X_in=X_train_f, y=y_train, upsample=False):
    X = PCA(n_components=2).fit_transform(X_in)
    plt.figure(figsize=(15, 8))
    plt.scatter(X[(y == 1), 0], X[(y == 1), 1], 
                c='b', 
                label='Yes Rain Tomorrow',
                alpha=0.1,
                s=20)
    plt.scatter(X[(y == 0), 0], X[(y == 0), 1], 
                c='r', 
                label='No Rain Tomorrow',
                alpha=0.1,
                s=20)
    t = 'Model Probability PCA Projection 2D'
    if upsample: t += ' - Upsample'
    plt.title(t)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
plot_2d_space()


# It is clear that our data has a lot of overlap in 2D, but that there are concentrations of classes in the upper right and lower left hand corners respectively. This plot shows that our base learners did a good job distinguishing the two classes since our class clusters seem more seperable in a higher dimension. Let's now perform upsampling on our predictions dataset and compare with the above plot. Since our data is numeric, we will use SMOTE to upsample our rainy target variables. SMOTE re-samples based on spatial proximity and is thus a powerful upsampling technique.

# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_train_f_sm, y_train_sm = smote.fit_sample(X_train_f, y_train)
# Ensure columns are homogeneous - imblearn recasts as numpy array
X_train_f_sm = pd.DataFrame(X_train_f_sm, columns=X_train_f.columns)


# In[ ]:


plt.hist(y_train_sm)
plt.show()


# In[ ]:


plot_2d_space(X_train_f_sm, y_train_sm, upsample=True)


# We see that our SMOTE fit data has a higher concentration of 'Yes' target variables. Our hope is that this upsampled space will improve model performance by feeding in a balanced target variable space. However, there is risk associated with upsampling since it is synthesizing new data and may lead to overfitting noise as well as yielding an 'unnatural' frequency of target occurence. Therefore, we will fit stacked models to both data sets and compare results.

# For scoring we will use F1-Score. As mentioned before, accuracy is a tricky metric for this dataset since it is imbalanced. Although we resampled the training set, the test set is still imbalanced, so a model that always predicts no rain tomorrow will be quite accurate. Therefore we will measure performance with F1-Score - this is a combination of precision and recall and will tell us more about how well the model is performing on each class vs. on the entire dataset.
# 
# In this particular dataset, it would be a personal preference to have a model with high recall, i.e. low false negative rate. This is because I would rather be told that it would rain tomorrow and be pleasantly surprised by sunshine vs. the other way around. By cross-validating on F1-score vs. pure recall however, we will build a model that has better overall performance.

# ## Non-Upsampled Model

# In[ ]:


def get_cv_model(X, y):
    """ Given a dataset, performs cross-validation on an XGBoost model 
        and returns best one. Scoring: F1-Score. """
    # A parameter grid for XGBoost
    params = {
            'min_child_weight': [6, 8, 10, 12],
            'gamma': [0.75, 1, 1.25, 1.5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }
    # Get model
    clf_xgb = XGBClassifier(learning_rate=0.1, 
                            n_estimators=500, 
                            objective='binary:logistic',
                            silent=True, 
                            nthread=1)
    # Get best hyperparameters
    random_search = RandomizedSearchCV(clf_xgb, 
                                       param_distributions=params, 
                                       n_iter=8, 
                                       scoring='f1', 
                                       n_jobs=-1, cv=5, 
                                       verbose=3, random_state=115)
    random_search.fit(X, y)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best F1 score')
    print(random_search.best_score_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    return random_search


# In[ ]:


# Get non-upsampled model
random_search = get_cv_model(X_train_f, y_train)


# In[ ]:


# Get final scores
def final_scores(model=random_search, X_test=X_test_f, y_test=y_test, upsample=False, model_name='Stacked XGBClassifier'):
    y_pred = model.predict(X_test)
    if upsample: print('-- Upsampled --')
    print(f'--- {model_name} Results ---')
    print('Test F1 Score: ', f1_score(y_test, y_pred)) 
    print('\n Classification Report: ')
    print(classification_report(y_test, y_pred))
    print('\n Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred))
    print('\n Test Accuracy Score: ', accuracy_score(y_test, y_pred))


# In[ ]:


final_scores()


# In[ ]:


from xgboost import plot_importance

# Plot feature importance
f = plot_importance(random_search.best_estimator_)


# ## Upsampled Model

# In[ ]:


random_search_u = get_cv_model(X_train_f_sm, y_train_sm)


# In[ ]:


final_scores(model=random_search_u, upsample=True)


# In[ ]:


# Plot feature importance
f_up = plot_importance(random_search_u.best_estimator_)


# ## Base Learners

# In[ ]:


# Compare to Base Learners
for key, model in models.items():
    model.fit(X_train, y_train)
    final_scores(model=model, X_test=X_test, model_name=key)
    print('\n\n')


# # Conclusion

# It seems that our non-upsampled stacked model outperformed all of our base learners in F1-Score ans accuracy. SGDClassifier had higher recall than the non-upsampled stacked model, but lower accuracy. Our stacked model also had the highest accuracy.
# 
# Our upsampled stacked model had some interesting results. Although it was by far the lowest in accuracy, it had a very high recall. Essentially, upsampling caused the model to predict rain far more often than the non-upsampled models. This came at the cost of predicting rain a lot when it wasn't going to happen. Which model to use depends greatly on the circumstances. For example, a meteorologist might prefer to be correct most often, desiring a more accurate model. A single person may want to enjoy surprise sunshine, like myself. 
# 
# If I had to choose a single model to keep working with, I would likely pick the non-upsampled stacked model. This model had high accuracy and f1-score, making it reliable and robust. Although we have shown how accuracy can be a dangerous metric, it is also the easiest to understand and explain. In the case of predicting something non-critical such as rainfall, it is safe to not worry so much about recall. If we were predicting something rare, yet dangerous, like fraud or cancer, we would definitely want to use upsampling to improve recall.

# # Next Steps
# 
# Recall that we did not optimize the hyperparameters of our base learners. This would most likely lead to overall improved performance at the cost of time. Another optimization could be using more base learners, again at the cost of time.
# 
# Furthermore, we could attempt to use other metrics to measure performance, such as roc-auc.

# ## Thank you!
