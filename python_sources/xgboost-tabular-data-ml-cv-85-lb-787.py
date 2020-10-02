#!/usr/bin/env python
# coding: utf-8

# # Melanoma Prediction Using Tabular Data
# In this solely based on feature engineering and using Machine learning Model to detect Skin Cancer. No Deep Learning Model has been used here.
# 
# **Special Thank to [oliver](https://www.kaggle.com/ogrellier) for sharing some key features in the data**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from glob import glob
from tqdm import tqdm
import seaborn as sns
sns.set(style = 'dark')
import matplotlib.pyplot as plt


# In[ ]:


train_files_dir = glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/*')
test_files_dir = glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/*')


# In[ ]:


train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Label Encoding

# ## Sex

# In[ ]:


train_df['sex'].fillna('unkown',inplace = True) # missing value


# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[ ]:


train_df['sex_enc'] = enc.fit_transform(train_df.sex.astype('str'))
test_df['sex_enc'] = enc.transform(test_df.sex.astype('str'))


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x = 'sex', hue = 'target', data = train_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## Anatom_site_general_challenge

# In[ ]:


test_df.anatom_site_general_challenge = test_df.anatom_site_general_challenge.fillna('unknown')
train_df.anatom_site_general_challenge = train_df.anatom_site_general_challenge.fillna('unknown')


# In[ ]:


train_df['anatom_enc']= enc.fit_transform(train_df.anatom_site_general_challenge.astype('str'))
test_df['anatom_enc']= enc.transform(test_df.anatom_site_general_challenge.astype('str'))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Age

# In[ ]:


train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].mode().values[0])
test_df['age_approx']  = test_df['age_approx'].fillna(test_df['age_approx'].mode().values[0]) # Test data doesn't have any NaN in age_approx


# In[ ]:


train_df['age_enc']= enc.fit_transform(train_df['age_approx'].astype('str'))
test_df['age_enc']= enc.transform(test_df['age_approx'].astype('str'))


# In[ ]:


plt.figure(figsize = (20,6))
sns.countplot(x = 'age_approx', hue = 'target', data = train_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Images Per Patient

# In[ ]:


train_df['n_images'] = train_df.patient_id.map(train_df.groupby(['patient_id']).image_name.count())
test_df['n_images'] = test_df.patient_id.map(test_df.groupby(['patient_id']).image_name.count())


# # Categorize Number of Images Per Patient

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
categorize = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
train_df['n_images_enc'] = categorize.fit_transform(train_df['n_images'].values.reshape(-1, 1)).astype(int).squeeze()
test_df['n_images_enc'] = categorize.transform(test_df['n_images'].values.reshape(-1, 1)).astype(int).squeeze()


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x = 'n_images_enc', hue = 'target', data = train_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Image Size 

# In[ ]:


train_images = train_df['image_name'].values
train_sizes = np.zeros(train_images.shape[0])
for i, img_path in enumerate(tqdm(train_images)):
    train_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))
    
train_df['image_size'] = train_sizes


test_images = test_df['image_name'].values
test_sizes = np.zeros(test_images.shape[0])
for i, img_path in enumerate(tqdm(test_images)):
    test_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))
    
test_df['image_size'] = test_sizes


# # Scaling Image Size

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scale = MinMaxScaler()
train_df['image_size_scaled'] = scale.fit_transform(train_df['image_size'].values.reshape(-1, 1))
test_df['image_size_scaled'] = scale.transform(test_df['image_size'].values.reshape(-1, 1))


# # Categorize Image Size

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
categorize = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
train_df['image_size_enc'] = categorize.fit_transform(train_df.image_size_scaled.values.reshape(-1, 1)).astype(int).squeeze()
test_df['image_size_enc'] = categorize.transform(test_df.image_size_scaled.values.reshape(-1, 1)).astype(int).squeeze()


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x = 'image_size_enc', hue = 'target', data = train_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Mean Color(Used previously saved data)

# In[ ]:


train_mean_color = pd.read_csv('/kaggle/input/mean-color-isic2020/train_color.csv')
test_mean_color = pd.read_csv('/kaggle/input/mean-color-isic2020/test_color.csv')


# In[ ]:


train_df['mean_color'] = train_mean_color.values
test_df['mean_color'] = test_mean_color.values


# # Categorize Mean Color

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
categorize = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
train_df['mean_color_enc'] = categorize.fit_transform(train_df['mean_color'].values.reshape(-1, 1)).astype(int).squeeze()
test_df['mean_color_enc'] = categorize.transform(test_df['mean_color'].values.reshape(-1, 1)).astype(int).squeeze()


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x = 'mean_color_enc', hue = 'target', data = train_df)


# # Min-Max age of Patient

# In[ ]:


train_df['age_id_min']  = train_df['patient_id'].map(train_df.groupby(['patient_id']).age_approx.min())
train_df['age_id_max']  = train_df['patient_id'].map(train_df.groupby(['patient_id']).age_approx.max())

test_df['age_id_min']  = test_df['patient_id'].map(test_df.groupby(['patient_id']).age_approx.min())
test_df['age_id_max']  = test_df['patient_id'].map(test_df.groupby(['patient_id']).age_approx.max())


# # Mean Encoding

# ## Plotting Barplot with number

# In[ ]:


def show_bar_plot(df, figsize = (12,6)):
 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    def show_values_on_bars(axs, h_v="v", space=0.4, v_space = 0.02, figsize = (12,6)):
        def _show_on_single_plot(ax):
            if h_v == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height()+ v_space
                    value = float(p.get_height())
                    ax.text(_x, _y, f'{value:.1f}', ha="center") 
            elif h_v == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height()+ v_space
                    value = int(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)
            

#     fig = plt.gcf()
#     fig.set_size_inches(12, 8)
    plt.figure(figsize = figsize)
    sns.set()
    plt.title('Probability')

    prob = df*100
    pal = sns.color_palette(palette='Blues_r', n_colors=len(prob))
    rank = prob.values.argsort().argsort() 
    #g=sns.barplot(x='day',y='tip',data=groupedvalues, palette=np.array(pal[::-1])[rank])
    br = sns.barplot(prob.index, prob.values, palette=np.array(pal[::-1])[rank])
    show_values_on_bars(br, "v", .50)
    plt.show()          


# In[ ]:


def kdeplot( df,col_name, figsize = (12,6)):

    plt.figure(figsize = figsize)
    sns.kdeplot(df[col_name][df.target==0], shade = True, color = 'b', label = '0')
    sns.kdeplot(df[col_name][df.target==1], shade = True, color = 'r', label = '1')
    plt.show()


# ## Age

# In[ ]:


train_df['age_approx_mean_enc'] = train_df['age_approx'].map(train_df.groupby(['age_approx'])['target'].mean())
test_df['age_approx_mean_enc'] = test_df['age_approx'].map(train_df.groupby(['age_approx'])['target'].mean())


# ## Probability of melanoma with respect to Age

# In[ ]:


show_bar_plot(train_df.groupby(['age_approx'])['target'].mean(), (20,10))


# ## Before Mean Encoding

# In[ ]:


col_name = 'age_approx'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After Mean Encoding

# In[ ]:


col_name = 'age_approx_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## Sex

# In[ ]:


train_df['sex_mean_enc'] = train_df.sex_enc.map(train_df.groupby(['sex_enc'])['target'].mean())
test_df['sex_mean_enc'] = test_df.sex_enc.map(train_df.groupby(['sex_enc'])['target'].mean())


# ## Probability of melanoma with respect to Sex

# In[ ]:


show_bar_plot(train_df.groupby(['sex'])['target'].mean())


# ## Before Mean Encoding

# In[ ]:


col_name = 'sex_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After Mean Encoding

# In[ ]:


col_name = 'sex_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## n_images

# In[ ]:


train_df['n_images_mean_enc'] = train_df['n_images_enc'].map(train_df.groupby(['n_images_enc'])['target'].mean())
test_df['n_images_mean_enc'] = test_df['n_images_enc'].map(train_df.groupby(['n_images_enc'])['target'].mean())


# ## Probability of melanoma with respect to Number of Image per Patient

# In[ ]:


show_bar_plot(train_df.groupby(['n_images_enc'])['target'].mean())


# In[ ]:


col_name = 'n_images'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After Encoding

# In[ ]:


col_name = 'n_images_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After Mean Encoding

# In[ ]:


col_name = 'n_images_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## Image Size

# In[ ]:


train_df['image_size_mean_enc'] = train_df['image_size_enc'].map(train_df.groupby(['image_size_enc'])['target'].mean())
test_df['image_size_mean_enc'] = test_df['image_size_enc'].map(train_df.groupby(['image_size_enc'])['target'].mean())


# ## Probability of melanoma with respect to Image Size

# In[ ]:


show_bar_plot(train_df.groupby(['image_size_enc'])['target'].mean())


# ## After Encoding

# In[ ]:


col_name = 'image_size'
kdeplot(train_df,col_name, figsize = (16,8))


# In[ ]:


col_name = 'image_size_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# In[ ]:


col_name = 'image_size_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## Anatom General Challenge

# In[ ]:


train_df['anatom_mean_enc'] = train_df['anatom_enc'].map(train_df.groupby(['anatom_enc'])['target'].mean())
test_df['anatom_mean_enc'] = test_df['anatom_enc'].map(train_df.groupby(['anatom_enc'])['target'].mean())


# ## Probability of melanoma with respect to Anatom General Challege

# In[ ]:


show_bar_plot(train_df.groupby(['anatom_enc'])['target'].mean())


# In[ ]:


col_name = 'anatom_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# In[ ]:


col_name = 'anatom_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# # Mean Color

# In[ ]:


train_df['mean_color_mean_enc'] = train_df['mean_color_enc'].map(train_df.groupby(['mean_color_enc'])['target'].mean())
test_df['mean_color_mean_enc'] = test_df['mean_color_enc'].map(train_df.groupby(['mean_color_enc'])['target'].mean())


# ## Probability of melanoma with respect to Mean Color

# In[ ]:


show_bar_plot(train_df.groupby(['mean_color_enc'])['target'].mean())


# ## Before encoding

# In[ ]:


col_name = 'mean_color'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After encoding

# In[ ]:


col_name = 'mean_color_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# ## After Mean Encoding

# In[ ]:


col_name = 'mean_color_mean_enc'
kdeplot(train_df,col_name, figsize = (16,8))


# # Mean Encoding is Angel or Devil?
# 
# As you can see mean encoding has some interesting effect on data. But there is a good chnace we will end up overfitting because we're depending on target of train data. What if train and test data have different distribution??? 

# # Correlation Matrix
# We can extract some interesting features from Corr Matrix. I leave that to the reader to find out some tricky features from Corr Matrix. Let me know you find one. 
# 
# **Spoiler Alert:** You can easily get **.80** just playing with these features

# In[ ]:


corr = train_df.corr(method = 'pearson')
corr = corr.abs()
corr.style.background_gradient(cmap='inferno')


# In[ ]:


# plt.figure(figsize = (20,20))
# sns.heatmap(corr, annot = True)


# In[ ]:


corr = test_df.corr(method = 'pearson')
corr = corr.abs()
corr.style.background_gradient(cmap='inferno')


# # Selecting Features

# In[ ]:


test_df.columns


# # I get rid of some features for best LB score

# In[ ]:


features = [
            'age_approx',
#             'age_enc',
#             'age_approx_mean_enc',
            'age_id_min',
            'age_id_max',
            'sex_enc',
#             'sex_mean_enc',
            'anatom_enc',
#             'anatom_mean_enc',
            'n_images',
#             'n_images_mean_enc',
#             'n_images_enc',
            'image_size_scaled',
#             'image_size_enc',
#             'image_size_mean_enc',
            'mean_color',
#             'mean_color_enc', 
#             'mean_color_mean_enc'
           ]


# In[ ]:


X = train_df[features]
y = train_df['target']

X_test = test_df[features]


# In[ ]:


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# # Training Xgboost
# ## Parameters were tunned using Gridsearch

# In[ ]:


model = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8, gamma=1, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.002, max_delta_step=0, max_depth=10,
             min_child_weight=1, missing=None, monotone_constraints=None,
             n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1,
             objective='binary:logistic', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8,
             tree_method=None, validate_parameters=False, verbosity=None)

kfold = StratifiedKFold(n_splits=5, random_state=1001, shuffle=True)
cv_results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc', verbose = 3)
cv_results.mean()


# In[ ]:


xgb = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8, gamma=1, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.002, max_delta_step=0, max_depth=10,
             min_child_weight=1, missing=None, monotone_constraints=None,
             n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1,
             objective='binary:logistic', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8,
             tree_method=None, validate_parameters=False, verbosity=None)

xgb.fit(X,y)
pred_xgb = xgb.predict(X_test)


# # Plot: Feature Importance

# In[ ]:


feature_important = xgb.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
plt.figure(figsize= (12,10))
sns.barplot(x = data.score , y = data.index, orient = 'h', palette = 'Blues_r')


# # Dimension Reduction (PCA)

# In[ ]:


from sklearn.decomposition import PCA

n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame({'x_pca_0':X_pca[:,0],
             'x_pca_1':X_pca[:,1],
             'y':y})


# In[ ]:


plt.figure(figsize = (10,10))
sns.scatterplot(
    x="x_pca_0", y="x_pca_1",
    hue="y",
    data=pca_df,
    legend="full",
    alpha=0.9
)


# # Prediction

# In[ ]:


sub = pd.DataFrame({'image_name':test_df.image_name.values,
                    'target':pred_xgb})
sub.to_csv('submission.csv',index = False)


# Though it is difficult to predict location of melanoma class from  plot but we look carefully we'll be able to notice that there are some regions where there is no melanoma class at all.

# There might be two possible reasons behind LB score differing from the CV score .
# 1. Only 30% test data is used in Public LB
# 2. Our Model has been ovefitted
# 
# Please let me know if I can improve my results. 
# ## Thank You Very Much
