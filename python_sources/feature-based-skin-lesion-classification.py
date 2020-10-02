#!/usr/bin/env python
# coding: utf-8

# # Goal
# The goal is to make a simple model that can go from an image (taken with a smartphone) to a prediction of how likely different allergens are to be present in the food. It could be part of a helpful app for people trying to avoid foods they might be allergic to.
# 
# ## Setup
# We basically take the precomputed color features and build simple models in order to determine if the food contains any of the 8 different allergens identified [here](https://www.kaggle.com/kmader/ingredients-to-allergies-mapping/). We try to create a balanced training group and a realistic validation group to know if the model is learning anything useful

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import os
from skimage.io import imread as imread
from skimage.util import montage
from PIL import Image
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
from skimage.color import label2rgb


# In[ ]:


dx_name_dict = {
    'nv': 'melanocytic nevi',
    'mel': 'melanoma',
    'bcc': 'basal cell carcinoma',
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'vasc': 'vascular lesions',
    'bkl': 'benign keratosis-like',
    'df': 'dermatofibroma'
}
dx_name_id_dict = {id: name for id, name in enumerate(dx_name_dict.values())}


# ## Read in the Color Features

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, categories="auto")

color_file = Path('..') / 'input' /  'skin-images-to-features' / 'color_features.json'
color_feat_df = pd.read_json(color_file)
color_feat_df['dx_vec'] = [x for x in ohe.fit_transform(color_feat_df['dx_id'].values.reshape(-1, 1))]
color_feat_df.sample(2)


# Split up the groups so we can validate our model on something besides the direct training data

# In[ ]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(color_feat_df, 
                 test_size = 0.2, 
                 random_state=2019,
                  # hack to make stratification work                  
                 stratify = color_feat_df['dx_name'])
train_df.reset_index(inplace=True)
valid_df.reset_index(inplace=True)
print(train_df.shape[0], 'training images')
print(valid_df.shape[0], 'validation images')


# In[ ]:


train_x_vec = np.stack(train_df['color_features'].values, 0)
train_y_vec = np.stack(train_df['dx_vec'], 0)
print(train_x_vec.shape, '->', train_y_vec.shape)
valid_x_vec = np.stack(valid_df['color_features'].values, 0)
valid_y_vec = np.stack(valid_df['dx_vec'], 0)
print(valid_x_vec.shape, '->', valid_y_vec.shape)


# # Display Results Nicely
# We want to have code to display our results nicely so we can see what worked well and what didn't

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
def show_model_results(in_model, use_split='valid', plot_type='swarm'):
    if use_split=='valid':
        x_vec = valid_x_vec
        y_vec = valid_y_vec
        example_df = valid_df
    elif use_split=='train':
        x_vec = train_x_vec
        y_vec = train_y_vec
        example_df = train_df
    else:
        raise ValueError('Unknown split: {}'.format(use_split))
    
    valid_pred = in_model.predict(x_vec)
    fig, m_axs = plt.subplots(4, 2, figsize=(20, 40))
    all_rows = []
    ax1 = m_axs[0,0]
    print(y_vec.shape, valid_pred.shape)
    for i, c_dx in dx_name_id_dict.items():
        tpr, fpr, _ = roc_curve(y_vec[:, i], valid_pred[:, i])
        auc = roc_auc_score(y_vec[:, i], valid_pred[:, i])
        acc = accuracy_score(y_vec[:, i], valid_pred[:, i]>0.5)
        ax1.plot(tpr, fpr, '.-', label='{}: AUC {:0.2f}, Accuracy: {:2.0%}'.format(c_dx, auc, acc))
        all_rows+=[{'dx_name': c_dx, 
                    'prediction': valid_pred[j, i], 
                    'class': 'Positive' if y_vec[j, i]>0.5 else 'Negative'} 
                         for j in range(valid_pred.shape[0])]
    
    d_ax = m_axs[0, 1]
    t_yp = np.mean(valid_pred, 0)
    t_y = np.mean(y_vec, 0)
    d_ax.barh(np.arange(len(dx_name_id_dict))+0.1, t_yp, alpha=0.5, label='Predicted')
    d_ax.barh(np.arange(len(dx_name_id_dict))-0.1, t_y+0.001, alpha=0.5, label='Ground Truth')
    d_ax.set_xlim(0, 1)
    d_ax.set_yticks(range(len(dx_name_id_dict)))
    d_ax.set_yticklabels(dx_name_id_dict.values(), rotation=0)
    d_ax.set_title('Overall')
    d_ax.legend()
    
    # show example images
    ax1.legend()
    for (_, c_row), (c_ax, d_ax) in zip(
        example_df.sample(m_axs.shape[0]).iterrows(), 
                               m_axs[1:]):
        
        c_ax.imshow(imread(c_row['image_path']))
        c_ax.set_title(c_row['dx_name'])
        c_ax.axis('off')
        t_yp = in_model.predict(np.expand_dims(c_row['color_features'], 0))
        t_y = np.array(c_row['dx_vec'])
        d_ax.barh(np.arange(len(dx_name_id_dict))+0.1, t_yp[0], alpha=0.5, label='Predicted')
        d_ax.barh(np.arange(len(dx_name_id_dict))-0.1, t_y+0.001, alpha=0.5, label='Ground Truth')
        d_ax.set_yticks(range(len(dx_name_id_dict)))
        d_ax.set_yticklabels(dx_name_id_dict.values(), rotation=0)
        d_ax.set_xlim(0, 1)
        d_ax.legend();
    
    # nice dataframe of output
    c_all_df = pd.DataFrame(all_rows)
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    if plot_type=='swarm':
        sns.swarmplot(data=c_all_df, hue='class', y='prediction', x='dx_name', size=2.0, ax=ax1)
    elif plot_type=='box':
        sns.boxplot(data=c_all_df, hue='class', y='prediction', x='dx_name', ax=ax1)
    ax1.set_ylim(-0.05, 1.05)
    return c_all_df


# # The Simplist Model
# Nearest Neighbor works by finding the most similar case from the training data using the feature vector. We can directly visualize this by showing which training image was being looked at.

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(train_x_vec, train_y_vec)


# ## Show the results
# We get incredibly good, nearly perfect results! Are we done now? Time to build an app and sell it to google for $$$?

# In[ ]:


show_model_results(knn, use_split='train', plot_type='box');


# ## Let's dig down a bit deeper, how does it work?

# In[ ]:


fig, m_axs = plt.subplots(6, 4, figsize=(30, 40))
dummy_web_image = Image.new(size=(1,1), mode='RGB').convert('P', palette='web')

for (c_ax, c_feat_ax, d_ax, d_feat_ax), (_, c_row) in zip(m_axs, 
                            color_feat_df.sample(m_axs.shape[0], random_state=2018).iterrows()):
    
    query_img = Image.open(c_row['image_path'])
    idx_to_color = np.array(query_img.convert('P', palette='web').getpalette()).reshape((-1, 3))/255.0
    c_ax.imshow(query_img)
    c_ax.set_title(c_row['lesion_id'][:25])
    c_ax.axis('off')
    counts, bins = np.histogram(np.ravel(query_img.convert('P', palette='web')), 
                                bins=np.arange(256))
    
    for i in range(counts.shape[0]):
        c_feat_ax.bar(bins[i], counts[i], color=idx_to_color[i], edgecolor='k', linewidth=0.1)
    c_feat_ax.set_yscale('log')
    c_feat_ax.set_xlabel('Color Id')
    c_feat_ax.set_ylabel('Pixel Count')
    c_feat_ax.set_title('Feature Vector')
    
    dist, idx = knn.kneighbors(np.expand_dims(c_row['color_features'], 0))
    m_row = train_df.iloc[idx[0][0]]
    matched_img = Image.open(m_row['image_path'])
    
    d_ax.imshow(matched_img)
    d_ax.set_title('Closest Match\n{}\nDistance: {:2.1%}'.format(m_row['lesion_id'][:25], dist[0][0]))
    d_ax.axis('off')
    
    counts, bins = np.histogram(np.ravel(matched_img.convert('P', palette='web')), 
                                bins=np.arange(256))
    
    for i in range(counts.shape[0]):
        d_feat_ax.bar(bins[i], counts[i], color=idx_to_color[i], edgecolor='k', linewidth=0.1)
    d_feat_ax.set_yscale('log')
    d_feat_ax.set_xlabel('Color Id')
    d_feat_ax.set_ylabel('Pixel Count')
    c_feat_ax.set_title('Matched Feature')


# ## Use on the validation split

# In[ ]:


show_model_results(knn, use_split='valid');


# # Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_x_vec, train_y_vec)


# In[ ]:


show_model_results(lr);


# ## Normalize the input
# 
# We can make a pipeline to normalize the input and remove bad features

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
lr_pipe = make_pipeline(RobustScaler(), VarianceThreshold(0.99), LinearRegression())
lr_pipe.fit(train_x_vec, train_y_vec)


# In[ ]:


show_model_results(lr_pipe);


# # More Complicated Models
# We can try decision trees to get better results

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
dt_pipe = make_pipeline(RobustScaler(), 
                        PCA(n_components=10), 
                        DecisionTreeRegressor(max_depth=5, min_samples_split=50))
dt_pipe.fit(train_x_vec, train_y_vec)
show_model_results(dt_pipe);


# In[ ]:


from sklearn.tree import export_graphviz
import graphviz
def show_tree(in_tree):
    return graphviz.Source(export_graphviz(in_tree, out_file=None))

show_tree(dt_pipe.steps[-1][1])


# # Fancier Models
# Here we can use much fancier models like random forest to even further improve the performance

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_pipe = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=200))
rf_pipe.fit(train_x_vec, train_y_vec)
show_model_results(rf_pipe);


# ## XGBoost
# One of the most powerful classification tools

# In[ ]:


from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
xg_pipe = make_pipeline(RobustScaler(), 
                        MultiOutputRegressor(XGBRegressor(objective='reg:linear')))
xg_pipe.fit(train_x_vec, train_y_vec)


# In[ ]:


show_model_results(xg_pipe);


# In[ ]:




