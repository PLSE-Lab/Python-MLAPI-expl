#!/usr/bin/env python
# coding: utf-8

# # Goal
# The goal is to make a simple model that can go from a satellite image to a prediction of how likely it is if there is any damage present
# 
# ## Setup
# We basically take the precomputed color features and build simple models in order to determine if the image contains any damage [here](https://www.kaggle.com/kmader/hurricane-damage-overview). We try to create a balanced training group and a realistic validation group to know if the model is learning anything useful

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
image_dir = Path('..') / 'input' / 'satellite-images-of-hurricane-damage'
mapping_file = Path('..') / 'input' / 'hurricane-damage-overview' / 'resnet_features.json'
image_df = pd.read_json(mapping_file)
image_df['damage_val'] = image_df['damage'].map(lambda x: x=='damage') 
image_df.sample(3)


# Split up the groups so we can validate our model on something besides the direct training data

# In[ ]:


image_df['data_split'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split
train_df = image_df.query('data_split=="train_another"')
train_df.reset_index(inplace=True)
print(train_df.shape[0], 'training images')


# In[ ]:


train_x_vec = np.stack(train_df['resnet_features'].values, 0)
train_y_vec = np.stack(train_df['damage_val'], 0)
print(train_x_vec.shape, '->', train_y_vec.shape)


# # Display Results Nicely
# We want to have code to display our results nicely so we can see what worked well and what didn't

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
def show_model_results(in_model, use_split=None, plot_type='swarm'):
    fig, m_axs = plt.subplots(4, 2, figsize=(15, 30))
    m_axs = m_axs.flatten()
    all_rows = []
    ax1 = m_axs[0]
    
    if use_split is None:
        cur_df = image_df.copy()
    else:
        cur_df = image_df.query('data_split=="{}"'.format(use_split)) 
    
    for c_split, example_df in cur_df.groupby('data_split'):
        example_df = example_df.reset_index()
        x_vec = np.stack(example_df['resnet_features'].values, 0)
        y_vec = np.stack(example_df['damage_val'], 0)

        valid_pred = in_model.predict(x_vec)
        tpr, fpr, _ = roc_curve(y_vec[:], valid_pred[:])
        auc = roc_auc_score(y_vec[:], valid_pred[:])
        acc = accuracy_score(y_vec[:], valid_pred[:]>0.5)
        ax1.plot(tpr, fpr, '.-', label='{}, AUC {:0.2f}, Accuracy: {:2.0%}'.format(c_split, auc, acc))
        all_rows += [pd.DataFrame({'class': y_vec[:], 'prediction': np.clip(valid_pred[:], 0, 1), 'type': 'damage', 
                                  'split': c_split})]
    
    c_all_df = pd.concat(all_rows)
        
    # show example images
    ax1.legend()
    for (_, c_row), (c_ax) in zip(
        example_df.sample(m_axs.shape[0]).iterrows(), 
                               m_axs[1:-1]):
        
        c_ax.imshow(imread(c_row['path']))
        t_yp = in_model.predict(np.expand_dims(c_row['resnet_features'], 0))
        c_ax.set_title('Class: {}\n Damage Prediction: {:2.2%}'.format(c_row['damage'], t_yp[0]))
        c_ax.axis('off')
        
        t_y = np.array(c_row['damage_val'])
    
    # nice dataframe of output
    
    ax1 = m_axs[-1]
    if plot_type=='swarm':
        # prevent overplotting
        sns.swarmplot(data=c_all_df.sample(500) if c_all_df.shape[0]>1000 else c_all_df,
                      hue='class', 
                      y='prediction', 
                      x='type', 
                      size=2.0, 
                      ax=ax1)
    elif plot_type=='box':
        sns.boxplot(data=c_all_df, hue='class', y='prediction', x='type', ax=ax1)
    elif plot_type=='violin':
        sns.violinplot(data=c_all_df, hue='class', y='prediction', x='type', ax=ax1)
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


show_model_results(knn, use_split='train_another', plot_type='box');


# ## Let's dig down a bit deeper, how does it work?

# In[ ]:


fig, m_axs = plt.subplots(6, 4, figsize=(30, 40))
for (c_ax, c_feat_ax, d_ax, d_feat_ax), (_, c_row) in zip(m_axs, 
                            image_df.sample(m_axs.shape[0], random_state=2018).iterrows()):
    
    query_img = Image.open(c_row['path'])
    idx_to_color = np.array(query_img.convert('P', palette='web').getpalette()).reshape((-1, 3))/255.0
    c_ax.imshow(query_img)
    c_ax.set_title(c_row['location'][:25])
    c_ax.axis('off')
    c_feat_ax.bar(np.arange(2048), c_row['resnet_features'], edgecolor='k', linewidth=0.1)
    c_feat_ax.set_xlabel('Color Id')
    c_feat_ax.set_ylabel('Pixel Count')
    c_feat_ax.set_title('Feature Vector')
    
    dist, idx = knn.kneighbors(np.expand_dims(c_row['resnet_features'], 0))
    m_row = train_df.iloc[idx[0][0]]
    matched_img = Image.open(m_row['path'])
    
    d_ax.imshow(matched_img)
    d_ax.set_title('Closest Match\n{}\nDistance: {:2.1%}'.format(m_row['location'][:25], dist[0][0]))
    d_ax.axis('off')
    
    d_feat_ax.bar(np.arange(2048), m_row['resnet_features'], edgecolor='k', linewidth=0.1)
    d_feat_ax.set_xlabel('Color Id')
    d_feat_ax.set_ylabel('Pixel Count')
    d_feat_ax.set_title('Matched Feature')


# ## Use on the validation split

# In[ ]:


show_model_results(knn);


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
xg_pipe = make_pipeline(RobustScaler(), 
                        XGBRegressor(objective='reg:linear'))
xg_pipe.fit(train_x_vec, train_y_vec)


# In[ ]:


show_model_results(xg_pipe);


# # Model Sensitivity
# How does changing the images slightly (saturation, contrast, brightness) affect the predictions of the model?

# In[ ]:


from keras.models import load_model
from PIL import ImageEnhance, Image
resnet_model = load_model('../input/hurricane-damage-overview/feature_model.h5')
def color_count_feature(raw_image):
    return resnet_model.predict(np.expand_dims(raw_image, 0))[0]


# In[ ]:


sample_images_df = image_df.query('data_split=="validation_another"').groupby('damage').apply(lambda x: x.sample(3)).reset_index(drop=True)


# ## Saturation

# In[ ]:


op_vals = [None, 0.15, 0.5, 2]
fig, m_axs = plt.subplots(sample_images_df.shape[0], len(op_vals)+1, figsize=(25, 25))

for (_, c_row), n_axs in zip(
        sample_images_df.iterrows(), m_axs):
        
        c_img = Image.open(c_row['path'])
        
        for c_op, c_ax in zip(op_vals, n_axs):
            if c_op is None:
                t_img = c_img
                t_caption = 'Original: {} \n'.format(c_row['damage'])
            else:
                t_img = ImageEnhance.Color(c_img).enhance(c_op)
                t_caption = 'Saturation: {:2.1%}\n'.format(c_op)
            c_feat = color_count_feature(t_img)
            c_ax.imshow(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))
            c_ax.set_title('{}Model Score:{:2.1%}'.format(t_caption, t_yp[0]))
            c_ax.axis('off')
        
        op_space = np.linspace(np.min(op_vals[1:]), np.max(op_vals[1:]), 50)
        im_score = np.zeros_like(op_space)
        
        for i, c_op in enumerate(op_space):
            t_img = ImageEnhance.Color(c_img).enhance(c_op)
            c_feat = color_count_feature(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))[0]
            im_score[i] = t_yp
        n_axs[-1].plot(100*op_space, 100*im_score.clip(0, 1), '.-', lw=0.2)
        n_axs[-1].set_ylabel('Model Prediction')
        n_axs[-1].set_xlabel('Saturation')
        n_axs[-1].set_ylim(-5, 105)
            


# ## Contrast

# In[ ]:


op_vals = [None, 0.25, 1.5, 4]
fig, m_axs = plt.subplots(sample_images_df.shape[0], len(op_vals)+1, figsize=(25, 25))

for (_, c_row), n_axs in zip(
        sample_images_df.iterrows(), m_axs):
        
        c_img = Image.open(c_row['path'])
        
        for c_op, c_ax in zip(op_vals, n_axs):
            if c_op is None:
                t_img = c_img
                t_caption = 'Original: {} \n'.format(c_row['damage'])
            else:
                t_img = ImageEnhance.Contrast(c_img).enhance(c_op)
                t_caption = 'Contrast: {:2.1%}\n'.format(c_op)
            c_feat = color_count_feature(t_img)
            c_ax.imshow(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))
            c_ax.set_title('{}Model Score:{:2.1%}'.format(t_caption, t_yp[0]))
            c_ax.axis('off')
        
        op_space = np.linspace(np.min(op_vals[1:]), np.max(op_vals[1:]), 50)
        im_score = np.zeros_like(op_space)
        
        for i, c_op in enumerate(op_space):
            t_img = ImageEnhance.Contrast(c_img).enhance(c_op)
            c_feat = color_count_feature(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))[0]
            im_score[i] = t_yp
        n_axs[-1].plot(100*op_space, 100*im_score.clip(0, 1), '.-', lw=0.2)
        n_axs[-1].set_ylabel('Model Score')
        n_axs[-1].set_xlabel('Image Contrast')
        n_axs[-1].set_ylim(-5, 105)
            


# # Brightness

# In[ ]:


op_vals = [None, 0.5, 1.5, 1/0.5]
fig, m_axs = plt.subplots(sample_images_df.shape[0], len(op_vals)+1, figsize=(25, 25))

for (_, c_row), n_axs in zip(
        sample_images_df.iterrows(), m_axs):
        
        c_img = Image.open(c_row['path'])
        
        for c_op, c_ax in zip(op_vals, n_axs):
            if c_op is None:
                t_img = c_img
                t_caption = 'Original: {} \n'.format(c_row['damage'])
            else:
                t_img = ImageEnhance.Brightness(c_img).enhance(c_op)
                t_caption = 'Brightness: {:2.1%}\n'.format(c_op)
            c_feat = color_count_feature(t_img)
            c_ax.imshow(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))
            c_ax.set_title('{}Model Score:{:2.1%}'.format(t_caption, t_yp[0]))
            c_ax.axis('off')
        
        op_space = np.linspace(np.min(op_vals[1:]), np.max(op_vals[1:]), 50)
        im_score = np.zeros_like(op_space)
        
        for i, c_op in enumerate(op_space):
            t_img = ImageEnhance.Brightness(c_img).enhance(c_op)
            c_feat = color_count_feature(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))[0]
            im_score[i] = t_yp
        n_axs[-1].plot(100*op_space, 100*im_score.clip(0, 1), '.-', lw=0.2)
        n_axs[-1].set_ylabel('Model Score')
        n_axs[-1].set_xlabel('Image Brightness')
        n_axs[-1].set_ylim(-5, 105)
            


# # Sharpness / Blurriness

# In[ ]:


op_vals = [None, -2.5, 1.5, 5]
fig, m_axs = plt.subplots(sample_images_df.shape[0], len(op_vals)+1, figsize=(25, 25))

for (_, c_row), n_axs in zip(
        sample_images_df.iterrows(), m_axs):
        
        c_img = Image.open(c_row['path'])
        
        for c_op, c_ax in zip(op_vals, n_axs):
            if c_op is None:
                t_img = c_img
                t_caption = 'Original: {} \n'.format(c_row['damage'])
            else:
                t_img = ImageEnhance.Sharpness(c_img).enhance(c_op)
                t_caption = 'Sharpness: {:2.1%}\n'.format(c_op)
            c_feat = color_count_feature(t_img)
            c_ax.imshow(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))
            c_ax.set_title('{}Model Score:{:2.1%}'.format(t_caption, t_yp[0]))
            c_ax.axis('off')
        
        op_space = np.linspace(np.min(op_vals[1:]), np.max(op_vals[1:]), 50)
        im_score = np.zeros_like(op_space)
        
        for i, c_op in enumerate(op_space):
            t_img = ImageEnhance.Sharpness(c_img).enhance(c_op)
            c_feat = color_count_feature(t_img)
            t_yp = xg_pipe.predict(np.expand_dims(c_feat, 0))[0]
            im_score[i] = t_yp
        n_axs[-1].plot(100*op_space, 100*im_score.clip(0, 1), '.-', lw=0.2)
        n_axs[-1].set_ylabel('Model Score')
        n_axs[-1].set_xlabel('Image Sharpness')
        n_axs[-1].set_ylim(-5, 105)
            


# In[ ]:




