#!/usr/bin/env python
# coding: utf-8

# Originally from @sergeyzlobin

# In[1]:


import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# In[ ]:





# # Load data
# 
# The data sample contains coordinates and time of photons reqistered in a Kiloton-Scale Liquid Scintillator Detector. There are two classes of events. The signal class contains double beta decay ($\beta\beta$-decay) events. The background class contains events of neutrino interactions due to $^{8}B$ decays in the sun. 
# 
# The detector has spherical form and registers photons on its surface. Each event is described by coordinates and time of registered photons. There are two types of photons in an event. Cherenkov photons fly in a cone and are produced by a particle due to Cherenkov effect. Scintillation photons fly in all directions and are produced due to interactions of photons and electrons with scintillator of the detector. 
# 
# Example of ideal (without scattering) signal (left) and background (right) events is shown in the following figure:
# 
# ![sig_bkg.png](attachment:sig_bkg.png)
# 
# Red and blue triangles correspond to Cherenkov photons, cyan dots - scintillation photons.

# In[3]:


data_path = '../input/bbdcay/'
pca_num=12


# In[4]:


# Read train data
data_train = pd.read_csv(f'{data_path}data_train.csv')
labels_train = pd.read_csv(f'{data_path}labels_train.csv')
# Read test data
data_test = pd.read_csv(f'{data_path}data_test.csv')


# In[5]:


data_train.head()


# In[6]:


labels_train.head()


# # Create images
# 
# For each event create an image in ($\theta$, $\phi$) coordinates with several time channels.

# In[7]:


def create_images(data, n_theta_bins=10, n_phi_bins=20, n_time_bins=6):
    images = []
    event_indexes = {}
    event_ids = np.unique(data['EventID'].values)

    # collect event indexes
    data_event_ids = data['EventID'].values
    for i in range(len(data)):
        i_event = data_event_ids[i]
        if i_event in event_indexes:
            event_indexes[i_event].append(i)
        else:
            event_indexes[i_event] = [i]

    # create images
    for i_event in event_ids:
        event = data.iloc[event_indexes[i_event]]
        X = event[['Theta', 'Phi', 'Time']].values
        one_image, edges = np.histogramdd(X, bins=(n_theta_bins, n_phi_bins, n_time_bins))
        images.append(one_image)

    return np.array(images)


# In[8]:


get_ipython().run_cell_magic('time', '', "X_train = create_images(data_train, n_theta_bins=15, n_phi_bins=20, n_time_bins=6)\nprint('train images created', X_train.shape)")


# In[10]:


get_ipython().run_cell_magic('time', '', "X_test = create_images(data_test, n_theta_bins=15, n_phi_bins=20, n_time_bins=6)\nprint('test images created', X_test.shape)")


# In[11]:


y = labels_train['Label'].values


# ## Cherenkov photons

# In[12]:


width, height = 3, 2

plt.figure(figsize=(9, 6))
for i in range(6):
    img = X_train[100][:, :, i]
    plt.subplot(height, width, i+1)
    plt.title(f't={i} y={y[100]}')
    plt.imshow(img)
    plt.colorbar()


# ## Scintillation photons

# In[13]:


width, height = 3, 2

plt.figure(figsize=(9, 6))
for i in range(6):
    img = X_train[79999][:, :, i]
    plt.subplot(height, width, i+1)
    plt.title(f't={i} y={y[79999]}')
    plt.imshow(img)
    plt.colorbar()


# ## min-max scaling

# In[14]:


d=np.concatenate((X_train,X_test))

scaler = StandardScaler()
scaler.fit(d.reshape(len(d), -1, ))
X_train=scaler.transform(X_train.reshape(len(X_train), -1, ))
X_test=scaler.transform(X_test.reshape(len(X_test), -1, ))


# ## PCA 

# In[15]:


d_scal=np.concatenate((X_train,X_test))

pca = PCA(n_components=pca_num)
pca.fit(d_scal.reshape(len(d_scal), -1, ))


# > # Start **KFold**

# In[16]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# In[19]:


from sklearn.ensemble import VotingClassifier
def do_train():
    oof_preds = np.zeros((len(X_train), ))
    preds = None

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        train_objects = X_train[trn_]
        valid_objects = X_train[val_]
        y_train = y[trn_]
        y_valid = y[val_]
        print('train len', len(train_objects))
        print('valid len', len(valid_objects))
        
        trainX_pca = pca.transform(train_objects.reshape(len(train_objects), -1, ))
        validX_pca = pca.transform(valid_objects.reshape(len(valid_objects), -1, ))

        model = VotingClassifier(n_jobs=-1, voting='soft', weights=None, 
                              estimators=[
                                  ('etc', ExtraTreesClassifier(n_estimators=1000, max_depth=10)), 
                                  ('rfc', RandomForestClassifier()),
                              ]
                             )
        model.fit(trainX_pca, y_train)

        y_pred = model.predict_proba(validX_pca)[:, 1]
        print(y_valid.shape)
        print(y_pred.shape)
        current_loss = roc_auc_score(y_valid, y_pred)
        print(current_loss)
        oof_preds[val_] = y_pred

        X_test_pca = pca.transform(X_test.reshape(len(X_test), -1, ))
        test_pred = model.predict_proba(X_test_pca)[:, 1]
        if preds is None:
            preds = test_pred
        else:
            preds += test_pred
        del model

    cv_loss = roc_auc_score(y, oof_preds)
    print('ALL FOLDS AUC: %.5f ' % cv_loss)
    oof_preds_df = pd.DataFrame()
    oof_preds_df['EventID'] = np.unique(data_train['EventID'].values)
    oof_preds_df['Proba'] = oof_preds
    oof_preds_df.to_csv('oof_preds.csv', index=False)
    return cv_loss, preds / folds.n_splits


# In[20]:


best_loss, preds = do_train()
print('CV:', best_loss)


# In[21]:


submission = pd.DataFrame()
submission['EventID'] = np.unique(data_test['EventID'].values)
submission['Proba'] = preds
submission.to_csv(f's_pca{pca_num}.csv', index=False, float_format='%.6f')


# In[ ]:




