#!/usr/bin/env python
# coding: utf-8

# https://rapidsai.github.io/projects/cuml/en/0.12.0/api.html?highlight=svc#support-vector-machines  
# > Support Vector Machines  
# > classcuml.svm.SVC(C-Support Vector Classification)  
# > Currently only binary classification is supported.  

# # Import libraries

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# https://www.kaggle.com/cdeotte/rapids\n\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib"] + sys.path \n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport os\n\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\nfrom cuml.svm import SVC\nfrom sklearn.metrics import accuracy_score, confusion_matrix\nfrom sklearn.model_selection import train_test_split\nfrom tqdm import tqdm')


# # Load datasets

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\npath_input = Path('/kaggle/input/digit-recognizer/')\n\ndf_train = pd.read_csv(path_input / 'train.csv')\ndf_test  = pd.read_csv(path_input / 'test.csv')\n\nX_train = df_train.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32)\ny_train = df_train.iloc[:, 0 ].values\n\nX_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)\n\nX_test  = df_test.values.reshape(-1, 28, 28).astype(np.float32)")


# In[ ]:


fig = plt.figure(figsize=(18, 9))
for row in range(5):
    for col in range(10):
        ax = fig.add_subplot(5, 10, 1+10*row+col)
        ax.imshow(X_train[10*row+col])
        ax.set_axis_off()
        ax.set_title(y_train[10*row+col])


# # Train

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlist_model = []\n\nfor ans in tqdm(range(10)):\n    model = SVC()\n    model.fit(X_train.reshape(-1, 28*28), y_train==ans)\n    list_model.append(model)')


# # Predict

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlist_h_valid = []\n\nfor ans in tqdm(range(10)):\n    h_valid = list_model[ans].predict(X_valid.reshape((-1, 28*28)))\n    h_valid = h_valid.values.get()\n    list_h_valid.append(h_valid)\n\nh_valid = np.array(list_h_valid).argmax(axis=0)')


# # validation

# In[ ]:


confusion_matrix(y_valid, h_valid)


# In[ ]:


accuracy_score(y_valid, h_valid)


# In[ ]:


fig = plt.figure(figsize=(18, 9))
for row in range(5):
    for col in range(10):
        ax = fig.add_subplot(5, 10, 1+10*row+col)
        ax.imshow(X_valid[y_valid != h_valid][10*row+col])
        ax.set_axis_off()
        ax.set_title(h_valid[y_valid != h_valid][10*row+col])

