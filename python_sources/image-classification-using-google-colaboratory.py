#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install PyDrive')


# In[ ]:


import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[ ]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[ ]:


get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
vcode = getpass.getpass()
get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# In[ ]:


get_ipython().system('mkdir -p drive')
get_ipython().system('google-drive-ocamlfuse drive')


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


# In[ ]:


download = drive.CreateFile({'id': '18s0FOQ-eR-RVuyrpS0sjzh9Lud8TXShg'})
download.GetContentFile('HAM10000_metadata.csv')


# In[ ]:


image_file = pd.read_csv('/content/drive/HAM10000_images/HAM10000_metadata.csv')


# In[ ]:


image_file.head()


# In[ ]:


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
image_file['le_type'] = image_file['dx'].map(lesion_type_dict.get) 
image_file['le_type_idx'] = pd.Categorical(image_file['le_type']).codes


# In[ ]:


image_file.isnull().sum()


# In[ ]:


image_file['age'].fillna((image_file['age'].median()), inplace=True)


# In[ ]:


image_file['le_type'].value_counts().plot(kind='bar')


# In[ ]:


image_file['localization'].value_counts().plot(kind='bar')


# In[ ]:


image_file['sex'].value_counts().plot(kind='bar')


# In[ ]:


image_file['age'].value_counts().plot(kind='bar')


# In[ ]:


image_file.dtypes


# In[ ]:


mod_image = []
for i in tqdm(range(image_file.shape[0])):
    img = image.load_img('/content/drive/HAM10000_images/'+image_file['image_id'][i]+'.jpg', target_size=(50,50,3))
    img = image.img_to_array(img)
    img = img/255
    mod_image.append(img)
    X = np.array(mod_image)


# In[ ]:


y=train['label'].values
y = to_categorical(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.2)


# In[ ]:


y=image_file['le_type_idx'].values
y = to_categorical(y)


# In[ ]:


X_train, X_validate, y_train, y_validate = train_test_split(X,y, test_size = 0.1, random_state = 2)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(50,50,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[ ]:


prediction = model.predict_classes(X_test)


# In[ ]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

