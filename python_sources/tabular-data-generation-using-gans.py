#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install -qU scikit-learn')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score,                            accuracy_score, classification_report,                            plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import os
from collections import Counter

np.random.seed(34)
path = '/kaggle/input/creditcardfraud/'


# # Data Exploration and Cleaning

# In[ ]:


# reading data
df = pd.read_csv(f'{path}creditcard.csv')
df.drop("Time", 1, inplace=True)
print(df.shape)
df.head()


# In[ ]:


# High class imbalance
df['Class'].value_counts(normalize=True)*100


# In[ ]:


# Checking for Null values
print(f"Number of Null values: {df.isnull().any().sum()}")


# In[ ]:


# checking for duplicate values
print(f"Dataset has {df.duplicated().sum()} duplicate rows")
# dropping duplicate rows
df.drop_duplicates(inplace=True)


# In[ ]:


# high skweness in Amount feature
plt.figure(figsize=(14,4))
df['Amount'].value_counts().head(50).plot(kind='bar')
plt.show()


# In[ ]:


# checking skewness of other columns
df.drop('Class',1).skew()


# In[ ]:


# taking log transform of high positively skewed features
skew_cols = df.drop('Class', 1).skew().loc[lambda x: x>2].index
for col in skew_cols:
    lower_lim = abs(df[col].min())
    normal_col = df[col].apply(lambda x: np.log10(x+lower_lim+1))
    print(f"Skew value of {col} after log transform: {normal_col.skew()}")


# In[ ]:


# Only applying log transform to Amount feature
df['Amount'] = df['Amount'].apply(lambda x: np.log10(x+1))


# In[ ]:


scaler = StandardScaler()
#scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('Class', 1))
y = df['Class'].values
print(X.shape, y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)


# # Training a Baseline Model

# In[ ]:


# simple linear regression
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))
plot_confusion_matrix(linear_model, X_test, y_test)
plt.show()


# # Using weighted regression to improve accuracy

# In[ ]:


weights = np.linspace(0.05, 0.95, 15)

gscv = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_res = gscv.fit(X, y)

print("Best parameters : %s" % grid_res.best_params_)


# In[ ]:


# plotting F1 scores 
plt.plot(weights, grid_res.cv_results_['mean_test_score'], marker='o')
plt.grid()
plt.show()


# In[ ]:


# training with best weights
wlr = LogisticRegression(**grid_res.best_params_)
wlr.fit(X_train, y_train)

y_pred = wlr.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))
plot_confusion_matrix(wlr, X_test, y_test)
plt.show()


# **Slight improvement when using weighted regression**

# # Using SMOTE for upsampling

# In[ ]:


# constructing pipeline
pipe = Pipeline([
        ('smote', SMOTE()),
        ('lr', LogisticRegression())
])
# training model with smote samples
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True)
plt.show()


# **Poor F1 score even when using SMOTE**

# # Grid Search on SMOTE and Regression

# In[ ]:


pipe = Pipeline([
        ('smote', SMOTE()),
        ('lr', LogisticRegression())
])
sm_ratio = np.linspace(0.2, 0.8, 10)
lr_weights = np.linspace(0.05, 0.95, 10)

gscv = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__sampling_strategy': sm_ratio,
        'lr__class_weight': [{0: x, 1: 1.0-x} for x in lr_weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gscv.fit(X, y)

print("Best parameters : %s" % grid_result.best_params_)


# In[ ]:


df_gs = pd.DataFrame(data=grid_result.cv_results_['mean_test_score'].reshape(10,10),
                     index=np.around(sm_ratio[::-1], 2), 
                     columns=np.around(lr_weights[::-1], 2))
plt.figure(figsize=(8,8))
sns.heatmap(df_gs,
            annot=True,
            linewidths=.5)
plt.show()


# In[ ]:


# training with best weights
pipe = Pipeline([
        ('smote', SMOTE(sampling_strategy= 0.2)),
        ('lr', LogisticRegression(class_weight={0: 0.95, 1: 0.05}))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True)
plt.show()


# **Using SMOTE with weighted regression improves results**

# # Using GANs to generate new data

# In[ ]:


from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
from sklearn.utils import shuffle


# In[ ]:


class cGAN():
    def __init__(self):
        self.latent_dim = 32
        self.out_shape = 29
        self.num_classes = 2
        self.clip_value = 0.01
        optimizer = Adam(0.0002, 0.5)
        #optimizer = RMSprop(lr=0.00005)

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # generating new data samples
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        gen_samples = self.generator([noise, label])

        self.discriminator.trainable = False

        # passing gen samples through disc. 
        valid = self.discriminator([gen_samples, label])

        # combining both models
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer,
                             metrics=['accuracy'])
        self.combined.summary()

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim))
        #model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256))
        #model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        #model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.out_shape, activation='tanh'))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        
        model_input = multiply([noise, label_embedding])
        gen_sample = model(model_input)

        return Model([noise, label], gen_sample, name="Generator")

    
    def build_discriminator(self):
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        model.add(Dense(512, input_dim=self.out_shape, kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(256, kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        model.add(Dense(128, kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        gen_sample = Input(shape=(self.out_shape,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.out_shape)(label))

        model_input = multiply([gen_sample, label_embedding])
        validity = model(model_input)

        return Model(inputs=[gen_sample, label], outputs=validity, name="Discriminator")


    def train(self, X_train, y_train, pos_index, neg_index, epochs, batch_size=32, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            #  Train Discriminator with 8 sample from postivite class and rest with negative class
            idx1 = np.random.choice(pos_index, 8)
            idx0 = np.random.choice(neg_index, batch_size-8)
            idx = np.concatenate((idx1, idx0))
            samples, labels = X_train[idx], y_train[idx]
            samples, labels = shuffle(samples, labels)
            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_samples = self.generator.predict([noise, labels])

            # label smoothing
            if epoch < epochs//1.5:
                valid_smooth = (valid+0.1)-(np.random.random(valid.shape)*0.1)
                fake_smooth = (fake-0.1)+(np.random.random(fake.shape)*0.1)
            else:
                valid_smooth = valid 
                fake_smooth = fake
                
            # Train the discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch([samples, labels], valid_smooth)
            d_loss_fake = self.discriminator.train_on_batch([gen_samples, labels], fake_smooth)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            # Condition on labels
            self.discriminator.trainable = False
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            if (epoch+1)%sample_interval==0:
                print (f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")


# In[ ]:


cgan = cGAN()


# In[ ]:


y_train = y_train.reshape(-1,1)
pos_index = np.where(y_train==1)[0]
neg_index = np.where(y_train==0)[0]
cgan.train(X_train, y_train, pos_index, neg_index, epochs=2000)


# In[ ]:


# generating new samples
noise = np.random.normal(0, 1, (400, 32))
sampled_labels = np.ones(400).reshape(-1, 1)

gen_samples = cgan.generator.predict([noise, sampled_labels])
gen_samples = scaler.inverse_transform(gen_samples)
print(gen_samples.shape)


# In[ ]:


gen_df = pd.DataFrame(data = gen_samples,
                      columns = df.drop('Class',1).columns)
gen_df.head()


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
ax[0].scatter(df[df['Class']==1]['Amount'], df[df['Class']==1]['V1'])
ax[1].scatter(gen_df['Amount'], gen_df['V1'])
plt.show()


# In[ ]:




