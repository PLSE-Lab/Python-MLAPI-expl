#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


import os
from pylab import *
import seaborn as sns
import pandas as pd
from collections import Counter


# In[ ]:


sns.set()


# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df


# <b>
# The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.<br>
# Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# </b>

# In[ ]:


df.info()


# ## Check data distribution

# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(20, 20))
c = 1
for col in list(df.columns[0:-1]):
    plt.subplot(6, 5, c)
    plt.boxplot(df[col])
    plt.legend([col])
    c+=1
    


# In[ ]:


sns.distplot(df.Time);


# In[ ]:


sns.boxenplot(y=df.Amount)


# In[ ]:


sns.countplot(x="Class", data=df)


# In[ ]:


counter = Counter(df["Class"])

print(f"""
# NotFraud: {counter[0]}
# Fraud: {counter[1]}

# ratio: {counter[1]/counter[0]}
""")


# There's a huge gap between a number of NOT fraud and fraud data <br>
# -> Imbalanced Dataset

# ## Find a linear relationship

# <b> correlation matrix

# In[ ]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# From the correlation matrix: <br>
# - Amount has a positive-linear tendency with V17, V20
# - Amount has a negative-linear tendency with V2, V5
# - Time has a negative-linear tendency with V3

# In[ ]:


sns.pairplot(data=df, x_vars=["V17", "V20"], y_vars=['Amount']);


# In[ ]:


sns.pairplot(data=df, x_vars=["V2", "V5"], y_vars=['Amount']);


# In[ ]:


sns.pairplot(data=df, x_vars=["V3"], y_vars=['Time']);


# ## Split data (for training & testing)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df_X, df_y = df[list(df.columns[:-1])], df["Class"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=3)


# In[ ]:


# train data ratio of n_frad / n_not_fraud
counter = Counter(y_train)
print(f"train data ratio of n_frad / n_not_fraud: {counter[1]/counter[0]}")


# In[ ]:


# test data ratio of n_frad / n_not_fraud
counter = Counter(y_test)
print(f"test data: {counter}", end="\n\n")
print(f"test data ratio of n_frad / n_not_fraud: {counter[1]/counter[0]}")


# Confirmed that training dataset and test dataset are proportionally split with respect to 'n_frad / n_not_fraud'

# ## Oversampling (to fix the imbalanced data)

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


X_samp, y_samp = SMOTE(random_state=1).fit_sample(X_train, y_train)


# In[ ]:


print(f"""
X_samp.shape: {X_samp.shape}
y_samp.shape: {y_samp.shape}
""")


# In[ ]:


counter = Counter(y_samp)

print(f"""
* After SMOTE

# NotFraud: {counter[0]}
# Fraud: {counter[1]}

-> Now it's balanced!
""")


# ## Analysis Using MA models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# <b>Random Forest

# In[ ]:


random_forest_clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)


# In[ ]:


random_forest_clf.fit(X_samp, y_samp)


# <b>Neural Network

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler


# In[ ]:


# input data scaling for training stability

scalerX = StandardScaler()
sc_X_samp = scalerX.fit_transform(X_samp)


# In[ ]:


tf.keras.backend.clear_session()


# In[ ]:


input_layer = tf.keras.Input(shape=(30,))

hl = Dense(64, activation="relu", kernel_initializer="he_normal")(input_layer)
hl = BatchNormalization()(hl)

hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)
hl = BatchNormalization()(hl)

hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)
hl = BatchNormalization()(hl)

hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)
hl = BatchNormalization()(hl)

output_layer = Dense(1, activation="sigmoid")(hl)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])


# In[ ]:


model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])


# In[ ]:


model.summary()


# In[ ]:


hist = model.fit(sc_X_samp, y_samp, batch_size=2**14, epochs=20, validation_split=0.1)


# In[ ]:


# training result plot

plt.plot(hist.history["loss"], label="loss")
plt.legend();


# In[ ]:


# training result plot

plt.plot(hist.history["precision"], label="precision")
plt.plot(hist.history["recall"], label="recall")
plt.plot(hist.history["val_precision"], label="val_precision")
plt.plot(hist.history["val_recall"], label="val_recall")
plt.legend();


# ## Test

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# <b> Random Forest

# In[ ]:


# on training dataset

y_pred = random_forest_clf.predict(X_samp)
print(classification_report(y_samp, y_pred))


# In[ ]:


# on test dataset

y_pred = random_forest_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


fi = pd.DataFrame()

fi['features'] = list(X_samp.columns)
fi['values'] = random_forest_clf.feature_importances_

fi = fi.sort_values(by="values", ascending=False);

sns.catplot(x="features", y="values", data=fi, kind="bar", 
            aspect=4, height=4);


# From the above graph, <br>
# the most influential variables for the detection are: V14, V10, V12, V4, ... (in order)

# <b> Neural Network

# In[ ]:


# on test dataset

y_pred = model.predict(scalerX.transform(X_test))


# In[ ]:


print(classification_report(y_test, np.round(y_test)))


# In[ ]:


confusion_matrix(y_test, np.round(y_test), labels=[0, 1])

