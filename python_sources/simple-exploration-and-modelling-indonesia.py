#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# <b> Carlo Abimanyu </b> <br>
# Ini adalah notebook pengerjaan saya dalam kompetisi Titanic di Kaggle untuk berlatih. Pekerjaan ini masih sangat jauh dari kata cukup, masih banyak kekurangannya. Sehingga sangat diharapkan ada feedback dari teman-teman. Semoga bisa menjadi gambaran singkat untuk teman-teman yang ingin belajar Data Science.

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# Pertama, saya melakukan eksplorasi agar lebih memahami data-nya.

# In[ ]:


print(train_df.columns.values)


# - Data Kategorikal: Survived, Sex, dan Embarked. Ordinal: Pclass
# - Data Numerik    : Kontinu: Age, Fare. Diskrit: SibSp, Parch <br>
# Adapun Ticket dan Cabin merupakan tipe data campuran (numeric dan alphanumeric)

# In[ ]:


train_df.info()


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# Kolom 'Cabin' memiliki banyak missing value. Saya akan mencoba meng-eksplorasi lebih jauh kolom tersebut.

# In[ ]:


train_df['Cabin'].describe()


# Ternyata kategori dari 'Cabin' pun sangat beragam, sehingga tidak saya lakukan One-Hot Encoding. Kolom 'Cabin' saya drop saja.

# In[ ]:


train_df = train_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin', axis=1)


# 'PassengerId', 'Name', 'Embarked' dan 'Ticket' secara intuitif seharusnya tidak berkontribusi terhadap selamat atau tidaknya penumpang. Maka kolom tersebut juga saya drop.

# In[ ]:


train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1)


# Selanjutnya saya akan menganalisis korelasi antara Pclass dengan Survived. Hipotesisnya, seharusnya yang berada di kelas 1 lebih banyak yang selamat karena biasanya berisi orang-orang kaya dan penting (diberikan fasilitas dan pelayanan lebih).

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Ternyata benar, yang berada di kelas lebih baik lebih banyak yang selamat (sekitar 63%). <br>
# Lalu saya juga ingin menganalisis korelasi Sex dengan Survived.

# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Ternyata yang perempuan lebih banyak selamat dibandingkan yang laki-laki.

# Age atau umur pasti sangat berpengaruh dalam menentukan selamat atau tidaknya suatu penumpang. <br>
# Berdasarkan analisis diatas, maka **Age**, **Pclass**, dan **Sex** akan digunakan sebagai *feature*.

# Secara intuitif, Fare memiliki kontribusi terhadap Survived, maka **Fare** akan digunakan sebagai *feature*. Namun, pada data test terdapat missing value pada kolom Fare.

# In[ ]:


train_df['Fare'].describe()


# In[ ]:


test_df['Fare'].describe()


# Dilihat dari sebaran datanya, lebih tepat jika missing value tersebut diisi dengan median, bukan mean.

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# Usia tentu sangat berkontribusi terhadap selamat atau tidaknya penumpang.

# In[ ]:


train_df['Age'].describe()


# In[ ]:


test_df['Age'].describe()


# Dilihat dari sebaran datanya, lebih tepat jika missing value tersebut diisi dengan median, bukan mean.

# In[ ]:


train_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)


# Sekarang sudah tidak ada missing value lagi. Selanjutnya, saya akan menangani data pada kolom Sex, karena masih ber-tipe object atau string. Berdasarkan analisis sebelumnya, perempuan lebih banyak selamat, maka saya rasa tidak perlu One-Hot Encoding, cukup Label Encoding saja. Sekaligus agar menunjukkan bahwa perempuan memiliki peluang selamat lebih besar.

# In[ ]:


train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# Sudah tidak ada lagi missing value mauapun data ber-tipe object atau string. Sehingga saya akan mulai membuat model machine learning.

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.copy()
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


# k-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[ ]:


# XGBoost

xgb = GradientBoostingClassifier(learning_rate=0.05)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb.score(X_train, y_train)
acc_xgb = round(xgb.score(X_train, y_train) * 100, 2)
acc_xgb


# In[ ]:


# test_df_1 = pd.read_csv('../input/titanic/test.csv')

# submission = pd.DataFrame({
#         "PassengerId": test_df_1["PassengerId"],
#         "Survived": y_pred_xgb
#     })

# submission.to_csv('submission.csv', index=False)


# Banyak sekali kekurangan dari pekerjaan ini. Kritik dan saran sangat diharapkan. Terima kasih.
