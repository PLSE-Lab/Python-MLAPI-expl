#!/usr/bin/env python
# coding: utf-8

# # Introduction

# **Konteks**
# 
# Dataset ini berasal dari National Institute of Diabetes and Digestive and Ginjal Diseases. Tujuan dari dataset adalah untuk memprediksi secara diagnostik apakah pasien memiliki diabetes, berdasarkan pengukuran diagnostik tertentu yang termasuk dalam dataset. Beberapa kendala ditempatkan pada pemilihan instance ini dari database yang lebih besar. Secara khusus, semua pasien di sini adalah wanita setidaknya 21 tahun dari warisan India Pima.
# 
# **Konten**
# 
# Kumpulan data terdiri dari beberapa variabel prediktor medis dan satu variabel target, Hasil. Variabel prediktor meliputi jumlah kehamilan yang pernah dialami pasien, BMI mereka, tingkat insulin, usia, dan sebagainya.

# # Table of Content
# 
# 1. Data Background
# 2. Data Preparation
# 3. Data Visualization
# 4. Building the Model
# 5. Testing Model

# # 1. Data Background

# **Source:**
# 
# Dataset ini tersedia di kaggle website: https://www.kaggle.com/uciml/pima-indians-diabetes-database, Dataset ini berasal dari National Institute of Diabetes and Digestive and Ginjal Diseases. Tujuan dari dataset adalah untuk memprediksi secara diagnostik apakah pasien memiliki diabetes, berdasarkan pengukuran diagnostik tertentu yang termasuk dalam dataset. Secara khusus, semua pasien di sini adalah wanita yang berumur 21 tahun dari Prima Indian Heritage.

# * Pregnancies = Jumlah kehamilan
# * Glucose = konsentrasi glukosa 2 jam dalam tes toleransi glukosa oral
# * BloodPressure = Tekanan darah diastolik (mm Hg)
# * SkinThickness = Ketebalan lipatan kulit (mm)
# * Insulin = 2-Jam serum insulin (mu U / ml)
# * BMI = Indeks massa tubuh (berat dalam kg / (tinggi dalam m) ^ 2)
# * DiabetesPedigreeFunction = Riwayat keturunan diabetes
# * Age = Umur (tahun)
# * Variabel Outcome = Class (0 atau 1) 268 dari 768 adalah 1, yang lain adalah 0

# # 2.Data Preparation

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.describe()


# # 3. Data Exploration

# In[ ]:


dataset.Outcome.value_counts()


# In[ ]:


sns.countplot(x='Outcome', data=dataset)


# # Distribusi Variabel Independen

# ### Variabel Pregnancies 

# In[ ]:


print(sns.distplot(dataset['Pregnancies']))

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.boxplot(data=dataset['Pregnancies'], ax=axes[0]);
sns.violinplot(data=dataset['Pregnancies'], ax=axes[1])

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(sns.distplot, 'Pregnancies')  .add_legend()
plt.title('PDF with Pregnancies')
plt.show()

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(plt.scatter, 'Outcome', 'Pregnancies') .add_legend()
plt.title('Sebaran Pasien Berdasarkan Pregnancies')
plt.show()


# ### Glucose

# In[ ]:


print(sns.distplot(dataset['Glucose']))

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.boxplot(data=dataset['Glucose'], ax=axes[0]);
sns.violinplot(data=dataset['Glucose'], ax=axes[1]);

sns.FacetGrid(dataset, hue="Outcome", height=5)  .map(sns.distplot, "Glucose")  .add_legend()
plt.title('PDF with Glucose')
plt.show()

sns.FacetGrid(dataset, hue = 'Outcome', height = 5).map(plt.scatter, 'Outcome', 'Glucose').add_legend()
plt.title('Distribusi Pasien Berdasarkan Glucose')
plt.show()


# ### Blood Presure

# In[ ]:


sns.distplot(dataset['BloodPressure'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.boxplot(data=dataset['BloodPressure'], ax=axes[0]);
sns.violinplot(data=dataset['BloodPressure'], ax=axes[1]);

sns.FacetGrid(dataset, hue="Outcome", height=5)  .map(sns.distplot, "BloodPressure")  .add_legend()
plt.title('PDF with BloodPressure')
plt.show()

sns.FacetGrid(dataset, hue = 'Outcome', height = 5).map(plt.scatter, 'Outcome', 'BloodPressure').add_legend()
plt.title('Distribusi Pasien Berdasarkan BloodPressure')
plt.show()


# ### SkinThickness

# In[ ]:


sns.distplot(dataset['SkinThickness'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.boxplot(data=dataset['SkinThickness'], ax=axes[0]);
sns.violinplot(data=dataset['SkinThickness'], ax=axes[1]);

sns.FacetGrid(dataset, hue="Outcome", height = 5)  .map(sns.distplot, "SkinThickness")  .add_legend()
plt.title('PDF with SkinThickness')
plt.show()

sns.FacetGrid(dataset, hue = 'Outcome', height = 5).map(plt.scatter, 'Outcome', 'SkinThickness').add_legend()
plt.title('Distribusi Pasien Berdasarkan SkinThickness')
plt.show()


# ### Insulin

# In[ ]:


sns.distplot(dataset['Insulin'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.boxplot(data=dataset['Insulin'], ax=axes[0]);
sns.violinplot(data=dataset['Insulin'], ax=axes[1]);

sns.FacetGrid(dataset, hue="Outcome", height=5)  .map(sns.distplot, "Insulin")  .add_legend()
plt.title('PDF with Insulin')
plt.show()

sns.FacetGrid(dataset, hue = 'Outcome', height = 5).map(plt.scatter, 'Outcome', 'Insulin').add_legend()
plt.title('Distribusi Pasien Berdasarkan Insulin')
plt.show()


# ### BMI 

# In[ ]:


sns.distplot(dataset['BMI'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
sns.barplot(data=dataset['BMI'], ax=axes[0]);
sns.violinplot(data=dataset['BMI'], ax=axes[1]);

sns.FacetGrid(dataset, hue='Outcome', height=5)  .map(sns.distplot, 'BMI')  .add_legend()
plt.title('PDF with BMI')
plt.show()

sns.FacetGrid(dataset, hue='Outcome', height=5)  .map(plt.scatter, 'Outcome', 'BMI')  .add_legend()
plt.title('Sebaran pasien berdasarkan BMI')
plt.show


# ### DiabetesPedigreeFunction

# In[ ]:


sns.distplot(dataset['DiabetesPedigreeFunction'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,6))
sns.boxplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[0]);
sns.violinplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[1])

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(sns.distplot, 'DiabetesPedigreeFunction')  .add_legend()
plt.title('PDF with DiabetesPedigreeFunction')
plt.show

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(plt.scatter, 'Outcome','DiabetesPedigreeFunction')  .add_legend()
plt.title('Sebaran pasien berdasarkan DiabetesPedigreeFunction')
plt.show()


# ### Age 

# In[ ]:


sns.distplot(dataset['Age'])

_, axes = plt.subplots(1,2, sharey=True, figsize=(10,6))
sns.boxplot(data=dataset['Age'], ax=axes[0]);
sns.violinplot(data=dataset['Age'], ax=axes[1])

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(sns.distplot, 'Age')  .add_legend()
plt.title('PDF with Age')
plt.show

sns.FacetGrid(data=dataset, hue='Outcome', height=5)  .map(plt.scatter, 'Outcome','Age')  .add_legend()
plt.title('Sebaran pasien berdasarkan Age')
plt.show()


# ### Rekap Visualisasi

# In[ ]:


sns.pairplot(data=dataset)


# # 4. Building the Model

# **Logistic Regression**
# 
# Regresi Logistik adalah salah satu tipe analisis regresi di statistik yang digunakan untuk memprediksi sebuah keluaran variabel dependen yang menjadi prediktor dari variabel independen. didalam regresi logistik banyak digunakan untuk memprediksi dan juga menghitung kesuksesan probabilitas

# In[ ]:


#Membuat variabel constanta
from statsmodels.tools import add_constant as add_constant
dataset_df = add_constant(dataset)
dataset_df.head()


# In[ ]:


import statsmodels.api as sm
column = dataset_df.columns[:-1]
model = sm.Logit(dataset_df.Outcome, dataset_df[column])
result=model.fit()
result.summary()


# Dari tabel diatas dapat dilihat bahwa variabel yang lebih dari 5% (>0.05) menunjukkan tidak mempengaruhi hubungan yang signifikan dengan diabetes. maka dari itu variabel tersebut harus dikeluarkan dengan variabel yg memiliki Pvalue terbesar (SkinThickness). hal itu dilakukan berulang sampai tidak ada variabel yang melebihi 0.05 

# **Backward Elimination**

# In[ ]:


def back_feature_elem (data_frame,dep_var,col_list):
   
    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(dataset_df,dataset.Outcome,column)
result.summary()


# **Interpreting the Results: Odds Ratio, Confidence Interval and Pvalues**

# In[ ]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue = round(result.pvalues,3)
conf['Pvalue']=pvalue
conf.columns = ['CI 95%', 'CI 97%', 'Odds Ratio', 'Pvalue']
print((conf))


# **Spliting Data**

# In[ ]:


X = dataset.iloc[:,[0,1,2,5,6]].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25 , random_state=0)

from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression(random_state=0)
lreg.fit(X_train, y_train)

y_pred = lreg.predict(X_test)


# # 5. Model Testing/Evaluation

# In[ ]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
akurasi = metrics.accuracy_score(y_test,y_pred)
print(cm)
print(cr)
print('akurasi yang dimiliki oleh model: %0.2f ' %(akurasi*100),'%')


# **Confusion Matrix**

# In[ ]:


from pandas import DataFrame
conf_matrix = pd.DataFrame(data=cm, columns=['Positif:1', 'Negatif:0'], index=['Positif:1','Negatif:0'])
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')


# Confusion Matrix menunjukkan 113 + 36 = 149 data prediksi benar dan 26+11 = 37 data prediksi salah
# 
# True Positive = 113
# 
# True Negative = 36
# 
# False Positive = 11
# 
# False Negative = 26

# In[ ]:


TP = cm[1,1]
TN = cm[0,0]
FN = cm[1,0]
FP = cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# **Model Evaluasi**

# In[ ]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',
      
'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)      


# **Predicted Probabillities**

# In[ ]:


y_pred_prob=lreg.predict_proba(X_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['no Diabetes','Diabetes'])
y_pred_prob_df.head()


# **Kurfa ROC**

# In[ ]:


from sklearn.metrics import roc_curve
import sklearn

logistik=lreg.predict_proba(X_test)
AUC = sklearn.metrics.roc_auc_score(y_test,logistik[:,1])
fpr, tpr, thresholds = roc_curve(y_test, logistik[:,1])

#Plotting AUC=0.5 Red Line
plt.plot([0,1],[0,1],color='red', linestyle = '--')

#Plotting ROC Graph (Blue)
plt.plot(fpr,tpr, label='ROC Curve (AUC= %0.2f' % AUC)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Diabetes')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.legend(loc='lower right')
plt.grid(True)


# # Kesimpulan

# 1. Semua atribut yang dipilih setelah proses eliminasi menunjukkan Pvalues lebih rendah dari 5% dan dengan demikian menunjukkan peran signifikan dalam prediksi penyakit Diabetes.
# 
# 2. BMI yang tinggi memiliki resiko 2 kali terkena penyakit Diabetes. kehamilan, glukosa, tekanan darah, dan Riwayat Keturunan juga berpengaruh dengan terjadinya penyakit Diabetes.
# 
# 3. Model diprediksi dengan akurasi 81%. Model ini lebih spesifik daripada sensitif.
# 
# 4. Area di bawah kurva ROC adalah 87% yang menunjukkan model cukup memuaskan.
# 
# 5. Model keseluruhan dapat ditingkatkan dengan lebih banyak data.

# In[ ]:




